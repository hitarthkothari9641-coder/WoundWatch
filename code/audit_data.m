%% audit_data.m
% WoundWatch data audit: image counts, dimensions, unreadable files,
% exact byte-identical files, and near-duplicates using a simple DCT pHash.
% Run this script from MATLAB. Outputs one row per discovered image to
% results/data_audit.csv. No image pixels or file names are printed.

clear; clc;
thisDir = fileparts(mfilename("fullpath"));
addpath(thisDir);
cfg = config();
rng(cfg.seed,"twister");
if ~isfolder(cfg.paths.results)
    [ok,msg] = mkdir(cfg.paths.results);
    if ~ok, error("WoundWatch:FolderCreateFailed","Cannot create results folder: %s",msg); end
end

%% Validate sources and collect image paths with dataset/class metadata
datasets = strings(0,1); classes = strings(0,1); splits = strings(0,1);
files = strings(0,1);
[files,datasets,classes,splits] = addSurgWound(cfg,files,datasets,classes,splits);
[files,datasets,classes,splits] = addFolderDataset(cfg.paths.kaggle,"Kaggle", ...
    files,datasets,classes,splits,cfg.audit.extensions);
[files,datasets,classes,splits] = addFolderDataset(cfg.paths.medetec,"Medetec", ...
    files,datasets,classes,splits,cfg.audit.extensions);
if isempty(files)
    error("WoundWatch:NoImages","No supported images were found in configured SurgWound, Kaggle, or Medetec roots.");
end

%% Read metadata, compute content hash and perceptual hash; record failures
n = numel(files);
width = nan(n,1); height = nan(n,1); exactHash = strings(n,1);
pHashBits = false(n,64); readable = false(n,1); errorText = strings(n,1);
for k = 1:n
    try
        info = imfinfo(files(k));
        width(k) = info(1).Width; height(k) = info(1).Height;
        bytes = readBytes(files(k));
        exactHash(k) = rollingFileHash(bytes);
        pHashBits(k,:) = perceptualHash(files(k),cfg.audit.nearDuplicateResize);
        readable(k) = true;
    catch ME
        errorText(k) = string(ME.message);
    end
    if mod(k,250)==0 || k==n
        fprintf("Inspected %d / %d images\n",k,n);
    end
end

%% Group exact byte duplicates; confirm bytes after matching hashes
exactGroup = strings(n,1);
readableIdx = find(readable);
uniqueHashes = unique(exactHash(readableIdx));
groupNumber = 0;
for h = 1:numel(uniqueHashes)
    idx = readableIdx(exactHash(readableIdx)==uniqueHashes(h));
    if numel(idx)<2, continue; end
    % Hash equality narrows candidates; byte comparison prevents hash-collision false matches.
    unassigned = idx;
    while ~isempty(unassigned)
        groupNumber = groupNumber+1;
        first = unassigned(1);
        same = first;
        for q = 2:numel(unassigned)
            if filesEqual(files(first),files(unassigned(q)))
                same(end+1,1) = unassigned(q); %#ok<AGROW>
            end
        end
        if numel(same)>1
            exactGroup(same) = "EXACT_" + compose("%05d",groupNumber);
        end
        unassigned=setdiff(unassigned,same,"stable");
    end
end

%% Find near duplicate pairs using pHash Hamming distance
% Compare only readable files. Pairwise work is chunked to cap temporary memory.
nearCount=zeros(n,1);
nearPartners=cell(n,1);
valid=find(readable);
bits=pHashBits(valid,:);
chunkSize=128;
for first=1:chunkSize:numel(valid)
    a=first:min(first+chunkSize-1,numel(valid));
    for ii=a
        for jj=ii+1:numel(valid)
            if exactGroup(valid(ii))~="" && exactGroup(valid(ii))==exactGroup(valid(jj))
                continue;
            end
            distance=sum(xor(bits(ii,:),bits(jj,:)));
            if distance<=cfg.audit.nearDuplicateHammingDistance
                i=valid(ii); j=valid(jj);
                nearCount(i)=nearCount(i)+1; nearCount(j)=nearCount(j)+1;
                nearPartners{i}(end+1)=j; %#ok<AGROW>
                nearPartners{j}(end+1)=i; %#ok<AGROW>
            end
        end
    end
end
nearFileList=strings(n,1);
for i=1:n
    if ~isempty(nearPartners{i})
        nearFileList(i)=join(files(nearPartners{i})," | ");
    end
end

%% Write detailed CSV and print aggregate counts without exposing file names
audit=table(datasets,classes,splits,files,width,height,readable,exactHash, ...
    exactGroup,nearCount,nearFileList,errorText,VariableNames=["dataset","class","split","file", ...
    "width","height","readable","fileHash","exactDuplicateGroup", ...
    "nearDuplicateCount","nearDuplicateFiles","readError"]);
writetable(audit,cfg.paths.auditCSV);

fprintf("\nWoundWatch data audit summary\n");
fprintf("Total discovered images: %d\n",n);
for d=["SurgWound","Kaggle","Medetec"]
    id=datasets==d;
    fprintf("%s: %d images, %d readable, %d unreadable\n",char(d),nnz(id),nnz(id&readable),nnz(id&~readable));
    cls=unique(classes(id));
    for c=cls(:)'
        fprintf("  %s: %d\n",char(c),nnz(id & classes==c));
    end
end
fprintf("Exact duplicate files (members of byte-identical groups): %d\n",nnz(exactGroup~=""));
fprintf("Images with >=1 near duplicate: %d (pHash Hamming threshold <= %d)\n", ...
    nnz(nearCount>0),cfg.audit.nearDuplicateHammingDistance);
fprintf("Unreadable files: %d\n",nnz(~readable));
fprintf("CSV written to: %s\n",cfg.paths.auditCSV);

%% Local functions
function [files,datasets,classes,splits] = addSurgWound(cfg,files,datasets,classes,splits)
root=cfg.paths.surgwound;
if ~isfolder(root)
    warning("WoundWatch:MissingDataset","SurgWound folder missing: %s",root);
    return
end
manifest=cfg.paths.surgwoundManifest;
if ~isfile(manifest)
    warning("WoundWatch:MissingManifest","SurgWound manifest missing: %s",manifest);
    return
end
T=readtable(manifest,TextType="string");
required=["file","split","label"];
if ~all(ismember(required,string(T.Properties.VariableNames)))
    error("WoundWatch:InvalidManifest","SurgWound CSV must contain columns file, split, label.");
end
if any(ismissing(T.file) | strlength(string(T.file))==0) || ...
        any(~ismember(string(T.label),["Low","Medium","High"]))
    error("WoundWatch:InvalidManifest","Manifest file entries must be nonempty and labels must be Low, Medium, or High.");
end
for i=1:height(T)
    rel=string(T.file(i));
    p=fullfile(root,rel);
    if ~isfile(p)
        p=fullfile(root,"images",string(T.split(i)),string(T.file(i)));
    end
    files(end+1,1)=p;
    datasets(end+1,1)="SurgWound";
    classes(end+1,1)=string(T.label(i));
    splits(end+1,1)=string(T.split(i));
end
end

function [files,datasets,classes,splits] = addFolderDataset(root,datasetName,files,datasets,classes,splits,extensions)
if ~isfolder(root)
    warning("WoundWatch:MissingDataset","%s folder missing: %s",datasetName,root);
    return
end
listing=dir(fullfile(root,"**","*"));
for i=1:numel(listing)
    if listing(i).isdir, continue; end
    [~,~,ext]=fileparts(listing(i).name);
    if ~ismember(lower(string(ext)),extensions), continue; end
    p=string(fullfile(listing(i).folder,listing(i).name));
    relative=extractAfter(p,strlength(string(root))+1);
    parts=split(relative,filesep);
    if numel(parts)>=2, className=parts(1); else, className="(root)"; end
    files(end+1,1)=p;
    datasets(end+1,1)=datasetName;
    classes(end+1,1)=className;
    if datasetName=="Medetec" && numel(parts)>=3
        splits(end+1,1)=parts(2);
    else
        splits(end+1,1)="(unspecified)";
    end
end
end

function bytes=readBytes(path)
fid=fopen(path,"rb");
if fid<0, error("WoundWatch:FileOpenFailed","Could not open image file."); end
cleanup=onCleanup(@()fclose(fid));
bytes=fread(fid,Inf,"*uint8");
end

function h=rollingFileHash(bytes)
% Deterministic two-prime modular byte hash; exact-match groups are verified
% byte-for-byte, so a hash collision cannot create a false exact duplicate.
p1=4294967291; p2=4294967279; a=257;
x1=1; x2=7;
for k=1:numel(bytes)
    x1=mod(x1*a+double(bytes(k)),p1);
    x2=mod(x2*a+double(bytes(k)),p2);
end
h=string([dec2hex(uint32(x1),8),dec2hex(uint32(x2),8)]);
end

function tf=filesEqual(a,b)
fa=fopen(a,"rb"); fb=fopen(b,"rb");
if fa<0 || fb<0
    if fa>=0, fclose(fa); end
    if fb>=0, fclose(fb); end
    tf=false; return
end
ca=onCleanup(@()fclose(fa)); cb=onCleanup(@()fclose(fb));
while true
    x=fread(fa,65536,"*uint8"); y=fread(fb,65536,"*uint8");
    if ~isequal(x,y), tf=false; return; end
    if isempty(x), tf=true; return; end
end
end

function bits=perceptualHash(path,resizeSize)
[A,map]=imread(path);
if ~isempty(map), A=ind2rgb(A,map); end
if ndims(A)==3
    if size(A,3)>=3, A=A(:,:,1:3); A=rgb2gray(A); else, A=A(:,:,1); end
end
A=imresize(im2double(A),resizeSize);
D=dct2(A);
low=D(1:8,1:8);
values=low(:);
threshold=median(values(2:end));
bits=(values>threshold)';
end

