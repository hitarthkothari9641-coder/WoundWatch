%% test_erythema_surgwound.m
% Exploratory image-level test of wound-minus-ring mean CIELAB a* on SurgWound.
% SurgWound has no patient IDs; no longitudinal drift is estimated here.

clear; clc;
thisDir=fileparts(mfilename("fullpath"));
addpath(thisDir);
cfg=config();
rng(cfg.seed+4,"twister");

if ~isfile(cfg.paths.surgwoundManifest)
    error("WoundWatch:MissingManifest","SurgWound manifest not found: %s",cfg.paths.surgwoundManifest);
end
if ~isfile(cfg.paths.segmentationModel)
    error("WoundWatch:MissingSegmenter","Train the segmenter first; model not found: %s",cfg.paths.segmentationModel);
end
saved=load(cfg.paths.segmentationModel,"net");
if ~isfield(saved,"net") || ~isa(saved.net,"dlnetwork")
    error("WoundWatch:InvalidSegmenterFile","Expected a trained dlnetwork named net in models/segmenter.mat.");
end
net=saved.net;

T=readtable(cfg.paths.surgwoundManifest,TextType="string");
required=["file","split","label"];
if ~all(ismember(required,string(T.Properties.VariableNames)))
    error("WoundWatch:InvalidManifest","Manifest must contain file, split, label.");
end
T.file=string(T.file); T.split=lower(string(T.split)); T.label=string(T.label);
isElevated=T.label=="Medium" | T.label=="High";
if any(~ismember(T.label,["Low","Medium","High"]))
    error("WoundWatch:InvalidManifest","Labels must be Low, Medium, or High.");
end
paths=strings(height(T),1);
for i=1:height(T)
    p=T.file(i);
    if ~isfile(p), p=fullfile(cfg.paths.surgwound,p); end
    if ~isfile(p), p=fullfile(cfg.paths.surgwoundImages,T.split(i),T.file(i)); end
    if ~isfile(p), error("WoundWatch:MissingImage","Manifest image is missing: %s",T.file(i)); end
    paths(i)=p;
end

erythema=nan(height(T),1);
for i=1:height(T)
    image=imread(paths(i));
    [wound,ring]=segmentWound(image,net);
    f=color_features(image,wound,ring);
    erythema(i)=f.erythema;
    if mod(i,100)==0 || i==height(T), fprintf("Processed %d / %d images.\n",i,height(T)); end
end
valid=isfinite(erythema);
y=isElevated(valid);
score=erythema(valid);
if numel(unique(y))~=2
    error("WoundWatch:InsufficientClasses","Need both Low and Elevated observations with valid masks.");
end
[~,~,~,auc]=perfcurve(y,score,true);
ci=bootstrapAUC(y,score,cfg.cv.bootstrapReplicates,cfg.seed+5);
pValue=ranksum(score(y),score(~y));
fprintf("\nSurgWound image-level erythema result (valid n=%d; elevated prevalence %.3f)\n", ...
    numel(y),mean(y));
fprintf("AUROC %.4f (stratified bootstrap 95%% CI %.4f to %.4f); 0.5 is chance-level ranking.\n", ...
    auc,ci(1),ci(2));
fprintf("Wilcoxon rank-sum p=%.5g; compares the two erythema distributions.\n",pValue);
fprintf("No patient-level inference or longitudinal drift is possible because patient IDs are absent.\n");

function ci=bootstrapAUC(y,scores,nBoot,seed)
rng(seed,"twister");
positive=find(y); negative=find(~y);
values=nan(nBoot,1);
for b=1:nBoot
    idx=[positive(randi(numel(positive),numel(positive),1)); ...
        negative(randi(numel(negative),numel(negative),1))];
    [~,~,~,values(b)]=perfcurve(y(idx),scores(idx),true);
end
ci=prctile(values,[2.5 97.5]);
end

