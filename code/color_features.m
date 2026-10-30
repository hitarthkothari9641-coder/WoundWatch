% color_features.m
function features = color_features(input,maskOrNet,skinRing)
% COLOR_FEATURES Compute wound and surrounding-skin CIELAB features.
%   F = color_features(IMG,WOUNDMASK,SKINRING) returns one row of Lab means,
%   medians, and erythema (mean wound a* minus mean ring a*).
%
%   F = color_features(RECORDS,SEGMENTER) accepts a table with patientID, day,
%   and file columns, infers masks with segmentWound, and adds per-patient
%   *_drift columns relative to day 1. Each patient must have at least one
%   day==1 record. If multiple day-1 records exist, their feature means define
%   that patient's baseline.
%
%   color_features expects RGB photos in sRGB. No color normalization is
%   performed. Empty wound/ring regions yield NaN measurements.

if istable(input)
    if nargin~=2
        error("WoundWatch:InputCount","Table form is color_features(records,segmenter).");
    end
    features=featureTableFromRecords(input,maskOrNet);
else
    if nargin~=3
        error("WoundWatch:InputCount","Image form is color_features(img,woundMask,skinRing).");
    end
    features=singleImageFeatures(input,maskOrNet,skinRing);
end
end

function features=featureTableFromRecords(records,segmenter)
required=["patientID","day","file"];
if ~all(ismember(required,string(records.Properties.VariableNames)))
    error("WoundWatch:InvalidRecords","Records table must contain patientID, day, and file columns.");
end
if ~isa(segmenter,"dlnetwork")
    error("WoundWatch:InvalidSegmenter","segmenter must be a trained dlnetwork.");
end
patientID=string(records.patientID);
day=records.day;
file=string(records.file);
if any(ismissing(patientID)|strlength(patientID)==0)
    error("WoundWatch:InvalidPatientID","patientID values must be present and nonempty.");
end
if ~isnumeric(day)||any(~isfinite(day))||any(day<1)
    error("WoundWatch:InvalidDay","day must contain finite numeric day numbers >= 1.");
end
if any(ismissing(file)|strlength(file)==0)
    error("WoundWatch:InvalidFile","file values must be present and nonempty.");
end
n=height(records);
base=emptyFeatureTable(n);
for i=1:n
    if ~isfile(file(i))
        error("WoundWatch:MissingImage","Image file not found: %s",file(i));
    end
    image=readRGB(file(i));
    [mask,ring]=segmentWound(image,segmenter);
    row=singleImageFeatures(image,mask,ring);
    base(i,:)=row;
end

features=table(patientID,day,file,VariableNames=["patientID","day","file"]);
features=[features base];
featureNames=base.Properties.VariableNames;
patients=unique(patientID);
for k=1:numel(featureNames)
    name=featureNames{k};
    driftName=[name 'Drift'];
    drift=nan(n,1);
    for p=1:numel(patients)
        isPatient=patientID==patients(p);
        dayOne=isPatient & day==1;
        if ~any(dayOne)
            error("WoundWatch:MissingDayOne", ...
                "Patient records need a day==1 baseline before drift can be computed.");
        end
        baseline=mean(base{dayOne,name},"omitnan");
        drift(isPatient)=base{isPatient,name}-baseline;
    end
    features.(driftName)=drift;
end
end

function features=singleImageFeatures(img,woundMask,skinRing)
rgb=readRGB(img);
if ~(islogical(woundMask)||isnumeric(woundMask)) || ...
        ~(islogical(skinRing)||isnumeric(skinRing))
    error("WoundWatch:InvalidMasks","woundMask and skinRing must be logical or numeric masks.");
end
if ~isequal(size(woundMask),size(rgb,[1 2])) || ~isequal(size(skinRing),size(rgb,[1 2]))
    error("WoundWatch:MaskSizeMismatch","Wound and skin-ring masks must match image height and width.");
end
if any(~ismember(unique(woundMask(:)),[0 1])) || any(~ismember(unique(skinRing(:)),[0 1]))
    error("WoundWatch:NonbinaryMask","woundMask and skinRing must contain only zero and one.");
end
woundMask=logical(woundMask); skinRing=logical(skinRing);
if any(woundMask & skinRing,"all")
    error("WoundWatch:OverlappingMasks","skinRing must not overlap the wound mask.");
end

lab=rgb2lab(rgb);
values=nan(1,13);
channelNames=["L","a","b"];
regions={woundMask,skinRing};
featureIndex=1;
for regionIndex=1:2
    region=regions{regionIndex};
    for channel=1:3
        plane=lab(:,:,channel);
        samples=plane(region);
        values(featureIndex)=mean(samples,"omitnan"); featureIndex=featureIndex+1;
        values(featureIndex)=median(samples,"omitnan"); featureIndex=featureIndex+1;
    end
end
% Indices 3 and 9 are wound and ring mean a*, respectively.
values(13)=values(3)-values(9);
names=["woundMeanL","woundMedianL","woundMeanA","woundMedianA", ...
    "woundMeanB","woundMedianB","ringMeanL","ringMedianL", ...
    "ringMeanA","ringMedianA","ringMeanB","ringMedianB","erythema"];
features=array2table(values,VariableNames=names);
end

function T=emptyFeatureTable(n)
names=["woundMeanL","woundMedianL","woundMeanA","woundMedianA", ...
    "woundMeanB","woundMedianB","ringMeanL","ringMedianL", ...
    "ringMeanA","ringMedianA","ringMeanB","ringMedianB","erythema"];
T=array2table(nan(n,numel(names)),VariableNames=names);
end

function rgb=readRGB(input)
if ischar(input)||isstring(input)
    [img,map]=imread(input);
    if ~isempty(map), img=im2uint8(ind2rgb(img,map)); end
else
    img=input;
end
if isempty(img)||~(isnumeric(img)||islogical(img))
    error("WoundWatch:InvalidImage","Image input must be a nonempty numeric array or file path.");
end
if ndims(img)==2
    img=repmat(img,1,1,3);
elseif ndims(img)==3 && size(img,3)==4
    img=img(:,:,1:3);
elseif ndims(img)~=3 || size(img,3)~=3
    error("WoundWatch:InvalidImageChannels","Image must be grayscale or RGB.");
end
if isfloat(img)
    if any(~isfinite(img(:)))
        error("WoundWatch:InvalidImageValues","Image values must be finite.");
    end
    if max(img(:))>1, img=double(img)/255; end
    if min(img(:))<0 || max(img(:))>1
        error("WoundWatch:InvalidImageRange","Floating point image values must be in [0,1] or [0,255].");
    end
    rgb=double(img);
else
    rgb=im2double(img);
end
end

