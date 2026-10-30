% quality_check.m
function [passed,reasons,measurements] = quality_check(img,segmenter,thresholdOverrides)
% QUALITY_CHECK Assess blur, clipped exposure, and predicted wound area.
%   [PASSED,REASONS,MEASUREMENTS] = quality_check(IMG,SEGMENTER) uses defaults
%   from config.m. SEGMENTER is the trained DeepLab dlnetwork loaded from
%   models/segmenter.mat. Optional THRESHOLDOVERRIDES is a scalar struct whose
%   fields override cfg.quality values for local tuning.
%
%   Colour-card detection is an explicit placeholder. Enabling checkColorCard
%   returns a failure reason because no detector is implemented.
%
%   Thresholds are initial engineering values only; tune them on representative
%   phone photos. This function is not a clinical acceptance rule.

if nargin<2 || nargin>3
    error("WoundWatch:InputCount","Use [passed,reasons,measurements] = quality_check(img,segmenter[,thresholdOverrides]).");
end
if isempty(img) || ~(isnumeric(img)||islogical(img)) || ~(ismatrix(img)||ndims(img)==3)
    error("WoundWatch:InvalidImage","img must be a nonempty grayscale or RGB numeric image.");
end
if ~isa(segmenter,"dlnetwork")
    error("WoundWatch:InvalidSegmenter","segmenter must be a trained dlnetwork from train_segmentation.m.");
end

cfg=config(); thresholds=cfg.quality;
if nargin==3 && ~isempty(thresholdOverrides)
    if ~isstruct(thresholdOverrides) || ~isscalar(thresholdOverrides)
        error("WoundWatch:InvalidThresholds","thresholdOverrides must be a scalar struct.");
    end
    names=fieldnames(thresholdOverrides);
    for k=1:numel(names)
        if ~isfield(thresholds,names{k})
            error("WoundWatch:UnknownThreshold","Unknown quality threshold field: %s",names{k});
        end
        thresholds.(names{k})=thresholdOverrides.(names{k});
    end
end
validateThresholds(thresholds);

fprintf("Quality thresholds: Laplacian variance >= %.6g; clipped fraction <= %.3f ", ...
    thresholds.blurVarianceMinimum,thresholds.maximumClippedFraction);
fprintf("(dark <= %.3f, bright >= %.3f); wound area fraction in [%.4f, %.3f].\n", ...
    thresholds.clippingLowIntensity,thresholds.clippingHighIntensity, ...
    thresholds.minimumWoundAreaFraction,thresholds.maximumWoundAreaFraction);
if thresholds.checkColorCard
    fprintf("Colour-card check requested, but detector is a placeholder and not implemented.\n");
end

rgb=qualityRGB(img);
gray=rgb2gray(rgb);
laplacianKernel=fspecial("laplacian",0.2);
laplacianResponse=imfilter(gray,laplacianKernel,"replicate","conv");
blurVariance=var(laplacianResponse(:),0);
lowClipFraction=nnz(gray<=thresholds.clippingLowIntensity)/numel(gray);
highClipFraction=nnz(gray>=thresholds.clippingHighIntensity)/numel(gray);
clippedFraction=lowClipFraction+highClipFraction;

[woundMask,skinRing]=segmentWound(img,segmenter);
woundAreaFraction=nnz(woundMask)/numel(woundMask);

reasons=strings(0,1);
if blurVariance<thresholds.blurVarianceMinimum
    reasons(end+1,1)="Image is too blurry (variance of Laplacian below threshold).";
end
if clippedFraction>thresholds.maximumClippedFraction
    reasons(end+1,1)="Exposure has excessive dark or bright histogram clipping.";
end
if woundAreaFraction<thresholds.minimumWoundAreaFraction
    reasons(end+1,1)="Predicted wound area is below the configured minimum.";
elseif woundAreaFraction>thresholds.maximumWoundAreaFraction
    reasons(end+1,1)="Predicted wound area exceeds the configured maximum.";
end
if thresholds.checkColorCard
    reasons(end+1,1)="Colour-card check requested, but detection is not implemented.";
end
passed=isempty(reasons);
measurements=struct( ...
    "blurVarianceLaplacian",blurVariance, ...
    "lowClippedFraction",lowClipFraction, ...
    "highClippedFraction",highClipFraction, ...
    "totalClippedFraction",clippedFraction, ...
    "woundAreaFraction",woundAreaFraction, ...
    "colourCardStatus","not implemented", ...
    "woundMask",woundMask, ...
    "skinRing",skinRing);

end

function validateThresholds(t)
required=["blurVarianceMinimum","clippingLowIntensity","clippingHighIntensity", ...
    "maximumClippedFraction","minimumWoundAreaFraction","maximumWoundAreaFraction","checkColorCard"];
for k=1:numel(required)
    name=char(required(k));
    if ~isfield(t,name)
        error("WoundWatch:MissingThreshold","Missing quality threshold: %s",required(k));
    end
end
numericFields=required(1:6);
for k=1:numel(numericFields)
    name=char(numericFields(k));
    value=t.(name);
    if ~isnumeric(value)||~isscalar(value)||~isfinite(value)
        error("WoundWatch:InvalidThreshold","Threshold %s must be a finite numeric scalar.",numericFields(k));
    end
end
if t.clippingLowIntensity<0 || t.clippingHighIntensity>1 || ...
        t.clippingLowIntensity>=t.clippingHighIntensity
    error("WoundWatch:InvalidThreshold","Clipping thresholds must satisfy 0 <= low < high <= 1.");
end
if t.maximumClippedFraction<0 || t.maximumClippedFraction>1
    error("WoundWatch:InvalidThreshold","maximumClippedFraction must be between 0 and 1.");
end
if t.minimumWoundAreaFraction<0 || t.maximumWoundAreaFraction>1 || ...
        t.minimumWoundAreaFraction>t.maximumWoundAreaFraction
    error("WoundWatch:InvalidThreshold","Wound-area thresholds must satisfy 0 <= minimum <= maximum <= 1.");
end
if ~islogical(t.checkColorCard)||~isscalar(t.checkColorCard)
    error("WoundWatch:InvalidThreshold","checkColorCard must be a logical scalar.");
end
end

function rgb=qualityRGB(img)
if ndims(img)==2
    img=repmat(img,1,1,3);
elseif size(img,3)==4
    img=img(:,:,1:3);
elseif size(img,3)~=3
    error("WoundWatch:InvalidImageChannels","img must have one, three, or four channels.");
end
if isfloat(img)
    if any(~isfinite(img(:)))
        error("WoundWatch:InvalidImageValues","Image values must be finite.");
    end
    if max(img(:))>1
        img=double(img)/255;
    end
    if min(img(:))<0 || max(img(:))>1
        error("WoundWatch:InvalidImageRange","Floating point img values must lie in [0,1] or [0,255].");
    end
    rgb=double(img);
else
    rgb=im2double(img);
end
end

