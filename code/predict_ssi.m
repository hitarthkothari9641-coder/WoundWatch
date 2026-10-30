% predict_ssi.m
function result = predict_ssi(imagePath)
%PREDICT_SSI Run quality gate, wound segmentation, and calibrated classifier.
% RESULT fields: status ('ok','retake','defer'), probability (Elevated),
% risk_band ('low','elevated','defer','unknown'), and reasons (string array).
% This is a research estimate and is not a clinical diagnosis.

if nargin~=1 || ~(ischar(imagePath) || (isstring(imagePath)&&isscalar(imagePath)))
    error("WoundWatch:InvalidImagePath","Provide one image path as text.");
end
imagePath=string(imagePath);
if strlength(imagePath)==0 || ~isfile(imagePath)
    error("WoundWatch:MissingImage","Image path does not exist.");
end
thisDir=fileparts(mfilename("fullpath"));
addpath(thisDir);
cfg=config();
try
    img=imread(imagePath);
catch
    error("WoundWatch:UnreadableImage","The specified image could not be decoded.");
end
if isempty(img) || ~(ismatrix(img)||ndims(img)==3) || ...
        (ndims(img)==3 && ~ismember(size(img,3),[1 3 4]))
    error("WoundWatch:InvalidImage","Image must be grayscale, RGB, or RGBA.");
end

persistent classifier segmenter classifierStamp segmenterStamp modelFields
dc=dir(cfg.paths.infectionClassifierModel); ds=dir(cfg.paths.segmentationModel);
if isempty(dc) || isempty(ds)
    error("WoundWatch:MissingModel","Train classifier and segmenter before calling predict_ssi.");
end
if isempty(classifier) || isempty(segmenter) || ...
        isempty(classifierStamp) || dc.datenum~=classifierStamp || ds.datenum~=segmenterStamp
    if ~isfile(cfg.paths.infectionClassifierModel) || ~isfile(cfg.paths.segmentationModel)
        error("WoundWatch:MissingModel","Train classifier and segmenter before calling predict_ssi.");
    end
    C=load(cfg.paths.infectionClassifierModel);
    S=load(cfg.paths.segmentationModel);
    if ~isfield(C,"finalNet") || ~isa(C.finalNet,"dlnetwork") || ...
            ~isfield(C,"calibrationParameters") || numel(C.calibrationParameters)~=2 || ...
            ~isfield(C,"thresholds") || ~isfield(S,"net") || ~isa(S.net,"dlnetwork")
        error("WoundWatch:InvalidModelFile","Saved model files lack expected trained networks or metadata.");
    end
    classifier=C.finalNet; segmenter=S.net; modelFields=C;
    classifierStamp=dc.datenum; segmenterStamp=ds.datenum;
end

% The quality gate includes segmentation-based area screening; reuse its mask.
[passed,gateReasons,measurements]=quality_check(img,segmenter);
if ~passed
    result=struct("status","retake","probability",NaN, ...
        "risk_band","unknown","reasons",gateReasons);
    return
end
if ~any(measurements.woundMask(:))
    result=struct("status","retake","probability",NaN, ...
        "risk_band","unknown","reasons","No wound pixels were found after quality screening.");
    return
end

rgb=toUint8RGB(img);
if isfield(modelFields,"imageSize")
    targetSize=modelFields.imageSize(1:2);
else
    targetSize=cfg.image.classifierSize(1:2);
end
input=single(imresize(rgb,targetSize));
scores=gather(extractdata(predict(classifier,input)));
scores=squeeze(scores);
if numel(scores)~=2 || any(~isfinite(scores(:)))
    error("WoundWatch:InvalidClassifierOutput","Classifier did not return two finite class scores.");
end
rawElevated=double(scores(2));
rawElevated=min(max(rawElevated,1e-7),1-1e-7);
rawLogit=log(rawElevated/(1-rawElevated));
parameters=double(modelFields.calibrationParameters(:));
calibrated=sigmoidStable(parameters(1)+parameters(2)*rawLogit);
thresholdName=char(cfg.inference.thresholdName);
if ~isfield(modelFields.thresholds,thresholdName)
    error("WoundWatch:MissingThreshold","Classifier file has no threshold named '%s'.",thresholdName);
end
threshold=double(modelFields.thresholds.(thresholdName));
if ~isfinite(threshold) || threshold<0 || threshold>1
    error("WoundWatch:InvalidThreshold","Saved classifier threshold must be in [0,1].");
end
if abs(calibrated-threshold)<=cfg.inference.deferHalfWidth
    result=struct("status","defer","probability",calibrated, ...
        "risk_band","defer","reasons",sprintf( ...
        "Calibrated probability is within %.3f of the %s threshold.", ...
        cfg.inference.deferHalfWidth,thresholdName));
else
    band="low";
    if calibrated>=threshold, band="elevated"; end
    result=struct("status","ok","probability",calibrated, ...
        "risk_band",band,"reasons",strings(0,1));
end
end

function rgb=toUint8RGB(img)
if ndims(img)==2 || size(img,3)==1
    img=repmat(img(:,:,1),1,1,3);
elseif size(img,3)==4
    img=img(:,:,1:3);
end
if isfloat(img)
    if any(~isfinite(img(:)))
        error("WoundWatch:InvalidImageValues","Image contains NaN or Inf values.");
    end
    if min(img(:))<0 || max(img(:))>255
        error("WoundWatch:InvalidImageRange","Floating point image values must lie in [0,1] or [0,255].");
    end
    if max(img(:))>1, img=img/255; end
    rgb=im2uint8(img);
elseif isa(img,"uint8")
    rgb=img;
else
    rgb=im2uint8(img);
end
end

function p=sigmoidStable(x)
if x>=0, p=1/(1+exp(-x)); else, ex=exp(x); p=ex/(1+ex); end
p=min(max(p,0),1);
end
