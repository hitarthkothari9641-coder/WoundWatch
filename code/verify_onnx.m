% verify_onnx.m
% Compare MATLAB and re-imported ONNX outputs on 20 deterministic images.
thisDir=fileparts(mfilename("fullpath")); addpath(thisDir);
cfg=config(); rng(cfg.seed,"twister");
if ~isfile(cfg.paths.onnxClassifier) || ~isfile(cfg.paths.onnxSegmenter)
    error("WoundWatch:MissingONNX","Run export_onnx.m before verify_onnx.m.");
end
if ~isfile(cfg.paths.infectionClassifierModel) || ~isfile(cfg.paths.segmentationModel)
    error("WoundWatch:MissingModel","Train both networks before verification.");
end
C=load(cfg.paths.infectionClassifierModel,"finalNet","imageSize");
S=load(cfg.paths.segmentationModel,"net","inputSize");
files=findVerificationImages(cfg.paths.surgwoundImages,cfg.onnx.verificationImageCount);
if numel(files)<cfg.onnx.verificationImageCount
    error("WoundWatch:InsufficientVerificationImages", ...
        "Found %d readable candidate paths; need %d images.",numel(files),cfg.onnx.verificationImageCount);
end
files=files(randperm(numel(files),cfg.onnx.verificationImageCount));
oldFolder=pwd; folderCleanup=onCleanup(@()cd(oldFolder)); cd(cfg.paths.models);
try
    importedClassifier=importNetworkFromONNX(cfg.paths.onnxClassifier,Namespace="ImportedClassifier");
    importedSegmenter=importNetworkFromONNX(cfg.paths.onnxSegmenter,Namespace="ImportedSegmenter");
catch ME
    error("WoundWatch:ONNXImportFailed", ...
        "Could not import ONNX. Confirm the ONNX converter add-on and supported layers. Details: %s",ME.message);
end
classifierErrors=zeros(numel(files),1); segmenterErrors=classifierErrors;
for i=1:numel(files)
    image=readRGB(files(i));
    % Both networks receive the same RGB uint8-derived single tensor (0..255).
    % Any input-layer normalization is embedded in, and applied by, each net.
    classifierInput=single(imresize(image,C.imageSize(1:2)));
    segmenterInput=single(imresize(image,S.inputSize(1:2)));
    classifierErrors(i)=compareOutputs(C.finalNet,importedClassifier,classifierInput,"classifier");
    segmenterErrors(i)=compareOutputs(S.net,importedSegmenter,segmenterInput,"segmenter");
end
maximumClassifierError=max(classifierErrors); maximumSegmenterError=max(segmenterErrors);
fprintf("Parity max absolute difference: classifier %.8g; segmenter %.8g.\n", ...
    maximumClassifierError,maximumSegmenterError);
fprintf("Tolerance %.8g; inputs are identical RGB tensors and each network applies its own embedded normalization.\n", ...
    cfg.onnx.parityTolerance);
if maximumClassifierError>cfg.onnx.parityTolerance || maximumSegmenterError>cfg.onnx.parityTolerance
    error("WoundWatch:ONNXParityFailed", ...
        "ONNX parity exceeds %.3g (classifier %.6g, segmenter %.6g).", ...
        cfg.onnx.parityTolerance,maximumClassifierError,maximumSegmenterError);
end
fprintf("ONNX parity passed on %d images for both networks.\n",numel(files));

function files=findVerificationImages(root,count)
extensions=["*.jpg","*.jpeg","*.png","*.bmp"];
files=strings(0,1);
for split=["train","val","test"]
    folder=fullfile(root,split);
    if ~isfolder(folder), continue; end
    for ext=extensions
        d=dir(fullfile(folder,ext));
        for j=1:numel(d)
            files(end+1,1)=string(fullfile(d(j).folder,d(j).name)); %#ok<AGROW>
            if numel(files)>=count*3, return; end
        end
    end
end
end

function image=readRGB(path)
[A,map]=imread(path);
if ~isempty(map), A=im2uint8(ind2rgb(A,map));
elseif ndims(A)==2, A=repmat(A,1,1,3);
elseif size(A,3)>=3, A=A(:,:,1:3);
else, error("WoundWatch:InvalidVerificationImage","A verification image has unsupported channels."); end
if ~isa(A,"uint8"), A=im2uint8(A); end
image=A;
end

function difference=compareOutputs(matlabNet,onnxNet,input,networkLabel)
a=gather(extractdata(predict(matlabNet,input)));
b=gather(extractdata(predict(onnxNet,input)));
a=double(a); b=double(b);
if ~isequal(size(a),size(b))
    error("WoundWatch:ONNXOutputShapeMismatch", ...
        "%s output shape differs between MATLAB and ONNX (%s vs %s).", ...
        networkLabel,mat2str(size(a)),mat2str(size(b)));
end
if any(~isfinite(a(:))) || any(~isfinite(b(:)))
    error("WoundWatch:ONNXNonfiniteOutput","%s produced nonfinite output values.",networkLabel);
end
difference=max(abs(a(:)-b(:)));
end
