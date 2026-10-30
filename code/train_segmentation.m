%% train_segmentation.m
% Train DeepLab v3+ with a ResNet-18 encoder on FUSeg train.
% Validate on FUSeg validation and externally test on labeled Medetec test.
% Requires MATLAB R2026a (code targets R2024b+), Deep Learning Toolbox, Computer Vision Toolbox,
% Image Processing Toolbox, and pretrained ResNet-18 weights.

clear; clc;
thisDir=fileparts(mfilename("fullpath"));
addpath(thisDir);
cfg=config();
rng(cfg.seed,"twister");

%% Resolve dataset folders and validate configuration
trainImageDir=fullfile(cfg.paths.fuseg,"train","images");
trainMaskDir=fullfile(cfg.paths.fuseg,"train","labels");
validationImageDir=fullfile(cfg.paths.fuseg,"validation","images");
validationMaskDir=fullfile(cfg.paths.fuseg,"validation","labels");
externalImageDir=fullfile(cfg.paths.fusegExternalTest,"images");
externalMaskDir=fullfile(cfg.paths.fusegExternalTest,"labels");
requiredFolders=[trainImageDir,trainMaskDir,validationImageDir,validationMaskDir, ...
    externalImageDir,externalMaskDir];
for k=1:numel(requiredFolders)
    if ~isfolder(requiredFolders(k))
        error("WoundWatch:MissingSegmentationFolder", ...
            "Required FUSeg folder does not exist: %s",requiredFolders(k));
    end
end
if ~isfolder(cfg.paths.models), mkdir(cfg.paths.models); end
exampleDir=fullfile(cfg.paths.results,"seg_examples");
if ~isfolder(exampleDir), mkdir(exampleDir); end

inputSize=cfg.image.segmentationSize(1:2);
assert(numel(inputSize)==2 && all(inputSize>0), ...
    "config.image.segmentationSize must contain positive height and width.");
trainPairs=pairImagesAndMasks(trainImageDir,trainMaskDir,"FUSeg train");
validationPairs=pairImagesAndMasks(validationImageDir,validationMaskDir,"FUSeg validation");
externalPairs=pairImagesAndMasks(externalImageDir,externalMaskDir,"Medetec external test");
fprintf("Paired images: train %d, validation %d, external test %d.\n", ...
    height(trainPairs),height(validationPairs),height(externalPairs));

%% Load and resize training images and binary ground-truth masks
[XTrain,YTrain]=loadSegmentationSet(trainPairs,inputSize,"FUSeg train");

%% Create DeepLab v3+ and train with crossentropy plus soft foreground Dice
classNames=["background","wound"];
% Categorical target values follow classNames order: 0=background, 1=wound.
categoricalTrain=categorical(YTrain,uint8([0 1]),classNames);
net=deeplabv3plus([inputSize 3],numel(classNames),"resnet18",DownsamplingFactor=16);
lossFcn=@(P,T) crossentropy(P,T)+foregroundDiceLoss(P,T);

useGPU=false;
try
    useGPU=canUseGPU;
catch
    useGPU=false;
end
if useGPU
    executionEnvironment="gpu";
    miniBatch=cfg.training.segmentationMiniBatchSizeGPU;
else
    executionEnvironment="cpu";
    miniBatch=cfg.training.segmentationMiniBatchSizeCPU;
end
options=trainingOptions("adam", ...
    MaxEpochs=cfg.training.segmentationMaxEpochs, ...
    MiniBatchSize=miniBatch, ...
    InitialLearnRate=cfg.training.initialLearnRate, ...
    Shuffle="every-epoch", ...
    Verbose=false, ...
    Plots="none", ...
    ExecutionEnvironment=executionEnvironment);
fprintf("Training on %s, batch size %d, input %dx%d.\n", ...
    executionEnvironment,miniBatch,inputSize(1),inputSize(2));
net=trainnet(XTrain,categoricalTrain,net,lossFcn,options);

%% Evaluate validation and external test Dice and IoU
validationMetrics=evaluateSegmentation(net,validationPairs,inputSize);
externalMetrics=evaluateSegmentation(net,externalPairs,inputSize);
fprintf("FUSeg validation: mean Dice %.4f; mean IoU %.4f.\n", ...
    validationMetrics.meanDice,validationMetrics.meanIoU);
fprintf("Medetec external test: mean Dice %.4f; mean IoU %.4f.\n", ...
    externalMetrics.meanDice,externalMetrics.meanIoU);
fprintf("Dice = 2*intersection/(predicted pixels + true pixels); IoU = intersection/union.\n");

%% Save 12 qualitative prediction overlays (six validation, six external)
nValidation=min(6,height(validationPairs));
nExternal=min(6,height(externalPairs));
if nValidation+nExternal<12
    error("WoundWatch:InsufficientOverlayImages", ...
        "Need at least 12 total validation and external images to save 12 overlays.");
end
saveOverlaySet(net,validationPairs,1:nValidation,inputSize,exampleDir,"validation");
saveOverlaySet(net,externalPairs,1:nExternal,inputSize,exampleDir,"external_test");
save(cfg.paths.segmentationModel,"net","inputSize","classNames","validationMetrics", ...
    "externalMetrics","cfg","-v7.3");
fprintf("Saved segmenter: %s\n",cfg.paths.segmentationModel);
fprintf("Saved 12 overlays under: %s\n",exampleDir);

%% Local functions
function pairs=pairImagesAndMasks(imageDir,maskDir,description)
imageListing=dir(fullfile(imageDir,"*"));
imageListing=imageListing(~[imageListing.isdir]);
maskListing=dir(fullfile(maskDir,"*"));
maskListing=maskListing(~[maskListing.isdir]);
validExt=[".jpg",".jpeg",".png",".bmp",".tif",".tiff"];
maskExt=[".png",".tif",".tiff"];
images=strings(0,1); masks=strings(0,1);
for i=1:numel(imageListing)
    [~,imageStem,imageExtension]=fileparts(imageListing(i).name);
    if ~ismember(lower(string(imageExtension)),validExt), continue; end
    matches=[];
    for j=1:numel(maskListing)
        [~,maskStem,maskExtension]=fileparts(maskListing(j).name);
        if strcmpi(imageStem,maskStem) && ismember(lower(string(maskExtension)),maskExt)
            matches(end+1)=j; %#ok<AGROW>
        end
    end
    if numel(matches)>1
        error("WoundWatch:AmbiguousMask","Multiple masks match an image in %s.",description);
    elseif numel(matches)==1
        images(end+1,1)=string(fullfile(imageListing(i).folder,imageListing(i).name)); %#ok<AGROW>
        j=matches(1);
        masks(end+1,1)=string(fullfile(maskListing(j).folder,maskListing(j).name)); %#ok<AGROW>
    end
end
if isempty(images)
    error("WoundWatch:NoImageMaskPairs","No matching image/mask pairs found for %s.",description);
end
if numel(images)~=numel(masks)
    error("WoundWatch:PairingFailure","Image/mask pairing failed for %s.",description);
end
pairs=table(images,masks,VariableNames=["imageFile","maskFile"]);
end

function [X,Y]=loadSegmentationSet(pairs,inputSize,description)
n=height(pairs); H=inputSize(1); W=inputSize(2);
X=zeros(H,W,3,n,"uint8");
Y=zeros(H,W,1,n,"uint8");
for i=1:n
    X(:,:,:,i)=readRGBImage(pairs.imageFile(i),inputSize);
    mask=readBinaryMask(pairs.maskFile(i));
    mask=imresize(mask,inputSize,"nearest");
    Y(:,:,1,i)=uint8(mask);
end
fprintf("Loaded %s (%d paired samples).\n",description,n);
end

function image=readRGBImage(path,inputSize)
[A,map]=imread(path);
if ~isempty(map)
    A=im2uint8(ind2rgb(A,map));
elseif ndims(A)==2
    A=repmat(A,1,1,3);
elseif size(A,3)>=3
    A=A(:,:,1:3);
else
    error("WoundWatch:UnsupportedImage","Image must be grayscale or RGB.");
end
A=imresize(A,inputSize);
if ~isa(A,"uint8"), A=im2uint8(A); end
image=A;
end

function mask=readBinaryMask(path)
M=imread(path);
if ndims(M)==3, M=M(:,:,1); end
values=unique(M(:));
if all(ismember(values,uint8([0 1])))
    mask=(M==1);
elseif all(ismember(values,uint8([0 255])))
    mask=(M==255);
else
    error("WoundWatch:UnexpectedMaskValues", ...
        "Mask must use only 0/1 or 0/255 values: %s",path);
end
mask=logical(mask);
end

function loss=foregroundDiceLoss(prediction,target)
% trainnet converts categorical pixel labels to one-hot targets. Under
% classNames=["background","wound"], channel 2 is the wound foreground.
p=prediction(:,:,2,:);
t=target(:,:,2,:);
intersection=sum(sum(p.*t,1),2);
denominator=sum(sum(p,1),2)+sum(sum(t,1),2);
dice=(2.*intersection+1)./(denominator+1);
loss=1-mean(dice,"all");
end

function metrics=evaluateSegmentation(net,pairs,inputSize)
n=height(pairs); dice=zeros(n,1); iou=zeros(n,1);
for i=1:n
    image=readRGBImage(pairs.imageFile(i),inputSize);
    truth=imresize(readBinaryMask(pairs.maskFile(i)),inputSize,"nearest");
    prediction=predictBinaryMask(net,image);
    intersection=nnz(prediction & truth);
    predictionCount=nnz(prediction); truthCount=nnz(truth);
    unionCount=nnz(prediction | truth);
    if predictionCount+truthCount==0
        dice(i)=1;
    else
        dice(i)=2*intersection/(predictionCount+truthCount);
    end
    if unionCount==0
        iou(i)=1;
    else
        iou(i)=intersection/unionCount;
    end
end
metrics=struct("meanDice",mean(dice),"meanIoU",mean(iou), ...
    "perImageDice",dice,"perImageIoU",iou,"numImages",n);
end

function mask=predictBinaryMask(net,image)
prediction=gather(extractdata(predict(net,single(image))));
prediction=squeeze(prediction);
if ndims(prediction)~=3 || size(prediction,3)~=2
    error("WoundWatch:UnexpectedSegmentationOutput", ...
        "Expected H-by-W-by-2 semantic scores from the segmenter.");
end
mask=prediction(:,:,2)>=prediction(:,:,1);
end

function saveOverlaySet(net,pairs,indices,inputSize,outputDir,prefix)
for k=1:numel(indices)
    row=indices(k);
    original=readRGBImage(pairs.imageFile(row),inputSize);
    truth=imresize(readBinaryMask(pairs.maskFile(row)),inputSize,"nearest");
    prediction=predictBinaryMask(net,original);
    overlay=im2double(original);
    alpha=0.38;
    % Predicted wound region is red; true-mask boundary is cyan.
    overlay(:,:,1)=overlay(:,:,1).*(1-alpha*prediction)+alpha*prediction;
    overlay(:,:,2)=overlay(:,:,2).*(1-alpha*prediction);
    overlay(:,:,3)=overlay(:,:,3).*(1-alpha*prediction);
    boundary=bwperim(truth);
    overlay(:,:,1)=overlay(:,:,1).*(~boundary);
    overlay(:,:,2)=overlay(:,:,2).*(~boundary)+double(boundary);
    overlay(:,:,3)=overlay(:,:,3).*(~boundary)+double(boundary);
    out=uint8(255*min(max(overlay,0),1));
    outPath=fullfile(outputDir,sprintf("%s_%02d.png",prefix,k));
    imwrite(out,outPath);
end
end

