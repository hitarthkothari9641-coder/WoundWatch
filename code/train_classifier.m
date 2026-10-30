%% train_classifier.m
% WoundWatch Low-vs-Elevated binary image classifier.
% MATLAB R2026a; compatible with R2024b+; requires Deep Learning, Image Processing, and Statistics
% and Machine Learning Toolboxes. Uses a cached uint8 MAT array so source JPEGs
% are decoded and resized once, not during each training epoch.
%
% CV is image-level because the supplied manifest has no patient identifiers.
% Consequently, patient-level leakage cannot be excluded.

clear; clc;
thisDir=fileparts(mfilename("fullpath"));
addpath(thisDir);
cfg=config();
rng(cfg.seed,"twister");

%% Validate configuration and input manifest
allowed=["resnet18","resnet50","efficientnetb0"];
backbone=lower(string(cfg.models.infectionClassifier));
if ~ismember(backbone,allowed)
    error("WoundWatch:InvalidBackbone", ...
        "config.models.infectionClassifier must be resnet18, resnet50, or efficientnetb0; got '%s'.",backbone);
end
if ~isfile(cfg.paths.surgwoundManifest)
    error("WoundWatch:MissingManifest","SurgWound manifest not found: %s",cfg.paths.surgwoundManifest);
end
if ~isfolder(cfg.paths.results), mkdir(cfg.paths.results); end
if ~isfolder(cfg.paths.models), mkdir(cfg.paths.models); end

T=readtable(cfg.paths.surgwoundManifest,TextType="string");
required=["file","split","label"];
if ~all(ismember(required,string(T.Properties.VariableNames)))
    error("WoundWatch:InvalidManifest","Manifest needs columns: file, split, label.");
end
T.file=string(T.file); T.split=lower(string(T.split)); T.label=string(T.label);
if isempty(T) || any(ismissing(T.file) | strlength(T.file)==0) || ...
        any(~ismember(T.label,["Low","Medium","High"]))
    error("WoundWatch:InvalidManifest","Manifest must have files and labels in Low, Medium, High.");
end
filePaths=strings(height(T),1);
for i=1:height(T)
    p=T.file(i);
    if ~isfile(p), p=fullfile(cfg.paths.surgwound,p); end
    if ~isfile(p), p=fullfile(cfg.paths.surgwoundImages,T.split(i),T.file(i)); end
    if ~isfile(p)
        error("WoundWatch:MissingImage","Image listed in manifest is missing: %s",T.file(i));
    end
    filePaths(i)=p;
end
binaryLabels=repmat("Elevated",height(T),1);
binaryLabels(T.label=="Low")="Low";
Y=categorical(binaryLabels,["Low","Elevated"]);
if min(countcats(Y))<cfg.cv.folds
    error("WoundWatch:InsufficientClassCount", ...
        "Each binary class needs at least %d images for stratified %d-fold CV.",cfg.cv.folds,cfg.cv.folds);
end
fprintf("Classifier backbone: %s. Images: %d (Low %d, Elevated %d).\n", ...
    backbone,height(T),nnz(Y=="Low"),nnz(Y=="Elevated"));
fprintf("CV is stratified by image; no patient IDs are available to group related images.\n");

%% Create or reuse the resized image cache
% Cache key includes file path, label, byte count, and modification time.
% If any source file or label changes, the cache is rebuilt automatically.
n=height(T); imageSize=cfg.image.classifierSize;
fileBytes=zeros(n,1); fileDatenum=zeros(n,1);
for i=1:n
    d=dir(filePaths(i));
    if isempty(d), error("WoundWatch:MissingImage","Cannot stat image: %s",filePaths(i)); end
    fileBytes(i)=d(1).bytes; fileDatenum(i)=d(1).datenum;
end
metadata=table(filePaths,T.label,fileBytes,fileDatenum, ...
    VariableNames=["file","severityLabel","bytes","modifiedDatenum"]);
cachePath=cfg.paths.classifierCache;
useCache=false;
if isfile(cachePath)
    old=load(cachePath,"images","metadata");
    useCache=isequal(old.metadata,metadata) && ...
        size(old.images,1)==imageSize(1) && size(old.images,2)==imageSize(2) && ...
        size(old.images,3)==imageSize(3) && size(old.images,4)==n;
end
if useCache
    images=old.images;
    fprintf("Reusing resized image cache: %s\n",cachePath);
else
    fprintf("Building 224x224 image cache from %d files...\n",n);
    images=zeros(imageSize(1),imageSize(2),3,n,"uint8");
    for i=1:n
        images(:,:,:,i)=readResizeRGB(filePaths(i),imageSize(1:2));
        if mod(i,100)==0 || i==n, fprintf("  Cached %d / %d\n",i,n); end
    end
    save(cachePath,"images","metadata","-v7.3");
end

%% Stratified five-fold CV with class-weighted crossentropy
% Assign folds locally. This avoids cvpartition's service dependency and keeps
% each class distributed across all folds using the fixed project seed.
foldAssignment=stratifiedFoldIDs(Y,cfg.cv.folds,cfg.seed);
oofRaw=nan(n,1); foldID=foldAssignment;
useGPU=false;
try
    useGPU=canUseGPU;
catch
    useGPU=false;
end
if useGPU
    executionEnvironment="gpu";
    miniBatch=cfg.training.miniBatchSizeGPU;
else
    executionEnvironment="cpu";
    miniBatch=cfg.training.miniBatchSizeCPU;
end
preprocessingEnvironment=string(cfg.training.preprocessingEnvironment);
fprintf("ExecutionEnvironment=%s; PreprocessingEnvironment=%s; minibatch=%d.\n", ...
    executionEnvironment,preprocessingEnvironment,miniBatch);

for f=1:cfg.cv.folds
    va=foldAssignment==f; tr=~va;
    nLow=nnz(Y(tr)=="Low"); nElev=nnz(Y(tr)=="Elevated");
    classWeights=single([nnz(tr)/(2*nLow),nnz(tr)/(2*nElev)]);
    lossFcn=dlaccelerate(@(P,Targ)crossentropy(P,Targ,classWeights));
    net=imagePretrainedNetwork(backbone,NumClasses=2);
    augmenter=imageDataAugmenter( ...
        RandRotation=cfg.augmentation.rotationDegrees, ...
        RandXReflection=cfg.augmentation.horizontalReflection, ...
        RandYReflection=cfg.augmentation.verticalReflection, ...
        RandXTranslation=cfg.augmentation.translationPixels, ...
        RandYTranslation=cfg.augmentation.translationPixels);
    trainDS=augmentedImageDatastore(imageSize(1:2),images(:,:,:,tr),Y(tr), ...
        DataAugmentation=augmenter);
    options=classifierTrainingOptions(cfg,executionEnvironment,preprocessingEnvironment,miniBatch);
    net=trainnet(trainDS,net,lossFcn,options);
    oofRaw(va)=predictElevatedScores(net,images(:,:,:,va));
    fprintf("Finished CV fold %d/%d.\n",f,cfg.cv.folds);
end
if any(~isfinite(oofRaw))
    error("WoundWatch:IncompleteOOF","Some observations did not receive out-of-fold predictions.");
end

%% Fit Platt calibration from OOF scores and calculate cross-fitted calibration metrics
% Use logit(raw probability) as the single Platt predictor. For unbiased
% calibration diagnostics, each row's diagnostic probability is produced by
% a Platt model fit on the other CV folds. The final deployable Platt model
% is then fitted on all OOF scores.
rawLogit=logitSafe(oofRaw);
yNumeric=double(Y=="Elevated");
oofCalibratedCV=nan(n,1);
for f=1:cfg.cv.folds
    fitRows=foldID~=f; heldRows=foldID==f;
    modelFold=fitglm(rawLogit(fitRows),yNumeric(fitRows), ...
        Distribution="binomial",Link="logit");
    oofCalibratedCV(heldRows)=predict(modelFold,rawLogit(heldRows));
end
plattModel=fitglm(rawLogit,yNumeric,Distribution="binomial",Link="logit");
calibrationParameters=plattModel.Coefficients.Estimate(:)';
calibratedOOF=min(max(predict(plattModel,rawLogit),0),1);
brierScore=mean((oofCalibratedCV-yNumeric).^2);

%% OOF discrimination, bootstrap AUROC interval, and decision thresholds
[auroc,auprc]=classificationMetrics(yNumeric,oofRaw);
prevalence=mean(yNumeric);
aucCI=bootstrapAUC(yNumeric,oofRaw,cfg.cv.bootstrapReplicates,cfg.seed+1);
[sensAt90,specAt90,thresholdAt90Raw]=thresholdAtSensitivity(yNumeric,oofRaw,0.90);
[youdenRaw,youdenSens,youdenSpec]=youdenThreshold(yNumeric,oofRaw);

% Thresholds are selected on cross-fitted calibrated probabilities because
% deployment predictions are calibrated probabilities.
[youdenCal,youdenCalSens,youdenCalSpec]=youdenThreshold(yNumeric,oofCalibratedCV);
[sensRuleCal,sensRuleSens,sensRuleSpec]=thresholdAtSensitivity(yNumeric,oofCalibratedCV,0.90);

fprintf("\nOOF CV results (pooled predictions)\n");
fprintf("AUROC %.4f (bootstrap 95%% CI %.4f to %.4f): ranking discrimination; 0.5 is chance.\n", ...
    auroc,aucCI(1),aucCI(2));
fprintf("AUPRC %.4f; elevated prevalence baseline %.4f. AUPRC measures precision-recall performance.\n", ...
    auprc,prevalence);
fprintf("Sensitivity %.4f at specificity %.4f for >=90%% specificity rule; raw-score threshold %.6f.\n", ...
    sensAt90,specAt90,thresholdAt90Raw);
fprintf("Youden raw threshold %.6f (sensitivity %.4f, specificity %.4f).\n", ...
    youdenRaw,youdenSens,youdenSpec);
fprintf("Calibrated OOF Brier score %.5f (lower is better; cross-fitted Platt predictions).\n",brierScore);
fprintf("Final Platt intercept %.6f, slope %.6f; calibrated Youden threshold %.6f.\n", ...
    calibrationParameters(1),calibrationParameters(2),youdenCal);
fprintf("Calibrated sensitivity>=0.90 threshold %.6f (sensitivity %.4f, specificity %.4f).\n", ...
    sensRuleCal,sensRuleSens,sensRuleSpec);
fprintf("A calibration plot compares predicted probability with observed elevated rate by score bin.\n");

%% Save out-of-fold predictions and calibration plot
results=table(filePaths,T.split,T.label,Y,foldID,oofRaw,oofCalibratedCV,calibratedOOF, ...
    VariableNames=["file","split","severityLabel","binaryLabel","fold", ...
    "oofElevatedProbability","crossFittedCalibratedProbability","fullFitCalibratedProbability"]);
writetable(results,fullfile(cfg.paths.results,"cv_results.csv"));
plotCalibration(oofCalibratedCV,yNumeric,fullfile(cfg.paths.results,"calibration_plot.png"));

%% Train final model on all manifest data and save network plus deployment parameters
nLow=nnz(Y=="Low"); nElev=nnz(Y=="Elevated");
finalWeights=single([n/(2*nLow),n/(2*nElev)]);
finalLoss=dlaccelerate(@(P,Targ)crossentropy(P,Targ,finalWeights));
finalNet=imagePretrainedNetwork(backbone,NumClasses=2);
finalAugmenter=imageDataAugmenter( ...
    RandRotation=cfg.augmentation.rotationDegrees, ...
    RandXReflection=cfg.augmentation.horizontalReflection, ...
    RandYReflection=cfg.augmentation.verticalReflection, ...
    RandXTranslation=cfg.augmentation.translationPixels, ...
    RandYTranslation=cfg.augmentation.translationPixels);
finalDS=augmentedImageDatastore(imageSize(1:2),images,Y,DataAugmentation=finalAugmenter);
finalOptions=classifierTrainingOptions(cfg,executionEnvironment,preprocessingEnvironment,miniBatch);
finalNet=trainnet(finalDS,finalNet,finalLoss,finalOptions);

thresholds=struct();
thresholds.youden=youdenCal;
thresholds.sensitivityAtLeast090=sensRuleCal;
thresholds.sensitivityAtLeast090AchievedSensitivity=sensRuleSens;
thresholds.sensitivityAtLeast090AchievedSpecificity=sensRuleSpec;
thresholds.rawYouden=youdenRaw;
thresholds.rawSensitivityAtLeast090=thresholdAt90Raw;
calibration=struct("method","Platt logistic calibration of logit(raw Elevated probability)", ...
    "intercept",calibrationParameters(1),"slope",calibrationParameters(2), ...
    "model",plattModel,"brierScoreCrossFitted",brierScore);
cvMetrics=struct("AUROC",auroc,"AUROC95CI",aucCI,"AUPRC",auprc, ...
    "ElevatedPrevalence",prevalence,"SensitivityAt90Specificity",sensAt90, ...
    "SpecificityAt90Rule",specAt90,"CrossFittedCalibratedBrier",brierScore);
save(cfg.paths.infectionClassifierModel,"finalNet","backbone","imageSize", ...
    "thresholds","calibration","calibrationParameters","finalWeights","cfg","cvMetrics","-v7.3");
fprintf("\nSaved final classifier: %s\n",cfg.paths.infectionClassifierModel);
fprintf("Saved OOF CV rows: %s\n",fullfile(cfg.paths.results,"cv_results.csv"));
fprintf("Saved calibration plot: %s\n",fullfile(cfg.paths.results,"calibration_plot.png"));

%% Local functions
function foldID=stratifiedFoldIDs(Y,nFolds,seed)
% Deterministic round-robin allocation after shuffling within each class.
if ~iscategorical(Y) || ~isscalar(nFolds) || nFolds<2 || nFolds~=floor(nFolds)
    error("WoundWatch:InvalidFoldConfiguration", ...
        "Y must be categorical and nFolds must be an integer of at least 2.");
end
classNames=categories(Y);
if numel(classNames)<2
    error("WoundWatch:InsufficientClasses","Stratified folds require at least two classes.");
end
classCounts=countcats(Y);
if any(classCounts<nFolds)
    error("WoundWatch:InsufficientClassCount", ...
        "Every class must contain at least %d samples for stratified folds.",nFolds);
end
rng(seed,"twister");
foldID=zeros(numel(Y),1);
for c=1:numel(classNames)
    classRows=find(Y==classNames{c});
    classRows=classRows(randperm(numel(classRows)));
    foldID(classRows)=mod((0:numel(classRows)-1)',nFolds)+1;
end
if any(foldID==0)
    error("WoundWatch:FoldAllocationFailed","Local stratified fold assignment left samples unassigned.");
end
end

function image=readResizeRGB(path,outputSize)
[A,map]=imread(path);
if ~isempty(map)
    A=im2uint8(ind2rgb(A,map));
elseif ndims(A)==2
    A=repmat(A,1,1,3);
elseif size(A,3)==4
    A=A(:,:,1:3);
elseif size(A,3)~=3
    error("WoundWatch:UnsupportedImage","Expected grayscale or RGB image: %s",path);
end
A=imresize(A,outputSize);
if ~isa(A,"uint8"), A=im2uint8(A); end
image=A;
end

function options=classifierTrainingOptions(cfg,execEnv,preprocessEnv,miniBatch)
options=trainingOptions("adam", ...
    MaxEpochs=cfg.training.classifierMaxEpochs, ...
    MiniBatchSize=miniBatch, ...
    InitialLearnRate=cfg.training.initialLearnRate, ...
    Shuffle="every-epoch", ...
    Verbose=false, ...
    Plots="none", ...
    ExecutionEnvironment=execEnv, ...
    PreprocessingEnvironment=preprocessEnv);
end

function scores=predictElevatedScores(net,images)
n=size(images,4); scores=zeros(n,1);
for i=1:n
    p=gather(extractdata(predict(net,single(images(:,:,:,i)))));
    p=squeeze(p);
    if numel(p)~=2
        error("WoundWatch:UnexpectedNetworkOutput", ...
            "Expected two network outputs (Low, Elevated); received %d values.",numel(p));
    end
    scores(i)=double(p(2));
end
scores=min(max(scores,eps),1-eps);
end

function z=logitSafe(p)
p=min(max(double(p(:)),1e-7),1-1e-7);
z=log(p./(1-p));
end

function [auc,auprc]=classificationMetrics(y,scores)
[~,~,~,auc]=perfcurve(y,scores,1);
[~,~,~,auprc]=perfcurve(y,scores,1,xCrit="reca",yCrit="prec");
end

function ci=bootstrapAUC(y,scores,nBoot,seed)
rng(seed,"twister");
positive=find(y==1); negative=find(y==0);
if isempty(positive) || isempty(negative)
    error("WoundWatch:OneClass","AUROC bootstrap requires both classes.");
end
values=nan(nBoot,1);
for b=1:nBoot
    idx=[positive(randi(numel(positive),numel(positive),1)); ...
         negative(randi(numel(negative),numel(negative),1))];
    [~,~,~,values(b)]=perfcurve(y(idx),scores(idx),1);
end
ci=prctile(values,[2.5 97.5]);
end

function [threshold,sensitivity,specificity]=youdenThreshold(y,scores)
[candidates,sens,spec]=thresholdStatistics(y,scores);
[~,i]=max(sens+spec-1);
threshold=candidates(i); sensitivity=sens(i); specificity=spec(i);
end

function [threshold,sensitivity,specificity]=thresholdAtSensitivity(y,scores,target)
[candidates,sens,spec]=thresholdStatistics(y,scores);
eligible=find(sens>=target);
if isempty(eligible)
    error("WoundWatch:ThresholdNotFound","No threshold reaches sensitivity %.3f.",target);
end
bestSpec=max(spec(eligible));
best=eligible(spec(eligible)==bestSpec);
% For ties, choose the highest threshold for the more conservative decision.
[~,j]=max(candidates(best));
i=best(j);
threshold=candidates(i); sensitivity=sens(i); specificity=spec(i);
end

function [thresholds,sensitivity,specificity]=thresholdStatistics(y,scores)
y=logical(y(:)); scores=double(scores(:));
thresholds=[-Inf; unique(scores); Inf];
sensitivity=zeros(numel(thresholds),1); specificity=sensitivity;
nPositive=nnz(y); nNegative=nnz(~y);
for i=1:numel(thresholds)
    predicted=scores>=thresholds(i);
    sensitivity(i)=nnz(predicted & y)/nPositive;
    specificity(i)=nnz(~predicted & ~y)/nNegative;
end
end

function plotCalibration(probability,y,path)
probability=min(max(probability(:),0),1); y=double(y(:));
nBins=min(10,numel(y));
[sortedProb,order]=sort(probability);
binID=zeros(size(probability));
binID(order)=ceil((1:numel(y))'*nBins/numel(y));
meanPredicted=nan(nBins,1); observedRate=nan(nBins,1); counts=zeros(nBins,1);
for b=1:nBins
    use=binID==b;
    if any(use)
        meanPredicted(b)=mean(probability(use));
        observedRate(b)=mean(y(use));
        counts(b)=nnz(use);
    end
end
fig=figure(Visible="off");
plot([0 1],[0 1],"k--","LineWidth",1); hold on;
scatter(meanPredicted,observedRate,24+4*sqrt(counts),"filled");
xlim([0 1]); ylim([0 1]); axis square; grid on;
xlabel("Mean predicted infection-risk probability");
ylabel("Observed elevated-label fraction");
title("WoundWatch calibration (cross-fitted Platt scores)");
saveas(fig,path); close(fig);
end

