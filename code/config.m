% config.m
function cfg = config()
% CONFIG.M  Central WoundWatch paths, model names, and run settings.
% All paths are derived from this file's location; no dataset absolute path is
% embedded here. Run from any current folder with cfg=config().

codeDir = fileparts(mfilename("fullpath"));
cfg.root = fileparts(codeDir);
cfg.compatibility.minimumMATLABRelease = "R2024b";
cfg.compatibility.developmentMATLABRelease = "R2026a";
cfg.paths.code = codeDir;
cfg.paths.data = fullfile(cfg.root,"data");
cfg.paths.raw = fullfile(cfg.paths.data,"raw");
cfg.paths.processed = fullfile(cfg.paths.data,"processed");
cfg.paths.surgwound = fullfile(cfg.paths.raw,"surgwound");
cfg.paths.surgwoundImages = fullfile(cfg.paths.surgwound,"images");
cfg.paths.surgwoundManifest = fullfile(cfg.paths.surgwound,"infection_risk.csv");
cfg.paths.dfuc = fullfile(cfg.paths.raw,"DFUC2021");
cfg.paths.kaggle = fullfile(cfg.paths.raw,"Kaggle","Wound_dataset copy");
cfg.paths.medetec = fullfile(cfg.paths.raw,"Medetec","medetec-dataset");
cfg.paths.fuseg = fullfile(cfg.paths.raw,"FUSeg","Foot Ulcer Segmentation Challenge");
cfg.paths.fusegExternalTest = fullfile(cfg.paths.raw,"FUSeg","Medetec_foot_ulcer_224","test");
cfg.paths.models = fullfile(cfg.root,"models");
cfg.paths.results = fullfile(cfg.root,"results");
cfg.paths.qualityGateModel = fullfile(cfg.paths.models,"quality_gate.mat");
cfg.paths.segmentationModel = fullfile(cfg.paths.models,"segmenter.mat");
cfg.paths.infectionClassifierModel = fullfile(cfg.paths.models,"infection_classifier.mat");
cfg.paths.onnxClassifier = fullfile(cfg.paths.models,"classifier.onnx");
cfg.paths.onnxSegmenter = fullfile(cfg.paths.models,"segmenter.onnx");
cfg.paths.classifierCache = fullfile(cfg.paths.processed,"classifier_images_224.mat");
cfg.paths.auditCSV = fullfile(cfg.paths.results,"data_audit.csv");

cfg.image.classifierSize = [224 224 3];
cfg.image.segmentationSize = [256 256 3];
cfg.image.qualityGateSize = [224 224 3];
cfg.seed = 20261031;
cfg.cv.folds = 5;
cfg.cv.bootstrapReplicates = 2000;
cfg.training.classifierMaxEpochs = 12;
cfg.training.segmentationMaxEpochs = 30;
cfg.training.miniBatchSizeGPU = 8;
cfg.training.miniBatchSizeCPU = 2;
cfg.training.segmentationMiniBatchSizeGPU = 4;
cfg.training.segmentationMiniBatchSizeCPU = 1;
cfg.training.initialLearnRate = 1e-4;
cfg.training.validationFraction = 0.2;
cfg.training.preprocessingEnvironment = "background";
cfg.augmentation.rotationDegrees = [-15 15];
cfg.augmentation.translationPixels = [-8 8];
cfg.augmentation.horizontalReflection = true;
cfg.augmentation.verticalReflection = true;
cfg.audit.nearDuplicateHammingDistance = 6;
cfg.audit.nearDuplicateResize = [32 32];
cfg.audit.extensions = [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif"];

% Initial quality-gate values are tuning starting points, not clinical cutoffs.
cfg.quality.blurVarianceMinimum = 1e-4;
cfg.quality.clippingLowIntensity = 0.01;
cfg.quality.clippingHighIntensity = 0.99;
cfg.quality.maximumClippedFraction = 0.05;
cfg.quality.minimumWoundAreaFraction = 0.002;
cfg.quality.maximumWoundAreaFraction = 0.80;
cfg.quality.checkColorCard = false;

% Deployment decision and ONNX parity settings.
cfg.inference.thresholdName = "youden";
cfg.inference.deferHalfWidth = 0.05;
cfg.onnx.parityTolerance = 1e-3;
cfg.onnx.verificationImageCount = 20;

% Only model identifiers documented for imagePretrainedNetwork are listed.
cfg.models.qualityGate = "mobilenetv2";
cfg.models.segmentationBackbone = "resnet18";
cfg.models.infectionClassifier = "resnet18";
cfg.models.allowedImagePretrainedNetwork = ["resnet18","resnet50","resnet101", ...
    "efficientnetb0","mobilenetv2","densenet201","googlenet","inceptionv3", ...
    "xception","squeezenet"];

% Create standard project output folders when config is called.
requiredFolders = [cfg.paths.code,cfg.paths.models,cfg.paths.results, ...
    cfg.paths.raw,cfg.paths.processed];
for k = 1:numel(requiredFolders)
    if ~isfolder(requiredFolders(k))
        [ok,msg] = mkdir(requiredFolders(k));
        if ~ok
            error("WoundWatch:FolderCreateFailed", ...
                "Could not create folder '%s': %s",requiredFolders(k),msg);
        end
    end
end
end

