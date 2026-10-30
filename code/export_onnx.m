% export_onnx.m
% Export the trained MATLAB classifier and segmenter to ONNX.
thisDir=fileparts(mfilename("fullpath")); addpath(thisDir);
cfg=config();
if ~isfile(cfg.paths.infectionClassifierModel)
    error("WoundWatch:MissingClassifier","Train the classifier first: %s",cfg.paths.infectionClassifierModel);
end
if ~isfile(cfg.paths.segmentationModel)
    error("WoundWatch:MissingSegmenter","Train the segmenter first: %s",cfg.paths.segmentationModel);
end
C=load(cfg.paths.infectionClassifierModel,"finalNet");
S=load(cfg.paths.segmentationModel,"net");
if ~isfield(C,"finalNet") || ~isa(C.finalNet,"dlnetwork") || ...
        ~isfield(S,"net") || ~isa(S.net,"dlnetwork")
    error("WoundWatch:InvalidModelFile","Expected finalNet and net dlnetwork variables in model MAT files.");
end
try
    exportONNXNetwork(C.finalNet,cfg.paths.onnxClassifier);
    exportONNXNetwork(S.net,cfg.paths.onnxSegmenter);
catch ME
    error("WoundWatch:ONNXExportFailed", ...
        "ONNX export failed. Check Deep Learning Toolbox Converter for ONNX Model Format is installed. Details: %s",ME.message);
end
fprintf("Exported classifier and segmenter ONNX files under the configured models folder.\n");
