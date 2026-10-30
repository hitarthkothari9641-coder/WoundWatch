% test_pipeline.m
function tests = test_pipeline
%TEST_PIPELINE matlab.unittest tests for predict_ssi input and output behavior.
tests=functiontests(localfunctions);
end

function testBadPathRaisesClearError(testCase)
addCodePath();
verifyError(testCase,@()predict_ssi(fullfile(tempdir,"woundwatch_missing_image.jpg")), ...
    "WoundWatch:MissingImage");
end

function testGrayscaleImageAccepted(testCase)
assumeModelsExist(testCase);
path=writeTemporaryImage(repmat(uint8(linspace(20,230,256)),256,1));
cleanup=onCleanup(@()deleteIfPresent(path)); %#ok<NASGU>
result=predict_ssi(path);
verifyValidResult(testCase,result);
end

function testTinyImageAcceptedOrRetake(testCase)
assumeModelsExist(testCase);
path=writeTemporaryImage(uint8([40 80;120 200]));
cleanup=onCleanup(@()deleteIfPresent(path)); %#ok<NASGU>
result=predict_ssi(path);
verifyValidResult(testCase,result);
end

function testProbabilityAndStatusRanges(testCase)
assumeModelsExist(testCase);
rng(20261031,"twister");
path=writeTemporaryImage(uint8(randi([30 220],256,256,3)));
cleanup=onCleanup(@()deleteIfPresent(path)); %#ok<NASGU>
result=predict_ssi(path);
verifyValidResult(testCase,result);
if isfinite(result.probability)
    verifyGreaterThanOrEqual(testCase,result.probability,0);
    verifyLessThanOrEqual(testCase,result.probability,1);
else
    verifyEqual(testCase,result.status,"retake");
end
end

function verifyValidResult(testCase,result)
verifyTrue(testCase,isstruct(result));
verifyTrue(testCase,all(isfield(result,cellstr(["status","probability","risk_band","reasons"]))));
verifyTrue(testCase,ismember(string(result.status),["ok","retake","defer"]));
verifyTrue(testCase,isnumeric(result.probability)&&isscalar(result.probability));
verifyTrue(testCase,isstring(result.reasons)||ischar(result.reasons));
end

function assumeModelsExist(testCase)
addCodePath(); cfg=config();
assumeTrue(testCase,isfile(cfg.paths.infectionClassifierModel)&&isfile(cfg.paths.segmentationModel), ...
    "Train the classifier and segmenter before running model-dependent pipeline tests.");
end

function addCodePath()
folder=fileparts(mfilename("fullpath")); addpath(folder);
end

function path=writeTemporaryImage(image)
path=string([tempname ".png"]);
imwrite(image,path);
end

function deleteIfPresent(path)
if isfile(path), delete(path); end
end
