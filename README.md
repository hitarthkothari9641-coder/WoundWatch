# WoundWatch

MATLAB R2026a development workspace for research on surgical-wound image
classification and wound segmentation. The MATLAB scripts use documented
functions and target R2024b or newer. This is an experimental project, not a
diagnostic tool; see [MODEL_CARD.md](MODEL_CARD.md).

## MATLAB run order

In MATLAB, set the current folder to this project root, then run:

```matlab
addpath("code")
cfg = config();
audit_data
```

Review `results/data_audit.csv`, then train the two required models (either
order is fine):

```matlab
train_classifier
train_segmentation
```

Run pipeline tests and make ONNX exports after both models have been trained:

```matlab
runtests("code/test_pipeline.m")
export_onnx
verify_onnx
```

To score one image from MATLAB after training:

```matlab
result = predict_ssi("path-to-an-image.jpg")
```

Training requires Deep Learning Toolbox, Image Processing Toolbox, Computer
Vision Toolbox, Statistics and Machine Learning Toolbox, pretrained ResNet-18
weights, and the data arranged as described in `code/config.m`. ONNX import and
export require Deep Learning Toolbox Converter for ONNX Model Format. Model
files and generated reports are intentionally excluded from Git.

## Python ONNX service

Build from the repository root:

```powershell
docker build -f service/Dockerfile -t woundwatch-service .
```

Mount the locally exported ONNX files read-only at `/models/classifier.onnx`
and `/models/segmenter.onnx`. Configure `PLATT_INTERCEPT`, `PLATT_SLOPE`, and
`DECISION_THRESHOLD` from the trained `infection_classifier.mat` artifact,
then start the container. The API exposes `GET /health` and `POST /predict`
(multipart field `file`). See `service/app.py` and `MODEL_CARD.md` for the
preprocessing and privacy contract.

The parity suite `service/test_parity.py` compares service results with a CSV
of MATLAB `predict_ssi` outputs; its expected columns are shown in
`service/matlab_parity_template.csv`. Both the model files and reference images
must be available to the test process.

## Data and outputs

`data/`, `models/`, and `results/` are local-only and ignored by Git. Review
source permissions in the model card before moving data or derived model
weights outside the local research environment. The project does not include
patient images, trained weights, or measured metrics in its source repository.
