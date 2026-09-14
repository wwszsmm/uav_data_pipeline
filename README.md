# UAV Vision Data Pipeline

[中文说明](README_zh.md)

A configurable UAV data preprocessing pipeline for converting raw videos into cleaned, traceable image datasets using multiprocessing, CNN feature extraction, and FAISS similarity search.

The project was developed from an internship workflow. The public repository contains code, evaluation summaries, and non-sensitive result figures only. Internal UAV data and model weights are not included. The repository also includes sanitized evaluation artifacts from a downstream YOLO-based road segmentation experiment.

## Pipeline

```text
UAV Videos
    ↓
Video Collection and Renaming
    ↓
Frame Extraction
    ↓
Image Gathering and Traceable Renaming
    ↓
CNN/FAISS Similarity Deduplication
    ↓
Image Integrity Check
    ↓
Pipeline Summary Report
```

## Main Features

- Recursively collects videos from nested folders.
- Generates stable filenames to avoid naming conflicts.
- Extracts frames with multiprocessing.
- Supports restart-safe frame extraction using completion markers.
- Gathers images into a unified directory and records source mappings.
- Uses CNN features and FAISS similarity search for image deduplication.
- Caches image features to avoid unnecessary repeated computation.
- Checks for empty or unreadable images.
- Generates a summary report for processed data.
- Uses a centralized configuration file for paths and processing parameters.

## Project Structure

```text
.
├── run_pipeline.py
├── video_copy.py
├── frame_extraction.py
├── image_gather.py
├── image_deduplication.py
├── check_images.py
├── generate_report.py
├── config.example.py
├── pipeline_config.py
├── pipeline_utils.py
├── requirements.txt
├── constraints-windows-py312.txt
├── tests/
└── model_result/
    ├── English_evaluation.md
    ├── Chinese_evaluation.md
    ├── error_analysis.csv
    ├── results.csv
    ├── results.png
    ├── MaskPR_curve.png
    └── confusion_matrix_normalized.png
```

## Installation

The tested baseline is **Windows x64, Python 3.12, CPU PyTorch**. Other OS/Python/CUDA combinations have not been validated. Create and activate a virtual environment before installing.

1. Clone the repository:

```bash
git clone https://github.com/wwszsmm/uav_data_pipeline.git
cd uav_data_pipeline
```

2. Install the required dependencies:

```bash
python -m pip install -r requirements.txt -c constraints-windows-py312.txt
```

3. Create a local configuration file from the example:

```bash
cp config.example.py config.py
```

Update the paths and parameters in `config.py` according to your local environment.

4. Run the full pipeline:

```bash
python run_pipeline.py
```

Individual modules can also be run separately if needed.

## Configuration

Copy the example configuration:

```powershell
Copy-Item config.example.py config.py
```

Then update the input path and processing parameters in `config.py`.

Important parameters include:

- `INPUT_PATH`
- `FRAME_INTERVAL`
- `MAX_COPY_WORKERS`
- `MAX_CUT_PROCESSES`
- `MIN_SIMILARITY_THRESHOLD`
- `BATCH_SIZE`
- `MAX_THREADS`
- `USE_GPU`

The local `config.py` is excluded from Git because it may contain machine-specific paths.

## Run the Pipeline

```powershell
python run_pipeline.py
```

The pipeline runs the configured stages in sequence and stops if one stage fails.

Individual modules can also be run separately:

```powershell
python video_copy.py
python frame_extraction.py
python image_gather.py
python image_deduplication.py
python check_images.py
python generate_report.py
```

## Runtime and recovery contract

- Paths in the selected configuration are resolved relative to that file. By default use `config.py`; set `UAV_CONFIG` to an absolute configuration-file path to use another configuration.
- Input and output roots must be separate and non-nested. Stage output directories must be distinct children of `OUTPUT_ROOT`. Use one process per output root; keep inputs and configuration unchanged during a run.
- **Migration:** use a new, empty `OUTPUT_ROOT` when upgrading from the original scripts. Non-empty legacy stage directories without an ownership manifest are rejected, preserving the original data. Do not delete manifests to force reuse.
- Copies are installed atomically and compared by SHA-256. Each stage records its current owned files. Changed/removed inputs are reflected in current outputs; only previously owned, unchanged obsolete files can be pruned. Unrelated files are not deleted.
- Frame snapshots are keyed by video content, interval and decoder configuration. Reuse requires intact recorded images. Old snapshots remain on disk, but gathering and reporting use only the current manifest. Storage can therefore grow after input/parameter changes.
- Failed writes, empty input and reported decoding shortfalls stop the pipeline. OpenCV does not always supply a reliable total frame count; successful decoding is not proof that every damaged source is detectable.
- Feature caches include image content hashes, model identity and package versions. A mismatch or damaged cache triggers rebuilding. Encoding uses the model's own preprocessing. The similarity threshold still needs validation on representative UAV images.
- Unreadable images are excluded from deduplicated output and listed in `deduped_images/rejected_images.json`. The final integrity check must pass. `pipeline_status.json` and `pipeline_report.md` distinguish failure, completion and completion with exclusions; duplicate rate uses valid input images as its denominator.
- Standalone modules require current manifests from their preceding stages. Re-run the full pipeline after changing inputs. Standalone reports cannot certify a new run, and mark changed configuration/data as unverified.
- `USE_GPU` controls video decoding only; standard `opencv-python` falls back to CPU if CUDA decoding is unavailable. CNN inference independently selects CUDA when supported by the installed PyTorch.
- The first CNN cache miss may download approximately 10 MB of public MobileNet ImageNet weights. For offline use, prepare that model cache first with `python -c "from imagededup.methods.cnn import CNN; CNN()"`, then preserve the PyTorch cache or set `TORCH_HOME`. Private segmentation weights are not required for preprocessing. `ultralytics` is not a runtime dependency of these six stages; the archived segmentation results are not regenerated by this command.

### Validation

Install test dependencies in the same virtual environment:

```bash
python -m pip install -r requirements-dev.txt -c constraints-windows-py312.txt
python -m pip check
python -m pytest -q
```

The default suite avoids model downloads and skips the full model test. To run the real CPU integration test with the prepared model cache, in PowerShell:

```powershell
$env:UAV_RUN_MODEL_TEST = "1"
python -m pytest -q
```

The integration test creates temporary synthetic videos and checks all six stages, cache reuse, same-name video replacement, input addition/removal and stale-report detection. It does not validate private UAV data quality, GPU execution or the archived YOLO metrics.

## Road Segmentation Evaluation

A YOLO-based segmentation model was evaluated on four road-related classes.

Key validation results:

- Mask mAP@0.5: approximately **0.902**
- Mask mAP@0.5:0.95: approximately **0.778**

![Training Results](model_result/results.png)

![Mask Precision-Recall Curve](model_result/MaskPR_curve.png)

![Normalized Confusion Matrix](model_result/confusion_matrix_normalized.png)

Manual review found that:

- Clear and medium-to-large road regions were generally segmented well.
- The main model errors were missed detections of distant or small regions.
- Some reference annotations were incomplete because classes had been labeled in separate tasks.
- In several reviewed cases, model predictions were consistent with the visible scene even when the reference annotation did not include every target.

Detailed summaries are available in:

- [`model_result/English_evaluation.md`](model_result/English_evaluation.md)
- [`model_result/Chinese_evaluation.md`](model_result/Chinese_evaluation.md)
- [`model_result/error_analysis.csv`](model_result/error_analysis.csv)

## Limitations

- This repository represents an engineering prototype, not a production deployment.
- Internal UAV images, complete datasets, and model weights are not included.
- Annotation incompleteness may affect the interpretation of validation metrics.
- Distant and small target regions remain the main weakness of the current model.
- Feature caches rebuild automatically when image content or the recorded model environment changes.

## Privacy

The public version does not include company data, customer information, internal server details, credentials, or proprietary model weights.
