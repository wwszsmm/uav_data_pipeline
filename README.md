# UAV Vision Data Pipeline

[中文说明](README_zh.md)

A configurable prototype for processing UAV videos into a cleaned image dataset and evaluating a road segmentation model.

The project was developed from an internship workflow. The public repository contains code, evaluation summaries, and non-sensitive result figures only. Internal UAV data and model weights are not included.

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
└── model_result/
    ├── English_evaluation.md
    ├── Chinese_evaluation.md
    ├── error_analysis.csv
    ├── results.csv
    ├── results.png
    ├── MaskPR_curve.png
    └── confusion_matrix_normalized.png
```

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
- Feature cache rebuilding is required when the image set changes.

## Privacy

The public version does not include company data, customer information, internal server details, credentials, or proprietary model weights.
