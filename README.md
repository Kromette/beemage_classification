# Beemage — BeeFree Species Classification POC

## 1. Overview

Beemage is a proof of concept (POC) for image-based insect classification in the BeeFree application developed by the *Des abeilles et nous* association.

The initial objective is to compare approaches for identifying three target species from photographs:

- `Vespa_velutina` — Asian hornet
- `Vespa_crabro` — European hornet
- `Vespula_sp` — Common wasp, corresponding to *Vespula vulgaris* in the scope of this POC

The project currently compares three pretrained model configurations:

1. **CLIP ViT-B/32** — zero-shot image classification using text prompts.
2. **YOLOv8s** — image-level classification derived from object detections.
3. **YOLOv10n** — image-level classification derived from object detections.

The models are evaluated on the same validation images. No fine-tuning is performed in this initial benchmark.

The goal is to establish a reproducible baseline before considering model fine-tuning, additional insect classes, or integration into BeeFree.

## 2. Dataset and pretrained models

### 2.1. Hornet3000+

The POC uses the publicly available Hornet3000+ dataset.

- **Dataset:** [Hornet3000+ on Kaggle](https://www.kaggle.com/datasets/marcoryvandijk/vespa-velutina-v-crabro-vespulina-vulgaris/data)
- **Associated GitHub repository:** [vespCV/hornet3000](https://github.com/vespCV/hornet3000)

The dataset provides images for the three target classes used in this benchmark.

The validation images are expected to be organized as follows:

```text
data/
└── hornet3000+/
    └── data3000/
        └── val/
            └── images/
                ├── Vespa_velutina/
                ├── Vespa_crabro/
                └── Vespula_sp/
```

The exact folder names must match those expected by `baselines.py`.

**Terminology note:** The dataset uses `Vespula_sp` as the class label for the common wasp targeted by this POC. This label is retained in the evaluation to match the dataset's directory structure and the pretrained YOLO models' class mapping.

### 2.2. Pretrained YOLO checkpoints

The associated GitHub repository provides pretrained YOLOv8s and YOLOv10n checkpoints.

Download both weights from the repository root:

```bash
mkdir -p models

curl -L \
  https://raw.githubusercontent.com/vespCV/hornet3000/refs/heads/main/yolov8sModel/weights/best.pt \
  --output models/yolov8s.pt

curl -L \
  https://raw.githubusercontent.com/vespCV/hornet3000/refs/heads/main/yolov10nModel/weights/best.pt \
  --output models/yolov10n.pt
```

The expected structure is:

```text
models/
├── yolov8s.pt
└── yolov10n.pt
```

The benchmark script expects these filenames. If the checkpoints are stored elsewhere or renamed, update the corresponding paths in `baselines.py`.

### 2.3. Additional YOLOv8 classes

The YOLOv8s checkpoint exposes the following class names:

| Class name | Interpretation                                                | Evaluation status |
| ---------- | ------------------------------------------------------------- | ----------------- |
| `vvel`     | `Vespa_velutina`                                              | Evaluated         |
| `vcra`     | `Vespa_crabro`                                                | Evaluated         |
| `vespsp`   | `Vespula_sp`                                                  | Evaluated         |
| `amel`     | Appears to correspond to *Apis mellifera* (Western honey bee) | Not evaluated     |
| `vzon`     | Appears to correspond to *Vespa zonata*                       | Not evaluated     |

The interpretations of `amel` and `vzon` are inferred from their abbreviations and should be confirmed against the original model documentation or training metadata.

These additional classes are not documented in the available project information, and the current validation dataset does not provide the necessary labelled examples to evaluate their performance.

Their presence in the checkpoint's class mapping suggests that YOLOv8s may recognize additional insect species beyond the three classes benchmarked here. However, **this capability has not yet been independently verified or quantitatively evaluated**.

The current benchmark therefore remains restricted to the three target classes.

## 3. Approaches under evaluation

### 3.1. CLIP zero-shot classification

The CLIP baseline uses [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) through Hugging Face Transformers.

CLIP compares an image with text descriptions of the target species. No fine-tuning is performed.

Example prompts:

- `a photo of a vespa velutina (Asian hornet)`
- `a photo of a vespa crabro (European hornet)`
- `a photo of a vespula vulgaris (Common wasp)`

The predicted class is the one with the highest image–text similarity score.

The resulting softmax scores are relative to the candidate prompts. They should not be interpreted as calibrated probabilities that the predictions are correct.

### 3.2. YOLO detection-derived image classification

The benchmark uses the pretrained YOLOv8s and YOLOv10n object-detection models from the Hornet3000 project.

Unlike CLIP, YOLO predicts bounding boxes, class labels, and confidence scores for detected objects.

To compare the models at image level, the script derives a single prediction for each image:

1. Run YOLO on the full image.
2. Keep detections corresponding to the three target classes.
3. Select the relevant detection with the highest confidence score.
4. Use its class label as the image-level prediction.
5. If no relevant detection is found, assign `no_detection`.

The `no_detection` outcome counts as an incorrect prediction for images known to contain one of the target classes. The script also reports the no-detection rate separately.

The class mapping is obtained from each checkpoint's metadata rather than assuming a fixed class-ID ordering.

### 3.3. Comparison

| Approach                         | Model         | Prediction method                     |
| -------------------------------- | ------------- | ------------------------------------- |
| Zero-shot classification         | CLIP ViT-B/32 | Highest-scoring species prompt        |
| Detection-derived classification | YOLOv8s       | Highest-confidence relevant detection |
| Detection-derived classification | YOLOv10n      | Highest-confidence relevant detection |

This is a comparison between different approaches to image-level prediction, not between models trained for exactly the same task. CLIP performs image–text matching, whereas YOLO performs object detection.

## 4. Running the benchmark

### 4.1. Prerequisites

The project uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management.

From the repository root, install the project dependencies:

```bash
uv sync
```

Before running the benchmark, check that:

- The Hornet3000+ dataset has been downloaded and extracted.
- The validation images are in `data/hornet3000+/data3000/val/images/`.
- Both YOLO checkpoints are in `models/`.
- The project package and dependencies are correctly configured.

### 4.2. Execute the baseline comparison

Run the benchmark from the repository root:

```bash
uv run python -m beemage.baselines
```

The script evaluates CLIP, YOLOv8s, and YOLOv10n on the same validation images.

The first execution downloads the CLIP model and processor from Hugging Face if they are not already cached locally.

### 4.3. Outputs

The script saves its results in the `results/` directory:

- `results/predictions.csv` — image-level ground-truth labels and predictions for each approach.
- `results/metrics.json` — aggregate evaluation metrics.

## 5. Initial validation results

The initial benchmark was run on **297 validation images**:

- `Vespa_velutina` (100 images)
- `Vespa_crabro` (100 images)
- `Vespula_sp` (97 images)

### 5.1. Aggregate metrics

| Approach      |   Accuracy |   Macro-F1 | No-detection rate |
| ------------- | ---------: | ---------: | ----------------: |
| CLIP ViT-B/32 |     49.83% |     45.85% |               N/A |
| YOLOv8s       | **97.31%** | **97.33%** |             0.00% |
| YOLOv10n      |     95.96% |     95.98% |             0.00% |

The no-detection rate applies to the YOLO approaches. It is not directly applicable to the CLIP classifier, which always selects one of the candidate prompts.

### 5.2. Interpretation

The initial results strongly favor the pretrained YOLO models over the zero-shot CLIP baseline on this validation set.

Key observations:

- **YOLOv8s achieves the best overall results**, with 97.31% accuracy and a macro-F1 score of 97.33%.
- **YOLOv10n also performs well**, reaching 95.96% accuracy and a macro-F1 score of 95.98%.
- Neither YOLO model produced a `no_detection` outcome on the evaluated images.
- **CLIP performs substantially worse** on this three-class task, with 49.83% accuracy and a macro-F1 score of 45.85%.

These findings suggest that the pretrained YOLO checkpoints are a promising starting point for the initial BeeFree classification use case. In particular, they may be more suitable than the tested zero-shot CLIP configuration when the objective is to distinguish these three species.

However, the results should be interpreted with caution:

- The benchmark uses a validation set that may have been used for choosing the best YOLO checkpoints rather than an independent test set.&#x20;
- The images may not represent the diversity and quality of photographs submitted by BeeFree users.
- The results do not establish how the models perform on bees, other insects, or images outside the three target classes.
- A zero no-detection rate on 297 images does not establish that the models will always detect the target insects in real-world conditions.

The benchmark should therefore be considered an initial comparison, not a definitive measure of deployment performance.

## 6. Current scope and future work

### Current scope

The current POC:

- Compares CLIP ViT-B/32, YOLOv8s, and YOLOv10n.
- Evaluates three target classes using Hornet3000+ validation images.
- Uses pretrained models without additional fine-tuning.
- Derives image-level classifications from YOLO detections.
- Records predictions and aggregate metrics for reproducibility.

### Future work

Potential next steps include:

- Inspecting per-class metrics and misclassified images.
- Evaluating the models on an independent test set.
- Collecting and annotating photographs representative of actual BeeFree submissions.
- Investigating the additional `amel` and `vzon` classes exposed by the YOLOv8s checkpoint.
- Verifying the full class mapping and training metadata of both YOLO checkpoints.
- Adding bees and an `Other insect` class.
- Considering an explicit unknown or out-of-scope outcome for images that do not contain one of the target species.
- Fine-tuning models on a curated BeeFree-specific dataset.
- Evaluating inference latency and deployment constraints for server-side or mobile use.
- Assessing robustness to image quality, backgrounds, viewing angles, and lighting conditions.

The current three-class benchmark is an initial milestone, not the complete classification system required by BeeFree.

## 7. Project status

**Status: Initial proof of concept — baseline comparison completed.**

The initial validation results indicate that the pretrained YOLOv8s and YOLOv10n models substantially outperform the tested CLIP zero-shot baseline on the three target classes.

YOLOv8s currently achieves the best aggregate metrics. Further evaluation on independent, representative data is needed before drawing conclusions about real-world performance or extending the system to additional insect classes.
