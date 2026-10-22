# BeeFree Species Classification POC

## 1. Overview

This project is a proof of concept (POC) for the BeeFree application developed by the *Des abeilles et nous* association.

The goal is to evaluate image-based approaches for identifying insect species from photographs submitted by users. The initial scope focuses on three species:

- `Vespa velutina` — Asian hornet
- `Vespa crabro` — European hornet
- `Vespula vulgaris` — Common wasp

The POC compares two approaches:

1. **Zero-shot image classification with CLIP**, without fine-tuning on the target dataset.
2. **Image-level classification derived from YOLO object detections**, using pretrained YOLO models.

The initial benchmark assumes that each image contains one target species. The objective is to establish a baseline before considering fine-tuning, additional classes, or integration into BeeFree.

## 2. Dataset and pretrained models

### Hornet3000+

The POC uses the publicly available Hornet3000+ dataset as a starting point.

- **Dataset:** [Hornet3000+ on Kaggle](https://www.kaggle.com/datasets/marcoryvandijk/vespa-velutina-v-crabro-vespulina-vulgaris/data)
- **Associated GitHub repository:** [vespCV/hornet3000](https://github.com/vespCV/hornet3000/tree)

The dataset and repository provide a useful starting point for evaluating approaches to insect recognition. The GitHub repository includes pretrained YOLOv8s and YOLOv10n weights.

According to the available project information, the pretrained YOLO models cover all three species targeted by this POC:

- `Vespa velutina`
- `Vespa crabro`
- `Vespula vulgaris`

They do **not** include the following classes relevant to the broader BeeFree application:

- Bee
- `Other insect`

Consequently, the existing models can be evaluated for the initial three-species benchmark, but they cannot yet provide complete classification coverage for all possible insect photographs submitted to BeeFree.

### Dataset limitations

Hornet3000+ is a starting point, not necessarily representative of the photographs that BeeFree users will submit in real-world conditions.

Potential limitations include:

- Differences in image quality, framing, lighting, and backgrounds.
- Differences between the dataset distribution and real user-submitted photographs.
- Limited class coverage compared with the eventual BeeFree requirements.
- Potential overlap or near-duplicates between existing dataset splits.

The initial results should therefore be interpreted as a baseline rather than as a reliable estimate of real-world application performance.

## 3. Approaches under evaluation

### 3.1. CLIP zero-shot classification

The initial CLIP baseline uses [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) through Hugging Face Transformers.

CLIP compares an image with text descriptions of the target species. No fine-tuning is performed for this initial experiment.

Example prompts:

- `a photograph of a Vespa velutina (Asian hornet)`
- `a photograph of a Vespa crabro (European hornet)`
- `a photograph of a Vespula vulgaris (common wasp)`

The model selects the species with the highest score among the three prompts.

The resulting scores are relative to the candidate prompts and should not be interpreted as calibrated probabilities that a prediction is correct.

### 3.2. YOLO detection-derived image classification

The second approach uses the pretrained YOLOv8s and YOLOv10n models provided by the Hornet3000 project.

Unlike CLIP, YOLO is an object-detection model: it predicts bounding boxes, class labels, and confidence scores for detected objects.

To compare its output with CLIP's image-level predictions, we derive a single prediction per image:

1. Run YOLO on the full image.
2. Keep detections belonging to the three target species.
3. Select the relevant detection with the highest confidence score.
4. Use its species label as the image-level prediction.
5. If no relevant detection is found, assign `no_detection`.

The `no_detection` outcome is counted as an incorrect prediction when evaluating images that are known to contain one of the three target species.

The class-ID mapping must be read from each checkpoint's metadata rather than hardcoded, as class ordering may differ between models.

### 3.3. Comparison

The initial benchmark compares three configurations:

| Approach                         | Model         | Prediction method                                    |
| -------------------------------- | ------------- | ---------------------------------------------------- |
| Zero-shot classification         | CLIP ViT-B/32 | Highest-scoring species prompt for the full image    |
| Detection-derived classification | YOLOv8s       | Species of the highest-confidence relevant detection |
| Detection-derived classification | YOLOv10n      | Species of the highest-confidence relevant detection |

This is a comparison between two different approaches to image-level prediction, not between models trained for exactly the same task: CLIP performs image–text matching, whereas YOLO performs object detection.

## 4. Evaluation methodology

All approaches should be evaluated on the same held-out test images with known ground-truth labels.

### Metrics

The following metrics will be reported:

- **Accuracy:** proportion of images assigned the correct species.
- **Macro-F1:** unweighted average of the F1 scores across the three species.
- **Per-class precision and recall:** to identify species-specific weaknesses.
- **Confusion matrix:** to inspect which species are most frequently confused.

For YOLO, `no_detection` counts as an incorrect prediction. It should also be reported separately as a diagnostic outcome, since a failure to detect an insect differs from detecting it but assigning the wrong species.

YOLO detection confidence scores and CLIP scores are not directly comparable and should not be treated as equivalent measures of certainty.

### Data splitting and leakage prevention

- Use a fixed test set that is not used for model training, fine-tuning, prompt selection, or threshold tuning.
- Keep near-duplicate images out of different splits wherever possible.
- Use the same test images for all approaches.
- Record the model checkpoint, configuration, and evaluation results to make experiments reproducible.

## 5. Current scope and future work

### Current scope

The initial POC focuses exclusively on the three target species:

- `Vespa velutina`
- `Vespa crabro`
- `Vespula vulgaris`

It evaluates pretrained models without fine-tuning them on a new BeeFree-specific dataset.

### Future work

Potential next steps include:

- Collecting and annotating photographs representative of actual BeeFree submissions.
- Fine-tuning models on a curated dataset and comparing them against the initial baselines.
- Adding bees and an `Other insect` class.
- Considering an explicit unknown or out-of-scope outcome for images that do not contain one of the target species.
- Evaluating robustness to image quality, backgrounds, viewing angles, and lighting conditions.
- Assessing inference latency and deployment constraints for server-side or mobile use.
- Investigating approaches that combine detection and classification if the initial results justify the added complexity.

The three-class benchmark is an initial milestone, not the final classification system required by BeeFree.

## 6. Setup

First, install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

Install the main Python dependencies:

```bash
uv sync
```

Download the Hornet3000+ dataset from Kaggle and obtain the pretrained YOLO checkpoints from the associated GitHub repository.

Check the repository documentation for the exact checkpoint filenames, model versions, and usage instructions. Verify that each checkpoint loads correctly with the installed Ultralytics version and that its class names match the expected species.

The CLIP model will be downloaded automatically by Hugging Face Transformers when first loaded, unless it is already available in the local cache.

## 7. Project status

**Status:** Initial proof of concept.

The first objective is to establish a reproducible baseline comparing CLIP zero-shot classification with image-level predictions derived from the pretrained YOLOv8s and YOLOv10n detection models.

The results will help determine whether fine-tuning, additional classes, or a more elaborate inference pipeline is justified for BeeFree.
