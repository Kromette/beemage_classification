import csv
import json
import re
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

from . import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = PROJECT_ROOT / "data" / "hornet3000+"
VAL_IMAGES_DIR = DATASET_DIR / "data3000" / "val" / "images"

YOLO_MODELS = {
    "yolov8s": PROJECT_ROOT / "models" / "yolov8s.pt",
    "yolov10n": PROJECT_ROOT / "models" / "yolov10n.pt",
    "yolov11s": PROJECT_ROOT / "models" / "yolov11s.pt"
}

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"
METRICS_JSON = RESULTS_DIR / "metrics.json"

CONF_THRESHOLD = 0.25

# Set to an integer for a quick smoke test, or None to evaluate all images.
MAX_IMAGES_PER_CLASS = None

SPECIES = [
    "Vespa_velutina",
    "Vespa_crabro",
    "Vespula_sp",
]

CLIP_PROMPTS = [
    "a photograph of a Vespa velutina (Asian hornet)",
    "a photograph of a Vespa crabro (European hornet)",
    "a photograph of a Vespula vulgaris (common wasp)",
]

# Additional names that the pretrained YOLO models use.
SPECIES_ALIASES = {
    # YOLOv8s & YOLOv11s
    "vvel": "Vespa_velutina",
    "vcra": "Vespa_crabro",
    "vespsp": "Vespula_sp",
    # The following two aliases cannot be confirmed from the hornet3000 repo and are guesses
    "amel": "Apis_mellifera",
    "vzon": "Vespa_zonata",

    # YOLOv10n
    "Vespula_vulgaris": "Vespula_sp"
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Normalize a class name for matching."""
    return re.sub(r"\s+", " ", name.replace("_", " ").strip().lower())


def canonical_species(name: str) -> str | None:
    """Map a dataset folder name or model class name to a target species."""
    if name in SPECIES:
        return name

    if name in SPECIES_ALIASES:
        return SPECIES_ALIASES[name]

    normalized = normalize_name(name)

    if normalized in SPECIES_ALIASES:
        return SPECIES_ALIASES[normalized]

    # Also support exact canonical species names.
    for species in SPECIES:
        if normalize_name(species) == normalized:
            return species

    return None


def collect_validation_images() -> list[tuple[Path, str]]:
    """Collect (image_path, ground_truth_species) pairs from class folders."""
    if not VAL_IMAGES_DIR.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found: {VAL_IMAGES_DIR}\n"
            "Check the dataset layout and update VAL_IMAGES_DIR."
        )

    samples = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for species in SPECIES:
        species_dir = VAL_IMAGES_DIR / species

        if not species_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class directory not found: {species_dir}"
            )

        image_paths = sorted(
            path
            for path in species_dir.iterdir()
            if path.is_file() and path.suffix.lower() in valid_extensions
        )

        if MAX_IMAGES_PER_CLASS is not None:
            image_paths = image_paths[:MAX_IMAGES_PER_CLASS]

        if not image_paths:
            raise ValueError(f"No images found in {species_dir}")

        samples.extend((path, species) for path in image_paths)

    return samples


# ---------------------------------------------------------------------------
# CLIP: zero-shot image classification
# ---------------------------------------------------------------------------

def load_clip(device: torch.device):
    """Load the pretrained CLIP model and processor."""
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    model.to(device)
    model.eval()

    return model, processor


@torch.inference_mode()
def predict_clip(image: Image.Image, model, processor, device):
    """Predict one species using CLIP and return its relative scores."""
    inputs = processor(
        text=CLIP_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)
    probabilities = outputs.logits_per_image.softmax(dim=1)[0]

    scores = probabilities.cpu().tolist()
    predicted_species = SPECIES[probabilities.argmax().item()]

    return predicted_species, scores


# ---------------------------------------------------------------------------
# YOLO: derive an image-level prediction from object detections
# ---------------------------------------------------------------------------

def load_yolo_models():
    """Load pretrained YOLO checkpoints and validate their classes."""
    models = {}

    for model_name, checkpoint in YOLO_MODELS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"YOLO checkpoint not found: {checkpoint}"
            )

        model = YOLO(str(checkpoint))

        # Check the actual class names rather than assuming class IDs.
        names = model.names
        print(f"\n{model_name} class names: {names}")

        recognized_species = {
            canonical_species(str(class_name))
            for class_name in names.values()
        }
        recognized_species.discard(None)

        missing_species = set(SPECIES) - recognized_species

        if missing_species:
            raise ValueError(
                f"{model_name} does not expose all target species. "
                f"Missing: {sorted(missing_species)}. "
                f"Actual class names: {names}"
            )

        models[model_name] = model

    return models


@torch.inference_mode()
def predict_yolo(image: Image.Image, model):
    """
    Return the highest-confidence target species detected in an image.

    If no target species is detected, return 'no_detection'.
    """
    result = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        verbose=False,
    )[0]

    best_species = "no_detection"
    best_confidence = 0.0

    if result.boxes is None:
        return best_species, best_confidence

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        class_name = model.names[class_id]
        species = canonical_species(str(class_name))

        if species is not None and confidence > best_confidence:
            best_species = species
            best_confidence = confidence

    return best_species, best_confidence


# ---------------------------------------------------------------------------
# Evaluation and reporting
# ---------------------------------------------------------------------------

def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute image-level classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=SPECIES,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=SPECIES,
            target_names=SPECIES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=SPECIES + ["no_detection"],
        ).tolist(),
    }


def save_predictions(rows: list[dict]):
    """Save per-image predictions and confidence scores."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No predictions to save.")

    with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    samples = collect_validation_images()
    print(f"Number of validation images: {len(samples)}")

    # Load models.
    clip_model, clip_processor = load_clip(device)
    yolo_models = load_yolo_models()

    y_true = []
    predictions = {
        "clip": [],
        **{
            key: [] for key in YOLO_MODELS
        }
    }

    rows = []

    for index, (image_path, ground_truth) in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {image_path.name}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        # CLIP prediction on the full image.
        clip_species, clip_scores = predict_clip(
            image,
            clip_model,
            clip_processor,
            device,
        )

        row = {
            "image": str(image_path.relative_to(PROJECT_ROOT)),
            "ground_truth": ground_truth,
            "clip_prediction": clip_species,
            **{
                f"clip_score_{species}": score
                for species, score in zip(SPECIES, clip_scores)
            },
        }

        predictions["clip"].append(clip_species)

        # YOLO predictions on the same full image.
        for model_name, model in yolo_models.items():
            species, confidence = predict_yolo(image, model)

            predictions[model_name].append(species)
            row[f"{model_name}_prediction"] = species
            row[f"{model_name}_confidence"] = confidence

        y_true.append(ground_truth)
        rows.append(row)

    # Save per-image results.
    save_predictions(rows)

    # Evaluate all approaches on the same images.
    metrics = {}

    for model_name, y_pred in predictions.items():
        metrics[model_name] = evaluate(y_true, y_pred)

        # Report the no-detection rate for YOLO.
        if model_name.startswith("yolo"):
            metrics[model_name]["no_detection_count"] = y_pred.count(
                "no_detection"
            )
            metrics[model_name]["no_detection_rate"] = (
                y_pred.count("no_detection") / len(y_pred)
            )

        print(f"\n=== {model_name} ===")
        print(f"Accuracy: {metrics[model_name]['accuracy']:.4f}")
        print(f"Macro-F1: {metrics[model_name]['macro_f1']:.4f}")

        if model_name.startswith("yolo"):
            print(
                "No-detection rate: "
                f"{metrics[model_name]['no_detection_rate']:.2%}"
            )

    with METRICS_JSON.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nPredictions saved to: {PREDICTIONS_CSV}")
    print(f"Metrics saved to: {METRICS_JSON}")


if __name__ == "__main__":
    main()
