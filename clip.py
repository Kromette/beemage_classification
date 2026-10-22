import os
import random

from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


def main():
    hornet_dataset_path = "data/hornet3000+/"

    # 1. Load the model and processor
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    # 2. Prepare image and text labels
    train_images_folder = os.path.join(hornet_dataset_path, "data3000/train/images")
    all_species = os.listdir(train_images_folder)
    sample_images = {}
    for species in all_species:
        species_images_path = os.path.join(train_images_folder, species)
        sample_images[species] = []
        filenames = list(os.listdir(species_images_path))
        random.shuffle(filenames)
        for filename in filenames[:15]:
            sample_images[species].append(os.path.join(species_images_path, filename))

    image_path = sample_images["Vespa_velutina"][0]
    print(f"Image path: {image_path}")
    image = Image.open(image_path)
    
    labels = [
        "a photo of a vespa velutina (Asian hornet)", 
        "a photo of a vespa crabro (European hornet)", 
        "a photo of a vespula vulgaris (Common wasp)",
        ]

    # 3. Process inputs
    inputs = processor(
        text=labels, images=image, return_tensors="pt", padding=True
    )

    # 4. Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Get probabilities via softmax
    logits_per_image = outputs.logits_per_image  # Image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    print("Label probabilities:", probs)


if __name__ == "__main__":
    main()
