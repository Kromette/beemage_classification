
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# Chemin vers la racine du dépôt
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_FILES = {
    "YOLOv8s": MODELS_DIR / "yolov8s.pt",
    "YOLOv11s": MODELS_DIR / "yolov11s.pt",
    "YOLOv10n": MODELS_DIR / "yolov10n.pt",
}

DISPLAY_NAMES = {
    "vvel": "Frelon asiatique",
    "vespa velutina": "Frelon asiatique",
    "vcra": "Frelon européen",
    "vespa crabro": "Frelon européen",
    "vespsp": "Guêpe",
    "vespula sp": "Guêpe",
    "vespula vulgaris": "Guêpe",
    "amel": "Abeille",
    "apis mellifera": "Abeille",
}

DISPLAY_LABELS = tuple(dict.fromkeys(DISPLAY_NAMES.values()))


def display_class_name(name: str) -> str:
    normalized_name = name.strip().lower().replace("_", " ")
    return DISPLAY_NAMES.get(normalized_name, "Autre")


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


st.set_page_config(
    page_title="BeeMage — Test des modèles",
    page_icon="🐝",
    layout="centered",
)

st.title("🐝 BeeMage")
st.subheader("Tester les modèles sur vos photos")

st.write(
    "Importez une photo pour tester la classification "
    "des abeilles, guêpes et frelons."
)

model_name = st.selectbox(
    "Modèle à tester",
    list(MODEL_FILES.keys()),
)

uploaded_file = st.file_uploader(
    "Choisir une photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Photo importée", use_container_width=True)

    if st.button("Analyser la photo", type="primary"):
        model_path = MODEL_FILES[model_name]

        if not model_path.exists():
            st.error(f"Modèle introuvable : {model_path.name}")
            st.stop()

        with st.spinner("Analyse en cours..."):
            model = load_model(str(model_path))
            result = model.predict(image, verbose=False)[0]

        st.subheader("Résultat")

        result.names = {
            class_id: display_class_name(class_name)
            for class_id, class_name in result.names.items()
        }

        # Affichage de l'image annotée
        annotated_image = result.plot()
        annotated_image = annotated_image[:, :, ::-1]
        st.image(
            annotated_image,
            caption="Résultat du modèle",
            use_container_width=True,
        )

        # Résultats d'un modèle de classification
        if result.probs is not None:
            probabilities = result.probs.data.cpu().tolist()
            class_names = model.names

            ranked = sorted(
                enumerate(probabilities),
                key=lambda item: item[1],
                reverse=True,
            )[:3]

            st.write("Classes les plus probables")

            for class_id, score in ranked:
                if score == 0:
                    continue

                st.write(
                    f"**{display_class_name(class_names[class_id])}** — {score:.1%}"
                )
                st.progress(float(score))

        # Résultats d'un modèle de détection
        elif result.boxes is not None:
            class_scores = {label: 0.0 for label in DISPLAY_LABELS}

            if len(result.boxes) == 0:
                st.info("Aucun objet détecté.")

            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                class_name = display_class_name(model.names[class_id])

                if class_name in class_scores and score > 0:
                    class_scores[class_name] = max(
                        class_scores[class_name], score
                    )

            nonzero_scores = {
                class_name: score
                for class_name, score in class_scores.items()
                if score > 0
            }

            if nonzero_scores:
                st.write("Scores de détection par classe")
                for class_name, score in nonzero_scores.items():
                    st.write(f"**{class_name}** — {score:.1%}")
                    st.progress(score)

        st.caption(
            "Scores de détection, pas des probabilités calibrées. "
            "La prédiction peut être erronée."
        )
