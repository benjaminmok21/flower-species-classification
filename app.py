import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Flower Species Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Flower Species Classifier")
st.write(
    "Upload a flower photo and the model will predict which of the 17 flower classes it most closely matches."
)

st.info(
    "Important: this model only knows the 17 flower classes used in the Oxford 17 Flowers dataset. "
    "If you upload a flower outside these classes, the model will still be forced to choose one of them."
)

# -----------------------------
# Class names
# The order must match the labels used during model training.
# -----------------------------
CLASS_NAMES = [
    "Daffodil",
    "Snowdrop",
    "Lily Valley",
    "Bluebell",
    "Crocus",
    "Iris",
    "Tigerlily",
    "Tulip",
    "Fritillary",
    "Sunflower",
    "Daisy",
    "Colts Foot",
    "Dandelion",
    "Cowslip",
    "Buttercup",
    "Windflower",
    "Pansy"
]

MODEL_PATH = "mobilenetv2_flower_classifier.keras"
IMAGE_SIZE = (224, 224)

# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource
def load_flower_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_flower_model()
except Exception as e:
    st.error("The model could not be loaded.")
    st.exception(e)
    st.stop()

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    # Resize to the same size used during training.
    resized_image = image.resize(IMAGE_SIZE)

    # Convert the image into numbers for the model.
    image_array = np.array(resized_image, dtype=np.float32)

    # Add a batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    image_batch = np.expand_dims(image_array, axis=0)

    # Important:
    # We do NOT apply MobileNetV2 preprocessing here because the saved model
    # already contains the preprocessing step used during training.
    predictions = model.predict(image_batch, verbose=0)[0]

    # Get the three highest-scoring predictions.
    top_3_indices = np.argsort(predictions)[-3:][::-1]

    best_index = top_3_indices[0]
    best_class = CLASS_NAMES[best_index]
    best_score = float(predictions[best_index])

    st.subheader("Prediction")
    st.success(f"{best_class} — {best_score:.1%}")

    st.subheader("Top 3 predictions")

    for index in top_3_indices:
        class_name = CLASS_NAMES[index]
        score = float(predictions[index])
        st.write(f"**{class_name}:** {score:.1%}")
        st.progress(min(max(score, 0.0), 1.0))

    st.caption(
        "The percentage shown is the model's confidence among the 17 classes it knows. "
        "It should not be treated as a guarantee that the flower truly belongs to that class."
    )
