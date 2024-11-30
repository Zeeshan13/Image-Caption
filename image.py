import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load necessary resources
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_captioning_model():
    return load_model("RNNmodelkeras_100.keras")

@st.cache_resource
def load_vgg16_model():
    base_model = VGG16()
    return Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

# Load resources
tokenizer = load_tokenizer()
model = load_captioning_model()
vgg_model = load_vgg16_model()
max_length = 34  # Assuming max length from the model training

# Feature extraction function
def extract_features(image, model):
    image = image.resize((224, 224))  # Resize for VGG16
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)  # Preprocess for VGG16
    feature = model.predict(image, verbose=0)
    return feature

# Caption generation function
def generate_caption(model, tokenizer, photo, max_length):
    input_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        input_text += " " + word
        if word == "endseq":
            break
    return input_text.split()[1:-1]

# Streamlit App
st.title("Automatic Image Caption")
st.write("Upload an image to generate a caption!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features dynamically
    photo_features = extract_features(image, vgg_model)
    
    # Generate caption
    generated_caption = generate_caption(model, tokenizer, photo_features, max_length)
    st.write("### Generated Caption:")
    st.write(" ".join(generated_caption))

