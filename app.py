import os.path
import json
from PIL import Image
import tensorflow as tf
import numpy as np
import streamlit as st
import requests
model_url = 'https://drive.google.com/uc?export=download&id=1a3wSgvCeI5Z8yTmJyq32GFpWXCseQz1j'
model_path='plant_disease_prediction_model.h5'

def download_model(url, file_path):
    if not os.path.exists(file_path):
        st.write("Downloading the model...")
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        st.write("Model downloaded successfully.")
download_model(model_url, model_path)

working_dir=os.path.dirname(os.path.abspath(__file__))
class_indices=json.load(open(f"{working_dir}/class_index.json"))
model=tf.keras.models.load_model(model_path)
def load_and_preprocess(img_path,target_size=(224,224)):
  img=Image.open(img_path)
  img=img.resize(target_size)
  img=np.array(img)
  img=np.expand_dims(img,axis=0)
  img=img/255.0
  return img

def predict_disease_class(model,img_path,class_idx):
  processed_img=load_and_preprocess(img_path)
  predictions=model.predict(processed_img)
  predicted_class_idx=np.argmax(predictions,axis=1)[0]
  predicted_class_name=class_idx[str(predicted_class_idx)]
  return predicted_class_name
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.pexels.com/photos/305821/pexels-photo-305821.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
        background-attachment: fixed;
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🪴 Plant Disease Classifier")
upl_img=st.file_uploader("Upload an image" , type=['JPG','jpg','jpeg','png'])
if upl_img is not None:
    image=Image.open(upl_img)
    col1,col2=st.columns(2)
    with col1:
        resized_img=image.resize((150,150))
        st.image(resized_img)
    with col2:
        if st.button("Classify"):
            prediction=predict_disease_class(model,upl_img,class_indices)
            st.success(f"Prediction : {str(prediction)}")

