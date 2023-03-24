import pandas as pd
import cv2
import numpy as np
import streamlit as st
import base64
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize
from tensorflow import keras
from PIL import Image
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib'
import ast
model = keras.models.load_model('best_our_model.h5')

def predict_class(img):
    prediction = model(img)
    return prediction
def process_image(img):
    if img is not None:
        image = Image.open(img)
        img = np.array(image)
        img = smart_resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img = (img - 32)/32
        img = img.reshape(1,32,32,1)
        return img
def main():
    st.set_page_config(page_title="My Streamlit App", page_icon=":memo:", layout="wide", initial_sidebar_state="expanded")
    st.title("Road object predictor")
    st.markdown('<style>div{color: Gray;}</style>', unsafe_allow_html=True)
    st.markdown('<style>h1{color: Gray;}</style>', unsafe_allow_html=True)
    img = st.file_uploader('Insert your image', type = ['png'])
    img =process_image(img)
    result = ""
    with open('future.jpg', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
    if st.button("Predict"):
        result= predict_class(img)
        classes_file = open("traffic_classes.txt", "r")
        classes = ast.literal_eval(classes_file.read())
        classes_file.close()
        predicted_classes = np.argmax(result)
        result = classes[predicted_classes]
    st.success('The sign is $ {}'.format(result))
if __name__ == '__main__':
    main()
