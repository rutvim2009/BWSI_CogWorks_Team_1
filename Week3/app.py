import streamlit as st
from text_to_image import *
from load_coco import resnet18_features
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
blank_model = ImageEmbedder()
model = load_weights(blank_model, "image_embedder_weights.pkl")
txt = st.text_input("Search:", placeholder= "Type here...")

image_database = ImageDatabase(model, resnet18_features)