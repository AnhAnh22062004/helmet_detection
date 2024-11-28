import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import glob
from PIL import Image
from predict_function import predict_with_yolo
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from predict_function import predict_with_sahi

models = {
    "YOLO V11m": YOLO(r"D:\UIT\Năm2\CV&UD\DOAN\yolo11m_best.pt"),
    "YOLO V8m": YOLO(r"D:\UIT\Năm2\CV&UD\DOAN\yolo8m_best.pt"),
    "YOLO V5m": YOLO(r"D:\UIT\Năm2\CV&UD\DOAN\yolo5m_best.pt"),
}

sahi_models = {
    "SAHI YOLO V11m": AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=r"D:\UIT\Năm2\CV&UD\DOAN\yolo11m_best.pt",
        confidence_threshold=0.5,
        device="cpu"
    ),
    "SAHI YOLO V8m": AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=r"D:\UIT\Năm2\CV&UD\DOAN\yolo8m_best.pt",
        confidence_threshold=0.5,
        device="cpu"
    ),
    "SAHI YOLO V5m": AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=r"D:\UIT\Năm2\CV&UD\DOAN\yolo5m_best.pt",
        confidence_threshold=0.5,
        device="cpu"
    ),
}


categories = ["no helmet", "helmet"]
st.set_page_config(layout="wide")


st.title("Helmet Detection App")


uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


conf_threshold = st.slider("Confidence Threshold:", min_value=0.01, max_value=1.0, value=0.5, step=0.01)


if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image)

    # YOLO Detection
    st.subheader("Detection with YOLO Models")
    yolo_columns = st.columns(len(models))

    for col, (model_name, model) in zip(yolo_columns, models.items()):
        with col:
            st.write(model_name)
            annotated_image = predict_with_yolo(model, image_np, conf_threshold)
            annotated_image_pil = Image.fromarray(annotated_image)
            st.image(annotated_image_pil, caption=f"{model_name}")

    st.subheader("Detection with SAHI Models")
    sahi_columns = st.columns(len(sahi_models))

    for col, (model_name, sahi_model) in zip(sahi_columns, sahi_models.items()):
        with col:
            st.write(model_name)
            annotated_image = predict_with_sahi(image_np, sahi_model, categories)
            annotated_image_pil = Image.fromarray(annotated_image)
            st.image(annotated_image_pil, caption=f"{model_name}")
