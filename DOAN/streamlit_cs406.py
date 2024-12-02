import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import glob
from PIL import Image
import tempfile
from predict_function import predict_with_yolo
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from predict_function import predict_with_sahi

models = {
    "YOLO V11m": YOLO(r"yolo11m_best.pt"),
    "YOLO V8m": YOLO(r"yolo8m_best.pt"),
    "YOLO V5m": YOLO(r"yolo5m_best.pt"),
}

sahi_models = {
    "SAHI YOLO V11m": AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=r"yolo11m_best.pt",
        confidence_threshold=0.6,
        device="cpu"
    ),
    "SAHI YOLO V8m": AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=r"yolo8m_best.pt",
        confidence_threshold=0.6,
        device="cpu"
    ),
    "SAHI YOLO V5m": AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=r"yolo5m_best.pt",
        confidence_threshold=0.6,
        device="cpu"
    ),
}


categories = ["no helmet", "helmet"]
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stSlider > div[data-baseweb="slider"] > div {
        border-radius: 10px;
        height: 10px;
    }
    .stSlider > div[data-baseweb="slider"] > div > div[role="slider"] {
        background-color: #ff6347;
        border: 2px solid #1e90ff;
        height: 10px;
        width: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Helmet Detection App")


uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])


conf_threshold = st.slider("Confidence Threshold:", min_value=0.01, max_value=1.0, value=0.5, step=0.01)


if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image)

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


if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.subheader("Original Video")
    st.video(video_path)

    st.subheader("Detection on Video")
    selected_model = st.selectbox("Select YOLO Model for Video:", list(models.keys()))

    if st.button("Process Video"):
        model = models[selected_model]
        cap = cv2.VideoCapture(video_path)
        output_frames = []

        stframe = st.empty() 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            annotated_frame = predict_with_yolo(model, frame_rgb, conf_threshold)

            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            stframe.image(annotated_frame, channels="RGB")

            output_frames.append(annotated_frame_bgr)

        cap.release()

        if output_frames:
            height, width, _ = output_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = "output_video.mp4"
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

            for frame in output_frames:
                out.write(frame)
            out.release()

            st.subheader("Processed Video")
            st.video(output_path)
