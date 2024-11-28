import numpy as np
import cv2
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import glob

categories = ["no helmet", "helmet"]
def predict_with_yolo(model, image, conf_thres):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 2 or image.shape[2] == 1:  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: 
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    
    results = model(image, conf=conf_thres)
    # annotated_image = np.array(results[0].plot())
    annotated_image = image.copy()

    bboxes = results[0].boxes.xyxy.cpu().numpy()  
    scores = results[0].boxes.conf.cpu().numpy() 
    labels = results[0].boxes.cls.cpu().numpy().astype(int) 

    for box, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        if label == 0:  
            label_text = f"{categories[label]}:{score:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(annotated_image, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else: 
            label_text = f"{categories[label]}:{score:.2f}"

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_image, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return annotated_image

    
model_paths = glob.glob(r"**\*.pt")
categories = ["no helmet", "helmet"]

def predict_with_sahi(image_np, detection_model, categories, slide_window_size= 512):
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_np

    result = get_sliced_prediction(
        image=image_rgb,
        detection_model=detection_model,
        slice_height=slide_window_size,
        slice_width=slide_window_size,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5,
    )

    annotated_image = image_np.copy()
    for obj in result.object_prediction_list:
        x_min, y_min, x_max, y_max = map(int, obj.bbox.to_voc_bbox())
        score = obj.score
        class_id = obj.category.id
        label_text = f"{categories[class_id]}"
        if class_id == 0:
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness=2)
            cv2.putText(annotated_image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)
            cv2.putText(annotated_image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return annotated_image

    
