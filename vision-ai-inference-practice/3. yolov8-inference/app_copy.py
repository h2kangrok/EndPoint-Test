# import cv2
# import gradio as gr
# import numpy as np
# from PIL import Image
# import random
# from ultralytics import YOLO
# color = []
# names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
# for i in range(len(names)):
#     j = random.randrange(0, 255)
#     k = 255 - random.randrange(0, 255)
#     l = 255 - random.randrange(0, 255) / 2
#     color.append((i, j, k))


# # 가상의 비전 AI API URL (예: 객체 탐지 API)
# VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/feed26fc-ce91-451d-89f3-8de7f4c880d9/inference"
# TEAM = "kdt2024_1-25"
# ACCESS_KEY = "tgQ2dmUkRR83hlNogvILG2eiD0Cfay6G52Hb8P7R"


# def process_image(image):
#     # Convert the image to OpenCV format
#     image = np.array(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Load YOLO model and run inference
#     model = YOLO("yolov8n.pt")
#     results = model(image)

#     # Draw results on the image
#     for result in results:
#         for con, cl, box in zip(result.boxes.conf, result.boxes.cls, result.boxes.xyxy):  # Extract coordinates for each bounding box
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(image, (x1, y1), (x2, y2), color[int(cl)], 2)
#             name = names[int(cl)] + 'score :' + str(con)

#             text_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # Get text size
#             text_width, text_height = text_size
#             padding = 5  # Add some padding around the text

#             # Coordinates for the rectangle
#             x1_rect = x1
#             y1_rect = y1 - text_height - padding
#             x2_rect = x1 + text_width + padding
#             y2_rect = y1 + padding

#             # Draw a filled rectangle behind the text
#             cv2.rectangle(image, (x1_rect, y1_rect), (x2_rect, y2_rect), color[int(cl)], -1)

#             cv2.putText(image, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


#     # Convert the image back to PIL format for Gradio
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(image)

# # Gradio 인터페이스 설정
# iface = gr.Interface(
#     fn=process_image,
#     inputs=gr.Image(type="pil"),
#     outputs="image",
#     title="Vision AI Object Detection",
#     description="Upload an image to detect objects using Vision AI.",
# )

# # 인터페이스 실행
# iface.launch(share=True)

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 가상의 비전 AI API URL (예: 객체 탐지 API)
VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/feed26fc-ce91-451d-89f3-8de7f4c880d9/inference"
TEAM = "kdt2024_1-25"
ACCESS_KEY = "tgQ2dmUkRR83hlNogvILG2eiD0Cfay6G52Hb8P7R"


def process_image(image):
    # Convert the image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Load YOLO model and run inference
    model = YOLO("yolov8n.pt")
    results = model(image)

    # YOLO가 자동으로 박스를 그리도록 처리
    annotated_image = results[0].plot()

    # Convert the image back to PIL format for Gradio
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_image)


# Gradio 인터페이스 설정
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs="image",
    title="Vision AI Object Detection",
    description="Upload an image to detect objects using Vision AI.",
)

# 인터페이스 실행
iface.launch(share=True)
