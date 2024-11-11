import cv2
import gradio as gr
import requests
import numpy as np
from PIL import Image
from requests.auth import HTTPBasicAuth
import io
import os
from ultralytics import YOLO

# 가상의 비전 AI API URL (예: 객체 탐지 API)
VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/feed26fc-ce91-451d-89f3-8de7f4c880d9/inference"
TEAM = "kdt2024_1-25"
ACCESS_KEY = "tgQ2dmUkRR83hlNogvILG2eiD0Cfay6G52Hb8P7R"


def process_image(image):
    # 이미지를 OpenCV 형식으로 변환
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # yolo 추가
    model = YOLO("yolov8n.pt")
    # Run inference with the YOLOv8n model on the 'bus.jpg' image
    results = model(image)
    print(results)

    # 이미지를 API에 전송할 수 있는 형식으로 변환
    _, img_encoded = cv2.imencode(".jpg", image)

    # API 호출 및 결과 받기 - 실습1

    response = requests.post(
        VISION_API_URL,
        auth=HTTPBasicAuth(TEAM, ACCESS_KEY),
        headers={"Content-Type": "image/jpeg"},
        # 사이트로 보낼때 byte형태
        data=io.BytesIO(img_encoded),
    )

    def get_color(point_name):
        colors = {
            "HOLE": (0, 255, 0),          # Green
            "RASPBERRY PICO": (0, 255, 255),  # Yellow
            "OSCILLATOR": (255, 0, 0),    # Blue
            "CHIPSET": (255, 255, 0),     # Cyan
            "BOOTSEL": (255, 0, 255),     # Magenta
            "USB": (0, 0, 255)            # Red
        }
        return colors.get(point_name)

    class_count = 1

    for obj in response.json()["objects"]:
        # class count 세기
        class_count += 1

        # print(obj)
        point_name = obj["class"]
        score_num = obj["score"]
        box = obj["box"]

        start_point = (box[0], box[1])
        end_point = (box[2], box[3])
        thickness = 2
        color = get_color(point_name)
        cv2.rectangle(image, start_point, end_point, color, thickness)

        # 텍스트 설정
        text = f"{point_name} : {round(score_num,2)}"  # 추가할 텍스트
        position = (box[0], box[1]-10)  # 텍스트 시작 위치 (x, y)

        font = cv2.FONT_HERSHEY_SIMPLEX  # 글꼴 설정
        font_scale = 0.3  # 글자 크기
        color = get_color(point_name)
        thickness = 1  # 글자 두께
        # 텍스트 추가
        cv2.putText(image, text, position, font, font_scale,
                    color, thickness, cv2.LINE_AA)

        class_count_text = str(class_count)  # 추가할 텍스트
        position = (20, 30)  # 텍스트 시작 위치 (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 글꼴 설정
        font_scale = 1  # 글자 크기
        color = (255, 0, 0)
        thickness = 2  # 글자 두께
        # 텍스트 추가
        cv2.putText(image, class_count_text, position, font, font_scale,
                    color, thickness, cv2.LINE_AA)

    # API 결과를 바탕으로 박스 그리기 - 실습2

    # BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


# Gradio 인터페이스 설정
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs="image",
    title="Vision AI Object Detection",
    description="Upload an image to detect objects using Vision AI.",
)

# 인터페이스 실행
iface.launch()
