import requests
import cv2
from requests.auth import HTTPBasicAuth
import os

# IMAGE_FILE_PATH = "image/50.jpg"

img_folder_path = 'image/'  # 이미지 폴더 경로 설정
output_folder_path = 'result/'  # 결과 저장 폴더 경로 설정

image_counter = 1  # 번호를 1부터 시작

for filename in os.listdir(img_folder_path):
    # 파일 경로 설정
    IMAGE_FILE_PATH = os.path.join(img_folder_path, filename)
    # IMAGE_FILE_PATH = cv2.imread(img_path)
    image = open(IMAGE_FILE_PATH, "rb").read()

    ACCESS_KEY = "tgQ2dmUkRR83hlNogvILG2eiD0Cfay6G52Hb8P7R"

    response = requests.post(
        url="https://suite-endpoint-api-apne2.superb-ai.com/endpoints/feed26fc-ce91-451d-89f3-8de7f4c880d9/inference",
        auth=HTTPBasicAuth("kdt2024_1-25", ACCESS_KEY),
        headers={"Content-Type": "image/jpeg"},
        data=image,
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

    # 이미지 불러오기
    # img_path = 'image/50.jpg'  # 이미지 파일 경로 설정
    # img = cv2.imread(img_path)
    # for filename in os.listdir(img_folder_path):
    #     # 파일 경로 설정
    #     img_path = os.path.join(img_folder_path, filename)
    img = cv2.imread(IMAGE_FILE_PATH)

    class_count = 1

    for obj in response.json()["objects"]:
        # class count 세기
        class_count += 1

        point_name = obj["class"]
        score_num = obj["score"]
        box = obj["box"]

        start_point = (box[0], box[1])
        end_point = (box[2], box[3])
        thickness = 2
        color = get_color(point_name)
        cv2.rectangle(img, start_point, end_point, color, thickness)

        # 텍스트 설정
        text = f"{point_name} : {round(score_num,2)}"  # 추가할 텍스트
        position = (box[0], box[1]-10)  # 텍스트 시작 위치 (x, y)

        font = cv2.FONT_HERSHEY_SIMPLEX  # 글꼴 설정
        font_scale = 0.3  # 글자 크기
        color = get_color(point_name)
        thickness = 1  # 글자 두께
        # 텍스트 추가
        cv2.putText(img, text, position, font, font_scale,
                    color, thickness, cv2.LINE_AA)

        # output_path = "result/result_img1.jpg"

    class_count_text = str(class_count)  # 추가할 텍스트
    position = (20, 30)  # 텍스트 시작 위치 (x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 글꼴 설정
    font_scale = 1  # 글자 크기
    color = (255, 0, 0)
    thickness = 2  # 글자 두께
    # 텍스트 추가
    cv2.putText(img, class_count_text, position, font, font_scale,
                color, thickness, cv2.LINE_AA)

    output_filename = f"result_{image_counter}.jpg"
    output_path = os.path.join(output_folder_path, output_filename)
    cv2.imwrite(output_path, img)

    image_counter += 1  # 이미지 번호 증가
    # cv2.imshow("img1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# class_count 텍스트 설정


# print(response.json())


# chipset
# 박스 칠 좌표 설정 (예: 좌측 상단 (50, 50), 우측 하단 (200, 200))
# start_point = (256, 425)  # 박스 시작 좌표 (x, y)
# end_point = (312, 478)  # 박스 끝 좌표 (x, y)
# color = (0, 255, 0)  # BGR 색상 (초록색)
# thickness = 2  # 박스 선의 두께
# # 박스 그리기
# cv2.rectangle(img, start_point, end_point, color, thickness)


# [x["box"] for x in response.json()["objects"]]


# # 텍스트 설정
# text = "Hello, OpenCV!"  # 추가할 텍스트
# position = (50, 50)  # 텍스트 시작 위치 (x, y)
# font = cv2.FONT_HERSHEY_SIMPLEX  # 글꼴 설정
# font_scale = 1  # 글자 크기
# color = (0, 255, 0)  # BGR 색상 (초록색)
# thickness = 2  # 글자 두께
# # 텍스트 추가
# cv2.putText(img, text, position, font, font_scale,
#             color, thickness, cv2.LINE_AA)

# cv2.imshow("img1", img)
# cv2.waitKey(0)
