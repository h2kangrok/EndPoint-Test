import time
import serial
import requests
import numpy
import io
from pprint import pprint
from requests.auth import HTTPBasicAuth
from collections import Counter
import cv2

colors = {'BOOTSEL': (255, 0, 0), 'USB': (255, 125, 0), 'CHIPSET': (
    255, 255, 0), 'OSCILLATOR': (125, 255, 0), 'RASBERRY PICO': (0, 255, 0), 'HOLE': (0, 0, 255)}
classes = {'BOOTSEL': 1, 'USB': 1, 'CHIPSET':  1,
           'OSCILLATOR': 1, 'RASBERRY PICO': 1, 'HOLE': 4}

ser = serial.Serial("/dev/ttyACM0", 9600)

api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/fdba7b1b-2e1b-4329-9ad2-9a8fcaa9bd26/inference"


def make_textbox(image, text, startx, starty, color, font_color, font_size=0.5, font_thick=1):
    text_size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thick)
    text_width, text_height = text_size
    padding = 5
    x1_rect = startx
    y1_rect = starty - text_height - padding
    x2_rect = startx + text_width + padding
    y2_rect = starty + padding
    cv2.rectangle(image, (x1_rect, y1_rect), (x2_rect, y2_rect), color, -1)
    cv2.putText(image, text, (startx, starty), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thick, cv2.LINE_AA)


def edgeDetecting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(img, 0.8, edges_colored, 0.2, 0)
    return result


def true_conditions(result_dictionary):
    why_false = ''
    falseList = []
    is_goodPD = True
    if sum(count_dict.values()) != sum(classes.values()):
        is_goodPD = False
        for key in classes.keys():
            if key not in result_dictionary.keys():
                result_dictionary[key] = 0
            if key in result_dictionary:
                if result_dictionary[key] != classes[key]:
                    falseList.append(key)
        if is_goodPD == False:
            why_false = f'Missing parts = {falseList}'
    return is_goodPD, why_false


def get_img():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera Error")
        exit(-1)
    ret, img = cam.read()
    cam.release()
    return img


def crop_img(img, size_dict):
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    img = img[y: y + h, x: x + w]
    return img


a = 0

while 1:
    data = ser.read()
    print(data)
    if data == b"0":
        img = get_img()
        crop_info = {"x": 200, "y": 100, "width": 400, "height": 400}
        if crop_info is not None:
            img = crop_img(img, crop_info)
            img = edgeDetecting(img)

        _, img_encoded = cv2.imencode(".jpg", img)
        height, width = img.shape[:2]

        response = requests.post(
            url=api_url,
            auth=HTTPBasicAuth(
                "kdt2024_1-25", "tgQ2dmUkRR83hlNogvILG2eiD0Cfay6G52Hb8P7R"),
            headers={"Content-Type": "image/jpeg"},
            data=io.BytesIO(img_encoded),
        )
        aaa = response.json()
        print(aaa)
        count_dict = dict(Counter(d["class"] for d in aaa['objects']))

        is_goodPD, why_false = true_conditions(count_dict)

        for obj in aaa['objects']:
            print(obj)
            box = obj['box']
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])
            color = colors[obj['class']]
            thickness = 2
            cv2.rectangle(img, start_point, end_point, color, thickness)
            text = obj['class'] + ' score :' + str(round(obj['score'], 2))
            make_textbox(img, text, box[0], box[1],
                         color, (255, 255, 255), 0.5)

        y1 = 20
        x1 = 0
        for key in count_dict.keys():
            text = key + ":" + str(count_dict[key])
            make_textbox(img, text, x1, y1, (0, 255, 0), (0, 0, 0))
            y1 += 20

        text = "total :" + str(sum(count_dict.values()))
        make_textbox(img, text, x1, y1, (0, 255, 0), (0, 0, 0))

        text = ' True' if is_goodPD else 'False'
        text += ' : ' + why_false
        iscolor = (0, 255, 0) if is_goodPD else (0, 0, 255)
        cv2.rectangle(img, (0, 0), (width, height), iscolor, 2)
        make_textbox(img, text, 0, height - 5, iscolor, (0, 0, 0), 0.5)

        filename = f"/home/rokey/Workspace/PicsForLearning/{a}.jpg"
        cv2.imwrite(filename, img)
        cv2.imshow("", img)
        cv2.waitKey(1)

        ser.write(b"1")
        a += 1
    else:
        pass
