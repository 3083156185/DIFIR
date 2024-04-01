import base64
import json
import urllib
import urllib.request

import cv2
import numpy as np


# 用来存储数据
def get_json_data(json_path):
    # 获取json里面数据
    with open(json_path, 'rb') as f:
        # 定义为只读模型，并定义名称为f
        params = json.load(f)
        # 加载json文件中的内容给params
        params['GPUID'] = (params['GPUID'] + 1) % 1
        dict = params
        # 将修改后的内容保存在dict中
    f.close()
    # 关闭json读模式
    return dict
    # 返回dict字典内容


def write_json_data(json_path, dict):
    # 写入json文件
    with open(json_path, 'w') as r:
        # 定义为写模式，名称定义为r
        json.dump(dict, r)
        # 将dict写入名称为r的文件中
    r.close()
    # 关闭json写模式


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def decode_image(base64_data):
    imgData = base64.b64decode(base64_data)
    nparr = np.fromstring(imgData, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def person_feature(person_engine, inputs, functions):
    output = {
        "context": {},
        "outputs": []
    }
    for input in inputs:
        image_url = input["image_url"]
        image_base64 = input["image_base64"]
        if not image_url and not image_base64:
            continue

        if image_url:
            img = url_to_image(image_url)
        else:
            img = decode_image(image_base64)

        result = {"objects": []}
        if "person" in functions["detection_type"]:
            persons = person_engine.get(img)
            # print("persons:{}".format(persons))
            if not persons:
                result["objects"] = []
            result = persons
        output["outputs"].append(result)
    return output
def face_feature(face_engine, inputs, functions):
    output = {
        "context": {},
        "outputs": []
    }
    for input in inputs:
        image_url = input["image_url"]
        image_base64 = input["image_base64"]
        if not image_url and not image_base64:
            continue

        if image_url:
            img = url_to_image(image_url)
        else:
            img = decode_image(image_base64)

        result = {"objects": []}
        if "face" in functions["detection_type"]:
            faces = face_engine.get(img, functions["feature_extract"])
            if not faces:
                result["objects"] = []
            for face in faces:
                xmin, ymin, w, h = face.bbox
                gender = 'Male'
                if face.gender == 0:
                    gender = 'Female'
                # print("face.embedding:{}___{}".format(type(face.embedding), face.embedding.shape))
                result["objects"].append(
                    {"type": "face",
                     "bbox": {
                         "x": int(xmin),
                         "y": int(ymin),
                         "w": int(w),
                         "h": int(h)
                     },
                     "confidence": float(face.confidence),
                     "feature": {
                         "feature_dim": face.embedding.shape[0],
                         "feature_content": base64.b64encode(face.embedding).decode()
                     },
                     "landmark": {
                         "pitch": float(face.pose[0, 0]),
                         "yaw": float(face.pose[1, 0]),
                         "roll": float(face.pose[2, 0])
                     },
                     "attributes": [{
                         "name": "gender",
                         "confidence": 0.88,
                         "value": str(gender)
                     },
                         {
                             "name": "age",
                             "confidence": 0.88,
                             "value": str(face.age)
                         },
                         {
                             "name": "glass",
                             "confidence": 0.9978376030921936,
                             "value": "noglass"
                         },
                         {
                             "name": "hat",
                             "confidence": 0.38203516602516174,
                             "value": "nohat"
                         },
                         {
                             "name": "race",
                             "confidence": 0.9891579151153564,
                             "value": "norace"
                         },
                         {
                             "name": "mask",
                             "confidence": 0.9997710585594177,
                             "value": "nomask"
                         },
                         {
                             "name": "color",
                             "confidence": 0.7762066125869751,
                             "value": "other"
                         }
                     ]
                     })

        output["outputs"].append(result)
    return output
