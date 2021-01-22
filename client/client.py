from __future__ import print_function
import requests
import json
import cv2
import jsonpickle
import numpy as np
import random

addr = 'http://localhost:5000'
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('/home/ravishekhartiwari/Desktop/object detection/images/demonstration-image.png')
# encode image as jpeg
_, img_encoded = cv2.imencode('.png', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print(json.loads(response.text))
print(response)


