import os

import json_numpy

json_numpy.patch()

import numpy as np
import requests


URL = os.getenv("OPENVLA_URL", "http://127.0.0.1:8000/act")

payload = {
    "image": np.zeros((224, 224, 3), dtype=np.uint8),
    "instruction": "move the robot arm forward",
    "unnorm_key": os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig"),
}

response = requests.post(URL, json=payload, timeout=120)
print(response.status_code)
print(response.text)
