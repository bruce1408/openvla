import json
import urllib.request
from pathlib import Path

IMAGE_COUNT = 100
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "test_data"

api = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=VyoJ%2FBridgeData-V2-Scripted-Images"
    f"&config=default&split=train&offset=0&length={IMAGE_COUNT}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(api, timeout=60) as response:
    rows = json.load(response)["rows"]

for index, item in enumerate(rows):
    image = item["row"]["first_image"]
    output = OUTPUT_DIR / f"bridge_sample_{index+1:04d}.jpg"

    if output.exists():
        print(f"[{index + 1}/{len(rows)}] skipped: {output}")
        continue

    with urllib.request.urlopen(image["src"], timeout=60) as response:
        output.write_bytes(response.read())

    print(f"[{index + 1}/{len(rows)}] saved: {output}")