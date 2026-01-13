import os
import json
import requests

os.makedirs("images",exist_ok=True)

with open("image.json",encoding="utf-8",) as f:
    urls = json.load(f)["images"]

for url in urls:
    filename = url.split("/")[-1]
    print(filename)

    content = requests.get(url).content

    with open(f"images/{filename}",'wb') as f:
        f.write(content)