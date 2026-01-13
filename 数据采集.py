import requests

url = "https://v1.jinrishici.com/all.json"

response = requests.get(url)

result = response.json()

print(result["content"])
print(result["origin"])
print(result["author"])
print(result["category"])