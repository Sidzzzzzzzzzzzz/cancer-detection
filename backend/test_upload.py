import requests

# Change filename to an actual file you have
files = {"file": open("test.png","rb")}
resp = requests.post("http://127.0.0.1:5000/predict/breast", files=files)
print("Status:", resp.status_code)
print("Response:", resp.text)



