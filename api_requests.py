import requests
import json

url = "https://nd0821-c3-starter-code-7h05.onrender.com/inference"
# url = "http://localhost:8000/inference"

data = {
    "age": 62,
    "workclass": "Private",
    "fnlgt": 57346,
    "education": "Doctorate",
    "education_num": 82,
    "marital_status": "Never-married",
    "occupation": "Exec-managerial",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 140,
    "capital_loss": 0,
    "hours_per_week": 67,
    "native_country": "United-States",
}

# Convert the Python dictionary to a JSON string
data_json = json.dumps(data)

# Make the POST request, setting the correct headers for JSON content
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=data_json, headers=headers)

# Print out the response from the server
print(response)
if response.status_code == 200:
    print("Success:")
    print(response.text)  # Prints the response text directly
else:
    print("Failed to get a successful response:")
    print(response.text)
