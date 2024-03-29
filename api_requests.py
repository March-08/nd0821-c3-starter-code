import requests

# The live API URL
url = "https://nd0821-c3-starter-code-7h05.onrender.com/inference"

# Sample data payload matching the Data model from your FastAPI app
data = {
    "age": 24,
    "workclass": "State-gov",
    "fnlgt": 584421,
    "education": "Bachelors",
    "education_num": 14,
    "marital_status": "Separated",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 3,
    "capital_loss": 2,
    "hours_per_week": 24,
    "native_country": "United-States",
}

response = requests.post(url, json=data)

# Printing out the response from the server
if response.status_code == 200:
    print("Success:")
    print(response.json())  # Assuming the response is JSON-formatted
else:
    print("Failed to get a successful response:")
    print(response.text)
