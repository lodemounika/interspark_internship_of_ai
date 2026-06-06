# Spam Detection API using Flask and Docker

## Project Overview

This project was developed as part of an AI Internship assignment. The objective of this project is to deploy a Machine Learning Spam Detection model using Flask and provide real-time predictions through a REST API.

The model is trained using SMS spam messages and can classify a given message as either Spam or Ham (Not Spam). The API accepts user input in JSON format and returns the prediction along with the spam probability score.

Docker support is also included to make the application portable and easy to deploy.

---

## Objectives

* Build a Spam Detection model using Machine Learning.
* Deploy the trained model using Flask.
* Create API endpoints for prediction.
* Return results in JSON format.
* Containerize the application using Docker.

---

## Technologies Used

* Python
* Flask
* Scikit-Learn
* Pandas
* NumPy
* NLTK
* TF-IDF Vectorization
* Logistic Regression
* Docker

---

## Project Structure

```text
task3_api_deployment/

├── app.py
├── train_model.py
├── spam_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── Dockerfile
├── report.pdf
├── screenshots/
│   ├── flask_server.png
│   └── api_testing.png
└── README.md
```

---

## Model Information

### Algorithm Used

* Logistic Regression

### Feature Extraction

* TF-IDF Vectorization

### Output Classes

* Spam
* Ham (Not Spam)

---

## Installation

### Step 1: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 2: Run the Flask Application

```bash
python app.py
```

Expected Output:

```text
* Running on http://127.0.0.1:5000
```

---

## API Endpoints

### Home Endpoint

**GET /**

Response:

```text
Spam Detection API Running Successfully!
```

---

### Prediction Endpoint

**POST /predict**

Sample Request:

```json
{
  "message": "Congratulations! You won a free iPhone"
}
```

Sample Response:

```json
{
  "message": "Congratulations! You won a free iPhone",
  "prediction": "Ham",
  "spam_probability": 0.4431
}
```

---

## Testing the API

### Test Case 1 - Normal Message

Request:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"message\":\"Hi, how are you today?\"}"
```

Response:

```json
{
  "message": "Hi, how are you today?",
  "prediction": "Ham",
  "spam_probability": 0.024
}
```

Result:

The model correctly identified the message as a normal (Ham) message.

---

### Test Case 2 - Spam Message

Request:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"message\":\"Claim your free prize now! Limited offer!\"}"
```

Response:

```json
{
  "message": "Claim your free prize now! Limited offer!",
  "prediction": "Spam",
  "spam_probability": 0.8756
}
```

Result:

The model successfully identified the message as Spam with a high probability score.

---

## Flask Server Output

The Flask application was executed successfully.

```text
python intern_app1.py

* Serving Flask app 'intern_app1'
* Debug mode: on
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5000
* Running on http://192.168.43.238:5000
Press CTRL+C to quit
```

Successful API Requests:

```text
127.0.0.1 - - [06/Jun/2026 15:18:14] "POST /predict HTTP/1.1" 200 -
127.0.0.1 - - [06/Jun/2026 15:18:19] "POST /predict HTTP/1.1" 200 -
127.0.0.1 - - [06/Jun/2026 15:18:39] "POST /predict HTTP/1.1" 200 -
```

HTTP Status Code 200 confirms that the API processed the requests successfully.

---

## Docker Setup

### Build Docker Image

```bash
docker build -t spam-api .
```

### Run Docker Container

```bash
docker run -p 5000:5000 spam-api
```

Access the application at:

```text
http://localhost:5000
```

---

## Screenshots

The repository contains screenshots demonstrating:

* Flask server execution
* Home endpoint response
* API testing using cURL
* Prediction results

Example:

```text
screenshots/
├── flask_server.png
└── api_testing.png
```

---

## Features

* Spam Detection
* REST API Deployment
* JSON Request and Response
* Real-Time Prediction
* Docker Containerization
* Machine Learning Model Integration

---

## Learning Outcomes

Through this project, I learned:

* How to train and save a Machine Learning model.
* How to create REST APIs using Flask.
* How to load and use trained models for inference.
* How to process JSON requests and responses.
* How Docker can be used for application deployment.

---

## Conclusion

This project successfully demonstrates the deployment of a Machine Learning Spam Detection model using Flask. The API can classify incoming messages as Spam or Ham and return the results in JSON format. The project provides a complete workflow from model training to deployment and testing.

---

## Author

L Mounika

