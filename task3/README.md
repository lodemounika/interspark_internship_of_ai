# Spam Detection API using Flask and Docker

## Project Description

This project was developed as part of an AI Internship assignment. The objective was to deploy a Machine Learning model using Flask and create an API that can predict whether a given message is Spam or Ham (Not Spam).

The model is trained on an SMS Spam Collection dataset using TF-IDF Vectorization and Logistic Regression. After training, the model is integrated with a Flask application to provide real-time predictions through API endpoints.

In addition, Docker support has been included to make the application easy to deploy across different environments.

---

## Features

* Spam Message Classification
* REST API using Flask
* Real-Time Prediction
* JSON Request and Response Handling
* Docker Containerization
* Easy Local Deployment

---

## Technologies Used

* Python
* Flask
* Scikit-Learn
* Pandas
* NumPy
* NLTK
* Docker

---

## Project Structure

task3_api_deployment/

├── app.py

├── train_model.py

├── spam_model.pkl

├── vectorizer.pkl

├── requirements.txt

├── Dockerfile

├── report.pdf

├── demo_video.mp4

└── README.md

---

## Model Information

### Machine Learning Algorithm

* Logistic Regression

### Feature Extraction Technique

* TF-IDF Vectorization

### Prediction Classes

* Spam
* Ham (Not Spam)

---

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Flask Application

```bash
python app.py
```

After execution, the application will start on:

```text
http://127.0.0.1:5000
```

---

## API Endpoints

### Home Endpoint

**GET /**

Response:

```text
Spam Detection API Running Successfully!
```

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

## Testing Using cURL

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"message\":\"Congratulations! You won a free iPhone\"}"
```

Example Output:

```json
{
  "message": "Congratulations! You won a free iPhone",
  "prediction": "Ham",
  "spam_probability": 0.4431
}
```

---

## Docker Commands

### Build Docker Image

```bash
docker build -t spam-api .
```

### Run Docker Container

```bash
docker run -p 5000:5000 spam-api
```

After running the container, open:

```text
http://localhost:5000
```

---

## Demo Video

A complete demonstration of the project execution is included in this repository.

File:

```text
demo_video.mp4
```

The video demonstrates:

* Running the Flask Application
* Testing the Home Endpoint
* Sending Prediction Requests
* Viewing JSON Responses
* Spam Detection Results

---

## Conclusion

This project helped me understand the complete deployment workflow of a Machine Learning model. I learned how to train a model, save it, integrate it with Flask APIs, and prepare it for deployment using Docker. The project also provided practical experience in handling API requests and returning predictions in JSON format.

---

## Author

L Mounika 
