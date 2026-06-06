import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Download stopwords
nltk.download('stopwords')

# ============================================================
# 2. LOAD DATASET
# ============================================================

# CHANGE PATH IF NEEDED
df = pd.read_csv(
    r"C:\Users\DELL\Downloads\archive\spam.csv",
    encoding='latin-1'
)

# Keep only needed columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("Dataset Shape:", df.shape)
print(df.head())

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

stop_words = set(stopwords.words('english'))

def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join(
        [char for char in text if char not in string.punctuation]
    )

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return ' '.join(tokens)

# Apply preprocessing
df['message'] = df['message'].apply(preprocess)

# Convert labels into numeric values
# ham = 0
# spam = 1
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nPreprocessing Completed!")

# ============================================================
# 4. FEATURE EXTRACTION USING TF-IDF
# ============================================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['message'])

y = df['label']

print("\nTF-IDF Shape:", X.shape)

# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ============================================================
# 6. MODEL TRAINING
# ============================================================

# Model 1
nb_model = MultinomialNB()

# Model 2
lr_model = LogisticRegression(max_iter=1000)

# Train models
nb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

print("\nModels Trained Successfully!")

# ============================================================
# 7. PREDICTIONS
# ============================================================

# Naive Bayes Predictions
nb_pred = nb_model.predict(X_test)

# Logistic Regression Predictions
lr_pred = lr_model.predict(X_test)

# Probabilities for ROC-AUC
nb_prob = nb_model.predict_proba(X_test)[:, 1]
lr_prob = lr_model.predict_proba(X_test)[:, 1]

# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_model(y_test, y_pred, y_prob, model_name):

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n================================================")
    print(f"{model_name} RESULTS")
    print("================================================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1, roc_auc

# ============================================================
# 9. MODEL EVALUATION
# ============================================================

nb_results = evaluate_model(
    y_test,
    nb_pred,
    nb_prob,
    "Naive Bayes"
)

lr_results = evaluate_model(
    y_test,
    lr_pred,
    lr_prob,
    "Logistic Regression"
)

# ============================================================
# 10. CROSS VALIDATION
# ============================================================

print("\n================================================")
print("CROSS VALIDATION")
print("================================================")

nb_cv = cross_val_score(
    nb_model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

lr_cv = cross_val_score(
    lr_model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("\nNaive Bayes Cross Validation Accuracy:")
print(nb_cv)
print("Average:", round(nb_cv.mean(), 4))

print("\nLogistic Regression Cross Validation Accuracy:")
print(lr_cv)
print("Average:", round(lr_cv.mean(), 4))

# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(6, 6))

cm = confusion_matrix(y_test, lr_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Ham', 'Spam']
)

disp.plot(ax=ax)

plt.title("Confusion Matrix - Logistic Regression")

plt.show()

# ============================================================
# 12. ROC CURVE
# ============================================================

fig, ax = plt.subplots(figsize=(6, 6))

RocCurveDisplay.from_estimator(
    lr_model,
    X_test,
    y_test,
    ax=ax
)

plt.title("ROC Curve - Logistic Regression")

plt.show()

# ============================================================
# 13. MODEL COMPARISON BAR CHART
# ============================================================

models = ['Naive Bayes', 'Logistic Regression']

accuracy_scores = [nb_results[0], lr_results[0]]

f1_scores = [nb_results[3], lr_results[3]]

x = np.arange(len(models))

width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(
    x - width/2,
    accuracy_scores,
    width,
    label='Accuracy'
)

ax.bar(
    x + width/2,
    f1_scores,
    width,
    label='F1 Score'
)

ax.set_xlabel('Models')

ax.set_ylabel('Scores')

ax.set_title('Model Comparison')

ax.set_xticks(x)

ax.set_xticklabels(models)

ax.legend()

plt.show()

# ============================================================
# 14. CUSTOM MESSAGE TESTING
# ============================================================

sample = [
    "Congratulations! You won a free iPhone. Click now!"
]

sample_processed = [preprocess(sample[0])]

sample_vector = vectorizer.transform(sample_processed)

prediction = lr_model.predict(sample_vector)[0]

probability = lr_model.predict_proba(sample_vector)[0][1]

print("\n================================================")
print("CUSTOM MESSAGE TEST")
print("================================================")

print("Message:", sample[0])

print("Prediction:",
      "Spam" if prediction == 1 else "Ham")

print("Spam Probability:",
      round(probability, 4))

# ============================================================
# 15. REUSABLE FUNCTION
# ============================================================

def check_spam(message):

    processed = preprocess(message)

    vector = vectorizer.transform([processed])

    result = lr_model.predict(vector)[0]

    prob = lr_model.predict_proba(vector)[0][1]

    return (
        "Spam" if result == 1 else "Ham",
        prob
    )

# ============================================================
# 16. INTERACTIVE TESTING
# ============================================================

print("\n================================================")
print("LIVE SPAM DETECTOR")
print("================================================")

while True:

    msg = input("\nEnter Message (type 'exit' to stop): ")

    if msg.lower() == 'exit':
        print("Program Ended!")
        break

    result, prob = check_spam(msg)

    print("Prediction:", result)

    print("Spam Probability:", round(prob, 4))

# ============================================================
# 17. BATCH TESTING
# ============================================================

test_messages = [

    "Win a free iPhone now!!!",

    "Meeting at 10 AM tomorrow",

    "Get cash reward instantly",

    "Let's study for exam",

    "Congratulations! Claim your prize now!",

    "Your bank account is secured"
]

print("\n================================================")
print("BATCH TESTING")
print("================================================")

for msg in test_messages:

    result, prob = check_spam(msg)

    print("\nMessage:", msg)

    print("Prediction:", result)

    print("Spam Probability:", round(prob, 4))

    print("-" * 50)

# ============================================================
# 18. FINAL CONCLUSION
# ============================================================

print("\n================================================")
print("FINAL CONCLUSION")
print("================================================")

if lr_results[3] > nb_results[3]:

    print("Logistic Regression performed better based on F1 Score.")

else:

    print("Naive Bayes performed better based on F1 Score.")

print("\nProject Completed Successfully!")
