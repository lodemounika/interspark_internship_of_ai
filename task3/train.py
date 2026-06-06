# ============================================================
# TRAIN SPAM DETECTION MODEL
# ============================================================

import pandas as pd
import string
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(r"C:\Users\DELL\Downloads\archive\spam.csv", encoding='latin-1')

# Keep only first 2 columns
df = df.iloc[:, :2]

# Rename columns
df.columns = ['label', 'message']

print(df.head())

# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join(
        [char for char in text if char not in string.punctuation]
    )

    return text

# Apply preprocessing
df['message'] = df['message'].apply(preprocess)

# Convert labels to numbers
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# ============================================================
# TF-IDF FEATURE EXTRACTION
# ============================================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['message'])

y = df['label']

# ============================================================
# TRAIN MODEL
# ============================================================

model = LogisticRegression()

model.fit(X, y)

print("\nModel Training Completed!")

# ============================================================
# SAVE MODEL
# ============================================================

pickle.dump(
    model,
    open("spam_model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

print("\nModel Saved Successfully!")
