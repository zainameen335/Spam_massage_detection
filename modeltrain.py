import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

os.makedirs("model", exist_ok=True)

data = pd.read_csv("dataset.csv")

X = data["message"]
y = data["label"]

model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X, y)

pickle.dump(model, open("model/spam_model.pkl", "wb"))

print("Model trained and saved!")