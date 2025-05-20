import pickle

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)
