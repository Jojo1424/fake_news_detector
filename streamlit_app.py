import streamlit as st
import pickle

# Load your trained model and vectorizer (must be in the same folder)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📰 Fake News Detector")

user_input = st.text_area("Paste a news article (title + content):")

if st.button("Predict"):
    if user_input.strip():
        # Apply the same text cleaning you used in training!
        input_clean = user_input.lower()
        input_vec = vectorizer.transform([input_clean])
        pred = model.predict(input_vec)[0]
        st.write("### Prediction:", "🟥 **FAKE NEWS**" if pred == 0 else "🟩 **TRUE NEWS**")
    else:
        st.warning("Please enter some text.")
