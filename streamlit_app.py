import streamlit as st
import pickle
import re
import numpy as np

@st.cache_resource
def load_vectorizer_model():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_vectorizer_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check if it's likely **Fake** or **Real**.")

input_text = st.text_area("News Article Text", height=200)

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(input_text)
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        label = "Real" if prediction == 1 else "Fake"
        color = "green" if label == "Real" else "red"
        st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {np.max(proba):.2%}")
        st.write("**Class probabilities:**")
        st.write({
            "Fake": f"{proba[0]:.2%}",
            "Real": f"{proba[1]:.2%}"
        })

st.markdown("---")
st.caption("Developed for Fake News Detection Dissertation. Powered by Streamlit & Scikit-learn.")
