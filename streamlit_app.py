import streamlit as st
import pickle

# ---- 1. FAKE NEWS DETECTOR UI ----
st.title("📰 Fake News Detector")

# Load your model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

user_input = st.text_area("Paste a news article (title + content):")
if st.button("Predict"):
    if user_input.strip():
        input_clean = user_input.lower()
        input_vec = vectorizer.transform([input_clean])
        pred = model.predict(input_vec)[0]
        st.write("### Prediction:", "🟥 **FAKE NEWS**" if pred == 0 else "🟩 **TRUE NEWS**")
    else:
        st.warning("Please enter some text.")

# ---- 2. SHOW TRAINING CODE (EXAMPLE) ----
st.header("📝 Model Training Code Example")
training_code = '''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and preprocess your data here

# Train vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

# Train model
rf_model = RandomForestClassifier()
rf_model.fit(X, df['label'])

# Save model and vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
'''
st.code(training_code, language='python')

# ---- 3. (Optional) DOWNLOAD NOTEBOOK ----
try:
    with open("Untitled8.ipynb", "rb") as f:
        st.download_button("Download Full Training Notebook", f, file_name="FakeNewsTraining.ipynb")
except FileNotFoundError:
    st.info("Notebook download will be available soon.")
