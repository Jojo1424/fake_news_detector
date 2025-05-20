import streamlit as st
import pickle

st.title("📰 Fake News Detector")

# Only LOAD the pre-trained vectorizer and model, do not SAVE them!
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

st.header("📝 Model Training Code Example")
training_code = '''
# (Paste your model training code here for display only!)
'''
st.code(training_code, language='python')
