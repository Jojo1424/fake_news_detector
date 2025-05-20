import streamlit as st
import pickle

st.title("📰 Fake News Detector")

# LOAD the pre-trained vectorizer and model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

user_input = st.text_area("Paste a news article (title + content):")
if st.button("Predict"):
    if user_input.strip():
        input_clean = user_input.lower()
        input_vec = vectorizer.transform([input_clean])
        pred = model.predict(input_vec)[0]
        st.write("### Prediction:", "🟥 **FAKE NEWS**" if pred == 0 else "🟩 **TRUE NEWS**")
    else:
        st.warning("Please enter some text.")

# If you want to display your code (not run it), use st.code():
st.header("📝 Model Training Code Example")
training_code = '''
# (paste your training code here, as a string!)
'''
st.code(training_code, language='python')
