import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open("spam_detection_model.pkl", "rb") as f:
    model=joblib.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer=joblib.load(f)

# Streamlit App UI
st.title("📩 SMS Spam Detector")
st.write("Enter a message to check if it's Spam or Ham.")

user_input=st.text_area("Type your message here:")
sample_text="""1. SMS. ac Sptv: The New Jersey Devils and the Detroit Red Wings play Ice Hockey. Correct or Incorrect? End? Reply END SPTV

2. Congratulations! You've been selected for a free iPhone 15. Click here to claim your prize!

3. Dear user, your Netflix subscription is about to expire. Click the link to renew now and avoid interruption.

4. Can you send me the notes for today's class?
"""
st.text_area("Or copy this sample SMS:", sample_text, height=250, disabled=True)

if st.button("Predict"):  
    if user_input.strip()=="":
        st.warning("⚠️ Please enter a message.")
    else:
        input_tfidf=vectorizer.transform([user_input])
        prediction=model.predict(input_tfidf)
        if prediction[0]==1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT spam.")

