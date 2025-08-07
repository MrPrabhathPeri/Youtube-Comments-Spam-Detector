import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already done
nltk.download('stopwords')

# Load the model and vectorizer
with open('spam_detector_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = text.lower()                  # Lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Streamlit UI
st.set_page_config(page_title="YouTube Spam Detector", page_icon="🔍")

st.title("🎥 YouTube Spam Comment Detector")
st.markdown("Enter a comment below to check whether it's **Spam** or **Not Spam**.")

user_input = st.text_area("📝 Enter a comment:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("🚨 This comment is classified as **Spam**.")
        else:
            st.success("✅ This comment is classified as **Not Spam**.")
