import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources (only needs to be done once)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title('Email/Spam Classifier')

# Input from user
input_sms = st.text_input('Enter your Message')

if st.button('Classify'):
    # Preprocess input
    transformed_sms = transform_text(input_sms)

    # Vectorize input
    vector_input = vectorizer.transform([transformed_sms])

    # Predict using the model
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')


