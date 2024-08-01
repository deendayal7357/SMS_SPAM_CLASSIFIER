import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the PorterStemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters and stopwords
    filtered_tokens = [ps.stem(w) for w in tokens if w.isalnum() and w not in stop_words]

    # Join the list of words back into a single string
    return " ".join(filtered_tokens)

# Streamlit UI
st.title("SMS Spam Classifier")

# Text input for user
input_sms = st.text_input("Enter the message")

# Button to trigger prediction
if st.button('Predict'):
    if input_sms:
        transformed_text = transform_text(input_sms)
        # Vectorize the transformed text
        vector = tfidf.transform([transformed_text])

        # Predict using the model
        result = model.predict(vector)[0]

        # Display the result
        if result == 1:
            st.header("SPAM")
        else:
            st.header("NOT SPAM")
    else:
        st.warning("Please enter a message to classify.")