import streamlit as st
import pickle
import numpy as np

# Load your trained model and CountVectorizer
# Save these objects from your notebook first:
# pickle.dump(model, open('spam_model.pkl', 'wb'))
# pickle.dump(cv, open('vectorizer.pkl', 'wb'))

model = pickle.load(open("C:\\Users\\chira\\OneDrive\\Desktop\\New folder\\major projects\\spam sma\\spam_model.pkl", 'rb'))
cv = pickle.load(open("C:\\Users\\chira\\OneDrive\\Desktop\\New folder\\major projects\\spam sma\\vectorizer.pkl", 'rb'))

st.title("Spam SMS Classifier")

user_input = st.text_area("Enter your SMS message:")

if st.button("Predict"):
    # Preprocess and vectorize input
    vect_input = cv.transform([user_input])
    prediction = model.predict(vect_input)
    if prediction == 1:
        st.error("This message is likely SPAM.")
    else:
        st.success("This message is NOT spam.")
