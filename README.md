# Spam-Sms-identifier-
This project is a spam SMS classifier that can distinguish between legitimate ("ham") and spam messages. The project is composed of a machine learning model, a dataset, and a web application for user interaction.

Here's a breakdown of the project components:
Dataset (spam.csv): The model is trained on a dataset of SMS messages, each labeled as either "ham" or "spam".
Machine Learning Model (spamsms.ipynb, spam_model.pkl, vectorizer.pkl):

A Jupyter Notebook (spamsms.ipynb) is used for the entire model development process. This includes loading and preprocessing the data from 
spam.csv, and then training a classification model
The notebook uses a CountVectorizer to convert the text messages into a numerical format that the machine learning model can process.
The trained model and vectorizer are saved as 
spam_model.pkl and vectorizer.pkl files, respectively.

Web Application (app.py):
A user-friendly web interface is created with Streamlit, allowing users to input their own SMS messages for classification.
The application loads the saved spam_model.pkl and vectorizer.pkl to make predictions on new, unseen messages.
When a user enters a message and clicks "Predict", the application uses the loaded model to classify the message and displays whether it is "SPAM" or "NOT spam".

Flow:

Start 🎬
Load Dataset: The process begins by loading the spam.csv file, which contains thousands of SMS messages labeled as either "ham" (legitimate) or "spam."
Preprocess Data: The text data is cleaned and prepared for the model. This involves removing punctuation, converting text to lowercase, and other text normalization techniques.
Vectorize Text: The cleaned text messages are converted into numerical data using the CountVectorizer. This step is crucial because machine learning models can only work with numbers, not words.
Train the Model: The numerical data is fed into a Logistic Regression classification algorithm. The model learns the patterns and characteristics of spam vs. ham messages from this data.
Save the Model and Vectorizer: Once the model is trained, both the model (spam_model.pkl) and the vectorizer (vectorizer.pkl) are saved as pickle files. This allows them to be loaded and used later without having to repeat the training process.
End of Training 🏆

2. Prediction
This is the "live" stage, where the trained model is put to work in the web application to classify new, unseen messages.

Flow:
Start 🚀
User Enters SMS: The user interacts with the Streamlit web application (app.py) and enters an SMS message into the text box.
Load Saved Model & Vectorizer: The application loads the pre-trained spam_model.pkl and vectorizer.pkl files.
Vectorize User Input: The user's message is transformed into a numerical format using the loaded CountVectorizer, ensuring it's in the same format as the data the model was trained on.
Make a Prediction: The vectorized input is fed into the loaded spam model, which predicts whether the message is spam or not.

Display the Result:
If the model predicts "SPAM," a warning message is displayed on the web page.
If the model predicts "NOT spam," a success message is shown.
End of Prediction 🎉
