import numpy as np
import joblib
import pandas as pd
import flask 
import nltk
from flask import Flask,render_template,request
model=joblib.load('lg.joblib')
import pickle as pkl
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pkl.load(f)
app=Flask(__name__)
import re
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)
import string
def remove_punctuation(text):
    # Using the string.punctuation constant to get all punctuations
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Function to apply lemmatization to a string
def apply_lemmatization(text):
    tokens = nltk.word_tokenize(text)  # Tokenize the text into words
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Apply lemmatization to each word
    return ' '.join(lemmatized_tokens)  # Join the lemmatized tokens back into a single string

from nltk.corpus import stopwords
#nltk.download('stopwords')
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)  # Tokenize the text into words
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]  # Remove stop words
    return ' '.join(filtered_tokens)  # Join the filtered tokens back into a single string

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        review=request.values
        n=review['reviewText']
        n=remove_urls(n)
        n=remove_punctuation(n)
        n=apply_lemmatization(n)
        n=remove_stopwords(n)
        x=cv.transform(n.split()).toarray()
        x=x.reshape(1,-1)

        # Assuming you have the sequences in 'x' as a list of lists

        # Set the maximum feature length
        max_len = 2500

        # Find the maximum sequence length
        # max_len = max(len(seq) for seq in x)

        # Perform zero-padding on the sequences
        padded_sequences = []
        for seq in x:
            if len(seq) < max_len:
                # Pad the sequence with zeros at the end
                padded_seq = seq + [0] * (max_len - len(seq))
            else:
                # Truncate the sequence if it exceeds the maximum length
                padded_seq = seq[:max_len]
            padded_sequences.append(padded_seq)

        # 'padded_sequences' now contains the zero-padded sequences
        value=model.predict(padded_sequences)
    if value[0]==1:
        return  render_template("index.html",prediction_text="Positive") 
    else:
        return render_template("index.html",prediction_text="Negative") 
    

if __name__=='__main__':
    app.run(debug=True)

