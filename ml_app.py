import streamlit as st
import numpy as np
import joblib
import spacy
import sklearn
import string
# python -m spacy download en_core_web_sm
@st.cache
def download_spacy():
    # Fetch data from URL here, and then clean it up.
    spacy.cli.download("en_core_web_sm")

download_spacy()


# nlp = spacy.load("en_core_web_sm-3.0.0\en_core_web_sm-3.0.0")
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation


def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)

    # print(doc)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

model = joblib.load('pipeline.joblib')
# st.write(model)

user_input = st.text_area('Enter Email Content')
button = st.button("Predict")


d = {  
  1:'Spam',
  0:'Not Spam'
}

if user_input and button :
    # test_sample
    output = model.predict([user_input])
    # st.write("Type:",type(output))
    # st.write("output: ", output)
    # prediction = output
    prediction = output[0]
    st.write("prediction: ", prediction)
    st.write("User Input: ",user_input)
    st.write("Prediction: ",d[prediction])