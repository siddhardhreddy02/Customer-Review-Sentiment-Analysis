import streamlit as st
import pickle

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model = pickle.load(open(r"C:\Users\vkred\OneDrive\Desktop\spyder\logistic_model.pkl",'rb'))
tfidf = pickle.load(open(r"C:\Users\vkred\OneDrive\Desktop\spyder\tfidf_vectorizer.pkl",'rb'))

st.title('Feedback Analysis App')
st.write("This Feedback Analysis App helps analyze customer reviews.")
st.write("It uses Natural Language Processing (NLP) to understand the sentiment of each review.")
st.write("Just enter any feedback, and the app will tell you if it's positive or negative.")
st.write("Businesses can use this to see how happy customers are and find ways to improve their service.")

inputText=st.text_area("Enter Your review Here")

corpus=[]

for i in inputText:
    review = re.sub('[^a-zA-Z]', ' ', inputText)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = ' '.join(review)
    corpus.append(review)
    
    review_vector = tfidf.transform([review])
    
if st.button('Predict Sentiment'):
    predict=model.predict(review_vector)
    if predict ==1:
        st.success('Your feedback is Positive Review')
    else:
        st.success('Your feedback is Negative Review')