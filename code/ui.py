from os import stat
import streamlit as st 
import numpy as np
import string 
import numpy as np 
from gensim.models import Word2Vec
from Preprocess import NepaliPreprocess
from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
nlp = NepaliPreprocess()

# @staticmethod
def load_Model():

    model = load_model("sentiment.h5")
    model_W2V = Word2Vec.load("w2v/nepaliW2V_5Million.model")
    
    return model, model_W2V

def word2token(word):
    try:
        return model_W2V.wv.key_to_index[word]
    except KeyError:
        return 0  
# COVERT TOKEN INTO WORD
def token2word(token):
    return model_W2V.wv.index_to_key[token]


translator = str.maketrans('', '', string.punctuation + '–')
def prepare_input(text):

    preprocessed_Text = nlp.Reg_and_Stemming(text)

    translator = str.maketrans('', '', string.punctuation + '–')
    news = preprocessed_Text.translate(translator)
    words = np.array([word2token(w) for w in news.split(' ')[:40] if w != ''])
    input_tokens = []
    for idx in words:
        input_tokens.append(idx) 

    set_x = pad_sequences([input_tokens], maxlen=50, padding='pre', value=0)
    return set_x

def get_CATEGORY(input_Seq):

    opt = model.predict(input_Seq)
    idx = np.argmax(opt)
    category = {'negative': 0, 'positive': 1}
    sentiment_class = list(category.keys())[list(category.values()).index(idx)]
    return sentiment_class

st.title("Nepali Sentiment Analysis : ")
from PIL import Image
image = Image.open('image\Bb.png')
st.image(image)

Sentiment_title = st.text_input('Input Sentence', '')

model, model_W2V = load_Model()

if st.button('Predict'):
    # padding the sentence
    padded_input = prepare_input(Sentiment_title)
    result = get_CATEGORY(padded_input)

    if result == 'negative' :
        st.error('Negative Sentiment')

    #else result == 'positive' :
        #st.success('Positive Sentiment')
        
    else:
        st.success('positive Sentiment')