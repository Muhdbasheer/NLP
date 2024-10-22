import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import wikipedia

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to fetch HTML content from URL and extract text
def get_text_from_url(url):
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    return text

# Function to process text (clean, stem, remove stopwords, normalize)
def process_text(text):
    # Tokenization
    words = word_tokenize(text)

    # Lowercasing and removing non-alphabetic characters
    words = [word.lower() for word in words if word.isalpha()]

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]



# URL of Wikipedia
url = 'https://en.wikipedia.org'

# Get text from URL
text = get_text_from_url(url)
# Set the language of the Wikipedia page (optional, defaults to English)
wikipedia.set_lang("en")

# Specify the title of the Wikipedia page you want to fetch
page_title = "Artificial intelligence"

# Fetch the content of the Wikipedia page
page_content = wikipedia.page(page_title).content



# Process text
processed_words = process_text(text)

# Print unique words
unique_words = set(processed_words)
print("Unique words:", unique_words)

# Tokenize the processed text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_words)
sequences = tokenizer.texts_to_sequences(processed_words)

# Pad sequences to ensure uniform length
max_sequence_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Define and compile the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
rnn_units = 128

model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(units=rnn_units),
    Dense(units=vocab_size, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


