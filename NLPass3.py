import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.cm as cm
from matplotlib import rcParams
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import MaxPooling1D
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Install Kaggle API if not installed
!pip install kaggle

# Provide your Kaggle API credentials
os.environ["KAGGLE_USERNAME"] = "muhdbasheer22"
os.environ["KAGGLE_KEY"] = "d932131db54289ab2c4efe6e4e3ec26e"

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset_name = "sentiment140/training.1600000.processed.noemoticon"
api.dataset_download_files(dataset_name, path="/kaggle/working", unzip=True)

# Now you can continue with your existing code to load the dataset
data = pd.read_csv("/kaggle/working/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", engine="python",
                   names=["label", "time", "date", "query", "username", "text"])

# Assign 1 to positive sentiment and 0 to negative sentiment
data.loc[data['label'] == 4, 'label'] = 1

# Filter the original dataset to separate positive and negative tweets
data_pos = data[data['label'] == 1].head(10000)
data_neg = data[data['label'] == 0].head(10000)

# Concatenate positive and negative tweets
data = pd.concat([data_pos, data_neg])

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet', "/kaggle/working/nltk_data/")
nltk.download('omw-1.4', "/kaggle/working/nltk_data/")

nltk.data.path.append("/kaggle/working/nltk_data/")
nltk.data.path.append("/kaggle/working/nltk_data/")

# Load dataset
dataa = pd.read_csv("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", engine="python",
                   names=["label", "time", "date", "query", "username", "text"])


# Assign 1 to positive sentiment and 0 to negative sentiment
data.loc[data['label'] == 4, 'label'] = 1

# Filter the original dataset to separate positive and negative tweets
data_pos = data[data['label'] == 1].head(10000)
data_neg = data[data['label'] == 0].head(10000)

# Concatenate positive and negative tweets
data = pd.concat([data_pos, data_neg])

# Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub('@[^\s]+', ' ', text)  # Remove emails
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)  # Remove URLs
    text = re.sub('[0-9]+', '', text)  # Remove numbers
    text = re.sub(r'(.)\1+', r'\1', text)  # Remove repeating characters
    translator = str.maketrans('', '', string.punctuation)  # Remove punctuations
    text = text.translate(translator)
    return text

stopwords_list = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords_list])

stemmer = PorterStemmer()
def stem_text(text):
    return [stemmer.stem(word) for word in text]

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(word) for word in text]

tokenizer = RegexpTokenizer(r'\w+')

# Apply preprocessing steps
data['text'] = data['text'].apply(preprocess_text)
data['text'] = data['text'].apply(remove_stopwords)
data['text'] = data['text'].apply(tokenizer.tokenize)
data['text'] = data['text'].apply(stem_text)
data['text'] = data['text'].apply(lemmatize_text)

# Prepare data for model input
X = data['text']
y = data['label']

max_len = 500
tok = Tokenizer(num_words=2000)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
sequences_matrix = pad_sequences(sequences, maxlen=max_len)

# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, y, test_size=0.3, random_state=2)

# Define CNN Model 1
def cnn_model_1(max_len=500):
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(2000, 50, input_length=max_len)(inputs)
    layer = Conv1D(64, 3, activation='relu')(layer)
    layer = GlobalMaxPooling1D()(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

# Train CNN Model 1
model_1 = cnn_model_1()
history_1 = model_1.fit(X_train, Y_train, batch_size=80, epochs=6, validation_split=0.1)

# Evaluate CNN Model 1
accr_1 = model_1.evaluate(X_test, Y_test)
print('CNN Model 1 - Test set\n  Accuracy: {:0.2f}'.format(accr_1[1]))

# Define CNN Model 2
def cnn_model_2(max_len=500):
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(2000, 50, input_length=max_len)(inputs)
    layer = Conv1D(128, 5, activation='relu')(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Conv1D(64, 5, activation='relu')(layer)
    layer = GlobalMaxPooling1D()(layer)
    layer = Dense(256, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

# Train CNN Model 2
model_2 = cnn_model_2()
history_2 = model_2.fit(X_train, Y_train, batch_size=80, epochs=6, validation_split=0.1)

# Evaluate CNN Model 2
accr_2 = model_2.evaluate(X_test, Y_test)
print('CNN Model 2 - Test set\n  Accuracy: {:0.2f}'.format(accr_2[1]))

# Define CNN Model 3
def cnn_model_3(max_len=500):
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(2000, 50, input_length=max_len)(inputs)
    layer = Conv1D(64, 3, activation='relu')(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Conv1D(64, 3, activation='relu')(layer)
    layer = GlobalMaxPooling1D()(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

# Train CNN Model 3
model_3 = cnn_model_3()
history_3 = model_3.fit(X_train, Y_train, batch_size=80, epochs=6, validation_split=0.1)

# Evaluate CNN Model 3
accr_3 = model_3.evaluate(X_test, Y_test)
print('CNN Model 3 - Test set\n  Accuracy: {:0.2f}'.format(accr_3[1]))
