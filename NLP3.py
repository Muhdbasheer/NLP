import os
import re
import numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.fasttext import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.models import Word2Vec
from gensim.downloader import load
import json
import pandas as pd

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



# Load Yelp tips data
yelp_tips_file_path = "C:/Users/hp/Documents/yelp_academic_dataset_tip.json"
yelp_datafile = pd.read_json(yelp_tips_file_path, lines=True)

# Extract 'text' attribute
texts = yelp_datafile['text'].tolist()

# Subset data for gensim FastText model
part_of_texts = texts[:10]  # select first 100 sample lines

# Print samples of sentences
print("\nSamples of Sentences\n{}".format(part_of_texts[:10]))

# Train yelp FastText model
embedding_size = 300
window_size = 5
min_word = 5
down_sampling = 1e-2

# Function to preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


# Preprocess text
processed_texts = [preprocess_text(text) for text in part_of_texts]



model = FastText(sentences=processed_texts, vector_size=embedding_size, window=window_size, min_count=min_word,
                 workers=4, sample=down_sampling)

if len(model.wv.index_to_key) >= 10:
    random_words = random.sample(list(model.wv.index_to_key), k=10)
else:
    random_words = list(model.wv.index_to_key)

# Choose 20 random words
random_words = random.sample(list(model.wv.index_to_key), k=10)


# Load Facebook pretrained model
pretrained_model_name = "cc.en.300.bin.gz"
pretrained_model = load_facebook_model(pretrained_model_name)
# Get similar and dissimilar words
similar_words_custom_model = {}
dissimilar_words_custom_model = {}

similar_words_pretrained_model = {}
dissimilar_words_pretrained_model = {}

for word in random_words:
    similar_words_custom_model[word] = model.wv.most_similar(word, topn=10)
    dissimilar_words_custom_model[word] = model.wv.most_similar(negative=[word], topn=10)

    similar_words_pretrained_model[word] = pretrained_model.most_similar(word, topn=10)
    dissimilar_words_pretrained_model[word] = pretrained_model.most_similar(negative=[word], topn=10)

# Print results
print("Similar and dissimilar words using custom model:")
for word in random_words:
    print(f"Word: {word}")
    print("Similar Words:")
    for similar_word, similarity in similar_words_custom_model[word]:
        print(f"\t{similar_word}: {similarity}")
    print("Dissimilar Words:")
    for dissimilar_word, similarity in dissimilar_words_custom_model[word]:
        print(f"\t{dissimilar_word}: {similarity}")

print("\nSimilar and dissimilar words using pretrained model:")
for word in random_words:
    print(f"Word: {word}")
    print("Similar Words:")
    for similar_word, similarity in similar_words_pretrained_model[word]:
        print(f"\t{similar_word}: {similarity}")
    print("Dissimilar Words:")
    for dissimilar_word, similarity in dissimilar_words_pretrained_model[word]:
        print(f"\t{dissimilar_word}: {similarity}")

# Update gensim fastText model using new data
new_data = [['aaa', 'bbbb', 'cccc', 'dddd', 'eeee', 'ffff'], ['wwww', 'xxxx', 'yyyy', 'zzzz', 'vvvv', 'mmmm']]

model.build_vocab(new_data, update=True)  # Update trained gensim fastText model
new_model = model.train(new_data, total_examples=model.corpus_count, epochs=10)  # Continue training the model

