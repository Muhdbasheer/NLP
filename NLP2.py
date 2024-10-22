import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import numpy as np

# Function to generate documents with random combinations of phrases
def generate_documents(phrases_list, num_documents):
    documents = []
    for _ in range(num_documents):
        phrases = [random.choice(field_phrases) for field_phrases in phrases_list]
        document = ' '.join(phrases)
        documents.append(document)
    return documents

# Sample phrases in different fields
phrases_list = [["machine learning", "deep learning"],
                ["data analysis", "data mining"],
                ["natural language processing", "text mining"]]

# Number of documents to generate
num_documents = 4

# Generate documents
documents = generate_documents(phrases_list, num_documents)

# Data processing
def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_and_normalize(text):
    tokens = word_tokenize(text.lower())
    return tokens

def stem(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

# Clean, tokenize, stem, and remove stopwords from documents
processed_documents = []
for document in documents:
    cleaned_document = clean_text(document)
    tokens = tokenize_and_normalize(cleaned_document)
    stemmed_tokens = stem(tokens)
    filtered_tokens = remove_stopwords(stemmed_tokens)
    processed_documents.append(' '.join(filtered_tokens))

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents to get TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(processed_documents)

# Get TF for each word for all documents
tf_matrix = tfidf_matrix.toarray()

# Get IDF for each word
idf_vector = vectorizer.idf_

# Reshape IDF vector to (n_features, 1)
idf_vector = idf_vector.reshape(-1, 1)

# Multiply TF * IDF to get TF-IDF
tfidf_matrix_weighted = tf_matrix * idf_vector[:, None]

# Normalize TF-IDF
tfidf_norm = tfidf_matrix_weighted / tfidf_matrix_weighted.sum(axis=1, keepdims=True)

# Get unique words
unique_words = vectorizer.get_feature_names_out()

print("Generated Documents:")
for i, doc in enumerate(processed_documents, 1):
    print(f"Document {i}: {doc}")

print("TF for each word:")
print(tf_matrix)

print("IDF for each word:")
print(idf_vector)

print("\nNormalized TF-IDF:")
print(tfidf_norm)

print("\nUnique Words:")
print(unique_words)

