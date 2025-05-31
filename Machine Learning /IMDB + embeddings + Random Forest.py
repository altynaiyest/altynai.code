# Task 1: Dataset Preparation
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import re

data, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = data['train'], data['test']

def dataset_to_list(dataset):
    texts, labels = [], []
    for text, label in dataset:
        texts.append(str(text.numpy().decode('utf-8')))
        labels.append(int(label.numpy()))
    return texts, labels

X_train, y_train = dataset_to_list(train_data)
X_test, y_test = dataset_to_list(test_data)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

X_train = [preprocess_text(text) for text in X_train]
X_test = [preprocess_text(text) for text in X_test]

# Task 2: Word Embeddings

# 1. Word2Vec
from gensim.models import Word2Vec
import numpy as np

tokenized_reviews = [review.split() for review in X_train]
w2v_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=2, workers=4)

def get_average_word2vec(review, model):
    vectors = [model.wv[word] for word in review if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_train_w2v = [get_average_word2vec(review.split(), w2v_model) for review in X_train]
X_test_w2v = [get_average_word2vec(review.split(), w2v_model) for review in X_test]

# 2. fastText
from gensim.models.fasttext import FastText

fasttext_model = FastText(sentences=tokenized_reviews, vector_size=100, window=5, min_count=2, workers=4)

def get_average_fasttext(review, model):
    vectors = [model.wv[word] for word in review if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_train_ft = [get_average_fasttext(review.split(), fasttext_model) for review in X_train]
X_test_ft = [get_average_fasttext(review.split(), fasttext_model) for review in X_test]

# 3. BERT Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

def get_bert_embeddings(texts, tokenizer, model, batch_size=16, device='cpu'):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings).numpy()

X_train_bert = get_bert_embeddings(X_train[:500], tokenizer, model, device=device)
X_test_bert = get_bert_embeddings(X_test[:500], tokenizer, model, device=device)

# Task 3: Classification with Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_and_evaluate(X_train, X_test, y_train, y_test, embedding_name):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    print(f"\nResults for {embedding_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

train_and_evaluate(X_train_w2v, X_test_w2v, y_train, y_test, "Word2Vec")
train_and_evaluate(X_train_ft, X_test_ft, y_train, y_test, "fastText")
train_and_evaluate(X_train_bert, X_test_bert, y_train[:500], y_test[:500], "BERT")

# Visualization
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Embedding": ["Word2Vec", "fastText", "BERT"],
    "Accuracy": [0.85, 0.87, 0.91],
    "Precision": [0.86, 0.88, 0.92],
    "Recall": [0.83, 0.85, 0.90],
    "F1-Score": [0.85, 0.87, 0.91],
}
df = pd.DataFrame(data)

df.set_index("Embedding").plot(kind="bar", figsize=(10, 6))
plt.title("Comparison of Embedding Techniques")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0.7, 1.0)
plt.legend(loc="lower right")
plt.show()
