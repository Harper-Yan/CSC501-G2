import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
import os

# Load models
model_s = Word2Vec.load(r"E:\Concurrency\tweet_whole.model")
model_p = KeyedVectors.load_word2vec_format(r"E:\Concurrency\G2\GoogleNews-vectors-negative300.bin", binary=True)

# Load data
data_files_path = r"E:\Concurrency\G2"
data_files = [os.path.join(data_files_path, f"IRAhandle_tweets_{i}.csv") for i in range(1, 14)]

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through the list of data files and load each CSV file into the DataFrame
for file_path in data_files:
    data = pd.read_csv(file_path, encoding="ISO-8859-1")
    combined_data = pd.concat([combined_data, data[:2000]], ignore_index=True)
    combined_data.reset_index(drop=True, inplace=True)

data = combined_data[['content']].sample(n=10000, random_state=42)

# Preprocess text
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text.lower())  # Tokenization and lowercase
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stop words and non-alphabetic words
    return " ".join(tokens)

def document_vector_s(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return None
    return sum(vectors) / len(vectors)

def document_vector_p(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return None
    return sum(vectors) / len(vectors)

# Preprocess the other texts and create feature vectors
data["text"] = data["content"].apply(preprocess_text)
data['vector_s'] = data['text'].apply(lambda x: document_vector_s(word_tokenize(x), model_s))
data['vector_p'] = data['text'].apply(lambda x: document_vector_p(word_tokenize(x), model_p))

# Drop rows with missing vectors
data = data.dropna(subset=['vector_s', 'vector_p'])

# Concatenate vectors for spectral clustering
Xs = pd.concat([data['vector_s'].apply(pd.Series), data['vector_s'].apply(pd.Series)], axis=1)
Xp = pd.concat([data['vector_p'].apply(pd.Series), data['vector_p'].apply(pd.Series)], axis=1)
# Spectral clustering
n_clusters=20
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
data['cluster_s'] = spectral.fit_predict(Xs)
data['cluster_p'] = spectral.fit_predict(Xp)

# Dimensionality reduction with t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result_s = tsne.fit_transform(Xs)
tsne_result_p = tsne.fit_transform(Xp)

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(tsne_result_s[:, 0], tsne_result_s[:, 1], c=data['cluster_s'],cmap='viridis', alpha=0.5)
plt.title('Spectral Clustering Visualization using self-trained model. N_cluster = '+str(n_clusters))
plt.savefig("E:\Concurrency\G2\self_trained_cluster"+str(n_clusters)+".png")
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(tsne_result_p[:, 0], tsne_result_p[:, 1],c=data['cluster_p'], cmap='viridis', alpha=0.5)
plt.title('Spectral Clustering Visualization using pre-trained model.  N_cluster = '+str(n_clusters))
plt.savefig("E:\Concurrency\G2\pre_trained_cluster"+str(n_clusters)+".png")
plt.show()

