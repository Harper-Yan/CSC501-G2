import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

data_files_path = r"E:\Concurrency\G2"
data_files = [os.path.join(data_files_path, f"IRAhandle_tweets_{i}.csv") for i in range(1, 14)]

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through the list of data files and load each CSV file into the DataFrame
for file_path in data_files:
    data = pd.read_csv(file_path, encoding="ISO-8859-1")
    combined_data = pd.concat([combined_data, data], ignore_index=True)

data=combined_data[['content']]

import nltk
nltk.download('punkt')
nltk.download('stopwords')


# 1. The cleaning of the text. We use only the colomn 'content' here, the detail process is like the code.
# We may talk about this part.
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # or handle it in a way that makes sense for your data
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text.lower())  # Tokenization and lowercase
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stop words and non-alphabetic words
    #return " ".join(tokens)
    return tokens


data["text"] = data["content"].apply(preprocess_text)

print(len(data))


#2. Self training word-embeddings, we may talk about the selection of parameters.
model = Word2Vec(sentences=data['text'], vector_size=100, window=20, min_count=1, workers=4)
vocabulary = model.wv.key_to_index
print("Vocabulary size:", len(vocabulary))
model.save("tweet_whole.model")