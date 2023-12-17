import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.data import find
from gensim.models import KeyedVectors
import os

model_s = Word2Vec.load(r"E:\Concurrency\tweet_whole.model")
vocabulary = model_s.wv.key_to_index
print("Vocabulary size:", len(vocabulary))
model_p = KeyedVectors.load_word2vec_format(r"E:\Concurrency\G2\GoogleNews-vectors-negative300.bin",binary=True)


data_files_path = r"E:\Concurrency\G2"
data_files = [os.path.join(data_files_path, f"IRAhandle_tweets_{i}.csv") for i in range(1, 14)]

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through the list of data files and load each CSV file into the DataFrame
for file_path in data_files:
    data = pd.read_csv(file_path, encoding="ISO-8859-1")
    combined_data = pd.concat([combined_data, data[:2000]], ignore_index=True)
    combined_data.reset_index(drop=True, inplace=True)

data=combined_data[['content']]

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

data = data.dropna(subset=['vector_s', 'vector_p'])

# Calculate cosine similarity with a reference text
reference_text1 = "News states: Liberals have a higher average IQ compared to consevertives."
reference_vector1s = document_vector_s(word_tokenize(preprocess_text(reference_text1)), model_s)
reference_vector1p = document_vector_p(word_tokenize(preprocess_text(reference_text1)), model_p)

reference_text2 = "The weather is Okay."
reference_vector2s = document_vector_s(word_tokenize(preprocess_text(reference_text2)), model_s)
reference_vector2p = document_vector_p(word_tokenize(preprocess_text(reference_text2)), model_p)

# Calculate similarity and print mean
data["similarity_cmp1s"] = data["vector_s"].apply(lambda x: cosine_similarity([reference_vector1s], [x])[0][0])
data["similarity_cmp2s"] = data["vector_s"].apply(lambda x: cosine_similarity([reference_vector2s], [x])[0][0])
data["similarity_cmp1p"] = data["vector_p"].apply(lambda x: cosine_similarity([reference_vector1p], [x])[0][0])
data["similarity_cmp2p"] = data["vector_p"].apply(lambda x: cosine_similarity([reference_vector2p], [x])[0][0])

print("Biased text: ", reference_text1)
print("Cos_similarity_by_self_trained_model:",data[["similarity_cmp1s"]].mean(axis=0))
print("Cos_similarity_by_pre_trained_model:",data[["similarity_cmp1p"]].mean(axis=0))

print("Unrevalent text: ",reference_text2)
print("Cos_similarity_by_self_trained_model:",data[["similarity_cmp2s"]].mean(axis=0))
print("Cos_similarity_by_pre_trained_model:",data[["similarity_cmp2p"]].mean(axis=0))
