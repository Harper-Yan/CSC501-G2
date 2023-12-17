# CSC501-G2

Running instruction:

Run the following commands to install the required Python libraries: 
pip install pandas nltk gensim scikit-learn matplotlib

Download Pre-trained Word Embeddings Model:
Download the pre-trained Word2Vec model ("GoogleNews-vectors-negative300.bin") from Google's Word2Vec page and save it in the "E:\Concurrency\G2" folder.

Following this order to excute the files:
  1. train_model.py. This is the file used for training models based on the tweets' texts.
  2. self_trained_verses_pre_content.py The is the file computing the average cos_similarity between column 'content'and reference texts.
  3. self_trained_verses_pre_tco1.py The is the file computing the average cos_similarity between column 'tco1_step'and reference texts.
  4. draw_cluster.py This is the file for spectral clustering, T-SNE dimension-reducing, and visualization.
  5. test.py. We use this file to compare the average cos_similarity between a set of netual, non-political texts and the reference text, as a supplement proof of the existence of political inclination in the tweets. The file used is given as "netural_text.csv".
