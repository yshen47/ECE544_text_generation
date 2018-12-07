import gensim
import ast
import nltk
import re
import pickle


with open("../dataset/raw_dataset.p", "rb") as pickle_d:
    raw_data = pickle.load(pickle_d)
'''
for i, v in enumerate(raw_data):
    raw_data[i][0] = nltk.word_tokenize(raw_data[i][0])
'''
word_embedding_input = []
for d in raw_data:
    word_embedding_input.append(d[0])

model = gensim.models.Word2Vec (word_embedding_input, size=100, window=10, min_count=2, workers=10)
model.train(word_embedding_input, total_examples=len(word_embedding_input), epochs=10)
model.save('word_embedding.model')

