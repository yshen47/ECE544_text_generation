import pickle
import nltk
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from datetime import datetime
from gensim.models import Word2Vec

with open("product_sentences.p", "rb") as pickle_d:
    raw_data = pickle.load(pickle_d)

word_embedding_model = Word2Vec.load("word_embedding.model")
words = []
max_len = 0
unknown = []

dataset = []
for sent in raw_data:
    count = 0
    curr_sent = [[]]
    curr_sent.append(sent[1])
    for word in sent[0]:
        if word not in word_embedding_model:
            unknown.append(word)
        else:
            curr_sent[0].append(word)
            words.append(word)
            count += 1
    dataset.append(curr_sent)
    max_len = max(max_len, count)
del raw_data

unique_words = list(set(words))

word2idx = {val:index for index, val in enumerate(unique_words)}

idx2word = {index:val for index, val in enumerate(unique_words)}

vocab_size = len(unique_words)

final_model = load_model('train_models2018_10_27_042532')


def predict_captions(image_encoding):
    start_word = ["START"]
    while True:
        par_caps = [word_embedding_model.wv[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = final_model.predict([np.array([image_encoding]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == "END" or len(start_word) > max_len:
            print(word_pred)
            break

    return ' '.join(start_word[1:-1])


for a in dataset:
    res = predict_captions(a[1])
    print(res)
    print(a[0])