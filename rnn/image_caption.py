import numpy as np
import pickle
import random
import ast
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers.wrappers import Bidirectional
import tqdm
import nltk
from keras.layers import *


token = 'dataset/fashion_sentence_dataset.txt'
sentences = open(token, 'r').read().strip().split('\n')
embedding = 'dataset/fashion_sentence_embedding_dataset.txt'
sentence_embeddings = open(embedding, 'r').read().strip().split('\n')

with open('dataset/vocabularies.json') as f:
    code = f.readlines()
code = ast.literal_eval(code[0])

def encode(editor_note, code=code):
    res = np.zeros(len(code), dtype=int)
    for i, c in enumerate(code):
        if c.lower() in editor_note.lower():
            res[i] = 1
        else:
            res[i] = 0
    return res


embedding_map = {}
for e in tqdm.tqdm(sentence_embeddings):
    temp = e.split(';')
    embedding_map[temp[0]] = ast.literal_eval("[" + temp[1].strip() + "]")


dataset = []
for s in tqdm.tqdm(sentences):
    temp = s.split(';')
    if len(temp) == 2:
        if 'shown here with' not in temp[1] and np.sum(encode(temp[1])) * 1.0 / np.sum(embedding_map[temp[0]]) > 0.4:
            dataset.append(temp)

random.shuffle(dataset)

X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=42)

# Calculating the unique words in the vocabulary.
caps = []
for sent in dataset:
    caps.append(sent[1])
words = [i.split() for i in caps]
unique = []
for i in words:
    unique.extend(i)

unique = list(set(unique))

# Mapping the unique words to indices and vice-versa
word2idx = {val:index for index, val in enumerate(unique)}
idx2word = {index:val for index, val in enumerate(unique)}

# Calculating the maximum length among all the captions
max_len = 0
for c in caps:
    c = c.split()
    if len(c) > max_len:
        max_len = len(c)

vocab_size = len(unique)

samples_per_epoch = 0
for ca in caps:
    samples_per_epoch += len(ca.split())-1


def data_generator(batch_size=32):
    partial_caps = []
    next_words = []
    images = []
    c = []
    count = 0
    while True:
        for j, datapoint in enumerate(X_train):

            current_image = embedding_map[datapoint[0]]
            text = datapoint[1].strip()
            for i in range(len(text.split()) - 1):
                count += 1

                partial = [word2idx[txt] for txt in text.split()[:i + 1]]
                partial_caps.append(partial)

                # Initializing with zeros to create a one-hot encoding matrix
                # This is what we have to predict
                # Hence initializing it with vocab_size length
                n = np.zeros(vocab_size)
                # Setting the next word to 1 in the one-hot encoded matrix
                n[word2idx[text.split()[i + 1]]] = 1
                next_words.append(n)

                images.append(current_image)

                if count >= batch_size:
                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                    yield [[images, partial_caps], next_words]
                    partial_caps = []
                    next_words = []
                    images = []
                    count = 0

# Let's create the model
embedding_size = 100

image_model = Sequential([
        Dense(embedding_size, input_shape=(1300,), activation='relu'),
        RepeatVector(max_len)
    ])

caption_model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=max_len),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])



mergedOut = Concatenate()([image_model.output,caption_model.output])
blstmOut = Bidirectional(LSTM(256, return_sequences=False))(mergedOut)
denseOut = Dense(vocab_size)(blstmOut)
softmaxOut = Activation('softmax')(denseOut)
final_model = Model([image_model.input,caption_model.input], softmaxOut)

final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

final_model.fit_generator(data_generator(batch_size=256), samples_per_epoch=samples_per_epoch, nb_epoch=1, verbose=1)
final_model.save()