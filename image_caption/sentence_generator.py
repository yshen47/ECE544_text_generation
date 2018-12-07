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
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import keras

with open("../dataset/raw_dataset.p", "rb") as pickle_d:
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
train_dataset, test_dataset = train_test_split(dataset, test_size=0.05, random_state=42)

unique_words = list(set(words))

word2idx = {val:index for index, val in enumerate(unique_words)}
idx2word = {index:val for index, val in enumerate(unique_words)}

vocab_size = len(unique_words)

def data_generator(batch_size=32):
    partial_caps = []
    next_words = []
    images = []
    c = []
    count = 0
    while True:
        for j, datapoint in enumerate(train_dataset):

            current_image = datapoint[1]
            words = datapoint[0]
            for i in range(len(words)-1):
                count += 1

                partial = [word_embedding_model.wv[txt] for txt in words[:i + 1]]
                partial_caps.append(partial)

                # Initializing with zeros to create a one-hot encoding matrix
                # This is what we have to predict
                # Hence initializing it with vocab_size length
                n = np.zeros(vocab_size)
                # Setting the next word to 1 in the one-hot encoded matrix
                n[word2idx[words[i + 1]]] = 1
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

embedding_size = 100

image_embedding = Sequential([
        Dense(embedding_size, input_shape=(1300,), activation='tanh'),
        Reshape((-1, embedding_size))
    ])

text_input = Input(shape=(max_len, 100), name='text_input')

mergedOut = Concatenate(axis=1)([image_embedding.output, text_input])
blstmOut = Bidirectional(LSTM(256, activation='relu', kernel_initializer=keras.initializers.glorot_normal(seed=None)))(mergedOut)
dropOut = Dropout(0.6)(blstmOut)
denseOut = Dense(vocab_size, kernel_initializer=keras.initializers.glorot_normal(seed=None),kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(dropOut)
softmaxOut = Activation('softmax')(denseOut)
final_model = Model([image_embedding.input,text_input], softmaxOut)
'''
final_model = Sequential()
final_model.add(Bidirectional(LSTM(256, activation="relu"),input_shape=(max_len, 100)))
final_model.add(Dropout(0.6))
final_model.add(Dense(vocab_size))
final_model.add(Activation('softmax'))
'''
final_model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=1, clipvalue=0.5), metrics=['accuracy'])
print(final_model.summary())
trained_model_id = datetime.now().strftime('%Y_%m_%d_%H%M%S')
checkpoint = ModelCheckpoint("train_models" + str(trained_model_id))
final_model.fit_generator(data_generator(batch_size=64), steps_per_epoch=10000, shuffle=True,callbacks = [checkpoint], nb_epoch=50,
                          verbose=1)

trained_model_id = datetime.now().strftime('%Y_%m_%d_%H%M%S')
final_model.save("trained_models" + str(trained_model_id))


trained_model_id = datetime.now().strftime('%Y_%m_%d_%H%M%S')
final_model.save("trained_models" + str(trained_model_id))

image_encoding = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
]
def predict_captions(image_encoding):
    start_word = ["<START>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = final_model.predict([np.array([image_encoding]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == "<END>" or len(start_word) > max_len:
            break

    return ' '.join(start_word[1:-1])

print(predict_captions(image_encoding))