import numpy as np
import ast
import re
import pickle
import json
from nltk.stem import PorterStemmer
from pymongo import MongoClient
client = MongoClient("localhost", 27017)
products = client["fashion_products"]["products"]
print(products.count())

with open('vocab_types.json') as f:
    vocab = f.readlines()
    vocab = ast.literal_eval(vocab[0])


def extract_keywords(editor_note):
    keyword_pairs = []
    has_type = False
    for i, c in enumerate(editor_note.split()):
        if c.lower() in vocab.keys():
            for cat in vocab[c.lower()]:
                if cat == 'type':
                    has_type = True
                keyword_pairs.append((cat, c.lower()))
    if has_type:
        return keyword_pairs
    else:
        return None


res = []
for p in products.find():
    des = p['descriptions'][0].lower().replace("\n", "")
    for sent in des.split("."):
        sent = sent.strip()
        if "this" in sent or "these" in sent:
            if 15 < len(sent.split()) < 20:
                res.append(sent)
res = list(set(res))


#replace frequency < 10 words with UNK
word_map = {}
for r in res:
    for w in r.split():
        if w not in word_map:
            word_map[w] = 0
        word_map[w] += 1

for i in range(len(res)):
    orig = res[i].split()
    new = ['UNK' if word_map[word] <= 10 else word for word in orig]
    curr_i = 1
    while True:
        if curr_i < len(new) and new[curr_i] == new[curr_i-1] and new[curr_i] == 'UNK':
            del new[curr_i]
            continue
        curr_i += 1
        if curr_i >= len(new):
            break
    res[i] = ' '.join(new)

final_res = []
for r in res:
    keyword_pairs = extract_keywords(r)
    if keyword_pairs and len(keyword_pairs) > 2:
        final_res.append((keyword_pairs, r))
print("total:", len(final_res))


train_threshold = int(len(final_res) * 0.95)

with open("train_dataset.p", "wb") as pickle_d:
    pickle.dump(final_res[:train_threshold], pickle_d)

with open('train_dataset.txt', 'w+') as f:
    for r in final_res[:train_threshold]:
        f.write(str(r[1]) + '\n')


with open("test_dataset.p", "wb") as pickle_d:
    pickle.dump(final_res[train_threshold:], pickle_d)

with open('test_dataset.txt', 'w+') as f:
    for r in final_res[train_threshold:]:
        f.write(str(r[1]) + '\n')


with open("total_dataset.p", "wb") as pickle_d:
    pickle.dump(final_res, pickle_d)
