from torch.utils.data import Dataset
import pickle
from lang import Lang
import torch
from Config import config
import nltk

SOS_token = 0
EOS_token = 1
Padding_token = 2
MAX_LENGTH = 21
MAX_MEM_SIZE = 10
device = config.device

def filter_keywors(pair):
    vocab = set()
    newSen = []
    for tuple in pair[0]:
        vocab.add(tuple[0])
        vocab.add(tuple[1])
    for word in pair[1].split():
        if word not in vocab:
            newSen.append(word)
    return (pair[0], " ".join(newSen))


def extract_tags(sentence):
    # tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(sentence)
    return [t[1] for t in tagged]

class FashionDataSet(Dataset):

    def __init__(self, directory):
        self.word_lang = Lang("normal word")
        self.num_normal_word = -1
        self.MAX_LENGTH = MAX_LENGTH
        self.MAX_MEM_SIZE = MAX_MEM_SIZE
        with open(directory, "rb") as pickle_d:
            self.raw_data = pickle.load(pickle_d)
        self.prepare_lang()

    def prepare_lang(self):
        tuples = []
        with open('../dataset/total_dataset.p', "rb") as pickle_d:
            self.total = pickle.load(pickle_d)
        # add sentence first
        for pair in self.total:
            pair = filter_keywors(pair)
            sen = pair[1]
            self.word_lang.addSentence(sen)
            for tuple in pair[0]:
                tuples.append(tuple)

        self.num_normal_word = self.word_lang.n_words

        # add tuples last
        for (category, keyword) in tuples:
            self.word_lang.addWord(category)
            self.word_lang.addWord(keyword)

        print("Counted words:")
        print(self.word_lang.name, self.word_lang.n_words)

    def indexes_from_sentence(self, sentence):
        return [self.word_lang.word2index[word] for word in sentence.split(' ')]

    def tensor_from_sentence(self, sentence):
        indexes = self.indexes_from_sentence(sentence)
        indexes.append(EOS_token)
        indexes.insert(0, SOS_token)
        # padding for batch
        for i in range(len(indexes), MAX_LENGTH):
            indexes.append(Padding_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensors_from_pair(self, pair):
        keyword_pairs = pair[0]
        sentence = pair[1]
        categories = [self.word_lang.word2index[category] for (category, _) in keyword_pairs]
        keywords = [self.word_lang.word2index[keyword] for (_, keyword) in keyword_pairs]
        # padding for batch
        for i in range(len(categories), MAX_MEM_SIZE):
            categories.append(Padding_token)
            keywords.append(Padding_token)
        categories = torch.tensor(categories[:MAX_MEM_SIZE], dtype=torch.long, device=device).view(-1, 1)
        keywords = torch.tensor(keywords[:MAX_MEM_SIZE], dtype=torch.long, device=device).view(-1, 1)
        tags = ['NN'] + extract_tags(sentence.split()) + ['NN']
        tags = tags[:MAX_LENGTH]
        sentence = self.tensor_from_sentence(sentence)
        if len(tags) < MAX_LENGTH:
            for i in range(len(tags), MAX_LENGTH):
                tags.append('NN')
        g_ground_truth = torch.zeros(sentence.size(0), device=device)

        for di in range(1, sentence.size(0) - 1):
            if sentence[di][0] in keywords.view(-1):
                g_ground_truth[di] = 0
            else:
                g_ground_truth[di] = 1
        return {"categories": categories, "keywords":keywords, "memory_size": min(len(keyword_pairs), MAX_MEM_SIZE), "sentence": sentence, "tags": tags, "g_truth": g_ground_truth}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        return self.tensors_from_pair(self.raw_data[index])
