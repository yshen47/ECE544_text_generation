import random
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy as bce_loss
from FashionDataSet import FashionDataSet
from model import FashionSentenceGenerator
import os
from tqdm import tqdm

BATCH_SIZE = 5
MEMORY_SIZE = 8.0
test_dataset = FashionDataSet('../dataset/test_dataset.p')

word_lang = test_dataset.word_lang

model = FashionSentenceGenerator(test_dataset.num_normal_word, word_lang.n_words - test_dataset.num_normal_word, word_lang=word_lang,
                                 max_len=test_dataset.MAX_LENGTH, batch_size=5)
MODEL_DIRECTORY = './model.pth'
model.eval()
if os.path.exists(MODEL_DIRECTORY):
    model.load_state_dict(torch.load(MODEL_DIRECTORY))
    print("Successfully load from previous results.")


def get_keywords(keywords_vector):
    keywords = []
    for word_idx in keywords_vector:
        word_idx = word_idx.item()
        keywords.append(word_lang.index2word[word_idx])
    return keywords


def generate_ground_truth_sentence(sentence_vector):
    sentence = []
    for word_idx in sentence_vector:
        word_idx = word_idx.item()
        sentence.append(word_lang.index2word[word_idx])
    return " ".join(sentence)


def test(model,  batch_size=BATCH_SIZE):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    total_loss = 0
    batch_count = 0
    for i_batch, sampled_batch in tqdm(enumerate(test_dataloader)):
        loss, g_history, topics = model(sampled_batch, False)
        generated_sentence = []
        for topic in topics:
            generated_sentence.append(word_lang.index2word[int(topic[0])])
            total_loss += loss[0]

        keywords = get_keywords(sampled_batch['keywords'][0])
        print('the keywords used: ', keywords)

        ground_truth_sentence = generate_ground_truth_sentence(sampled_batch['sentence'][0])
        print("original sentence: ", ground_truth_sentence)

        print("generated sentence: " + " ".join(generated_sentence))
        batch_count += 1

        print('------------------------------------------------------')

    print("total average loss: ", total_loss.item() / batch_count)


if __name__ == '__main__':
    test(model)
