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


def sanitize_word_vector(word_vector, target='Padding'):
    while target in word_vector:
        word_vector.remove('Padding')


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
    return sentence


def quality_evaluation(keywords, generated_sentence):
    keywords = set(keywords)
    unique_generated_keywords = 0
    for word in keywords:
        if word in generated_sentence:
            unique_generated_keywords += 1

    all_generated_keywords = 0
    for word in generated_sentence:
        if word in keywords:
            all_generated_keywords += 1

    keywords_coverage = unique_generated_keywords / len(keywords)
    keywords_repetition_rate = (all_generated_keywords - unique_generated_keywords) / len(generated_sentence)
    enrichment = (len(generated_sentence) - all_generated_keywords) / len(generated_sentence)
    return keywords_coverage, keywords_repetition_rate, enrichment


def test(model_type='gru',  batch_size=BATCH_SIZE):

    model = FashionSentenceGenerator(test_dataset.num_normal_word, word_lang.n_words - test_dataset.num_normal_word,
                                     model_type=model_type,
                                     word_lang=word_lang,
                                     max_len=test_dataset.MAX_LENGTH, batch_size=BATCH_SIZE)
    MODEL_DIRECTORY = './models/{}_model_batch{}.pth'.format(model_type, BATCH_SIZE)
    model.eval()
    if os.path.exists(MODEL_DIRECTORY):
        model.load_state_dict(torch.load(MODEL_DIRECTORY))
        print("Successfully load {} from previous results.".format(model_type))

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    total_loss = 0
    batch_count = 0
    total_kw_coverage = 0
    total_kw_rep_rate = 0
    total_enrichment = 0
    for i_batch, sampled_batch in tqdm(enumerate(test_dataloader)):
        loss, g_history, topics = model(sampled_batch, False)
        generated_sentence = []
        for topic in topics:
            generated_sentence.append(word_lang.index2word[int(topic[0])])
            total_loss += loss[0]

        keywords = get_keywords(sampled_batch['keywords'][0])
        sanitize_word_vector(keywords)
        print('the keywords used: ', keywords)

        ground_truth_sentence = generate_ground_truth_sentence(sampled_batch['sentence'][0])
        sanitize_word_vector(ground_truth_sentence)
        print("original sentence: ", " ".join(ground_truth_sentence))

        sanitize_word_vector(generated_sentence)
        print("generated sentence: " + " ".join(generated_sentence))

        kw_coverage, kw_rep_rate, enrichment = quality_evaluation(keywords, generated_sentence)
        total_kw_coverage += kw_coverage
        total_kw_rep_rate += kw_rep_rate
        total_enrichment += enrichment

        print("keywords_coverage: ", kw_coverage, "; keywords_repetition_rate: ", kw_rep_rate, "; enrichment", enrichment)

        batch_count += 1

        print('------------------------------------------------------')

    print("{} tests ran".format(batch_count))
    print("total average loss: ", total_loss.item() / batch_count)
    print("total_average_keywords_coverage: ", total_kw_coverage / batch_count)
    print("total_average_keywords_repetition_rate: ", total_kw_rep_rate / batch_count)
    print("total_average_enrichment", total_enrichment / batch_count)


if __name__ == '__main__':
    test(model_type='lstm')
