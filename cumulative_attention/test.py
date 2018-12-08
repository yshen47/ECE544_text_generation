import random
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy as bce_loss
from FashionDataSet import FashionDataSet
from model import FashionSentenceGenerator
import os
from tqdm import tqdm

test_dataset = FashionDataSet('../dataset/test_dataset.p')

word_lang = test_dataset.word_lang

model = FashionSentenceGenerator(test_dataset.num_normal_word, word_lang.n_words - test_dataset.num_normal_word, word_lang=word_lang,
                                 max_len=test_dataset.MAX_LENGTH, batch_size=1)
MODEL_DIRECTORY = './model.pth'
model.eval()
if os.path.exists(MODEL_DIRECTORY):
    model.load_state_dict(torch.load(MODEL_DIRECTORY))
    print("Successfully load from previous results.")


def test(model, num_workers=0, gate_coefficient=20):
    test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, drop_last=True)

    # Validation
    with torch.set_grad_enabled(False):
        validation_loss = 0
        for i_batch, sampled_batch in enumerate(test_data_loader):
            loss, g_history, generated_sent_indices = model.predict(sampled_batch)
            ground_truth_sent = []
            generated_sent = []
            print("==========================================")
            #print("ground_truth_keywords: ", str(sampled_batch["keywords"][0]))
            print("ground_truth_history: ", str(sampled_batch["g_truth"][0]))
            print("generated history: ", str(g_history[0]))
            for i, word_index in enumerate(generated_sent_indices):
                next_word = word_lang.index2word[int(word_index[0])]
                generated_sent.append(next_word)
                ground_truth_sent.append(word_lang.index2word[int(sampled_batch["sentence"][0][i])])
            print("ground_truth_sent: ", ' '.join(ground_truth_sent))
            print("generated history: ", ' '.join(generated_sent))
            print("==========================================")


test(model)
