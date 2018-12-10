from torch.utils.data import DataLoader
import torch
from torch import nn
from FashionDataSet import FashionDataSet
from model import FashionSentenceGenerator
import os
from tqdm import tqdm
#import torch.multiprocessing as mp
import random
from Config import config

BATCH_SIZE = 5
EPOCH_SIZE = 30
MEMORY_SIZE = 8.0
# print(os.getcwd())
device =config.device
train_dataset = FashionDataSet('../dataset/train_dataset.p')
test_dataset = FashionDataSet('../dataset/test_dataset.p')

word_lang = train_dataset.word_lang

#model.share_memory()


def train(model_type='gru', save_every_batch_num=1000, epoch_size=EPOCH_SIZE, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, gate_coefficient=1, teacher_forcing_ratio=0.95):

    model = FashionSentenceGenerator(train_dataset.num_normal_word, word_lang.n_words - train_dataset.num_normal_word,
                                     model_type=model_type,
                                     word_lang=word_lang,
                                     max_len=train_dataset.MAX_LENGTH, batch_size=BATCH_SIZE)

    MODEL_DIRECTORY = './models/{}_model_batch{}.pth'.format(model_type, BATCH_SIZE)
    if os.path.exists(MODEL_DIRECTORY):
        model.load_state_dict(torch.load(MODEL_DIRECTORY))
        print("Successfully load from previous {} results.".format(model_type))

    criterion_sentence = nn.NLLLoss()
    criterion_gating = nn.BCELoss()
    decoder_optimizer = torch.optim.Adam(model.parameters())

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    for i in tqdm(range(1, epoch_size + 1)):
        print("Running epoch ", str(i))
        for i_batch, sampled_batch in tqdm(enumerate(train_data_loader)):
            decoder_optimizer.zero_grad()
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            loss, g_history, topis = model(sampled_batch, use_teacher_forcing)
            if not use_teacher_forcing:
                generated_sentence = []
                for topic in topis:
                    generated_sentence.append(word_lang.index2word[int(topic[0])])
                print(" ".join(generated_sentence))
            for i in range(batch_size):
                loss += gate_coefficient * criterion_gating(g_history[i], sampled_batch['g_truth'][i])
            loss.backward()
            decoder_optimizer.step()

            if i_batch % save_every_batch_num == 0:
                torch.save(model.state_dict(), './models/{}_model_batch{}.pth'.format(model_type, BATCH_SIZE))
                print("saved model")

        torch.save(model.state_dict(), './models/{}_model_batch{}.pth'.format(model_type, BATCH_SIZE))
        print("saved {} model".format(model_type))
        # Validation
        with torch.set_grad_enabled(False):
            validation_loss = 0
            for i_batch, sampled_batch in enumerate(test_data_loader):
                loss, g_history, _ = model(sampled_batch, use_teacher_forcing=True)
                for i in range(batch_size):
                    loss += gate_coefficient * criterion_gating(g_history[i], sampled_batch['g_truth'][i])
                validation_loss += loss
            with open('./results/validation_loss_{}_batch{}.txt'.format(model_type, BATCH_SIZE), 'a+') as f:
                f.write(str(validation_loss) + '\n')
            print(validation_loss)

# def run():
#     processes = []
#     for i in range(4):  # No. of processes
#         p = mp.Process(target=train, args=(model,))
#         p.start()
#         processes.append(p)
#     for p in processes: p.join()

if __name__ == '__main__':
    print("device:", device)
    train(model_type='lstm')

