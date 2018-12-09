import torch
import torch.nn.functional as F
from torch import nn
from Config import config
from FashionDataSet import extract_tags
import numpy as np
import nltk

device = config.device
available = False


class FashionSentenceGenerator(nn.Module):

    def __init__(self, normal_vocab_size, keyword_vocab_size, word_lang=None, max_len=30, max_mem_size=10, num_layers=1,
                 embedding_dim=50, batch_size=5, tag_constant = config.TAG_CONSTANT):
        """
        Constructor for fashion sentence generator
        :param normal_vocab_size:   int     total number of normal vocabulary
        :param keyword_vocab_size:  int     total number of keyword vocabulary
        :param max_mem_size:        int     maximum number of memory pairs
        :param max_len:             int     maximum length of input sentences
        :param num_layers:          int     number of LSTM layers
        :param embedding_dim:       int     word embedding size
        """
        super(FashionSentenceGenerator, self).__init__()
        self.hidden_size = embedding_dim
        self.embedding_dim = embedding_dim
        self.word_lang = word_lang
        self.word_embedder = nn.Sequential(
            nn.Embedding(normal_vocab_size + keyword_vocab_size, self.embedding_dim),
            nn.Dropout(0.1)
        )
        self.max_mem_size = max_mem_size
        self.normal_vocab_size = normal_vocab_size
        self.max_len = max_len
        self.device = device
        self.batch_size = batch_size
        self.W_Ct_reshape = nn.Linear(5 * self.embedding_dim, self.hidden_size)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.tag_constant = tag_constant

        self.W_n = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        nn.init.eye_(self.W_n.weight)
        nn.init.eye_(self.W_k.weight)
        nn.init.eye_(self.W_v.weight)

        self.W_n_proj = nn.Linear(self.hidden_size, normal_vocab_size, bias=True)

        self.normal_vocab_linear_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, normal_vocab_size),
            torch.nn.Softmax(dim=1)
        )

        self.gated_linear_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, max_mem_size),
            torch.nn.Sigmoid()
        )

        self.W_g = nn.Linear(self.hidden_size, 1)
        self.attn_HN = torch.nn.Sequential(
            torch.nn.Linear((self.max_len + 1) * self.hidden_size, self.max_len),
            torch.nn.Tanh()
        )
        self.attn_HK = torch.nn.Sequential(
            torch.nn.Linear((self.max_len + 1) * self.hidden_size, self.max_len),
            torch.nn.Tanh()
        )
        self.attn_HV = torch.nn.Sequential(
            torch.nn.Linear((self.max_len + 1) * self.hidden_size, self.max_len),
            torch.nn.Tanh()
        )
        self.attn_MK = torch.nn.Sequential(
            torch.nn.Linear(4 * self.hidden_size, self.max_mem_size),
            torch.nn.Tanh()
        )
        self.attn_MV = torch.nn.Sequential(
            torch.nn.Linear(4 * self.hidden_size, self.max_mem_size),
            torch.nn.Tanh()
        )

    def prepare_memory(self, batch_data):
        self.current_mem_sizes = batch_data["memory_size"]
        self.key_memory = torch.zeros(self.batch_size, self.max_mem_size, self.embedding_dim, dtype=torch.float,
                                      device=device)
        self.value_memory = torch.zeros(self.batch_size, self.max_mem_size, self.embedding_dim, dtype=torch.float,
                                        device=device)
        for i in range(self.batch_size):
            for j in range(self.current_mem_sizes[i]):
                self.key_memory[i, j, :] = self.word_embedder(batch_data["keywords"][i, j, 0])
                self.value_memory[i, j, :] = self.word_embedder(batch_data["categories"][i, j, 0])

    def prepare_history(self):
        # ====== build history =====
        self.hist_N = torch.zeros(self.batch_size, self.max_len, self.hidden_size, dtype=torch.float, device=device)
        self.hist_K = torch.zeros(self.batch_size, self.max_len, self.hidden_size, dtype=torch.float, device=device)
        self.hist_V = torch.zeros(self.batch_size, self.max_len, self.hidden_size, dtype=torch.float, device=device)

        # ====== build prev_hidden h_(t-1) =====
        if available:
            avg_mk = (torch.sum(self.key_memory, 1).t() / self.current_mem_sizes.type(torch.cuda.FloatTensor)).t()
            avg_mv = (torch.sum(self.value_memory, 1).t() / self.current_mem_sizes.type(torch.cuda.FloatTensor)).t()
        else:
            avg_mk = (torch.sum(self.key_memory, 1).t() / self.current_mem_sizes.float()).t()
            avg_mv = (torch.sum(self.value_memory, 1).t() / self.current_mem_sizes.float()).t()

        # ====== build h_(t-1)^N h_(t-1)^K h_(t-1)^V =====
        for i in range(self.batch_size):
            initial_hidden = (avg_mv[i, :] + avg_mk[i, :]) / 2
            self.hist_N[i, 0, :] = initial_hidden
            self.hist_K[i, 0, :] = initial_hidden
            self.hist_V[i, 0, :] = initial_hidden
        self.prev_hiddens = self.hist_N[:, 0, :].view(self.batch_size, -1)
        # ====== memorize t =====
        self.t = 1

    def update_history(self, di, hidden_Ns, hidden_Ks, hidden_Vs):
        temp_N = self.hist_N.clone()
        temp_K = self.hist_K.clone()
        temp_V = self.hist_V.clone()
        temp_N[:, di, :] = hidden_Ns
        temp_K[:, di, :] = hidden_Ks
        temp_V[:, di, :] = hidden_Vs
        self.hist_N = temp_N
        self.hist_K = temp_K
        self.hist_V = temp_V

    def forward(self, batch_data, use_teacher_forcing):
        # Single example
        self.prepare_memory(batch_data)
        self.prepare_history()
        # input_length = sentence_tensor.size[0]
        loss = 0
        prev_word_embeddings = self.word_embedder(batch_data["sentence"][:, 0, :])
        g_history = torch.zeros(self.batch_size, self.max_len, 1, device=device, dtype=torch.float)
        topis = []
        for di in range(1, self.max_len):

            # ===================== compute context ========================
            context_HNs = self.apply_attention_HN(self.prev_hiddens)
            context_HKs = self.apply_attention_HK(self.prev_hiddens)
            context_HVs = self.apply_attention_HV(self.prev_hiddens)
            context_Hs = torch.cat((context_HNs, context_HKs, context_HVs), 2)
            context_MKs = self.apply_attention_MK(context_Hs)
            context_MVs = self.apply_attention_MV(context_Hs)

            cur_contexts = torch.cat((context_Hs, context_MKs, context_MVs), 2)
            reshaped_curr_contexts = self.W_Ct_reshape(cur_contexts)

            _, (hiddens, _) = self.lstm(prev_word_embeddings,
                                        (self.prev_hiddens.squeeze().unsqueeze(0),
                                         reshaped_curr_contexts.squeeze().unsqueeze(0)))
            # out: tensor of shape (batch_size, seq_length, hidden_size*2)

            # ===================== compute next output =====================

            hidden_Ns = self.W_n(hiddens).squeeze()
            hidden_Ks = self.W_k(hiddens).squeeze()
            hidden_Vs = self.W_v(hiddens).squeeze()

            self.update_history(di, hidden_Ns, hidden_Ks, hidden_Vs)

            P_Ns = self.normal_vocab_linear_layer(hidden_Ns)
            P_MKs = F.softmax(torch.bmm(self.key_memory, hidden_Ks.unsqueeze(2)))
            P_MVs = F.softmax(torch.bmm(self.value_memory, hidden_Vs.unsqueeze(2)))

            # bridge the the decoder’s semantic space with the memory’s semantic space
            P_Ms = ((P_MKs + P_MVs) / 2).squeeze()

            # gating mechanism
            new_g = torch.sigmoid(self.W_g(hiddens.squeeze()))

            # compute the distribution for the next decoder output
            P_xts = torch.cat((P_Ns * new_g, P_Ms * (1 - new_g)), dim=1)
            # some greedy algorithm that outputs the most likely word in distribution P_xt
            # output = greedy(P_xt, self.embedded_vocab, self.embedded_mem_val, self.embedded_mem_key)

            # === increment t ===
            self.t += 1
            self.prev_hiddens = hiddens.squeeze()
            next_word_indices = batch_data["sentence"][:, di, :].squeeze()
            for batch_i in range(self.batch_size):
                next_word = next_word_indices[batch_i]
                if next_word < self.normal_vocab_size:
                    loss += -torch.log(P_xts[batch_i][next_word])
                else:
                    associated_memory_index = 0
                    for curr_i, curr_word in enumerate(batch_data["keywords"][batch_i]):
                        if curr_word[0] == next_word:
                            associated_memory_index = curr_i
                            break
                    loss += -torch.log(P_xts[batch_i, self.normal_vocab_size + associated_memory_index])
            g_history[:, di, :] = new_g
            if use_teacher_forcing:
                prev_word_embeddings = self.word_embedder(batch_data["sentence"][:, di, :])
            else:
                curr_word_embeddings = torch.zeros(self.batch_size, 1, self.embedding_dim, device=device)
                for batch_i in range(self.batch_size):
                    topv, topi = P_xts[batch_i][:self.normal_vocab_size + self.current_mem_sizes[batch_i]].topk(1)

                    if topi >= self.normal_vocab_size:
                        topi = batch_data["keywords"][batch_i][topi % self.normal_vocab_size].view(-1)
                    curr_word_embeddings[batch_i] = self.word_embedder(topi.squeeze())

                    if batch_i == 0:
                        topis.append(topi)
                prev_word_embeddings = curr_word_embeddings

        return loss, g_history, topis

    def get_words_from_indexes(self, indexes):
        words = []
        for i in range(len(indexes)):
            words.append(self.word_lang.index2word[int(indexes[i])] )
        return words

    def predict(self, batch_data):
        # Single example
        self.prepare_memory(batch_data)
        self.prepare_history()
        # input_length = sentence_tensor.size[0]
        loss = 0
        prev_word_embeddings = self.word_embedder(batch_data["sentence"][:, 0, :])
        generated_sent = []
        g_history = torch.zeros(self.batch_size, self.max_len, 1, device=device, dtype=torch.float)
        for di in range(1, self.max_len):

            # ===================== compute context ========================
            context_HNs = self.apply_attention_HN(self.prev_hiddens)
            context_HKs = self.apply_attention_HK(self.prev_hiddens)
            context_HVs = self.apply_attention_HV(self.prev_hiddens)
            context_Hs = torch.cat((context_HNs, context_HKs, context_HVs), 2)
            context_MKs = self.apply_attention_MK(context_Hs)
            context_MVs = self.apply_attention_MV(context_Hs)

            cur_contexts = torch.cat((context_Hs, context_MKs, context_MVs), 2)
            reshaped_curr_contexts = self.W_Ct_reshape(cur_contexts)

            _, (hiddens, _) = self.lstm(prev_word_embeddings,
                                        (self.prev_hiddens.squeeze().view(1, 1, -1),
                                         reshaped_curr_contexts.view(1, 1, -1)))
            # out: tensor of shape (batch_size, seq_length, hidden_size*2)

            # ===================== compute next output =====================

            hidden_Ns = self.W_n(hiddens).view(1, -1)
            hidden_Ks = self.W_k(hiddens).view(1, -1)
            hidden_Vs = self.W_v(hiddens).view(1, -1)

            P_Ns = self.normal_vocab_linear_layer(hidden_Ns)
            P_MKs = F.softmax(torch.bmm(self.key_memory, hidden_Ks.unsqueeze(2)))
            P_MVs = F.softmax(torch.bmm(self.value_memory, hidden_Vs.unsqueeze(2)))

            # bridge the the decoder’s semantic space with the memory’s semantic space
            P_Ms = ((P_MKs + P_MVs) / 2).view(1, -1)

            # gating mechanism
            new_g = torch.sigmoid(self.W_g(hiddens.view(1, -1)))

            # compute the distribution for the next decoder output
            P_xts = torch.cat((P_Ns * new_g, P_Ms * (1 - new_g)), dim=1)
            # some greedy algorithm that outputs the most likely word in distribution P_xt
            # output = greedy(P_xt, self.embedded_vocab, self.embedded_mem_val, self.embedded_mem_key)

            topv, topi = P_xts[0][:self.normal_vocab_size + self.current_mem_sizes].topk(1)
            # === increment t ===
            self.t += 1
            self.prev_hiddens = hiddens.view(1, -1)
            next_word_indices = batch_data["sentence"][:, di, :].squeeze()
            if topi >= self.normal_vocab_size:
                topi = batch_data["keywords"][0][topi % self.normal_vocab_size].view(-1)

            generated_sent.append(topi)
            prev_word_embeddings = self.word_embedder(topi).unsqueeze(0)
            g_history[:, di, :] = new_g

        return loss, g_history, generated_sent

    def apply_attention_HN(self, prev_hiddens):
        # print("self.hist_N.size():", self.hist_N.size())
        # print("prev_hidden.size():", prev_hidden.size())
        concatenated = self.attn_HN(
            torch.cat((self.hist_N.view(self.batch_size, -1), prev_hiddens), 1).view(self.batch_size, 1, -1))
        attn_weights = F.softmax(concatenated, dim=2)
        attn_applied = torch.bmm(attn_weights, self.hist_N)
        return attn_applied

    def apply_attention_HK(self, prev_hiddens):
        concatenated = self.attn_HK(
            torch.cat((self.hist_K.view(self.batch_size, -1), prev_hiddens), 1).view(self.batch_size, 1, -1))
        attn_weights = F.softmax(concatenated, dim=2)
        attn_applied = torch.bmm(attn_weights, self.hist_K)
        return attn_applied

    def apply_attention_HV(self, prev_hiddens):
        concatenated = self.attn_HV(
            torch.cat((self.hist_V.view(self.batch_size, -1), prev_hiddens), 1).view(self.batch_size, 1, -1))
        attn_weights = F.softmax(concatenated, dim=2)
        attn_applied = torch.bmm(attn_weights, self.hist_V)
        return attn_applied

    def apply_attention_MK(self, context_Hs):
        prev_hidden_Ks = self.hist_K[:, self.t - 1, :].view(self.batch_size, 1, -1)
        concatenated = torch.cat((prev_hidden_Ks, context_Hs), 2)
        concatenated = self.attn_MK(concatenated)
        attn_weights = F.softmax(concatenated, dim=2)
        attn_applied = torch.bmm(attn_weights, self.key_memory)
        return attn_applied

    def apply_attention_MV(self, context_Hs):
        prev_hidden_Vs = self.hist_V[:, self.t - 1, :].view(self.batch_size, 1, -1)
        concatenated = torch.cat((prev_hidden_Vs, context_Hs), 2)
        concatenated = self.attn_MV(concatenated)
        attn_weights = F.softmax(concatenated, dim=2)
        attn_applied = torch.bmm(attn_weights, self.value_memory)
        return attn_applied
