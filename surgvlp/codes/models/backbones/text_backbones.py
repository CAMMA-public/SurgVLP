import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import re
import torch.nn.functional as F
import numpy as np
from ..backbones.text_basicblocks import Transformer, LayerNorm
from ...registry import MODELS

@MODELS.register_module(name='text_backbones/BertEncoder')
class BertEncoder(nn.Module):
    def __init__(self, text_bert_type, text_last_n_layers, 
                text_aggregate_method, text_norm, text_embedding_dim, text_freeze_bert, text_agg_tokens):
        super(BertEncoder, self).__init__()

        self.bert_type = text_bert_type
        self.last_n_layers = text_last_n_layers
        self.aggregate_method = text_aggregate_method
        self.norm = text_norm
        self.embedding_dim = text_embedding_dim
        self.freeze_bert = text_freeze_bert
        self.agg_tokens = text_agg_tokens

        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}


        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):

        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, ids=None, attn_mask=None, token_type=None, input_embedding=None):

        if input_embedding is not None:
            outputs = self.model(inputs_embeds=input_embedding, attention_mask=attn_mask)
        else:
            outputs = self.model(ids, attn_mask, token_type)

        # aggregate intermetidate layers
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers :]
            )  # layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3)

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            sent_embeddings = embeddings.mean(axis=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1)
                sent_embeddings = sent_embeddings.sum(axis=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            # outputs: ['last_hidden_state', 'pooler_output', 'hidden_states']
            word_embeddings, sent_embeddings = outputs[0], outputs[1]
            sents = None

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        return word_embeddings, sent_embeddings, sents
