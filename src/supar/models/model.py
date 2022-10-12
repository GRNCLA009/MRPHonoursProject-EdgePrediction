# Model Class
# Yu Zhang (adapted by Claudia Greenberg)
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# Parent class to other models
# Used in this adaptation
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from supar.modules import (CharLSTM, ELMoEmbedding, IndependentDropout,
                           SharedDropout, TransformerEmbedding,
                           TransformerWordEmbedding, VariationalLSTM)
from supar.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from supar.utils import Config, transform, fn


class Model(nn.Module):

    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 elmo_bos_eos=(True, True),
                 elmo_dropout=0.5,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 encoder_dropout=.33,
                 pad_index=0,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())

        if encoder == 'lstm': # used in BiLSTM module
            self.word_embed = nn.Embedding(num_embeddings=self.args.n_words, # used
                                           embedding_dim=self.args.n_embed)

            n_input = self.args.n_embed
            if self.args.n_pretrained != self.args.n_embed:
                n_input += self.args.n_pretrained
            if 'tag' in self.args.feat:
                self.tag_embed = nn.Embedding(num_embeddings=self.args.n_tags,
                                              embedding_dim=self.args.n_feat_embed)
                n_input += self.args.n_feat_embed
            if 'char' in self.args.feat: # used
                self.char_embed = CharLSTM(n_chars=self.args.n_chars,
                                           n_embed=self.args.n_char_embed,
                                           n_hidden=self.args.n_char_hidden,
                                           n_out=self.args.n_feat_embed,
                                           pad_index=self.args.char_pad_index,
                                           dropout=self.args.char_dropout)
                n_input += self.args.n_feat_embed
            if 'lemma' in self.args.feat:
                self.lemma_embed = nn.Embedding(num_embeddings=self.args.n_lemmas,
                                                embedding_dim=self.args.n_feat_embed)
                n_input += self.args.n_feat_embed
            if 'elmo' in self.args.feat:
                self.elmo_embed = ELMoEmbedding(n_out=self.args.n_plm_embed,
                                                bos_eos=self.args.elmo_bos_eos,
                                                dropout=self.args.elmo_dropout,
                                                finetune=self.args.finetune)
                n_input += self.elmo_embed.n_out
            if 'bert' in self.args.feat:
                self.bert_embed = TransformerEmbedding(model=self.args.bert,
                                                       n_layers=self.args.n_bert_layers,
                                                       n_out=self.args.n_plm_embed,
                                                       pooling=self.args.bert_pooling,
                                                       pad_index=self.args.bert_pad_index,
                                                       mix_dropout=self.args.mix_dropout,
                                                       finetune=self.args.finetune)
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=self.args.embed_dropout)
            # LSTM used
            self.encoder = VariationalLSTM(input_size=n_input,
                                           hidden_size=self.args.n_encoder_hidden//2,
                                           num_layers=self.args.n_encoder_layers,
                                           bidirectional=True,
                                           dropout=self.args.encoder_dropout)
            self.encoder_dropout = SharedDropout(p=self.args.encoder_dropout)
        elif encoder == 'transformer':
            self.word_embed = TransformerWordEmbedding(n_vocab=self.args.n_words,
                                                       n_embed=self.args.n_embed,
                                                       pos=self.args.pos,
                                                       pad_index=self.args.pad_index)
            self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)
            self.encoder = TransformerEncoder(layer=TransformerEncoderLayer(n_heads=self.args.n_encoder_heads,
                                                                            n_model=self.args.n_encoder_hidden,
                                                                            n_inner=self.args.n_encoder_inner,
                                                                            attn_dropout=self.args.encoder_attn_dropout,
                                                                            ffn_dropout=self.args.encoder_ffn_dropout,
                                                                            dropout=self.args.encoder_dropout),
                                              n_layers=self.args.n_encoder_layers,
                                              n_model=self.args.n_encoder_hidden)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
        elif encoder == 'bert': # used in BERT module
            # BERT encoder used
            self.encoder = TransformerEmbedding(model=self.args.bert,
                                                n_layers=self.args.n_bert_layers,
                                                pooling=self.args.bert_pooling,
                                                pad_index=self.args.pad_index,
                                                mix_dropout=self.args.mix_dropout,
                                                finetune=True)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
            self.args.n_encoder_hidden = self.encoder.n_out

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    # embeddings for the BiLSTM module used
    def embed(self, words, feats=None, feat_=None):
        
        # feat_ is the character embeddings (see encode function). 
        # In an adaptation, this should be changed to a tuple 
        #   if additional features are used. 

        ext_words = words
        
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            if self.args.n_embed == self.args.n_pretrained:
                word_embed += pretrained
            else:
                word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)

        # feature extraction
        feat_embed = []
        if 'tag' in self.args.feat:
            feat_embed.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat: # used in our adaptation (feat.pop(0) occurs in encode function)
            feat_embed.append(self.char_embed(feat_))
        if 'elmo' in self.args.feat:
            feat_embed.append(self.elmo_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embed.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embed.append(self.lemma_embed(feats.pop(0)))
        
        # dropout (occurs in our case)
        if isinstance(self.embed_dropout, IndependentDropout):
            if len(feat_embed) == 0:
                # initially, feat was not allowed to be empty. This was changed.
                print("feat currently empty!")
                # dropout
                embed = torch.cat(self.embed_dropout(word_embed), -1)
            else:
                # concatenate word and feature embeddings (char, in our case)
                embed = torch.cat(self.embed_dropout(word_embed, torch.cat(feat_embed, -1)), -1)
        
        # no dropout (does not occur in our case, therefore untouched)
        else:
            embed = word_embed
            if len(feat_embed) > 0:
                embed = torch.cat((embed, torch.cat(feat_embed, -1)), -1)
            embed = self.embed_dropout(embed)
        return embed

    # encodes the nodes
    def encode(self, words, sens, feats=None):
        
        if self.args.encoder == 'lstm': # used in our BiLSTM module
            
            # separate tokens into two tensors, one with start tokens and one with end tokens
            words_start, words_end = self.separate_words(words, sens, "words")     
                        
            if feats != []:
                # extract character embeddings 
                # our adaptation only used character features
                # the below should should be adapted if more features are used)
                chars = feats.pop(0)
                
                # separate token characters into two tensors, one with start characters and one with end characters
                chars_start, chars_end = self.separate_words(chars, sens, "chars")
                            
            else:
                chars_start, chars_end = None, None

            # start and end tensors are treated independently
            
            x_end = pack_padded_sequence(self.embed(words_end, feats, chars_end), words_end.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x_start = pack_padded_sequence(self.embed(words_start, feats, chars_start), words_start.ne(self.args.pad_index).sum(1).tolist(), True, False)
            
            x_end, _ = self.encoder(x_end)
            x_start, _ = self.encoder(x_start)
            
            x_end, _ = pad_packed_sequence(x_end, True, total_length=words_end.shape[1])
            x_start, _ = pad_packed_sequence(x_start, True, total_length=words_start.shape[1])
            
            # final embedding is the subtraction of end and start token embeddings
            x = torch.subtract(x_end, x_start)
            
        elif self.args.encoder == 'transformer':
            x = self.encoder(self.embed(words, feats), words.ne(self.args.pad_index))
        else: # used in our BERT module
            x = self.encoder(words, sens)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError
    
    # separates words or characters into two separate tensors
    # this forms part of the span adaptation
    def separate_words(self, words, sens, word_type):
        
        starts, ends = [], []

        # extract start and end token informaton
        if(type(sens) == transform.CoNLLSentence):
            starts.append(sens.values[2])
            ends.append(sens.values[3])
            sens = [sens]
        
        else:
            for i, sen in enumerate(sens):
                starts.append(sen.values[2])
                ends.append(sen.values[3])
        
        
        # append bos token
        
        # each word is represented by one value
        if word_type == "words":
            
            #words example: torch.tensor([[2, 34, 35, 36], [2, 0, 3, 4]])
            
            words_start, words_end = [], []
            
            for sent in range(len(starts)):
                words_start.append([0])
                words_end.append([2])
        # each word is represnted by an array of values (for all characters)            
        elif word_type == "chars":
            
            #words example: torch.tensor([[[2, 0, 0], [34, 5, 0], [35, 0, 0], [36, 12, 14]], [[0, 60, 1], [10, 1, 12], [3, 1, 45], [4, 3, 6]]])
            
            words_start, words_end = [], []
            
            for sent in range(len(starts)):
                temp_start, temp_end = [], []
                
                for char in range(len(words[0][0])):
                    temp_start.append(0)

                    if(char == 0):
                        temp_end.append(2)
                    else:
                        temp_end.append(0)
                
                words_start.append([temp_start])
                words_end.append([temp_end])
        
        # UNK = 1, BOS = 2, EOS = 3 <not used>
        
        # append separate values to tensors

        for sen in range(len(starts)): # for each sentence
            for index in range(len(starts[sen])): # for each node
                if word_type == "words":
                    
                    words_start[sen].append(int(words[sen][int(starts[sen][index]) + 1]))

                    # since some nodes only spans one token, the end token is the first token after the span ends
                    if(int(ends[sen][index]) < (len(words[sen]) - 1)):
                        words_end[sen].append(int(words[sen][int(ends[sen][index]) + 1]))
                
                    # when the end of the span is the final token, a PAD token is set
                    elif(int(ends[sen][index]) == (len(words[sen]) - 1)): 
                        words_end[sen].append(0)
                
                    else:
                        print("PROBLEM IN SEPARATE WORDS!")
                
                elif word_type == "chars":
                    
                    words_start[sen].append(words[sen][int(starts[sen][index]) + 1].tolist()) #words[0][1] 

                    if(int(ends[sen][index]) < (len(words[sen]) - 1)):
                        words_end[sen].append(words[sen][int(ends[sen][index]) + 1].tolist())
                
                    elif(int(ends[sen][index]) == (len(words[sen]) - 1)): #add PAD token
                        temp = [0]*(len(words[sen][0].tolist()))
                        
                        words_end[sen].append(temp)
                
                    else:
                        print("PROBLEM IN SEPARATE CHARS!")

            # convert arrays to tensors                                        
            words_start[sen] = torch.tensor(words_start[sen])
            words_end[sen] = torch.tensor(words_end[sen])
        
        if(torch.cuda.is_available()): # for chpc

            words_start = fn.pad(words_start).cuda()
            words_end = fn.pad(words_end).cuda()

        else: # for laptop
            
            words_start = fn.pad(words_start)
            words_end = fn.pad(words_end)
        
        return words_start, words_end
