# Transformer Embedding Class, Elmo Embedding Class (unused), Scalar Mixx Class
# Yu Zhang (adapted by Claudia Greenberg)
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# BERT (used) and ELMo (unused) Embeddings and an assisting Scalar Mix class 
# Used in this adaptation
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from supar.utils.fn import pad
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils import transform

class TransformerEmbedding(nn.Module): # used for BERT embeddings
    r"""
    Bidirectional transformer embeddings of words from various transformer architectures :cite:`devlin-etal-2019-bert`.

    Args:
        model (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of BERT layers to use. If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        stride (int):
            A sequence longer than max length will be splitted into several small pieces
            with a window size of ``stride``. Default: 10.
        pooling (str):
            Pooling way to get from token piece embeddings to token embedding.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        pad_index (int):
            The index of the padding token in BERT vocabulary. Default: 0.
        mix_dropout (float):
            The dropout ratio of BERT layers. This value will be passed into the :class:`ScalarMix` layer. Default: 0.
        finetune (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(
        self,
        model: str,
        n_layers: int,
        n_out: int = 0,
        stride: int = 256,
        pooling: str = 'mean',
        pad_index: int = 0,
        mix_dropout: float = .0,
        finetune: bool = False
    ) -> TransformerEmbedding:
        super().__init__()

        from transformers import AutoModel
        try: # loading BERT model
            self.bert = AutoModel.from_pretrained(model, output_hidden_states=True, local_files_only=True)
        except Exception:
            self.bert = AutoModel.from_pretrained(model, output_hidden_states=True, local_files_only=False)
        self.bert = self.bert.requires_grad_(finetune)
        self.tokenizer = TransformerTokenizer(model) # tokenizer class used

        self.model = model
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.pooling = pooling
        self.pad_index = pad_index
        self.mix_dropout = mix_dropout
        self.finetune = finetune
        self.max_len = int(max(0, self.bert.config.max_position_embeddings) or 1e12) - 2
        self.stride = min(stride, self.max_len)

        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"stride={self.stride}, pooling={self.pooling}, pad_index={self.pad_index}"
        if self.mix_dropout > 0:
            s += f", mix_dropout={self.mix_dropout}"
        if self.finetune:
            s += f", finetune={self.finetune}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, subwords: torch.Tensor, sens) -> torch.Tensor:
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.

        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        # main BERT span adaptation
        # subwords are split into start and end span subwords
        # the embeddings are created and are then
        # concated by subtration
        
        starts, ends, subwords_start, subwords_end = [], [], [], []

        # extract start and end span information
        # append start tokens
        if(type(sens) == transform.CoNLLSentence):        
            starts.append(sens.values[2])
            ends.append(sens.values[3])
            subwords_start.append([[0]*(len(subwords[0][0]))])
            subwords_end.append([subwords[0][0].tolist()]) 
            sens = [sens]

        else:
            for i, sen in enumerate(sens):
                starts.append(sen.values[2])
                ends.append(sen.values[3])
                subwords_start.append([[0]*(len(subwords[0][0]))])
                subwords_end.append([subwords[0][0].tolist()])

        # add subwords to start and end tensors
        for sen in range(len(starts)): # for each sentence
            for index in range(len(starts[sen])): # for each subword in the sentence
                
                subwords_start[sen].append(subwords[sen][int(starts[sen][index]) + 1].tolist())

                if(int(ends[sen][index]) < (len(subwords[sen]) - 1)):
                    subwords_end[sen].append(subwords[sen][int(ends[sen][index]) + 1].tolist())
            
                elif(int(ends[sen][index]) == (len(subwords[sen]) - 1)): # add SEP token (EOS)
                    temp = [0]*(len(subwords[sen][0]))
                    temp[0] = 102
                    subwords_end[sen].append(temp)
            
                else:
                    print("PROBLEM IN PRETRAINED!")
            
            # convert to tensors
            subwords_start[sen] = torch.tensor(subwords_start[sen])
            subwords_end[sen] = torch.tensor(subwords_end[sen])
	
        if(torch.cuda.is_available()): # for chpc

            subwords_start = pad(subwords_start).cuda()
            subwords_end = pad(subwords_end).cuda()

        else: # for laptop
            subwords_start = pad(subwords_start)
            subwords_end = pad(subwords_end)

        # perform embeddings for start tokens

        mask_start = subwords_start.ne(self.pad_index)
        lens_start = mask_start.sum((1, 2))
        # [batch_size, n_subwords]        
        subwords_start = pad(subwords_start[mask_start].split(lens_start.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        bert_mask_start = pad(mask_start[mask_start].split(lens_start.tolist()), 0, padding_side=self.tokenizer.padding_side)

        # return the hidden states of all layers
        bert_start = self.bert(subwords_start[:, :self.max_len], attention_mask=bert_mask_start[:, :self.max_len].float())[-1]
        # [n_layers, batch_size, max_len, hidden_size]
        bert_start = bert_start[-self.n_layers:]
        # [batch_size, max_len, hidden_size]
        bert_start = self.scalar_mix(bert_start)
        # [batch_size, n_subwords_start, hidden_size]
        for i in range(self.stride, (subwords_start.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part_start = self.bert(subwords_start[:, i:i+self.max_len], attention_mask=bert_mask_start[:, i:i+self.max_len].float())[-1]
            bert_start = torch.cat((bert_start, self.scalar_mix(part_start[-self.n_layers:])[:, self.max_len-self.stride:]), 1)

        # [batch_size, seq_len]
        bert_lens_start = mask_start.sum(-1)
        bert_lens_start = bert_lens_start.masked_fill_(bert_lens_start.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed_start = bert_start.new_zeros(*mask_start.shape, self.hidden_size).masked_scatter_(mask_start.unsqueeze(-1), bert_start[bert_mask_start])
        
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            embed_start = embed_start[:, :, 0]
        elif self.pooling == 'last':
            embed_start = embed_start.gather(2, (bert_lens_start-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        else:
            embed_start = embed_start.sum(2) / bert_lens_start.unsqueeze(-1)
        embed_start = self.projection(embed_start) # HERE IS BERT!

        # perform embeddings for end tokens

        mask_end = subwords_end.ne(self.pad_index)
        lens_end = mask_end.sum((1, 2))
        # [batch_size, n_subwords]
        subwords_end = pad(subwords_end[mask_end].split(lens_end.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        bert_mask_end = pad(mask_end[mask_end].split(lens_end.tolist()), 0, padding_side=self.tokenizer.padding_side)

        # return the hidden states of all layers
        bert_end = self.bert(subwords_end[:, :self.max_len], attention_mask=bert_mask_end[:, :self.max_len].float())[-1]
        # [n_layers, batch_size, max_len, hidden_size]
        bert_end = bert_end[-self.n_layers:]
        # [batch_size, max_len, hidden_size]
        bert_end = self.scalar_mix(bert_end)
        # [batch_size, n_subwords_end, hidden_size]
        for i in range(self.stride, (subwords_end.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part_end = self.bert(subwords_end[:, i:i+self.max_len], attention_mask=bert_mask_end[:, i:i+self.max_len].float())[-1]
            bert_end = torch.cat((bert_end, self.scalar_mix(part_end[-self.n_layers:])[:, self.max_len-self.stride:]), 1)

        # [batch_size, seq_len]
        bert_lens_end = mask_end.sum(-1)
        bert_lens_end = bert_lens_end.masked_fill_(bert_lens_end.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed_end = bert_end.new_zeros(*mask_end.shape, self.hidden_size).masked_scatter_(mask_end.unsqueeze(-1), bert_end[bert_mask_end])
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            embed_end = embed_end[:, :, 0]
        elif self.pooling == 'last':
            embed_end = embed_end.gather(2, (bert_lens_end-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        else:
            embed_end = embed_end.sum(2) / bert_lens_end.unsqueeze(-1)
        embed_end = self.projection(embed_end) #HERE IS BERT!

        # concatenate end and start token embeddings by subtraction (end - start)
        embed = torch.subtract(embed_end, embed_start)

        return embed

class ELMoEmbedding(nn.Module):
    r"""
    Contextual word embeddings using word-level bidirectional LM :cite:`peters-etal-2018-deep`.

    Args:
        model (str):
            The name of the pretrained ELMo registered in `OPTION` and `WEIGHT`. Default: ``'original_5b'``.
        bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of sentence outputs.
            Default: ``(True, True)``.
        n_out (int):
            The requested size of the embeddings. If 0, uses the default size of ELMo outputs. Default: 0.
        dropout (float):
            The dropout ratio for the ELMo layer. Default: 0.
        finetune (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.
    """

    OPTION = {
        'small': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json',  # noqa
        'medium': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json',  # noqa
        'original': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',  # noqa
        'original_5b': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',  # noqa
    }
    WEIGHT = {
        'small': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',  # noqa
        'medium': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5',  # noqa
        'original': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',  # noqa
        'original_5b': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',  # noqa
    }

    def __init__(
        self,
        model: str = 'original_5b',
        bos_eos: Tuple[bool, bool] = (True, True),
        n_out: int = 0,
        dropout: float = 0.5,
        finetune: bool = False
    ) -> ELMoEmbedding:
        super().__init__()

        from allennlp.modules import Elmo

        self.elmo = Elmo(options_file=self.OPTION[model],
                         weight_file=self.WEIGHT[model],
                         num_output_representations=1,
                         dropout=dropout,
                         finetune=finetune,
                         keep_sentence_boundaries=True)

        self.model = model
        self.bos_eos = bos_eos
        self.hidden_size = self.elmo.get_output_dim()
        self.n_out = n_out or self.hidden_size
        self.dropout = dropout
        self.finetune = finetune

        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_out={self.n_out}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.finetune:
            s += f", finetune={self.finetune}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, chars: torch.LongTensor) -> torch.Tensor:
        r"""
        Args:
            chars (~torch.LongTensor): ``[batch_size, seq_len, fix_len]``.

        Returns:
            ~torch.Tensor:
                ELMo embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        x = self.projection(self.elmo(chars)['elmo_representations'][0])
        if not self.bos_eos[0]:
            x = x[:, 1:]
        if not self.bos_eos[1]:
            x = x[:, :-1]
        return x

class ScalarMix(nn.Module):
    r"""
    Computes a parameterized scalar mixture of :math:`N` tensors, :math:`mixture = \gamma * \sum_{k}(s_k * tensor_k)`
    where :math:`s = \mathrm{softmax}(w)`, with :math:`w` and :math:`\gamma` scalar parameters.

    Args:
        n_layers (int):
            The number of layers to be mixed, i.e., :math:`N`.
        dropout (float):
            The dropout ratio of the layer weights.
            If dropout > 0, then for each scalar weight, adjusts its softmax weight mass to 0
            with the dropout probability (i.e., setting the unnormalized weight to -inf).
            This effectively redistributes the dropped probability mass to all other weights.
            Default: 0.
    """

    def __init__(self, n_layers: int, dropout: float = .0) -> ScalarMix:
        super().__init__()

        self.n_layers = n_layers

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            tensors (List[~torch.Tensor]):
                :math:`N` tensors to be mixed.

        Returns:
            The mixture of :math:`N` tensors.
        """

        return self.gamma * sum(w * h for w, h in zip(self.dropout(self.weights.softmax(-1)), tensors))
