# Biaffine Semantic Dependency Model Class & VISemantic Dependency Model Class (unused)
# Yu Zhang (adapted by Claudia Greenberg)
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# Main model used in this project
# Used in this adaptation
# -*- coding: utf-8 -*-

import torch.nn as nn, numpy as np

from supar.models.model import Model
from supar.modules import MLP, Biaffine, Triaffine
from supar.structs import SemanticDependencyLBP, SemanticDependencyMFVI
from supar.utils import Config
from supar.utils.maxtree import MaxTree


class BiaffineSemanticDependencyModel(Model):
    r"""
    The implementation of Biaffine Semantic Dependency Parser :cite:`dozat-manning-2018-simpler`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, required if lemma embeddings are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word representations. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 1200.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_edge_mlp (int):
            Edge MLP size. Default: 600.
        n_label_mlp  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of edge MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['tag', 'char', 'lemma'],
                 loss_type='entropy',
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.2,
                 n_encoder_hidden=1200,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_edge_mlp=600,
                 n_label_mlp=600,
                 edge_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        # FFNNs used for representations
        self.edge_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.edge_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.label_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)
        self.label_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)

        # Biaffine scoring functions used
        self.edge_attn = Biaffine(n_in=n_edge_mlp, n_out=2, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        
        # Loss functions
        self.loss_type = loss_type
        if loss_type == 'entropy':
            #Cross entropy loss
            self.criterion = nn.CrossEntropyLoss()
        
        elif loss_type == 'margin':
            #Maximum margin loss
            self.criterion = nn.MarginRankingLoss()

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained)
        return self

    def forward(self, words, sens, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            sens:
                batch sentences
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len, 2]`` holds scores of all possible edges.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each edge.
        """

        x = self.encode(words, sens, feats)

        # send through FFNNs for representations
        edge_d = self.edge_mlp_d(x)
        edge_h = self.edge_mlp_h(x)
        label_d = self.label_mlp_d(x)
        label_h = self.label_mlp_h(x)

        # send to biaffine scoring function

        # [batch_size, seq_len, seq_len, 2]
        s_edge = self.edge_attn(edge_d, edge_h).permute(0, 2, 3, 1)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_edge, s_label # scores!

    def loss(self, s_edge, s_label, labels, mask, type, label_preds = None):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            type:
                current stage (currently, either train or eval <validation>)
            label_preds:
                predicted labels (used in max margin loss)

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        if self.loss_type == "entropy":
            edge_mask = labels.ge(0) & mask
            edge_loss = self.criterion(s_edge[mask], edge_mask[mask].long())
            label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
        
        elif self.loss_type == "margin":
            pass # place max margin loss here (torch.nn.MarginRankingLoss)
            
        # print loss for epoch evaluation experimentation

        if(type == "train"):
            print("train: edge loss: " , edge_loss.item(), " label loss: ", label_loss.item())

        elif(type == "eval"):
            print("eval: edge loss: " , edge_loss.item(), " label loss: ", label_loss.item())

        return self.args.interpolation * label_loss + (1 - self.args.interpolation) * edge_loss

    def decode(self, s_edge, s_label, nodes):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            nodes:
                Sentence nodes; used for maximum spanning tree algorithm

        Returns:
            ~torch.LongTensor:
                Predicted labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        # Maximum Spanning Tree Algorithm
        # This had not been integrated but is recommended as future work.
        
        edge_scores = [] 
        
        # append the scores to edge_scores
        for row in s_edge[0]:
            row_scores = []
            for col in row:
                if float(col[0]) < 0.00001: # beginning scores (epoch 0) are 0
                    row_scores.append(float(col[1]))
                else:
                    row_scores.append(0)
            if row_scores != []:
                edge_scores.append(row_scores)
        
        final_edges = {}

        """if type(nodes) == list:
            old_input, old_output, representatives = MaxTree.olds(len(nodes[0]), np.array(edge_scores))
            MaxTree.chu_liu_edmonds(len(nodes[0]) - 1, np.array(edge_scores), list(nodes[0].values[1]), final_edges,  old_input, old_output, representatives)
        
        else:
            old_input, old_output, representatives = MaxTree.olds(len(nodes), np.array(edge_scores))
            MaxTree.chu_liu_edmonds(len(nodes) - 1, np.array(edge_scores), list(nodes.values[1]), final_edges,  old_input, old_output, representatives)"""

        # at this point, the argmax and max tree scores would be placed in a union
        # this union would be the theorised answer (would need to be tested)

        return s_label.argmax(-1).masked_fill_(s_edge.argmax(-1).lt(1), -1) # argmax function for greedy algorithm

# class not used directly in project
class VISemanticDependencyModel(BiaffineSemanticDependencyModel):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference :cite:`wang-etal-2019-second`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, required if lemma embeddings are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 1200.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_edge_mlp (int):
            Unary factor MLP size. Default: 600.
        n_pair_mlp (int):
            Binary factor MLP size. Default: 150.
        n_label_mlp  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of unary edge factor MLP layers. Default: .25.
        pair_mlp_dropout (float):
            The dropout ratio of binary factor MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        inference (str):
            Approximate inference methods. Default: ``mfvi``.
        max_iter (int):
            Max iteration times for inference. Default: 3.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.2,
                 n_encoder_hidden=1200,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_edge_mlp=600,
                 n_pair_mlp=150,
                 n_label_mlp=600,
                 edge_mlp_dropout=.25,
                 pair_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 inference='mfvi',
                 max_iter=3,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.edge_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.edge_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.pair_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_pair_mlp, dropout=pair_mlp_dropout, activation=False)
        self.pair_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_pair_mlp, dropout=pair_mlp_dropout, activation=False)
        self.pair_mlp_g = MLP(n_in=self.args.n_encoder_hidden, n_out=n_pair_mlp, dropout=pair_mlp_dropout, activation=False)
        self.label_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)
        self.label_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)

        self.edge_attn = Biaffine(n_in=n_edge_mlp, bias_x=True, bias_y=True)
        self.sib_attn = Triaffine(n_in=n_pair_mlp, bias_x=True, bias_y=True)
        self.cop_attn = Triaffine(n_in=n_pair_mlp, bias_x=True, bias_y=True)
        self.grd_attn = Triaffine(n_in=n_pair_mlp, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.inference = (SemanticDependencyMFVI if inference == 'mfvi' else SemanticDependencyLBP)(max_iter)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                The first and last are scores of all possible edges of shape ``[batch_size, seq_len, seq_len]``
                and possible labels on each edge of shape ``[batch_size, seq_len, seq_len, n_labels]``.
                Others are scores of second-order sibling, coparent and grandparent factors
                (``[batch_size, seq_len, seq_len, seq_len]``).

        """

        x = self.encode(words, feats)

        edge_d = self.edge_mlp_d(x)
        edge_h = self.edge_mlp_h(x)
        pair_d = self.pair_mlp_d(x)
        pair_h = self.pair_mlp_h(x)
        pair_g = self.pair_mlp_g(x)
        label_d = self.label_mlp_d(x)
        label_h = self.label_mlp_h(x)

        # [batch_size, seq_len, seq_len]
        s_edge = self.edge_attn(edge_d, edge_h)
        # [batch_size, seq_len, seq_len, seq_len], (d->h->s)
        s_sib = self.sib_attn(pair_d, pair_d, pair_h)
        s_sib = (s_sib.triu() + s_sib.triu(1).transpose(-1, -2)).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, seq_len], (d->h->c)
        s_cop = self.cop_attn(pair_h, pair_d, pair_h).permute(0, 3, 1, 2)
        s_cop = s_cop.triu() + s_cop.triu(1).transpose(-1, -2)
        # [batch_size, seq_len, seq_len, seq_len], (d->h->g)
        s_grd = self.grd_attn(pair_g, pair_d, pair_h).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_edge, s_sib, s_cop, s_grd, s_label

    def loss(self, s_edge, s_sib, s_cop, s_grd, s_label, labels, mask):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_cop (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-coparent triples.
            s_grd (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-grandparent triples.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        edge_mask = labels.ge(0) & mask
        edge_loss, marginals = self.inference((s_edge, s_sib, s_cop, s_grd), mask, edge_mask.long())
        label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
        loss = self.args.interpolation * label_loss + (1 - self.args.interpolation) * edge_loss
        return loss, marginals

    def decode(self, s_edge, s_label):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.LongTensor:
                Predicted labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        return s_label.argmax(-1).masked_fill_(s_edge.lt(0.5), -1)
