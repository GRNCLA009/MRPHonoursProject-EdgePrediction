# Biaffine Semantic Dependency Parser, VI Semantic Dependency Parser (unused)
# Yu Zhang (adapted by Claudia Greenberg)
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# Main parser used
# Used in this adaptation
# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn

from typing import Tuple

from supar.models import (BiaffineSemanticDependencyModel,
                          VISemanticDependencyModel)
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ChartMetric
from supar.utils.parallel import parallel, sync
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import CoNLL
from supar.utils.fn import pad

logger = get_logger(__name__)


class BiaffineSemanticDependencyParser(Parser):
    r"""
    The implementation of Biaffine Semantic Dependency Parser :cite:`dozat-manning-2018-simpler`.
    """

    NAME = 'biaffine-semantic-dependency'
    MODEL = BiaffineSemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.TAG = self.transform.TAG
        self.LABEL = self.transform.EDGES # edges used to be in PHEAD

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (Union[str, Iterable]):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()), form = self.get_form()) # additional form requirment

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, verbose=True, **kwargs):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for evaluation. Both a filename and a list of instances are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()), form = self.get_form()) # additional form requirment

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, prob=False,
                verbose=True, **kwargs):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for prediction.
                - a filename. If ends with `.txt`, the parser will seek to make predictions line by line from plain texts.
                - a list of instances.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """
        
        print(self.args) # printing statement
        return super().predict(**Config().update(locals()), form = self.args.form) # additional form requirment

    @parallel()
    def _train(self, loader, form: Tuple):
        bar, metric = progress_bar(loader), ChartMetric()
        
        for i, batch in enumerate(bar, 1):
            *feats, labels = batch # words removed and set separately

            if form[1] is not None:
                feats = ['char'] # used for this adaptation; should be changed if a future adaptation occurs
            
            # add to words
            # add to sens (sentences) for tracking
            if(form[0] is not None):
                words = [] # append here
                sens = []
                for sen in batch.sentences:
                    if(sen in loader.dataset.sentences):
                        words.append(next(form[0].transform([sen.tokens])))
                        sens.append(sen)
                    else:
                        print("Error in training!")
            
                words = pad(words)

            # add to chars
            if form[1] is not None:
                chars = []
                for sen in batch.sentences:
                    if(sen in loader.dataset.sentences):
                        chars.append(next(form[1].transform([sen.tokens])))
                        
                    else:
                        print("Error in char training!")

                chars = pad(chars)
                feats = [chars] #only useful for this adaptation

            mask = batch.mask
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            with sync(self.model, i % self.args.update_steps == 0):
                with torch.autocast(self.device, enabled=self.args.amp):
                    # scores
                    s_edge, s_label = self.model(words, sens, feats)
                    # predicted labels
                    label_preds = self.model.decode(s_edge, s_label, loader.dataset.sentences[i-1]) # moved here for max margin 
                    # loss calculation
                    loss = self.model.loss(s_edge, s_label, labels, mask, "train", label_preds)
                    loss = loss / self.args.update_steps
                self.scaler.scale(loss).backward()
            if i % self.args.update_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

            metric += ChartMetric(loss, label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
                        
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - {metric}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader, form: Tuple):
        metric = ChartMetric()

        for batch in progress_bar(loader):
            *feats, labels = batch # words done separately

            if form[1] is not None:
                feats = ['char'] # only useful for this adaptation

            if(form[0] is not None):
                words = [] # words append here
                sens = [] # tracking the sentences
                for sen in batch.sentences:
                    if(sen in loader.dataset.sentences):
                        words.append(next(form[0].transform([sen.tokens])))
                        sens.append(sen)
                    else:
                        print("Error in evaluation!")
            words = pad(words)

            # append characters here
            if form[1] is not None:
                chars = []
                for sen in batch.sentences:
                    if(sen in loader.dataset.sentences):
                        chars.append(next(form[1].transform([sen.tokens])))
                        
                    else:
                        print("Error in char evaluation!")

                chars = pad(chars)
                feats = [chars]

            mask = batch.mask
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                # scores
                s_edge, s_label = self.model(words, sens, feats)            
                # prediction
                label_preds = self.model.decode(s_edge, s_label, loader.dataset.sentences) #not i-1 like in train?
                # loss
                loss = self.model.loss(s_edge, s_label, labels, mask, "eval", label_preds)
            metric += ChartMetric(loss, label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader, form: Tuple):
        for batch in progress_bar(loader):
            feats = batch # removed words here

            if form[1] is not None:
                feats = ['char'] # only good for this adaptation
            
            # words appended
            # tracking sentences appended
            if(form[0] is not None):
                words = [] # append here
                sens = []
                for sen in batch.sentences:
                    if(sen in loader.dataset.sentences):
                        words.append(next(form[0].transform([sen.tokens])))
                        sens.append(sen)
                    else:
                        print("Error in prediction!")
            words = pad(words)

            # chars appended
            if form[1] is not None:
                chars = []
                for sen in batch.sentences:
                    if(sen in loader.dataset.sentences):
                        chars.append(next(form[1].transform([sen.tokens])))
                        
                    else:
                        print("Error in char predict!")
                chars = pad(chars)
                feats = [chars]

            mask, lens = batch.mask, (batch.lens - 1).tolist()
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                # scores
                s_edge, s_label = self.model(words, sens, feats)
            # prediction
            label_preds = self.model.decode(s_edge, s_label, loader.dataset.sentences).masked_fill(~mask, -1) #not i-1 like train?
            batch.labels = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                                                   for row in chart[1:i, :i].tolist()])
                            for i, chart in zip(lens, label_preds)]
            if self.args.prob: # probabilities
                batch.probs = [prob[1:i, :i].cpu() for i, prob in zip(lens, s_edge.softmax(-1).unbind())]
            yield from batch.sentences

    @classmethod
    def build(cls, path, min_freq=7, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (Dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(parser.device)
            return parser

        logger.info("Building the fields")
        TAG, LEMMA = None, None

        # set form
        cls.set_form(cls, args.encoder, args.fix_len, args.feat)
        form = cls.get_form(cls)
        
        # initialise tag and lemma
        if args.encoder != 'bert':
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=PAD, unk=UNK, bos=BOS, lower=True)

        # labels!
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        STARTS, ENDS = (), ()
        
        # initialise transform (ConLL object)
        transform = CoNLL(TAG=TAG, START = STARTS, END = ENDS, EDGES=LABEL)

        # initialise dataset
        train = Dataset(transform, args.train, form = form, **args)
        
        # build fields
        if args.encoder != 'bert':
            form[0].build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(train)
            if form[1] is not None:
                form[1].build(train)
            if LEMMA is not None:
                LEMMA.build(train)
    
        LABEL.build(train)
        
        # update args
        args.update({
            'n_words': len(form[0].vocab) if args.encoder == 'bert' else form[0].vocab.n_init,
            'n_labels': len(LABEL.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(form[1].vocab) if form[1] is not None else None,
            'char_pad_index': form[1].pad_index if form[1] is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': form[3].pad_index if form[3] is not None else None,
            'pad_index': form[0].pad_index,
            'unk_index': form[0].unk_index,
            'bos_index': form[0].bos_index,
            'form': form
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(form[0].embed if hasattr(form[0], 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser
    
    def get_form(self):
        return self.FORM
    
    # the form is the main variable carried around because of the span adaptation
    def set_form(self, enc, fix_len, feats=None, f=None):
        # type is either args or no args
        
        bert = "SpanBERT/spanbert-base-cased"
        
        if(f is None):
            
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True) #no eos!
            
            CHAR, ELMO, BERT = None, None, None
            if enc == 'bert':
                t = TransformerTokenizer(bert)
                WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=fix_len, tokenize=t) #not empty
                WORD.vocab = t.vocab 
            
            else:
                
                if(feats is not None):
                    if 'char' in feats:
                        CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=fix_len)
                    
                    if 'bert' in feats:
                        t = TransformerTokenizer(bert)
                        BERT = SubwordField('bert', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=fix_len, tokenize=t)
                        BERT.vocab = t.vocab
        
            f = (WORD, CHAR, ELMO, BERT)
                    
        self.FORM = f


class VISemanticDependencyParser(BiaffineSemanticDependencyParser):
    r"""
    The implementation of Semantic Dependency Parser using Variational Inference :cite:`wang-etal-2019-second`.
    """

    NAME = 'vi-semantic-dependency'
    MODEL = VISemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, amp=False, cache=False,
              verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (Union[str, Iterable]):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, verbose=True, **kwargs):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for evaluation. Both a filename and a list of instances are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, amp=False, cache=False, prob=False,
                verbose=True, **kwargs):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for prediction.
                - a filename. If ends with `.txt`, the parser will seek to make predictions line by line from plain texts.
                - a list of instances.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (Dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @parallel()
    def _train(self, loader):
        bar, metric = progress_bar(loader), ChartMetric()

        for i, batch in enumerate(bar, 1):
            words, *feats, labels = batch
            mask = batch.mask
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            with sync(self.model, i % self.args.update_steps == 0):
                with torch.autocast(self.device, enabled=self.args.amp):
                    s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
                    loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd, s_label, labels, mask)
                    loss = loss / self.args.update_steps
                self.scaler.scale(loss).backward()
            if i % self.args.update_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

            label_preds = self.model.decode(s_edge, s_label)
            metric + ChartMetric(loss, label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
            
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - {metric}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = ChartMetric()

        for batch in progress_bar(loader):
            words, *feats, labels = batch
            mask = batch.mask
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
                loss, s_edge = self.model.loss(s_edge, s_sib, s_cop, s_grd, s_label, labels, mask)
            label_preds = self.model.decode(s_edge, s_label)
            metric += ChartMetric(loss, label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        for batch in progress_bar(loader):
            words, *feats = batch
            mask, lens = batch.mask, (batch.lens - 1).tolist()
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_sib, s_cop, s_grd, s_label = self.model(words, feats)
                s_edge = self.model.inference((s_edge, s_sib, s_cop, s_grd), mask)
            label_preds = self.model.decode(s_edge, s_label).masked_fill(~mask, -1)
            batch.labels = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                                                   for row in chart[1:i, :i].tolist()])
                            for i, chart in zip(lens, label_preds)]
            if self.args.prob:
                batch.probs = [prob[1:i, :i].cpu() for i, prob in zip(lens, s_edge.unbind())]
            yield from batch.sentences
