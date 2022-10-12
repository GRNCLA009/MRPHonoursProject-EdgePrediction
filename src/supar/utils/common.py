# Common Variables
# Yu Zhang (unedited)
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# Used in this adaptation
# -*- coding: utf-8 -*-

import os

PAD = '<pad>'
UNK = '<unk>'
BOS = '<bos>'
EOS = '<eos>'
NUL = '<nul>'

MIN = -1e32
INF = float('inf')

CACHE = os.path.expanduser('~/.cache/supar')
