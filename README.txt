UCT Honours Project: 
   Meaning Representation Parsing: 
   The Edge Prediction Component of a Semantic Graph Parser
   by Claudia Greenberg, GRNCLA009
September 2022
README File for the Codebase Presented

How to Run Main Module:
- A virtual environment is recommended:
    -- python3 -m venv venv
    -- To run the virtual environment, type ".\venv\Scripts\activate". "(venv)" should appear.
    -- pip3 install -Ur requirements.txt (only those used in the project are included)
- BERT module training example:
    -- python -u .\src\biaffine_sdp.py train -b 
        --encoder bert 
        -c .\file_examples\config.txt 
        --train .\file_examples\train_data.conllu 
        --dev .\file_examples\dev_data.conllu 
        --test .\file_examples\test_data.conllu 
        --num_epochs 5 
        --path BERT_module 
        --lr 5e-05
- BiLSTM module training example:
    -- python -u .\src\biaffine_sdp.py train -b 
        --encoder lstm 
        -f char 
        -c .\file_examples\config.txt 
        --train .\file_examples\train_data.conllu 
        --dev .\file_examples\dev_data.conllu 
        --test .\file_examples\test_data.conllu 
        --num_epochs 5 
        --path BiLSTM_module 
        --lr 5e-03
- See src.supar.biaffine_sdp.py and src.supar.cmds.cmd.py for additional parameter information.

If something goes wrong, please try replacing all "\" with "/" or perhaps try "python3" not "python".

This code should run smoothly. For any technical issues, please email grncla009@myuct.ac.za.

All classes and additional files (including license and citation) are retained from the original codebase. 
This is to reduce any complications. Most unused and unedited classes are left untouched.
The original codebase is well-documented. Additional documentation is provided when necessary.
All classes are marked if unused and/or unedited. 
The classes unedited but used were analysed at the same intricacy level and magnitude as the edited classes.
The original codebase is available at: https://github.com/yzhangcs/parser
    - Accompanying documentation is available at: https://parser.yzhang.site/en/latest/

*FULL CLASS LIST:*

General Classes:
    - supar/__init__.py
    - biaffine_sdp.py 
        -- main class to run the parser code
        -- to run: see above
    - data_conversion.py
        -- converts raw data into appropriate form
        -- to run: python .\src\data_conversion.py 
        -- files are set and must be modified in the code if needed
    - epoch_eval.py
        -- reads parser output and evaluates loss values
        -- to run: python .\src\epoch_eval.py
        -- files are set and must be modified in the code if needed
    - eval.py
        -- compares gold and predicted sentences
        -- to run: python .\src\eval.py
        -- files are set and must be modified in the code if needed
    - extract-convert-mrs.py
        -- extract class for raw data extraction
        -- no run command is provided
        -- see https://gitlab.cs.uct.ac.za/jbuys/mrs-processing 
        --      for information on how to run this
        --      (only accessible by UCT staff and students)
    - main_predict.py
        -- main method to run predict command
        -- to run: python ./src/main_predict.py
        -- files are set and must be modified in the code if needed
        -- this cannot be run immediately as no trained module is provided in the submission
    - parse-convert-mrs.py 
        -- parse class for raw data extraction
        -- no run command is provided
        -- see https://gitlab.cs.uct.ac.za/jbuys/mrs-processing 
        --      for information on how to run this
        --      (only accessible by UCT staff and students)
    - semantics.py
        -- semantics class for raw data extraction
        -- no run command is provided
        -- see https://gitlab.cs.uct.ac.za/jbuys/mrs-processing 
        --      for information on how to run this
        --      (only accessible by UCT staff and students)
    - setup.py
        -- setup file
        -- no run command
        -- unedited and unused in this adaptation
    - syntax.py 
        -- syntax class for raw data extraction
        -- no run command is provided
        -- see https://gitlab.cs.uct.ac.za/jbuys/mrs-processing 
        --      for information on how to run this
        --      (only accessible by UCT staff and students)

cmds:
    - __init__.py
    - aj_con.py
        -- unedited and unused in this adaptation
    - biaffine_dep.py
        -- unedited and unused in this adaptation
    - cmd.py
        -- main commands class; additional parameters added to parser
    - crf_con.py
        -- unedited and unused in this adaptation
    - crf_dep.py
        -- unedited and unused in this adaptation
    - crf2o_dep.py
        -- unedited and unused in this adaptation
    - vi_con.py
        -- unedited and unused in this adaptation
    - vi_dep.py
        -- unedited and unused in this adaptation
    - vi_sdp.py
        -- unedited and unused in this adaptation

models:
    - __init__.py
        -- unedited and unused in this adaptation
    - const.py
        -- unedited and unused in this adaptation
    - dep.py
        -- unedited and unused in this adaptation
    - model.py
        -- parent class to other models
    - sdp.py
        -- main model used in this project

modules:
    - __init__.py
    - affine.py
        -- biaffine scoring function layer
    - dropout.py
        -- encoding and embedding dropouts
    - gnn.py
        -- unedited and unused in this adaptation
    - lstm.py
        -- LSTM models used for the BiLSTM module developed
    - mlp.py
        -- MLP (FFNN) used for token representations
    - pretrained.py
        -- BERT (used) and ELMo (unused) embeddings and an assisting scalar mix class 
    - transformer.py
        -- unedited and unused in this adaptation

parsers:
    - __init__.py
    - const.py
        -- unedited and unused in this adaptation
    - dep.py
        -- unedited and unused in this adaptation
    - parser.py
        -- parent class to other parsers
    - sdp.py
        -- main parser used

structs:
    - __init__.py
    - chain.py
        -- unedited and unused in this adaptation
    - dist.py
        -- unedited and unused in this adaptation
    - fn.py
        -- additional functions
    - semiring.py
        -- unedited and unused in this adaptation
    - tree.py
        -- unedited and unused in this adaptation
    - vi.py
        -- unedited and unused in this adaptation

utils:
    - __init__.py
    - common.py
        -- common variables
    - config.py
        -- configurations
    - data.py
        -- classes which deal with the data storage and manipulation
    - embed.py
        -- embedding classes
    - field.py
        -- field objects to store the words and characters
    - fn.py
        -- additional functions
    - logging.py
        -- an automatic logging system is outputted when a model is trained, evaluated or predicting
    - maxtree.py
        -- attempted integration of the Chu-Liu Edmonds Algorithm for finding the Maximum Spanning Tree
    - metric.py
        -- varying metrics used for evaluations
    - optim.py 
        -- learning rate manipulation
    - parallel.py
        -- parallelism functions
    - tokenizer.py
        -- tokeniser functions
    - transform.py
        -- main data storage 
    - vocab.py
        -- vocab used in BiLSTM module; BERT module retained Hugging Face's fixed vocab

FILE EXAMPLES (in file_examples folder):
    - BERT_20_epochs_logger.logging
        -- logger example which is outputted with the model
    - config.txt   
        -- empty configuration file used in run command
    - dev_data.conllu 
        -- a snippet of the dev data provided from the LinGo Redwoods Treebank
    - eval_20.txt
        -- precision, recall and evaluation scores for a BERT module with 20 epochs
    - module_prediction.conllu
        -- a module prediction output example
    - new20.job.txt
        -- module output file example (with original tracing statements)
    - test_data.conllu
        -- a snippet of the test data provided from the aforementioned treebank
    - train_data.conllu
        -- a snippet of the training data provided from the aforementioned treebank
    - wsj21a.eds
        -- an originally extracted eds file from the aforementioned treebank

Notes:
    - Depending on your software, you may need to replace "\" with "/".

Thank you!