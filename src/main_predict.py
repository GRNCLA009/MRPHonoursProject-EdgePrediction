# Prediction Class
# Claudia Greenberg (lightly based off of a program by Chase Ting Cong)
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# Main method to run predict command
# This is a class to be run independently to the parser code

from supar.parsers import parser

# specify the module
parser = parser.Parser.load('./newnewmodel_20') 

dataset = parser.predict('file_examples/test_data.conllu', pred = 'file_examples/module_prediction.conllu', lang='en', prob=True, verbose=False)