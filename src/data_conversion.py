# Data Conversion Class
# Claudia Greenberg
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# Converts original datasets (eds/json files) to appropriate form 
# This is a class to be run independently to the parser code

"""
OUTPUT:

! Tokens separated by spaces

1. ID:
    Token counter, starting at 1
2. LABEL:
    Node label
3. START: - needed
    Start token of the span
4. END - needed
    First token after the span

All sentences are separated by a newline. 
The final sentence is followed by two newlines 
    to indicate the end of the file.
"""

import json, os

def raw_data_conversion(type, paths, output_file):
    
    #array of json_arrays
    main_array = []
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # loop through paths
    for path in paths:
        
        # original used:
        #file_name = "data/extracted/" + type + "/" + path

        file_name = path

        print(file_name) # printed to track progress
        with open(file_name) as src:
            for line in src: # for each sentence in the file
                    
                # 1: Initialise
                json_array = []
                
                json_load = (json.loads(line))

                first_line = ["!"]

                first_sentence = ""

                # 2: loop through TOKENS to append to first line of sentence
                for token in json_load['tokens']:
                    first_sentence = first_sentence + token['form'] + "\t" 
                
                first_line.append(first_sentence)
                json_array.append(first_line)

                # 3: loop through NODES to append to json line array
                for node in json_load['nodes']:

                    # 1. ID
                    id = node['id'] + 1 #start from 1 for each sentence

                    # 2. LABEL
                    label = node['label']

                    # 3. START
                    start = node['anchors'][0]['from']

                    # 4. END
                    end = node['anchors'][0]['end'] + 1

                    # 5. EDGES - placeholder
                    edges = "_"


                    json_array.append([id, label, start, end, edges])                    
                
                # 4: loop through EDGES  
                for edge in json_load['edges']:
                    
                    source_node = edge['source'] # this is the head
                    target_node = edge['target'] # this is the id of the node - it is situated in one of the rows
                    label = edge['label'] # this is the relation

                    if(len(json_array[target_node]) > 4):
                        if(json_array[target_node][4] == "_"): # it is currently empty, just add the edge regularly
                            json_array[target_node][4] = str(source_node) + ":" + str(label)
                        else: # there is already one there, add a "|"
                            json_array[target_node][4] = json_array[target_node][4] + "|" + str(source_node) + ":" + str(label) 

                main_array.append(json_array)

    # append to file
    with open(output_file, 'a') as f:
        for array in main_array:
            for line in array: # for each word
                # for each value in array, append to string
                json_line = ""
                for index in range(len(line)):
                    json_line = json_line + "\t" + str(line[index])
                    if(index == len(line) - 1): #last index gets /n
                        json_line = json_line + "\n"
                f.write(json_line)
            f.write("\n")
                     
    print("Done")


# this is run when the class is run

print("Starting...")
print("Note: This need only be run once.")
raw_data_conversion("train", ["file_examples/wsj21a.eds"], "file_examples/train.conllu")

"""
ORIGINAL FILES USED:

raw_data_conversion("train", os.listdir("data/extracted/train"), "data/train.conllu")
print("Next up: dev")
raw_data_conversion("dev", os.listdir("data/extracted/dev"), "data/dev.conllu")
print("Last up: test")
raw_data_conversion("test", os.listdir("data/extracted/test"), "data/test.conllu")
"""