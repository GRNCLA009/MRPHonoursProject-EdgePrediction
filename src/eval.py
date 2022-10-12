# Evaluation Class
# Claudia Greenberg (lightly based off of a program by Chase Ting Chong)
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# Compares gold and predicted sentences
# This is a class to be run independently to the parser code

def edge_from_conllu(edge): # reading in edges
    tgt_edge = edge.split("|")
    
    tgt = tgt_edge[0]

    src_label = tgt_edge[1].split(":")

    src = src_label[0]

    label = src_label[1]
    
    return src + " " + label + " " + tgt

def evaluate(gold, predicted):
    if predicted is None:
        return
        
    gold_edges = set()
    pre_edges = set()

    temp_edges = []
    for row in gold:
        units = row.strip().split("\t")
        temp_e = units[4]

        if(temp_e != "_"): 
            es = temp_e.split("|") # here are the edges
            for e in es:
                temp_edges.append(units[0] + "|" + e)

    # for each edge in expected, add it
    for edge in temp_edges:
        gold_edges.add(edge_from_conllu(edge))
    
    temp_edges = []
    for row in predicted:
        units = row.strip().split("\t")
        temp_e = units[4]

        if(temp_e != "_"): 
            es = temp_e.split("|") # here are the edges
            for e in es:
                temp_edges.append(units[0] + "|" + e)

    # for each edge in predicted, add it
    for edge in temp_edges:
        pre_edges.add(edge_from_conllu(edge))

    tp, fp = 0, 0
    
    # get edge positives
    while(len(pre_edges) > 0):
        edge = pre_edges.pop() # take first predicted edge in set
        if edge in gold_edges: # if you find the predicted edge in the gold edges
            tp += 1
            gold_edges.remove(edge)
        else:
            fp += 1
    
    # edges remaining in the set are false negatives
    fn = len(gold_edges)

    # edge precision
    if tp + fp == 0:
        precision = 1
    else:
        precision = tp / (tp + fp)

    # edge recall
    if (tp + fn) == 0: 
        recall = 1
    else:
        recall = tp / (tp + fn)
    
    # edge f1
    if (precision + recall) == 0:
        f1 = 0    
    else:
        f1 = 2 * (precision * recall)/(precision + recall)
    
    return {'precision': precision,
            'recall': recall,
            'f1 score': f1}

def calc_final_f1(file): # calculate final F1 score
    precision_arr = []
    recall_arr = []
    f1_arr = []
    
    with open(file) as src:
        for line in src:
            sections = line.strip().split(" ")
            if(len(sections) < 7):
                continue

            precision = sections[1]
            recall = sections[3]
            f1 = sections[6]
            
            precision_arr.append(float(precision))
            recall_arr.append(float(recall))
            f1_arr.append(float(f1))
    
    avgp = sum(precision_arr)/len(precision_arr)
    avgr = sum(recall_arr)/len(recall_arr)
    avgf = sum(f1_arr)/len(f1_arr)

    return (avgp, avgr, avgf)

# this is run when the class is run

main_array = []
output_file = "file_examples/final_eval.txt"

with open("file_examples/test_data.conllu", "r") as gold:
    with open("file_examples/module_prediction.conllu", "r") as predicted:
        
        # read in graphs
        # gold
        gold_graphs = []
        graph = []
        for line in gold: # for each line in input file, separate into graphs
            if(line == "\n"):
                gold_graphs.append(graph)
                graph = [] # reset for new graph
            
            elif(line.split("\t")[1] == "!"):
                continue 

            else:
                graph.append(line)

        # predicted
        predicted_graphs = []
        for line in predicted: # for each line in input file, separate into graphs
            
            if(line == "\n"):
                predicted_graphs.append(graph)
                graph = [] # reset for new graph
            
            else:
                graph.append(line)

        # send each to evaluate
        if(len(gold_graphs) == len(predicted_graphs)):
            for index in range(len(gold_graphs)):
                main_array.append(evaluate(gold_graphs[index], predicted_graphs[index]))

        else:
            print("Problem with input files. Inconsistent number of graphs.")

        with open(output_file, 'w') as f:
            
            for array in main_array:
                for val in array:
                    f.write(val + " " + str(array[val]) + " ")
                f.write("\n")
            f.write("\n")
                     
        #print("Done")

#res = calc_final_f1(output_file)

#print("Precision: ", res[0])
#print("Recall: ", res[1])
#print("F1 Score: ", res[2])