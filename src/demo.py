# Demo Class
# Claudia Greenberg
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# This is a class to be run independently to the parser code

from supar.parsers import parser
from epoch_eval import epocheval
from eval import calc_final_f1, evaluate
from tabulate import tabulate

# STEP 1: Train model
print("Step One: Train the Model")
print("Training takes several hours on the CHPC.")

# STEP 2: Evaluate the different epochs
print("\nStep Two (Optional): Evaluate the F1 Scores on the Validation Data and Loss Values at Different Epoch Values")
print("Reading in a module output file (consisting of loss values computed during training) (new20.job.txt).\nWriting to an evaluation file (epoch_vals.txt).")

main_arr = []

# specify files to read
epochs = ['file_examples/new20.job.txt']
for ep in epochs:
    epocheval(ep, main_arr)

# specify file to write to
with open('file_examples/epoch_vals.txt', 'w') as f:
    for line in main_arr:
        f.write(line)
        f.write("\n")
    f.close()

print("\nThe loss values are depicted below and in the paper:\n")

loss = [["Training Edge Loss", 0.0395, 0.0183, 0.0132, 0.0105, 0.0050, 0.0030, 0.0018, 0.0011, 0.0007, 0.0004, 0.0002],
["Validation Edge Loss", 0.2229, 0.0799, 0.0508, 0.0356, 0.0120, 0.0065, 0.0035, 0.0020, 0.0010, 0.0004, 0.0002],
["Training Label Loss", 0.0172, 0.0110, 0.0093, 0.0085, 0.0078, 0.0079, 0.0083, 0.0090, 0.0104, 0.0111, 0.0124],
["Validation Label Loss", 0.0857, 0.0578, 0.0593, 0.0533, 0.0544, 0.0606, 0.0596, 0.0680, 0.0720, 0.0716, 0.0737]]

print (tabulate(loss, headers=["Epochs", 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]))

print("\nThe F1 scores on the validation data was calculated using several trained models with varying epochs.")
print("The F1 scores are depicted below:\n")

f1 = [["Precision", 93.5423, 94.7980, 95.6875, 95.9832, 96.6694, 97.2165, 97.4682, 97.4411, 97.4513, 97.5553, 97.7618],
["Recall", 91.6390, 93.1172, 94.2084, 94.5656, 95.1165, 95.7330, 95.8974, 95.9710, 96.0274, 95.9888, 96.1609],
["F1", 92.3935, 93.7999, 94.8104, 95.1615, 95.8113, 96.4169, 96.6249, 96.6506, 96.6592, 96.7208, 96.9081]]

print (tabulate(f1, headers=["Epochs", 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]))

print("\nThe chosen number of epochs, according to the F1 scores and loss values is 20.")

# STEP 3: Predict a set of sentences with the model
print("\nStep Three: Predict a Set of Sentences with the Module")
print("Using an edge prediction module (BERT encoder, Biaffine network, Cross Entropy Loss)\ntrained with 20 epochs.")
print("Reading in a set of test data (test_data.conllu)\nWriting to a prediction file (module_prediction.conllu)")
print("The module parameters are depicted below:\n")
parser = parser.Parser.load('./newnewmodel_20')
#parser = parser.Parser.load('./newlstmmodel_20_5e03') 
dataset = parser.predict('file_examples/test_data.conllu', pred = 'file_examples/module_prediction.conllu', lang='en', prob=True, verbose=False)

#STEP 3: Evaluate prediction
print("\nStep Four: Evaluate the Prediction by Comparing to the Gold Sentences")
print("Reading in predictions and gold sentences from above\nWriting values to file (final_evals.txt)\nOutputting evaluation metric results below:\n")

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

res = calc_final_f1(output_file)

print("Precision: ", res[0])
print("Recall: ", res[1])
print("F1 Score: ", res[2])