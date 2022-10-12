# Epoch Evaluation Class
# Claudia Greenberg
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# Reads parser output and evaluates loss values
# This is a class to be run independently to the parser code


def epocheval(input_file, arr):

    """
    FILE INPUT:

    Output statements include loss values (which is what we are concerned with)
    
    Loss values come in the form:
    train/eval: edge loss:  0.6931464672088623  label loss:  2.302584648132324
    
    """

    train_edge_loss, train_label_loss, dev_edge_loss, dev_label_loss = [], [], [], []
    epoch = 1
    
    # specify the epochs you want to evaluate
    epoch_group = [2, 3, 4, 5, 10, 15, 20]
    
    # used to regster when a new epoch occurs
    last = "eval:"

    with open(input_file) as f:
        for line in f:
            sections = line.strip().split()
            
            # filter out what we don't want
            if len(sections) == 0 or (sections[0] != "train:" and sections[0] != "eval:"):
                continue
                
            else:
                if sections[0] == "train:":
                    # append values to array
                    if last == "eval:": # new epoch
                        if epoch in epoch_group: # previous epoch should be recorded now
                            teavg = sum(train_edge_loss)/len(train_edge_loss)
                            tlavg = sum(train_label_loss)/len(train_label_loss)
                            deavg = sum(dev_edge_loss)/len(dev_edge_loss)
                            dlavg = sum(dev_label_loss)/len(dev_label_loss)
                            train_edge_loss, train_label_loss, dev_edge_loss, dev_label_loss = [], [], [], [] #reset!
                            arr.append("epoch " + str(epoch) + ":\t" + str(teavg) + "\t" + str(tlavg) + "\t" + str(deavg) + "\t" + str(dlavg))
                    
                        epoch = epoch + 1

                    if epoch in epoch_group: # append!
                        train_edge_loss.append(float(sections[3]))
                        train_label_loss.append(float(sections[6]))
            
                 
                if sections[0] == "eval:" and epoch in epoch_group: # append!
                    dev_edge_loss.append(float(sections[3]))
                    dev_label_loss.append(float(sections[6]))
                
                last = sections[0]

# this is run when the class is run

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
