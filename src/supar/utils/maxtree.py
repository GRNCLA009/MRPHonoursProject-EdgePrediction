# Chu-Liu Edmonds Maximum Spanning Tree 
# Compiled by Claudia Greenberg ; Original developers cited in code
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# This code is produced separately to the MST algorithm in supar.structs.fn
# The code retains the original tracing statements used for understanding
# Integrating this, or the supar.structs.fn code is highly recommended as future work
# This code is left here as proof of attempt, despite the unsuccessful integration
# Used in this adaptation

import numpy
from typing import List, Dict, Set, Tuple

# the following functions were developed by allennlp developers
# Title: AllenNLP: A Deep Semantic Natural Language Processing Platform
# Class: chu_liu_edmonds.py
# Developers: Garder, M., Grus, J., Neumann, M., Tafjord, 
#   O., Dasigi, P., Liu, N., Peters, M., Schmitz, M., Zettlemoyer, L.
# Affiliation: Allen Institute for Artificial Intelligence
# Version: "2.10.0"
# Codebase available at https://github.com/allenai/allennlp
# Class available at https://github.com/allenai/allennlp/blob/main/allennlp/nn/chu_liu_edmonds.py
class MaxTree:
    # the following function initialises the old_input and old_output 
    def olds(length, score_matrix):
        old_input = numpy.zeros([length, length], dtype=numpy.int32)    # 0s
        old_output = numpy.zeros([length, length], dtype=numpy.int32)   # 0s
        representatives: List[Set[int]] = []                            # empty reps

        for node1 in range(length):                                     # for each node
            # original_score_matrix[node1, node1] = 0.0
            score_matrix[node1, node1] = 0.0                            # its val to itself (diag) is 0
            representatives.append({node1})                             # append node to reps

            for node2 in range(node1 + 1, length):                      # for the remaining nodes
                old_input[node1, node2] = node1                         # [0, 1] = 0
                old_output[node1, node2] = node2                        # [0, 1] = 1

                old_input[node2, node1] = node2                         # [1, 0] = 1
                old_output[node2, node1] = node1                        # [1, 0] = 0

        return old_input, old_output, representatives

    def _find_cycle(
        parents: List[int], length: int, current_nodes: List[bool]
    ) -> Tuple[bool, List[int]]:
                                                                    
        added = [False for _ in range(length)]                      # added = false for every node
        added[0] = True                                             # first value of added is true
        cycle = set()                                               # new set is the cycle
        has_cycle = False                                           # currently no cycle
        for i in range(1, length):                                  # for each value (nodes)
            if has_cycle:                                           # break if there's a cycle
                break
            # don't redo nodes we've already
            # visited or aren't considering.
            if added[i] or not current_nodes[i]:                    # no redoing
                continue
            # Initialize a new possible cycle.
            this_cycle = set()                                      # this cycle is a new set
            this_cycle.add(i)                                       # add node to cycle
            added[i] = True                                         # it is known to be added
            has_cycle = True                                        # has cycle is created and set to true
            next_node = i                                           # new node that has just been added
            while parents[next_node] not in this_cycle:             # while its parent is not in the cycle yet
                next_node = parents[next_node]                      # place that as the next node
                # If we see a node we've already processed,
                # we can stop, because the node we are
                # processing would have been in that cycle.
                if added[next_node]:                                # stopping condition - no cycle
                    has_cycle = False
                    break
                added[next_node] = True                             # setting this node to have been added
                this_cycle.add(next_node)                           # adding it to the cycle

            if has_cycle:                                           # there was actually a cycle
                original = next_node                                # new variable: original = current node
                cycle.add(original)                                 # add this node to the cycle set
                next_node = parents[original]                       # next node is now the parent
                while next_node != original:                        # while the next node is not the original node
                    cycle.add(next_node)                            # add the next node
                    next_node = parents[next_node]                  # make next node the next node's parent
                break

        return has_cycle, list(cycle)                               # param1 = true/false; param2 = cycle set

    def chu_liu_edmonds(self,
        length: int,
        score_matrix: numpy.ndarray,
        current_nodes: List[bool],
        final_edges: Dict[int, int],
        old_input: numpy.ndarray,
        old_output: numpy.ndarray,
        representatives: List[Set[int]],
    ):
        """
        Applies the chu-liu-edmonds algorithm recursively
        to a graph with edge weights defined by score_matrix.
        Note that this function operates in place, so variables
        will be modified.
        # Parameters
        length : `int`, required.
            The number of nodes.
        score_matrix : `numpy.ndarray`, required.
            The score matrix representing the scores for pairs
            of nodes.
        current_nodes : `List[bool]`, required.
            The nodes which are representatives in the graph.
            A representative at it's most basic represents a node,
            but as the algorithm progresses, individual nodes will
            represent collapsed cycles in the graph.
        final_edges : `Dict[int, int]`, required.
            An empty dictionary which will be populated with the
            nodes which are connected in the maximum spanning tree.
        old_input : `numpy.ndarray`, required.
        old_output : `numpy.ndarray`, required.
        representatives : `List[Set[int]]`, required.
            A list containing the nodes that a particular node
            is representing at this iteration in the graph.
        # Returns
        Nothing - all variables are modified in place.
        """
        # Set the initial graph to be the greedy best one.
        
        parents = [-1]
        for node1 in range(1, length):                              # [node1 = 1] for each node
            parents.append(0)                                       # [parent of node1] parent's first node is 0
            if current_nodes[node1]:                                # if the indexed node exists
                max_score = score_matrix[0, node1]                  # [between node0 and node1] the max score is the first index of the node's scores
                for node2 in range(1, length):                      # for each node in the rest of the nodes
                    if node2 == node1 or not current_nodes[node2]:  # if they're the same node, or the second node is not in the list
                        continue                                    # ignore them!

                    new_score = score_matrix[node2, node1]          # [between node2 and node1] the new score is the score between node 2 and node 1
                    if new_score > max_score:                       # if the new score beats the old score
                        max_score = new_score                       # [max for node1] the new score replaces the max score
                        parents[node1] = node2                      # [parent of node1] parents of node 

        
        # Check if this solution has a cycle.
        has_cycle, cycle = self._find_cycle(parents, length, current_nodes)
        # If there are no cycles, find all edges and return.
        if not has_cycle:                                           # no cycle scenario!
            final_edges[0] = -1                                     # first edge: 0 -> -1
            for node in range(1, length):                           # for each remaining node
                if not current_nodes[node]:                         # if the node does not exist in the current set, ignore
                    continue
                parent = old_input[parents[node], node]             # the parent is the parent, child value in old input
                # e.g. tensor[0: [1, 2, 3],
                #      1: [4, 5, 6],
                #      2: [7, 8, 9]]
                # if node = 2 and parents[2] = 1, 
                # old_input[1, 2] = 6

                child = old_output[parents[node], node]             # the child is the parent, child value in old output
                final_edges[child] = parent                         # the edge to the child is the parent
            
            return # returns nothing 

        # Otherwise, we have a cycle so we need to remove an edge.
        # From here until the recursive call is the contraction stage of the algorithm.
        cycle_weight = 0.0                                          # cycle weight init
        # Find the weight of the cycle.
        index = 0                                                       # index init
        for node in cycle:                                              # for each node in the cycle
            index += 1                                                  # add 1 to index
            cycle_weight += score_matrix[parents[node], node]           # weight's score increased by parent-child edge score

        # For each node in the graph, find the maximum weight incoming
        # and outgoing edge into the cycle.
        cycle_representative = cycle[0]                                 # cycle rep is the first node in the cycle
        for node in range(length):                                      # for each node
            if not current_nodes[node] or node in cycle:                # node not in current set or its in the cycle, ignore it
                continue

            in_edge_weight = float("-inf")                              # in edge weight is minus infinity
            in_edge = -1                                                # in edge is -1
            out_edge_weight = float("-inf")                             # out edge weigh is minus infinity
            out_edge = -1                                               # out edge is -1

            for node_in_cycle in cycle:                                 # for each node in the cycle
                if score_matrix[node_in_cycle, node] > in_edge_weight:  # if the score in the matrix is larger than the in edge weight (-1)
                    in_edge_weight = score_matrix[node_in_cycle, node]  # the in edge weight is now this weight
                in_edge = node_in_cycle                                 # the in edge is now the node

                # Add the new edge score to the cycle weight
                # and subtract the edge we're considering removing.
                score = (                                                   # the score is  
                    cycle_weight                                            # cycle weight
                    + score_matrix[node, node_in_cycle]                     # + new score
                    - score_matrix[parents[node_in_cycle], node_in_cycle]   # - old score
                )

                if score > out_edge_weight:                                 # if score is greater than out edge weight (-1)
                    out_edge_weight = score                                 # out edge weight is updated - new score
                    out_edge = node_in_cycle                                # out edge is now the node

            score_matrix[cycle_representative, node] = in_edge_weight       # score matrix of cycle rep and node is the in edge weight        
            old_input[cycle_representative, node] = old_input[in_edge, node]# old input of cycle rep and node is the old input of in edge and node
            old_output[cycle_representative, node] = old_output[in_edge, node]# old output of cycle rep and node is the old ourput of in edge and node

            score_matrix[node, cycle_representative] = out_edge_weight      # score matrix of node and cycle rep is the out edge weight
            old_output[node, cycle_representative] = old_output[node, out_edge]# old output of node and cycle rep is the old output of the node and out edge
            old_input[node, cycle_representative] = old_input[node, out_edge]# old input of the node and cycle rep is the old input of the node and out edge

        # For the next recursive iteration, we want to consider the cycle as a
        # single node. Here we collapse the cycle into the first node in the
        # cycle (first node is arbitrary), set all the other nodes not be
        # considered in the next iteration. We also keep track of which
        # representatives we are considering this iteration because we need
        # them below to check if we're done.
        considered_representatives: List[Set[int]] = []
        for i, node_in_cycle in enumerate(cycle):
            considered_representatives.append(set())
            if i > 0:
                # We need to consider at least one
                # node in the cycle, arbitrarily choose
                # the first.
                current_nodes[node_in_cycle] = False

            for node in representatives[node_in_cycle]:
                considered_representatives[i].add(node)
                if i > 0:
                    representatives[cycle_representative].add(node)

        # RECURSIVE STEP #

        self.chu_liu_edmonds(
            length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives
        )

        # Expansion stage.
        # check each node in cycle, if one of its representatives
        # is a key in the final_edges, it is the one we need.
        found = False
        key_node = -1
        for i, node in enumerate(cycle):
            for cycle_rep in considered_representatives[i]:
                if cycle_rep in final_edges:
                    key_node = node
                    found = True
                    break
            if found:
                break

        previous = parents[key_node]
        while previous != key_node:
            child = old_output[parents[previous], previous]
            parent = old_input[parents[previous], previous]
            final_edges[child] = parent
            previous = parents[previous]

# attempted testing below

"""nodes = ['proper_q', 'named', 'compound']

s_edge = [[1.0, 0.8, 0.6], [1.4, 9.0, 2.1], [2.2, 1.2, 1.3]]

old_input, old_output, representatives = olds(3, numpy.array(s_edge))

final_edges = {}

chu_liu_edmonds(3, numpy.array(s_edge), nodes, final_edges,  old_input, old_output, representatives)

print(final_edges)"""