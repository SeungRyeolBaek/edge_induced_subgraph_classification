# Edge Induced Subgraph Classification
This is an official implementation of the paper "Edge-Induced Subgraph Representation Learning"
You can access to the paper with the following link : https://dl.acm.org/doi/10.1145/3770855.3818085 

Run shell commands written in command.txt to reproduce our experiment results

#### Requirements
Tested combination: Python 3.9.6 + [PyTorch 1.9.0](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 1.7.2](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Other required python libraries include: numpy, scikit-learn, pyyaml etc.

#### Dataset
All files about dataset needed to run the experiment are already prepared in edge_dataset directory.

#### Experiment
shell commands to run the experiment code is written in command.txt. We provide the experiment code to reproduce our result of GNNseg, GNNplain, GLASS, SSNP, ISNN

Run the commands written right below the comment '#Ablation Study' to reproduce the result of ablation study. 
test_edge_model_union.py is the code about reproducing the result of Variant 1. 
test_edge_model_node_induced.py is the code about reproducint the result of Variant 2.

Run the commands written right below the comment '# Alpha sweep' to reproduce the result of 'Effect of the Mixing Coefficient 𝛼'.
Run the command written right below the comment '# Plot Alpha sweep' to plot the result. You may see the plot the reproduce the Figure 5

