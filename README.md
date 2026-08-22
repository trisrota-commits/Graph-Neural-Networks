# Transformer Sequence Experiments on the Cora Citation Graph

A research prototype investigating whether Transformer sequence models can learn predictive structure from random walks sampled from the Cora citation graph.

The project evaluates three main tasks:

1. **Past labels → next label**
2. **Past node features → future node features**
3. **Past node features → future node labels**

It also includes a stricter appendix using Cora's official train, validation, and test node masks.

## How the earlier and current experiments are connected

The Cora project grew out of an earlier Transformer experiment on the Zachary Karate Club graph. Both projects represent graph walks as short sequences and use Transformer models to learn patterns between nodes.

The earlier experiment used engineered graph features such as degree, betweenness, closeness, PageRank, and community membership. The Cora experiment uses Cora's native 1,433-dimensional sparse word features and seven publication classes.

## What was missing in the earlier experiment

The earlier sequence-to-sequence model received the features for the complete eight-node walk and predicted the labels for that same walk:

x1, x2, x3, x4, x5, x6, x7, x8    -> y1, y2, y3, y4, y5, y6, y7, y8

Therefore, when predicting y5, the encoder had access to x5, the feature vector belonging to the target node. This does not prove that the model relied on x5, but the information was available.

The model also used teacher forcing. When predicting y5, the decoder received the previous correct labels, such as `y1`–`y4`, but not `y5` itself. Teacher forcing and target-feature exposure are separate issues:


Teacher forcing:
previous correct labels -> current label

Target-feature exposure:
full feature sequence, including x5 -> prediction of y5


The earlier data was split into random walks rather than disjoint node sets. Consequently, the same physical node could appear in both training and test walks. Its accuracy therefore measured performance on new sampled walks, not strict generalisation to unseen nodes.

The earlier work also focused on label prediction and did not separately test future-feature reconstruction. It lacked simple feature baselines, repeated random seeds, and a direct GNN comparison. It tracked validation loss but did not restore the best validation checkpoint before testing.

## What the Cora experiment changes

Each Cora walk is divided into an observed prefix and a future segment:


past features:   x1, x2, x3, x4
future features: x5, x6, x7, x8
future labels:   y5, y6, y7, y8


The future-label task is:

x1, x2, x3, x4 -> y5, y6, y7, y8


The encoder no longer receives `x5`–`x8` while predicting their labels.

Experiment 2 separately evaluates future-feature prediction:

x1, x2, x3, x4 -> x5, x6, x7, x8


Experiments 2 and 3 use teacher forcing during training and autoregressive generation during evaluation. Teacher-forced results use the correct previous target, while generated results require the decoder to use its own previous predictions.

The main Cora results use a walk-level split. The strict appendix generates walks separately inside Cora's official node masks, preventing walks from crossing between the training, validation, and test partitions.

This is a stricter node-partition diagnostic, but it is not fully inductive: it still uses the full graph structure, and test-node features are used as observed inputs during test-time evaluation. The difference between the main and strict results should therefore not be attributed to node overlap alone, since the mask restriction also changes the available walk structure and sample distribution.

## Results from the supplied run

| Task / setting | Main metric |
|---|---:|
| Past labels → next label | 81.13% accuracy |
| Past features → future features | 0.011614 MSE |
| Future-feature cosine similarity | 0.2931 |
| Future-feature nearest-neighbour diagnostic | 46.75% |
| Future labels, teacher forced | 81.07% token accuracy |
| Future labels, generated | 76.76% token accuracy |
| Strict node split, teacher forced | 47.93% token accuracy |
| Strict node split, generated | 22.51% token accuracy |
| Strict node split, generated sequence accuracy | 9.71% |

## Remaining limitations

The main results are walk-level rather than node-disjoint. The study uses one dataset and one random seed. Experiment 2 still needs simple reconstruction baselines, and its nearest-neighbour score searches the full Cora feature matrix, including test nodes.

There is no direct GCN, GraphSAGE, or graph-Transformer comparison. The sequence models also do not restore the best validation checkpoint before testing. Finally, the BERT-style encoders are randomly initialised rather than language-pretrained.
