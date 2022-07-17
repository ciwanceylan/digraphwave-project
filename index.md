# Digraphwave: Scalable Extraction of Structural Node Embeddings via Diffusion on Directed Graphs
### Ciwan Ceylan, Kambiz Ghoorchian and Danica Kragic

Structural node embeddings, vectors capturing local connectivity information for each node in a graph, have many applications in data mining and machine learning, e.g., network alignment and node classification, clustering and anomaly detection.
For the analysis of directed graphs, e.g., transactions graphs, communication networks and social networks, the capability to capture directional information in the structural node embeddings is highly desirable, as is scalability of the embedding extraction method.
Most existing methods are nevertheless only designed for undirected graph.
Therefore, we present Digraphwave -- a scalable algorithm for extracting structural node embeddings on directed graphs.
The Digraphwave embeddings consist of compressed diffusion pattern signatures, which are twice enhanced to increase their discriminate capacity.
By proving a lower bound on the heat contained in the local vicinity of a diffusion initialization node, theoretically justified diffusion timescale values are established, and Digraphwave is left with only two easy-to-interpret hyperparameters: the embedding dimension and a neighbourhood resolution specifier.
In our experiments, the two embedding enhancements, named transposition and aggregation, are shown to lead to a significant increase in macro F1 score for classifying automorphic identities, with Digraphwave outperforming all other structural embedding baselines.
Moreover, Digraphwave either outperforms or matches the performance of all baselines on real graph datasets, displaying a particularly large  performance gain in a network alignment task, while also being scalable to graphs with millions of nodes and edges, running up to 30x faster than a previous diffusion pattern based method and with a fraction of the memory consumption. 



<div style="text-align:center"><img src="figures/pipeline_drawing.svg" width="900"/></div>