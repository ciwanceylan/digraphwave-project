# rolxrefex

Clone the repo and then install into your environment using

```bash
pip install ./rolx-refex
```

Usage example

```python
import networkx as nx
import rolxrefex.refex as refex

graph = nx.karate_club_graph()
adj = nx.adjacency_matrix(graph).T
features = refex.refex(adj, use_weights=False)
print(features.shape)
```