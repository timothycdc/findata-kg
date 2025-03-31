
### A Brief Mathematical Background on Graph Signal Processing

> A graph example for this section is a network of weather stations, where each station is a node and the connections between them are edges. 

A graph is defined by $G = (V, E)$, where
- $V$ (vertices) is the set of nodes (e.g. weather stations, sensors), and
- $E$ (edges) is the set of connections between nodes. (e.g. distances between weather stations)

A **graph signal** is a function that assigns a value to each node at a particular instant in time. For example, imagine recording the temperature at all weather stations simultaneously. This measurement is a graph signal:
$$
f : V \to \mathbb{R}
$$

Example: we have $f$ take a reading at each node in $V$, and output a scalar value (the temperature) at that node. 

In this case, each node is assigned a single scalar value (its temperature). Thus, a signal can be seen as a snapshot of some property across the entire network.

For our example, assume we have $n$ connected weather stations, thus $n$ nodes. For the whole graph, we can represent the signal as a vector:
$$
f = [f_1, f_2, \dots, f_n]^T \in \mathbb{R}^n
$$

The topology of the network is described by matrices that capture the connectivity among nodes:

- **Adjacency Matrix ($W$):**  
  An $n \times n$ matrix where $W_{ij}$ reflects the weight of the edge between nodes $i$ and $j$. A higher value indicates a stronger connection.

- **Degree Matrix ($D$):**  
  A diagonal matrix where each entry is the sum of the weights of the edges connected to a node:
  $$
  D_{ii} = \sum_j W_{ij}
  $$

The **graph Laplacian** combines this information to reflect how signals diffuse over the network. We define it as:
- **Unnormalised Laplacian:**
$$
L = D - W
$$
- **Normalised (symmetric) Laplacian:**
$$
L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}
$$

The Laplacian characterises the graph’s structure. Its eigenvectors form the graph Fourier basis, and the eigenvalues act as frequencies– which are can be applied to clustering and filtering actions on graph signals.

#### Weather Stations

Consider a network of three weather stations:
- **Vertices:** $V = \{1, 2, 3\}$
- **Edges:** $E = \{(1,2), (2,3)\}$, where each edge has weight 1 (a similar score for geographic proximity/terrain similarity)

Graphically, the network is:
```
(1) -- (2) -- (3)
```

At a given moment, suppose we record the following temperatures:
- Station 1: $f_1 = 15^\circ$ C
- Station 2: $f_2 = 16^\circ$ C
- Station 3: $f_3 = 15.5^\circ$ C

The temperature readings form the graph signal:
$$
f = \begin{bmatrix} 15 \\ 16 \\ 15.5 \end{bmatrix}
$$
This signal is simply a snapshot— each station (node) has one corresponding measurement.

Now, let’s look at the matrices that describe the network:

- **Adjacency Matrix ($W$):**
$$
W = \begin{bmatrix}
0 & 1 & 0\\[6pt]
1 & 0 & 1\\[6pt]
0 & 1 & 0
\end{bmatrix}
$$

- **Degree Matrix ($D$):**
$$
D = \begin{bmatrix}
1 & 0 & 0\\[6pt]
0 & 2 & 0\\[6pt]
0 & 0 & 1
\end{bmatrix}
$$

- **Graph Laplacian ($L$):**
$$
L = D - W = \begin{bmatrix}
1 & -1 & 0\\[6pt]
-1 & 2 & -1\\[6pt]
0 & -1 & 1
\end{bmatrix}
$$
