# Knowledge Graphs as Signal Aggregators for Macroeconomic Predictive Models

### Knowledge Graphs (KGs)
Knowledge graphs structure information as triples: (Entity, Relationship, Entity) that describe causal or associative relations between entities. In the industry, they are used to integrate qualitative domain knowledge into a more structured form for information retrieval purposes. 

### Graph Signal Processing (GSP)
Graph Signal Processing has historically been used to employ background of signal generating mechanisms to define a graph as a signal domain. This allows for certain analytical techniques which can incorporate signal similarity and spatial locality.

### The Problem in Finance
Portfolio Managers (PMs), especially those in discretionary or macro buy-side roles often rely on predictive models using conventional, structured time-series data e.g. economic indicators, market prices, yield curves, etc. These models are not relied on solely for trading– at times, portfolio managers may override model decisions especially in times of sudden market volatility. 

In a sudden market event, the correlations between the feature (input) variables will shift greatly. E.g., during a crisis or a panic sell-off, almost all securities will move together downwards. It may be helpful for PMs to have a mechanism to quickly integrate their views on causal relationships without retraining the a predictive model, as it is often the case that these causal changes are often intermittent and do not last long enough to warrant a full retraining of the model.

Modelling covariances between multiple responses is often an uncharted problem in finance, especially when in most modelling cases, the correlations between features are assumed to be constant. [^1].

### How KGs Might Help (High-Level)
KGs allow portfolio managers to input their views into a predictive model, by highlighting causal links between market entities and macroeconomic indicators. These causal links are a way of updating the correlations between the features in a predictive model in a specific moment in time. KGs also allow portfolio managers to plug-and-play temporary or novel data streams into a predictive model without retraining the model weights. 

### Methodology Assumptions
- Assume that we have structured data feeds (time series) that are already linked to a conventional predictive model.
- KG nodes must explicitly symbolise or link directly to these time series data feeds. 



### The Role of Correlations in Prediction
Correlations capture how variables move together. Explicitly modeling correlations is crucial because predictions rarely depend solely on isolated factors—often, predictive power arises from relationships between multiple variables:

- **Econometrics Example:** Predicting returns of different stocks typically benefits from understanding correlations (e.g., tech stocks often move together). If correlations suddenly change due to geopolitical events, predictions not explicitly handling these shifts become inaccurate.
- **Macroeconomics Example:** Predicting inflation involves multiple indicators (unemployment rate, GDP growth, commodity prices). Explicitly modeling correlations captures dynamic interdependencies—such as the inverse correlation between unemployment and inflation (Phillips curve)—enhancing prediction accuracy as economic conditions evolve.

### Importance of Input-Varying Correlations
Typically, correlations are assumed constant, but real-world correlations change over time, significantly affecting prediction accuracy during critical periods. Modelling these dependent covariances remains largely uncharted, especially in financial forecasting.


### Original Linear Predictor
The initial linear predictive model is defined as:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3
$$

where $x_1, x_2, x_3$ are original data features and $w_1, w_2, w_3$ are their corresponding weights.


### Mathematical Methodology: Graph Signal Processing

Fully worked-out expression for filtered signals $x' = (x_1', x_2', x_3')^\top$ using a simple low-pass filter $H(L) = I - \alpha L$ on a 3-node graph:

#### Define the Graph and Laplacian

- **Adjacency Matrix** $A$:

$$
A = \begin{pmatrix}
0 & c_{12} & c_{13}\\
c_{21} & 0 & c_{23}\\
c_{31} & c_{32} & 0
\end{pmatrix}, \quad c_{ij} = c_{ji}
$$

- **Degree Matrix** $D$:

$$
D_{ii} = \sum_{j=1}^{3} A_{ij}, \quad D_{11} = c_{12} + c_{13}, \quad D_{22} = c_{21} + c_{23}, \quad D_{33} = c_{31} + c_{32}
$$

- **Graph Laplacian** $L$:

$$
L = D - A
$$

$$
H(L) = I - \alpha L = I - \alpha (D - A) = I - \alpha D + \alpha A
$$

Set $B = I + \alpha (A - D)$, thus:

$$
 x' = B x = \bigl(I + \alpha (A - D)\bigr) x
$$

#### Filtered Features Explicitly

Let $x = (x_1, x_2, x_3)^\top$. In coordinates, for $i \in \{1,2,3\}$:

$$
 x'_i = x_i + \alpha \Bigl(\sum_{j} c_{ij} x_j - D_{ii} x_i \Bigr).
$$

Expanding explicitly for each node:

- **Node 1** ($x_1'$): $x'_1 = \bigl(1 - \alpha (c_{12} + c_{13})\bigr) x_1 + \alpha c_{12} x_2 + \alpha c_{13} x_3$
- **Node 2** ($x_2'$): $x'_2 = \bigl(1 - \alpha (c_{21} + c_{23})\bigr) x_2 + \alpha c_{21} x_1 + \alpha c_{23} x_3$
- **Node 3** ($x_3'$): $x'_3 = \bigl(1 - \alpha (c_{31} + c_{32})\bigr) x_3 + \alpha c_{31} x_1 + \alpha c_{32} x_2$

#### Updated Linear Model

After filtering, predictions become:

$$
\hat{y}_{new}=w_1 x_1'+w_2 x_2'+w_3 x_3'
$$

#### Use of KGs:
A portfolio manager can tweak values of $c_{ij}$ in the adjacency matrix $A$ to reflect their views on correlations between features. For example, if $x_1$ and $x_2$ are expected to be more correlated due to a recent economic event, the portfolio manager can increase $c_{12}$ and $c_{21}$ in the adjacency matrix. 

For this example, assume all other correlations to be zero. We have the following adjacency matrix:
$$
A = \begin{pmatrix}
0 & 0.8 & 0\\
0.8 & 0 & 0\\
0 & 0 & 0
\end{pmatrix}
$$

For simplicity let $\alpha = 1$. 

Then the new features become:
$$
\begin{aligned}
x'_1 &= \bigl(1 - 0.8\bigr) x_1 + 0.8 x_2 + 0 x_3\\
     &= 0.2 x_1 + 0.8 x_2\\
x'_2 &= \bigl(1 - 0.8\bigr) x_2 + 0.8 x_1 + 0 x_3\\
        &= 0.2 x_2 + 0.8 x_1\\
x'_3 &= \bigl(1 - 0\bigr) x_3 + 0 x_1 + 0 x_2\\
        &= x_3
\end{aligned}
$$

This means that we 'blend' the features $x_1$ and $x_2$ together in the prediction model. In order for this to work, we need to assume that $x_1, x_2, x_3$ are all normalised to the same scale.


#### Hot-Swapping New Data Streams
Let's say we have a new data stream $x_4$ that we think is correlated with $x_3$, and will be so temporarily. We can add this new data stream to the model by adding a new node to the graph and updating the adjacency matrix $A$ accordingly.
$$
A = \begin{pmatrix}
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0.5\\
0 & 0 & 0.5 & 0
\end{pmatrix}
$$
Recall:

$$
 x'_i = x_i + \alpha \Bigl(\sum_{j} c_{ij} x_j - D_{ii} x_i \Bigr).
$$


Then, the new features become:
$$
\begin{aligned}
x'_1 &= x_1 \\
x'_2 &= x_2\\
x'_3 &= \bigl(1 - 0.5\bigr) x_3 + 0 x_1 + 0 x_2 + 0.5 x_4\\
        &= 0.5 x_3 + 0.5 x_4\\
\end{aligned}
$$

We still only have three features in the model, but we have added a new data stream $x_4$ to the model without retraining the weights.





[^1]: Wilson, A. G. and Ghahramani, Z. (n.d.) Modelling Input Varying Correlations between Multiple Responses. Unpublished working paper, University of Cambridge. Accessed 2025. https://mlg.eng.cam.ac.uk/pub/pdf/WilGha12a.pdf


