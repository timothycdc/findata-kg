# Incorporating Domain Knowledge via Graph-Regularised Factor Forecasting

## 1. Introduction

Factor models are foundational in quantitative portfolio management, allowing asset returns to be decomposed into systematic (factor-driven) and idiosyncratic components. Traditionally, latent factors and their loadings are estimated from asset-level data. However, in practice, portfolio managers (PMs) often have informed views about how macroeconomic conditions influence markets. Rather than encoding these insights directly into asset-level relationships, we propose a framework where macroeconomic signals are used to predict factor returns, and PM views are injected via a knowledge graph (KG) over macro variables.

In our setting, asset returns $X \in \mathbb{R}^{n \times t}$ (for $n$ assets over $t$ time periods) are modelled as:
$$
X \approx L F
$$
where $F \in \mathbb{R}^{k \times t}$ represents the returns of $k$ observable market factors (e.g. interest rates, inflation shocks, industrial production), and $L \in \mathbb{R}^{n \times k}$ is the matrix of asset exposures to those factors. The key challenge is to forecast $F$, the factor return matrix, using $k$ macroeconomic data feeds.

To encode PM beliefs, we place a structured prior on the **covariance of the factor returns**, $\Sigma_F = \mathrm{Cov}(F)$. We assume PM views can be expressed via a knowledge graph $G$ over macroeconomic drivers, which then guides the geometry of $F$ through a graph-regularised forecasting model.

---

## 2. Problem Formulation

Let $M \in \mathbb{R}^{k \times t}$ be a matrix of $k$ macroeconomic indicators over $t$ months (e.g. inflation, GDP growth, unemployment, Fed funds rate). We aim to model the factor returns $F$ as a function of $M$, i.e.
$$
F = g(M) + \varepsilon,
$$
where $g(\cdot)$ is a predictive model (e.g. linear regression, kernel model, or neural network), and $\varepsilon$ is a noise term.

The asset return matrix is then reconstructed as:
$$
\hat{X} = L \hat{F} = L g(M)
$$

Our key innovation is to guide the structure of $F$ using a **graph regularisation penalty** that enforces smoothness or correlation structure based on a KG defined over macroeconomic variables.

---

## 3. Graph Regularisation for Factor Forecasting

We construct a knowledge graph $G = (V, E)$ where each node represents a macroeconomic variable (a row of $M$), and edges represent discretionary or structural dependencies among variables (e.g. "rate hikes reduce inflation", "GDP affects employment").

Let $A \in \mathbb{R}^{k \times k}$ be the adjacency matrix of the KG. Define:
- **Degree matrix:**
  $$
  D_{ii} = \sum_{j=1}^k A_{ij}, \quad D \in \mathbb{R}^{k \times k}
  $$
- **Graph Laplacian:**
  $$
  L_G = D - A
  $$

We impose the following regularisation on the factor forecast matrix $\hat{F} = g(M)$:
$$
\min_g \; \|F - g(M)\|_F^2 + \lambda\, \mathrm{Tr}(g(M)^\top L_G g(M))
$$

The term $\mathrm{Tr}(g(M)^\top L_G g(M))$ penalises large differences in predicted returns between macro variables connected in the KG. Effectively, it acts as a structural prior on the **covariance of factor returns**:
- If $A_{ij}$ is high, then $f_i$ and $f_j$ are expected to be correlated.
- The regularisation encourages $\mathrm{Cov}(f_i, f_j)$ to align with the KG structure.

---

## 4. Example: A 3×3 Macro Graph

Suppose we model three macroeconomic variables $x_1 =$ inflation, $x_2 =$ unemployment, $x_3 =$ interest rate. The PM believes that inflation is closely tied to both unemployment and rates, but the latter two are conditionally independent.

### 4.1 Adjacency Matrix $A$
$$
A = \begin{pmatrix}
0 & 1 & 1 \\
1 & 0 & 0 \\
1 & 0 & 0 \\
\end{pmatrix}
$$

### 4.2 Degree Matrix $D$
$$
D = \begin{pmatrix}
2 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{pmatrix}
$$

### 4.3 Laplacian $L_G$
$$
L_G = D - A = \begin{pmatrix}
2 & -1 & -1 \\
-1 & 1 & 0 \\
-1 & 0 & 1 \\
\end{pmatrix}
$$

This structure enforces smoothness in the predicted returns of the inflation, unemployment, and rate factors.

---

## 5. Tuning the Graph Regularisation Parameter $\lambda$

To ensure the KG meaningfully influences the forecast, we recommend a relative-scale heuristic:

1. Fit a baseline model $g_0$ with $\lambda = 0$
2. Compute the error and regularisation energy:
   $$
   E_0 = \|F - g_0(M)\|_F^2, \quad G_0 = \mathrm{Tr}(g_0(M)^\top L_G g_0(M))
   $$
3. Choose:
   $$
   \lambda = \alpha \cdot \frac{E_0}{G_0}, \quad \alpha \in [0.1, 1]
   $$

Cross-validation across a grid of $\lambda$ values is advised.

---

## 6. Conclusion

This macro-informed framework repositions the factor model as a two-stage system: regress asset returns on latent or interpretable factors, then forecast factor returns using macroeconomic signals. By imposing a knowledge graph prior over the covariance of macro factors, PMs can encode structural relationships directly into the predictive model. This approach bridges the gap between asset-level modelling and discretionary macro views, allowing a PM to express beliefs without hand-tuning asset-by-asset exposures. It also cleanly separates cross-sectional structure (via $L$) from temporal forecasting (via $F = g(M)$), improving interpretability and modularity.

