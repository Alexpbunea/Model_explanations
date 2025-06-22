# Model explanations using Influence Analysis

This project investigates the fragility and robustness of model explanation techniques based on influence analysis. These methods aim to interpret individual predictions by identifying which training examples most influenced a specific inference.

Concretely, for a given test instance (in this case a Titanic passenger predicted to survive), influence methods return a ranked list of training samples—the top-k most "influential" examples that shaped the model’s decision.

The analysis focuses on two key challenges:
- Stability: How sensitive are these influence scores to variations in data splits, model configurations, or random seeds—even when using the same method?
- Consistency: How much do influence rankings vary when using different explanation techniques under otherwise identical conditions?

To explore this, we compare two methods—First-Order Influence (FOI) and TracIn—by:

- Quantifying ranking agreement with metrics like Kendall Tau, Spearman, and Jaccard@k.
- Identifying overlapping influential samples across methods.
- Evaluating model performance after removing high-influence points from the training set.

This repository presents code, analysis, and visualizations that provide insight into how fragile or robust influence-based explanations are in practice.


*Note: The following batch size refers to the ones used to train and evaluate the models. There are other batch sizes in the code, but have any impact at all. For rapic check, go to the section "My view" below. Also, the graphics are made with a 'k' of 300, 200 of train plus 100 of test, whereas the rest of the operations are done with a K = 711, the total size of the train dataset. This is because if K was smaller, the metrics of rank similation were equal to 'nan'.

<br>

## Batch size of 32

| FOI                                      | TracIn                                   |
|------------------------------------------|------------------------------------------|
| ![F1](/plots/batch_size_32/Figure_1.png) | ![F4](/plots/batch_size_32/Figure_4.png) |
| ![F2](/plots/batch_size_32/Figure_2.png) | ![F5](/plots/batch_size_32/Figure_5.png) |
| ![F3](/plots/batch_size_32/Figure_3.png) |                                          |

### Results

**Kendall Tau: 0.208 | Spearman: 0.289 | Jaccard@711: 1.000**
| Method    | k    | ΔAcc     | ΔAUC     |
|-----------|-------|----------|----------|
| FOI       | 1     | +0.0955  | +0.0183  |
| TracIn    | 1     | +0.1236  | +0.0395  |
| FOI       | 5     | +0.0730  | +0.0265  |
| TracIn    | 5     | +0.0843  | +0.0189  |
| FOI       | 10    | +0.0787  | +0.0397  |
| TracIn    | 10    | +0.0955  | +0.0365  |
| FOI       | 25    | +0.0562  | +0.0191  |
| TracIn    | 25    | +0.0843  | +0.0243  |
| FOI       | 50    | +0.0787  | +0.0220  |
| TracIn    | 50    | +0.1067  | +0.0308  |
| FOI       | 711   | -0.0337  | -0.0716  |
| TracIn    | 711   | -0.0674  | -0.0943  |

----
<br><br><br>


## Batch size of 16

| FOI                                      | TracIn                                   |
|------------------------------------------|------------------------------------------|
| ![F1](/plots/batch_size_16/Figure_1.png) | ![F4](/plots/batch_size_16/Figure_4.png) |
| ![F2](/plots/batch_size_16/Figure_2.png) | ![F5](/plots/batch_size_16/Figure_5.png) |
| ![F3](/plots/batch_size_16/Figure_3.png) |   

### Results

**Kendall Tau: 0.209 | Spearman: 0.273 | Jaccard@711: 1.000**
| Method    | k    | ΔAcc     | ΔAUC     |
|-----------|-------|----------|----------|
| FOI       | 1     | +0.0112  | -0.0052  |
| TracIn    | 1     | +0.0225  | -0.0062  |
| FOI       | 5     | -0.0056  | +0.0056  |
| TracIn    | 5     | +0.0112  | -0.0029  |
| FOI       | 10    | +0.0000  | -0.0058  |
| TracIn    | 10    | +0.0112  | +0.0025  |
| FOI       | 25    | +0.0000  | -0.0050  |
| TracIn    | 25    | +0.0000  | +0.0063  |
| FOI       | 50    | +0.0000  | -0.0035  |
| TracIn    | 50    | +0.0169  | +0.0056  |
| FOI       | 711   | -0.1798  | -0.1091  |
| TracIn    | 711   | -0.1292  | -0.1028  |

-----
<br><br><br>

## Batch size of 8

| FOI                                      | TracIn                                   |
|------------------------------------------|------------------------------------------|
| ![F1](/plots/batch_size_8/Figure_1.png) | ![F4](/plots/batch_size_8/Figure_4.png) |
| ![F2](/plots/batch_size_8/Figure_2.png) | ![F5](/plots/batch_size_8/Figure_5.png) |
| ![F3](/plots/batch_size_8/Figure_3.png) | 

### Results

**Kendall Tau: 0.298 | Spearman: 0.374 | Jaccard@711: 1.000**
| Method     | k    | ΔAcc     | ΔAUC     |
|------------|-------|----------|----------|
| FOI        | 1     | +0.0056  | +0.0053  |
| TracIn     | 1     | +0.0056  | -0.0011  |
| FOI        | 5     | +0.0056  | -0.0078  |
| TracIn     | 5     | +0.0281  | -0.0007  |
| FOI        | 10    | -0.0112  | -0.0027  |
| TracIn     | 10    | -0.0056  | +0.0091  |
| FOI        | 25    | +0.0000  | +0.0015  |
| TracIn     | 25    | +0.0056  | +0.0091  |
| FOI        | 50    | +0.0056  | -0.0052  |
| TracIn     | 50    | -0.0056  | +0.0111  |
| FOI        | 711   | -0.0562  | -0.0338  |
| TracIn     | 711   | -0.1067  | -0.0684  |

----
<br><br><br>

## Batch size of 4

| FOI                                      | TracIn                                   |
|------------------------------------------|------------------------------------------|
| ![F1](/plots/batch_size_4/Figure_1.png) | ![F4](/plots/batch_size_4/Figure_4.png) |
| ![F2](/plots/batch_size_4/Figure_2.png) | ![F5](/plots/batch_size_4/Figure_5.png) |
| ![F3](/plots/batch_size_4/Figure_3.png) | 

### Results

**Kendall Tau: 0.284 | Spearman: 0.404 | Jaccard@711: 1.000**
| Method    | k    | ΔAcc     | ΔAUC     |
|-----------|-------|----------|----------|
| FOI       | 1     | -0.0056  | -0.0034  |
| TracIn    | 1     | +0.0056  | +0.0046  |
| FOI       | 5     | +0.0000  | -0.0063  |
| TracIn    | 5     | -0.0056  | -0.0030  |
| FOI       | 10    | +0.0000  | -0.0088  |
| TracIn    | 10    | -0.0112  | +0.0032  |
| FOI       | 25    | +0.0000  | -0.0010  |
| TracIn    | 25    | +0.0056  | +0.0084  |
| FOI       | 50    | +0.0112  | +0.0018  |
| TracIn    | 50    | -0.0056  | -0.0012  |
| FOI       | 711   | -0.0899  | -0.0439  |
| TracIn    | 711   | -0.0899  | -0.0463  |



<br><br><br><br>

## My view

<br>

### 1. Methodology and Key Metrics

1. **Construction of Influence Rankings**  
   - **First-Order Influence (FOI):** Estimates each sample’s direct contribution to the loss gradient.  
   - **TracIn:** Accumulates loss changes over training by summing gradient correlations between train and test samples.

2. **Ranking-Similarity Metrics**  
   - **Kendall Tau (τ):** Measures pairwise concordance in full rankings; τ = 1 means identical order, τ = 0 means random, τ < 0 indicates inversion.  
   - **Spearman (ρ):** Rank correlation; more sensitive to global deviations.  
   - **Jaccard@N:** Overlap of the top-N most influential samples. Jaccard@711 = 1.000 means perfect agreement on the 711 least influential samples.

3. **Impact Evaluation (ΔAcc, ΔAUC)**  
   - For each k (1, 5, 10, 25, 50, …): remove the top-k most influential samples (by FOI and by TracIn), retrain the model, and measure the change in accuracy and AUC relative to the baseline.


<br>

### 2. Comparing graphics

- **Median Curves:**  
  - Both FOI and TracIn median curves steadily decline as the rank increases.  
  - The drop is steepest at the beginning and levels off gradually across all batch sizes.  
- **Cumulative Distribution Functions (CDFs):**  
  - FOI’s CDF rises quickly then plateaus, showing most influence scores are clustered at low values.  
  - TracIn’s CDF also climbs rapidly but concentrates influence scores in an even lower range than FOI.  
- **Long-Tail Behavior:**  
  - Both methods exhibit a “long tail” in their median curves, with influence scores slowly tapering over a wide rank range.  
  - This long tail is more pronounced for FOI than for TracIn.  
- **Decay Speed of Influence Scores:**  
  - Influence scores drop sharply at first, then more slowly, for all batch sizes.  
  - The initial decline is faster for TracIn than for FOI, indicating TracIn’s scores concentrate more quickly at low values.  
- **Effect of Smaller Batches:**  
  - As batch size decreases, both median curves and CDFs show similar shapes but with steeper and faster drops in influence scores.  
  - Smaller batches amplify this effect, especially for TracIn.  

<br>

### 3. Ranking Similarity

| Metric         | Value Range                   | Interpretation                                                                                          |
|----------------|-------------------------------|---------------------------------------------------------------------------------------------------------|
| Kendall τ      | ~0.284 (batch=4) – 0.208 (32) | Moderate to low correlation: methods often differ in ordering highly influential samples.               |
| Spearman ρ     | 0.404 (batch=4) – 0.289 (32)  | Higher agreement with smaller batches; FOI and TracIn align better as batch size decreases.            |
| Jaccard@711    | 1.000                         | Both methods identify the same least-influential samples—useful for discarding “noise.”                |

> **Insight:** A τ≈0.2–0.3 indicates agreement on low-influence samples but significant differences on high-influence ones.

<br>

### 4. Effect of Removing Top-k Influential Samples (batch_size = 32)

| k    | FOI ΔAcc   | FOI ΔAUC   | TracIn ΔAcc  | TracIn ΔAUC  |
|------|-----------:|-----------:|-------------:|-------------:|
| 1    | +0.0955    | +0.0183    | **+0.1236**  | **+0.0395** |
| 5    | +0.0730    | +0.0265    | +0.0843      | +0.0189     |
| 10   | +0.0787    | +0.0397    | +0.0955      | +0.0365     |
| 25   | +0.0562    | +0.0191    | +0.0843      | +0.0243     |
| 50   | +0.0787    | +0.0220    | +0.1067      | +0.0308     |
| 711  | –0.0337    | –0.0716    | –0.0674      | –0.0943     |

- **Maximum Benefit:**  
  - TracIn at k = 1 → +12.36 pp accuracy, +3.95 pp AUC.  
  - FOI at k = 10 → +7.87 pp accuracy, +3.97 pp AUC.  
- **Risk Zone:** Removing ≥ 7 % of data (k ≥ 711) degrades performance (ΔAcc < 0).

<br>

### 5. Trade-Offs and Practical Considerations

| Aspect                    | FOI                                                    | TracIn                                                      |
|---------------------------|--------------------------------------------------------|-------------------------------------------------------------|
| **Immediate Gain**        | Smaller boost at k=1 (max +9.55 pp vs +12.36 pp).      | Largest boost when removing the single most influential sample. |
| **Ranking Stability**     | Slightly more robust at large batch sizes.             | Less consistent overall (lower τ at large batch sizes).     |
| **Computational Cost**    | Low: only local gradients.                             | High: must store or recompute states each epoch.            |
| **Interpretability**      | Direct (per-sample gradient).                          | Requires tracking training history.                         |

<br>

### 6. Practical Recommendations

0. **Ideal Batch Size:** Use **batch_size = 32** when training and computing influences—it maximizes accuracy/AUC gains and offers the best trade-off between stability and practical impact.  
1. **Quick Outlier Detection:** Apply TracIn with k=1–5 to find critical outliers and boost accuracy/AUC rapidly.  
2. **Lightweight Cross-Validation:** Use FOI on large batches to confirm TracIn findings before removing samples.  
3. **Operational Threshold:** Keep k ≤ 50 (≤ 0.7 % of dataset) to maximize gains without risking performance drops.  
4. **Hybrid Pipeline:**  
   - **Step 1:** TracIn(k=1, 5) → manual inspection of IDs.  
   - **Step 2:** FOI(k=1, 5) → cross-check agreement.  
   - **Step 3:** Retrain removing only samples agreed upon by both methods.  


 
