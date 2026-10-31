# 📑 TRUSTRATE AI: MATHEMATICAL FORMULATIONS SUMMARY

**Project:** TrustRate AI — Review Authenticity Engine, True Rating Deflator & Failure Diagnostics  
**Repository:** [faizaan1845-wq/AIML-PROJECT-SEM3](https://github.com/faizaan1845-wq/AIML-PROJECT-SEM3)  

---

<!-- ======================================================================= -->
<!-- PAGE 1: THE CORE 5 FORMULAS                                            -->
<!-- ======================================================================= -->

# 📄 PAGE 1: THE 5 MAIN FORMULAS (CORE ENGINE)

These 5 formulas represent the primary mathematical backbone connecting Machine Learning with Calculus optimization:

```
[1. TF-IDF] ──▶ [2. Sigmoid Activation] ──▶ [3. Binary Cross-Entropy Loss]
                                                        │
                                                        ▼
[5. Rating Deflator] ◀───────────────────── [4. Gradient Descent Update]
```

---

### Formula 1: Text-to-Number Vectorization (Sublinear TF-IDF)
$$\text{TF-IDF}(t, d) = \Big(1 + \ln(f_{t, d})\Big) \times \left( \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1 \right)$$
* **Why it is used:** Machine learning models cannot process raw words or sentences directly; they require structured numerical matrices.
* **How it works:** It counts word frequency while applying a logarithm to prevent repeated words from dominating, and downweights common words (`"the"`, `"product"`) while giving high weights to deceptive terms (`"miracle"`, `"five stars buy now"`).

---

### Formula 2: Prediction Probability Mapping (Logistic Sigmoid)
$$z = \sum_{j=1}^n w_j x_j + b \quad \implies \quad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Why it is used:** To convert unbounded linear scores $z \in (-\infty, +\infty)$ into a true probability between 0% and 100% indicating if a review is fake.
* **How it works:** It multiplies each word value $x_j$ by its learned weight $w_j$, adds bias $b$, and squashes the result through the S-shaped Sigmoid curve; if $\hat{y} \ge 0.50$, the review is classified as fake.

---

### Formula 3: Error & Mistake Measurement (Binary Cross-Entropy Loss)
$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \Big[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \Big]$$
* **Why it is used:** Provides a continuous, differentiable calculus objective function that quantifies exactly how wrong the AI is across all $m$ training reviews.
* **How it works:** If the true review is fake ($y=1$) but the AI predicts genuine ($\hat{y} \approx 0$), $-\ln(\hat{y})$ imposes a massive mathematical penalty, creating a smooth convex error bowl that can be minimized.

---

### Formula 4: How the AI Learns (Gradient Descent Update via Chain Rule)
$$w_j^{(t+1)} := w_j^{(t)} - \alpha \frac{\partial J}{\partial w_j} = w_j^{(t)} - \alpha \left( \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} \right)$$
* **Why it is used:** This is the core Calculus algorithm that trains the model by adjusting word weights in the direction that minimizes prediction errors.
* **How it works:** It takes the partial derivative of loss via the multivariate Chain Rule $\left(\frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}\right)$, multiplies the error-weighted gradient by learning rate $\alpha$, and steps downhill towards the minimum error point.

---

### Formula 5: True Organic Rating Deflator
$$\Delta R = R_{\text{advertised}} - R_{\text{authentic}} \quad \text{where} \quad R_{\text{authentic}} = \frac{1}{|G|} \sum_{i \in G} r^{(i)}$$
* **Why it is used:** To eliminate deceptive rating inflation caused by sponsored 5-star reviews on e-commerce platforms like Amazon.
* **How it works:** It discards all reviews flagged as fake by the ML model ($G = \{\text{genuine reviews}\}$), calculates the true organic star average $R_{\text{authentic}}$, and computes the inflation delta $\Delta R$ (e.g. 4.5★ advertised vs 3.8★ genuine = **+0.7★ inflation**).

---
\pagebreak
---

<!-- ======================================================================= -->
<!-- PAGE 2: COMPLETE MATHEMATICAL SPECIFICATION                            -->
<!-- ======================================================================= -->

# 📄 PAGE 2: COMPLETE REFERENCE OF ALL FORMULAS USED

This page lists every single mathematical equation used across all modules of TrustRate AI.

---

### Part A: NLP & Feature Extraction Formulations (`train.py`)

1. **Sublinear Term Frequency (TF):**
   $$\text{TF}(t, d) = \begin{cases} 1 + \ln(f_{t, d}), & \text{if } f_{t, d} > 0 \\ 0, & \text{otherwise} \end{cases}$$
2. **Smooth Inverse Document Frequency (IDF):**
   $$\text{IDF}(t, D) = \ln\left( \frac{1 + N}{1 + \text{DF}(t)} \right) + 1$$
3. **Composite TF-IDF Weight:**
   $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
4. **Euclidean ($L_2$) Vector Normalization:**
   $$\mathbf{x} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{k=1}^n v_k^2}}$$

---

### Part B: Classifier Hypothesis & Probability (`model_utils.py`)

5. **Linear Logit Combination:**
   $$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^n w_j x_j + b$$
6. **Sigmoid Activation Function:**
   $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$$
7. **Sigmoid Analytical Derivative (Chain Rule Lemma):**
   $$\frac{d\sigma}{dz} = \sigma(z)\Big(1 - \sigma(z)\Big) = \hat{y}(1 - \hat{y})$$
8. **Binary Decision Boundary:**
   $$\text{Class} = \begin{cases} 1 \text{ (Suspicious / Deceptive Review)}, & \text{if } \hat{y} \ge 0.50 \iff z \ge 0 \\ 0 \text{ (Genuine Organic Review)}, & \text{if } \hat{y} < 0.50 \iff z < 0 \end{cases}$$

---

### Part C: Loss Function & Calculus Optimization (`optimization.py` & `train.py`)

9. **Regularized Binary Cross-Entropy (BCE) Loss:**
   $$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \Big[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \Big] + \frac{\lambda}{2m} \|\mathbf{w}\|_2^2$$
10. **Multivariate Chain Rule Gradient Decomposition:**
    $$\frac{\partial J}{\partial w_j} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$
    $$\frac{\partial J}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}, \quad \frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y}), \quad \frac{\partial z}{\partial w_j} = x_j$$
    $$\text{Product: } \left[\frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}\right] \cdot \Big[\hat{y}(1 - \hat{y})\Big] \cdot x_j = (\hat{y} - y) x_j$$
11. **Analytical Gradient with $L_2$ Regularization:**
    $$\nabla_{\mathbf{w}} J = \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m \Big(\hat{y}^{(i)} - y^{(i)}\Big) x_j^{(i)} + \frac{\lambda}{m} w_j$$
12. **Bias (Intercept) Gradient:**
    $$\nabla_b J = \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m \Big(\hat{y}^{(i)} - y^{(i)}\Big)$$
13. **Gradient Descent Parameter Update Rule:**
    $$w_j^{(t+1)} := w_j^{(t)} - \alpha \frac{\partial J}{\partial w_j}$$
    $$b^{(t+1)} := b^{(t)} - \alpha \frac{\partial J}{\partial b}$$
14. **Vectorized Matrix Gradient Formulation:**
    $$\mathbf{e} = \sigma(\mathbf{X}\mathbf{w} + b) - \mathbf{y}$$
    $$\mathbf{w} := \mathbf{w} - \alpha \left( \frac{1}{m}\mathbf{X}^T\mathbf{e} + \frac{\lambda}{m}\mathbf{w} \right)$$
    $$b := b - \alpha \left( \frac{1}{m}\sum_{i=1}^m e_i \right)$$

---

### Part D: Rating Deflation & Diagnostics (`model_utils.py` & `aspect_analyzer.py`)

15. **Advertised Store Star Rating:**
    $$R_{\text{advertised}} \in [1.0, 5.0] \quad (\text{scraped directly from Amazon link})$$
16. **Genuine Organic Star Rating:**
    $$R_{\text{authentic}} = \frac{1}{|G|} \sum_{i \in G} r^{(i)} \quad \text{where } G = \{i \mid \text{is\_suspicious}^{(i)} = 0\}$$
17. **Rating Inflation Delta:**
    $$\Delta R = R_{\text{advertised}} - R_{\text{authentic}}$$
18. **Percentage Rating Inflation:**
    $$\text{Inflation \%} = \left( \frac{\Delta R}{R_{\text{authentic}}} \right) \times 100\%$$
19. **Aspect Positive Sentiment Ratio:**
    $$\text{Sentiment Ratio}_k = \left( \frac{\text{Pos}_k}{\max(\text{Pos}_k + \text{Neg}_k, 1)} \right) \times 100\%$$
20. **Hardware Failure Mode Incidence Rate:**
    $$\text{Incidence Rate}_k = \left( \frac{\text{Complaints}_k}{\text{Total Critical Reviews (1-3★)}} \right) \times 100\%$$

---

### Part E: Classification Evaluation Metrics (`app.py`)

21. **Classification Accuracy:**
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \mathbf{75.8\%}$$
22. **Precision (Suspicious Class):**
    $$\text{Precision} = \frac{TP}{TP + FP} = \mathbf{70.1\%}$$
23. **Recall / Sensitivity:**
    $$\text{Recall} = \frac{TP}{TP + FN} = \mathbf{75.5\%}$$
24. **Harmonic Mean ($F_1$-Score):**
    $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{0.727}$$
25. **Area Under ROC Curve (ROC-AUC):**
    $$\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt = \mathbf{0.828}$$
