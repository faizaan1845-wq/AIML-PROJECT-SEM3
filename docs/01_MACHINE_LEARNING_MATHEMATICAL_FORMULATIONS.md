# 📄 DOCUMENT 1: MACHINE LEARNING MATHEMATICAL FORMULATIONS & ARCHITECTURE GUIDE
**Project:** TrustRate AI — Review Authenticity Engine, True Rating Deflator & Failure Diagnostics  
**Academic Module:** AIML Semester 3 Project  
**Author / Team:** faizaan1845-wq & Team  

---

## 1. Executive Overview & System Architecture
TrustRate AI is an end-to-end Machine Learning system engineered to detect fraudulent/incentivized product reviews, calculate genuine customer rating deflation, and extract hardware failure diagnostics from e-commerce platforms.

### End-to-End Mathematical Pipeline:
```
Raw Customer Review Text 
       │
       ▼ [Formula 1: Sublinear TF-IDF Vectorization]
High-Dimensional Numerical Feature Vector: x ∈ ℝⁿ
       │
       ▼ [Formula 2: Linear Combination & Logit Score: z = wᵀx + b]
Unbounded Linear Activation: z ∈ (-∞, +∞)
       │
       ▼ [Formula 3: Sigmoid Hypothesis Mapping: ŷ = σ(z)]
Calibrated Posterior Probability: ŷ = P(Suspicious = 1 | x) ∈ [0, 1]
       │
       ▼ [Formula 4: Rating Deflation Calculation]
Genuine Authentic Rating: R_authentic & Inflation Delta: ΔR
       │
       ▼ [Formula 5: Aspect Diagnostic Sentiment & Failure Incidence]
Aspect Scorecards, Root-Cause Defect Breakdown, and Pros & Cons
```

---

## 2. Mathematical Formulations

### Formula 1: Feature Representation via Sublinear TF-IDF
* **Where Used in Code:** `train.py` via `TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))`
* **Mathematical Definition:**
  Converts raw text into an $n$-dimensional normalized feature space.

  1. **Sublinear Term Frequency (TF):**
     $$\text{TF}(t, d) = \begin{cases} 1 + \ln(f_{t, d}), & \text{if } f_{t, d} > 0 \\ 0, & \text{otherwise} \end{cases}$$
     *where $f_{t,d}$ is the raw count of term $t$ in review $d$.*
     *Logarithmic dampening ensures that a promotional word repeated 10 times does not exert 10 times more influence than a word appearing once.*

  2. **Smooth Inverse Document Frequency (IDF):**
     $$\text{IDF}(t, D) = \ln\left( \frac{1 + N}{1 + \text{DF}(t)} \right) + 1$$
     *where $N$ is the total corpus size and $\text{DF}(t)$ is the number of reviews containing term $t$.*

  3. **Composite Weight & Euclidean ($L_2$) Normalization:**
     $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
     $$\mathbf{x} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{k=1}^n v_k^2}}$$
* **Why It Is Used:** Review texts vary significantly in word length (from 6 words to 250 words). $L_2$ normalization projects every review vector onto a unit hypersphere $\|\mathbf{x}\| = 1$, eliminating length bias while giving high mathematical weights to distinctive fraud tokens (e.g. `"miracle"`, `"five stars buy immediately"`) versus generic vocabulary (`"the"`, `"product"`).

---

### Formula 2: Linear Hypothesis & Decision Boundary
* **Where Used in Code:** `models/best_model.pkl` & `model_utils.py`
* **Mathematical Definition:**
  Computes the linear decision boundary as the inner product of the learned weight vector $\mathbf{w}$ and the input feature vector $\mathbf{x}$, offset by bias $b$:
  $$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{n} w_j x_j + b$$
* **Why It Is Used:**
  * Each learned parameter $w_j$ acts as a directional dial:
    * Positive weights ($w_j > 0$): Increase suspicion (e.g., promotional buzzwords, excessive exclamation marks).
    * Negative weights ($w_j < 0$): Signal genuine purchase authenticity (e.g., practical usage descriptions, wear-and-tear observations).
  * $b$ is the base bias reflecting the baseline prior distribution of the training data.

---

### Formula 3: Sigmoid Hypothesis Function (Posterior Probability)
* **Where Used in Code:** `model_utils.py` (predicting `is_suspicious`)
* **Mathematical Definition:**
  Maps the unbounded linear score $z \in (-\infty, +\infty)$ into a strictly bounded probability $\hat{y} \in [0, 1]$:
  $$\hat{y} = h_{\mathbf{w}, b}(\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + \exp\left(-\left(\sum_{j=1}^n w_j x_j + b\right)\right)}$$

  **Classification Decision Rule:**
  $$\text{Prediction} = \begin{cases} 1 \text{ (Suspicious / Promotional Review)}, & \text{if } \hat{y} \ge 0.50 \iff z \ge 0 \\ 0 \text{ (Genuine Authentic Review)}, & \text{if } \hat{y} < 0.50 \iff z < 0 \end{cases}$$
* **Why It Is Used:** Hard boolean outputs cannot quantify model confidence. The Sigmoid activation yields a calibrated posterior probability $P(y=1 \mid \mathbf{x})$ that allows threshold tuning and probabilistic auditing.

---

### Formula 4: True Rating Deflation & Rating Inflation Delta
* **Where Used in Code:** `model_utils.py` inside `compute_rating_inflation()`
* **Mathematical Definition:**
  1. **Advertised Store Star Rating ($R_{\text{advertised}}$):**
     Extracted directly and immutably from the live Amazon product page:
     $$R_{\text{advertised}} \in [1.0, 5.0]$$

  2. **Authentic Organic Star Rating ($R_{\text{authentic}}$):**
     Computed by filtering out all reviews predicted as suspicious by the classifier:
     $$G = \left\{ i \in \{1, \dots, m\} \mid \text{is\_suspicious}^{(i)} = 0 \right\}$$
     $$R_{\text{authentic}} = \frac{1}{|G|} \sum_{i \in G} r^{(i)}$$
     *where $r^{(i)}$ is the star rating of the $i$-th customer review.*

  3. **Rating Inflation Delta ($\Delta R$):**
     $$\Delta R = R_{\text{advertised}} - R_{\text{authentic}}$$

  4. **Percentage Inflation:**
     $$\text{Inflation \%} = \left( \frac{\Delta R}{R_{\text{authentic}}} \right) \times 100\%$$
* **Why It Is Used:** Online stores display simple averages that include paid, bot-generated, and promotional 5-star reviews. By eliminating detected fake reviews, this formula exposes the genuine organic rating of the product.

---

### Formula 5: Aspect Diagnostic Sentiment & Failure Incidence Rate
* **Where Used in Code:** `aspect_analyzer.py` inside `extract_aspect_sentiment_and_failures()`
* **Mathematical Definition:**
  Partitions reviews across 6 hardware dimensions (Battery, Build, Audio, Connectivity, Ergonomics, Software).

  1. **Aspect Positive Sentiment Ratio:**
     $$\text{Sentiment Ratio}_k = \left( \frac{\text{Pos}_k}{\max(\text{Pos}_k + \text{Neg}_k, 1)} \right) \times 100\%$$
     *where $\text{Pos}_k$ represents verified organic positive mentions and $\text{Neg}_k$ represents complaint mentions.*

  2. **Hardware Failure Mode Incidence Rate:**
     $$\text{Incidence Rate}_k = \left( \frac{\text{Complaints}_k}{\text{Total Critical Reviews (1-3★)}} \right) \times 100\%$$

  3. **Severity Classification Function:**
     $$\text{Severity Level} = \begin{cases} \text{Critical Failure Risk}, & \text{if Incidence Rate} \ge 40\% \text{ or contains severe safety hazard} \\ \text{High Recurrence}, & \text{if } 20\% \le \text{Incidence Rate} < 40\% \\ \text{Moderate Defect Rate}, & \text{if Incidence Rate} < 20\% \end{cases}$$
* **Why It Is Used:** Converts subjective unstructured text into structured hardware failure categories, telling buyers exactly why a product fails and presenting verified Pros and Cons without duplication.

---

### Formula 6: Model Evaluation & Validation Metrics
* **Where Used in Code:** `AI Model Studio & Evaluation` ([app.py](file:///C:/Users/venka/.gemini/antigravity/scratch/AIML-PROJECT-SEM3/app.py))
* **Mathematical Formulations:**
  Evaluated on held-out test split of 1,100 reviews (Confusion Matrix: $TP = 383, TN = 430, FP = 163, FN = 124$):

  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{383 + 430}{1100} = \mathbf{75.8\%}$$
  $$\text{Precision} = \frac{TP}{TP + FP} = \frac{383}{383 + 163} = \mathbf{70.1\%}$$
  $$\text{Recall} = \frac{TP}{TP + FN} = \frac{383}{383 + 124} = \mathbf{75.5\%}$$
  $$F_1\text{-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{0.727}$$
  $$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt = \mathbf{0.828}$$

---

## 3. Summary of Formulas in the Project

| Formula | Mathematical Concept | Application in Code | Purpose |
| :--- | :--- | :--- | :--- |
| **Sublinear TF-IDF** | Logarithmic Term Frequency & Smooth IDF | `train.py` | Converts text into 5,000+ length-invariant numerical features |
| **Linear Combination** | Matrix Inner Product $z = \mathbf{w}^T\mathbf{x} + b$ | `model_utils.py` | Computes weighted linguistic score for review |
| **Sigmoid Activation** | Logistic Sigmoid Function $\sigma(z)$ | `model_utils.py` | Converts score to probability $P(\text{fake} \mid \mathbf{x}) \in [0, 1]$ |
| **Rating Deflator** | Sample Mean Shift $\Delta R = R_{\text{adv}} - R_{\text{auth}}$ | `model_utils.py` | Calculates true genuine star rating without fake reviews |
| **Aspect Sentiment** | Ratio of Praise vs Complaint | `aspect_analyzer.py` | Diagnoses hardware failures and generates Pros & Cons |
| **Classification Metrics**| Confusion Matrix, $F_1$, ROC-AUC | `app.py` | Validates model performance on held-out test data |
