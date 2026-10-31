# 📄 DOCUMENT 2: CALCULUS INTEGRATION & GRADIENT DESCENT OPTIMIZATION
**Project:** TrustRate AI — Real-World Application of Optimization using Gradient Descent  
**Academic Module:** CIE-3 Mathematics / Calculus Project & Viva Component  
**Component Marks:** Part 1: Project (10 Marks) — Report: 6 Marks, Viva: 4 Marks  

---

## 1. Problem Description & Real-World Application
In e-commerce, millions of deceptive reviews artificially inflate product ratings, misleading consumers into purchasing defective goods. 

To solve this, our project trains a binary classifier to automatically identify fake/promotional reviews ($y = 1$) versus genuine purchases ($y = 0$).

### How Calculus is Applied:
A Machine Learning model does not magically know how to classify text. It starts with completely random weights and makes severe errors.
* **Calculus provides the objective function:** We define a continuous, differentiable cost function called **Binary Cross-Entropy Loss** $J(\mathbf{w}, b)$ that quantifies the classification error across all customer reviews.
* **Calculus computes the direction of error reduction:** Using the **multivariate Chain Rule**, we calculate the **gradient vector** $\nabla J(\mathbf{w}, b)$, which mathematically points in the direction where error increases fastest.
* **Calculus drives the parameter update (Gradient Descent):** By iteratively stepping in the exact opposite direction of the gradient ($-\nabla J$), we systematically minimize classification error until the model reaches optimal convergence and 75.8% accuracy.

---

## 2. Mathematical Formulation

### 2.1 The Hypothesis Function
The model computes a linear combination of $n$ text features ($z$), transformed through the non-linear Sigmoid activation function:
$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{n} w_j x_j + b$$
$$h_{\mathbf{w}, b}(\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$$

#### Lemma: Analytical Derivative of the Sigmoid Function
Using the Power Rule and Chain Rule:
$$\frac{d\sigma}{dz} = \frac{d}{dz}\left( (1 + e^{-z})^{-1} \right) = -1 \cdot (1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z})$$
$$= -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

Splitting the numerator into factors:
$$\frac{d\sigma}{dz} = \left( \frac{1}{1 + e^{-z}} \right) \cdot \left( \frac{e^{-z}}{1 + e^{-z}} \right) = \left( \frac{1}{1 + e^{-z}} \right) \cdot \left( \frac{1 + e^{-z} - 1}{1 + e^{-z}} \right)$$
$$= \sigma(z) \cdot \left( 1 - \frac{1}{1 + e^{-z}} \right) = \sigma(z)(1 - \sigma(z))$$

$$\therefore \frac{d\sigma}{dz} = h(1 - h)$$

*Calculus Significance:* The derivative is expressed purely as a function of its own output, requiring zero transcendental exponential re-evaluations during gradient calculation.

---

### 2.2 The Cost Function: Regularized Binary Cross-Entropy (BCE)
Given $m$ labeled customer reviews with labels $y^{(i)} \in \{0, 1\}$, the loss across the entire training dataset is:
$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln\left(h^{(i)}\right) + (1 - y^{(i)}) \ln\left(1 - h^{(i)}\right) \right] + \frac{\lambda}{2m} \|\mathbf{w}\|_2^2$$

where:
* $h^{(i)} = \sigma\left(\mathbf{w}^T \mathbf{x}^{(i)} + b\right)$ is the predicted probability that review $i$ is deceptive.
* $\frac{\lambda}{2m} \|\mathbf{w}\|_2^2 = \frac{\lambda}{2m} \sum_{j=1}^n w_j^2$ is the $L_2$ Ridge regularization term to ensure numerical stability.

---

### 2.3 Analytical Gradient Derivation via the Multivariate Chain Rule
To minimize $J(\mathbf{w}, b)$, we must compute the vector of partial derivatives:
$$\nabla J(\mathbf{w}) = \left[ \frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial w_2}, \dots, \frac{\partial J}{\partial w_n} \right]^T \quad \text{and} \quad \frac{\partial J}{\partial b}$$

Let the per-sample loss be:
$$L^{(i)} = -\left[ y^{(i)} \ln(h^{(i)}) + (1 - y^{(i)}) \ln(1 - h^{(i)}) \right]$$

Applying the multivariate Chain Rule across the 3 composite function layers ($L \to h \to z \to w_j$):
$$\frac{\partial L^{(i)}}{\partial w_j} = \frac{\partial L^{(i)}}{\partial h^{(i)}} \cdot \frac{\partial h^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_j}$$

#### Step 1: Differentiate Cost $L^{(i)}$ with respect to Hypothesis $h^{(i)}$
$$\frac{\partial L^{(i)}}{\partial h^{(i)}} = -\left[ \frac{y^{(i)}}{h^{(i)}} - \frac{1 - y^{(i)}}{1 - h^{(i)}} \right] = -\left[ \frac{y^{(i)}(1 - h^{(i)}) - (1 - y^{(i)})h^{(i)}}{h^{(i)}(1 - h^{(i)})} \right]$$
$$= -\left[ \frac{y^{(i)} - y^{(i)}h^{(i)} - h^{(i)} + y^{(i)}h^{(i)}}{h^{(i)}(1 - h^{(i)})} \right] = -\left[ \frac{y^{(i)} - h^{(i)}}{h^{(i)}(1 - h^{(i)})} \right] = \frac{h^{(i)} - y^{(i)}}{h^{(i)}(1 - h^{(i)})}$$

#### Step 2: Differentiate Hypothesis $h^{(i)}$ with respect to Linear Score $z^{(i)}$
From our Sigmoid lemma:
$$\frac{\partial h^{(i)}}{\partial z^{(i)}} = h^{(i)}(1 - h^{(i)})$$

#### Step 3: Differentiate Linear Score $z^{(i)}$ with respect to Feature Weight $w_j$
$$\frac{\partial z^{(i)}}{\partial w_j} = \frac{\partial}{\partial w_j}\left( \sum_{k=1}^n w_k x_k^{(i)} + b \right) = x_j^{(i)}$$

#### Step 4: Multiply the Factors (The Exact Cancellation)
$$\frac{\partial L^{(i)}}{\partial w_j} = \left[ \frac{h^{(i)} - y^{(i)}}{h^{(i)}(1 - h^{(i)})} \right] \cdot \left[ h^{(i)}(1 - h^{(i)}) \right] \cdot x_j^{(i)}$$

$$\frac{\partial L^{(i)}}{\partial w_j} = (h^{(i)} - y^{(i)}) x_j^{(i)}$$

*Calculus Insight:* The non-linear denominator $h(1 - h)$ cancels the Sigmoid derivative exactly, leaving an intuitive linear error-weighted gradient!

#### Step 5: Average across all $m$ samples and add Regularizer Gradient
$$\frac{\partial}{\partial w_j} \left( \frac{\lambda}{2m} \sum_{k=1}^n w_k^2 \right) = \frac{\lambda}{m} w_j$$

$$\therefore \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m \left( h^{(i)} - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j$$

For the bias term $b$ (since $\frac{\partial z^{(i)}}{\partial b} = 1$ and the bias is unregularized):
$$\therefore \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m \left( h^{(i)} - y^{(i)} \right)$$

---

## 3. The Gradient Descent Optimization Algorithm

### 3.1 Parameter Update Formulation
Using the calculated gradient vector, weights and bias are updated iteratively in the direction of steepest descent:

$$w_j^{(t+1)} := w_j^{(t)} - \alpha \frac{\partial J}{\partial w_j}$$
$$b^{(t+1)} := b^{(t)} - \alpha \frac{\partial J}{\partial b}$$

*where:*
* $\alpha > 0$ is the **learning rate** (step size).
* $t \in \{1, 2, \dots, T\}$ denotes the iteration / epoch number.

### 3.2 Vectorized Matrix Form
In our NumPy implementation (`optimization.py`), the entire gradient vector is evaluated simultaneously using BLAS matrix operations:
$$\mathbf{e} = \sigma(\mathbf{X}\mathbf{w} + b) - \mathbf{y} \quad \in \mathbb{R}^m$$
$$\nabla_{\mathbf{w}} J = \frac{1}{m} \mathbf{X}^T \mathbf{e} + \frac{\lambda}{m} \mathbf{w} \quad \in \mathbb{R}^n$$
$$\nabla_b J = \frac{1}{m} \sum_{i=1}^m e_i \quad \in \mathbb{R}$$
$$\mathbf{w} := \mathbf{w} - \alpha \nabla_{\mathbf{w}} J$$
$$b := b - \alpha \nabla_b J$$

---

## 4. Geometric Interpretation & Convexity Proof
Because Binary Cross-Entropy is strictly convex with respect to $\mathbf{w}$:
$$\nabla^2 J(\mathbf{w}) = \frac{1}{m} \mathbf{X}^T \mathbf{S} \mathbf{X} + \frac{\lambda}{m} \mathbf{I} \succ 0$$
where $\mathbf{S} = \text{diag}(h^{(i)}(1 - h^{(i)})) > 0$.

* **Convexity Guarantee:** The Hessian matrix $\nabla^2 J$ is strictly positive definite.
* **Global Minimum:** The cost function has **zero local minima** and **zero flat saddle points**; it is a single convex bowl.
* **Convergence Guarantee:** For any learning rate satisfying $0 < \alpha < \frac{2}{\lambda_{\max}(\mathbf{X}^T\mathbf{X})}$, Gradient Descent is guaranteed to converge strictly to the global minimum $\mathbf{w}^*$.

---

## 5. Learning Rate ($\alpha$) Sensitivity Analysis

| Learning Rate ($\alpha$) | Mathematical Behavior | Real-World Convergence Result |
| :--- | :--- | :--- |
| **Too Small ($\alpha = 0.01$)** | Steps $\Delta \mathbf{w} = -\alpha \nabla J$ are minuscule. | Requires >5,000 iterations; converges too slowly. |
| **Optimal ($\alpha = 0.15$)** | Balanced step size. Smooth exponential decay in loss. | Loss drops from $0.6931 \to 0.0284$ (95.9% reduction) in 50 epochs. |
| **Too Large ($\alpha \ge 0.80$)** | Overshoots the bottom of the parabolic cost bowl. | Loss oscillates wildly, fails to settle, or diverges ($J \to \infty$). |

---

## 6. Implementation Variants (`optimization.py`)

1. **Batch Gradient Descent (BGD):**
   * Computes the exact gradient using all $m$ training samples per step.
   * Trajectory: Perfectly smooth, deterministic descent.
2. **Mini-Batch Gradient Descent (MBGD):**
   * Partitions data into chunks of size $B = 32$ or $64$.
   * Trajectory: Fast, GPU/BLAS vectorized, optimal for large real-world corpora.
3. **Stochastic Gradient Descent (SGD):**
   * Updates weights after every single sample ($m = 1$).
   * Trajectory: Noisy zig-zag that escapes shallow local traps.
4. **Gradient Ascent (Maximum Likelihood):**
   * Replaces cost minimization with log-likelihood maximization:
     $$\mathbf{w}^{(t+1)} := \mathbf{w}^{(t)} + \alpha \nabla \ell(\mathbf{w})$$

---

## 7. Blackboard Viva Walkthrough (For 4/4 Viva Marks)

If the teacher asks: **"Derive how the error term $(h - y)$ appears using the Chain Rule on the board"**:

1. **Write the 3 Chain Rule components:**
   $$\frac{\partial J}{\partial w_j} = \frac{\partial J}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$
2. **Differentiate Loss:**
   $$\frac{\partial J}{\partial h} = -\frac{y}{h} + \frac{1 - y}{1 - h} = \frac{h - y}{h(1 - h)}$$
3. **Differentiate Sigmoid:**
   $$\frac{\partial h}{\partial z} = h(1 - h)$$
4. **Multiply together:**
   $$\left( \frac{h - y}{h(1 - h)} \right) \cdot \Big( h(1 - h) \Big) \cdot x_j = (h - y)x_j$$
5. **State the conclusion to the teacher:**
   *"The term $h(1 - h)$ in the denominator cancels the derivative of the Sigmoid activation exactly, leaving the intuitive linear error-weighted gradient $(h - y)x_j$."*
