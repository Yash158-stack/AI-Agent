# sample_data.py
"""
Pre-packaged 1-Click Study Deck: Machine Learning Fundamentals
Provides instant onboarding experience without requiring manual document uploads.
"""
import os
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from quiz_parser import parse_quiz_output

SAMPLE_DECK_FILENAME = "Machine_Learning_Fundamentals.txt"

SAMPLE_DOCUMENT_TEXT = """# Machine Learning Fundamentals: Core Principles & Architectures

## Module 1: Taxonomy of Machine Learning
Machine Learning (ML) is broadly classified into three primary paradigms:
1. Supervised Learning: The algorithm learns an approximation function f: X -> Y mapping feature vectors X to ground-truth labels Y using labeled training pairs. Typical tasks include regression (continuous outputs, e.g., predicting housing prices) and classification (discrete categories, e.g., spam detection).
2. Unsupervised Learning: The algorithm discovers latent structures, clusters, or probability densities within unlabeled feature sets X. Popular algorithms include K-Means clustering, Principal Component Analysis (PCA), and Gaussian Mixture Models (GMMs).
3. Reinforcement Learning (RL): An agent interacts with a dynamic environment via a Markov Decision Process (MDP), taking actions to maximize cumulative expected rewards over time via exploration and exploitation.

## Module 2: The Bias-Variance Tradeoff & Regularization
Generalization error on unseen test data is mathematically decomposed into three components:
Total Error = Bias^2 + Variance + Irreducible Noise
- Bias represents the error introduced by approximating a complex real-world relationship with an overly simplistic model. High bias leads to underfitting.
- Variance represents model sensitivity to small fluctuations in the training set. High variance leads to overfitting, where the model memorizes noise.
- Regularization techniques penalize model complexity:
  * L1 Regularization (Lasso): Adds the sum of absolute weight values (lambda * sum|w|) to the loss, promoting parameter sparsity and feature selection.
  * L2 Regularization (Ridge): Adds the sum of squared weights (lambda * sum(w^2)), shrinking coefficients smoothly towards zero without setting them strictly to zero.
  * Dropout: Randomly deactivates neurons during forward passes with probability p, preventing co-adaptation of features.

## Module 3: Neural Networks & Optimization
Modern Artificial Neural Networks (ANNs) consist of input, hidden, and output layers parameterized by weights W and biases b.
- Forward Propagation: z = W*x + b, followed by non-linear activation functions such as ReLU (max(0, z)), GELU, or Softmax for multi-class probability distributions.
- Loss Function: Cross-Entropy Loss for classification and Mean Squared Error (MSE) for continuous regression.
- Backpropagation: Computes gradients of the loss with respect to all network parameters via the multivariable chain rule.
- Gradient Descent & Optimizers:
  * Stochastic Gradient Descent (SGD): Updates parameters using mini-batches.
  * Adam (Adaptive Moment Estimation): Computes adaptive learning rates by maintaining exponential moving averages of both past gradients (first moment) and squared gradients (second moment).

## Module 4: Evaluation Metrics & Diagnostic Curves
Evaluating predictive performance requires metrics resilient to class imbalance:
- Confusion Matrix: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).
- Precision = TP / (TP + FP): Proportion of predicted positives that were actually positive. Critical in spam filters.
- Recall (Sensitivity) = TP / (TP + FN): Proportion of actual positives correctly identified. Critical in cancer screening.
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall): The harmonic mean of precision and recall.
- ROC-AUC: Area under the Receiver Operating Characteristic curve, measuring the True Positive Rate against the False Positive Rate across all discrimination thresholds.
"""

SAMPLE_SUMMARY = """### Executive Summary: Machine Learning Fundamentals

Machine Learning is defined across three fundamental paradigms: **Supervised Learning** (predicting known labels via regression and classification), **Unsupervised Learning** (discovering latent patterns via clustering and dimensionality reduction), and **Reinforcement Learning** (policy optimization via environmental rewards).

A central challenge in empirical risk minimization is the **Bias-Variance Tradeoff**. High bias produces underfitting, whereas excessive model capacity leads to high variance and overfitting. Practitioners mitigate overfitting using **L1 (Lasso) and L2 (Ridge) regularization**, early stopping, and **Dropout**.

Deep learning architectures rely on multi-layer perceptrons trained via **backpropagation** and the multivariable chain rule. Modern optimizers like **Adam** leverage first and second gradient moments for robust convergence. Finally, model diagnostics on imbalanced distributions prioritize **Precision, Recall, and the harmonic F1-Score** over naive classification accuracy.
"""

SAMPLE_NOTES = """# 📚 Comprehensive Study Notes: Machine Learning Fundamentals

---

### 1. High-Level Paradigm Comparison

| Paradigm | Input Data | Objective | Key Algorithms |
| :--- | :--- | :--- | :--- |
| **Supervised** | Labeled $(X, Y)$ | Learn mapping $f: X \\to Y$ | Linear Regression, Logistic Regression, Random Forests, SVMs |
| **Unsupervised** | Unlabeled $X$ | Discover latent structure | K-Means, DBSCAN, PCA, Autoencoders |
| **Reinforcement** | States, Actions, Rewards | Maximize cumulative return | Q-Learning, PPO, Deep Q-Networks (DQN) |

---

### 2. Generalization & The Bias-Variance Tradeoff

$$\\text{Expected Error} = \\text{Bias}^2 + \\text{Variance} + \\sigma^2_{\\text{irreducible}}$$

* **High Bias (Underfitting)**:
  * Symptoms: Poor training accuracy, poor validation accuracy.
  * Remedy: Increase model capacity, add polynomial features, decrease regularization parameter $\\lambda$.
* **High Variance (Overfitting)**:
  * Symptoms: Very high training accuracy, poor validation accuracy.
  * Remedy: Collect more training samples, apply L1/L2 weight penalties, apply Dropout ($p=0.2 - 0.5$), perform cross-validation.

---

### 3. Regularization Mechanics

* **L1 Regularization (Lasso)**:
  $$\\mathcal{L}_{\\text{Lasso}} = \\mathcal{L}_0 + \\lambda \\sum_{j=1}^{d} |w_j|$$
  * *Property*: Produces sparse weight vectors; functions as automatic feature selector.
* **L2 Regularization (Ridge)**:
  $$\\mathcal{L}_{\\text{Ridge}} = \\mathcal{L}_0 + \\lambda \\sum_{j=1}^{d} w_j^2$$
  * *Property*: Damps large parameter magnitudes, highly effective against multicollinearity.

---

### 4. Neural Network Optimization

1. **Forward Pass**:
   $$z = W x + b, \\quad a = \\text{ReLU}(z) = \\max(0, z)$$
2. **Backpropagation**:
   $$\\frac{\\partial \\mathcal{L}}{\\partial W} = \\frac{\\partial \\mathcal{L}}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z} \\cdot \\frac{\\partial z}{\\partial W}$$
3. **Adam Optimizer**:
   Combines momentum (exponential average of gradients) with RMSProp (scaling by root mean square of gradients) to achieve stable step sizes.

---

### 5. Classification Metrics Reference

* **Precision**: $\\frac{TP}{TP + FP}$ (Minimizes false alarms)
* **Recall (Sensitivity)**: $\\frac{TP}{TP + FN}$ (Minimizes missed detections)
* **F1-Score**: $\\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$
* **ROC-AUC**: Threshold-invariant ranking quality metric.
"""

SAMPLE_QUIZ_RAW = """### 1. 5 SHORT Important Questions

1. What is the fundamental difference between supervised and unsupervised learning?
Answer: Supervised learning uses labeled dataset pairs (X, Y) to learn a predictive mapping function, whereas unsupervised learning discovers latent patterns or groupings directly from unlabeled features X.

2. What is the mathematical decomposition of generalization error in machine learning?
Answer: Total Expected Error = Bias^2 + Variance + Irreducible Noise.

3. Why does L1 regularization (Lasso) lead to sparse feature weights?
Answer: L1 has a diamond-shaped constraint boundary with sharp corners at coordinate axes, causing gradient trajectories to hit zero exactly, eliminating redundant features.

4. What is the harmonic mean formula for the F1-Score?
Answer: F1-Score = 2 * (Precision * Recall) / (Precision + Recall).

5. What two gradient moments does the Adam optimizer track to adapt learning rates?
Answer: The first moment (exponential moving average of gradients) and the second moment (uncentered variance of squared gradients).

---

### 2. 5 LONG Descriptive Questions

1. Analyze the Bias-Variance tradeoff. How do model complexity, training dataset scale, and regularization techniques influence both components?
- Thoroughly define bias and variance in terms of model capacity and underfitting/overfitting.
- Explain the role of irreducible noise.
- Detail how regularization (L1/L2/Dropout) shifts the optimal operating point on the error curve.

2. Compare and contrast L1 (Lasso) and L2 (Ridge) regularization mathematically and practically.
- Formulate both cost functions with penalty parameters.
- Contrast their geometric constraint spaces and sparsity outcomes.
- Discuss appropriate use cases for each method in high-dimensional datasets.

3. Detail the backpropagation algorithm in deep neural networks. How does the multivariable chain rule propagate error gradients?
- Derive the gradient of the loss with respect to hidden layer weights.
- Discuss activation functions (Sigmoid vs ReLU) and the vanishing/exploding gradient phenomenon.
- Explain why non-linear activations are strictly required for universal function approximation.

4. Evaluate classification performance under severe class imbalance (e.g., 99% negative, 1% positive). Why is raw accuracy deceptive?
- Define True Positives, False Positives, True Negatives, and False Negatives.
- Compare Precision, Recall, F1-Score, and ROC-AUC curve behavior.
- Recommend optimal threshold selection strategies for safety-critical systems.

5. Explain the architecture and update mechanics of the Adam optimizer compared to classical Stochastic Gradient Descent (SGD).
- Formulate the first and second moment equations with bias correction.
- Contrast learning rate behavior in sparse vs dense parameter regimes.
- Discuss hyperparameters beta1, beta2, and epsilon.

---

### 3. 5 MCQs

MCQ 1. In the Bias-Variance tradeoff, what symptom is typically observed when a model suffers from excessive variance?
a) High training error and high test error
b) Very low training error but high test error
c) Equal training and test error with underfitting
d) Zero irreducible noise
Answer: b
Explanation: High variance means the model has memorized training noise (overfitting), achieving near-perfect training scores but failing to generalize to unseen test data.

MCQ 2. Which regularization technique adds a penalty proportional to the sum of absolute parameter weights?
a) Ridge Regularization (L2)
b) Dropout
c) Lasso Regularization (L1)
d) Batch Normalization
Answer: c
Explanation: L1 (Lasso) penalizes the L1-norm of the weight vector, driving weights to exact zeros and enforcing sparsity.

MCQ 3. In medical screening where missing a true disease diagnosis is far more catastrophic than a false alarm, which metric should be prioritized?
a) Precision
b) Specificity
c) Recall (Sensitivity)
d) Accuracy
Answer: c
Explanation: Recall measures TP / (TP + FN). Maximizing recall minimizes False Negatives, ensuring fewer actual positive cases are missed.

MCQ 4. What is the primary purpose of applying non-linear activation functions (like ReLU) in multi-layer neural networks?
a) To prevent all weights from going to zero
b) To enable the network to learn non-linear decision boundaries
c) To speed up hard disk storage operations
d) To eliminate the need for backpropagation
Answer: b
Explanation: Without non-linear activation functions, a composition of linear layers collapses mathematically into a single linear transformation, regardless of depth.

MCQ 5. How does the Adam optimization algorithm adapt individual parameter learning rates?
a) By randomly resetting weights every epoch
b) By computing moving averages of both past gradients and past squared gradients
c) By calculating the exact Hessian matrix on each step
d) By doubling the learning rate whenever loss increases
Answer: b
Explanation: Adam tracks both the first moment (momentum) and second moment (uncentered variance of gradients) to adjust per-parameter learning rates dynamically.
"""


def load_sample_deck(upload_dir: str, faiss_dir: str, get_embeddings_fn):
    """
    Load the pre-packaged Machine Learning study deck into the session environment.
    Writes sample document, builds FAISS index, and returns structured data.
    """
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(faiss_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, SAMPLE_DECK_FILENAME)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_DOCUMENT_TEXT)

    # Ingest into FAISS
    from ingest import compute_document_set_id, text_splitting_recursive, _base_metadata
    doc_set_id = compute_document_set_id([file_path])
    chunks = text_splitting_recursive(SAMPLE_DOCUMENT_TEXT)
    base_meta = _base_metadata(file_path, doc_set_id)

    docs = []
    for idx, c in enumerate(chunks, start=1):
        meta = {
            **base_meta,
            "chunk_index": idx,
            "chunk_id": f"{base_meta['source_id']}:{idx}",
        }
        docs.append(Document(page_content=c, metadata=meta))

    embeddings = get_embeddings_fn()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(os.path.abspath(faiss_dir))

    parsed_quiz = parse_quiz_output(SAMPLE_QUIZ_RAW)

    return {
        "file_path": file_path,
        "filename": SAMPLE_DECK_FILENAME,
        "document_set_id": doc_set_id,
        "total_chunks": len(docs),
        "summary": SAMPLE_SUMMARY,
        "notes": SAMPLE_NOTES,
        "quiz_data": parsed_quiz,
        "quiz_raw": SAMPLE_QUIZ_RAW,
        "chunks": [d.page_content for d in docs],
    }
