# Lab 14 Report: Diagnosing and Fixing Bias and Variance in Machine Learning Models
---

## 1. Introduction

When developing machine learning models, one of the most critical challenges is achieving a model that generalizing well to new, unseen data. Two primary sources of error that prevent models from generalizing are **bias** and **variance**.

*   **Bias** is the error introduced by approximating a highly complex real-world problem with a simplified model. A model with high bias makes strong assumptions about the data (e.g., using a linear model for highly non-linear data). This results in **underfitting**, where the model fails to capture the underlying patterns in both the training and test sets.
*   **Variance** is the error introduced by the model's sensitivity to small fluctuations in the training set. A model with high variance is overly complex and "memorizes" the noise in the training data rather than the actual signal. This leads to **overfitting**, where the model performs exceptionally well on the training data but poorly on the test set.

The tradeoff between bias and variance is fundamentally a question of model complexity. Our objective is to find the "just right" level of complexity. We can evaluate our models and identify these issues by analyzing their **training error** and **cross-validation (CV) error** against a baseline level of acceptable performance.

## 2. Diagnosing Model Performance Issues

Before we can improve our model, we must accurately diagnose whether it suffers from high bias or high variance. We do this by observing the learning errors:

*   **High Bias (Underfitting):** Both the training error and the CV error are unacceptably high relative to the baseline performance. Furthermore, the gap between the training error and the CV error is usually very small. The model simply isn't learning enough.
*   **High Variance (Overfitting):** The training error is low (sometimes near zero), but the CV error is significantly higher. There is a large gap between the two errors, indicating that the model has memorized the training set but fails to generalize.

## 3. Addressing High Bias (Underfitting)

When a model exhibits high bias, it lacks the necessary complexity to capture the relationships within the data. Based on our practical lab experiments, we can fix high bias using the following techniques:

### 3.1. Adding Polynomial Features
**How it works:** By introducing higher-order polynomial features (such as $x^2$, $x^3$, etc.), we expand the hypothesis space, giving the model the flexibility to fit non-linear patterns.
**Lab Observation:** In our experiments with a simple linear regression model, increasing the polynomial degree allowed the model to fit a complex curve. As we added polynomial features, both the training and CV errors dropped significantly closer to the baseline level, demonstrating that the underfitting issue was resolved.

### 3.2. Acquiring Additional Features
**How it works:** Sometimes a model underfits because it simply does not have enough information or variables to make accurate predictions. Adding new, relevant features provides the model with more dimensions to learn the true underlying target function.
**Lab Observation:** When we introduced a second meaningful feature to our dataset, the training error dropped sharply, and the CV performance improved significantly compared to using only a single feature.

### 3.3. Decreasing the Regularization Parameter ($\lambda$)
**How it works:** Regularization (such as Ridge/L2 regularization) penalizes large model weights to prevent overfitting. However, if the regularization parameter ($\lambda$) is set too high, the model becomes overly constrained. It acts like a much simpler model, introducing high bias.
**Lab Observation:** By decreasing $\lambda$, we allowed the model's weights to grow sufficiently to learn the necessary patterns. This loosening of restrictions successfully brought the error metrics down to acceptable baseline levels.

## 4. Addressing High Variance (Overfitting)

When a model exhibits high variance, it has learned the training data *too* well, including the random noise, preventing it from generalizing. The following methods correct overfitting:

### 4.1. Increasing the Regularization Parameter ($\lambda$)
**How it works:** A higher $\lambda$ forces the learning algorithm to keep model parameters (weights) small. This restricts the model's flexibility and smooths out the hypothesis curve, reducing its tendency to chase noisy data points and "wiggle" unnecessarily.
**Lab Observation:** We increased $\lambda$ for a high-degree polynomial model. This penalized the overly complex weights, reducing the CV error significantly and narrowing the large gap between the training and CV performance.

### 4.2. Using Smaller Sets of Features
**How it works:** Having too many features—especially irrelevant ones—can cause the model to build complex correlations that don't actually exist (essentially memorizing noise). Selecting a smaller set of features constraints the model's complexity.
**Lab Observation:** When a random, irrelevant feature (like an ID column) was added to the data, the CV error skyrocketed as the model tried to learn from it, even as the training error decreased. Removing this irrelevant feature immediately restored the model's generalization capabilities.

### 4.3. Getting More Training Examples
**How it works:** An overly complex model might easily overfit a small dataset. However, as more training data is introduced, it becomes increasingly difficult for the model to memorize the entire dataset. The model is eventually forced to learn the actual underlying function rather than the noise.
**Lab Observation:** Plotting the learning curves revealed that as the training set size increased, the CV error gradually approached the training error, mitigating the high variance issue. It is important to note, however, that adding more data does *not* solve high bias (if the model is too simple, the training error remains flat and high regardless of dataset size).

## 5. Conclusion

Improving model performance is a systematic process of diagnosis and targeted remediation. The key to building robust machine learning models is mastering the balance between model complexity and generalization. 

If a model suffers from **high bias**, the goal is to *increase model capability*: add polynomial features, gather additional relevant features, or decrease the regularization penalty. Conversely, if a model suffers from **high variance**, the goal is to *decrease model capability or increase the data sample size*: remove irrelevant features, increase the regularization penalty, or collect more training examples. By properly applying these tools, we can successfully guide our models toward the optimal "just right" state, ensuring they perform exceptionally well on real-world, unseen data.
