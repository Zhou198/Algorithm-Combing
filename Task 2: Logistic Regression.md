## 1. Link & Difference between Logistic and Linear Regression
* **Link:** Both Logistic and Linear Regression belong to GLM (Generalized Linear Models).
*  **Difference:** 
   + The response of Logistic Regression is binary (discrete) while that can be any number (continuous) from $\mathbb{R}^1$ in Linear Regression, which means the former is used to do classification (or probability prediction) while the latter is to do any numerical prediction.
   + $\mu=\mathbf{E}(Y|X=x)$  can be map into $\mathbb{R}^1$ through link function $g(\mu)=\mu$ in Linear Regression while through $g(\mu)=\log{\frac{\mu}{1-\mu}}$ in Logistic Regression.
   + Different loss functions. Linear Regression's is involved with quadratic function while Logistic Regression's related to likelihood function.

## 2. Principle of Logistic Regression
The goal is to model $p(x):=P(Y=1|X=x)$ and Linear Regression would model it as $p(x)=x^\top\beta$. However, it is not a good idea since $p(x)\in(0, 1)$. So we need a certain function map the probability into the whole $\mathbb{R}^1$. $$\log{\frac{p(x)}{1-p(x)}}=x^\top\beta \quad \Rightarrow \quad p(x)=\frac{1}{1+\exp(-x^\top\beta)},$$ and we call$\frac{p(x)}{1-p(x)}=\exp(x^\top\beta)$ as odds.

## 3. Derive and Optimize for Loss Function of Logistic Regression
* **For $y_i = 0, 1$ scenario:**
1) Assume data: $\lbrace (x_i, y_i), i = 1, 2, \cdots, n \rbrace$ and $y_i=0, 1, Y_i|X_i=x_i \sim Bernoulli(p(x_i))$, then the log-likelihood is $$\begin{aligned}
L(\beta, X)&=\log\left(\prod_{i=1}^np(x_i)^{y_i}(1-p(x_i))^{1-y_i}\right)\\\\
&=\sum_{i=1}^n{y_i\log(p(x_i))+(1-y_i)\log(1-p(x_i))}\\\\
&=\sum_{i=1}^n{y_i\log{\frac{p(x_i)}{1-p(x_i)}}+\log(1-p(x_i))}\\\\
&=\sum_{i=1}^n{y_i(x_i^\top\beta)-\log(1+\exp(x_i^\top\beta))}.
\end{aligned}$$ Therefore, maximizing above log-likelihood is also equivalent to minimize below loss function$$\ell(\beta, X)=\sum_{i=1}^n{-y_i(x_i^\top\beta)+\log(1+\exp(x_i^\top\beta))},$$which is also called total deviance.
2) We use Newton-Raphson to get our solution, which means
$$\beta^{(k+1)}\leftarrow \beta^{(k)}-\left[\ddot\ell\left(\beta^{(k)}, X\right)\right]^{-1}\dot\ell\left(\beta^{(k)}, X\right),$$ where 
$$\begin{aligned}
\dot\ell\left(\beta, X\right)&=\sum_{i=1}^{n}(-y_ix_i+\frac{\exp(x_i^\top\beta)x_i}{1+\exp(x_i^\top\beta)})=\sum_{i=1}^{n}x_i(-y_i+p(x_i))=-X^\top\left(Y-P\right),\\\\
\ddot\ell\left(\beta, X\right)&=\sum_{i=1}^{n}x_i\frac{\exp(x_i^\top\beta)x_i^\top(1+\exp(x_i^\top\beta))-\exp(x_i^\top\beta)^2x_i^\top}{(1+\exp(x_i^\top\beta))^2}\\\\
&=\sum_{i=1}^{n}x_i\frac{\exp(x_i^\top\beta)x_i^\top}{(1+\exp(x_i^\top\beta))^2}\\\\
&=\sum_{i=1}^{n}x_ix_i^\top p(x_i)(1-p(x_i))=X^\top WX,
\end{aligned}$$ where $W=\text{Diag}\left(p(x_i)(1-p(x_i)\right).$ Note convergence may not be guaranteed if $W$ or $X^\top WX$ is not invertible, especially when 2 classes are well separated with $p(x_i)$'s much close to 0 or 1.
   
* **Alternative coding for response:**
1) Recall the unit loss function when coding $y_i=0, 1$ is $${-y_i(x_i^\top\beta)+\log(1+\exp(x_i^\top\beta))},$$which can be re-expressed as 
$$\begin{cases}
\log[1+\exp(-x_i^\top\beta)] & y_i=1\\\\
\log[1+\exp(x_i^\top\beta)] & y_i=0\\\\
\end{cases}.$$This can be put together into $\log(1+\exp(-y_ix_i^\top\beta))$ if we code class 0 as class -1. Now the loss function becomes$$\ell(\beta, X)=\sum_{i=1}^{n}\log\left(1+\exp(-y_ix_i^\top\beta)\right)$$ for $\pm1$ scenario.

2) In this case, we choose Gradient Descent to do optimization, which is
$$\beta^{(k+1)}\leftarrow\beta^{(k)}-\gamma\dot\ell\left(\beta^{(k)}, X\right),$$ where 
$$\dot\ell\left(\beta, X\right)=\sum_{i=1}^{n}\frac{-\exp(-y_ix_i^\top\beta)y_ix_i}{1+\exp(-y_ix_i^\top\beta)}=\sum_{i=1}^{n}\left(\frac{1}{1+\exp(-y_ix_i^\top\beta)}-1\right)y_ix_i.$$

## 4. Regularization and Model Assessment
* **Sparse Logistic Regression:**
$$\mathop {\min }\limits_\beta \frac{1}{n}\ell\left(\beta, X\right) + \lambda \mathcal{P}(\beta)$$
1) $\ell_1$ penalty (Lasso): $\mathcal{P}(\beta)=\sum_{j=1}^{p}|\beta_j|$, it can shrink some coefficients exactly to 0.
2) $\ell_2$ penalty (Ridge): $\mathcal{P}(\beta)=\sum_{j=1}^{p}\beta_j^2$, it can shrink some correlated variables simultaneously.
3) Elastic Net: $\mathcal{P}(\beta)=\sum_{j=1}^{p}\left(0.5(1-\alpha)\beta_j^2+\alpha |\beta_j|\right)$, where tuning parameter $\alpha \in [0, 1]$. Make a compromise between Lasso and Ridge penalties. Additionally, it works with advantages of both Lasso and Ridge.
4) Grouped Lasso:   $\mathcal{P}(\beta)=\sum_{j=1}^{p}||\overrightarrow{\beta}_j||_2$, it works especially for multinomial problem.
5) Nonconvex penalties: SCAD penalty, MC+ penalty. Both are concave and we are interested in them when $p$ is really large.
 * **Assessment:**
1) Misclassification Error & Accuracy: $E(f; D)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left({f(x_i)\neq y_i}\right)$, the smaller the better.  $acc(f;D)=1-E(f;D)$.
2) Precision & Recall: $P=\frac{TP}{TP+FP}, R=\frac{TP}{TP+FN}$. Precision and Recall are a pair of contradictory measures. The relationship between them is that one falls as another rises.

|   |Predicted Positive|Predicted Negative|
|:------ |:------:|:------:|
|Real Positive|TP|FN|
|Real Negative|FP|TN|

3) ROC & AUC: Inspect cut point.

## 5. Pros and Cons of Logistic Regression
* Pros:
1) It is much easy to interpretation and understanding.
2) The computation is not fairly expensive.
3) Since the result is probability,  we can try different threshold to adjust our final decisions under different background. 

* Cons:
1) When classes are well separated,  it may not converge.
2) In some situation it is not powerful because of its simple model structure. 

## 6. Dealing with imbalanced sample
* **Change weights:** We can change the weights for different classes in loss function when training. Say, improve weight for the class with less sample size and decrease that for the class with more sample size.
* **Add extra sample:** Add more data for the class whose sample size is less. Obviously, this method will cost much money and time.
* **Oversampling:** Sample on the small class with replacement until the sample size is close to the large one's.
* **Undersampling:** Take a sub-sample on the large class, such that the sample size close to the small one's. Actually, this method wastes a part of resource.

## 7. Parameters of Sklearn for Logistic Regression 
```{python}
LogisticRegression(penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="warn", max_iter=100, multi_class="warn", verbose=0, warm_start=False, n_jobs=None)
```
* **penalty:** The norm used in regularization: "l1" or "l2". "l2" penalty is default. The "liblinear" solver supports both penalties.
* **dual:** Dual or primal formulation. Prefer dual=False when n_samples > n_features.
* **tol:** Tolerance of accuracy for stopping criteria.
* **C:** Inverse of lambda, the smaller C is, the stronger the penal is. Default value is 1.0.
* **fit_intercept:** Weather include the intercept term. Default is True.
* **intercept_scaling:** Only works when using "liblinear" and fit_intercept=True. Default is 1.0.
* **class_weight:** Weights for classes with "balanced" mode or dictionary form {class_label: weight}. Default is None, representing each class has equal weight. If choosing "balanced", it will automatically adjust the weights.
* **random_state:** The seed of pseudo number generator when shuffling the data.
* **solver:** Algorithm to optimize objective function. "newton-cg", "lbfgs", "liblinear", "sag" and "saga". Default is "liblinear". "liblinear" is good for small datasets while "sag" and "saga" compute faster for large datasets. When dealing with muticlass problem, "newton-cg", "lbfgs", "sag" and "saga" can handle multinomial loss while "liblinear" works for one-vs-rest schemes.
* **max_iter:** The maxmum number of iterations to converge, default is 100. Works when choosing "newton-cg", "lbfgs" or "sag" as a solver.
* **multi_class:** "ovr" means fitting of binary problem for each label. "multinomial" (not useful for "liblinear") means loss fit cross over the whole distribution. "auto" means choosing "ovr" if data is binary or solver is "liblinear", otherwise "multinomial" is selected. Default is "ovr".
* **verbose:** Default is 0 and any positive int set when using "liblinear" or "lbfgs".
* **warm_start:** Use previous call to fit if set it True, default is False.
* **n_jobs:** Like LinearRegression, number of CPU to do computation. None means 1, -1 means use all CPUs.
- [ ] *Attributes* from the model are  "classes_", "coef_", "intercept_" and "n_iter_", representing target labels in training data, an array for $\hat\beta$, intercept term or 0 when fit_intercept=False and number of iterations respectively.



## Reference
* **1.** https://www.quora.com/Whats-the-relationship-between-linear-and-logistic-regression-What-are-the-similarities-and-differences
* **2.** Hastie, Trevor, Tibshirani, Robert, and Friedman, J. H. (2009). The elements of statistical learning: Data mining, inference, and prediction, 2nd Edition
* **3.** 周志华 (2016). [机器学习](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLbook2016.htm).
* **4.** https://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf
