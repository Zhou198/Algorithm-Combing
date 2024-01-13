[//]:  # (<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>)
[//]: # (\usepackage{bm})
## 1.  Concept of Machine Learning
* **Supervised Learning:** The goal is to predict the value of outcome (or do classification for observations) based on input variables (like Regression, KNN). It is involved with target variable .
* **Unsupervised Learning:** The goal is to find out the associations or somewhat patterns among input variables (like PCA, ICA, Clustering). It is not involved with target variable .
* **Generalization:** Use the model learning from training data to predict (or do classification) on testing data and observe its corresponding capacity or accuracy. 
* **Overfitting:** Model performs fairly well on training data but less well on testing data, which may appear on complex models. We can discard some variables or introduce a regularization term to reduce overfitting.
* **Underfitting:** Model performs badly both on training and testing data, which may appear on simpler models. We can add more variables to improve models' performance.
* **Bias-Variance Tradeoff:** Complex models have less bias (small changes to the data change the result a lot $\Rightarrow$ a high-variance solution) while simpler models have less variance (model does not fit data very well $\Rightarrow$ a biased solution). For instance, in ordinary linear regression, although $\hat\beta=(X^\top X)^{-1}X^\top Y$ is an unbiased estimator for true $\beta$, another estimator, say $\tilde\beta$ obtained when being introduced regularization (like ridge or lasso), has lower variance.
* **Cross-Validation:** The whole data can be divided into training set, validation set and testing set based certain ratio. Validation set (tuning set) is used to estimate prediction error for the purpose of model selection. We can use cross-validation to select the parameter (model) with smallest validation error.

## 2.  Principal of Linear Regression
$$ Y=X \mathbf{\beta} + \mathbf{\varepsilon},  \quad \mathbf{\varepsilon} \sim N\left(\mathbf{0}, \sigma^2 I_n\right),$$ where $Y_{n\times 1}=\left(y_1, y_2, \cdots, y_n\right)^\top, X_{n\times (p+1)}=\left(x_1, x_2, \cdots, x_n\right)^\top, x_i=\left(1, x_{i1}, x_{i2}, \cdots, x_{ip}\right)^\top.$
Linear regression is a method trying to find a best hyper-plane $h(x, \beta)$ to minimize the difference between points and this plane.

## 3. Functions of Linear Regression
* **Loss function:**
$$\left(y_i-h(x_i, \beta)\right)^2, \quad \text{for} \quad i = 1, 2, \cdots, n.$$
* **Cost function:**
$$\ell(\mathbf{\beta})=\frac{1}{n}\sum_{i=1}^n\left(y_i-h(x_i, \beta)\right)^2=\frac{1}{n}\left(Y-X\mathbf{\beta}\right)^\top\left(Y-X\mathbf{\beta}\right).$$
* **Objective function:**
$$\min  \frac{1}{n}\left(Y-X\mathbf{\beta}\right)^\top\left(Y-X\mathbf{\beta}\right)+\lambda||\beta||_p^p.$$ If we do not consider sparsity, we can set $\lambda =0$.

## 4.  Optimization Methods
* **Gradient Descent:**  $\beta^{(k+1)}\leftarrow\beta^{(k)}-\gamma\dot\ell\left(\beta^{(k)}\right)$, where $0\leq\gamma\leq 1$ is step size.
* **Newton-Raphson:** $\beta^{(k+1)}\leftarrow \beta^{(k)}-\left[\ddot\ell\left(\beta^{(k)}\right)\right]^{-1}\dot\ell\left(\beta^{(k)}\right).$
<div align=center> <img src="https://user-images.githubusercontent.com/47863455/53610596-775b0680-3b99-11e9-8d9f-ccda57c65f1c.jpeg" width="50%" height="50%" “Plot for Newton-Raphson”></div>

* **Quasi-Newton:** $\beta^{(k+1)}\leftarrow \beta^{(k)}-\alpha_kB_k^{-1}\dot\ell\left(\beta^{(k)}\right),$ where $B_k$ is approximated by some algorithms like DEP, BFGS, Broyden.
* **Link & Difference among above methods:**  Gradient descent directly updates points towards the direction of steepest descent for $\ell\left(\beta\right)$, while Newton-Raphson indirectly optimizes by finding the root of $\dot\ell\left(\beta\right)=0$. Newton-Raphson should converge faster than gradient descent, however it requires the second partial derivative (Hessian) matrix to be positive definite. Since there might be no inverse of Hessian matrix or the inverse is computationally expensive, someone devised other methods to approximate it, which reduces the load of computation.


## 5. Assessment and Selection for Linear Regression
There are several criterion used to assess linear models.
* **(Adjusted) R-square:** Reflect the goodness of fitting. The larger  R-square is, the better the model fits. However, it cannot detect overfitting. The harder we try to fit the data, the more the training error underestimates the testing error. Therefore, we cannot rely on training error only.
* **MSE (Mean Squared Error) on testing data:** The criterion is based on how the model will work on the outside real-word data, which is independent on training set.
$$MSE=\frac{1}{n}\sum_{i=1}^n\left(Y_i-\hat Y_i\right)^2.$$
* **Other measures similar to the MSE:**
$$MAE=\frac{1}{n}\sum_{i=1}^n\left|Y_i-\hat Y_i\right|, \qquad MAPE=\frac{1}{n}\sum_{i=1}^n\left|\frac{Y_i-\hat Y_i}{Y_i}\right|.$$Actually, when using the MAPE measure, it should guarantee $Y_i  \neq 0$ for all $i$'s.
* **Information Criterion:**
$$AIC=2p-2\ln\left(L(\hat\beta)\right), \qquad BIC=p\ln(n)-2\ln\left(L(\hat\beta)\right).$$

## 6.  Parameters in Sklearn
```
LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
```
* **fit_intercept:** Whether to include intercept term in this model, default is True.
* **normalize:** Whether to normalize the input $X$, default is False. If True, $X\leftarrow \frac{X-\bar X}{||X||_2}$.
* **copy_X:** $X$ will be copied if True, otherwise it will be overwritten.
* **n_jobs:** Number of CPU to do computation. None means 1, -1 means use all CPUs.
- [ ] *Attributes* from the model are  "intercept_" and "coef_", representing $\hat\beta_0$ and $\hat\beta_1, \hat\beta_2,\cdots, \hat\beta_p$ respectively.
```
fit(X, y, sample_weight=None)
```
* **X:** Training data with array-like or sparse matrix.
* **y:** Target values with array-like form.
* **sample_weight:** Individual weight for $\hat\beta_i$'s.
```
predict(X)
```
* **X:** Testing data with array-like form.  It returns the predicted target values.



## Reference
**1.** Hastie, Trevor, Tibshirani, Robert, and Friedman, J. H. (2009). The elements of statistical learning: Data mining, inference, and prediction, 2nd Edition

**2.** http://www.seas.ucla.edu/~vandenbe/236C/lectures/qnewton.pdf

**3.** https://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf



















