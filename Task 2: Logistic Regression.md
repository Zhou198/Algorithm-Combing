## 1. Link & Difference between Logistic and Linear Regression

## 2. Principle of Logistic Regression

## 3. Derivation and Optimization for Loss function of Logistic Regression

## 4. Regularization and Model Assessment

## 5. Pros and Cons of Logistic Regression

## 6. Dealing with imbalanced sample
* **Change weights:** We can change the weights for different classes in loss function when training. Say, improve weight for the class with less sample size and decrease that for class with more sample size.
* **Add extra sample:** Add more data for the class whose sample size is less. Obviously, this method will cost much money and time.
* **Oversampling:** Sample on the small class with replacement until the sample size is close to the large one.
* **Undersampling:** Take a sub-sample on the large class, such that the sample size close to the small one. Actually, this method wastes a part of resource.

## 7. Parameters of Sklearn for Logistic Regression 
```{python}
LogisticRegression(penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="warn", max_iter=100, multi_class="warn", verbose=0, warm_start=False, n_jobs=None)
```
* **penalty:** The norm used in regularization: "l1" or "l2". "l2" penalty is defualt. The "liblinear" solver supports both penalties.
* **dual:** Dual or primal formulation. Prefer dual=False when n_samples > n_features.
* **tol:** Tolerance of accuracy for stopping criteria.
* **C:** Inverse of lambda, the smaller C is, the stronger the penal is. Default value is 1.0.
* **fit_intercept:** Weather include the intercept term. Default is True.
* **intercept_scaling:** Only works when using "liblinear" and fit_intercept=True. Default is 1.0.
* **class_weight:** Weights for classes with "balanced" mode or dictionary form {class_label: weight}. Default is None, representing each class has equal weight. If choosing "balanced", it will automatically adjust the weights.
* **random_state:** The seed of pseudo number generator when shuffling the data.
* **solver:** Algorithm to optimize objective function. "newton-cg", "lbfgs", "liblinear", "sag" and "saga". Default is "liblinear". "liblinear" is good for small datasets while "sag" and "saga" compute faster for large datasets. When dealing with muticlass problem, "newton-cg", "lbfgs", "sag" and "saga" can handle multinomial loss while "liblinear" works for one-vs-rest schemes.
* **max_iter:** The maxmum number of iterations to converge, default is 100. Works when choosing "newton-cg", "lbfgs" or "sag" as a solver.
* **multi_class:** "ovr" means fitting of binary problem for each label. "mutinomial" (not useful for "liblinear") means loss fit cross over the whole distribution. "auto" means choosing "ovr" if data is binary or solver is "liblinear", otherwise "multinomial" is selected. Default is "ovr".
* **verbose:** Default is 0 and any positive int set when using "liblinear" or "lbfgs".
* **warm_start:** Use previous call to fit if set it True, default is False.
* **n_jobs:** Like LinearRegression, number of CPU to do computation. None means 1, -1 means use all CPUs.
- [] *Attributes* from the model are  "classes_", "coef_", "intercept_" and "n_iter_", representing target labels in training data, an array for $\hat\beta$, intercept term or 0 when fit_intercept=False and number of iterations respectively.






## Reference
* **1.** 
* **2.**
* **3.** https://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf
