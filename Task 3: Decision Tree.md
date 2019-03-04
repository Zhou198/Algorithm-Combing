## The Basics of Information Theory
* **Entropy:** A measure describing the impurity of a set. The smaller the purer. $(p_1, p_2, \cdots, p_K)$ can be viewed as probability mass for classes in set $D$. The entropy is defined as $$\text{Ent}(D)=-\sum_{k=1}^Kp_k\log p_k.$$

* **Joint Entropy:** The entropy of a joint (multivariate) probability distribution. For instance, $(X, Y) \sim p(x, y)$, then the joint entropy is $$\text{Ent}(X, Y)=-\sum_{x\in \mathcal{X}}\sum_{y\in \mathcal{Y}}p(x, y)\log p(x, y).$$

* **Conditional Entropy:** Given joint and conditional distributions $p(x, y)$ and $p(y|x)$, then the conditional entropy of $Y$ given $X$ is $$\text{Ent}(Y|X)=-\sum_{x\in \mathcal{X}}\sum_{y\in \mathcal{Y}}p(x, y)\log p(y|x).$$

* **Information Gain:** Suppose feature $a$ has $V$ many levels, then mark those samples falling into level $v$ as $D^v$ and the sample size is $|D^v|$, where $v=1, 2, \cdots, V$. Given a sub-sample set, we still can get its entropy $\text{Ent}(D^v)$, then the information gain is defined as $$\text{Gain}(D, a)=\text{Ent}(D)-\sum_{v=1}^V\frac{|D^v|}{|D|}\text{Ent}(D^v).$$ Actually, the greater information gain, the better splitting. This means after splitting based on this feature, we can get a much less average entropy.

* **Gini Impurity:** A different measure to describe the impurity. $$\text{Gini}(D)=\sum_{j\neq k}p_jp_k=1-\sum_{k=1}^Kp_k^2.$$


## Principle & Application Scenario for Different Algorithms
* **ID3:** Choose variables to split based on information gain. Then our target is $$a^\ast = \arg \mathop{\max}\limits_{a\in A} \text{Gain}(D, a).$$
* **C4.5:** Choose variables to split based on gain ratio instead of information gain, which is defined as $$\text{Gain_ratio}(D, a)=\frac{\text{Gain}(D, a)}{\text{IV}(a)},$$ where $\text{IV}(a)=-\sum\limits_{v=1}^{V}\frac{|D^v|}{|D|}\log\frac{|D^v|}{|D|}.$ Then our target variable is $$a^\ast = \arg \mathop{\max}\limits_{a\in A} \text{Gain_ratio}(D, a).$$ Actually, "Information Gain" criteria prefers features with much levels. In order to avoid the influence that might bring, C4.5 algorithm uses "Gain ratio" instead of "Information Gain".  However, the more levels a feature has, the greater $IV$ is. 
Heuristically, C4.5 filters features with larger information gain at first, from which a feature with highest gain ratio (feature with less levels) is selected finally.

* **CART:** Choose variables to split based on gini index, which is defined as $$\text{Gini_index}(D, a)=\sum_{v=1}^V\frac{|D^v|}{|D|}\text{Gini}(D^v),$$ and our target variable is $$a^\ast = \arg \mathop{\min}\limits_{a\in A} \text{Gini_index}(D, a).$$ It can be used to do regression and classification.

## Principle of Regression Tree
Intuitively, the prediction value for new data $x$ falling into the node $\tau$ is equal to the average value of observations in node $\tau$, which is $$\hat{y}(x|\tau)=\bar{y}(x_i\in\tau)=\frac{\sum_{i=1}^n\mathbf{1}{(x_i\in\tau)}{y_i}}{\sum_{i=1}^n\mathbf{1}{(x_i\in\tau)}},$$ and the mean square error at node $\tau$ is defined as $$\hat{\sigma}^2(\tau)=\frac{\sum_{i=1}^n\mathbf{1}{(x_i\in\tau)}{ ((y_i-\hat y(x|\tau))^2}}{\sum_{i=1}^n\mathbf{1}{(x_i\in\tau)}}.$$ Actually, it is a sample variance for training data in node $\tau$.

In Classification Tree, we use some impurity measure to select and split features. Similarly, in Regression Tree, we minimize the weighted sum of prediction mean squared errors for two daughter nodes. That is, we want to find the $j$-th variable and split $s$, to minimize $$p_L\hat{\sigma}^2(\tau_L)+p_R\hat{\sigma}^2(\tau_R),$$ where $p_L$ and $p_R$ are the relative proportions of samples in two nodes.

## Method to Avoid Decision Tree Overfitting
* **Prepruning:** During the tree's growing, we always use validation data to decide whether we need to split at this node. Specifically, after splitting this node, if the accuracy of validation data is not improved compared before, we stop splitting at this node. Although prepruning is much faster than postpruning, it may lead underfitting.

* **Postpruning:** Let the tree grow to saturation at first, then use validation set to prune the branch,  from bottom to up, if the accuracy of validation set is improved after pruning. Since postpruning requires to grow a largest tree and then prune the tree step by step, this will lead training to be much time-consuming than prepruning.

## Model Assessment
* For Regression Tree, we can use MSE on testing data (generalized error) to assess models. 
* For Classification Tree, we can use various indexes to measure a model. Actually, this means to assess a classier's power, which is same to content related on assessment in following link:
https://github.com/Zhou198/Algorithm-Combing/blob/master/Task%202:%20Logistic%20Regression.md

## Parameters of Sklearn & Plot Decision Tree with Python
```python
DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                       min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, 
                       min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

DecisionTreeRegressor(criterion="mse", ...)
```
* **criterion:** Measure the quality of splitting. "gini" or "entropy" for classifier. "mse", "friedman_mse" or "mae" for regressor.
* **splitter:** "best" or "random". Method to split at each node.
* **max_depth:** "None" or int. The maximum depth of the tree. If None, tree will grow up until each leaf node is pure or  contains sample less than min_samples_split.
* **min_samples_split:** The minimum number of samples to split a node. If int, then it is the minimum number; if a float, then it is $\lceil$n $\times$ min_ samples_split$\rceil$.
* **min_samples_leaf:** The minimum sample size in each leaf node. If int, then it is the minimum sample size in each node; if a float, then it is $\lceil$n $\times$ min_ samples_leaf$\rceil$.
* **min_weight_fraction_leaf:** The minimum weighted fraction to be at leaf node. They will have equal weights if sample_weight is not provided.
* **max_features:** The number of features considered when splitting. If int, consider max_features at each splitting. If float, it is $\lceil$n_features $\times$ max_features$\rceil$. If "auto", it is sqrt(n_features). If "log2", it is log2(n_features). If None, it just is n_features. 
* **min_impurity_split:** Stop growing when reach this value, then this node becomes a leaf node.
* **class_weight:** Dictionary-like value, "balanced" or None. To solve imbalanced sample problem
* **presort:** Whether to presort the data to speed training. It is not recommended with large dataset. 

_Attributes_ 
1) **class_** array of class labels.
2) **max_features_** The inferred value of max_features.
3) **n_classes_** The number of each class.
4) **n_features_** Number of features when fit is performed.

```python
### Build a decision tree with training data (X, y) ###
fit(X, y, sample_weight=None, check_input=True, X_idx_sorted=None)

### Predict class or do regression for data X ###
predict(X, check_input=True)

### Predict the probability for each X classified as one label ###
predict_proba(X, check_input=True)
```
* **Use built-in iris data to plot a tree:**
```python
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_graphviz

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

export_graphviz(clf, out_file = "tree.dot", filled = True, rounded = True,
                class_names = iris.target_names, special_characters = True)
```
<div align=center> <img src="https://user-images.githubusercontent.com/47863455/53703007-e5424080-3dda-11e9-92bd-3fcfeeee49bf.png" width="70%" height="70%"></div>

## Reference
* **1.** 周志华 (2016). [机器学习](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLbook2016.htm)
* **2.** http://www.inf.ed.ac.uk/teaching/courses/fmcs1/slides/lecture25.pdf
* **3.** Hastie, Trevor, Tibshirani, Robert, and Friedman, J. H. (2009). The elements of statistical learning: Data mining, inference, and prediction, 2nd Edition
* **4.** https://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf
* **5.** https://www.cnblogs.com/pinard/p/6056319.html
