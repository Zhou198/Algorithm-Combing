## Hard Margin SVMs (Linearly separable)
Assume linearly separable data $\\{(x_i, y_i)\\}_{i=1}^N$ with dimension $N\times p$ .
*  **Hyperplanes**: The separating hyperplane $f(x)=\omega^\top x + b=0$ has a $p-1$ dimension. The positive plane is $f(x)=\omega^\top x + b=A$ for some $A>0$. So for those $y_i=+1$ samples with feature $x_i$'s, $\omega^\top x_i + b\geq A$. Conversely, the negative plane is $f(x)=\omega^\top x + b=-A$ and those $y_i=-1$ samples with feature $x_i$'s satisfy $\omega^\top x_i + b\leq-A$. Therefore, for any data point from linearly separated samples, we have $$y_if(x_i)=y_i(\omega^\top x_i + b)\geq A.$$

* **Margin**: Denote these points on $\pm$ planes as support vectors with notation $x_{SV}$. Therefore, $$|f(x_{SV})|=|\omega^\top x_{SV} + b|=A,$$ and the margin defined as the distance between $\pm$ planes  and that we want to maximize is $$\frac{2A}{||\omega||}.$$

* **Objective Problem**: Since we can arbitrarily multiply a non-zero constant for two sides of $f(x)=\omega^\top x + b=A$, for simplicity, we just scale the $\omega$ such that the $A=1$. Then the problem becomes to $$\max_\limits{\omega, b}\frac{2}{||\omega||},   \quad \text{s.t.  } y_i(\omega^\top x_i + b)\geq 1,$$ which is equivalent to  $$\min_\limits{\omega, b}\frac{1}{2}||\omega||^2,   \quad \text{s.t.  } y_i(\omega^\top x_i + b)\geq 1.$$

* **Solve Problem**: Use Lagrangian method to deal with this problem. $$L=\frac{1}{2}||\omega||^2-\sum\limits_{i=1}^N\alpha_i[y_i(\omega^\top x_i + b)-1],$$ where $\alpha_i\geq 0$ are Lagrangian multipliers.
$$\begin{cases}
\frac{\partial{L}}{\partial{\omega}}=\omega-\sum\limits_{i=1}^N\alpha_iy_ix_i=0\\\\
\frac{\partial{L}}{\partial{b}}=-\sum\limits_{i=1}^N\alpha_iy_i=0\\\\
\frac{\partial{L}}{\partial{\alpha_i}}=y_i(\omega^\top x_i + b)-1=0
\end{cases} \Rightarrow
\begin{cases}
\omega=\sum\limits_{i=1}^N\alpha_iy_ix_i\\\\
\sum\limits_{i=1}^N\alpha_iy_i=0\\\\
b=y_i-\omega^\top x_i
\end{cases}.$$
Therefore, the classifier is $$f(x)=\omega^\top x+b=\sum\limits_{i=1}^N\alpha_iy_ix_i^\top x +b.$$


## Soft Margin SVMs (Non-separable Case)
For linearly separable case, we have 
$$\min_\limits{\omega, b}\frac{1}{2}||\omega||^2,   \quad \text{s.t.  } y_i(\omega^\top x_i + b)\geq 1.$$
However, if $y_i(\omega^\top x_i + b)\geq 1$ is impossible for all data points, we need to  introduce a slack variable $\xi_i\geq 0$ for each data point and relax $y_i(\omega^\top x_i + b)\geq 1$ to $y_i(\omega^\top x_i + b)\geq 1-\xi_i$.
Actually, $\xi_i$ represents the misclassification somehow and we want $\sum\limits_{i=1}^N\xi_i$ to be small. So we rewrite the objective function as $$\min_\limits{\omega, b}\frac{1}{2}||\omega||^2 + C\sum\limits_{i=1}^N\xi_i,   \quad \text{s.t.  } y_i(\omega^\top x_i + b)\geq 1-\xi_i,  \xi_i\geq 0.$$

## SVMs Dual Problem
Lagrangian: $$L(\omega, b, \xi, \alpha, \beta)=\frac{1}{2}||\omega||^2+\sum\limits_{i=1}^N\\{C\xi_i-\alpha_i\left[y_i\left(\omega^\top x_i+b\right)+\xi_i-1\right]-\beta_i\xi_i\\},$$ where $\alpha_i\geq 0, \beta_i\geq 0$ are KKT multipliers.

* **KKT stationary condition**: requires $\frac{\partial{L}}{\partial\omega}, \frac{\partial{L}}{\partial b}, \frac{\partial{L}}{\partial\xi_i}$ to equal to 0. $$\begin{cases}
\frac{\partial{L}}{\partial{\omega}}=\omega-\sum\limits_{i=1}^N\alpha_iy_ix_i=0\\\\
\frac{\partial{L}}{\partial{b}}=-\sum\limits_{i=1}^N\alpha_iy_i=0\\\\
\frac{\partial{L}}{\partial{\xi_i}}=C-\alpha_i-\beta_i=0
\end{cases} \Rightarrow
\begin{cases}
\omega=\sum\limits_{i=1}^N\alpha_iy_ix_i\\\\
\sum\limits_{i=1}^N\alpha_iy_i=0\\\\
\alpha_i+\beta_i=C
\end{cases}.$$Substitute above results into Lagrangian formula then get $$\begin{aligned}
L(\omega, b, \xi, \alpha, \beta)&=\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\alpha_i\alpha_jy_iy_jx_i^\top x_j-\sum\limits_{i=1}^N\alpha_i\left[y_i\left(\sum\limits_{j=1}^N\alpha_jy_jx_i^\top x_j + b\right)-1\right]\\\\
&=\sum\limits_{i=1}^N\alpha_i-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\alpha_i\alpha_jy_iy_jx_i^\top x_j,
\end{aligned}$$ which curiously is identical to that for the hard margin.

* **KKT complementary slackness condition**: requires $$\begin{cases}
\alpha_i[y_i(\omega^\top x_i+b)+\xi_i-1]=0\\\\
\beta_i\xi_i=0
\end{cases}.$$ 

*Support Vectors and $\alpha$*: 
1. For those points with non-zero slack variable ($\xi_i>0$), their corresponding $\alpha_i$'s are exactly equal to $C$.
2. For those points on $\pm$ planes ($\xi_i=0$), their corresponding $\alpha_i$'s falls into $(0, C)$.

Finally, the dual problem becomes $$\begin{aligned}
&\max_\limits{\alpha} \sum\limits_{i=1}^N\alpha_i-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\alpha_i\alpha_jy_iy_jx_i^\top x_j, \\\\ 
&\text{s.t. } \sum\limits_{i=1}^N\alpha_iy_i=0, \alpha_i \in [0, C].
\end{aligned}$$ Note: If $C \rightarrow +\infty$, this scenario becomes hard margin SVMs.


## SVMs with Kernel Trick
Map original data $x$ into a different feature space with much higher dimension: $$x=(x_1, x_2, ..., x_p) \mapsto \phi(x)=\left(\phi_1(x), \phi_2(x), ..., \phi_P(x)\right).$$
A Kernel is a function $K$, such that for all $x, y \in X$, $$K(x, y)=\phi(x)^\top \phi(y).$$ $K(x,y)$ is much easily computational instead of find a mapping and expanding $x$ into $\phi(x)$.
* **Linear Kernel**: $K(x, y)=x^\top y$.
* **Polynomial Kernel**: $K(x, y)=(c + x^\top y)^d$.
ex: $x=(x1, x2)^\top, y=(y_1, y_2)^\top$ and take kernel function $$\begin{aligned}
K(x, y)&=(1+x^\top y)^2\\\\
&=(1+x_1y_1+x_2y_2)^2\\\\
&=1+x_1^2y_1^2+x_2^2y_2^2+2x_1y_1+2x_2y_2+2x_1x_2y_1y_2\\\\
&=(1, x_1^2, \sqrt{2}x_1x_2, x_2^2, \sqrt{2}x_1, \sqrt{2}x_2)\cdot (1, y_1^2, \sqrt{2}y_1y_2, y_2^2, \sqrt{2}y_1, \sqrt{2}y_2)^\top\\\\
&=\phi(x)^\top\phi(y)
\end{aligned}$$
* **Gaussian (Radial Basis Function) Kernel**: $K(x, y)=\exp\left(-\frac{||x-y||^2}{\sigma^2}\right)$.

Similarly, for those SVMs applied with kernel trick, the classifier is $$f(x)=\omega^\top x+b=\sum\limits_{i=1}^N\alpha_iy_iK(x_i, x) +b,$$ and the Lagrangian Dual Problem is $$\begin{aligned}
&\max_\limits{\alpha} \sum\limits_{i=1}^N\alpha_i-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i, x_j), \\\\ 
&\text{s.t. } \sum\limits_{i=1}^N\alpha_iy_i=0, \alpha_i \in [0, C].
\end{aligned}$$
## Sequential Minimal Optimization (SMO) 
In SMO algorithm, each loop we just adjust 2 variables, say $\alpha_1, \alpha_2$. 
Dual problem: $$\begin{aligned}
W(\alpha)&=\sum_{i=1}^N-\frac{1}{2}\left(\alpha_1y_1\phi(x_1)+\alpha_2y_2\phi(x_2)+\sum_{i=3}^N{\alpha_iy_i\phi(x_i)}\right)^\top \left(\alpha_1y_1\phi(x_1)+\alpha_2y_2\phi(x_2)+\sum_{j=3}^N{\alpha_jy_j\phi(x_j)}\right)^\top\\\\
&=\alpha_1+\alpha_2+\sum_{i=3}^N\alpha_i-\frac{1}{2}\left(\alpha_1^2K_{11}+\alpha_2^2K_{22}+\sum_{i=3}^N{\sum_{j=3}^N{\alpha_i\alpha_jy_iy_jK_{ij}+2\alpha_1\alpha_2y_1u_2K_{12}+2\alpha_1y_1\sum_{i=3}^N{\alpha_iy_iK_{1i}+2\alpha_2y_2\sum_{i=3}^N{\alpha_iy_iK_{2i}}}}}\right)\\\\
&=-\frac{1}{2}\alpha_1^2K_{11}-\frac{1}{2}\alpha_2^2K_{22}-\alpha_1\alpha_2y_1y_2K_{12}-\alpha_1y_1\sum_{i=3}^N{\alpha_iy_iK_{1i}}-\alpha_2y_2\sum_{i=3}^N{\alpha_iy_iK_{2i}}+\alpha_1+\alpha_2+\cdots.\\\\
\end{aligned}$$
Note: "$\cdots$" means we just omit those terms irrelevant to $\alpha_1$ and $\alpha_2$.
Since $$\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^N\alpha_iy_i=\alpha_1^{old}y_1+\alpha_2^{old}y_2,$$ then we can get $$\alpha_1+\alpha_2y_1y_2=-y_1\sum_{i=3}^N\alpha_iy_i=\alpha_1^{old}+\alpha_2^{old}y_1y_2.$$
For simplicity, set $y_1y_2=s$ and $-y_1\sum_{i=3}^N\alpha_iy_i=\gamma$. Therefore, $$\alpha_1+\alpha_2s=\alpha_1^{old}+\alpha_2^{old}s=\gamma,$$ where $\gamma$ is a constant since all the $y_i$'s are known and $\left\\{\alpha_i\right\\}_{i=3}^N$ are fixed.

So $\alpha_1=\gamma-\alpha_2s$ and pluge this into dual problem. $$\begin{aligned}
W(\alpha)&=-\frac{1}{2}(\gamma-\alpha_2s)^2K_{11}-\frac{1}{2}\alpha_2^2K_{22}-(\gamma-\alpha_2s)\alpha_2sK_{12}-(\gamma-\alpha_2s)y_1\sum_{i=3}^N{\alpha_iy_iK_{1i}}-\alpha_2y_2\sum_{i=3}^N{\alpha_iy_iK_{2i}}+\gamma-\alpha_2s+\alpha_2+\cdots\\\\
&=-\frac{1}{2}(\alpha_2^2-2\gamma s\alpha)K_{11}-\frac{1}{2}\alpha_2^2K_{22}-\gamma sK_{12}\alpha_2+\alpha_2^2K_{12}+\alpha_2y_2\sum_{i=3}^N{\alpha_iy_iK_{1i}}-\alpha_2y_2\sum_{i=3}^N{\alpha_iy_iK_{2i}}+(1-s)\alpha_2+\cdots\\\\
&=-\frac{1}{2}\left(K_{11}+K_{22}-2K_{12}\right)\alpha_2^2+\left(\gamma s(K_{11}-K_{12})+1-s+y_2\sum_{i=3}^N{\alpha_iy_i(K_{1i}-K_{2i})}\right)\alpha_2+\cdots.
\end{aligned}$$

Moreover, $$\begin{aligned}
-\frac{1}{2}\left(K_{11}+K_{22}-2K_{12}\right)&=-\frac{1}{2}\left(\phi(x_1)^\top\phi(x_1)+\phi(x_2)^\top\phi(x_2)-2\phi(x_1)^\top\phi(x_2)\right)\\\\
&=-\frac{1}{2}||\phi(x_1)-\phi(x_2)||^2\\\\
&\leq 0,
\end{aligned}$$
so $W(\alpha)$ is convex upward and achieves to minimum at $$\alpha_2^{new, uncon}=\frac{\gamma s(K_{11}-K_{12})+1-s+y_2\sum_{i=3}^N{\alpha_iy_i(K_{1i}-K_{2i})}}{K_{11}+K_{22}-2K_{12}}$$ if there is no extra restrictions.


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
* **2.** Cristianini, N. and Shawe-Taylor, J. (2000), An introduction to Support Vector Machines: and other kernel-based learning methods, Cambridge University Press.
* **3.** Hastie, Trevor, Tibshirani, Robert, and Friedman, J. H. (2009). The elements of statistical learning: Data mining, inference, and prediction, 2nd Edition
* **4.** https://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf
* **5.** https://www.cnblogs.com/pinard/p/6056319.html
