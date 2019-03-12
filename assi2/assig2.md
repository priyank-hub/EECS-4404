



<h1 style="text-align:center">EECS 4404</h1>

<h2 style="text-align:center">Assignment 2</h2>

























<ul>
    <h4 style="text-align:center">Name: Bochao Wang</h4>
    <h4 style="text-align:center">Student ID: 215237902</h4>
    <h4 style="text-align:center">Prism: bochao</h4>
    <h4 style="text-align:center">Date: Feb. 20th</h4>
</ul>

​    







**1. Gradient computation **

​	Let $X \in \reals ^{d * d}$ be some matrix. Consider the function $f:\ \Reals^d \rightarrow \Reals$ defined by $f(w) = w^T X w$, show that 	the gradient of this function with repect to $w$ is $\bigtriangledown f(w) = w^T(X+X^T)$

- Proof:
  - $\bigtriangledown f(w) = \begin{pmatrix}\frac{\partial f}{\partial w_1},&\frac{\partial f}{\partial w_2},&\ldots &\frac{\partial f}{\partial w_d}\end{pmatrix}​$ 
  - $ f(w)= \begin{bmatrix}w_1&w_2&w_3&\ldots &w_d\end{bmatrix}\begin{bmatrix}x_{11}&x_{12}&x_{13}&\ldots &x_{1d}\\\\x_{21}&x_{22}&x_{23}&\ldots &x_{2d}\\\\x_{31}&x_{32}&x_{33}&\ldots &x_{3d}\\\\ \vdots &&&& \vdots\\\\ x_{d1}&x_{d2}&x_{d3}&\ldots &x_{dd} \end{bmatrix} \begin{bmatrix}w_1\\\\w_2\\\\w_3 \\\\ \vdots \\\\ w_d\end{bmatrix}$ 
  - $f(w)=\begin{bmatrix}\sum_{i=1}^d w_{i}x_{i1}, & \sum_{i=1}^d w_{i}x_{i2}, & \ldots & \sum_{i=1}^d w_{i}x_{id}\end{bmatrix} w= \sum_{j=1}^d \sum_{i=1}^d w_{i} x_{ij} w_{j}$ 
  - Simplify the function $f(w) = \sum_{j=1}^d (x_{jj}w_j^2+ \sum_{i \not=j}w_{i} x_{ij} w_{j})$
  - we can compute the partial derivative with element $k$, $\frac{\partial}{\partial w_k}[\sum_{j=1}^d (x_{jj}w_j^2+ \sum_{i \not=j}w_{i} x_{ij} w_{j})] =\sum_{j=1}^d (\frac{\partial}{\partial w_k}x_{jj}w_j^2+ \sum_{i \not=j}\frac{\partial}{\partial w_k}w_{i} x_{ij} w_{j}) = 2x_{kk}w_{kk} + \sum_{i \not=k} x_{ki}w_{i}+\sum_{i \not=k}w_ix_{ik}​$
  - $\because x_{kk}w_{kk} =x_{kk}w_{kk}$ Then, $\frac{\partial}{\partial w_k}[\sum_{j=1}^d (x_{jj}w_j^2+ \sum_{i \not=j}w_{i} x_{ij} w_{j})]=\sum_{i=1}^d x_{ki}w_{i}+\sum_{i=1}^dw_ix_{ik}$
  - $\bigtriangledown f(w)=\begin{bmatrix} \sum_{i=1}^d x_{1i}w_{i}+\sum_{i=1}^dw_ix_{i1}, &\sum_{i=1}^d x_{2i}w_{i}+\sum_{i=1}^dw_ix_{i2}, & \ldots &\sum_{i=1}^d x_{di}w_{i}+\sum_{i=1}^dw_ix_{id}\end{bmatrix}$ 
  - $\bigtriangledown f(w)=\begin{bmatrix} \sum_{i=1}^dw_ix_{i1}, &\sum_{i=1}^dw_ix_{i2}, & \ldots &\sum_{i=1}^dw_ix_{id}\end{bmatrix}+\begin{bmatrix} \sum_{i=1}^d x_{1i}w_{i}, &\sum_{i=1}^d x_{2i}w_{i}, & \ldots &\sum_{i=1}^d x_{di}w_{i}\end{bmatrix}​$ 
  - $\begin{bmatrix} \sum_{i=1}^dw_ix_{i1}, &\sum_{i=1}^dw_ix_{i2}, & \ldots &\sum_{i=1}^dw_ix_{id}\end{bmatrix}=w^TX$
  - $\begin{bmatrix} \sum_{i=1}^d x_{1i}w_{i}, &\sum_{i=1}^d x_{2i}w_{i}, & \ldots &\sum_{i=1}^d x_{di}w_{i}\end{bmatrix} = w^TX^T$ 
  - Thus, $\bigtriangledown f(w) = w^TX+w^TX^T=w^T(X+X^T) $



**2. Stochastic Gradient Descent**

Recall the logistic loos function, point wise defined as

​		$\mathcal{l}^{logist} (y_{\pmb{w}},(\pmb{x}, t))=ln(1+exp(-t<\pmb{w},\pmb{x}>))​$

We know that empirical logistic loss over a dataset

​		$\mathcal{L}^{logist} (y_{\pmb{w}})=\frac{1}{N} \sum_{n=1}^Nln(1+exp(-t_n<\pmb{w},\pmb{x}_n>))​$

(a) Compute the gradient with respect to $w​$ of the logistic loss $\mathcal{l}^{logist} (y_{\pmb{w}},(\pmb{x}, t))​$ on a data point $(x, t)​$.

- Solve:
  - $\mathcal{l}^{logist} (y_{\pmb{w}},(\pmb{x}, t))=ln(1+exp(-t * \sum_{i}^N w_ix_i))​$
  - $\bigtriangledown f(\pmb{w}) = \begin{pmatrix}\frac{\partial f}{\partial w_1},&\frac{\partial f}{\partial w_2},&\ldots &\frac{\partial f}{\partial w_d}\end{pmatrix}​$ 
  - $\bigtriangledown \mathcal{l}^{logist} (y_{\pmb{w}},(\pmb{x}, t))=\begin{pmatrix}\frac{1}{1+exp(-t * \sum_{i}^N w_ix_i)} \frac{\partial(1+exp(-t * \sum_{i}^N w_ix_i))}{\partial w_1}, && \frac{1}{1+exp(-t * \sum_{i}^N w_ix_i)} \frac{\partial(1+exp(-t * \sum_{i}^N w_ix_i))}{\partial w_2}, && \ldots, && \frac{1}{1+exp(-t * \sum_{i}^N w_ix_i)} \frac{\partial(1+exp(-t * \sum_{i}^N w_ix_i))}{\partial w_d}\end{pmatrix}​$
  - we compute the $k​$ element:
  - $\frac{\partial(1+exp(-t  \sum_{i}^N w_ix_i))}{\partial w_k} = (-t  \sum_{i}^N w_ix_i)  \frac{\partial(-t  \sum_{i}^N w_ix_i)}{\partial w_k} = (-t  \sum_{i}^N w_ix_i)  (-tx_k)= (-t <\pmb{w}, \pmb{x}>)(-tx_k) $ 
  - $\bigtriangledown \mathcal{l}^{logist} (y_{\pmb{w}},(\pmb{x}, t))=\begin{pmatrix}\frac{1}{1+exp(-t * \sum_{i}^N w_ix_i)} (-t <\pmb{w}, \pmb{x}>)(-tx_1), && \frac{1}{1+exp(-t * \sum_{i}^N w_ix_i)} (-t <\pmb{w}, \pmb{x}>)(-tx_2), && \ldots, && \frac{1}{1+exp(-t * \sum_{i}^N w_ix_i)} (-t <\pmb{w}, \pmb{x}>)(-tx_d)\end{pmatrix}$
  - $\bigtriangledown \mathcal{l}^{logist} (y_{\pmb{w}},(\pmb{x}, t))=\begin{pmatrix}\frac{1}{1+exp(-t <\pmb{w}, \pmb{x}>)} (-t <\pmb{w}, \pmb{x}>)(-tx_1), && \frac{1}{1+exp(-t <\pmb{w}, \pmb{x}>)} (-t <\pmb{w}, \pmb{x}>)(-tx_2), && \ldots, && \frac{1}{1+exp(-t <\pmb{w}, \pmb{x}>)} (-t <\pmb{w}, \pmb{x}>)(-tx_d)\end{pmatrix}$
  - $\bigtriangledown \mathcal{l}^{logist} (y_{\pmb{w}},(\pmb{x}, t))=\frac{1}{1+exp(-t <\pmb{w}, \pmb{x}>)} (-t <\pmb{w}, \pmb{x}>)(-t\pmb{x})$



(b) Describe Stochastic Gradient Descent with a fixed stepsize η with respect to the logistic loss

- Solve:

  - ```pseudocode
    In: data D = ((x_1, t_1), (x_2, t_2), ..., (x_N, t_N))
    Parameters: T, lambda, n
    Initialize: theta_0 = 0
    For j = 0, ..., T
    	Set w_j = (1/(lambda * j)) * theta_j
    	Choose a random index i from {1, ..., N}
    	if
    ```

  - 





 