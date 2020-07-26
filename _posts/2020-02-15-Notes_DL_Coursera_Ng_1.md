---
layout: post
title: Notes on Neural Network and Deep Learning
date: 2020-02-15 21:49:02
tags:
- Machine-Learning
---

**This is the 1st course of the [Deep Learning Specialization on Coursera by Andrew Ng ](https://www.coursera.org/specializations/deep-learning#courses)**
<!-- more -->

### Definitions
* $m$ : number of examples in the dataset
* $n_x$ : input size (number of input features)
* $n^{[l]}$ : number of hidden units in $l^\text{th}$ layer
* $X$ : $(n_x, m)$ input matrix
* $\hat{y}$ : $(1, m)$ predicted label vector
* $W^{[l]}$ : $(n^{[l]}, n^{[l-1]})$ weight matrix of the $l^\text{th}$ hidden layer
* $w^{[l]}_{jk}$ : weight from the $k^\text{th}$ neuron in the $[l-1]^\text{th}$ layer to the $j^\text{th}$ neuron in the $l^\text{th}$ neuron
* $b^{[l]}$ :  $(n^{[l]})$ bias vector of the $l^\text{th}$ hidden layer
* $A^{[l]}$ : $(n^{[l]}, m)$ activation matrix of the $l^\text{th}$ hidden layer
* $Z^{[l]}$ : $(n^{[l]}, m)$ linear product of the $l^\text{th}$ hidden layer
* $g^{[l]}$ : activation function of the $l^\text{th}$ hidden layer (e.g., `relu`, `sigmoid`)
* $J$ : cost function
* $\alpha$ : learning rate
* $\cdot$ : matrix multiplication
* $\odot$ : element-wise multiplication, hadamard product

### Forward Propagation
For the $l^\text{th}$ hidden layer, forward propagation is

$$
\begin{aligned}
  Z^{[l]} & = W^{[l]} \cdot A^{[l-1]} + b^{[l]} \quad \text{(linear combination)} \\
  A^{[l]} & = g^{[l]}(Z^{[l]}) \quad \text{(activation)}
\end{aligned}
$$

and the cost function is

$$
J = \frac{1}{m} \sum\limits_{i=1}^m L_i(y_i, \hat{y}_i)
$$

where $L_i$ is the $i$th loss function for one instance.

### Backward Propagation
In the backward propagation process, we care about how the change of the variables in the network affects the cost, i.e., to compute the derivative of the variables. At the last layer (layer $K$), the activation is essentially the output, i.e., $A^{[K]} = \hat{y}$ and its derivative is

$$
\begin{aligned}
  \mathrm{d} A^{[K]} = \frac{\mathrm{d} J}{\mathrm{d} \hat{y}},
  \end{aligned}
$$

where the notation $\mathrm{d} X$ represents the derivative of the cost function $J$ relative to $X$[^1].

Following the chain rule, for the $l^\text{th}$ hidden layer[^2], we can find the derivative for the linear product

$$
\mathrm{d} Z^{[l]} = \mathrm{d} A^{[l]} \odot \frac{\mathrm{d}g^{[l]}}{\mathrm{d}Z^{[l]}}.
$$

The hadamard product $\odot$ shows up because $g$ is a scalar function with element-wise mapping. Same procedure can be applied to obtain the derivative for the weight matrix and bias vector of the hidden layer

$$
\begin{aligned}
\mathrm{d} W^{[l]} & = \mathrm{d} Z^{[l]} \cdot (A^{[l-1]})^T, \quad \\
\mathrm{d} b^{[l]} & = \sum\limits_{i=1}^m \mathrm{d} Z^{[l]},
\end{aligned}
$$

The first equation above can be broke down to 

$$
\frac{\mathrm{d} J}{\mathrm{d} w^{[l]}_{ij}} = \sum\limits_{k=1}^m \frac{\mathrm{d} J}{\mathrm{d}z_{i k}^{[l]}} \times \frac{\mathrm{d}z_{ik}^{[l]}}{\mathrm{d}w_{ij}^{[l]}}= \sum\limits_{k=1}^m \frac{\mathrm{d} J}{\mathrm{d}z_{i k}^{[l]}} \times a^{[l-1]}_{jk} = \mathrm{d} Z^{[l]} \cdot (A^{[l-1]})^T
$$

The linear property of the dot product promises a nice concise from to compute the derivative. Then the activation of the next backward layer is computed as

$$
\mathrm{d} A^{[l-1]} = (W^{[l]})^T \cdot \mathrm{d} Z^{[l]}.
$$

### Gradient Descent

A simple gradient descent step is applied to update the weight matrix and the bias vector,

$$
\begin{aligned}
W^{[l]} & = W^{[l]} - \alpha \cdot \mathrm{d} W^{[l]} \\
b^{[l]} & = b^{[l]} - \alpha \cdot \mathrm{d} b^{[l]}
\end{aligned}
$$

This forms a complete process of the fundamental computation in a neural network.

[^1]: This definition is different from that in the course. There Prof. Ng actually defines $\mathrm{d} A^{[l]} = \frac{\mathrm{d} L}{\mathrm{d} A^{[l]}}$ and $\mathrm{d} W^{[l]} = \frac{\mathrm{d} J}{\mathrm{d} W^{[l]}}$, this causes a bit confusion.
[^2]: Following the last note, because of the different definiton here, the $1/m$ factor in equations calculating $dW$ and $db$ in the course won't show up here.


### References
- https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome