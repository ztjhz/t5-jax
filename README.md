# JAX Implementation of T5

This project is a JAX implementation of the [T5](https://arxiv.org/pdf/1910.10683.pdf) model.

- [JAX Implementation of T5](#jax-implementation-of-t5)
  - [Setup Instructions](#setup-instructions)
  - [Analysis](#analysis)
    - [1. Jax precision on TPU is low by default](#1-jax-precision-on-tpu-is-low-by-default)
    - [2. Layer normalisation](#2-layer-normalisation)
    - [3. Dropout](#3-dropout)
    - [4. Scaling QK matrices](#4-scaling-qk-matrices)
    - [5. Relative Attention Bias / Relative Position Bias](#5-relative-attention-bias--relative-position-bias)
    - [6. Layer norm in T5 does not subtract mean](#6-layer-norm-in-t5-does-not-subtract-mean)
    - [7. T5 employs a final layer norm on the output of the encoder and decoder](#7-t5-employs-a-final-layer-norm-on-the-output-of-the-encoder-and-decoder)
    - [8. T5 uses tied word embeddings](#8-t5-uses-tied-word-embeddings)

## Setup Instructions

1. Install [jax](https://jax.readthedocs.io/en/latest/#installation)

   ```bash
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. Then install requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Analysis

### 1. Jax precision on TPU is low by default

By default, jax uses `bfloat16` on TPU, even when the data type is float32.

### 2. Layer normalisation

[Hugging Face T5](https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/t5/modeling_flax_t5.py#L488) performs pre-layer norm instead of post-layer norm.

Attention:

- `(layer norm -> self attention -> dropout -> add)` instead of
- `(self-attention -> dropout -> add -> layer norm)`

Feed foward:

- `(layer norm -> densereludense -> dropout -> add)` instead of
- `(densereludense -> dropout -> add -> layernorm)`

### 3. Dropout

- drop out performed once at the end in ff `(linear -> linear -> dropout)` instead of twice after each linear layer `(linear -> dropout -> linear -> dropout)`

### 4. Scaling QK matrices

[Hugging Face T5](https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/t5/modeling_flax_t5.py#L436) does not scale the QK matrices

> The T5 paper did not mention the exclusion of QK matrix scaling.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(QK^T\right)V
$$

instead of

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 5. Relative Attention Bias / Relative Position Bias

[Hugging Face Flax T5](https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/t5/modeling_flax_t5.py#L268) relative attention bias (relative position bias) is different from [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155).

1. Uses binned relative attention bias to reduce time complexity for long sequences
2. Only applies the bias before $\text{softmax}$

> The T5 paper did not mention how relative position bias was used.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(QK^T + X\right)V
$$

instead of

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + X}{\sqrt{d_k}}\right)V + Y
$$

Where:

- $Q$ is the query matrix
- $K$ is the key matrix
- $V$ is the value matrix
- $d_k$ is the dimension of the keys

In the case of multi-head attention, the above process is performed multiple times with different learned linear transformations of the original \(Q\), \(K\), and \(V\). If we have \(h\) heads, then we have:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
$$

where each head is defined as:

$$
\text{head}_i = \text{Attention}(QW_{Qi}, KW_{Ki}, VW_{Vi})
$$

### 6. Layer norm in T5 does not subtract mean

**Layer Norm Definition**

Given an input $x \in \mathbb{R}^{d}$, the output of layer normalization $y \in \mathbb{R}^{d}$ is calculated as:

$$
y = \gamma \left( \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta
$$

Where:

- $\mu$ is the mean of the input $x$: $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$
- $\sigma^2$ is the variance of the input $x$: $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$
- $\gamma$ and $\beta$ are learnable parameters (the weight and bias), which have the same dimension as $x$.
- $\epsilon$ is a small number for numerical stability, typically on the order of $10^{-5}$ to $10^{-8}$.

**T5 Layer Norm**

[HuggingFace T5 Layer Norm](https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/t5/modeling_flax_t5.py#L71) does not subtract the mean ($\mu$) and does not have a bias ($\beta$). They utilise [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).

> The T5 paper did not mention that they used Root Mean Square Layer Normalization

Root mean Square Layer Normalization Formula:

$$
\bar{a_i}=\frac{a_i}{RMS(a)}g_i\textrm{, where }RMS(a)=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a_i^2}
$$

Where:

- $g_i$ is the gain (weight) parameter
- $a_i$ is the inputs
- $\bar{a_i}$ is the scaled values of the inputs
- $RMS(a)$ is the root mean square $a$.

### 7. T5 employs a final layer norm on the output of the encoder and decoder

In the original transformer model proposed by Vaswani et al., 2017, there is no final layer normalization on the outputs of the encoder and decoder. The outputs of these components are fed directly into subsequent operations.

In the T5 model, there is a final layer normalization step after the output from both the encoder and decoder.

### 8. T5 uses tied word embeddings

T5 uses `tied word embeddings`, which is layered upon the output of the final decoder. This differs from the conventional Transformer architecture, which uses a linear layer for the language model head (`lm_head`).

However, for T5 during training, the `lm_head` is the transpose of the word embedding. This reduces the number of trainable parameters in the model by sharing the same embeddings for the input and output layers. This not only decreases the computational load, but also helps in regularizing the model, leading to an improved generalization ability and potentially better performance on unseen data.

> The output of the final decoder block is fed into a dense layer with a softmax output, whose weights are shared with the input embedding matrix.
