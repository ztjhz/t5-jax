# JAX Implementation of T5

This project is a JAX implementation of the [T5](https://arxiv.org/pdf/1910.10683.pdf) model.

- [JAX Implementation of T5](#jax-implementation-of-t5)
  - [Setup Instructions](#setup-instructions)
  - [Examples](#examples)
  - [Analysis](#analysis)
    - [1. Jax precision on TPU is low by default](#1-jax-precision-on-tpu-is-low-by-default)
    - [2. Layer normalisation](#2-layer-normalisation)
    - [3. Dropout](#3-dropout)
    - [4. Scaling QK matrices](#4-scaling-qk-matrices)
    - [5. Relative Attention Bias / Position embeddings](#5-relative-attention-bias--position-embeddings)
    - [6. Layer norm in T5 does not subtract mean](#6-layer-norm-in-t5-does-not-subtract-mean)
    - [7. T5 employs a final layer norm on the output of the encoder and decoder](#7-t5-employs-a-final-layer-norm-on-the-output-of-the-encoder-and-decoder)
    - [8. T5 uses tied word embeddings](#8-t5-uses-tied-word-embeddings)
    - [9. T5 also rescales the decoder output for tied word embedding in the language model head](#9-t5-also-rescales-the-decoder-output-for-tied-word-embedding-in-the-language-model-head)
  - [Results](#results)
    - [Input and Output](#input-and-output)
    - [Time taken](#time-taken)
    - [Conclusion](#conclusion)

## Setup Instructions

1. Install [jax](https://jax.readthedocs.io/en/latest/#installation)

   ```bash
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. Then install requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Examples

1. Tokenize inputs

   ```python
   from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

   model = FlaxT5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base")
   tokenizer = AutoTokenizer.from_pretrained("t5-base")
   inputs = tokenizer(
      ["summarize: My friends are cool but they eat too many carbs."], return_tensors="np"
   )
   input_ids = inputs["input_ids"]
   ```

2. Encoder

   ```python
   from model.transformer_encoder import fwd_transformer_encoder

   encoder_output = fwd_transformer_encoder(
      encoder_params=model.params["encoder"],
      embedding_params=model.params["shared"],
      input_ids=input_ids,
   )
   ```

3. Decoder

   ```python
   from model.transformer_decoder import fwd_transformer_decoder

   decoder_start_token_id = model.config.decoder_start_token_id
   decoder_input_ids = (
      jnp.ones((encoder_input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
   )

   decoder_output = fwd_transformer_decoder(
      decoder_params=model.params["decoder"],
      embedding_params=model.params["shared"],
      decoder_input_ids=decoder_input_ids,
      encoder_output=encoder_output,
   )
   ```

4. Generate

   ```python
   from model.t5_generate import fwd_t5_generate

   sequences = fwd_t5_generate(
      model.params,
      encoder_input_ids=input_ids,
      eos_token_id=model.config.eos_token_id,
      decoder_start_token_id=model.config.decoder_start_token_id,
   )
   output = tokenizer.batch_decode(sequences, skip_special_tokens=True)
   ```

## Analysis

### 1. Jax precision on TPU is low by default

By default, jax uses `bfloat16` on TPU, even when the data type is float32.

### 2. Layer normalisation

T5 performs pre-layer norm instead of post-layer norm.

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

### 5. Relative Attention Bias / Position embeddings

T5's position embeddings (relative attention bias) is different from [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155). _([Hugging Face's implementation](<(https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/t5/modeling_flax_t5.py#L268)>))_

1. Uses binned relative attention bias to reduce time complexity for long sequences
2. Only applies the bias before $\text{softmax}$

> It is not mentioned in the T5 paper that they only apply the bias before the $\text{softmax}$

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

T5's layer norm does not subtract the mean ($\mu$) and does not have a bias ($\beta$). They utilise [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). _([HuggingFace's implementation](https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/t5/modeling_flax_t5.py#L71))_

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

### 9. T5 also rescales the decoder output for tied word embedding in the language model head

> The rescaling of decoder output before passing it into the lm_head is not mentioned in the T5 paper

However, their [T5](https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586) implementation scales the decoder output.

$$
\mathrm{lm\_head}(x) = \frac{x}{\sqrt{d_{\text{model}}}}W_e \textrm{\quad instead of\quad} \mathrm{lm\_head}(x) = xW_e
$$

$$
y = \text{Softmax}(\mathrm{lm\_head}(x))
$$

Where:

- $x$ is the decoder output.
- $y$ is the logits.
- $d_{\text{model}}$ is the dimensionality of the model.
- $W_e$ is the input embeddings used for tie word embeddings.
- $\mathrm{lm\_head}$ is the input embeddings used for tie word embeddings.

## Results

### Input and Output

| Input                                                                                                                                                                                          | Hugging Face Output                                                     | My Output                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| translate English to German: That is good.                                                                                                                                                     | 'Das ist gut so.'                                                       | 'Das ist gut so.'                                                       |
| cola sentence: The course is jumping well.                                                                                                                                                     | acceptable                                                              | acceptable                                                              |
| stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.                                                                                                       | 4.0                                                                     | 4.0                                                                     |
| summarize: In recent times, rapid advancements in technology have revolutionized various industries, enhancing efficiency, connectivity, and convenience for individuals and businesses alike. | rapid advancements in technology have revolutionized various industries | rapid advancements in technology have revolutionized various industries |

### Time taken

| Hugging Face | Mine   | Speed Improvement |
| ------------ | ------ | ----------------- |
| 16.14s       | 13.14s | 18.57% faster     |

### Conclusion

In a direct comparison, my implementation achieves comparable results to Hugging Face's implementation, while also demonstrating superior performance in terms of speed.
Both implementations produced identical translations, acceptability scores, and summarization outputs in the provided examples.
However, my implementation outperforms Hugging Face's implementation, completing the tasks approximately 18.57% faster.
