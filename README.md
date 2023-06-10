# JAX Implementation of T5

[**[Research Report]**](https://github.com/ztjhz/t5-jax/blob/main/research-report.pdf) | [**[WandB Runs]**](https://wandb.ai/jinghua/t5-jax-fr-en-finetune?workspace=user-jinghua)

This project is an enhanced implementation of the [T5](https://arxiv.org/pdf/1910.10683.pdf) model using JAX. By adopting a functional approach and harnessing the capabilities of JAX, this implementation strives for superior performance and compatibility with Google Cloud TPU. Beyond the technical advancements, this project's motivation stems from the desire to establish a cleaner T5 codebase and to serve as a valuable educational resource for both AI researchers and engineers, facilitating their understanding and exploration of Transformer-based LLM architectures.

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

This project is inspired by [ayaka/bart-base-jax](https://github.com/ayaka14732/bart-base-jax/), while the code for this project is entirely written by myself.

- [Setup Instructions](#setup-instructions)
- [Usage examples](#usage-examples)
- [Discoveries](#discoveries)
- [Analysis](#analysis)
  - [1. JAX precision](#1-jax-precision)
  - [2. Layer normalisation](#2-layer-normalisation)
  - [3. Dropout](#3-dropout)
  - [4. Scaling QK matrices](#4-scaling-qk-matrices)
  - [5. Relative Attention Bias / Position embeddings](#5-relative-attention-bias--position-embeddings)
  - [6. Layer norm in T5 does not subtract mean](#6-layer-norm-in-t5-does-not-subtract-mean)
  - [7. T5 employs a final layer norm on the output of the encoder and decoder](#7-t5-employs-a-final-layer-norm-on-the-output-of-the-encoder-and-decoder)
  - [8. T5 uses tied word embeddings](#8-t5-uses-tied-word-embeddings)
  - [9. T5 also rescales the decoder output for tied word embedding in the language model head](#9-t5-also-rescales-the-decoder-output-for-tied-word-embedding-in-the-language-model-head)
- [T5 Jax Implementation Results](#t5-jax-implementation-results)
  - [Input and Output](#input-and-output)
  - [Time taken](#time-taken)
  - [Conclusion](#conclusion)
- [Fine-tuning](#fine-tuning)
  - [Dataset](#dataset)
  - [Results](#results)
  - [BLEU Score](#bleu-score)
  - [Task Performance](#task-performance)

## Setup Instructions

1. Install [jax](https://jax.readthedocs.io/en/latest/#installation)

   ```bash
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. Then install requirements.txt:

   ```bash
   pip install -r requirements.txt
   ```

## Usage examples

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

2. Initialize model parameters

   ```python
   from utils.params_utils import init_params_pretrained

   params = init_params_pretrained()
   ```

3. Encoder

   ```python
   from model.transformer_encoder import fwd_transformer_encoder

   encoder_output = fwd_transformer_encoder(
      encoder_params=params["encoder"],
      embedding_params=params["shared"],
      input_ids=input_ids,
   )
   ```

4. Decoder

   ```python
   from model.transformer_decoder import fwd_transformer_decoder

   decoder_start_token_id = model.config.decoder_start_token_id
   decoder_input_ids = (
      jnp.ones((encoder_input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
   )

   decoder_output = fwd_transformer_decoder(
      decoder_params=params["decoder"],
      embedding_params=params["shared"],
      decoder_input_ids=decoder_input_ids,
      encoder_output=encoder_output,
   )
   ```

5. Generate

   ```python
   from model.t5_generate import fwd_t5_generate
   from config import config

   sequences = fwd_t5_generate(
      params,
      encoder_input_ids=input_ids,
      eos_token_id=config.EOS_TOKEN_ID,
      decoder_start_token_id=config.DECODER_START_TOKEN_ID,
   )
   output = tokenizer.batch_decode(sequences, skip_special_tokens=True)
   ```

## Discoveries

I discovered an issue in the Hugging Face transformers FlaxT5. Their hidden states output were not consistent with my outputs.

I observed that my encoder and decoder block `11` `hidden state` does not align with their block `11` `hidden_state` even though my `hidden states` from block `0` to `10` aligns with the their `hidden states` from block `0` to `10`. Additionally, my `final hidden state` (after applying the layer norm) also aligns with their `final hidden state` after the layer norm.

I then raised an [issue](https://github.com/huggingface/transformers/issues/23960) and made a [PR](https://github.com/huggingface/transformers/pull/24027) to fix this issue.

## Analysis

### 1. JAX precision

1. On TPU, JAX defaults to using `bfloat16` for matrix multiplication even when the data type is specified as `float32`. While this may speed up training, some precision is lost.
2. When utilizing GPU, the Hugging Face transformers model exhibits distinct precision compared to JAX.

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

## T5 Jax Implementation Results

### Input and Output

| Input                                                                                                                                                                                          | Hugging Face Output                                                     | My Output                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| translate English to German: That is good.                                                                                                                                                     | 'Das ist gut so.'                                                       | 'Das ist gut so.'                                                       |
| cola sentence: The course is jumping well.                                                                                                                                                     | acceptable                                                              | acceptable                                                              |
| stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.                                                                                                       | 4.0                                                                     | 4.0                                                                     |
| summarize: In recent times, rapid advancements in technology have revolutionized various industries, enhancing efficiency, connectivity, and convenience for individuals and businesses alike. | rapid advancements in technology have revolutionized various industries | rapid advancements in technology have revolutionized various industries |

### Time taken

The inputs above are fed into the Hugging Face transformers model and my own model. Generation was repeated 100 times and here is the total time taken:

| Device | Hugging Face | Mine   | Speed Improvement |
| ------ | ------------ | ------ | ----------------- |
| GPU    | 190.63s      | 64.36s | 66.24% faster     |
| TPU    | 466.59s      | 42.31s | 90.93% faster     |

### Conclusion

In a direct comparison, my implementation achieves comparable results to Hugging Face's implementation, while also demonstrating superior performance in terms of speed.
Both implementations produced identical translations, acceptability scores, and summarization outputs in the provided examples.
However, my implementation outperforms Hugging Face's implementation, completing the tasks approximately 90.93% faster on TPU and 66.24% faster on GPU.

## Fine-tuning

Upon reading the original T5 paper, I discovered that it primarily focused on translating English to German, French, and Romanian. This sparked my curiosity about whether the model could also handle translating from French to English. To test this, I utilized the pre-trained model and applied a task prefix of "translate French to English: ". Unfortunately, the model proved incapable of performing the desired translation. Determined to overcome this limitation, I embarked on the journey of fine-tuning my own model specifically tailored for the task of French to English translation.

> For more in-depth information regarding my fine-tuning process, please read the [**research report**](https://github.com/ztjhz/t5-jax/blob/main/research-report.pdf) or visit the [GitHub branch](https://github.com/ztjhz/t5-jax/tree/train) or explore the [WandB runs](https://wandb.ai/jinghua/t5-jax-fr-en-finetune?workspace=user-jinghua). These resources provide additional insights into the details of my fine-tuning procedure.

### Dataset

To finetune my model, I utilized the [wmt-14](https://huggingface.co/datasets/wmt14) fr-en dataset, which consists of approximately 40 million data entries for the training set, and around 3,000 rows for the test and validation sets.

### Results

> For a comprehensive understanding and detailed analysis of my findings, I invite you to explore my [**research report**](https://github.com/ztjhz/t5-jax/blob/main/research-report.pdf).

Through rigorous experimentation with different factors such as initialising the language model head with embeddings, scaling the decoder output, task prefix, trying different Adafactor learning rates, and testing various optimisers, my results reveal that the optimal configuration comprises of using the Adafactor optimiser with a learning rate of 0.001, coupled with a scaled decoder output and embedding initialised `lm_head`.

### BLEU Score

| Optimiser                 | Steps  | Generation BLEU | One Pass BLEU |
| ------------------------- | ------ | --------------- | ------------- |
| Original (No fine-tuning) |        | 1.01            | 16.43         |
| Adafactor, 0.001, scale   | 20,480 | 29.96           | 34.89         |
|                           | 40,960 | 30.61           | 35.44         |
|                           | 61,440 | 30.64           | 35.50         |
|                           | 81,920 | **31.20**       | **36.12**     |

Prior to fine-tuning the model specifically for French to English translation, it only managed to secure a modest BLEU score of **16.43**. After fine-tuning the model with the best configuration (Adafactor optimizer, learning rate of 0.001, scaled decoder output, and embedding lm_head), the fine-tuned model achieved a BLEU score to **31.20**, almost twice of the non-fine-tuned model.

### Task Performance

| Task                                  | Input                                                                                                                                                                                            | Output                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| English to German                     | "translate english to german: That is good"                                                                                                                                                      | "That is good news."                                                                                                                                                                                                                                                                                                                                                                                  |
| Summarize                             | "summarize: In recent times, rapid advancements in technology have revolutionized various industries, enhancing efficiency, connectivity, and convenience for individuals and businesses alike." | "Rapid advances in technology have revolutionized various industries recently, increasing efficiency, connectivity, and convenience for individuals and businesses alike."                                                                                                                                                                                                                            |
| Semantic Textual Similarity Benchmark | "stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field."                                                                                                       | "sentence1: The rhinograzed on the grass. sentence2: A rhino is grazing in a field. sentence3: A rhino is grazing in a field. sentence4: A rhino is grazing in a field. sentence5: A rhino is grazing in a field. sentence6: A rhino is grazing in a field. sentence7: A rhino is grazing in a field. sentence8: A rhino is grazing in a field. sentence8: A rhino is grazing in a field. sentence8:" |
| Corpus of Linguistic Acceptability    | "cola sentence: The course is jumping well."                                                                                                                                                     | "The course is jumping well. The course is jumping well."                                                                                                                                                                                                                                                                                                                                             |
| French to English                     | "translate french to english: Une stratégie"                                                                                                                                                     | "A Strategy"                                                                                                                                                                                                                                                                                                                                                                                          |
|                                       | "translate french to english: Cette année, je pense que c'est la bonne."                                                                                                                         | "This year I think it's nice."                                                                                                                                                                                                                                                                                                                                                                        |
|                                       | "translate french to english: L'effet de la vitamine D sur le cancer n'est pas clairement établi non plus."                                                                                      | "Vitamin D's effect on cancer is not clear either."                                                                                                                                                                                                                                                                                                                                                   |
|                                       | "translate french to english: Une boîte noire dans votre voiture?"                                                                                                                               | "Black box in your car?"                                                                                                                                                                                                                                                                                                                                                                              |
|                                       | "translate french to english: Le sportif Jhonathan Florez a sauté jeudi d'un hélicoptère au-dessus de Bogota, la capitale colombienne."                                                          | "Jhonathan Florez crashed helicopter over Bogotá City Thursday night. He survived injuries sustained by teammates from teammates from Bogotá City."                                                                                                                                                                                                                                                   |
|                                       | "translate french to english: j'aime manger du riz au poulet le matin."                                                                                                                          | "I like eating rice chicken morning."                                                                                                                                                                                                                                                                                                                                                                 |

As seen from the table, the model perform quite well on French to English translation, but it fails to perform original tasks well. This demonstrates a striking example of catastrophic forgetting in machine learning, a predicament that affects not only the original tasks - including translation from English to German, summarization, STS-B, and CoLA - but also persists even in models fine-tuned for a small number of steps such as 20,480, 40,960, 61,440, and 81,920. Despite these careful, incremental adjustments, the models continue to exhibit catastrophic forgetting, underlining the challenge of maintaining the proficiency of AI models in their originally trained tasks while integrating new knowledge.
