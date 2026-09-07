+++
date = '2024-05-14T00:00:00+08:00'
draft = false
title = 'Chapter2 Using 🤗 Transformers'
description = 'NLP Course of Hugging Face'
aliases = ['/2024/05/14/NLP-Course-of-Hugging-Face/']
tags = ['NLP', 'Hugging Face']
categories = ['Learning']
language = 'en'
+++

# Behind the pipeline

`pipeline()` groups preprocessing, model inference, and postprocessing together.

![Pipeline overview](https://cdn.jsdelivr.net/gh/Daucloud/imagecdn/test/202404291430930.png)

## Preprocessing with a tokenizer

A tokenizer converts raw text into vectors by:

1. splitting the input into tokens, such as words, subwords, and punctuation;
2. mapping each token to an integer;
3. adding any extra inputs the model needs.

```python
from transformers import AutoTokenizer

checkpoint="checkoutName"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)# all the preprocessing needs to be done exactly the same way as when the model was pretrained
raw_input="context"
inputs=tokenizer(raw_inputs,padding=True,truncation=True,return_tensors="pt")

# as for the return_tensors, pt: Pytorch, tf: TensorFlow, np: Numpy, jax: JAX
print(inputs)

### the results, is a dictionary, which contains two key-value pair
{
 "input_ids": tensor([1,1,1]), # the unique identifiers for each token
 "attention_mask": tensor([2,2,2])
}
```

## Going through the model

The model converts input IDs into logits.

![Model output](https://raw.githubusercontent.com/Daucloud/imagecdn/main/test/202405140923963.png)

🤗 provides the `AutoModel` class, which corresponds to the hidden-states step.

```python
from transformers import AutoModel
checkout="checkoutname"
model=AutoModel.from_pretrained(checkout)
outputs=(**inputs)

# the Automodel converts the inputs(seen last part) into a three-dimensional vector:
"""
1. batch size: the number of sequences processed at a time
2. sequence length
3. Hidden size: usually very large(768, 3072, even more)
"""
print(outputs.last_hidden_state.size) # torch.Size([2,16,768])
```

There are also task-specific architectures. You can understand them as `AutoModel` followed by a task head:

- `ForCausalLM`
- `ForMaskedLM`
- `ForMultipleChoice`
- `ForQuestionAnswering`
- `ForSequenceClassification`
- `ForTokenClassification`
- other 🤗 task heads

```python
from transformers import AutoModelForSequenceClassification
checkout="checkoutName"
model=AutoModelForSequenceClassification.from_pretrained(checkout)
outputs=(**inputs)
print(outputs.logits.shape) #torch.Size([2,2]), the size is much smaller than the results of AutoModel
```

## Postprocessing the output

Postprocessing converts logits into human-readable predictions, usually by turning raw unnormalized scores into probabilities.

Usually we use a softmax layer for this step.

```python
import torch
predictions=torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
'''
tensor([[5.0e-2,9.5e-1],[1.0e-1,9.0-1]],grad_fn=<SoftmaxBackward>)
'''

# we can inspect the id2label the following way:
model.config.id2label # {0:'NEGATIVE',1:'POSITIVE'}
```

# Models

`AutoClass` and its relatives are wrappers that can infer the architecture from the checkpoint.

## Creating A Transformer

```python
from transformers import BertConfig, BertModel,
config=BertConfig()
model=BertModel(config)# in this way, you will get a randomly initialized bert model, which will output gibberish
model=BertModel.from_pretrained("bert-base-cased")# the model is instantiate with the checkpoint trained by the bert team, which is less time-consuming and more environment-friendly; you can also replace the BertModel with AutoClass. Actually, it is more suggested to use AutoModel rather than a specific model, which will also work even if the architechture is different
```

Once you use the checkpoint, the weights are downloaded and cached to `~/.cache/huggingface/transformers`.

You can use `save_pretrained` to save the model to disk:

```python
model.save_pretrained("path/to/directory")
!ls path/to/directory
"""
config.json pytorch_model.bin

# the two files go hand in hand, the `config.json` contains the attributes necessary to build the architechture, and the `pytorch_model.bin` contains the weights(checkpoints) which are the parameters of your model
"""
```

# Tokenizers

## Algorithms for tokenization

### Word-Based

1. Split words according to marks such as spaces and punctuation.
2. Map each word to an ID. The ID is determined through the vocabulary, which is usually very large. For example, an English vocabulary may be as large as 500,000.
3. Words outside the vocabulary are often represented by an unknown token such as `[UNK]` or `<unk>`. This loses information, so it is sensible to avoid unknown tokens as much as possible.

### Character-Based

Split sentences by characters.

> This method will definitely reduce the number of unknown tokens. However, it may also make sequences too long and the results less meaningful.

### Subword Tokenization

Represent rare words as combinations of frequently used subwords.

> This saves vocabulary space while preserving semantic meaning as much as possible, which is especially useful for agglutinative languages such as Turkish.

Examples:

- Byte-level BPE: GPT-2
- WordPiece: BERT
- SentencePiece or Unigram: multilingual models

## Loading and Saving

Loading and saving tokenizers is nearly the same as loading and saving models: use `from_pretrained` and `save_pretrained`.

- Cached algorithms are similar to model architectures.
- Cached vocabularies are similar to model weights.

## Encoding

There are two steps when encoding text.

### Tokenization

```python
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("modelName")
tokens=tokenizer.tokenize("sequence")

print(tokens) # print the results of spilting
```

### From tokens to input IDs

Models only accept tensors as inputs, so tokens must be mapped into numbers. The vocabulary is the dictionary that maps tokens to IDs.

# Handling Multiple Sequences

## Models expect a batch of inputs

 

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model=AutoModelForSequenceClassification.from_pretrained("checkpointName")
tokenizer=AutoTokenizer.from_pretrained("checkpointName")

tokens=tokenizer.tokenize("raw inputs")
token_ids=tokenizer.convert_tokens_to_ids(tokens)
input_ids=torch.tensor(token_ids)

model(input_ids)
"""
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
"""
model([input_ids]) # this line will succeed
```

## Padding the inputs

Batches must contain sequences of the same length because tensors are rectangular. This is why we introduce `padding_id` to pad inputs. You can access it with the tokenizer's `pad_token_id`.

## Attention masks

One key feature of Transformers is the attention layer that contextualizes each token. As a consequence, padding IDs can also affect the output, which is not expected.

Attention masks solve this. A value of `0` means the corresponding token should be ignored; `1` means it should be used.

```python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

## Longer sequences

All Transformer models have a sequence length limit. If you want to pass a sequence longer than the limit, you should **truncate** your sequence or switch to a model that allows longer inputs.

# Putting it all together

## Put the tokenization steps together

As a review, there are three steps we need to tokenize raw inputs:

```text
tokenizer.tokenize() -> tokenizer.convert_tokens_to_ids() -> torch.tensor()
```

For convenience, `🤗 transformers` provides a high-level function that puts all these steps together: `tokenizer()` itself.

```python
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("checkpointName")

# 1
sequence="1"
input1=tokenizer(sequence) # valid

# 2
sequences=["1","2"]
input2=tokenizer(sequneces)

# 3 pad
input3=tokenizer(sequences, padding="longest") # pad to the maximum sequece length
input4=tokenizer(sequences, padding="max_length") # pad to the model maximum length
input5=tokenizer(sequences, padding="max_length", max_length=8) # pad to the specified maximum length

# 4 truncate
input6=tokenizer(sequences, truncation=True) # truncate the sequences longer than the model limit
input7=tokenizer(sequences, max_length=8, truncation=True) # truncate the sequences longer than the specified length

# 5 switch the type of tensors returned
input8=tokenizer(sequences, padding=True, return_tensors="pt") # pt stands for the Pytorch, tf for TensorFlow, np for NumPy
```

## Special words

Some models add special tokens such as `[CLS]` at the beginning; some add `[SEP]` at the end; some add both; and some add none.

## Wrapping up: From tokenizer to model

```python
import torch
from transfomers import AutoTokenizer, AutoModel

tokenizer=Autokenizer.form_pretrained("checkpointName")
model=AutoModel.form_pretrained("checkpointName")

sequence="1"
Input=tokenizer(sequence)
Output=model(**Input)
```
