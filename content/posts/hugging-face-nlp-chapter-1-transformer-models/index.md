+++
date = '2024-05-10T00:00:00+08:00'
draft = false
title = 'Chapter1 Transformer Models'
description = 'NLP Course of Hugging Face'
aliases = ['/2024/05/10/NLP-Course-of-Hugging-Face/']
tags = ['NLP', 'Hugging Face']
categories = ['Learning']
language = 'en'
+++
> 最近打算入门 NLP，在自学 🤗 的 [NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)，但是感觉自己过于摆烂了。于是打算边学边做笔记，争取在期末之前把本课程学完

# Pipeline

- the most basic object in the 🤗 Transformers llibrary
- It can:
  - feature-extraction(get a vector representing the text)
  - fill-task
  - ner(entity recognition)
  - question-answering
  - sentiment-analysis
  - summarization
  - text-generation
  - translation
  - zero-shot classification

## Zero-shot Classification

- you can casually assign the labels

```python
classifier = pipeline("zero-shot-classification")
classifier(
   "I play Genshin Impact!",
   candidate_labels=["op","2-dimensional"]
)
```

## Text Generation

- involves randomness

```python
generator=pipeline("text-generation")
generator(
  "prompts",
  max_length=15, #the max length of the text
  num_return_sequence=3 #How many texts are gonna generated
)
```

## Mask filling

```python
unmasker=pipeline("fill-mask")
unmasker("I play <mask> Impact!",top_k=2) #top_k decides the time it does;<mask> depends on what model you are using
```

## Named entity recognition

```python
ner=pipeline("ner",grouped_entities=True)#'grouped_entities=True' is used to enable the model to put multi-words together
```

## Question answering

- answer a question using given context
- extracting answers from the context instead of generating answers

```python
question_answerer=pipeline("question-answering")
question_answerer(
  question="where do I work?",
  context="My name is Sylvain and I work at Hugging Face in Brooklyn"
)
```

## Summarization

```python
summarizer=pipeline("summarization")
summarizer(
   """
context
   """
)
```

## Translation

```python
translator=pipeline("translation",model="modelName")
translator("contex")
```

# Brief intro to Transformers

## General categorization

- GPT-like: auto-regressive
- BERT-like: auto-encoding
- BART/T5-like: sequence-to-sequence

## Transfer Learning

### Pretraning

- the act of traing a model from scratch  
  ![](https://github.com/daucloud/imagecdn/raw/main/test/202404251741650.png)

### Fine-tuning

- training on the top of pretrained models with a dataset specific to the target task  
  ![image.png](https://github.com/daucloud/imagecdn/raw/main/test/202404251746195.png)

## General architecture

- Transformers model is generally composed of two blocks:
  - Encoder: receives inputs and builds representations for them
  - Decoder: use the outputs of encoder to generate target outputs
- Various tasks requires different blocks:
  - Encoder-only: sentence classfication; NER
  - Decoder-only: text generation
  - Encoder-decoder models or sequence-to-sequence models: translation or summarization

## Attention layer

- pay specific attention to certain words while ignoring others more or less

## Original Architechture

- the encoder translate all the words
- the decoder is only allowed to translate by the past words; but later it can get all the outpus of encoer to better translate the word  
  ![image.png](https://github.com/daucloud/imagecdn/raw/main/test/202404271755969.png)
