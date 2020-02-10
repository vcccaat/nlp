Table of Contents
=================

  \* [Table of Contents](#table-of-contents)

  \* [Data Preparation](#data-preparation)

  \* [Seq2Seq with Attention](#seq2seq-with-attention)

   \* [Introduction](#introduction)

   \* [Network architecture](#network-architecture)

  \* [Pointer-generator](#pointer-generator)

   \* [Introduction](#introduction-1)

   \* [Network architecture](#network-architecture-1)

​     \* [Copy distribution](#copy-distribution)

​     \* [Coverage mechanism](#coverage-mechanism)

   \* [Implementation](#implementation)

   \* [Model Evaluation](#model-evaluation)

   \* [Result for financial dataset](#result-for-financial-dataset)

  \* [BERT](#bert)

  \* [Sentiment Analysis](#sentiment-analysis)

<br><br>

# Data Preparation

* **Datasets**: 
  * Non-financial
    * [CNN and Daily Mail](https://github.com/vcccaat/cnn-dailymail)
    * Yelp Review Dataset
  * Financial
    * [Reuters dataset (100k news) ](https://github.com/duynht/financial-news-dataset)
* **Packages used**:
  * nltk 
  * keras
  * pandas
  * sklearn
  * numpy
  * corenlp-stanford
* **Data Preprocessing Pipeline for Reuters Dataset**:
  * tokenize 
  * remove stop words
  * ... 

<br><br>

# Seq2Seq with Attention

## Introduction 

Encoder contains the input words that want to be transformed (translate, generate summary), and each word is a vector that go through forward and backward activation with bi-directional RNN. Then calculate the attention value for each words in encoder reflects its importance in a sentence. Decoder generates the output word one at a time, by taking dot product of the feature vector and their corresponding attention for each timestamp. 

![image-20200210110013495](readme.assets/image-20200210110013495.png) 

<br><br>

## Network architecture

![image-20200207155242372](readme.assets/image-20200207155242372.png) 

* **Encoder**: Bi-directional RNN, feature vector `a` at timestamp `t` is the concatenation of forward RNN and backward RNN 

  ![image-20200210110118887](readme.assets/image-20200210110118887.png)  

  <br><br>

* **Attention**: ![img](readme.assets/clip_image002-1065144.png): the amount of attention ![img](readme.assets/clip_image002-1062674.png) should pay to ![img](readme.assets/clip_image002-1062690.png)

  * Done by a neural network takes previous word ![img](readme.assets/clip_image002-1062718.png) in the decoder and ![img](readme.assets/clip_image002-1062690.png) in the encoder generate ![img](readme.assets/clip_image002-1062765.png) go through softmax to generate ![img](readme.assets/clip_image002-1062650.png)

  

  ​         ![image-20200206104243639](readme.assets/image-20200206104243639.png) ![image-20200207180539638](readme.assets/image-20200207180539638.png)   

  * additive attention for neural network: 

    ![image-20200210121315826](readme.assets/image-20200210121315826.png)  

  * simplier ways can choose dot-product attention:

    ![image-20200210110149495](readme.assets/image-20200210110149495.png)  

    <br><br>

* **Decoder**: RNN of dot product between attention and activation

  ![image-20200210121232055](readme.assets/image-20200210121232055.png)  

<br><br><br><br>

# Pointer-generator 

## Introduction

Abstrative text summarization requires sequence-to-sequence models, these models have two shortcomings: they are liable to reproduce factual details inaccurately, and they tend to repeat themselves. The state-of-the-art pointer-generator model came up by Google Brain at 2017 solves these problems. In addition to attention model, it add two features: first, it **copys** words from the source text via *pointing* which aids accurate repro- duction of information. Second, it uses **coverage** to keep track of what has been summarized, which discourages repetition. 

![image-20200210105957682](readme.assets/image-20200210105957682.png) 

<br><br>

## Network architecture

In addition to attention, we add two things:



### Copy distribution

* **Copy** frequent words occur in the text by adding distribution of the same word

  ![image-20200210121154603](readme.assets/image-20200210121154603.png) 

  ![image-20200207164857450](readme.assets/image-20200207164857450.png) 

   <br><br>

* **Combine** copy distribution `Pcopy`with general attention vocabulary distribution `Pvocab`(computed in attention earlier: ![img](readme.assets/clip_image002-1062650.png)) with certain weight `Pgen`:  *p*gen ∈ [0, 1] for timestep *t* is calculated from the context vector `a`∗, the decoder state `s`and the decoder input `c` :

  ![image-20200210121140811](readme.assets/image-20200210121140811.png)  

  ![image-20200210121130835](readme.assets/image-20200210121130835.png) 

  <br><br>

* **Training**: use `Pfinal` to compute sigmoid probability  

<br><br>

### Coverage mechanism 

record certain sentences that have appear in decoder many times

* **Sum the attention** over all previous decoder timesteps, `c`  represents the degree of coverage that those words have received from the attention mechanism so far.

     ![image-20200210121105048](readme.assets/image-20200210121105048.png) 

* **additive attention** of previous seq2seq attention model has changed to:

  ![image-20200210121045197](readme.assets/image-20200210121045197.png)  

* **add one more term for loss**

​    **loss = softmax loss +** ![image-20200210121018318](readme.assets/image-20200210121018318.png)  

<br><br>

## Implementation

 [**GitHub Code Here**](https://github.com/vcccaat/pointer-generator)

<br><br>

## Model Evaluation

![image-20200210105922693](readme.assets/image-20200210105922693.png) 



**Example from [Paper:](https://arxiv.org/abs/1704.04368)**

![image-20200210105854973](readme.assets/image-20200210105854973.png) 

<br><br>

## Result for financial dataset

Incoming.. 

<br><br><br><br>

# BERT

Incoming...

<br><br><br><br>

# Sentiment Analysis

**VADER**

focused on social media and short texts unlike Financial News, used available package in nltk, easy to use.

<br><br>

**CNN**

on yelp dataset

<br><br>

**LSTM** with transformer

on yelp dataset

<br><br>

**DPCNN**

on yelp dataset

<br><br>

**Deep and Wide Learning (Google)**

on yelp dataset