# Introduction to the project

Natural Language Processing or NLP, refers to using machine learning to process natural text, with ‘natural’ referring to the kind of text we find in books and newspapers, as opposed to, for example computer programming code.

### Transfer Learning in NLP

Transfer learning,is a technique where a deep learning model trained on a large dataset is used to perform similar tasks on another dataset. NLP shares a great deal of simlarities in linguistric expressions and structural presentations. Moreover Tranfer Learning have proved to have yield state-of-the-atr results for NLP tasks over and over again. However, training a computer vision or natural language model can be expensive. It requires lots of data, and in the case of computer vision, that data needs to be labelled by humans, and it can take lot of time to train on expensive hardware.

### Training vs Fine-tuning

Here’s the good news though, we can download these models for free on the internet, pre-trained on enormous datasets and ready to go. What’s even better is that we can fine-tune these models really quickly to work on the nuances of your specific dataset. To give a sense of the difference between fine-tuning and training a model, fine-tuning is like taking your car to the mechanic and getting new spark plugs whereas training/pre-training is like getting a whole new engine. If we were to train a model from scratch it would likely take a few hours or longer, need an enormous amount of data that we’d need to collect and pre-process yourself and would be less accurate than the fine-tuned model. 

### Tensorflow Hub

Website link:- https://tfhub.dev/

TensorFlow Hub is a repository of pre-trained TensorFlow models.

In this project, we will use pre-trained models from TensorFlow Hub with tf.keras for text classification. Transfer learning makes it possible to save training resources and to achieve good model generalization even when training on a small dataset. In this project, we will demonstrate this by training with several different TF-Hub modules.

### Learning Objectives

- Use various pre-trained NLP text embedding models from TensorFlow Hub
- Perform transfer learning to fine-tune models on your own text data
- Visualize model performance metrics with TensorBoard

# Setup your TensorFlow and Colab Runtime

You will only be able to use the Colab Notebook after you save it to your Google Drive folder. Click on the File menu and select “Save a copy in Drive. You will only be able to use the Colab Notebook after you save it to your Google Drive folder. Click on the File menu and select “Save a copy in Drive".

# Download and Import the Quora Insincere Questions Dataset

A downloadable copy of the Quora Insincere Questions Classification data can be found https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip. Decompress and read the data into a pandas DataFrame.

![image](https://user-images.githubusercontent.com/57663083/179395185-c9f1adec-fcfc-41bf-a53a-6b99be4224d2.png)

![image](https://user-images.githubusercontent.com/57663083/179395281-065d378b-ca7f-4ec8-803e-d7b86b5bedd1.png)

# TensorFlow Hub for Natural Language Processing

Tensorflow Hub provides a number of modules to convert sentences into embeddings such as Universal sentence ecoders, NNLM, BERT and Wikiwords. In this project, we will demonstrate this by training with several different TF-Hub modules.

- tf2-preview/gnews-swivel-20dim (https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1)
  Token based text embedding trained on English Google News 130GB corpus. Returns 20-dimensional embedding vectors.

- nnlm-en-dim50 (https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1)
  Token based text embedding trained on English Google News 7B corpus. Returns 50-dimensional embedding vectors.
  
- nnlm-en-dim128 (https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1)
Token based text embedding trained on English Google News 200B corpus. Returns 128-dimensional embedding vectors.

# Define Function to Build and Compile Models & Train Various Text Classification Models

We build and comple our model adding some dense layers to the initial hub layer. Then the model is compiled,trained and tested. We also keep a "trainable parameter" to referene the model to enable fine-tuning.

![image](https://user-images.githubusercontent.com/57663083/179395942-e086d102-c87e-420a-a7df-f5923a02a65b.png)

# Compare Accuracy and Loss Curves

After the model has been fitted, we compare accuracies and loss curves using matplotlib. 

## Conclusions:-

As predicted the fine-tuned models perform significantly better than their counterparts. On the whole, the "nlm-en-dim128" module with embed-size=128 gives the best results followed by the "nlm-en-dim50" with embed-size=50, preceded by "gnews-swivel-20dim" with embed-size=20.   

![image](https://user-images.githubusercontent.com/57663083/179396386-f6471290-06f6-4f4b-971c-f27477fd145c.png)

![image](https://user-images.githubusercontent.com/57663083/179396424-00972fea-acd2-4a97-8ee7-58fb931ca0ff.png)

# Train Bigger Models and Visualize Metrics with TensorBoard

- universal-sentence-encoder (https://tfhub.dev/google/universal-sentence-encoder/4)
  Encoder of greater-than-word length text trained on a variety of data. Returns 512-dimensional embedding vectors.
  
- universal-sentence-encoder-large (https://tfhub.dev/google/universal-sentence-encoder-large/5)
  Encoder of greater-than-word length text trained on a variety of data. Returns 512-dimensional embedding vectors.

# Compare Accuracy and Loss Curves
After the new models has been fitted, we compare accuracies and loss curves using matplotlib.

## Conclusions:-
The "universal-sentence-encoder-large" module whhch is based on transformer architecture model outperforms every other model on the list. The "universal-sentence-encoder" which is based on averaging architecture does worse than "universal-sentence-encoder-large" module, but significantly better than "nnlm-en-dim128" module. 
However we also notice that the fine-tuned "nnlm-en-dim128" module performs at par with the 512-embedding vector "universal-sentence-encoder". So we achieved a significantly higher accuracy from a lighter module by just fine tuning it with our datasets.

![image](https://user-images.githubusercontent.com/57663083/179397514-71ebe354-c710-42fe-80b6-02b790fe03ee.png)

![image](https://user-images.githubusercontent.com/57663083/179397518-b91594a2-2185-478d-bd61-56dd986e4916.png)
