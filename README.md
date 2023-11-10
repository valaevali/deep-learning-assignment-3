# Deep Learning Lecture (2023)

This repository contains the 3rd assignment of the Deep Learning Lecture. The main file is the Jupyter Notebook 'code/assignment3.ipynb'. The html-Version with the code blocks output inlined is 'assignment3.html'.

The goal in the end was to achieve the highest accuracy score for an unknown data set - I achieved the 4-5 place (from about 30 people) and gained 100/100 percent in total score for this assignment.

## Task

Given an image, the task of image captioning is to describe its visual content. This task is very challenging compared to other downstream tasks, e.g., object classification and detection, as the algorithm should generate descriptions that capture individual objects, express how these objects relate to each other, and describe their attributes and the activities they are involved in. Image captioning is a prominent application, where both computer vision and natural language processing methods are used. In this assignment, your task is to train an image captioning model using the tools that you have learned in class.

### Data
In this assignment, you will use *Flickr8k* dataset (This dataset is too big for this repository). The dataset is split in the following ways:

- a training set containing 7000 images, where each image has 5 descriptions
- a validation set containing 500 images, where each image has 5 descriptions
- a test set containing 500 images, where each image has 5 descriptions

You are given image-caption pairs in the training and validation sets, but only images in the test set. Once the model is trained, you will run it on images in the test set and obtain the corresponding captions (descriptions). In the end, you will submit your code and descriptions. Note that the get loader.py file should not be edited.

### Image Captioning Task
The standard architecture for this task is an encoder-decoder type, i.e., the image is encoded via encoder to obtain a visual descriptor and then this information is used in the decoder (a sequence model) to generate the description.

You are free to use any architecture for the encoder (e.g., CNN, Vision Transformer, etc.) and decoder (RNN, LSTM, Transformers, etc.). More specifically, for the encoder
- you can have any pretrained model that is trained either in an unsupervised (e.g., DINO) or a supervised (available in PyTorch) way
- You can define your own model from scratch and train it jointly with the decoder. You can also pretrain your own model in an unsupervised way on large datasets (e.g., imagenet) by using self-supervised learning methods. Please note that this will take time to train and your time is limited.

We require to train the decoder from scratch.
