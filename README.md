# Human Activity Recognition (HAR)

This repository is about human activity recognition project.

Data explanation

Data set name: WISDM version 1.1
Website:  http://www.cis.fordham.edu/wisdm/dataset.php
Format: 6 columns per row

user, activity, timestamp, x-acceleration, y-acceleration, z-acceleration

Activity: Walking, Jogging, Sitting, Standing, Upstairs, Downstairs 

## Table of Contents
<!--ts-->
[Getting Started](##gettingstarted)
[Running the tests](##runningthetests)
[Author](##author)
[Reference](##reference)

<!--te-->
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- python 3.7 (but it's also work, if you use python 3.6)
- tensroflow (version 2 but in the code already change compatibility to version 1)
- pandas
- matplotlib
- Keras
- seaborn
- sklearn
- [virsual environment](https://docs.python.org/3/tutorial/venv.html)
Before you run the code,
1. make directory for data set named 'data' in the project directory. In 'data' directory must has 'CNN-data' and 'LSTM-data' directory.
2. set Virtualenv that has all inpenndencies.

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be


## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why


### And coding style tests

Explain what these tests test and why


## Author

* **Phataratah Sa-nguannarm** - *Graduate student at Sun Moon University*

## Reference
- LSTM model based on [TensorFlow on Android for Human Activity Recognition with LSTMs](https://github.com/curiousily/TensorFlow-on-Android-for-Human-Activity-Recognition-with-LSTMs)
- CNN model based on [Human Activity Recognition using CNN in Keras](https://github.com/Shahnawax/HAR-CNN-Keras)
- Github TOC [gh-md-toc](https://github.com/ekalinin/github-markdown-toc/blob/master/README.md)
- Save and restore the model [here](https://github.com/AISangam/Save-and-Restore-Model-Tensorflow)
