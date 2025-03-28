This repo follows the Andrej Karpathy zero to hero lectures.
Reference is [here](https://karpathy.ai/zero-to-hero.html).

## [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6013s&ab_channel=AndrejKarpathy)
micrograd is a tiny engine that implements backpropagation and a small neural network. This only operates on scalar values.
```
cd micrograd
```
### how virtual environment was created
```
python3 -m venv .venv
```
### activate virtual environment
```
source .venv/bin/activate
which python3
```
### deactivate virtual environment
```
deactivate
```
### brew install graphviz
```
brew install graphviz
```
### list installed packages in virtual envrionment
```
python3 -m pip list
```
### install libraries in virtual environment
```
python3 -m pip install -r requirements.txt
```
### freeze installed packages to requirements.txt in virtual environment
```
python3 -m pip freeze > requirements.txt
```
### demo micgrad
run **demo_micrograd.ipynb** in an ipython notebook. 

---

## [makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&ab_channel=AndrejKarpathy)
A neural net that will learn to make more of whatever dataset is supplied. 

Under the hood, makemore is a character level language model. It treats every single line as an example and each example is a sequence of individual characters. It knows how to predict the next character in a sequence. 
```
cd makemore
```
The **names.txt** file here was taken from Karpathy's [makemore repo](https://github.com/karpathy/makemore/blob/master/names.txt).

---

## GPT
This folder contains an exercise builing GPT from scratch.
