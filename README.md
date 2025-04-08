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
makemore is a neural net that will learn to make more of whatever dataset is supplied. 

Under the hood, makemore is a character level language model. It treats every single line as an example and each example is a sequence of individual characters. It knows how to predict the next character in a sequence. 
```
cd makemore
```
The **names.txt** file here was taken from Karpathy's [makemore repo](https://github.com/karpathy/makemore/blob/master/names.txt).

---

## [GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&ab_channel=AndrejKarpathy)
```
cd gpt
```
This folder contains an exercise builing GPT from scratch. `input.txt` contains the Tiny Shakespeare collection. Run the demo at **gpt.ipynb**

### Steps to setup GPU instance
1. Go to AWS account, services, EC2. Click on Launch Instance.
2. Launch EC2 instance. 
    - Name: **ML Training**
    - AMI from catalog: **[Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6 (Ubuntu 22.04)](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-6-ubuntu-22-04/)** It is optimized for deep learning and includes PyTorch. Supported EC2 instances: G4dn, G5, G6, Gr6, G6e, P4, P4de, P5, P5e, P5en
    - Instance type: **g4dn.xlarge**
    - Key Pair (login): **aws.pem**
    - Network settings: Network is the VPC you want to launch the instance into. Create security group that will allow SSH traffic from My IP. 
3. Connect to the instance. ```ssh -i "aws.pem" ubuntu@ec2-instance.us-west-1.compute.amazonaws.com```
4. Activate pytorch virtual environment venv in the instance. Git clone this repo. Pip install requirements.
```
source /opt/pytorch/bin/activate
git clone https://github.com/salazardenise/neural_networks.git
python3 -m pip install -r requirements.txt
```
5. Try training bigram model. Try training transformer model with current hyperparameters.
```
python3 bigram.py
```
5. Update hyperparameters for bigger network and run transformer.py.
```
vim transformer.py
```
6. IMPORTANT: Stop the instance.

## [Tokenization](https://www.youtube.com/watch?v=zduSFxRajkE&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=9&ab_channel=AndrejKarpathy)
```
cd tokenization
```
This folder explores the Byte Pair Encoding algrithm for tokenization.

## Future Work
- try the suggested exercises per lecture
