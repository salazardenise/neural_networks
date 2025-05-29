## Table of Contents
- [Libraries](#libraries)
    - [Numpy](#numpy)
    - [PyTorch](#pytorch)
- [Micrograd](#micrograd)
- [Makemore](#makemore)
    - [Bigram](#bigram)
    - [MLP](#mlp)

## Libraries
### NumPy
- `import numpy as np`
- `np.arange(-5,5,0.2)` - [np.arange()](https://numpy.org/doc/2.1/reference/generated/numpy.arange.html) returns evenly spaced values within a given interval. inputs are start, stop, step.
- `np.tanh(np.arange(-5,5,0.2))` - [np.tanh()](https://numpy.org/doc/2.1/reference/generated/numpy.tanh.html#) computes hyperbolic tangent element-wise. 
- `np.linspace(-3, 3, 100)` - [np.linspace()](https://numpy.org/doc/2.1/reference/generated/numpy.linspace.html#) return evenly spaced numbers over a specified interval. Given num of 100, returns 100 evenly spaced samples, calculated over the interval -3 to 3, inclusive. End point excluded if `endpoint` is set to False. 


### PyTorch
- An open sourced deep learning [library](https://pytorch.org/)
- `torch.tensor` - Creates a tensor. torch.tensor constructs tensor with no autograd history (also known as a leaf tensor) by copying data. torch.tensor infers the dtype automatically. It also has the dtype argument.
- `t = torch.Tensor`- Another way to create a tensor. [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#) returns a torch.FloatTensor, aka default type is float32.
- `t.shape` - [t.shape](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.shape.html) returns the size of tensor t. 
- `t.requires_grad = True` - PyTorch assumes by default that leaf nodes do not require gradients. This is for efficiency reasons. You can set it to True explicitly, if needed.
- `t.item()`, `t.data.item()` and `t.grad.item()` - takes tensor of single item and strips the tensor. t.item() and t.data.item() produce the same result in PyTorch.
- `t.backward()` - t is a tensor object that has a [backward](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) function. backward computes the gradient of current tensor wrt graph leaves. This function accumulates gradients in the leaves - **you might need to zero .grad attributes or set them to None before calling it**.
- `N = torch.zeros((27, 27), dtype=torch.int32) ` - [zeros](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html creates tensor of 27 rows x 27 columns array of zeros. Numbers type defaults to single precision floating point, aka float32. 
- `g = torch.Generator().manual_seed(2147483647)` - [torch.Generator()](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) creates a random number generator.
- `p = torch.rand(3, generator=g)` - [torch.randn](https://docs.pytorch.org/docs/stable/generated/torch.randn.html)  creates a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution), aka from a uniform distribution on the interval [0,1). Random numbers generated using the given generator object.
- `p_samples = torch.multinomial(p, num_samples=20, replacement=True, generator=gg)` - Use [torch.multinomial](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html) to draw samples from the torch tensor of probabilities, p. Set replacement to True so that when we draw an element, we can then put it back in list of indicies we can draw from again. This defaults to False. 
- `P = N.float()` - casts all items in N tensor to floats
- `Psums = P.sum(1, keepdim=True)` - To calculate the sum along specific dimensions, the following syntax is used: `torch.sum(input, dim=None, keepdim=False, *, dtype=None) → Tensor`. For a P of size 27x27, if we set keepdim to True, the output shape is now 27x1, aka a column vector of counts, going horizontally.
- `P = (N+1).float()` - add a count of 1 to N to smooth the model out. The more you add here, the more uniform the model you will have, and the less you add the more peaks you will have.
- `import torch.nn.functional as F; xenc = F.one_hot(xs, num_classes=27).float() ` - The torch.nn.functional module has a functional approach to work on the input data, aka the functions of the module work directly on the input data, without creating an instance of a neural network layer. The functions in this module are stateless. [torch.nn.functional.one_hot()](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html) takes LongTensor with index values of shape (*) and returns a tensor of shape (*, num_classes) that have zeros everywhere except where the index of last dimension matches the corresponding value of the input tensor, in which case it will be 1.
- `logits = xenc @ W` - @ is used for matrix multiplication
- `counts = logits.exp()` - [torhc.exp()](https://docs.pytorch.org/docs/stable/generated/torch.exp.html). Returns a new tensor with the exponential of the elements of the input tensor input. i.e. $y_i = e^{x_i}$
- `concatenated = torch.cat(list_of_tensors, 1)` - [torch.cat()](https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) concatenates lists on dimension 1.
- `torch.cat(torch.unbind(mat, 1), 1)` - [torch.unbind()](https://pytorch.org/docs/stable/generated/torch.unbind.html) function which removes a tensor dimension and tuple of all slices along a given dimension.
- `mat.view(32, 6)` - [torch.Tensor.view()](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) returns a new tensor with the same data at the self tensor but of a different shape. The new view size must be compatible with its original size; aka size input to view must multiple to same original size. The view operation is extremely efficient! Each tesnor has underlying storage which is just the numbers in a 1D vector.  In view, no memory is changed copied etc, the storage is identical, only attributes of view (stride, offset, etc) is manipulated.
- `a.storage()` - [torch.Tensor.storage()](https://pytorch.org/docs/stable/generated/torch.Tensor.storage.html).
- `import torch.nn.functional as F; loss = F.cross_entropy(logits, Y) ` - [torch.nn.functional.cross_entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) computes the cross entropy loss between input logits and target.


## Micrograd
| Term | Defintion |
| ---- | --------- |
| Neural Network | A model with at least one hidden layer. Each neuron in a neural network connects to all of the nodes in the next layer. |
| Layer | A layer is a set of neurons in a neural network layer. Types of layers include the input layer, hidden layer, and output layer. |
| Activation Function | A mathematical function that determines the output of a neuron based on its input. It introduces non-linearity into the model, allowing the model to learn and represent complex patterns in data. If a neural network were purely linear, it would be limited to learning only linear patterns. Common types of activation functions include: tanh function. |
| tanh | A hyperbolic tangent function maps the input to a range between -1 and 1. This is a common activation function. |
| Neuron | A fundamental unit within a neural network layer. |
| Weight | A value that a model multiplies by another value. Training is the process of determining a model's ideal weights; inference is the process of using those learned weights to make predictions. |
| Bias | Bias is a parameter in machine learning models. In a simple 2D line, bias is the "y-intercept." | 
| Parameters | All the weights and biases in the neural net. These are learned during training, i.e. initialized and then optimized. |
| Prediction | A model's output. | 
| Ground truth | The reality as opposed to the prediction. | 
| Forward pass | During the forward pass, the model processes examples to get predictions. The loss is computed by comparing each pedictiomn to each label value. |
| Backward pass | During the backward pass (backpropagation), the model reduces the loss by adjusting the weights of all the neurons in all the hidden layers. | 
| Backpropagation | The algorithm that implements gradient descent in neural networks. This is done with many iterations of the forward pass and backward pass. In calculus terms, backpropagation implements the chain rule. That is, backpropagation calculates the partial derivative of the error with respect to each parameter. | 
| Tensor | A multi-dimensional array that serves as the fundamental data structure for representing and manipulating data in machine learning models. Tensors can leverage GPUs for accelerated computation, which is essential for training deep learning models efficiently. | 
| Multi Layer Perceptron (MLP) | A feedforward neural network contianing connected neurons in layers with nonlinear activation functions. The MLP is trained using backpropagation. | 
| Binary Classifier | A type of classification task that predicts one of two mutually exclusive classes. | 
| Loss | A single number calculated that indicates the performance of the neural network. We want to train the model such that the loss is minimized, aka the targets are close to predictions. |
| Mean Squared Error loss (MSE) | $Mean Squared Error = L_2 loss / Number of Examples$, aka the average L2 loss per example. Used for linear regression. $L_2$ is a function that calculates the square of the difference between actual label values and the values that a model predicts. Due to squaring, $L_2$ loss amplifies the influence of outliers, aka bad predictions. |


## Makemore

### Bigram
| Term | Defintion |
| ---- | --------- |
| Bigram | A bigram is a sequence of two adjacent elements from a string of tokens, which are typically letters, syllables, or words. A bigram is an n-gram for n=2. Bigrams, along with other n-grams, are used in most successful language models. | 
| Bigram character level language model | This type of model predicts the probability of the next character in a sequence, based on the preceding character. It treats each character as a token and uses the frequency of character pairs (bigrams) to estimate the likelihood of one character following another.  |
| Data normalization | Data normalization (or feature scaling) includes methods that rescale input data so that the features have the same range, mean, variance, or other statistical properties. |
| Multinomial | In probability theory, the multinomial distribution is a generalization of the binomial distribution. In machine learning, it's often used in classification tasks. In essence, the multinomial distribution provides a way to quantify the probabilities of different outcomes when dealing with multiple, mutually exclusive possibilities in machine learning tasks. | 
| [Broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) | PyTorch broadcasting is a powerful mechanism that allows operations on tensors with different shapes. It avoids the need to manually reshape tensors to make them compatible for element-wise operations, leading to more concise and efficient code. |
| [Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) | Maximum likelihood estimation is a way of estimating unknown parameters using observed data. |
| Negative log likelihood | A cost function that is used as loss for machine learning models, telling us how bad it’s performing, the lower the better. Used for classification applications. |
| Monotonic |  Monotonic means it either never decreases or never increases |
| Model smoothing | Model smoothing in machine learning refers to techniques used to improve the robustness and generalization of models, particularly in the context of classification and language models. These techniques aim to make models less overconfident and more adaptable to various inputs, including unseen data. |
| Training set | In machine learning, a training set is the portion of a dataset used to teach a model how to make predictions or classifications. It's the data the model learns from, allowing it to identify patterns and relationships that enable it to make predictions on new, unseen data. |
| One-hot encoding | One-hot encoding is a technique that converts categorical data into a numerical format where each category is represented by a separate column with a 1 indicating its presence and a 0 for all other categories. These vectors of floats can then feed into neural nets. | 
| Gradient descent | Gradient descent is a popular optimization algorithm used in machine learning to find the minimum of a function, often a cost or loss function. It works by iteratively adjusting the parameters of the function until it reaches a minimum.  |
| Regularization | Regularization is an important technique in machine learning that helps to improve model accuracy by preventing overfitting which happens when a model learns the training data too well including noise and outliers and perform poor on new data. By adding a penalty for complexity it helps simpler models to perform better on new data. aka augment the loss function with a small component regularization loss. |
| [logits](https://telnyx.com/learn-ai/logits-ai) | Logits are the outputs of a neural network before the activation function is applied. They are the unnormalized probabilities of the item belonging to a certain class. Logits are often used in classification tasks, where the goal is to predict the class label of an input. |
| [softmax](https://en.wikipedia.org/wiki/Softmax_function) | The softmax function is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes. softmax function converts raw output scores, also known as logits, into probabilities by taking the exponential of each output and normalizing these values by dividing by the sum of all the exponentials. |

### MLP
| Term | Defintion |
| ---- | --------- |
| Stochastic gradient descent (SGD) | SGD is an optimization algorithm used in machine learning to train models. SGD uses a single data point or a small batch to calculate the gradient and update the parameters, while GD uses the entire training dataset to calculate the gradient. |
| Natural Language Processing (NLP) | Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It enables computers to understand, interpret, and generate human language. |
| Embedding lookup table | An embedding is a vector representation of data in embedding space. A model finds potential embeddings by projecting the high-dimensional space of initial data vectors into a lower-dimensional space. In an embedding, the distance between any two items can be calculated mathematically, and can be interpreted as a measure of relative similarity between those two items. Embeddings make it easier to do machine learning on large feature vectors. |
| Block size | The context length, i.e. how many characters do we take to predict the next one? |
| Hidden layer | A layer in a neural network between the input layer (the features) and the output layer (the prediction). Each hidden layer consists of one or more neurons. A deep neural network contains more than one hidden layer. |
| Classification | A model whose prediction is a class. In contrast, regression models predict numbers rather than classes. |
| Cross entropy function | A loss function that is commonly used for classification tasks. Also known as logarithmic loss or log loss for binary classification. |
| Overfitting | Overfitting means the model has memorized your data very well but won't give any new type of output. Occurs when training and dev loss are  diverging. |
| Underfitting | If loss_train and loss_dev are about the same, then we are underfitting. That means our network is very tiny. So to imporve performance, we need to increase the size of the neural net. Take note that the bigger the network, the longer it may take to converge. |
| Batches | In practice, people usually do forward and backward pass on a batch of data, a randomly selected batch of data. When training on mini batches, the quality of the gradient is low and the direction of the gradeint is not as reliable, aka not the actual gradient direction. But with a good batch size, this should be good enough. | 
| Learning rate | At late stages of training, when the loss is starting to plateau, do a learning rate decay, aka decrease the learning rate.|
| Learning rate decay | Learning rate decay is a technique in machine learning, particularly in neural network training, where the learning rate is gradually reduced as the training progresses. This helps the model converge to a better solution by taking smaller steps towards the minimum. |
| Dev / Validation set |  The training set is used to optimize the parameters of the model. The dev/validation set is used to optimize the hyperparameters and settings of the model. like size of hidden layer, strength of regularization, you can try different variations. |
| Test set | The test set is used to evaluate the performance of the model at the end, this is done very few times, otherwise you risk overfitting on the model. |


## Papers
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) 2003


## Recommended References
- [Activation functions in Neural Networks geeks for geeks page](https://www.geeksforgeeks.org/activation-functions-neural-networks/)
- [Recommended blog on internals of torch tensors](https://blog.ezyang.com/2019/05/pytorch-internals/)
