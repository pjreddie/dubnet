![dubnet logo](figs/cyberdawg.png)

# dubnet: Homework 0 #

Welcome friends,

For the first assignment we'll be diving right in to neural networks. What are they? How do they work? Do they really require blood sacrifices on the third Saturday of odd-numbered months? We'll find out the answers to these questions and more!

## What the heck is this codebase?? ##

During this class you'll be building out your own neural network framework. I'll take care of some of the more boring parts like loading data, stringing things together, etc. so you can focus on the important parts.

Those of you who took vision with me will recognize a lot of the image codebase as stuff that you did! Early on we'll be focussing on applications of deep learning to problems in computer vision so there's a lot of code in `image.c` for loading images and performing some basic operations. No need to go into the details, just glance through `image.h` for a quick summary of the kinds of operations you can perform on images. We won't need any of that for this homework though!

More pertinent to this homework is the tensor library described in `tensor.h` and fleshed out in `tensor.c`. As we've learned in class, the heavy lifting in deep learning is done through tensor operations. One special case of tensors is matrices. We'll be implementing matrix transposition and multiplication in `matrix.c`.

Remember, the codebase is in C. There is a `Makefile` but every time you make a change to the code you have to run `make` for your executable to be updated. Try it right now by running:

    git clone https://github.com/pjreddie/dubnet
    cd dubnet
    make
    ./dubnet test hw0

This will tell you how bad you are doing on the homework so far (kind of like a loss function). Better get started...

## 0. Tensors ##

If you check out `tensor.h` you see that our tensors are pretty simple, a number of dimensions `n`, a list of sizes for those dimensions in `size` and a field for the `data`. Now check out `tensor.c` and read through the functions for making new (zeroed out) matrices and randomly filled matrices.

### 0.0 `tensor_copy` ###

Fill in the code for making a copy of a tensor.

### 0.1 `tensor_scale_` ###

Fill in the code to scale a tensor in-place. Note how `tensor_scale` is functional version, it just makes a new tensor and then calls the in-place version.

### 0.2 `tensor_axpy_` ###

Fill in the code to do elementwise `y = ax + y` on two tensors (must be the same size).

### Other tensor computations ###

Take a look at the rest of the file there's some fun stuff in there. You can do a bunch of elementwise binary tensor operations that have automatic broadcasting built-in by using functions like `tensor_add`, `tensor_mul`, etc. These are recursively implemented and not *super* fast but still good enough.

You can also sum tensors, either the whole tensor with `tensor_sum`, or along a certain dimension using `tensor_sum_dim`. Summing along a dimension is very useful for, example, aggregating gradient information across samples in a batch.

## 0. Matrices ##

Matrices are just tensors that only have `n==2`. But they get their own file in `matrix.c`.

### 1.0 `matrix_transpose` ###

Fill in the code to transpose a matrix. First make sure the matrix is the right size. Then fill it in. This might be handy: https://en.wikipedia.org/wiki/Transpose

![transpose example](figs/Matrix_transpose.gif)

### 1.1 `matrix_multiply` ###

Implement matrix multiplication. No need to do anything crazy, 3 `for` loops oughta do it. However, good cache utilization will make your operation much faster! Once you've written something you can run: `./dubnet time`. This doesn't actually check if your code is correct but it will time a bunch of matrix multiplications for you. Try optimizing your loop ordering to get the fastest time. You don't need to change anything except the loop order, there's just one ordering that's MUCH faster. On this test for my code it takes about 1.2 seconds (13 gflops) but your processing speed may vary!

## 2. Activation functions ##

An important part of machine learning, be it linear classifiers or neural networks, is the activation function you use. We'll define our activation functions in `activation_layer.c`. This is also a good time to check out `dubnet.h`, which gives you an overview of the structures will be using for building our models and, important for this, what activation functions we have available to us!

For Leaky ReLU use an alpha = 0.01 if you want to pass the tests!

### 2.0 `forward_activation_layer` ###

Fill in `tensor forward_activation_layer(layer *l, tensor x)` to modify `y` to be `f(y)` applied elementwise where the function `f` is given by what the activation `a` is.

Remember that for our softmax activation we will take e^x for every element x, but then we have to normalize each element by the sum as well. Each element in the first dimension of the tensor is a separate data point so we want to normalize over each data point separately.

### 2.1 `backward_activation_layer` ###

From dL/dy we need to calculate dL/dx = f'(x) * dL/dy. We'll take each element of dL/dx and multiply it by the appropriate gradient f'(x). Guess i sorta just said that twice.

The gradient of a linear activation is just 1 everywhere. The gradient of our softmax will also be 1 everywhere because we will only use the softmax as our output with a cross-entropy loss function, for an explaination of why see [here](https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa) or [here](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/#softmax-and-cross-entropy-loss).

The gradient of our logistic function is discussed [here](https://en.wikipedia.org/wiki/Logistic_function#Derivative):

    f'(x) = f(x) * (1 - f(x))

I'll let you figure out on your own what `f'(x)` given `f(x)` is for RELU and Leaky RELU.

## 3. Connected Layers ##

For this homework we'll be implementing a fully connected neural network layer as our first layer type! This should be fairly straightforward. First let's check out our layer definition in `src/dubnet.h`:

    typedef struct layer {
        tensor x;
    
        // Weights
        tensor w;
        tensor dw;
    
        // Biases
        tensor b;
        tensor db;
    
        ACTIVATION activation;
    
        tensor  (*forward)  (struct layer *, struct tensor);
        tensor  (*backward) (struct layer *, struct tensor);
        void   (*update)   (struct layer *, float rate, float momentum, float decay);
    } layer;

### 3.0 `forward_connected_layer` ###

Our layer outputs a matrix called `y = xw + b` where: `x` is the input, `w` is the weights, `b` is the bias.

To compute the output of the model we first will want to do a matrix multiplication involving the input and the weights for that layer. Remember, the weights are stored under `l->w`.

Then we'll want to add in our biases for that layer, stored under `l->b`. Elementwise tensor sum using broadcasting comes in handy here!

Remember to `tensor_free` any intermediate results that you allocated but don't need anymore.

### 3.1 `backward_connected_layer` ###

This is the trickiest one probably.

To update our model we want to calculate `dL/dw` and `dL/db`.

We also want to calculate the deltas for the previous layer (aka the input to this layer) which is `dL/dx`.

What we have to start with is `dL/dy`, which is equivalent to `dL/d(xw+b)`.

First we need `dL/db` to calculate our bias updates. Actually, you can show pretty easily that `dL/db = dL/d(in*w+b)` so we're almost there. Except we have bias deltas for every example in our batch. So we just need to sum up the deltas for each specific bias term over all the examples in our batch to get our final `dL/db`. You can use `tensor_sum_dim` to sum the deltas over examples in the batch and then you can add them in to `l->db`.

Then we need to get `dL/dxw` which by the same logic as above we already have as `dL/d(xw+b)`. So taking our `dL/d(xw)` we can calculate `dL/dw` and `dL/dx` as described in lecture. Don't reset `l->dw`, just add in the new gradient using axpy. Good luck!

### 3.2 `update_connected_layer` ###

We want to update our weights using SGD with momentum and weight decay.

Our normal SGD update with learning rate η is:

    w = w - rate*dL/dw

With weight decay `decay` we get:

    w = w - rate*dL/dw - rate*decay*w

With momentum our update will be:
    
    w = w - rate*dL/dw - rate*decay*w + rate*momentum*prev

We can undistribute a `-rate` to get:

    w = w - rate*[dL/dw + decay*w - momentum*prev]

note that for next itertion `-prev` is just the stuff in the brackets:

    -prev_{i+1} = [dL/dw + decay*w - momentum*prev]


In our case, we'll always try to keep track of `-prev` in our layer's `l.dw`. When we start the update method, our `l.dw` will store:

    l.dw = dL/dw - momentum*prev

Our first step is to add in the weight decay (using axpy) so we have:

    l.dw = dL/dw - momentum*prev + decay*w

This value is gradient of our full loss function (that includes decay) and includes the negative momentum term. Next we need to subtract a scaled version of this into our current weights (also using axpy).

    l.w = l.w - rate*l.dw

After we apply the updates, `l.dw = -prev` (for the next iteration). Thus our final step is to scale this vector by our momentum term so that:

    l.dw = momentum * -prev

Then during our next backward phase we are ready to add in some new `dL/dw`s!

You will do a similar process with your biases but there is no need to add in the weight decay, just the momentum.

## 4. Training your network! ##

### 4.0 Understanding the network ###

First check out `net.c` to see how we run the network forward, backward, and updates. It's pretty simple.

Now, little did you know, but our C library has a Python API all ready to go! The API is defined in `dubnet.py`, but you don't have to worry about it too much. Mainly, check out `tryhw0.py` for an example of how to train on the MNIST data set. But wait, you don't have any data yet!

### 4.1 Get MNIST data ###

Background: MNIST is a classic computer vision dataset of hand-drawn digits - http://yann.lecun.com/exdb/mnist/

I already have it all ready to go for you, just run:

    wget https://pjreddie.com/media/files/mnist.tar.gz
    tar xvzf mnist.tar.gz

### 4.2 Train a model ###

Make sure your executable and libraries are up to date by running:

    make

Then try training a model on MNIST!

    python tryhw0.py

Every batch the model will print out the loss and at the end of training your model will run on the training and testing data and give you final accuracy results. Try playing around with different model structures and hyperparameter values. Can you get >97% accuracy?

### 4.3 Train on CIFAR ###

The CIFAR-10 dataset is similar in size to MNIST but much more challenging - https://www.cs.toronto.edu/~kriz/cifar.html. Instead of classifying digits, you will train a classifier to recognize these objects:

![cifar10](figs/cifar.png)

First get the dataset:

    wget https://pjreddie.com/media/files/cifar.tar.gz
    tar xvzf cifar.tar.gz

Then modify the `tryhw0.py` script to train on CIFAR:

    python tryhw0.py

How do your results compare to MNIST? If you changed your model for MNIST, do similar changes affect your CIFAR results in the same way?

## Running on the GPU with PyTorch ##

Navigate to: https://colab.research.google.com/

and upload the iPython notebook provided: `hw0.ipynb`

Complete the notebook to train a PyTorch model on the MNIST dataset.


## Turn it in ##

First run the `collate_hw0.sh` script by running:

    bash collate_hw0.sh
    
This will create the file `hw0.tar.gz` in your directory with all the code you need to submit. The command will check to see that your files have changed relative to the version stored in the `git` repository. If it hasn't changed, figure out why, maybe you need to download your ipynb from google?

Submit `hw0.tar.gz` in the file upload field for Homework 0 on Canvas.

