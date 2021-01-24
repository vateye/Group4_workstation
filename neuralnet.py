################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2021
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
import time
import sys
import logging
import csv
import pickle
import argparse
import matplotlib.pyplot as plt


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    # return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance.
    """
    inp = inp.copy().astype("float")
    mean_feature = np.mean(inp, axis=0, keepdims=True)
    inp -= mean_feature
    
    std_feature = np.std(inp, axis=0, keepdims=True)
    inp /= std_feature
    return inp


def one_hot_encoding(labels, num_classes=10):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    n_samples = len(labels)
    tmp = np.ones((n_samples, ))
    tmp = tmp * np.arange(num_classes)[:, None]

    labels = ((tmp == labels[None, ...]) > 0).astype(float)
    return labels.T


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x, axis = -1):
    """
    TODO: Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    # raise NotImplementedError("Softmax not implemented")
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x), axis = axis, keepdims=True)


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU", "identity"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)
        
        else:
            return a

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()
       
        elif self.activation_type == "leakyReLU":
            grad = self.grad_leakyReLU()
            
        else:
            grad = np.ones_like(delta)
            
        return grad * delta

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        # raise NotImplementedError("Sigmoid not implemented")
        return 1. / (1 + np.exp(-x))

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        # raise NotImplementedError("Tanh not implemented")
        return np.tanh(x)

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        # raise NotImplementedError("ReLu not implemented")
        return np.maximum(x, 0)

    def leakyReLU(self, x):
        """
        TODO: Implement leaky ReLU here.
        """
        # raise NotImplementedError("leakyReLu not implemented")
        return np.maximum(x, 0.1*x)

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        # raise NotImplementedError("Sigmoid gradient not implemented")
        tmp = self.sigmoid(self.x)
        return tmp * (1 - tmp)

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        # raise NotImplementedError("tanh gradient not implemented")
        tmp = self.tanh(self.x)
        return 1 - tmp ** 2

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        # raise NotImplementedError("ReLU gradient not implemented")
        tmp = np.zeros_like(self.x)
        tmp[self.x > 0] = 1
        return tmp

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        # raise NotImplementedError("leakyReLU gradient not implemented")
        tmp = np.ones_like(self.x)
        tmp[self.x <= 0] = 0.1
        return tmp


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        # * np.sqrt(2 / (out_units + in_units))
        self.w = np.random.randn(in_units, out_units)     # Declare the Weight matrix
        self.b = np.random.randn(out_units,)    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        # raise NotImplementedError("Layer forward pass not implemented.")
        self.x = x
        self.a = np.dot(x, self.w) + self.b
        return self.a

    def backward(self, delta):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # raise NotImplementedError("Backprop for Layer not implemented.")
        # delta = dA (number samples, out_units)
        m = self.x.shape[0]

        self.d_w = 1. / m * np.dot(self.x.T, delta)
        self.d_b = 1. / m * np.sum(delta, axis = 0, keepdims = False)
        self.d_x = np.dot(delta, self.w.T)

        return self.d_x
        


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.config = config
        
        self.cache = {}      # Hooker for forward propagate

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        # raise NotImplementedError("Forward not implemented for NeuralNetwork")
        
        self.cache["Z0"] = x
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                self.cache["A" + str(i//2 + 1)] = layer(self.cache["Z" + str(i//2)])
            else:
                self.cache["Z" + str(i//2 + 1)] = layer(self.cache["A" + str(i//2 + 1)])
                
        self.cache["Z" + str(len(self.layers)//2 + 1)] = softmax(self.cache["A" + str(len(self.layers)//2 + 1)])
        logits = self.cache["Z" + str(len(self.layers)//2 + 1)]
        if targets is None:
            return logits
        else:
            if len(targets.shape) == 1:
                targets = targets.reshape(1, -1)
            self.y = targets
            loss = self.loss(logits, targets)
            return logits, loss

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        # raise NotImplementedError("Loss not implemented for NeuralNetwork")
        m = targets.shape[0]

        l2_loss = 0
        for layer in self.layers:
            if isinstance(layer, Layer):
                l2_loss += np.sum(np.square(layer.w))
        loss = -np.mean(targets * np.log(logits + 1e-9)) + l2_loss * self.config["L2_penalty"] / (2 * m)
        return loss

    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        # raise NotImplementedError("Backprop not implemented for NeuralNetwork")
        layer_nums = len(self.layers)
        self.grad = {}
        self.grad["dA" + str(layer_nums//2 + 1)] = (self.cache["Z" + str(layer_nums//2 + 1)] - self.y) / self.y.shape[-1]

        for i, layer in zip(reversed(range(layer_nums)), reversed(self.layers)):
            if i % 2 == 0:
                self.grad["dZ" + str(i//2)] = layer.backward(self.grad["dA" + str(i//2 + 1)])
            else:
                self.grad["dA" + str(i//2 + 1)] = layer.backward(self.grad["dZ" + str(i//2 + 1)])
                
    @property
    def state_dict(self,):
        weights = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer):
                weights["W"+str(i//2 + 1)] = layer.w
                weights["b"+str(i//2 + 1)] = layer.b
        return weights
    
    def load_state_dict(self, state_dict):
        if len(state_dict) // 2 != len(self.layers) // 2 + 1:
            print("State dict match failed")
            return
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Layer):
                layer.w = state_dict["W"+str(i//2 + 1)]
                layer.b = state_dict["b"+str(i//2 + 1)]

                
def save(state_dict, filename = "./checkpoint.pkl"):
    '''Save model's parameters to file
    Args:
        state_dict: dict, model parameters
        filename: string, direction of the checkpoint file
    Returns:
        None
    '''
    with open(filename, 'wb') as f:
        pickle.dump(state_dict, f)
        

def load(filename = "./checkpoint.pkl"):
    '''load model's parameters from file
    Args:
        filename: string, direction of the checkpoint file
    Returns:
        state_dict: dict the model parameters
    '''
    with open(filename, 'rb') as f:
        state_dict = pickle.load(f)
    return state_dict
        

def get_grads(net, config):
    """
    Grab gradient for SGD.
    """
    grad = {}
    lamba = config["L2_penalty"]
    m = net.y.shape[0]
    for i, layer in enumerate(net.layers):
        if isinstance(layer, Layer):
            grad["dW" + str(i//2+1)] = layer.d_w + layer.w * lamba / m
            grad["db" + str(i//2+1)] = layer.d_b
    
    return grad


def initialize_velocity(grads):
    """
    Initializes the velocity for momentum
    """
    
    L = len(grads) // 2 # number of layers in the neural networks
    v = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((grads["dW" + str(l+1)].shape))
        v["db" + str(l+1)] = np.zeros((grads["db" + str(l+1)].shape))

    return v


def update_sgd(net, config, grads, v = None):
    """
    Stocahstic Gradient Descent update parameters
    v is the previous velocity for momentum
    if v is None, it will create one for it
    """
    
    grads = get_grads(net, config)
    lr = config["learning_rate"]
    L = len(grads) // 2
    
    if config["momentum"]:
        gamma = config["momentum_gamma"]
        if v is None:
            v = initialize_velocity(grads)
        for l in range(L):
            v["dW" + str(l+1)] = gamma * v["dW" + str(l+1)] + (1 - gamma) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = gamma * v["db" + str(l+1)] + (1 - gamma) * grads["db" + str(l+1)]
        
        for i, layer in enumerate(net.layers):
            if isinstance(layer, Layer):
                layer.w = layer.w - lr * v["dW" + str(i//2+1)]
                layer.b = layer.b - lr * v["db" + str(i//2+1)]
        return v
    else:
        for i, layer in enumerate(net.layers):
            if isinstance(layer, Layer):
                layer.w = layer.w - lr * grads["dW" + str(i//2+1)]
                layer.b = layer.b - lr * grads["db" + str(i//2+1)]
        return None
    
    
def data_loader(X, y, batch_size = 64, seed = 1):
    """
    Creates a list of random batchs from (X, y)
    
    Arguments:
    X -- input data with shape (number of samples, features_dims)
    y -- ground truth label (number of samples, xxx)
    batch_size -- size of the mini-batches, integer
    
    Returns:
    batches -- list of synchronous (batch_X, batch_y)
    """
    m = X.shape[0]
    batches = []
    
    # shuffle X, y
    np.random.seed(seed)
    idx = list(np.random.permutation(m))
    shuffled_X = X[idx]
    shuffled_y = y[idx]
    
    num_batches = m // batch_size
    for k in range(num_batches):
        batch_X = shuffled_X[k * batch_size : (k+1) * batch_size]
        batch_y = shuffled_y[k * batch_size : (k+1) * batch_size]
        batch = (batch_X, batch_y)
        batches.append(batch)
        
    if m % batch_size != 0:
        batch_X = shuffled_X[batch_size * num_batches:]
        batch_y = shuffled_y[batch_size * num_batches:]
        batch = (batch_X, batch_y)
        batches.append(batch)
    
    return batches


def accuarcy(logits, y):
    """
    Calculate accuarcy for NN output
    """
    pred = np.argmax(logits, axis = -1)
    gt = np.argmax(y, axis = -1)
    return np.mean(pred == gt)


def calculate_gain(nonlinearity):
    """Return the recommended gain value for the given nonlinearity function.
    The values are as follows: (Adopt from PyTorch)

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function
    """
    if nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'ReLU':
        return np.sqrt(2.0)
    elif nonlinearity == 'leakyReLU':
        negative_slope = 0.1
        return np.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def kaiming_initilization(model, nonlinearity = "ReLU"):
    """
    Do kaiming initialization for the network
    """
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Layer):
            m_in, m_out = layer.w.shape
            gain = calculate_gain(nonlinearity)
            layer.w = layer.w * gain / np.sqrt(m_out)
            layer.b = np.zeros_like(layer.b)

            
class AverageMeter(object):
    """Computes and stores the average and current value adopted from ImageNet"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)            


def generate_config_path(config):
    """
    generate path for checkpoint for specific config
    """
    
    path = "./exps/{}_{}_{:.1e}_{:.1e}_{}_{}_{}_{:.1f}".format(
        config["batch_size"], config["epochs"],
        config["learning_rate"], config["L2_penalty"],
        "-".join([str(i) for i in config["layer_specs"][1:-1]]),
        config["activation"], 
        str(config["momentum"]), config["momentum_gamma"]
    )
    
    return path


class SetupLogger(object):
    """
    Log console output to log file
    """
    
    def __init__(self, save_dir, filename="./sample.txt"):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, filename), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append('{:.4f}'.format(values[col]))

        self.logger.writerow(write_values)
        self.log_file.flush()
        

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    
    config_path = generate_config_path(config)
    train_logger = Logger(os.path.join(config_path, 'train.log'), ['Epoch', 'Loss', 'Accuracy'])
    valid_logger = Logger(os.path.join(config_path, 'valid.log'), ['Epoch', 'Loss', 'Accuracy'])
    
    # weight initilization via kaiming method
    kaiming_initilization(model, nonlinearity=config["activation"])
    
    num_epochs = config["epochs"]
    batch_size = config["batch_size"]
    v = None # momentum velocity
    
    best_acc = -1

    for epoch in range(1, num_epochs + 1):
        print("\nStarting for Epoch {}".format(epoch))
        train_loader = data_loader(x_train, y_train, batch_size=batch_size, seed=epoch)
        valid_loader = data_loader(x_valid, y_valid, batch_size=batch_size, seed=epoch)
        
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        # train the data with one epoch
        print("==========> Training for Epoch {}".format(epoch))
        for i, (X, y) in enumerate(train_loader):
            logits, loss = model.forward(X, y)
            
            model.backward()
            grads = get_grads(model, config) # get gradient via backward
            v = update_sgd(model, config, grads, v) # do sgd and return current velocity for next update
            
            acc = accuarcy(logits, y) # training accuarcy
            
            losses.update(loss, X.shape[0])
            top1.update(acc, X.shape[0])
            
            print("Epoch [{:3d}]/[{:3d}] | Iter [{:3d}]/[{:3d}] | Loss: {:4.4f} ({:4.4f}) | Valid Acc: {:4.4f} ({:4.4f})".format(
                epoch, num_epochs, i + 1, len(train_loader), 
                losses.val, losses.avg, top1.val, top1.avg
            ))
            
        train_loss.append(losses.avg)
        train_acc.append(top1.avg)
        train_logger.log({
            'Epoch': epoch,
            'Loss': losses.avg,
            'Accuracy': top1.avg
        })
        
        # Start validation
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        print("==========> Validation for Epoch {}".format(epoch))
        for i, (X, y) in enumerate(valid_loader):
            logits, loss = model.forward(X, y)
            
            acc = accuarcy(logits, y) # training accuarcy
            
            losses.update(loss, X.shape[0])
            top1.update(acc, X.shape[0])
            
            print("Epoch [{:3d}]/[{:3d}] | Iter [{:3d}]/[{:3d}] | Loss: {:4.4f} ({:4.4f}) | Valid Acc: {:4.4f} ({:4.4f})".format(
                epoch, num_epochs, i + 1, len(valid_loader), 
                losses.val, losses.avg, top1.val, top1.avg
            ))
            
        valid_loss.append(losses.avg)
        valid_acc.append(top1.avg)
        valid_logger.log({
            'Epoch': epoch,
            'Loss': losses.avg,
            'Accuracy': top1.avg
        })
        
        if top1.avg > best_acc:
            best_acc = top1.avg
            save(model.state_dict, filename=os.path.join(config_path, "checkpoint_best.pkl"))
    
    return train_loss, train_acc, valid_loss, valid_acc


def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """

    preds = []
    gts = []
    
    test_loader = data_loader(X_test, y_test, batch_size=32)
    for (X, y) in test_loader:
        pred = model.forward(X)
        preds.append(pred)
        gts.append(y)
    
    preds = np.concatenate(preds, axis = 0)
    gts = np.concatenate(gts, axis = 0)
    
    test_acc = accuarcy(preds, gts)
    
    print("Test Accuracy: {:4.4f}".format(test_acc))
    return test_acc


if __name__ == "__main__":
    # Load the configuration.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, 
                        default="./config.yaml", help="Configuration File")
    parser.add_argument("--data_path", type=str, 
                        default="./", help="Data path where you store xxx.tar.gz")
    args = parser.parse_args()

    config = load_config(args.config)
    config_path = generate_config_path(config)
    
    if not os.path.exists(config_path):
        os.makedirs(config_path)
        
    sys.stdout = SetupLogger(config_path, "main.log")
    print("Args:", config)

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path=args.data_path, mode="train")
    x_test,  y_test  = load_data(path=args.data_path, mode="t10k")

    # TODO: Create splits for validation data here.
    np.random.seed(42)
    idx = np.random.permutation(range(x_train.shape[0]))
    percentage = 0.8
    train_idx, val_idx = idx[:int(percentage * len(idx))], idx[int(percentage * len(idx)):]
    
    x_valid, y_valid = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    
    # TODO: train the model
    train_loss, train_acc, valid_loss, valid_acc = train(model, x_train, y_train, x_valid, y_valid, config)
    
    model.load_state_dict(load(os.path.join(config_path, "checkpoint_best.pkl")))
    test_acc = test(model, x_test, y_test)
    
    with open(os.path.join(config_path, "test_acc.txt"), 'w') as f:
        f.write("{:.4f}".format(test_acc))

    # TODO: Plot
    # Plot Loss
    plt.figure(figsize = (6, 4))
    plt.plot(np.arange(1, config["epochs"] + 1), train_loss, '-o', c = 'r', markersize = 2, label = "Train Loss")
    plt.plot(np.arange(1, config["epochs"] + 1), valid_loss, '-o', c = 'b', markersize = 2, label = "Valid Loss")
    plt.xticks(fontsize = 'large')
    plt.yticks(fontsize = 'large')
    plt.xlabel('Epoch', size = 'large')
    plt.ylabel('Loss', size = 'large')
    plt.legend(fontsize = 'large')
    plt.savefig(os.path.join(config_path, "loss.pdf"), format = 'pdf')
    
    
    # Plot accuracy
    plt.figure(figsize = (6, 4))
    plt.plot(np.arange(1, config["epochs"] + 1), train_acc, '-o', c = 'r', markersize = 2, label = "Train Acc.")
    plt.plot(np.arange(1, config["epochs"] + 1), valid_acc, '-o', c = 'b', markersize = 2, label = "Valid Acc.")
    plt.xticks(fontsize = 'large')
    plt.yticks(fontsize = 'large')
    plt.xlabel('Epoch', size = 'large')
    plt.ylabel('Accuracy', size = 'large')
    plt.legend(fontsize = 'large')
    plt.savefig(os.path.join(config_path, "accuracy.pdf"), format = 'pdf')
