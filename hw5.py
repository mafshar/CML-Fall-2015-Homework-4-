#!usr/bin/env python

'''
Mohammad Afshar
N19624829
CSCI-GA 3033-12: Computational Machine Learning
HW 5
Due: 11/29/15 at 00:00
'''

from sklearn.datasets import load_iris ## obtaining the iris dataset
from scipy.spatial import distance ## for calculating euclidean distance
from sklearn.preprocessing import normalize ## for normalizing data
from operator import sub
from operator import add
import matplotlib.pyplot as plt
import numpy as np
import random ## for seeding the permutation of the iris data
import copy
import sys
import pylab

DUE_DATE = 112915

## h can vary from 0.1 to 0.5
h = 0.5

## code from scikit-learn -- used only for graphing
def huber(y_true, y_pred):
    z = y_pred * y_true
    loss = -4 * z
    loss[z >= -1] = (1 - z[z >= -1]) ** 2
    loss[z >= 1.] = 0
    return loss

## with help from scikit-learn modules on loss functions
def plot_loss_functions():
    x_min, x_max = -2, 2
    xx = np.linspace(x_min, x_max, 100)
    lw = 2
    plt.plot([x_min, 0, 0, x_max], [1, 1, 0, 0], color='green', lw=lw)
    plt.plot(xx, np.where(xx < 1, 1 - xx, 0), color='red', lw=lw)
    plt.plot(xx, huber(xx, 1), color='blue', lw=lw)
    plt.ylim((0, 4))
    plt.show()
    return

def get_data(n, dim):
    np.random.seed(DUE_DATE)
    C = np.array([[0., -0.23], [0.83, .23]])
    temp_vals = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    X = []
    for i in range(len(temp_vals)):
        temp_vals[i] = temp_vals[i] / np.linalg.norm(temp_vals)
        X.append(np.append(temp_vals[i], [1], None)) ## adding extra dim
    X = np.array(X)
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    rng = np.random.RandomState(random.seed(DUE_DATE))
    permutation = rng.permutation(len(X))
    X, y = X[permutation], y[permutation]
    return X, y

def dot_product(inputs, weights):
    ''' computes the dot product of the input vector with the weight vector
        assumes that both vectors are of the same dimensionality '''
    if len(inputs) == 1:
        return inputs*weights
    return sum(value * weight for value, weight in zip(inputs, weights))

def huber_hinge_loss(u):
    if u > (1 + h):
        return 0
    elif abs(1 - u) <= h:
        return ((1 + h - u)**2)/(4 * h)
    elif u < (1 - h):
        return 1-u

def grad_huber_hinge_loss(u):
    if u > (1 + h):
        return 0
    elif abs(1 - u) <= h:
        return ((1 + h - u))/(2 * h)
    elif u < (1 - h):
        return 1

def compute_obj(X, y, w, c=1):
    ''' F(w) = (l-2 norm(w))^2 +
                    (c/n)sum from i to n of(huber-hinge-loss(yi, f(xi)))
        where f(xi) = <w^T, x>
        let n = 2 + 1 = 3
        returns update w '''
    # val = [np.linalg.norm(w)**2]
    val = 0
    scalar = float(c)/3
    for i in range(len(X)):
        ## X[i] is a vector with dimension 3
        u = y[i]*dot_product(X[i], w)
        ## now plug in u into the loss function
        val += huber_hinge_loss(u)
    return [(np.linalg.norm(w)**2) + (scalar*val)]

def compute_grad(X, y, w, c=1):
    val = 0
    scalar = float(c)/3
    for i in range(len(X)):
        ## X[i] is a vector with dimension 3
        u = y[i]*dot_product(X[i], w)
        ## now plug in u into the loss function
        val += grad_huber_hinge_loss(u)
    return [val]

def my_gradient_descent(X, y, w=[0.0], eta=0.1, \
                                max_iter=1000, convergence=False):
    # w = [0]*len(X[0]+1) ## if in multiple dimensions
    ## note that comments show improvement in speed of calculation
    old_update = w
    iterations = 0
    for t in range(max_iter):
        # w = w - [eta*item for item in compute_grad(w)] ## if in multiple dim
        # [w] = [w] - [eta*item for item in compute_grad(X, y, \
                                                    # compute_obj(X, y, w))]
        # w = map(sub, w, eta*compute_grad(X, y, compute_obj(X, y, w)))
        # w = map(sub, w, [i * eta for i in \
        #                         compute_grad(X, y, compute_obj(X, y, w))])
        # w = map(sub, w, map(lambda x: x * eta, \
        #                             compute_grad(X, y, compute_obj(X, y, w))))
        new_update = [eta * i for i in compute_grad(X, y, compute_obj(X, y, w))]
        ## for faster convergence
        if t == 0:
            w = [float(w[0]) - float(new_update[0])]
            continue
        if old_update[0] != new_update[0]:
            w = [float(w[0]) - float(new_update[0])]
            old_update[0] = new_update[0]
            iterations = t
        else:
            if convergence:
                break
            else:
                continue
    return w, iterations

## takes a relatively long amount of time to iterate through
def my_svm(X, y, w=[0.0], eta=0.1, max_iter=1000):
    vector = []
    for t in range(max_iter):
        vector.append(my_gradient_descent(X, y, \
                                    max_iter=max_iter, convergence=True))
        w = min(vector)
        ## to see what iteration it's on
        # print "iteration", t+1
    return w[0]


if __name__ == "__main__":
    np.random.seed(DUE_DATE)
    # plot_loss_functions()

    ### train
    # X_train, y_train = get_data(500, 2)
    # print X_train
    # print y_train
    # my_gradient_descent(X_train, y_train, eta=1.1, max_iter=1000)
    # w = my_svm(X_train, y_train, max_iter=100)

    ### test
    # X_test, y_test = get_data(500, 2)
    # w = my_svm(X_test, y_test, w=w, max_iter=200)
    # print X_test
    # print y_test
