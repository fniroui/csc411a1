""" Methods for doing logistic regression."""

import numpy as np
from utils import *

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    # M = weights.shape[1]
    # N = data.shape[0]
    # y = np.zeros((N,1))
    data2 = np.ones((data.shape[0], data.shape[1] + 1))
    data2[:, :-1] = data

    x= np.dot(data2,weights)

    return sigmoid(x)

    # for x in range(0,N):
    #     y[x,1] = sigmoid(data2*weights)
    #
    # return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    ce = np.take(-1*np.dot(targets.T,np.log(y))-np.dot((1-targets).T,np.log(1-y)),0)

    correct = 0
    for x in range(0, len(targets)):
        if round(y[x, 0]) == round(targets[x, 0]):
            correct += 1

    frac_correct = float(correct)/float(len(targets))

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        f = -1*np.sum(targets*np.log(y)+(1-targets)*np.log(1-y))

        data2 = np.ones((data.shape[0], data.shape[1] + 1))
        data2[:, :-1] = data
        df = np.dot(data2.T,y-targets)

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    y = logistic_predict(weights, data)
    a = hyperparameters['weight_decay']
    data2 = np.ones((data.shape[0], data.shape[1] + 1))
    data2[:, :-1] = data

    f = -1*np.sum(targets*np.log(y)+(1-targets)*np.log(1-y))+np.dot(weights.T,weights)*a/2+np.log(np.sqrt(2*np.pi/a))
    df = np.dot(data2.T, y - targets) + a*weights
    df[len(weights)-1] = np.sum(y-targets)

    
    return f, df
