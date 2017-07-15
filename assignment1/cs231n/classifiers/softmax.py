import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
    
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.amax(scores)
    correct_class_score = scores[y[i]]
    norm = np.sum(np.exp(scores))

    loss += -np.log(np.exp(correct_class_score)/norm)
    
    for j in xrange(num_classes):
        dW[:,j] += -X[i].T*(1*(j == y[i]) - np.exp(scores[j])/norm)
  
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW /= num_train
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  scores = X.dot(W)
    
  # Subtract maximum of each row
  maxes = np.amax(scores, axis=1)
  maxes = maxes[:, np.newaxis]
  scores = scores - maxes
  norms = np.sum(np.exp(scores), axis = 1)
  softmax = np.divide(np.exp(scores[np.arange(num_train), y]), norms)
  
  logloss = np.sum(-np.log(softmax))              
  ##################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  norms = np.power(norms, -1)
  scores = np.exp(scores).T
  scores = scores * norms
  
  scores[y, np.arange(num_train)] -= 1
  dW = np.dot(scores, X).T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  logloss /= num_train
  logloss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  
  return logloss, dW  
