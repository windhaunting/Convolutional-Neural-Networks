import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype



    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #pass

    C, H, W = input_dim
    F, HH, WW = num_filters, filter_size, filter_size
    
    self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
    self.params['b1'] = np.zeros(F) 

    self.params['W2'] = weight_scale * np.random.randn(F*H/2*W/2, hidden_dim)                 # HH/2*WW/2
    self.params['b2'] = np.zeros(hidden_dim) 

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)                 #HH/2*WW/2
    self.params['b3'] = np.zeros(num_classes) 


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
	
    scores = None

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #pass
    outPool, cachePool = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

    outAffine, cacheAffine = affine_relu_forward(outPool, W2, b2)
    scores, cache = affine_forward(outAffine, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    #pass
    loss, dscore = softmax_loss(scores, y)
    dAffine, grads['W3'], grads['b3'] = affine_backward(dscore, cache)
    dPool, grads['W2'], grads['b2'] = affine_relu_backward(dAffine, cacheAffine)

    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dPool, cachePool)

    loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))   # add regularization

    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


#pass


# consider spatial batch_norm or not
class ThreeLayerConvNetBatchNorm(object):
  """
     conv - batch_norm -- relu - 2x2 max pool - affine -batch_norm  -- relu - affine - softmax
  """
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, vanilla_batch_norm = False, spatial_batch_norm = False):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    --vanilla_batch_norm
    --spatial_batch_norm
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    #added by fubao
    self.batch_norm = {}
    self.vanilla_batch_norm = vanilla_batch_norm
    self.spatial_batch_norm = spatial_batch_norm

    if self.vanilla_batch_norm:
        self.vanilla_bn_params = {'mode': 'train'}

    if self.spatial_batch_norm:
        self.spatial_bn_params = {'mode': 'train'}


    C, H, W = input_dim
    F, HH, WW = num_filters, filter_size, filter_size
    
    self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
    self.params['b1'] = np.zeros(F) 

    if self.vanilla_batch_norm:
        self.params['gamma1'] = np.ones(F*H*W)
 	self.params['beta1'] = np.ones(F*H*W)

    self.params['W2'] = weight_scale * np.random.randn(F*H/2*W/2, hidden_dim)                 # HH/2*WW/2
    self.params['b2'] = np.zeros(hidden_dim) 

    if self.spatial_batch_norm:
        self.params['gamma2'] = np.ones(hidden_dim)
 	self.params['beta2'] = np.ones(hidden_dim)

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)                 #HH/2*WW/2
    self.params['b3'] = np.zeros(num_classes) 

  # for vanilla and spatial batch_normalization usage
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


    if self.vanilla_batch_norm:
        self.vanilla_bn_params['mode'] = mode

    if self.spatial_batch_norm:
        self.spatial_bn_params['mode'] = mode

    scores = None
 

    outPool, cachePool = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

    outAffine, cacheAffine = affine_relu_forward(outPool, W2, b2)
    scores, cache = affine_forward(outAffine, W3, b3)


    if y is None:
      return scores

    loss, grads = 0, {}
 
    loss, dscore = softmax_loss(scores, y)
    dAffine, grads['W3'], grads['b3'] = affine_backward(dscore, cache)
    dPool, grads['W2'], grads['b2'] = affine_relu_backward(dAffine, cacheAffine)

    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dPool, cachePool)

    loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))   # add regularization

    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3



    return loss, grads



#added by fubao using leaky relu
class ThreeLayerConvNetLeakyRelu(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - leakyRelu - 2x2 max pool - affine - leakyRelu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype



    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #pass

    C, H, W = input_dim
    F, HH, WW = num_filters, filter_size, filter_size
    
    self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
    self.params['b1'] = np.zeros(F) 

    self.params['W2'] = weight_scale * np.random.randn(F*H/2*W/2, hidden_dim)                 # HH/2*WW/2
    self.params['b2'] = np.zeros(hidden_dim) 

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)                 #HH/2*WW/2
    self.params['b3'] = np.zeros(num_classes) 


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
	
    scores = None

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #pass
    outPool, cachePool = conv_leakyRelu_pool_forward(X, W1, b1, conv_param, pool_param)

    outAffine, cacheAffine = affine_leakyRelu_forward(outPool, W2, b2)
    scores, cache = affine_forward(outAffine, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    #pass
    loss, dscore = softmax_loss(scores, y)
    dAffine, grads['W3'], grads['b3'] = affine_backward(dscore, cache)
    dPool, grads['W2'], grads['b2'] = affine_leakyRelu_backward(dAffine, cacheAffine)

    dx, grads['W1'], grads['b1'] = conv_leakyRelu_pool_backward(dPool, cachePool)

    loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))   # add regularization

    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


#pass


