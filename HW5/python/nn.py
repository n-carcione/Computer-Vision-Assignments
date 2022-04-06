import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    # Which equation for variance do we use, the one from the homework or the one
    #from the paper?  The one from the paper gives std dev in expected range
    # var = 2 / (in_size + out_size)
    var = np.sqrt(6) / np.sqrt(in_size + out_size)
    W = np.random.uniform(-var,var,(in_size,out_size))

    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):

    res = 1 / (1+np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """

    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    pre_act = X @ W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = np.zeros(x.shape)

    for row in np.arange(x.shape[0]):
        x_vec = np.exp(x[row,:] - np.max(x[row,:]))
        res[row,:] = x_vec / np.sum(x_vec)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss = -np.sum(y*np.log(probs))

    guesses = np.argmax(probs,1)
    guess_matrix = np.zeros(probs.shape)
    for row in np.arange(guesses.shape[0]):
        guess_matrix[row,guesses[row]] = 1
    acc = np.sum(y * guess_matrix) / y.shape[0]

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """

    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    deriv = activation_deriv(post_act)
    grad_X = (deriv * delta) @ W.T
    grad_W = X.T @ (deriv * delta)
    grad_b = np.sum(deriv*delta,axis=0)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    
    #Create a vector that contains all possible indices randomly ordered
    num_samples = x.shape[0]
    inds = np.random.choice(num_samples, size=(num_samples), replace=False)
    while inds.size != 0:
        #Draw from the random index vector the first batch_size indices and get the
        #corresponding x and y sample
        x_batch = x[inds[0:batch_size]]
        y_batch = y[inds[0:batch_size]]
        #Remove the first batc_size indices from the index vector since they
        #have now been used
        inds = np.delete(inds,np.arange(batch_size))
        #Append the x and y batch arrays to the batches list
        batches.append([x_batch, y_batch])
    
    return batches
