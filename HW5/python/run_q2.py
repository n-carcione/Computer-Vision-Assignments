import numpy as np
# you should write your functions in nn.py
from nn import *
from util import *


# fake data
# feel free to plot it in 2D
# what do you think these 4 classes are?
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])
# we will do XW + B
# that implies that the data is N x D

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# turn to one_hot
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1

# parameters in a dictionary
params = {}

# Q 2.1
# initialize a layer
initialize_weights(2,25,params,'layer1')
initialize_weights(25,4,params,'output')
assert(params['Wlayer1'].shape == (2,25))
assert(params['blayer1'].shape == (25,))

#expect 0, [0.05 to 0.12]
print("{}, {:.2f}".format(params['blayer1'].sum(),params['Wlayer1'].std()**2))
print("{}, {:.2f}".format(params['boutput'].sum(),params['Woutput'].std()**2))

# Q 2.2.1
# implement sigmoid
test = sigmoid(np.array([-1000,1000]))
print('should be zero and one\t',test.min(),test.max())
# implement forward
h1 = forward(x,params,'layer1')
print(h1.shape)
# Q 2.2.2
# implement softmax
probs = forward(h1,params,'output',softmax)
# make sure you understand these values!
# positive, ~1, ~1, (40,4)
print(probs.min(),min(probs.sum(1)),max(probs.sum(1)),probs.shape)

# Q 2.2.3
# implement compute_loss_and_acc
loss, acc = compute_loss_and_acc(y, probs)
# should be around -np.log(0.25)*40 [~55] and 0.25
# if it is not, check softmax!
print("{}, {:.2f}".format(loss,acc))

# here we cheat for you
# the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
delta1 = probs
delta1[np.arange(probs.shape[0]),y_idx] -= 1

# we already did derivative through softmax
# so we pass in a linear_deriv, which is just a vector of ones
# to make this a no-op
delta2 = backwards(delta1,params,'output',linear_deriv)
# Implement backwards!
backwards(delta2,params,'layer1',sigmoid_deriv)

# W and b should match their gradients sizes
for k,v in sorted(list(params.items())):
    if 'grad' in k:
        name = k.split('_')[1]
        print(name,v.shape, params[name].shape)

# Q 2.4
batches = get_random_batches(x,y,5)
# print batch sizes
print([_[0].shape[0] for _ in batches])
batch_num = len(batches)

# TRAINING LOOP
max_iters = 500
learning_rate = 1e-3
# with default settings, you should get loss < 35 and accuracy > 75%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        # forward
        h1_t = forward(xb,params,'layer1',activation=sigmoid)
        probs_t = forward(h1_t,params,'output',activation=softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss_t, acc_t = compute_loss_and_acc(yb,probs_t)
        total_loss += loss_t
        avg_acc += acc_t
        # backward
        delta1_t = probs_t
        delta1_t[np.arange(probs_t.shape[0]),np.argmax(yb,1)] -= 1
        delta2_t = backwards(delta1_t,params,'output',linear_deriv)
        delta3_t = backwards(delta2_t,params,'layer1',sigmoid_deriv)
        
        # apply gradient
        params['Wlayer1'] -= learning_rate*params['grad_Wlayer1']
        params['Woutput'] -= learning_rate*params['grad_Woutput']
        params['blayer1'] -= learning_rate*params['grad_blayer1']
        params['boutput'] -= learning_rate*params['grad_boutput']

    avg_acc /= batch_num 
    if itr % 100 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))


# Q 2.5 should be implemented in this file
# you can do this before or after training the network. 

#Find the gradients for the entire training set using the most recent weights
h1_t = forward(x,params,'layer1',activation=sigmoid)
probs_t = forward(h1_t,params,'output',activation=softmax)
# loss
# be sure to add loss and accuracy to epoch totals 
loss_t, acc_t = compute_loss_and_acc(y,probs_t)
# backward
delta1_t = probs_t
delta1_t[np.arange(probs_t.shape[0]),y_idx] -= 1
delta2_t = backwards(delta1_t,params,'output',linear_deriv)
delta3_t = backwards(delta2_t,params,'layer1',sigmoid_deriv)

# save the old params
import copy
params_orig = copy.deepcopy(params)

eps = 1e-6
for k,v in params.items():
    if '_' in k: 
        continue
    # we have a real parameter!
    # for each value inside the parameter
    #   add epsilon
    #   run the network
    #   get the loss
    #   compute derivative with central diffs
    
    #For each element in the weight matrices and bias vectors
    #increase the given (single) element by epsilon
    #forward propagate to find the loss (loss 1)
    #decrease the element by 2*epsilon (now a total change of -epsilon)
    #forward propagate again to find the loss (loss 2)
    #calculate (loss1 - loss2) / (2*epsilon) and place result into gradient matrix/vector
    if 'W' in k:    
        rows = v.shape[0]
        cols = v.shape[1]
        for i in np.arange(rows):
            for j in np.arange(cols):
                orig_val = v[i,j]
                params[k][i,j] += eps
                h1_n1 = forward(x,params,'layer1',activation=sigmoid)
                probs_n1 = forward(h1_n1,params,'output',activation=softmax)
                loss_1, acc_n1 = compute_loss_and_acc(y,probs_n1)
                
                params[k][i,j] -= 2*eps
                h1_n2 = forward(x,params,'layer1',activation=sigmoid)
                probs_n2 = forward(h1_n2,params,'output',activation=softmax)
                loss_2, acc_n2 = compute_loss_and_acc(y,probs_n2)
                
                grad = (loss_1 - loss_2) / (2*eps)
                params["grad_"+k][i,j] = grad
                params[k][i,j] = orig_val
    elif 'b' in k:
        rows = v.shape[0]
        for i in np.arange(rows):
            orig_val = v[i]
            params[k][i] += eps
            h1_n1 = forward(x,params,'layer1',activation=sigmoid)
            probs_n1 = forward(h1_n1,params,'output',activation=softmax)
            loss_1, acc_n1 = compute_loss_and_acc(y,probs_n1)
            
            params[k][i] -= 2*eps
            h1_n2 = forward(x,params,'layer1',activation=sigmoid)
            probs_n2 = forward(h1_n2,params,'output',activation=softmax)
            loss_2, acc_n2 = compute_loss_and_acc(y,probs_n2)
            
            grad = (loss_1 - loss_2) / (2*eps)
            params["grad_"+k][i] = grad
            params[k][i] = orig_val
    

total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]),np.abs(params_orig[k]))
        err = err.sum()
        print('{} {:.2e}'.format(k, err))
        total_error += err
# should be less than 1e-4
print('total {:.2e}'.format(total_error))
