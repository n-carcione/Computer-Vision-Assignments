import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'layer2')
initialize_weights(hidden_size,hidden_size,params,'layer3')
initialize_weights(hidden_size,1024,params,'output')

total_loss = np.zeros((max_iters))
# should look like your previous training loops
for itr in range(max_iters):
    total_loss[itr] = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        
        # forward
        h1 = forward(xb,params,'layer1',activation=relu)
        h2 = forward(h1,params,'layer2',activation=relu)
        h3 = forward(h2,params,'layer3',activation=relu)
        im_out = forward(h3,params,'output',activation=sigmoid)
        # loss
        total_loss[itr] += np.sum((im_out-xb)**2)
        # backward
        delta1 = -2*(xb-im_out)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        delta5 = backwards(delta4,params,'layer1',relu_deriv)
        
        # apply gradient
        params['m_Wlayer1'] = 0.9*params['m_Wlayer1'] - learning_rate*params['grad_Wlayer1']
        params['Wlayer1'] += params['m_Wlayer1']
        params['m_Wlayer2'] = 0.9*params['m_Wlayer2'] - learning_rate*params['grad_Wlayer2']
        params['Wlayer2'] += params['m_Wlayer2']
        params['m_Wlayer3'] = 0.9*params['m_Wlayer3'] - learning_rate*params['grad_Wlayer3']
        params['Wlayer3'] += params['m_Wlayer3']
        params['m_Woutput'] = 0.9*params['m_Woutput'] - learning_rate*params['grad_Woutput']
        params['Woutput'] += params['m_Woutput']
        
        params['m_blayer1'] = 0.9*params['m_blayer1'] - learning_rate*params['grad_blayer1']
        params['blayer1'] += params['m_blayer1']
        params['m_blayer2'] = 0.9*params['m_blayer2'] - learning_rate*params['grad_blayer2']
        params['blayer2'] += params['m_blayer2']
        params['m_blayer3'] = 0.9*params['m_blayer3'] - learning_rate*params['grad_blayer3']
        params['blayer3'] += params['m_blayer3']
        params['m_boutput'] = 0.9*params['m_boutput'] - learning_rate*params['grad_boutput']
        params['boutput'] += params['m_boutput']

    total_loss[itr] /= train_x.shape[0]
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss[itr]))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
#Plot the loss curve
plt.figure()
plt.plot(total_loss)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
#Note: each class has 100 samples --> i.e., class 1 is samples 0-99, class 2 is 100-199,...
h1_v = forward(valid_x,params,'layer1',activation=relu)
h2_v = forward(h1_v,params,'layer2',activation=relu)
h3_v = forward(h2_v,params,'layer3',activation=relu)
im_val = forward(h3_v,params,'output',activation=sigmoid)
class_nums = np.array([3, 3, 3, 3, 8, 8, 8, 8, 17, 17, 17, 17,
                       23, 23, 23, 23, 34, 34, 34, 34])
fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
ctr = 0
for ax in grid:
    c = class_nums[ctr]
    if ctr % 4 == 0:
        im = valid_x[100*c,:].reshape(32,32)
    elif ctr % 4 == 1:
        im = im_val[100*c,:].reshape(32,32)
    elif ctr % 4 == 2:
        im = valid_x[100*c+1,:].reshape(32,32)
    elif ctr % 4 == 3:
        im = im_val[100*c+1,:].reshape(32,32)
    ax.imshow(im,cmap='gray')
    ctr += 1
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
total_psnr = 0
for image in np.arange(valid_x.shape[0]):
    total_psnr += peak_signal_noise_ratio(valid_x[image,:],im_val[image,:])
avg_psnr = total_psnr / valid_x.shape[0]
print("avergae PSNR: {:.2f}".format(avg_psnr))
