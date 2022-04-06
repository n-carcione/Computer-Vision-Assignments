import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 36
learning_rate = 5e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,36,params,'output')

#Visualize first layer weights immediately after initialization
fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

im_num = 0
for ax in grid:
    im = params['Wlayer1'][:,im_num].reshape((32,32))
    ax.imshow(im)
    im_num += 1
plt.show()

total_loss = np.zeros((max_iters))
total_acc = np.zeros((max_iters))
valid_loss = np.zeros((max_iters))
valid_acc = np.zeros((max_iters))
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        # forward
        h1_t = forward(xb,params,'layer1',activation=sigmoid)
        probs_t = forward(h1_t,params,'output',activation=softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss_t, acc_t = compute_loss_and_acc(yb,probs_t)
        total_loss[itr] += loss_t
        total_acc[itr] += acc_t
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

    total_acc[itr] /= batch_num
    total_loss[itr] /= train_x.shape[0]

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss[itr],total_acc[itr]))

    # run on validation set and report accuracy! should be above 75%
    h1_valid = forward(valid_x,params,'layer1',activation=sigmoid)
    probs_valid = forward(h1_valid,params,'output',activation=softmax)
    valid_loss[itr], valid_acc[itr] = compute_loss_and_acc(valid_y,probs_valid)
    valid_loss[itr] /= valid_x.shape[0]

plt.figure()
plt.plot(total_acc, label='Test')
plt.plot(valid_acc,label='Validation')
plt.title("Accuracy Curves for Learning Rate = {:.1e}".format(learning_rate))
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.legend()

plt.figure()
plt.plot(total_loss,label='Test')
plt.plot(valid_loss,label='Validation')
plt.title("Loss Curves for Learning Rate = {:.1e}".format(learning_rate))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# run on test set and report accuracy
h1_test = forward(test_x,params,'layer1',activation=sigmoid)
probs_test = forward(h1_test,params,'output',activation=softmax)
test_loss, test_acc = compute_loss_and_acc(test_y,probs_test)

print('Validation accuracy: ',valid_acc[itr])
print('Test accuracy: ',test_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
#****COMMENTED OUT TO PRESERVE BEST NETWORK***
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
# visualize weights here
fig2 = plt.figure()
grid2 = ImageGrid(fig2, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

im_num2 = 0
for ax2 in grid2:
    im2 = params['Wlayer1'][:,im_num2].reshape((32,32))
    ax2.imshow(im2)
    im_num2 += 1
plt.show()

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
#Get the guessed and actual classes
guesses = np.argmax(probs_test,1)
actuals = np.argmax(test_y,1)
#Iterate through each sample to and add 1 to the confusion matrix at the spot
#where the actual and guessed classes are (y axis actual, x axis guessed)
for sample in np.arange(len(guesses)):
    confusion_matrix[actuals[sample],guesses[sample]] += 1

#Visualize the confusion matrix
import string
plt.figure()
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()