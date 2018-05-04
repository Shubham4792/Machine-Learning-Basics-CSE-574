'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''
import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
import datetime

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return  1/(1 + np.exp(-z))

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    x = np.ones((training_data.shape[0],1))
    train_data_with_bias = np.append(training_data,x,1)
    hidden_output = sigmoid(np.dot(w1,np.transpose(train_data_with_bias)))
    x = np.ones((1,training_data.shape[0]))
    hidden_output_with_bias = np.append(hidden_output,x,0)
    o = np.transpose(sigmoid(np.dot(w2,hidden_output_with_bias)))
    
    y = []
    for i in range(0,train_label.shape[0]):
        for j in range(0,n_class):
           x = np.zeros((1,n_class))[0]
           if(train_label[i] == j):
               x[j] = 1
               y.append(x)
               break
    y = np.asarray(y)
    total = 0
    total += np.sum(y*np.log(o))
    total += np.sum((1-y)*np.log(1-o))
    obj_val = -total/train_data_with_bias.shape[0]    
    hidden_output_with_bias = np.transpose(hidden_output_with_bias)
    
    hl_err = o - y
    hl_err = np.transpose(hl_err)
    gradW2 =  np.dot(hl_err, hidden_output_with_bias)
    gradW2 = gradW2/train_data_with_bias.shape[0]
    
    tmp = w2[:,0:w1.shape[0]]
    errSum = np.transpose(np.dot(np.transpose(hl_err),tmp))
    first =  1 - hidden_output
    second = 1 - first 
    prod =first * second * errSum
    gradW1 = np.dot(prod,train_data_with_bias)
    gradW1 = gradW1/train_data_with_bias.shape[0]
    
    reg = lambdaval * (np.sum(np.sum(w1*w1)) + np.sum(np.sum(w2*w2)))
    reg = reg/(train_data_with_bias.shape[0]*2)
    obj_val = obj_val + reg
    obj_grad = np.concatenate((gradW1.flatten(), gradW2.flatten()),0)
    return (obj_val, obj_grad)


# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here
    x = np.ones((data.shape[0],1))
    data_with_bias = np.append(data,x,1)
    hidden_output = sigmoid(np.dot(w1,np.transpose(data_with_bias)))
    hidden_output = np.transpose(hidden_output)
    hidden_output_with_bias = np.concatenate((hidden_output,x),1)
    output = sigmoid(np.dot(hidden_output_with_bias,np.transpose(w2)))
    labels = np.argmax(output,1)
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
begin = datetime.datetime.now()

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
end = datetime.datetime.now()

print('\nTime taken(in seconds): '+str((end - begin).total_seconds()))