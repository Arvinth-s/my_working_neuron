

#print('om namo narayana')



#This model is the implementation of mutilayer neural networks using stochastic gradient descent algorithm
#The user has the flexibility to choose the number of layers and number of neuron in a single layer
#For each epoch the performance of the batch is tested with the test data provided to the function
#implemetation for future:This specific model is linear, this model can be converted to sigmoid or tanh model with some modifications in the activation and bacckprop functions

import numpy as np
from time  import sleep
import os

def move_down(lines):
    for i in range(lines):
        print('\n')

def clear():
    s = '\n' * 15
    print(s)

def title(_title_, lines, ch):
    space = len(_title_)
    space = '*'*(28 - space)
    if ch == 1:
        print('*******************************************************************************************************************************************************************\n')
        print('*******************************************************************', _title_ ,space+'******************************************************************\n')
        print('*******************************************************************************************************************************************************************\n')
    else:
        print('*******************************************************************', _title_ ,space+'******************************************************************\n')
    for i in range(lines):
        print('\n')
        sleep(0.2)
def get_random_W(dim_layer):
    W = []
    for layer in range(len(dim_layer) - 1):
        W.append(np.random.randint(1, 4, [dim_layer[layer], dim_layer[layer+1]]) *1.0)
    return W
    #W is a 3D list



def get_random_bias(dim_layer):
    bias = []

    for layer in range(len(dim_layer) - 1):
        bias.append(np.random.randint(1, 3, [dim_layer[layer+1]]) * 1.0)
    return bias
    #bias is a 2D list




def get_activations(X, W, bias):
    X = np.array(X)
    activations = []
    activations.append(X)
    for layer in range(len(W)):
        activations.append(np.dot(W[layer].T, activations[layer].reshape(activations[layer].shape[0], 1)).T[0] + bias[layer])
    return activations
    #activations is a 2D list




def get_prediction(X, W, bias):
    activations = X
    for layer in range(len(W)):
        activations = np.dot(W[layer].T, activations.reshape(activations.shape[0], 1)).T[0] + bias[layer]
    return activations




def back_prop(X, Y, W, b, activations, dim_layer, alpha):
    A = W[:]
    l = len(dim_layer) - 1
    matrix = np.array(activations[l] - Y)
    for i in range(len(dim_layer) - 1):
        W[l - 1 - i] = A[l - 1 - i] - alpha *  np.dot(activations[l - i - 1].reshape(activations[l - 1 - i].shape[0], 1) , matrix.reshape(1, len(matrix))) 
        b[l - 1 - i] -= alpha * matrix
        matrix = np.dot(matrix.reshape(1, matrix.shape[0]), A[l - 1 - i].T)[0]
    return(W, b)




def get_error(X, Y, W ,bias):
    #X is a single input numpy array and Y is the single output numpy array
    predictions = get_prediction(X, W, bias)
    error = sum((predictions - Y)**2)
    return error




def evaluate(input_test_data, output_test_data, W, bias):
    predictions = np.array([get_prediction(input_test_data[i], W, bias) for i in range(input_test_data.shape[0])])
    error = np.sum((predictions/pow(input_test_data.shape[0], 0.5) - output_test_data/pow(input_test_data.shape[0], 0.5))**2)
    return error




#input_data is a 3D nupmy array
def mini_SGD(input_data, output_data, input_test_data, output_test_data, dict_ ={}):
    

    log = {'epochs': 10, 'maximum_error':10e10 , 'dim_layer': [2, 3, 2], 'alpha' : 0.01}
    for key in log.keys():
        if(dict_[key] != None):
            log[key] = dict_[key]
    dim_layer  = log.get('dim_layer')
    alpha = log.get('alpha')
    error_list = []
    weight_list = []
    
    for epoch in range(log.get('epochs')):

        W = get_random_W(dim_layer)
        bias = get_random_bias(dim_layer)

        for i in range(input_data.shape[0]):
            activations = get_activations(input_data[i], W, bias)
            W, bias = back_prop(input_data[i], output_data[i], W, bias, activations, dim_layer, log.get('alpha'))
        batch_error = evaluate(input_test_data, output_test_data, W, bias)
        error_list.append(batch_error)
        weight_list.append((W, bias))
        print('epoch_number:{} error:{}'.format(epoch + 1, batch_error))
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx min error: {}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n'.format(min(error_list)))
    sleep(1)
    #print('$$$$$Weights and bias corresponding to the minimum error.{}.format$$$$'.format(weight_list[error_list.index(min(error_list))]))
    return weight_list[error_list.index(min(error_list))], min(error_list)




input_data = np.random.randint(1, 100, [10000, 2])
output_data = input_data * 3.0
input_test_data = np.random.randint(1, 100, [100, 2])
output_test_data = input_test_data * 3.0








def f1(alpha, epochs):
    #function definitinon f1(x) = 2 * x
    title('        FUNCTION    f(X) = 2*X           ', 1, 0)
    input_data = np.random.randint(1, 100, [1000, 1])
    output_data = input_data * 2
    input_test_data = np.random.randint(1, 100, [100, 1])
    output_test_data = input_test_data * 2
    title('WITH 1 HIDDEN LAYER', 1, 0)
    print('train data size', input_data.shape[0])
    print('test data size', input_test_data.shape[0])
    sleep(3)
    A, error = mini_SGD(input_data, output_data, input_test_data, output_test_data, {'epochs':epochs, 'maximum_error':10e10, 'dim_layer': [1, 1, 1], 'alpha' : alpha })
    title('WITH 1 HIDDEN LAYER', 0, 0)
    title('2 PERCEPTRON IN HIDDEN LAYER', 1, 0)
    A, error = mini_SGD(input_data, output_data, input_test_data, output_test_data, {'epochs':epochs, 'maximum_error':10e10, 'dim_layer': [1, 2, 1], 'alpha' : alpha })
    title('WITH 2 HIDDEN LAYERS', 1, 0)
    A, error = mini_SGD(input_data, output_data, input_test_data, output_test_data, {'epochs':epochs, 'maximum_error':10e10, 'dim_layer': [1, 1, 1, 1], 'alpha' : alpha })





def f2(alpha, epochs):
    #function definition = f2(x, y) = 3*x + 4*y
    title('        FUNCTION    f(X, Y) = 3*X + 4*Y          ', 1, 0)
    input_data = np.random.randint(1, 100, [1000, 2])
    output_data = (input_data.T[0] * 3 + input_data.T[1] * 4).reshape(input_data.shape[0], 1)
    input_test_data = np.random.randint(1, 100, [100, 2])
    output_test_data = (input_test_data.T[0] * 3 + input_test_data.T[1] * 4).reshape(input_test_data.shape[0], 1)
    title('WITH 1 HIDDEN LAYER', 1, 0)
    print('train data size', input_data.shape[0])
    print('test data size', input_test_data.shape[0])
    sleep(3)
    A, error = mini_SGD(input_data, output_data, input_test_data, output_test_data, {'epochs':epochs, 'maximum_error':10e10, 'dim_layer': [2, 1, 1], 'alpha' : alpha })
    title('WITH 1 HIDDEN LAYER', 0, 0)
    title('2 PERCEPTRON IN HIDDEN LAYER', 1, 0)
    A, error = mini_SGD(input_data, output_data, input_test_data, output_test_data, {'epochs':epochs, 'maximum_error':10e10, 'dim_layer': [2, 2, 1], 'alpha' : alpha })
    title('WITH 2 HIDDEN LAYERS', 1, 0)
    A, error = mini_SGD(input_data, output_data, input_test_data, output_test_data, {'epochs':epochs, 'maximum_error':10e10, 'dim_layer': [2, 1, 1], 'alpha' : alpha })

clear = lambda : os.system('clear')

if __name__=='__main__':

    move_down(10)
    
    title('NEURON NETWORK', 5, 1)

    f1(0.0000001, 100)#with alpha value 0.0000001 and numbee of epochs 10

    f2(0.0000001, 100)#with alpha value 0.0000001

    f1(0.00001, 50)

    f1(0.00005, 100)

    f1(0.00003, 10)

    f1(0.00003, 10)


    #best for f1 0.00005 for 1 and 0.00002 for next 2

    #best for f2 is 0.00003 







