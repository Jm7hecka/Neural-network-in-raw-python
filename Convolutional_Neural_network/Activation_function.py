import numpy as np

def sigmoid(x): #output in range(0, 1)
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): #find the rate of change of sigmoid function
    return x * (1-x)       #finding derivation is limit of difference quotient, which is f(x+h) - f(x) / h
                        #in other words it is dy/dx, if you know maths you will know the m of gradient = dy/dx
                        #this definition will be useful later

def relu(x): #turn negative value into 0
    return np.maximum(0, x)

def relu_derivative(x): #find the rate of change of relu function
    return (x > 0).astype(float)
