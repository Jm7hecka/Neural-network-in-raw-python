import numpy as np
from keras.datasets import mnist #include dataset 
from matplotlib import pyplot 

def sigmoid(x): #activative function
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): #find the rate of change of sigmoid function
    return x * (1-x)       #finding derivation is limit of difference quotient, which is f(x+h) - f(x) / h
                           #in other words it is dy/dx, if you know maths you will know the m of gradient = dy/dx
                           #this definition will be useful later

def relu(x): #activation function
    return np.maximum(0, x)

def relu_derivative(x): #find the rate of change of relu function
    return (x > 0).astype(float)

def mean_squared_error(x, y):
    return np.mean((x - y) **2) 

class Convulational_neural_network():
    def __init__(self, train_data, train_label, test_data, test_label, learning_rate):
        self.train_X = train_data.reshape(-1, 28*28)/ 255 #turn all train image into 1D array
        self.imgdata = test_data
        self.test_X = test_data.reshape(-1, 28*28) / 255 #turn all test image into 1D array
        self.train_y = train_label
        self.test_y = test_label
        self.input = 28*28 #define input neuron 
        self.hidden = 256 #define neuron in hidden layer
        self.output = 10 #define output neuron(1-10)
        self.learning_rate = learning_rate 
        self.weight_1 = 0.02 * np.random.random((self.input, self.hidden)) - 0.01 #define weight for input layer
        self.weight_2 = 0.2 * np.random.random((self.hidden, self.output)) - 0.1 #defile weight for hidden layer
        self.bias_1 = np.zeros((1, self.hidden)) #define bias for input layer  
        self.bias_2 = np.zeros((1, self.output)) #define bias for hidden layer

    def transformLabels(self, labels):     #turn label into one-hot data
        zeroArray = [0,0,0,0,0,0,0,0,0,0]  #for example 1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        transformed = []
        for i in range(len(labels)):
            label = labels[i]
            transformed.append(zeroArray[0:label] + [1] + zeroArray[label+1:])
        return transformed

    def reform(self, array): #turn one-hot label into result
        max = 0              
        num = 0
        for i in range(10): 
            if array[i] > max: 
                num = i
                max = array[i]
        return num
    
   
    def forward(self, x): #forward propagation, find output from input value
        self.output = []
        self.layer_1 = relu(np.dot(x, self.weight_1) + self.bias_1) #hidden layer process
        self.output = sigmoid(np.dot(self.layer_1, self.weight_2 + self.bias_2)) #output layer process
    
    def backward(self, x, y): #changing weight and bias value base on error rate 
        x = x.reshape(-1, len(x))
        error_2 = y - self.output #find error rate of layer 2 by doing subtraction between actual output and predicted output
        gradient_decent_2 = error_2 * sigmoid_derivative(self.output) #finding gradient decent(mx) of layer 2
        error_1 = gradient_decent_2.dot(self.weight_2.T) #find error rate of layer 1
        gradient_decent_1 = error_1 * relu_derivative(self.layer_1) #find gradient decent of layer 1
        self.weight_2 += self.layer_1.T.dot(gradient_decent_2) * self.learning_rate #update weight value of layer 2
        self.weight_1 += x.T.dot(gradient_decent_1) * self.learning_rate #update weight value of layer 1
        self.bias_2 += np.sum(gradient_decent_2, axis=0) * self.learning_rate #update bias value of layer 2
        self.bias_1 += np.sum(gradient_decent_1, axis=0) * self.learning_rate #update bias value of layer 1 

    def train(self, epochs): 
        print('Training data...')
        y = self.transformLabels(self.train_y) #transform all train label into one-hot value
        for i in range(epochs): #epochs represents how many times for all train data will be tested
            for j in range(len(self.train_X)):
                self.forward(self.train_X[j])
                self.backward(self.train_X[j], y[j])
            if i % 1== 0: #find loss for each 100 epochs
                print(f"Epoch {i}: Loss = {mean_squared_error(self.output, y)}")
                
        print(f"Training done. Total Epochs: {epochs} \n Final Loss = {mean_squared_error(self.output, y)}")
    
    def test_data(self): #find the accuracy
        correct = 0
        X = self.test_X
        for i in range(1000): 
            self.forward(X[i])
            result = self.reform(self.output[0])
            if result == self.test_y[i]: #find if test result is correct or not
                correct += 1
            
            print(f"Test case #{i+1}: Predict output = {result}, Actual output = {self.test_y[i]} ")
            print(f'Accuracy: {round(((correct/( i+1)) * 100), 2)}%')

    def showimg(self, X):  
        pyplot.imshow(X, cmap=pyplot.get_cmap('gray'))
        pyplot.show()    
       
    def predict(self, num): #find result and show image used
        self.forward(self.test_X[num])
        result = self.reform(self.output[0])
        print(f"Image shows: {result}")
        self.showimg(self.imgdata[num])


if __name__ == '__main__':
    (train_X, train_y),(test_X, test_y) = mnist.load_data()
    ml = Convulational_neural_network(train_X, train_y, test_X, test_y, 0.01)
    ml.train(1)
    ml.test_data()
    while True:
        number = int(input("Enter number from 1-10000:"))
        ml.predict(number)
