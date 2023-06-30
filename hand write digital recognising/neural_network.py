import numpy as np
from keras.models import save_model
from matplotlib import pyplot 
import csv 

class Activation_function():
    def sigmoid(self, x): #output in range(0, 1)
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x): #find the rate of change of sigmoid function
        return x * (1-x)       #finding derivation is limit of difference quotient, which is f(x+h) - f(x) / h
                            #in other words it is dy/dx, if you know maths you will know the m of gradient = dy/dx
                            #this definition will be useful later

    def relu(self, x): #turn negative value into 0
        return np.maximum(0, x)

    def relu_derivative(self, x): #find the rate of change of relu function
        return (x > 0).astype(float)


class MLP():
    def __init__(self):
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.input = None
        self.hidden = None
        self.output = None
        self.weight_1 = None
        self.weight_2 = None
        self.weight_3 = None
        self.bias_1 = None
        self.bias_2 = None
        self.bias_3 = None
        self.function = Activation_function()
        self.twolayers = False

    def import_data(self, train_data, train_label, test_data, test_label):
        self.train_X = train_data.reshape(-1, 28*28)/ 255 #turn all train image into 1D array
        self.imgdata = test_data
        self.test_X = test_data.reshape(-1, 28*28) / 255 #turn all test image into 1D array
        self.train_y = train_label
        self.test_y = test_label

    def mean_squared_error(self, x, y):
         return np.mean((x - y) **2) 

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
        self.output = None
        if self.twolayers:
            self.layer_1 = self.function.relu(np.dot(x, self.weight_1) + self.bias_1) #hidden layer 1 process
            self.layer_2 = self.function.relu(np.dot(self.layer_1, self.weight_2) + self.bias_2) #hidden layer 2 process
        else:
            self.layer_2 = self.function.relu(np.dot(x, self.weight_2) + self.bias_2) #hidden layer 2 process
        self.output = self.function.sigmoid(np.dot(self.layer_2, self.weight_3 + self.bias_3)) #output layer process
    
    def backpropagation(self, x, y): #changing weight and bias value base on error rate 
        x = x.reshape(-1, len(x))
        error_3 = y - self.output #find error rate of output layer by doing subtraction between actual output and predicted output
        gradient_decent_3 = error_3 * self.function.sigmoid_derivative(self.output) #finding gradient decent(mx) of output layer
        error_2 = gradient_decent_3.dot(self.weight_3.T) #find error rate of hidden layer 2
        gradient_decent_2 = error_2 * self.function.relu_derivative(self.layer_2) #find gradient decent of hidden layer 2
        if self.twolayers:
            error_1 = gradient_decent_2.dot(self.weight_2.T) #find error rate of layer 1
            gradient_decent_1 = error_1 * self.function.relu_derivative(self.layer_1) #find gradient decent of layer 1
            self.weight_2 += self.layer_1.T.dot(gradient_decent_2) * self.learning_rate #update weight value of output layer
            self.weight_1 += x.T.dot(gradient_decent_1) * self.learning_rate #update weight value of layer 1
            self.bias_1 += np.sum(gradient_decent_1, axis=0) * self.learning_rate #update bias value of layer 1  
        else:
            self.weight_2 += x.T.dot(gradient_decent_2) * self.learning_rate #update weight value of output layer    
        self.weight_3 += self.layer_2.T.dot(gradient_decent_3) * self.learning_rate #update weight value of output layer
        self.bias_3 += np.sum(gradient_decent_3, axis=0) * self.learning_rate #update bias value of output layer 
        self.bias_2 += np.sum(gradient_decent_2, axis=0) * self.learning_rate #update bias value of hidden layer 2
        

    def train(self, input_neuron, hidden_neuron, output_neuron, learning_rate, epochs): 
        print('Training data...')
        if self.train_X.all() == None:
            print("ERROR: Dataset not found")
            return 0
        self.input = input_neuron #define input neuron 
        self.hidden = list(hidden_neuron) #define neuron in hidden layer
        self.output = output_neuron #define output neuron(1-10)
        self.learning_rate = learning_rate 
        if len(self.hidden) == 2:
            self.weight_1 = 0.002 * np.random.random((self.input, self.hidden[0])) - 0.001 #define weight for input layer
            self.bias_1 = np.zeros((1, self.hidden[0])) #define bias for input layer  
            self.weight_2 = 0.02 * np.random.random((self.hidden[0], self.hidden[1])) - 0.01 #define weight for hidden layer 1
            self.weight_3 = 0.2 * np.random.random((self.hidden[1], self.output)) - 0.1 #defile weight for hidden layer 2
            self.bias_2 = np.zeros((1, self.hidden[1])) #define bias for hidden layer 1  
            self.bias_3 = np.zeros((1, self.output)) #define bias for hidden layer 2
            self.twolayers  =True
        elif len(self.hidden) > 2:
            print('ERROR: Only 2 hidden layers can be used')
            return 0      
        else:
            self.weight_2 = 0.02 * np.random.random((self.input, self.hidden[0])) - 0.01 #define weight for hidden layer 1
            self.weight_3 = 0.2 * np.random.random((self.hidden[0], self.output)) - 0.1 #defile weight for hidden layer 2
            self.bias_2 = np.zeros((1, self.hidden[0])) #define bias for hidden layer 1  
            self.bias_3 = np.zeros((1, self.output)) #define bias for hidden layer 2
        y = self.transformLabels(self.train_y) #transform all train label into one-hot value
        for i in range(epochs): #epochs represents how many times train data will be tested
            for j in range(len(self.train_X)):
                self.forward(self.train_X[j])
                self.backpropagation(self.train_X[j], y[j])
            if i % 1== 0: #find loss for each epoch
                print(f"Epoch {i+1}: Loss = {self.mean_squared_error(self.output, y)}")
                
        print(f"Training done. Total Epochs: {epochs} \n Final Loss = {self.mean_squared_error(self.output, y)}")
    
    def test_data(self): #find the accuracy
        if self.test_X.all() == None:
            print('ERROR: Dataset not found')
            return 0
        correct = 0
        X = self.test_X
        for i in range(1000): 
            self.forward(X[i])
            result = self.reform(self.output)
            if result == self.test_y[i]: #find if test result is correct or not
                correct += 1
            
            print(f"Test case #{i+1}: Predict output = {result}, Actual output = {self.test_y[i]} ")
            print(f'Accuracy: {round(((correct/( i+1)) * 100), 2)}%')

    def save_model(self, filename):
        file = str(filename) 
        if not filename: 
            print('ERROR: Please provide filename.')
        elif not file.endswith('.csv'):
            print("ERROR: File must be CSV file.")
        with open(filename, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            if self.twolayers:
                #save layer 1 model
                writer.writerow(['Layer 1 Weight'])
                writer.writerows(self.weight_1)
                writer.writerow(['Layer 1 Bias'])
                writer.writerows(self.bias_1)
                #save layer 2 model
                writer.writerow(['Layer 2 Weight'])
                writer.writerows(self.weight_2)
                writer.writerow(['Layer 2 Bias'])
                writer.writerows(self.bias_2)
                #save layer 3 model
                writer.writerow(['Layer 3 Weight'])
                writer.writerows(self.weight_3)
                writer.writerow(['Layer 3 Bias'])
                writer.writerows(self.bias_3)
            else:
                #save layer 1 model
                writer.writerow(['Layer 1 Weight'])
                writer.writerows(self.weight_2)
                writer.writerow(['Layer 1 Bias'])
                writer.writerows(self.bias_2)
                #save layer 2 model
                writer.writerow(['Layer 2 Weight'])
                writer.writerows(self.weight_3)
                writer.writerow(['Layer 2 Bias'])
                writer.writerows(self.bias_3)
        print('Trained model saved.')
        self.load_model(filename)

    def load_model(self, filename):
            print('Loading trained model...')
            self.weight_1 = []
            self.weight_2 = []
            self.weight_3 = []
            self.bias_1 = None
            self.bias_2 = None
            self.bias_3 = None
            self.twolayers = False
            #try:
            with open(filename, 'r') as csvf:
                reader = csv.reader(csvf)
                layer = None
                for row in reader:
                    if len(row) > 0 and row[0].startswith('Layer'):
                        layer = row[0]
                    elif layer == 'Layer 1 Weight':
                        self.weight_1.append(np.array([float(val) for val in row]))
                    elif layer == 'Layer 1 Bias':
                        self.bias_1 = np.array([float(val) for val in row])
                    elif layer == 'Layer 2 Weight':
                        self.weight_2.append(np.array([float(val) for val in row]))
                    elif layer == 'Layer 2 Bias':
                        self.bias_2 = np.array([float(val) for val in row])
                    elif layer == 'Layer 3 Weight':
                        self.weight_3.append(np.array([float(val) for val in row]))
                        self.twolayers = True
                    elif layer == 'Layer 3 Bias':
                        self.bias_3 = np.array([float(val) for val in row])
                if self.twolayers:
                    self.weight_1 = np.array(self.weight_1)
                    self.weight_2 = np.array(self.weight_2)
                    self.weight_3 = np.array(self.weight_3)
                else:
                    self.bias_3 = self.bias_2
                    self.bias_2 = self.bias_1
                    self.weight_3 = np.array(self.weight_2)
                    self.weight_2 = np.array(self.weight_1)
                    
            print("Trained model loaded.")
            #except:
             #   print('ERROR: File not found')
              #  return 0
    def showimg(self, X):  
        pyplot.imshow(X, cmap=pyplot.get_cmap('gray'))
        pyplot.show()    
       
    def predict(self, num): #find result and show image used
        self.forward(self.test_X[num])
        result = self.reform(self.output)
        return result
    
