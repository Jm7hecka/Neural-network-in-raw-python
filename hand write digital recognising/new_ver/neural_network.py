import numpy as np
import time
from math import sqrt
import numpy as np
from keras.models import save_model
import csv 
import threading
import Activation_function as af
import ProgressBar
class MLP():
    def __init__(self):
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.output = None
        self.weight = []
        self.bias = []
        self.batch_gradient_descent = None
        self.layer_output = []
        self.label_classes = []
        self.isclasses = False
        self.TrainOverview = "" 
        self.correct = 0
        self.batch_size = 0
        self.mini_batch = []
   
    def Flatten(self, image):
        array = np.array(image)
        array = array.reshape(image.shape[0], -1)/ 255
        return array
    
    
    def ImportData(self, train_data, train_label, test_data, test_label, label_class:list = None):
        self.train_X = self.Flatten(train_data) #turn all train image into 1D array
        self.imgdata = test_data
        self.test_X = self.Flatten(test_data) #turn all test image into 1D array
        self.train_y = train_label
        self.test_y = test_label
        if label_class is None:
            self.isclasses = False
        else:
            self.isclasses = True
            self.label_classes = label_class

    def mean_squared_error(self, y_pred, y_actual):
        return np.mean((y_actual - y_pred) ** 2) 
    
    def accuracy(self, x, y, total):
    # Calculate accuracy
        
        if x == y:
            self.correct +=1
        return round(((self.correct/(total+1))), 4)

    def TransformLabels(self, labels, label_classes):     #turn label into one-hot data
        zeroArray = np.zeros(len(label_classes), int).tolist() #for example 1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        transformed = []
        for i in range(len(labels)):
            label = int(labels[i])
            transformed.append(zeroArray[0:label] + [1] + zeroArray[label+1:])
        return transformed

    def Reform(self, output, label_classes: list = None): #turn one-hot label into result
        max = 0              
        num = 0
        output = np.array(output)
        array = output.ravel()

        for i in range(10): 
            if array[i] > max: 
                num = i
                max = array[i]
        if label_classes is not None:
            return label_classes[num]
        else:
            return output
    
    def Create_MiniBatch(self, inputx, inputy, batch_size):
        mini_x = []
        mini_y= []
        for i in range(0, inputx.shape[0], batch_size):
            mini_x.append(inputx[i: i+batch_size])
            mini_y.append(inputy[i: i+batch_size])
        mini_batch = []
        mini_batch.append(mini_x)
        mini_batch.append(mini_y)
        return mini_batch

    def forward(self, x): #forward propagation, find output from input value
        self.output = None
        self.layer_output = []
        self.layer_output.append(x)
        for i in range(len(self.weight)-1): 
            self.layer_output.append(af.Activation_function(np.dot(self.layer_output[-1], self.weight[i]) + self.bias[i], 'relu'))
            
        self.layer_output.append(af.Activation_function(np.dot(self.layer_output[-1], self.weight[-1]) + self.bias[-1], 'sigmoid'))

        return self.layer_output
    
    def backpropagation(self, layer_output, y_actual): #changing weight and bias value base on error rate 
        x = layer_output[0].reshape(len(layer_output[0]), -1)
        gradient_descent = None
        error = None
        gradient_w = [np.zeros_like(w) for w in self.weight]
        gradient_b = [np.zeros_like(b) for b in self.bias]
        self.batch_gradient_descent = [None, None]
        error = y_actual - layer_output[-1]
        gradient_descent = error * af.Activation_function(layer_output[-1], 'sigmoid', True)
        gradient_w[-1] = np.sum([np.matmul(self.layer_output[-2][j].reshape(1, -1).T, gradient_descent[j].reshape(1, -1)) for j in range(self.batch_size)], axis=0)
        gradient_b[-1] = np.sum((gradient_descent[j] for j in range(self.batch_size)), axis=0)
        
        for i in range(len(self.layer_output)-2, 0, -1):
            error = gradient_descent.dot(self.weight[i].T)
            gradient_descent = error * af.Activation_function(self.layer_output[i], 'relu', True)
            gradient_w[i-1] = np.sum([self.layer_output[i-1][j].reshape(1, -1).T.dot(gradient_descent[j].reshape(1, -1)) for j in range(self.batch_size)], axis=0)
            gradient_b[i-1] = np.sum((gradient_descent[j] for j in range(self.batch_size)), axis=0)
       
        self.batch_gradient_descent = [gradient_w, gradient_b]
        
        for i in range(len(self.batch_gradient_descent[0])):
            self.weight[i] += self.batch_gradient_descent[0][i] * self.learning_rate
            self.bias[i] += self.batch_gradient_descent[1][i] * self.learning_rate
        
        return self.batch_gradient_descent
    

    def LazyTrain(self,  epochs: int, layer_neuron: list, learning_rate, batch_size:int = 1): 
        print('Training data...')
        if self.train_X.all() == None:
            raise ValueError("ERROR: Dataset not found")
            return 0

        self.learning_rate = learning_rate 
        self.batch_size = batch_size
        #initialise weight and bias
        
        for i in range(len(layer_neuron)-1):
            new_weight = np.random.uniform(low=-sqrt(6 / (layer_neuron[i] + layer_neuron[i+1])), high=sqrt(6 / (layer_neuron[i] + layer_neuron[i+1])), size=(layer_neuron[i], layer_neuron[i+1]))
            self.weight.append(new_weight)
            self.bias.append(np.zeros((1, layer_neuron[i+1])))

        if self.isclasses:
           y = self.TransformLabels(self.train_y, self.label_classes) #transform all train label into one-hot value

        for i in range(epochs): #epochs represents how many times train data will be tested
            print(f'\nEpoch {i+1}/{epochs}')
            self.correct = 0
            self.mini_batch = self.Create_MiniBatch(self.train_X, y, batch_size)
            p = ProgressBar.ProgressBar(epochs, len(self.mini_batch[0]))
            s = p.run()
            for j in range(len(self.mini_batch[0])):
                self.output = self.forward(self.mini_batch[0][j])
                gd = self.backpropagation(self.output, self.mini_batch[1][j])
                accuracy = self.accuracy(self.Reform(self.output[-1][-1], self.label_classes), self.Reform(self.mini_batch[1][j][-1], self.label_classes), j)
                loss = self.mean_squared_error(self.output[-1][-1], self.mini_batch[1][j][-1])
                s(loss, accuracy)
        self.TrainOverview = f"Total Epochs: {epochs} | Time taken: {p.time} | Accuracy = {accuracy} | Final Loss = {self.mean_squared_error(self.output[-1][-1], self.mini_batch[1][j])} \n Layers: {layer_neuron}, Learning rate:{learning_rate}, Batch size: {batch_size}, Total Batch: {len(self.mini_batch[0])} "
        print(f"Training done. {self.TrainOverview}")
    
    def TestData(self): #find the accuracy
        if self.test_X.all() == None:
            raise ValueError('ERROR: Dataset not found')
        correct = 0
        X = self.test_X
        for i in range(1000): 
            output = self.forward(X[i])[-1]
            if self.isclasses:
                result = self.Reform(output, self.label_classes)
            else:
                result = output
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
            if self.twolayers:
                #save layer 3 model
                writer.writerow(['Layer 3 Weight'])
                writer.writerows(self.weight_3)
                writer.writerow(['Layer 3 Bias'])
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
                    self.weight_3 = np.array(self.weight_3)
                self.weight_1 = np.array(self.weight_1)
                self.weight_2 = np.array(self.weight_2)
            print("Trained model loaded.")
            #except:
             #   print('ERROR: File not found')
              #  return 0
       
    def predict(self, image): #find result and show image used
        self.forward()
        result = self.reform(self.output)
        return result
    
