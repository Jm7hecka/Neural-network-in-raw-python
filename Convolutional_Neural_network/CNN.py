import numpy as np
from keras.models import save_model
import csv 
from matplotlib import pyplot 
import Activation_function as AF

class CNN():
    def __init__(self):
        self.weight = []
        self.bias = []
        self.kernal = []
        self.kernal_bias = []
        self.feature_map = []
        self.pooled_output = []
        self.weight_gradient = None
        self.kernal_graident = None
        self.kernal_layer = 0

    def Use_activation(self, input,  activation_func, derivative:bool = False):
        if derivative:
            if activation_func.lower() == "sigmoid":
                return AF.sigmoid_derivative(input)
            elif activation_func.lower() == "relu":
                return AF.relu_derivative(input)
        else:
            if activation_func.lower() == "sigmoid":
                return AF.sigmoid(input)
            elif activation_func.lower() == "relu":
                return AF.relu(input)

    def Conv(self, input_channel:int, output_channel: int, kernal_size: int, activation_func):
        seq = self.kernal_layer
        self.kernal.append(np.random.random((output_channel, input_channel, kernal_size, kernal_size)))
        self.kernal_bias.append(np.random.random(output_channel))
        def calculation(input: list):
            input = np.array(input)
            if input.ndim == 2:
                input = input[np.newaxis, :, :]

            output = np.zeros((output_channel, (input.shape[1] - kernal_size + 1), (input.shape[2] - kernal_size + 1)))
            for h in range(output_channel):
                for i in range(output.shape[1]):
                    for j in range(output.shape[2]):
                        output[h, i, j] = np.sum(input[0:input_channel, i:(i + kernal_size), j:(j + kernal_size)] * self.kernal[seq][h]) + self.kernal_bias[seq][h]
            output = self.Use_activation(output, activation_func)
            self.feature_map.append(output)
            return output

        self.kernal_layer += 1
        return calculation
    
    def MaxPool(self, kernal_size: int):
        def calculation(input: list):
            input = np.array(input)
            if input.ndim == 2:
                input = input[np.newaxis, :, :]

            output = np.zeros((input.shape[0], (input.shape[1]//kernal_size), (input.shape[2]//kernal_size)))

            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    for k in range(output.shape[2]):
                        output[i, j, k] = np.amax(input[i, j * kernal_size: j * kernal_size + 2, k * kernal_size: k * kernal_size + 2])
            
            self.pooled_output = output
            return output
        return calculation

cnn = CNN()
