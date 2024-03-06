from keras.datasets import mnist #include dataset 
from neural_network import MLP

ml = MLP() #type of the neural network using
(train_X, train_y),(test_X, test_y) = mnist.load_data() 
ml.import_data(train_X, train_y, test_X, test_y)
ml.train(28*28, [256, 256], 10, 0.001, 10) #[input neuron, hidden layers' neuron, output neuron, learning rate, epoches]
ml.save_model('trained_model.csv')
ml.test_data()
print("DONE")
