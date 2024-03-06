from keras.datasets import mnist #include dataset 
from neural_network import MLP

ml = MLP()
(train_X, train_y),(test_X, test_y) = mnist.load_data()
ml.ImportData(train_X, train_y, test_X, test_y, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ml.LazyTrain(1, [28*28, 256, 10], 0.01, 32)
ml.TestData()
print(ml.TrainOverview)
print("DONE")
