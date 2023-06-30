
Hand write digital recognising machine learning
=======================================================
This neural network is really simple and common, using MNIST dataset. This neural network recognise hand write digital. 
We are using a type of neural network called 'Multilayer Perception'(MLP), with 2 hidden layers so that it can run more complicate datasets. This can be used on other dataset. 

Explain
=======================================================
- Program will first initialise data for training and testing 
- Then it will create weight and bias randomly
- In train process, it will do forward porpagation and backward porpagation( change weight and bias) 
- In forward porpagation, input value is multiplied by weights, each neuron has different weight. 2 layers will be done in this neural network
- Sigmoid function is used as activation function to introduce non-linearality. The sigmoid function is defined as 1/1+e^-x.
- It will return result looks like this [[0.49701919 0.49459193 0.4982384  0.49093732 0.48924195 0.49455886 0.48850142 0.50313756 0.49729235 0.50388361]]
- In backward porpagation, predicted output will be subtracted by actual output to get the error rate (loss).  
- Then error rate will be multiplied by rate of change of formula (derivation of function) to find gradient decent. The formula of derivative is limΔx→0f(x+Δx)−f(x) / Δx. This is a limit of difference quotient, in other words it is just dy/dx
- The error rate of layer 1 is also needed by dotting gradient decent and layer 1 output. Same way to find gradient decent
- The update of weight is calculated by dotting layer of outputs and gradient decent times learning rate( dont ask me why, still figuring out :)
- The update of bias is calculated by summing gradient decent and multiply it by learning rate
- Once it is done, save the model( you dont want to train data everytime you use it right?) and test it with the test case to find accuracy
- Then it is done!

### The theory of neural network is still confusing me, contact me if i understand it incorrectly.
