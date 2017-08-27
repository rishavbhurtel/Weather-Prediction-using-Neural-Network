I have selected data of 5 years to train and test the neural network.

1) Define Network

The first step is to define my neural network. Neural networks are defined in Keras as a sequence of layers. Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. TensorFlow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. It was developed with a focus on enabling fast experimentation.  The container for these layers is the Sequential class. The first step is to create an instance of the Sequential class. Then create layers and add them in the order that they should be connected.
For example, we can do this in two steps:
model = Sequential()
model.add(Dense(2))
we can also do this in one step by creating an array of layers and passing it to the constructor of the Sequential but I have used the first step.
layers = [Dense(2)]
model = Sequential(layers)
The first layer in the network must define the number of inputs to expect. The way that this is specified can differ depending on the network type, but in my case for a Multilayer Perceptron model this is specified by the input_dim attribute. For example, a small Multilayer Perceptron model with 2 inputs in the visible layer, 5 neurons in the hidden layer and one neuron in the output layer can be defined as:
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(1))
Think of a Sequential model as a pipeline with your raw data fed in at the bottom and predictions that come out at the top.
This is a helpful conception in Keras as concerns that were traditionally associated with a layer can also be split out and added as separate layers, clearly showing their role in the transform of data from input to prediction. For example, activation functions that transform a summed signal from each neuron in a layer can be extracted and added to the Sequential as a layer-like object called Activation.
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))
The choice of activation function is most important for the output layer as it will define the format that predictions will take. For example, below are some common predictive modeling problem types and the structure and standard activation function that you can use in the output layer:
Regression: Linear activation function or ‘linear’ and the number of neurons matching the number of outputs.
Binary Classification (2 class): Logistic activation function or ‘sigmoid’ and one neuron the output layer.
Multiclass Classification (>2 class): Softmax activation function or ‘softmax’ and one output neuron per class value, assuming a one-hot encoded output pattern.

2)Compile Network

Once i have defined our network, i must compile it. Compilation is an efficiency step. It transforms the simple sequence of layers that we defined into a highly efficient series of matrix transforms in a format intended to be executed on your GPU or CPU, depending on how Keras is configured. Think of compilation as a precompute step for the network.
Compilation is always required after defining a model. This includes both before training it using an optimization scheme as well as loading a set of pre-trained weights from a save file. The reason is that the compilation step prepares an efficient representation of the network that is also required to make predictions on your hardware. Compilation requires a number of parameters to be specified, specifically tailored to training your network. Specifically the optimization algorithm to use to train the network and the loss function used to evaluate the network that is minimized by the optimization algorithm. For example, below is a case of compiling a defined model and specifying the stochastic gradient descent (sgd) optimization algorithm and the mean squared error (mse) loss function, intended for a regression type problem.
model.compile(optimizer='sgd', loss='mse')
But in my case I used binary_crossentropy for loss and adam for optimizer
model.compile(loss='binary_crossentropy', optimizer='adam')
The type of predictive modeling problem imposes constraints on the type of loss function that can be used. For example, below are some standard loss functions for different predictive model types:
Regression: Mean Squared Error or ‘mse‘.
Binary Classification (2 class): Logarithmic Loss, also called cross entropy or ‘binary_crossentropy‘.
Multiclass Classification (>2 class): Multiclass Logarithmic Loss or ‘categorical_crossentropy‘.
The most common optimization algorithm is stochastic gradient descent, but Keras also supports a suite of other state of the art optimization algorithms. Perhaps the most commonly used optimization algorithms because of their generally better performance are:
Stochastic Gradient Descent or ‘sgd‘ that requires the tuning of a learning rate and momentum.
ADAM or ‘adam‘ that requires the tuning of learning rate.
RMSprop or ‘rmsprop‘ that requires the tuning of learning rate.
Finally, i also specified metrics to collect while fitting the model in addition to the loss function. Generally, the most useful additional metric to collect is accuracy for classification problems. The metrics to collect are specified by name in an array.For example:
model.compile(optimizer='adam', loss='binary_crossentrophy', metrics=['accuracy'])

3)Fit Network

Once the network is compiled, it can be fit, which means adapt the weights on a training dataset. Fitting the network requires the training data to be specified, both a matrix of input patterns X and an array of matching output patterns y. The network is trained using the backpropagation algorithm and optimized according to the optimization algorithm and loss function specified when compiling the model. The backpropagation algorithm requires that the network be trained for a specified number of epochs or exposures to the training dataset. Each epoch can be partitioned into groups of input-output pattern pairs called batches. This define the number of patterns that the network is exposed to before the weights are updated within an epoch. It is also an efficiency optimization, ensuring that not too many input patterns are loaded into memory at a time. The way I fitted my network is as follows:
model.fit(X_train, Y_train, nb_epoch=150, batch_size=10)
Once fit, a object is returned that provides a summary of the performance of the model during training. This includes both the loss and any additional metrics specified when compiling the model, recorded each epoch.

4) Evaluate Network

Once the network is trained, it can be evaluated. The network can be evaluated on the training data, but this will not provide a useful indication of the performance of the network as a predictive model, as it has seen all of this data before.
I evaluate the performance of the network on a separate dataset, unseen during testing. This will provide an estimate of the performance of the network at making predictions for unseen data in the future. The model evaluates the loss across all of the test patterns, as well as any other metrics specified when the model was compiled, like classification accuracy. A list of evaluation metrics is returned. For example, for a model compiled, i could evaluate it on a new dataset as follows:
scores = model.evaluate(X_test, Y_test)

5) Make Predictions

Finally, once we are satisfied with the performance of our fit model, we can use it to make predictions on new data. This is as easy as calling the predict() function on the model with an array of new input patterns.
model.predict(x=pre_data)
where pre_data is the data given witten in txt file which will be read as
pre_data = np.loadtxt('predict.txt')
In predict.txt file the values will be in the order which we had given the network before i.e.
1 19 1 4 -1 95 4 1024 1019 7 1 3
The predictions will be returned in the format provided by the output layer of the network. For a binary classification problem, the predictions may be an array of probabilities for the first class that can be converted to a 1 or 0 by rounding. For a multiclass classification problem i.e in my case the results will be in the form of an array of probabilities. From that i choose the highest value as 1 and other as 0 giving me the possible outcome as 0001 or 0010 or 0100 or 1000 which are Thunderstorm, Rain, Fog and sunny respectively.
