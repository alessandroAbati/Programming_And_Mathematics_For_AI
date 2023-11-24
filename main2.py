import numpy as np
import matplotlib.pyplot as plt



# ReLU layer class
class ReLU:
    '''
    A class representing the Rectified Linear Unit (reLu) activation function.
    '''
    def __init__(self):
        self.input = None # placeholder for storing the input to the layer

    def forward_pass(self, input_data):
        self.input = input_data # store the input to use it in the backward pass
        return np.maximum(0, input_data) # apply the relu function: if x is negative, max(0, x) will be 0; otherwise, will be x

    def backward_pass(self, output_gradient):
        '''
        Compute the backward pass through the reLu activation function.

        The method calculates the gradient of the reLu function with respect 
        to its input 'x', given the gradient of the loss function with respect 
        to the output of the relu layer ('gradient_values').

        Parameters:
        - gradient_values (numpy.ndarray): The gradient of the loss function with respect 
                                           to the output of the relu layer.

        Returns:
        - numpy.ndarray: The gradient of the loss function with respect to the 
                         input of the relu layer.
        '''
        # apply the derivative of the relu function: if the input is negative, the derivative is 0; otherwise, the derivative is 1
        return output_gradient * (self.input > 0)
        #return output_gradient * np.where(self.input > 0, 1.0, 0.0) 
    
# Sigmoid layer class
class Sigmoid:
    '''
    A class representing the Sigmoid activation function.
    '''
    def __init__(self):
        self.output = None # placeholder for storing the output of the forward pass

    def forward_pass(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data)) # apply the sigmoid function: f(x) = 1 / (1 + exp(-x))
        return self.output

    def backward_pass(self, output_gradient):
        '''
        Computes the backward pass of the Sigmoid activation function.

        Given the gradient of the loss function with respect to the output of the
        Sigmoid layer ('output_gradient'), this method calculates the gradient with respect
        to the Sigmoid input.

        Parameters:
        - output_gradient (numpy.ndarray): The gradient of the loss function with respect
                                           to the output of the Sigmoid layer.

        Returns:
        - numpy.ndarray: The gradient of the loss function with respect to the
                         input of the Sigmoid layer.
        '''
        return output_gradient * (self.output * (1 - self.output))
    
# Softmax layer class
class Softmax:
    def __init__(self):
        self.dinputs = None

    def forward_pass(self, input_data):
        # Shift the input data to avoid numerical instability in exponential calculations
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return output

    def backward_pass(self, dvalues):
        # The gradient of loss with respect to the input logits 
        # directly passed through in case of softmax + categorical cross-entropy
        self.dinputs = dvalues
        return self.dinputs

# Dense layer class
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = 0.01 * np.random.normal(0, 1/np.sqrt(input_size), (input_size, output_size)) # Normal distribution initialisation
        self.biases = np.full((1, output_size), 0.001) # Initialise biases with a small positive value
        self.input = None

    def forward_pass(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward_pass(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Update weights and biases
        self.weights += learning_rate * weights_gradient
        self.biases += learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient

# Neural Network wrapper class
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_history = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_pass(self, input_data):
        for layer in self.layers:
            input_data = layer.forward_pass(input_data)
        return input_data

    def backward_pass(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            if isinstance(layer, Layer):
                output_gradient = layer.backward_pass(output_gradient, learning_rate)
            else:
                output_gradient = layer.backward_pass(output_gradient)
    
    def compute_categorical_cross_entropy_loss(self, y_pred, y_true):
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate the negative log of the probabilities of the correct class
        # Multiply with the one-hot encoded true labels and sum across classes
        loss = np.sum(y_true * -np.log(y_pred_clipped), axis=1)

        # Average loss over all samples
        return np.mean(loss)

    def compute_categorical_cross_entropy_gradient(self, y_pred, y_true):
        # Assuming y_true is one-hot encoded and y_pred is the output of softmax
        y_pred_gradient = (y_pred - y_true) / len(y_pred)
        return y_pred_gradient

    def train(self, X_train, y_train, epochs=100, learning_rate=0.001, batch_size=32, verbose = 1):
        n_samples = len(X_train)
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_x = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]

                output = self.forward_pass(batch_x)
                loss_gradient = self.compute_categorical_cross_entropy_gradient(batch_y, output)
                self.backward_pass(loss_gradient, learning_rate)

            output = self.forward_pass(X_train)
            loss = self.compute_categorical_cross_entropy_loss(y_train, output)
            self.loss_history.append(loss)

            if verbose == 1:
                print(f"Epoch {epoch+1}/{epochs} --- Loss: {loss}")

    def predict(self, X_test):
        output = self.forward_pass(X_test)

        # Convert probabilities to class predictions
        predictions = np.argmax(output, axis=1)
        return predictions

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


def standardize_data(X):
    # Calculate the mean and standard deviation for each feature
    means = X.mean(axis=0)
    stds = X.std(axis=0)

    # Avoid division by zero in case of a constant feature
    stds[stds == 0] = 1

    # Standardize each feature
    X_standardized = (X - means) / stds
    return X_standardized

#################################### TESTING WITH 2D DATA POINTS #####################################
"""
def generate_data_multiclass(num_samples=1000):
    # Generate points for three classes
    class_0 = np.random.randn(num_samples, 2) + np.array([-5, -5])
    class_1 = np.random.randn(num_samples, 2) + np.array([5, 5])
    class_2 = np.random.randn(num_samples, 2) + np.array([0, 10])

    # Labels (One-Hot Encoded)
    labels_0 = np.array([1, 0, 0] * num_samples).reshape(-1, 3)
    labels_1 = np.array([0, 1, 0] * num_samples).reshape(-1, 3)
    labels_2 = np.array([0, 0, 1] * num_samples).reshape(-1, 3)

    # Combine data
    X = np.vstack((class_0, class_1, class_2))
    y = np.vstack((labels_0, labels_1, labels_2))

    return X, y

X, y = generate_data_multiclass()

X = standardize_data(X)

# Create the model
network = NeuralNetwork()
network.add_layer(Layer(2, 64))
network.add_layer(ReLU())
network.add_layer(Layer(64, 3))
network.add_layer(Softmax())

# Training
network.train(X, y, learning_rate=0.01, batch_size=32)

# Plot loss function
network.plot_loss()

# Visualisation of the results
def create_grid(X, steps=100, buffer=1):
    x_min, x_max = X[:, 0].min() - buffer, X[:, 0].max() + buffer
    y_min, y_max = X[:, 1].min() - buffer, X[:, 1].max() + buffer
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    return xx, yy

def plot_decision_boundary_multiclass(network, X, y):
    xx, yy = create_grid(X, steps=100, buffer=1)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred_func = np.vectorize(lambda x, y: np.argmax(network.forward_pass(np.array([[x, y]])), axis=1))
    Z = pred_func(xx, yy)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), edgecolors='b')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plotting
plot_decision_boundary_multiclass(network, X, y)
"""

################################## TESTING WITH MNIST DATASET ##################################
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Standardize the features
X = standardize_data(X)

# One-hot encode the labels
one_hot_encoder = OneHotEncoder(sparse=False)
y = one_hot_encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the neural network model
network = NeuralNetwork()
network.add_layer(Layer(64, 128))  # 64 inputs (8x8 images)
network.add_layer(ReLU())
network.add_layer(Layer(128, 10))  # 10 classes
network.add_layer(Softmax())

# Train the network
network.train(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=8)

network.plot_loss()

# Evaluate the performance of the model
y_pred = network.predict(X_test)
y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_test)
print(f"\nAccuracy: {accuracy}")