
import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Weights and biases initialization as per report (mean 0, std 0.1)
        self.weights1 = np.random.normal(0, 0.1, (input_dim, hidden_dim))
        self.bias1 = np.random.normal(0, 0.1, (hidden_dim,))
        self.weights2 = np.random.normal(0, 0.1, (hidden_dim, output_dim))
        self.bias2 = np.random.normal(0, 0.1, (output_dim,))

    def forward(self, state_vector):
        # Ensure state_vector is a 1D array for dot product
        state_vector = np.asarray(state_vector).flatten()
        
        # Input to hidden layer
        hidden_layer_input = np.dot(state_vector, self.weights1) + self.bias1
        hidden_layer_output = np.tanh(hidden_layer_input) # tanh activation

        # Hidden to output layer
        output = np.dot(hidden_layer_output, self.weights2) + self.bias2
        return output[0] # Return scalar value as state value estimation

    def train(self, state_vector, target_value, learning_rate=0.01):
        # Forward pass to get current prediction
        state_vector = np.asarray(state_vector).flatten()
        hidden_layer_input = np.dot(state_vector, self.weights1) + self.bias1
        hidden_layer_output = np.tanh(hidden_layer_input)
        predicted_value = np.dot(hidden_layer_output, self.weights2) + self.bias2

        # Calculate error
        error = target_value - predicted_value

        # Backpropagation (simplified gradient descent as per report)
        # Update output layer weights and biases
        delta_output = error
        self.weights2 += learning_rate * np.outer(hidden_layer_output, delta_output)
        self.bias2 += learning_rate * delta_output

        # Update hidden layer weights and biases
        # Simplified error propagation for hidden layer as per report (fixed coefficient 0.1)
        # This is not a standard backprop, but follows the report's simplified description.
        delta_hidden = np.dot(delta_output, self.weights2.T) * (1 - hidden_layer_output**2) # Derivative of tanh
        self.weights1 += learning_rate * np.outer(state_vector, delta_hidden)
        self.bias1 += learning_rate * delta_hidden

    def save_model(self, path):
        np.savez(path, weights1=self.weights1, bias1=self.bias1, weights2=self.weights2, bias2=self.bias2)
        print(f"Model saved to {path}")

    def load_model(self, path):
        data = np.load(path)
        self.weights1 = data["weights1"]
        self.bias1 = data["bias1"]
        self.weights2 = data["weights2"]
        self.bias2 = data["bias2"]
        print(f"Model loaded from {path}")


