import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def is_fraction_of_y(x, y):
    # Ensure y is divisible by 10
    if y % 10 != 0:
        return False
    
    # Calculate n
    n = (10 * x) / y
    
    # Check if n is an integer and within the range 1 to 9
    return n.is_integer() and 1 <= n <= 9

# XOR dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Initialize weights
np.random.seed(1)
input_layer_neurons = 2
hidden_layer_neurons = 4
output_neurons = 1

weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

# Training parameters
learning_rate = 0.6
epochs = int(input("How many epochs? "))

# Lists to store error values for plotting
error_list = []
epoch_list = []

# Training the neural network
for epoch in range(epochs):
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    error = outputs - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate

    # Store the error every 500 epochs
    if epoch % 500 == 0:
        mean_error = np.mean(np.abs(error))
        error_list.append(mean_error)
        epoch_list.append(epoch)
        print(f"Epoch {epoch}: Mean Error = {mean_error}")

# Save the weights after training
np.save('xor_weights_input_hidden.npy', weights_input_hidden)
np.save('xor_weights_hidden_output.npy', weights_hidden_output)

print("Training complete. Weights saved.")

# Plotting the error vs. epochs
plt.figure(figsize=(10, 6))
plt.plot(epoch_list, error_list, marker='o', linestyle='-', color='b')
plt.title('Error vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.grid(True)
plt.savefig('xor_error_vs_epochs.png')
plt.show()
