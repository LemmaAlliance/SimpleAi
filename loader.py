import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load the saved weights
weights_input_hidden = np.load('weights_input_hidden.npy')
weights_hidden_output = np.load('weights_hidden_output.npy')

# Prediction function
def predict(inputs):
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    return predicted_output

# Get user input
input1 = int(input("Enter the first boolean value (0 or 1): "))
input2 = int(input("Enter the second boolean value (0 or 1): "))

# Ensure the inputs are valid boolean values
if input1 not in [0, 1] or input2 not in [0, 1]:
    print("Invalid input. Please enter 0 or 1 for both inputs.")
else:
    # Prepare the input array for prediction
    user_input = np.array([[input1, input2]])

    # Predict and print the output
    predicted_output = predict(user_input)
    print(f"Predicted output for inputs ({input1}, {input2}): {predicted_output[0][0]:.4f}")
