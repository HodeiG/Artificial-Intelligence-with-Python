import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Load input data
"""
The data contains 2 columns for the output to represent the valid 4 labels
(2 bits information)
"""
text = np.loadtxt('data_simple_nn.txt')

# Separate it into datapoints and labels
data = text[:, 0:2]
labels = text[:, 2:]


# Minimum and maximum values for each dimension
dim1_min, dim1_max = data[:, 0].min(), data[:, 0].max()
dim2_min, dim2_max = data[:, 1].min(), data[:, 1].max()

# Define the number of neurons in the output layer
num_output = labels.shape[1]

# Define a single-layer neural network
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
nn = nl.net.newp([dim1, dim2], num_output)

# Train the neural network
error_progress = nn.train(data, labels, epochs=100, show=20, lr=0.03)

# Plot input data
plt.figure()
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')
weights = np.append(nn.layers[0].np['b'], nn.layers[0].np['w'])
for inputs, target in zip(data, labels):
    colour = None
    if target[0] == 0 and target[1] == 0:
        colour = 'bo'
    elif target[0] == 0 and target[1] == 1:
        colour = 'go'
    elif target[0] == 1 and target[1] == 0:
        colour = 'ro'
    elif target[0] == 1 and target[1] == 1:
        colour = 'co'

    plt.plot(inputs[0], inputs[1], colour)
for i in np.linspace(np.amin(data), np.amax(data)):
    # Plot input data
    slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
    intercept = -weights[0]/weights[2]
    # y = mx+c, m is slope and c is intercept
    y = (slope*i) + intercept
    plt.plot(i, y, 'ko')


# Plot the training progress
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()

plt.show()

# Run the classifier on test datapoints
print('\nTest results:')
data_test = [[0.4, 4.3], [4.4, 0.6], [4.7, 8.1]]
for item in data_test:
    print(item, '-->', nn.sim([item])[0])
