import numpy as np
import matplotlib.pyplot as plt
# https://pythonhosted.org/neurolab/intor.html
# https://pythonhosted.org/neurolab/lib.html
# See https://www.youtube.com/watch?v=V7s0VfAahJs
import neurolab as nl

# Load input data
text = np.loadtxt('data_perceptron.txt')

# Separate datapoints and labels
data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

# Define minimum and maximum values for each dimension
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

# Number of neurons in the output layer
num_output = labels.shape[1]

# Define a perceptron with 2 input neurons (because we
# have 2 dimensions in the input data)
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
"""
neurolab.net.newp(minmax, cn, transf)
    Create one layer perceptron

    Parameters:
        minmax (range of input value): list of list, the outer list is the
        number of input neurons, inner lists must contain 2 elements:
        min and max

        cn (number of neurons): int, number of output neurons

        transf (activation function): func (default HardLim)
        >>> # create network with 2 inputs and 10 neurons

        >>> net = newp([[-1, 1], [-1, 1]], 10)
"""
perceptron = nl.net.newp([dim1, dim2], num_output)

# Train the perceptron using the data
# Get error_progress every 20 epochs
error_progress = perceptron.train(data, labels, epochs=100, show=20, lr=0.03)

# Plot input data
plt.figure()
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')
weights = np.append(perceptron.layers[0].np['b'], perceptron.layers[0].np['w'])
for inputs, target in zip(data, labels):
    plt.plot(inputs[0], inputs[1], 'ro' if (target[0] == 1) else 'bo')
#  plt.scatter(data[:, 0], data[:, 1])
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
