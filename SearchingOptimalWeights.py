import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
import time


# Function to model a neuron with step and sigmoid activation functions
def neuron_model(inputs, weights, activation='step', threshold=0.001):
    # Add bias term (-1) to input matrix
    bias = -1 * np.ones((inputs.shape[0], 1))
    inputs_with_bias = np.hstack((bias, inputs))

    # Calculate neuron activation (weighted sum)
    activation_values = np.dot(inputs_with_bias, weights)

    # Apply activation function
    if activation == 'step':
        outputs = np.where(activation_values >= 0, 1, 0)
    elif activation == 'sigmoid':
        outputs = 1 / (1 + np.exp(-activation_values / threshold))
    return outputs


# Function to perform one learning step in perceptron training
def update_weights(error, inputs, weights, learning_rate=0.1): #eror == y-y'
    # Add bias term (-1) to input matrix
    bias = -1 * np.ones((inputs.shape[0], 1))
    inputs_with_bias = np.hstack((bias, inputs))

    # Update weights based on error
    new_weights = weights + learning_rate * inputs_with_bias.T @ error
    return new_weights


# Initialize training and testing data
x_train = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])
y_train = np.array([[1], [0], [0], [0]])
x_test = np.random.rand(100, x_train.shape[1])
y_test = np.zeros(x_test.shape[0])

# Initial weights and learning parameters
initial_weights = np.array([[0.5], [-1], [1]])
learning_rate = 0.1
mse_threshold = 0.05
num_epochs = 50

# Plot initial decision boundary with training points
xx = np.array([-5, 5])
decision_boundary = -(initial_weights[1] / initial_weights[2]) * xx + (initial_weights[0] / initial_weights[2])
plt.plot(xx, decision_boundary, color='green', linewidth=2)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(x_train[y_train.flatten() == 1, 0], x_train[y_train.flatten() == 1, 1], marker='+', color='red', s=100,
            label="Class 1")
plt.scatter(x_train[y_train.flatten() == 0, 0], x_train[y_train.flatten() == 0, 1], marker='o', color='red', s=100,
            label="Class 0")
plt.xlabel('EEG Delta Power (x1)')
plt.ylabel('EEG Theta Power (x2)')
plt.title('Initial Decision Boundary')
plt.legend()
plt.show()

# Training the perceptron over multiple epochs
weights = initial_weights
mse_history = []

for epoch in range(num_epochs):
    # Forward pass
    predictions = neuron_model(x_train, weights)
    errors = y_train - predictions
    mse = np.mean(errors ** 2)
    mse_history.append(mse)

    # Weight update
    weights = update_weights(errors, x_train, weights, learning_rate)

    # Plot updated decision boundary with test points classified
    decision_boundary = -(weights[1] / weights[2]) * xx + (weights[0] / weights[2])
    plt.plot(xx, decision_boundary, color='green', linewidth=2, label="Decision Boundary")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.scatter(x_train[y_train.flatten() == 1, 0], x_train[y_train.flatten() == 1, 1], marker='+', color='red', s=100,
                label="Class 1")
    plt.scatter(x_train[y_train.flatten() == 0, 0], x_train[y_train.flatten() == 0, 1], marker='o', color='red', s=100,
                label="Class 0")

    # Classify test points and plot them with different markers
    test_predictions = neuron_model(x_test, weights)
    plt.scatter(x_test[test_predictions.flatten() == 1, 0], x_test[test_predictions.flatten() == 1, 1], marker='+',
                color='blue', s=50, label="Class 1 (Test)")
    plt.scatter(x_test[test_predictions.flatten() == 0, 0], x_test[test_predictions.flatten() == 0, 1], marker='o',
                color='blue', s=50, facecolors='none', label="Class 0 (Test)")

    plt.xlabel('EEG Delta Power (x1)')
    plt.ylabel('EEG Theta Power (x2)')
    plt.title(f'Decision Boundary after Epoch {epoch + 1}')
    plt.legend(loc="upper right")
    plt.show()

    time.sleep(0.5)

    # Stop if MSE is below threshold
    if mse < mse_threshold:
        break

# Plot MSE over epochs
plt.plot(mse_history, label='Mean Squared Error')
plt.axhline(y=mse_threshold, color='r', linestyle='--', label='MSE Threshold')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE over Epochs')
plt.legend()
plt.show()

# Classification using sklearn's Perceptron model
perceptron_model = Perceptron()
perceptron_model.fit(x_train, y_train.ravel())
perceptron_predictions = perceptron_model.predict(x_test)

# Plot decision boundary for Perceptron with test points
weights = perceptron_model.coef_.flatten()
intercept = perceptron_model.intercept_[0]
decision_boundary = -(weights[0] / weights[1]) * xx - (intercept / weights[1])
plt.plot(xx, decision_boundary, color='green', linewidth=2, label="Decision Boundary")
plt.scatter(x_train[y_train.flatten() == 1, 0], x_train[y_train.flatten() == 1, 1], marker='+', color='red', s=100)
plt.scatter(x_train[y_train.flatten() == 0, 0], x_train[y_train.flatten() == 0, 1], marker='o', color='red', s=100)
plt.scatter(x_test[perceptron_predictions == 1, 0], x_test[perceptron_predictions == 1, 1], marker='+', color='blue',
            s=50)
plt.scatter(x_test[perceptron_predictions == 0, 0], x_test[perceptron_predictions == 0, 1], marker='o', color='blue',
            s=50, facecolors='none')

plt.xlabel('EEG Delta Power (x1)')
plt.ylabel('EEG Theta Power (x2)')
plt.title('Decision Boundary with sklearn Perceptron')
plt.legend(loc="upper right")
plt.show()

# Classification using SVM model
svm_model = LinearSVC(class_weight='balanced')
svm_model.fit(x_train, y_train.ravel())
svm_predictions = svm_model.predict(x_test)

# Plot decision boundary for SVM with test points
weights = svm_model.coef_.flatten()
intercept = svm_model.intercept_[0]
decision_boundary = -(weights[0] / weights[1]) * xx - (intercept / weights[1])
plt.plot(xx, decision_boundary, color='green', linewidth=2, label="Decision Boundary")
plt.scatter(x_train[y_train.flatten() == 1, 0], x_train[y_train.flatten() == 1, 1], marker='+', color='red', s=100)
plt.scatter(x_train[y_train.flatten() == 0, 0], x_train[y_train.flatten() == 0, 1], marker='o', color='red', s=100)
plt.scatter(x_test[svm_predictions == 1, 0], x_test[svm_predictions == 1, 1], marker='+', color='blue', s=50)
plt.scatter(x_test[svm_predictions == 0, 0], x_test[svm_predictions == 0, 1], marker='o', color='blue', s=50,
            facecolors='none')

plt.xlabel('EEG Delta Power (x1)')
plt.ylabel('EEG Theta Power (x2)')
plt.title('Decision Boundary with SVM')
plt.legend(loc="upper right")
plt.show()
