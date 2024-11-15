#include "neural_network.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Helper function to initialize weights with small random values
float random_weight() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(-0.1, 0.1);
    return dis(gen);
}

// Constructor: initializes the weights
NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size) {
    hidden_weights.resize(input_size * hidden_size);
    output_weights.resize(hidden_size * output_size);
    
    // Initialize weights for hidden and output layers with small random values
    std::generate(hidden_weights.begin(), hidden_weights.end(), random_weight);
    std::generate(output_weights.begin(), output_weights.end(), random_weight);
}

// Forward pass: calculates the output of the network for a given input
std::vector<float> NeuralNetwork::forward(const std::vector<float> &inputs) {
    // Hidden layer
    std::vector<float> hidden_layer(hidden_weights.size() / inputs.size());
    for (int i = 0; i < hidden_layer.size(); ++i) {
        hidden_layer[i] = 0.0f;
        for (int j = 0; j < inputs.size(); ++j) {
            hidden_layer[i] += inputs[j] * hidden_weights[j * hidden_layer.size() + i];
        }
        // Apply ReLU activation
        hidden_layer[i] = std::max(0.0f, hidden_layer[i]);
    }

    // Output layer
    std::vector<float> output(output_weights.size() / hidden_layer.size());
    for (int i = 0; i < output.size(); ++i) {
        output[i] = 0.0f;
        for (int j = 0; j < hidden_layer.size(); ++j) {
            output[i] += hidden_layer[j] * output_weights[j * output.size() + i];
        }
    }
    
    // Apply softmax to the output layer
    float sum_exp = 0.0f;
    for (float &val : output) {
        val = std::exp(val);
        sum_exp += val;
    }
    for (float &val : output) {
        val /= sum_exp;
    }

    return output;
}

// Backward pass: updates the weights based on error
void NeuralNetwork::backward(const std::vector<float> &inputs, const std::vector<float> &outputs, int label) {
    // One-hot encode the label
    std::vector<float> target(outputs.size(), 0.0f);
    target[label] = 1.0f;

    // Calculate output layer errors (cross-entropy derivative)
    std::vector<float> output_errors(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
        output_errors[i] = outputs[i] - target[i];
    }

    // Hidden layer errors
    std::vector<float> hidden_errors(hidden_weights.size() / inputs.size());
    for (int i = 0; i < hidden_errors.size(); ++i) {
        hidden_errors[i] = 0.0f;
        for (int j = 0; j < output_errors.size(); ++j) {
            hidden_errors[i] += output_errors[j] * output_weights[i * output_errors.size() + j];
        }
        // Derivative of ReLU (only propagate error if hidden neuron was active)
        hidden_errors[i] = hidden_errors[i] > 0 ? hidden_errors[i] : 0;
    }

    // Update output weights
    for (int i = 0; i < hidden_errors.size(); ++i) {
        for (int j = 0; j < output_errors.size(); ++j) {
            output_weights[i * output_errors.size() + j] -= 0.01f * hidden_errors[i] * output_errors[j];
        }
    }

    // Update hidden weights
    for (int i = 0; i < inputs.size(); ++i) {
        for (int j = 0; j < hidden_errors.size(); ++j) {
            hidden_weights[i * hidden_errors.size() + j] -= 0.01f * inputs[i] * hidden_errors[j];
        }
    }
}

// Train the neural network
void NeuralNetwork::train(const std::vector<std::vector<float> > &images, const std::vector<int> &labels, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int correct_predictions = 0;
        
        for (int i = 0; i < images.size(); ++i) {
            // Forward pass
            std::vector<float> outputs = forward(images[i]);

            // Calculate prediction accuracy
            if (predict(images[i]) == labels[i]) {
                ++correct_predictions;
            }

            // Backward pass to update weights
            backward(images[i], outputs, labels[i]);
        }
        
        // Print training progress for each epoch
        float accuracy = static_cast<float>(correct_predictions) / images.size();
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << ", Accuracy: " << accuracy * 100 << "%\n";
    }
}

// Predict the label for a single image
int NeuralNetwork::predict(const std::vector<float> &image) {
    std::vector<float> outputs = forward(image);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

// Calculate accuracy over the dataset
float NeuralNetwork::accuracy(const std::vector<std::vector<float> > &images, const std::vector<int> &labels) {
    int correct = 0;
    for (int i = 0; i < images.size(); ++i) {
        if (predict(images[i]) == labels[i]) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / images.size();
}

void NeuralNetwork::printArchitecture() {
    std::cout << "Model: Multilayer Perceptron\n";
    std::cout << "Layer (type)              Shape\n";
    std::cout << "=================================\n";
    std::cout << "Input Layer                (" << 784 << ")\n";
    std::cout << "Hidden Layer (ReLU)        (" << 128 << ")\n";
    std::cout << "Output Layer (Softmax)     (" << 10 << ")\n";
}
