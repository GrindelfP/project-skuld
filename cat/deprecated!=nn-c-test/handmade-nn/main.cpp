#include "neural_network.h"
#include "mnist_loader.h"
#include <iostream>

int main() {
    const int input_size = 784;  // 28x28 images
    const int hidden_size = 128; // Size of the hidden layer
    const int output_size = 10;  // 10 classes for 0-9 digits
    const int epochs = 10;
    const float learning_rate = 0.01;

    // Load MNIST data
    auto train_images = load_mnist_images("train-images.idx3-ubyte");
    auto train_labels = load_mnist_labels("train-labels.idx1-ubyte");

    // Initialize the neural network
    NeuralNetwork net(input_size, hidden_size, output_size);
    net.printArchitecture();

    // Train the model
    net.train(train_images, train_labels, epochs, learning_rate);

    // Evaluate the model
    float accuracy = net.accuracy(train_images, train_labels);
    std::cout << "Training accuracy: " << accuracy * 100 << "%\n";

    return 0;
}
