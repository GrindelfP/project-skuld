// main.cpp
#include <iostream>
#include <vector>
#include <fann.h>
#include "neural_network_fann.h"
#include "mnist_loader.h"

int main() {
    try {
        // Load MNIST dataset
        std::string image_file = "train-images.idx3-ubyte";
        std::string label_file = "train-labels.idx1-ubyte";
        
        std::vector<std::vector<float>> images = load_mnist_images(image_file);
        std::vector<int> labels = load_mnist_labels(label_file);

        // Create neural network with input size (28*28=784), 128 hidden units, and 10 output units
        NeuralNetwork nn(784, 128, 10);
        
        // Train the network
        nn.train(images, labels, 50, 0.01);  // 50 epochs, learning rate 0.01
        
        // Save the trained network
        nn.save("mnist_nn.net");

        // Evaluate a test image (for example, the first image from the dataset)
        std::vector<float> test_image = images[0];
        float result = nn.evaluate(test_image);
        std::cout << "Prediction: " << result << std::endl;
        
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
