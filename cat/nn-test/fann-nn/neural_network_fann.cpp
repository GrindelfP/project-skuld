// neural_network_fann.cpp
#include "neural_network_fann.h"
#include <iostream>
#include <fann.h>

NeuralNetwork::NeuralNetwork(int num_input, int num_hidden, int num_output) {
    // Create a neural network with FANN: num_input, num_hidden, num_output
    ann = fann_create_standard(3, num_input, num_hidden, num_output);
    if (ann == nullptr) {
        throw std::runtime_error("Error creating neural network.");
    }
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
}

NeuralNetwork::~NeuralNetwork() {
    fann_destroy(ann);  // Free up memory when done
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, int epochs, float learning_rate) {
    unsigned int num_samples = images.size();
    struct fann_train_data* train_data = fann_create_train(num_samples, images[0].size(), 1);
    
    // Prepare training data
    for (unsigned int i = 0; i < num_samples; ++i) {
        for (unsigned int j = 0; j < images[i].size(); ++j) {
            train_data->input[i][j] = images[i][j];
        }
        train_data->output[i][0] = (labels[i] == 1 ? 1.0f : -1.0f);  // Simplified for binary classification (e.g., digit '1' vs others)
    }

    // Train the neural network
    fann_train_on_data(ann, train_data, epochs, 1, learning_rate);

    fann_destroy_train(train_data);
}

float NeuralNetwork::evaluate(const std::vector<float>& image) {
    float* output = fann_run(ann, image.data());
    return output[0];  // In this case, we assume binary classification for simplicity
}

void NeuralNetwork::save(const std::string& filename) {
    fann_save(ann, filename.c_str());
}

void NeuralNetwork::load(const std::string& filename) {
    ann = fann_create_from_file(filename.c_str());
}
