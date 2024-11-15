// neural_network_fann.h
#ifndef NEURAL_NETWORK_FANN_H
#define NEURAL_NETWORK_FANN_H

#include <fann.h>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int num_input, int num_hidden, int num_output);
    ~NeuralNetwork();
    
    void train(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, int epochs, float learning_rate);
    float evaluate(const std::vector<float>& image);
    void save(const std::string& filename);
    void load(const std::string& filename);

private:
    struct fann* ann;  // Pointer to FANN neural network
};

#endif
