// neural_network.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    void train(const std::vector<std::vector<float> > &images, const std::vector<int> &labels, int epochs, float learning_rate);
    int predict(const std::vector<float> &image);
    float accuracy(const std::vector<std::vector<float> > &images, const std::vector<int> &labels);
    void printArchitecture();

private:
    std::vector<float> forward(const std::vector<float> &inputs);
    void backward(const std::vector<float> &inputs, const std::vector<float> &outputs, int label);
    
    std::vector<float> hidden_weights;
    std::vector<float> output_weights;
};

#endif
