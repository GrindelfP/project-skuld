#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>

std::vector<std::vector<float> > load_mnist_images(const std::string &file_path);
std::vector<int> load_mnist_labels(const std::string &file_path);

#endif
