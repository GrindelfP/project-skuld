// mnist_loader.cpp
#include "mnist_loader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>

// Function to load MNIST images from a binary file
std::vector<std::vector<float> > load_mnist_images(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + file_path);
    }

    // Read the header of the file (16 bytes)
    uint32_t magic_number, num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    file.read(reinterpret_cast<char*>(&num_cols), 4);

    // Convert from big-endian to little-endian if necessary
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    // Check the magic number for validity
    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST image file.");
    }

    // Read the images into a vector
    std::vector<std::vector<float> > images(num_images, std::vector<float>(num_rows * num_cols));
    for (uint32_t i = 0; i < num_images; ++i) {
        for (uint32_t j = 0; j < num_rows * num_cols; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = pixel / 255.0f;  // Normalize pixel value to [0, 1]
        }
    }

    file.close();
    return images;
}

// Function to load MNIST labels from a binary file
std::vector<int> load_mnist_labels(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + file_path);
    }

    // Read the header of the file (8 bytes)
    uint32_t magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    // Convert from big-endian to little-endian if necessary
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    // Check the magic number for validity
    if (magic_number != 2049) {
        throw std::runtime_error("Invalid MNIST label file.");
    }

    // Read the labels into a vector
    std::vector<int> labels(num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = label;
    }

    file.close();
    return labels;
}
