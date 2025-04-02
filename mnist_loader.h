#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>

bool load_MNIST(const std::string& images_file, std::vector<std::vector<float>>& images);

#endif
