#include <fstream>
#include <sstream>
#include <vector>
#include <string>

bool load_MNIST(const std::string& images_file, std::vector<std::vector<float>>& images) {
    std::ifstream file(images_file);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<float> image;
        float pixel;
        while (ss >> pixel) {
            image.push_back(pixel);
        }
        images.push_back(image);
    }

    return true; 
}
