#include <iostream>
#include <fstream>
#include <vector>

void load_MNIST(const char* images_file, const char* labels_file,
    std::vector<std::vector<float>> &images,
    std::vector<int> &labels) {
int rows = 70000, cols = 784;

std::ifstream file(images_file);
if (!file) {
std::cerr << "Error: Could not open " << images_file << std::endl;
exit(1);
}

images.resize(rows, std::vector<float>(cols));

std::cout << "Reading first 10 images..." << std::endl;

for (int i = 0; i < rows; i++) {  // 🔥 Only read first 10 images
for (int j = 0; j < cols; j++) {
if (!(file >> images[i][j])) {
    std::cerr << "Error reading file at image " << i << ", pixel " << j << std::endl;
    exit(1);
}
}
}

std::cout << "Finished reading first 10 images." << std::endl;
file.close();

std::ifstream file2(labels_file);
if (!file2) {
std::cerr << "Error: Could not open " << labels_file << std::endl;
exit(1);
}

labels.resize(rows);

std::cout << "Reading first 10 labels..." << std::endl;

for (int i = 0; i < rows; i++) {
if (!(file2 >> labels[i])) {
std::cerr << "Error reading label " << i << std::endl;
exit(1);
}
}

std::cout << "Finished reading first 10 labels." << std::endl;
file2.close();
}

int main() {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    load_MNIST("mnist-images.txt", "mnist-labels.txt", images, labels);

    // ✅ Debugging: Check if images and labels are loaded correctly
    if (images.empty() || labels.empty()) {
        std::cerr << "Error: Images or labels failed to load!" << std::endl;
        return 1;
    }

    std::cout << "No. Images: " << images.size() << std::endl;
int displayImage = 2;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++)
            std::cout << images[displayImage][i * 28 + j] << " ";  // 🔥 Ensure images[0] is valid
        std::cout << std::endl;
    }

    std::cout << "Image is " << labels[displayImage] << std::endl;

    return 0;
}
