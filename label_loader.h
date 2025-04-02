#include <fstream>
#include <vector>
bool load_labels(const std::string& filename, std::vector<int>& labels) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    int label;
    while (file >> label) {
        labels.push_back(label);
    }

    return true;
}
