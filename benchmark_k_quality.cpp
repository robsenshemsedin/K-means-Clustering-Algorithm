#include "mnist_loader.h"
#include "utils.h"
#include "mini_batch_kmeans.h"
#include "parallel_kmeans.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <limits>
#include "label_loader.h"

int main() {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    // Load data
    if (!load_MNIST("mnist-images.txt", images)) {
        std::cerr << "Failed to load images." << std::endl;
        return 1;
    }

    if (!load_labels("mnist-labels.txt", labels)) {
        std::cerr << "Failed to load labels." << std::endl;
        return 1;
    }

    if (images.size() != labels.size()) {
        std::cerr << "Image count and label count mismatch!" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << images.size() << " images and labels." << std::endl;

    // Try different values of K
    std::vector<int> k_values = {5, 10, 15, 20};
    int best_k = -1;
    double best_nmi = -1.0;

    for (int K : k_values) {
        std::vector<std::vector<float>> centroids;
        initialize_centroids(centroids, images, K); // Modified to pass K
        std::cout << "\nRunning Mini-Batch K-Means with K = " << K << " ...\n";

        auto start = std::chrono::high_resolution_clock::now();

        parallel_mini_batch_kmeans(images, centroids, 1000, K); // batch size = 1000

        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        std::vector<int> predicted = assign_clusters(images, centroids);
        double nmi = compute_NMI(predicted, labels, K);

        std::cout << "K = " << K << ", NMI = " << nmi << ", Time = " << time << " sec\n";

        if (nmi > best_nmi) {
            best_nmi = nmi;
            best_k = K;
        }
    }

    std::cout << "\n✅ Best K = " << best_k << " with NMI = " << best_nmi << std::endl;

    return 0;
}
