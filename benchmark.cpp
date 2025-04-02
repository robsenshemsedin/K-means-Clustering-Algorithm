#include "mnist_loader.h"
#include "parallel_kmeans.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include "mini_batch_kmeans.h"



int main() {
    std::vector<std::vector<float>> images;
    load_MNIST("mnist-images.txt", images);
    std::cout << "Loaded " << images.size() << " images.\n";

    // List of batch sizes to test
    std::vector<int> batch_sizes = {10, 50, 100, 500, 1000};

    // Test for each batch size
    for (int b : batch_sizes) {
        std::vector<std::vector<float>> centroids;
        initialize_centroids(centroids, images);

        // Set batch size for the algorithm
        std::cout << "\nRunning with batch size b = " << b << " ...\n";
        
        auto start = std::chrono::high_resolution_clock::now();

        // Run parallel version with current batch size
        parallel_mini_batch_kmeans(images, centroids, b); // <== We’ll update the function to take `b`

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Execution Time for b = " << b << ": " << duration << " ms\n";
    }

    return 0;
}
