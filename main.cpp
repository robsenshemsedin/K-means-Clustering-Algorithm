#include "mnist_loader.h"
#include "mini_batch_kmeans.h"
#include "parallel_kmeans.h"
#include <iostream>
#include <chrono>

int main() {
    std::vector<std::vector<float>> images;
    load_MNIST("mnist-images.txt", images);

    std::vector<std::vector<float>> centroids;
   // initialize_centroids(centroids, images);

    //mini_batch_kmeans(images, centroids);
   
    
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Running Parallel Mini-Batch K-Means...\n";
   // parallel_mini_batch_kmeans(images, centroids,5,5);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution Time: " << duration << " ms\n";

    return 0;
}

