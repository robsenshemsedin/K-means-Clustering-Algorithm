#include "mnist_loader.h"
#include "utils.h"
#include "mini_batch_kmeans.h"
#include "parallel_kmeans.h"
#include <omp.h>
#include <chrono>
#include <iostream>
#include <vector>

int main() {
    std::vector<std::vector<float>> images;
    load_MNIST("mnist-images.txt", images);

    const int K = 20;
    const int batch_size = 500;
    const std::vector<int> thread_counts = {1, 2, 4, 8, 16};  // Modify based on CPU

    double T1 = -1.0;

    std::cout << "Loaded " << images.size() << " images.\n";
    std::cout << "Running with K = " << K << ", batch size = " << batch_size << "\n";

    for (int threads : thread_counts) {
        std::vector<std::vector<float>> centroids;
        initialize_centroids(centroids, images, K);

        omp_set_num_threads(threads);  // Set OpenMP thread count
        std::cout << "\nRunning with " << threads << " thread(s)...\n";

        auto start = std::chrono::high_resolution_clock::now();
        parallel_mini_batch_kmeans(images, centroids, batch_size, K);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        elapsed /= 1000.0; // to seconds

        if (threads == 1) T1 = elapsed;

        double speedup = (threads == 1) ? 1.0 : T1 / elapsed;
        std::cout << "Execution Time: " << elapsed << " sec | Speed-up: " << speedup << "\n";
    }

    return 0;
}
