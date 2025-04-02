#include "mnist_loader.h"
#include "utils.h"
#include "parallel_kmeans.h"
#include <chrono>
#include <iostream>
#include "mini_batch_kmeans.h"

int main() {
    std::vector<std::vector<float>> original_images;
    load_MNIST("mnist-images.txt", original_images);

    int K = 20;
    int batch_size = 500;
    int max_minutes = 5;
    int max_seconds = max_minutes * 60;

    int multiplier = 1;
    float stddev = 20.0;
    bool done = false;

    std::cout << "Starting Task 5: Clustering with Augmented Data within " << max_minutes << " minutes\n";

    while (!done) {
        std::vector<std::vector<float>> augmented = augment_with_noise(original_images, multiplier, stddev);
        std::vector<std::vector<float>> centroids;

        initialize_centroids(centroids, augmented, K);

        auto start = std::chrono::high_resolution_clock::now();
        parallel_mini_batch_kmeans(augmented, centroids, batch_size, K);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        std::cout << "Multiplier " << multiplier << " | Images: " << augmented.size()
                  << " | Time: " << elapsed_sec << " sec" << std::endl;

        if (elapsed_sec >= max_seconds) {
            std::cout << "\n✅ Largest dataset clustered under " << max_minutes << " minutes: "
                      << (multiplier - 1) * original_images.size() << " images.\n";
            break;
        }

        multiplier++;
    }

    return 0;
}
