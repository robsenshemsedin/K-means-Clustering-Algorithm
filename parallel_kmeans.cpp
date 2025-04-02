#include "utils.h"
#include <omp.h>
#include <random>
#include <iostream>

void parallel_mini_batch_kmeans(std::vector<std::vector<float>>& images,
                                std::vector<std::vector<float>>& centroids,
                                int batch_size,
                                int K) {
    int num_features = images[0].size();
    int num_iterations = 100;
    int n_samples = images.size();

    std::vector<int> cluster_counts(K, 0);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(0, n_samples - 1);

    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<int> batch_indices(batch_size);
        for (int i = 0; i < batch_size; i++) {
            batch_indices[i] = dist(gen);
        }

        std::vector<std::vector<float>> sum_centroids(K, std::vector<float>(num_features, 0.0f));
        std::vector<int> local_counts(K, 0);

        // Assign batch points to nearest centroid
        #pragma omp parallel for
        for (int i = 0; i < batch_size; ++i) {
            int idx = batch_indices[i];
            int nearest = find_closest_centroid(images[idx], centroids);

            #pragma omp critical
            {
                for (int j = 0; j < num_features; ++j) {
                    sum_centroids[nearest][j] += images[idx][j];
                }
                local_counts[nearest]++;
            }
        }

        // Update centroids
        for (int k = 0; k < K; ++k) {
            if (local_counts[k] > 0) {
                float eta = 1.0f / (++cluster_counts[k]);
                for (int j = 0; j < num_features; ++j) {
                    centroids[k][j] = (1 - eta) * centroids[k][j] + eta * (sum_centroids[k][j] / local_counts[k]);
                }
            }
        }

        std::cout << "Parallel Iteration " << iter + 1 << " completed." << std::endl;
    }

    std::cout << "Parallel Mini-Batch K-Means clustering completed!" << std::endl;
}
