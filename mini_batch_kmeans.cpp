#include "mini_batch_kmeans.h"
#include "utils.h"
#include <iostream>
#include <random>
#include <unordered_set>

const int K = 5;              
const int BATCH_SIZE = 50;
const int MAX_ITERATIONS = 100;


void initialize_centroids(std::vector<std::vector<float>>& centroids,
                          const std::vector<std::vector<float>>& images,
                          int K) {
    centroids.clear();
    std::unordered_set<int> selected;
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(0, images.size() - 1);

    while (centroids.size() < K) {
        int idx = dist(gen);
        if (selected.insert(idx).second) {
            centroids.push_back(images[idx]);
        }
    }
}


void mini_batch_kmeans(std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& centroids) {
    std::vector<int> cluster_counts(K, 0);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        std::vector<int> batch_indices(BATCH_SIZE);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, images.size() - 1);

        for (int i = 0; i < BATCH_SIZE; i++) {
            batch_indices[i] = dist(gen);
        }

        for (int i : batch_indices) {
            int closest = find_closest_centroid(images[i], centroids);
            cluster_counts[closest] += 1;
            float eta = 1.0 / cluster_counts[closest];

            for (size_t j = 0; j < images[i].size(); j++) {
                centroids[closest][j] = (1 - eta) * centroids[closest][j] + eta * images[i][j];
            }
        }

        std::cout << "Iteration " << iter + 1 << " completed.\n";
    }

    std::cout << "Mini-Batch K-Means clustering completed!" << std::endl;
    std::cout << "\nFinal Centroids:\n";
for (int centroid = 0; centroid < K; centroid++) {
    std::cout << "centroid:"<< centroid<<" \n";
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++)
            std::cout << images[centroid][i * 28 + j] << " ";  // 🔥 Ensure images[0] is valid
        std::cout << std::endl;
    }
   
}
}
