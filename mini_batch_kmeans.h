#ifndef MINI_BATCH_KMEANS_H
#define MINI_BATCH_KMEANS_H

#include <vector>

void mini_batch_kmeans(std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& centroids);
int find_closest_centroid(const std::vector<float>& image, const std::vector<std::vector<float>>& centroids);
void initialize_centroids(std::vector<std::vector<float>>& centroids, const std::vector<std::vector<float>>& images, int K);

#endif
