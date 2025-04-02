#ifndef PARALLEL_KMEANS_H
#define PARALLEL_KMEANS_H

#include <vector>

void parallel_mini_batch_kmeans(std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& centroids, int batch_size, int K);

#endif
