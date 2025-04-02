#ifndef UTILS_H
#define UTILS_H

#include <vector>

float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b);
int find_closest_centroid(const std::vector<float>& image, const std::vector<std::vector<float>>& centroids);
double compute_NMI(const std::vector<int>& predicted, const std::vector<int>& ground_truth, int K);
std::vector<int> assign_clusters(const std::vector<std::vector<float>>& images, const std::vector<std::vector<float>>& centroids);
std::vector<std::vector<float>> augment_with_noise(const std::vector<std::vector<float>>&, int multiplier, float stddev);


#endif
