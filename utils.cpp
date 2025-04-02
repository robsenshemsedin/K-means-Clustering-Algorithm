#include "utils.h"
#include <random>

#include <cmath>

float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

int find_closest_centroid(const std::vector<float>& image, const std::vector<std::vector<float>>& centroids) {
    int best_index = 0;
    float best_dist = euclidean_distance(image, centroids[0]);

    for (int i = 1; i < centroids.size(); i++) {
        float dist = euclidean_distance(image, centroids[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_index = i;
        }
    }
    return best_index;
}

std::vector<int> assign_clusters(const std::vector<std::vector<float>>& images, const std::vector<std::vector<float>>& centroids) {
    std::vector<int> assignments(images.size());

    #pragma omp parallel for
    for (int i = 0; i < images.size(); ++i) {
        assignments[i] = find_closest_centroid(images[i], centroids);
    }

    return assignments;
}
#include <unordered_map>
#include <cmath>
#include <random>

double compute_NMI(const std::vector<int>& predicted, const std::vector<int>& ground_truth, int K) {
    int N = predicted.size();
    std::unordered_map<int, int> label_freq;
    std::unordered_map<int, int> cluster_freq;
    std::unordered_map<int, std::unordered_map<int, int>> contingency;

    for (int i = 0; i < N; ++i) {
        int label = ground_truth[i];
        int cluster = predicted[i];
        label_freq[label]++;
        cluster_freq[cluster]++;
        contingency[cluster][label]++;
    }

    double I = 0.0; // Mutual Information
    for (const auto& cluster_pair : contingency) {
        int cluster = cluster_pair.first;
        for (const auto& label_pair : cluster_pair.second) {
            int label = label_pair.first;
            int n_ij = label_pair.second;
            double p_ij = static_cast<double>(n_ij) / N;
            double p_i = static_cast<double>(cluster_freq[cluster]) / N;
            double p_j = static_cast<double>(label_freq[label]) / N;
            I += p_ij * log(p_ij / (p_i * p_j));
        }
    }

    // Entropies
    double H_pred = 0.0, H_true = 0.0;
    for (const auto& pair : cluster_freq) {
        double p = static_cast<double>(pair.second) / N;
        H_pred -= p * log(p);
    }
    for (const auto& pair : label_freq) {
        double p = static_cast<double>(pair.second) / N;
        H_true -= p * log(p);
    }

    double NMI = (H_pred + H_true == 0) ? 0 : (2 * I) / (H_pred + H_true);
    return NMI;
}


std::vector<std::vector<float>> augment_with_noise(const std::vector<std::vector<float>>& original, int multiplier, float stddev) {
    std::vector<std::vector<float>> augmented;
    std::default_random_engine generator;
    std::normal_distribution<float> noise(0.0, stddev);

    for (int m = 0; m < multiplier; ++m) {
        for (const auto& img : original) {
            std::vector<float> noisy = img;
            for (float& pixel : noisy) {
                pixel += noise(generator);
                pixel = std::max(0.0f, std::min(pixel, 255.0f)); // Clamp
            }
            augmented.push_back(noisy);
        }
    }
    return augmented;
}
