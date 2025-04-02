# Mini-Batch K-Means Clustering (C++ with OpenMP)
This project implements a parallel Mini-Batch K-Means algorithm in C++ using OpenMP and applies it to the MNIST dataset. It explores how clustering performance varies with batch size, cluster count (K), thread count, and dataset size.

# Project Structure
- mini_batch_kmeans.cpp / .h – Sequential Mini-Batch K-Means

- parallel_kmeans.cpp / .h – Parallel version using OpenMP

- mnist_loader.cpp / .h – Loads images and labels from .txt files

- utils.cpp / .h – Includes distance, NMI calculation, and noise augmentation

- benchmark.cpp – Benchmarks different batch sizes

- benchmark_k_quality.cpp – Finds the best K using NMI

- benchmark_threads.cpp – Evaluates speed-up with varying threads

- benchmark_dataset_limit.cpp – Tests clustering with large datasets

# Requirements
- C++ compiler with OpenMP support (e.g., g++)
- MNIST dataset in .txt format (mnist-images.txt, mnist-labels.txt)

# Tasks Covered
- Batch size vs. runtime
- Best K (NMI)
- Thread scaling
- Clustering large datasets (up to 140,000 images)
