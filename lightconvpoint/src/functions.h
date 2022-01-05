# pragma once
#include <torch/extension.h>

// KNN computation
torch::Tensor knn(const torch::Tensor points, const torch::Tensor queries, const size_t K);

torch::Tensor radius(const torch::Tensor points, const torch::Tensor queries, const float radius, const size_t max_K);

// sampling
torch::Tensor sampling_quantized(const torch::Tensor points, const size_t nqueries);

torch::Tensor sampling_fps(const torch::Tensor points, const size_t nqueries);

torch::Tensor sampling_random(const torch::Tensor points, const size_t nqueries);

// sampling convpoint would not be efficient as it requires computing knn, use sampling_knn_convpoint instead

// sampling + KNN
std::vector<torch::Tensor> sampling_knn_random(const torch::Tensor points, const size_t nqueries, const size_t K);

std::vector<torch::Tensor> sampling_knn_fps(const torch::Tensor points, const size_t nqueries, const size_t K);

std::vector<torch::Tensor> sampling_knn_quantized(const torch::Tensor points, const size_t nqueries, const size_t K);

std::vector<torch::Tensor> sampling_knn_convpoint(const torch::Tensor points, const size_t nqueries, const size_t K);

