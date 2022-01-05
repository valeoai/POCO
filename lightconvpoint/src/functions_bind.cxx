#include "functions.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn", &knn, "knn computation");
  m.def("radius", &radius, "radius nn computation");
  m.def("sampling_quantized", &sampling_quantized, "sampling computation with voxels");
  m.def("sampling_fps", &sampling_fps, "sampling computation with farthest point sampling");
  m.def("sampling_random", &sampling_random, "sampling computation with random sampling");
  m.def("sampling_knn_random", &sampling_knn_random, "knn computation with random sampling");
  m.def("sampling_knn_convpoint", &sampling_knn_random, "knn computation with convpoint sampling");
  m.def("sampling_knn_quantized", &sampling_knn_random, "knn computation with quantized sampling");
  m.def("sampling_knn_fps", &sampling_knn_random, "knn computation with farthest point sampling");
}