
// KDTree
#include "functions.h"
#include "nanoflann.hpp"
#include "KDTreeTableAdaptor.h"
using namespace nanoflann;


#include <random>
#include <algorithm>
#include <iterator>
#include <numeric>

using namespace std;

#include <omp.h>

#include <torch/extension.h>

typedef KDTreeTableAdaptor< float, float> KDTree;

torch::Tensor knn(const torch::Tensor points, const torch::Tensor queries, const size_t K){

	// get the sizes
	size_t B = points.size(0);
	size_t D = points.size(1);
	size_t N = points.size(2);

	// get the points
	auto pts = points.transpose(1,2).contiguous();
	assert(pts.dtype() == torch::kFloat32);
	const float* pts_data = pts.data_ptr<float>();

	// queries are points
	if(queries.dtype() == torch::kFloat32){
		size_t Q = queries.size(2);
		auto qrs = queries.transpose(1,2).contiguous();
		const float* qrs_data = qrs.data_ptr<float>();

		// create the tensor for indices
		auto indices = torch::zeros({long(B), long(Q), long(K)}, torch::kLong);
		auto indices_a = indices.accessor<long,3>();

		// iterate over the batch
		#pragma omp parallel for
		for(size_t b=0; b < B; b++){

			// get a float pointer to data
			const float* pts_b = &pts_data[b*N*D];
			const float* qrs_b = &qrs_data[b*Q*D];

			// create the kdtree
			KDTree mat_index(N, D, pts_b, 10);
			mat_index.index->buildIndex();

			// create the containers for the queries
			std::vector<float> out_dists_sqr(K);
			std::vector<size_t> out_ids(K);

			// iterate over the queries
			for(size_t q=0; q<Q; q++){
				
				nanoflann::KNNResultSet<float> resultSet(K);
				resultSet.init(&out_ids[0], &out_dists_sqr[0] );

				mat_index.index->findNeighbors(resultSet, &qrs_b[q*D], nanoflann::SearchParams(10));
				// fill the queries and neighborhoods
				for(size_t j=0; j<K; j++){
					indices_a[b][q][j] = long(out_ids[j]);
				}
			}
		}

		return indices;

	}else{ // assume it is an integer tensor with indices
		size_t Q = queries.size(1);
		
		// create the tensor for indices
		auto indices = torch::zeros({long(B), long(Q), long(K)}, torch::kLong);
		auto indices_a = indices.accessor<long,3>();
		auto queries_a = queries.accessor<long,2>();


		// iterate over the batch
		#pragma omp parallel for
		for(size_t b=0; b < B; b++){

			// get a float pointer to data
			const float* pts_b = &pts_data[b*N*D];

			// create the kdtree
			KDTree mat_index(N, D, pts_b, 10);
			mat_index.index->buildIndex();

			// create the containers for the queries
			std::vector<float> out_dists_sqr(K);
			std::vector<size_t> out_ids(K);

			// iterate over the queries
			for(size_t q=0; q<Q; q++){
				
				nanoflann::KNNResultSet<float> resultSet(K);
				resultSet.init(&out_ids[0], &out_dists_sqr[0] );

				mat_index.index->findNeighbors(resultSet, &pts_b[queries_a[b][q]*D], nanoflann::SearchParams(10));
				// fill the queries and neighborhoods
				for(size_t j=0; j<K; j++){
					indices_a[b][q][j] = long(out_ids[j]);
				}
			}
		}

		return indices;
	}
}
