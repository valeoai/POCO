
// KDTree
#include "functions.h"
#include "nanoflann.hpp"
#include "KDTreeTableAdaptor.h"
using namespace nanoflann;


#include <random>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <iostream>

using namespace std;

#include <omp.h>

#include <torch/extension.h>

typedef KDTreeTableAdaptor< float, float> KDTree;

torch::Tensor sampling_random(const torch::Tensor points, const size_t nqueries){

	// create the random machine
	mt19937 mt_rand(time(0));

	// get the sizes
	size_t B = points.size(0);
	size_t D = points.size(1);
	size_t N = points.size(2);

	auto pts = points.transpose(1,2).contiguous();
	assert(pts.dtype() == torch::kFloat32);
	const float* pts_data = pts.data_ptr<float>();

	auto indices_queries = torch::zeros({long(B), long(nqueries)}, torch::kLong);
	auto indices_queries_a = indices_queries.accessor<long,2>();

	// iterate over the batch
	#pragma omp parallel for
	for(size_t b=0; b < B; b++){

		// get a float pointer to data
		const float* pts_b = &pts_data[b*N*D];

		// create the index vector
		std::vector<int> rand_indices(N);
		std::iota(rand_indices.begin(), rand_indices.end(), 0);
		std::random_shuffle ( rand_indices.begin(), rand_indices.end() );

		// iterate over the queries
		for(size_t q=0; q<nqueries; q++){
			indices_queries_a[b][q] = rand_indices[q%N];
		}
	}

	return indices_queries;
}


std::vector<torch::Tensor> sampling_knn_random(const torch::Tensor points, const size_t nqueries,
		const size_t K){

	// create the random machine
	mt19937 mt_rand(time(0));

	// get the sizes
	size_t B = points.size(0);
	size_t D = points.size(1);
	size_t N = points.size(2);

	auto pts = points.transpose(1,2).contiguous();
	assert(pts.dtype() == torch::kFloat32);
	const float* pts_data = pts.data_ptr<float>();

	auto indices = torch::zeros({long(B), long(nqueries), long(K)}, torch::kLong);
	auto indices_a = indices.accessor<long,3>();
	auto indices_queries = torch::zeros({long(B), long(nqueries)}, torch::kLong);
	auto indices_queries_a = indices_queries.accessor<long,2>();
	auto query_points = torch::zeros({long(B), long(D), long(nqueries)}, torch::kFloat32);
	auto query_points_a = query_points.accessor<float,3>();

	// iterate over the batch
	#pragma omp parallel for
	for(size_t b=0; b < B; b++){

		// get a float pointer to data
		const float* pts_b = &pts_data[b*N*D];

		// create the index vector
		std::vector<int> rand_indices(N);
		std::iota(rand_indices.begin(), rand_indices.end(), 0);
		std::random_shuffle ( rand_indices.begin(), rand_indices.end() );

		// create the kdtree
		KDTree mat_index(N, D, pts_b, 10);
		mat_index.index->buildIndex();

		// iterate over the queries
		for(size_t q=0; q<nqueries; q++){
			
			// create the containers for the queries
			std::vector<float> out_dists_sqr(K);
			std::vector<size_t> out_ids(K);
			nanoflann::KNNResultSet<float> resultSet(K);
			resultSet.init(&out_ids[0], &out_dists_sqr[0] );

			vector<float> query_pt(3);
			for(size_t j=0; j<D; j++){
				query_pt[j] = pts_b[rand_indices[q%N]*D+j];
			}

			mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

			// fill the queries and neighborhoods
			for(size_t j=0; j<K; j++){
				indices_a[b][q][j] = long(out_ids[j]);
			}
			indices_queries_a[b][q] = rand_indices[q%N];
			for(size_t j=0; j<D; j++){
				query_points_a[b][j][q] = pts_b[rand_indices[q%N]*D+j];
			}
		}
	}

	return {indices_queries, indices, query_points};
}
