// KDTree
#include "knn.h"
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



std::vector<torch::Tensor> convpoint_pick_knn(const torch::Tensor points, const size_t nqueries,
		const size_t K){

	// create the random machine
	mt19937 mt_rand(time(0));

	// get the sizes
	size_t B = points.size(0);
	size_t D = points.size(1);
	size_t N = points.size(2);

	if(nqueries >= N){
		return random_pick_knn(points, nqueries, K);
	}

	auto pts = points.transpose(1,2).contiguous();
	assert(pts.dtype() == torch::kFloat32);
	const float* pts_data = pts.data_ptr<float>();

	auto indices = torch::zeros({long(B), long(nqueries), long(K)}, torch::kLong);
	auto indices_a = indices.accessor<long,3>();
	auto indices_queries = torch::zeros({long(B), long(nqueries)}, torch::kLong);
	auto indices_queries_a = indices_queries.accessor<long,2>();
	auto query_points = torch::zeros({long(B), long(D), long(nqueries)}, torch::kFloat32);
	auto query_points_a = query_points.accessor<float,3>();

	#pragma omp parallel for
	for(size_t b=0; b < B; b++){

		// get a float pointer to data
		const float* pts_b = &pts_data[b*N*D];

		// create the kdtree
		KDTree mat_index(N, D, pts_b, 10);
		mat_index.index->buildIndex();

		// create the memory vector for used points
		vector<int> used(N, 0);

		// iterate over points
		int current_id = 0;
		// iterate over the queries
		for(size_t q=0; q<nqueries; q++){

			// get the possible points
			vector<size_t> possible_ids;
			while(possible_ids.size() == 0){
				for(size_t i=0; i<N; i++){
					if(used[i] == current_id){
						possible_ids.push_back(i);
					}
				}
				if(possible_ids.size() == 0){
					current_id = *std::min_element(std::begin(used), std::end(used));
				}
			}

			// get an index
			size_t index = possible_ids[mt_rand()%possible_ids.size()];

			// create the query
			vector<float> query_pt(3);
			for(size_t j=0; j<D; j++){
				query_pt[j] = pts_b[index*D+j];
			}
			
			// create the containers for the queries
			std::vector<float> out_dists_sqr(K);
			std::vector<size_t> out_ids(K);
			nanoflann::KNNResultSet<float> resultSet(K);
			resultSet.init(&out_ids[0], &out_dists_sqr[0] );

			mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

			// fill the queries and neighborhoods
			for(size_t j=0; j<K; j++){
				indices_a[b][q][j] = long(out_ids[j]);
			}
			indices_queries_a[b][q] = index;
			for(size_t j=0; j<D; j++){
				query_points_a[b][j][q] = pts_b[index*D+j];
			}

			for(size_t j=0; j<K; j++){
				used[out_ids[j]] ++;
			}
			used[index] += 100;

		}
	}

	return {indices_queries, indices, query_points};
}


std::vector<torch::Tensor> convpoint_pick_knn_dev(const torch::Tensor points, const size_t nqueries,
		const size_t K){

	// create the random machine
	mt19937 mt_rand(time(0));

	// get the sizes
	size_t B = points.size(0);
	size_t D = points.size(1);
	size_t N = points.size(2);

	if(nqueries >= N){
		return random_pick_knn(points, nqueries, K);
	}

	auto pts = points.transpose(1,2).contiguous();
	assert(pts.dtype() == torch::kFloat32);
	const float* pts_data = pts.data_ptr<float>();

	auto indices = torch::zeros({long(B), long(nqueries), long(K)}, torch::kLong);
	auto indices_a = indices.accessor<long,3>();
	auto indices_queries = torch::zeros({long(B), long(nqueries)}, torch::kLong);
	auto indices_queries_a = indices_queries.accessor<long,2>();
	auto query_points = torch::zeros({long(B), long(D), long(nqueries)}, torch::kFloat32);
	auto query_points_a = query_points.accessor<float,3>();

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

		// create the memory vector for used points
		vector<size_t> used(N, 0);

		size_t q=0;
		size_t i=0;
		size_t current_point_level = 0;
		while(q<nqueries){
			
			// iterate to find the next candidate
			size_t ptid=rand_indices[i];
			size_t jump=1e7;
			size_t count=0;
			while(used[ptid]!=current_point_level and count<N){
				ptid = rand_indices[(i+count)%N];
				jump = std::min(jump, used[ptid] - current_point_level);
				count+=1;
			}

			if(count==N){
				// we have seen all the points operate a jump
				current_point_level += jump;
				continue;
			}

			// create the query
			vector<float> query_pt(3);
			for(size_t j=0; j<D; j++){
				query_pt[j] = pts_b[ptid*D+j];
			}
			
			// create the containers for the queries
			std::vector<float> out_dists_sqr(K);
			std::vector<size_t> out_ids(K);
			nanoflann::KNNResultSet<float> resultSet(K);
			resultSet.init(&out_ids[0], &out_dists_sqr[0] );

			mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

			// fill the queries and neighborhoods
			for(size_t j=0; j<K; j++){
				indices_a[b][q][j] = long(out_ids[j]);
			}
			indices_queries_a[b][q] = ptid;
			for(size_t j=0; j<D; j++){
				query_points_a[b][j][q] = pts_b[ptid*D+j];
			}
			for(size_t j=0; j<K; j++){
				used[out_ids[j]]++;
			}
			used[ptid] += 100;

			// next query
			i = (i+count)%N;
			q+=1;
		}
	}

	return {indices_queries, indices, query_points};
}
