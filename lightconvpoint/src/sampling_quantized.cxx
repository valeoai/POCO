// KDTree
#include "functions.h"
#include "nanoflann.hpp"
#include "KDTreeTableAdaptor.h"
using namespace nanoflann;
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <iostream>
#include <queue>

using namespace std;

#include <omp.h>
#include <torch/extension.h>
#include <fstream>


typedef KDTreeTableAdaptor< float, float> KDTree;

struct VectorComp {
	bool operator() (const vector<int>& lhs, const vector<int>& rhs) const
	{
		if(lhs[0]<rhs[0]) return true;
		else if(lhs[0]==rhs[0]){
			if(lhs[1]<rhs[1]) return true;
			else if(lhs[1]==rhs[1]){
				if(lhs[2]<rhs[2]) return true;
			}
		}
		return false;
	}
};

struct Vect{
	int x,y,z;
	Vect(){};
	Vect(int x_, int y_, int z_):x(x_), y(y_), z(z_){};
	Vect operator+(const Vect& v1) const{ 
		return Vect(v1.x+x, v1.y+y, v1.z+z);
	}
};


inline bool in_bounding_box(const Vect& v, const Vect& mini, const Vect& maxi){
	return (v.x >= mini.x and v.x <= maxi.x) and (v.y >= mini.y and v.y <= maxi.y) and (v.z >= mini.z and v.z <= maxi.z);
}

struct VectComp {
	bool operator() (const Vect& lhs, const Vect& rhs) const
	{
		if(lhs.x<rhs.x) return true;
		else if(lhs.x==rhs.x){
			if(lhs.y<rhs.y) return true;
			else if(lhs.y==rhs.y){
				if(lhs.z<rhs.z) return true;
			}
		}
		return false;
	}
};

torch::Tensor sampling_quantized(const torch::Tensor points, const size_t nqueries){

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
		
		// get min and max
		float min_x=1e7, min_y=1e7, min_z=1e7;
		float max_x=-1e7, max_y=-1e7, max_z=-1e7;
		for(size_t ptid=0; ptid<N; ptid++){
			if(pts_b[ptid*D+0]<min_x) min_x = pts_b[ptid*D+0];
			if(pts_b[ptid*D+0]>max_x) max_x = pts_b[ptid*D+0];
			if(pts_b[ptid*D+1]<min_y) min_y = pts_b[ptid*D+1];
			if(pts_b[ptid*D+1]>max_y) max_y = pts_b[ptid*D+1];
			if(pts_b[ptid*D+2]<min_z) min_z = pts_b[ptid*D+2];
			if(pts_b[ptid*D+2]>max_z) max_z = pts_b[ptid*D+2];
		}

		// create the index vector
		std::vector<int> rand_indices(N);
		std::iota(rand_indices.begin(), rand_indices.end(), 0);
		std::random_shuffle ( rand_indices.begin(), rand_indices.end() );

		// compute the expected size of the voxel
		float vox_size = std::sqrt((max_x-min_x)*(max_x-min_x)+(max_y-min_y)*(max_y-min_y)+(max_z-min_z)*(max_z-min_z)) / std::sqrt(nqueries);
		vector<size_t> selected_points;
		while(selected_points.size() < nqueries){
			std::set<Vect, VectComp > discrete_set;
			std::vector<int> next_rand_indices; 
			for(auto ptid : rand_indices){
				Vect v = { int(pts_b[ptid*D+0]/vox_size), int(pts_b[ptid*D+1]/vox_size), int(pts_b[ptid*D+2]/vox_size)};
				if(discrete_set.insert(v).second){ // has been inserted
					selected_points.push_back(ptid);
					if(selected_points.size() >= nqueries) break;
				}else{
					next_rand_indices.push_back(ptid);
				}
			}
			rand_indices = next_rand_indices;
			vox_size /=2;
			if(discrete_set.size()==0) break;
		}

		// iterate over the queries
		for(size_t q=0; q<nqueries; q++){
			size_t ptid = selected_points[q%selected_points.size()];
			indices_queries_a[b][q] = ptid;
		}
	}

	return indices_queries;
}

std::vector<torch::Tensor> sampling_knn_quantized(const torch::Tensor points, const size_t nqueries, const size_t K){

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

		// create the vectors for selected points
		
		// get min and max
		float min_x=1e7, min_y=1e7, min_z=1e7;
		float max_x=-1e7, max_y=-1e7, max_z=-1e7;
		for(size_t ptid=0; ptid<N; ptid++){
			if(pts_b[ptid*D+0]<min_x) min_x = pts_b[ptid*D+0];
			if(pts_b[ptid*D+0]>max_x) max_x = pts_b[ptid*D+0];
			if(pts_b[ptid*D+1]<min_y) min_y = pts_b[ptid*D+1];
			if(pts_b[ptid*D+1]>max_y) max_y = pts_b[ptid*D+1];
			if(pts_b[ptid*D+2]<min_z) min_z = pts_b[ptid*D+2];
			if(pts_b[ptid*D+2]>max_z) max_z = pts_b[ptid*D+2];
		}

		// compute the expected size of the voxel
		float vox_size = std::sqrt((max_x-min_x)*(max_x-min_x)+(max_y-min_y)*(max_y-min_y)+(max_z-min_z)*(max_z-min_z)) / std::sqrt(nqueries);
		vector<size_t> selected_points;
		while(selected_points.size() < nqueries){
			std::set<Vect, VectComp > discrete_set;
			std::vector<int> next_rand_indices; 
			for(auto ptid : rand_indices){
				Vect v = { int(pts_b[ptid*D+0]/vox_size), int(pts_b[ptid*D+1]/vox_size), int(pts_b[ptid*D+2]/vox_size)};
				if(discrete_set.insert(v).second){ // has been inserted
					selected_points.push_back(ptid);
					if(selected_points.size() >= nqueries) break;
				}else{
					next_rand_indices.push_back(ptid);
				}
			}
			rand_indices = next_rand_indices;
			vox_size /=2;
			if(discrete_set.size()==0) break;
		}

		// create the kdtree for
		KDTree mat_index(N, D, pts_b, 10);
		mat_index.index->buildIndex();

		// iterate over the queries
		for(size_t q=0; q<nqueries; q++){
			
			size_t ptid = selected_points[q%selected_points.size()];

			// create the containers for the queries
			std::vector<float> out_dists_sqr(K);
			std::vector<size_t> out_ids(K);
			nanoflann::KNNResultSet<float> resultSet(K);
			resultSet.init(&out_ids[0], &out_dists_sqr[0] );

			// create the query
			vector<float> query_pt(D);
			for(size_t j=0; j<D; j++){
				query_pt[j] = pts_b[ptid*D+j];
			}

			// search for the KNN
			mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

			// fill the queries and neighborhoods
			for(size_t j=0; j<K; j++){
				indices_a[b][q][j] = long(out_ids[j]);
			}
			indices_queries_a[b][q] = ptid;
			for(size_t j=0; j<D; j++){
				query_points_a[b][j][q] = pts_b[ptid*D+j];
			}
		}
	}

	return {indices_queries, indices, query_points};

}
