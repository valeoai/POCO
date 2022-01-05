
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

torch::Tensor radius(const torch::Tensor points, const torch::Tensor queries, const float radius, const size_t max_K){

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
		auto indices = torch::full({long(B), long(Q), long(max_K)}, -1, torch::kLong);
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

			for (size_t q=0; q<Q; q++){

				std::vector<std::pair<size_t,float> >   ret_matches;
				nanoflann::SearchParams params;
				const size_t nMatches = mat_index.index->radiusSearch(&qrs_b[q*D], radius, ret_matches, params);

				if(nMatches > max_K){
					std::random_shuffle ( ret_matches.begin(), ret_matches.end() );
				}

				size_t num_ret = std::min(nMatches, max_K);
				for(size_t j=0; j<num_ret; j++){
					indices_a[b][q][j] = long(ret_matches[j].first);
				}
			}



		}

		return indices;

	}else{ // assume it is an integer tensor with indices
		size_t Q = queries.size(1);
		
		// create the tensor for indices
		auto indices = torch::full({long(B), long(Q), long(max_K)},-1, torch::kLong);
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

			// iterate over the queries
			for(size_t q=0; q<Q; q++){

				std::vector<std::pair<size_t,float> >   ret_matches;
				nanoflann::SearchParams params;
				const size_t nMatches = mat_index.index->radiusSearch(&pts_b[queries_a[b][q]*D], radius, ret_matches, params);

				if(nMatches > max_K){
					std::random_shuffle ( ret_matches.begin(), ret_matches.end() );
				}

				size_t num_ret = std::min(nMatches, max_K);
				for(size_t j=0; j<num_ret; j++){
					indices_a[b][q][j] = long(ret_matches[j].first);
				}

			}
		}

		return indices;
	}
}
