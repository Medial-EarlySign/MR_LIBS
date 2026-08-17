//
// MedCluster - Clustering algorithms
//

#include "MedAlgo.h"
#include <MedUtils/MedUtils/MedGlobalRNG.h>
#include <MedUtils/MedUtils/MedGenUtils.h>
#include <MedMat/MedMat/MedMat.h>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Jacobi>


#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

#define KMEANS_MAX_DIST		1e10
#define KMEANS_EPSILON		1e-3
#define KMEANS_DEF_MAX_ITER	100

using namespace Eigen;

//================================================================================
// KMEANS
//================================================================================


//.................................................................................................
float Euclidian_Norm(float *x, float *y, int n)
{
	float L2 = 0;

	for (int i = 0; i < n; i++)
		L2 += (x[i] - y[i])*(x[i] - y[i]);

	//	L2 = sqrt(L2);
	return L2;
}

//.................................................................................................
int KMeans_choose_start_centers(float *x, int nrows, int ncols, int K, float *centers)
{
	int i, j, k;

	// first center is simply a random one
	int r = rand_N(nrows);
	for (j = 0; j < ncols; j++)
		centers[j] = x[r*ncols + j];

	vector<float> dists(nrows, (float)KMEANS_MAX_DIST);
	for (k = 1; k < K; k++) {

		// we now choose the next point as the one for which the min distance to the already chosen points is maximal
		float max_dist = 0;
		int max_ind = -1;
		for (i = 0; i < nrows; i++) {
			float d = Euclidian_Norm(&centers[(k - 1)*ncols], &x[i*ncols], ncols);
			if (d < dists[i])
				dists[i] = d;
			if (dists[i] > max_dist) {
				max_dist = dists[i];
				max_ind = i;
			}
		}

		if (max_ind >= 0)
			for (j = 0; j < ncols; j++)
				centers[k*ncols + j] = x[max_ind*ncols + j];
		else
			return -1;

	}

	return 0;
}

//.................................................................................................
// sizes are: x: nrows x ncols , centers K x ncols , clusters: nrows, dists: nrows x K
int KMeans(float *x, int nrows, int ncols, int K, int max_iter, float *centers, int *clusters, float *dists, bool verbose_print)
{
	// choose start points
	if (KMeans_choose_start_centers(x, nrows, ncols, K, centers) < 0) {
		MERR("KMeans: Failed choosing start centers\n");
		return -1;
	}

	// loop up to convergence or max_iter
	int niter = 0;
	int go_on = 1;
	int i, j, k;
	vector<float> new_centers(static_cast<size_t>(K) * static_cast<size_t>(ncols), (float)0);
	vector<int> n_in_cluster(K, 0);

	float rss_prev = (float)nrows*(float)ncols*(float)1000;
	float rss_curr = 0;
	while (go_on) {

		fill(new_centers.begin(), new_centers.end(), (float)0);
		fill(n_in_cluster.begin(), n_in_cluster.end(), 0);
		rss_curr = 0;

		// calculate dists and assign clusters
		for (i = 0; i < nrows; i++) {
			float min_dist = (float)KMEANS_MAX_DIST;
			int min_ind = -1;
			for (k = 0; k < K; k++) {
				float d = Euclidian_Norm(&centers[k*ncols], &x[i*ncols], ncols);
				dists[i*K + k] = d;
				if (d < min_dist) {
					min_dist = d;
					min_ind = k;
				}
			}

			clusters[i] = min_ind;
			rss_curr += min_dist;

			// accumulating for next iteration
			for (j = 0; j < ncols; j++)
				new_centers[clusters[i] * ncols + j] += x[i*ncols + j];
			n_in_cluster[clusters[i]]++;
		}

		// stop criteria
		if (verbose_print)
			MLOG("KMeans %d: Iter %d/%d : prev_RSS %f curr_RSS %f : \n", K, niter, max_iter, rss_prev, rss_curr);
		//for (k=0; k<K; k++) MLOG(" (%d) %d",k,n_in_cluster[k]);
		//MLOG("\n");
		niter++;
		if (niter > max_iter || abs(rss_prev - rss_curr) < (float)KMEANS_EPSILON*rss_prev) {
			go_on = 0;
		}
		else {
			// assign new centers and iterate once more
			for (k = 0; k < K; k++)
				if (n_in_cluster[k] > 0)
					for (j = 0; j < ncols; j++)
						centers[k*ncols + j] = new_centers[k*ncols + j] / (float)n_in_cluster[k];
			rss_prev = rss_curr;
		}

	}

	if (verbose_print) {
		MLOG("##KMeans (%d x %d) %d: Iter %d/%d : prev_RSS %f curr_RSS %f : ", nrows, ncols, K, niter, max_iter, rss_prev, rss_curr);
		//for (k=0; k<K; k++) MLOG(" (%d) %d",k,n_in_cluster[k]);
		MLOG("\n");
	}

	return 0;
}

//.................................................................................................
int KMeans(MedMat<float> &x, int K, int max_iter, MedMat<float> &centers, vector<int> &clusters, MedMat<float> &dists)
{
	centers.resize(K, x.ncols);
	clusters.resize(x.nrows);
	dists.resize(x.nrows, K);

	centers.signals = x.signals;

	return KMeans(x.data_ptr(), x.nrows, x.ncols, K, max_iter, centers.data_ptr(), VEC_DATA(clusters), dists.data_ptr());
}

//.................................................................................................
int KMeans(MedMat<float> &x, int K, MedMat<float> &centers, vector<int> &clusters, MedMat<float> &dists)
{
	return KMeans(x, K, KMEANS_DEF_MAX_ITER, centers, clusters, dists);
}

//.................................................................................................
int KMeans(float *x, int nrows, int ncols, int K, float *centers, int *clusters, float *dists)
{
	return KMeans(x, nrows, ncols, K, KMEANS_DEF_MAX_ITER, centers, clusters, dists);
}


//================================================================================
// PCA
//================================================================================

//.................................................................................................
int MedPCA(MedMat<float> &x, MedMat<float> &pca_base, vector<float> &varsum)
{

	int i, j;
	// We first prepare a matrix in Eigen format
	MatrixXf x_in;

	x_in.resize(x.nrows, x.ncols);
	for (i = 0; i < x.nrows; i++)
		for (j = 0; j < x.ncols; j++)
			x_in(i, j) = x(i, j);

	// Actual SVD
	JacobiSVD<MatrixXf> J(x_in, ComputeThinV);

	// get base mat and load it to output mat
	MatrixXf V = J.matrixV().leftCols(x.ncols);

	pca_base.resize(x.ncols, x.ncols);
	for (i = 0; i < x.ncols; i++)
		for (j = 0; j < x.ncols; j++)
			pca_base(i, j) = V(j, i);

	// get the singular values (which are roots of the eigen values), and the cummulative relative size
	VectorXf sv = J.singularValues();

	float sum = 0;
	varsum.resize(x.ncols);
	for (i = 0; i < x.ncols; i++) {
		sum += sv(i)*sv(i);
		varsum[i] = sum;
	}

	if (sum > 0) {
		for (i = 0; i < x.ncols; i++)
			varsum[i] = varsum[i] / sum;
	}

	return 0;

}

//.................................................................................................
int MedPCA_project(MedMat<float> &x, MedMat<float> &pca_base, int dim, MedMat<float> &projected)
{
	int i, j, k;

	projected.resize(x.nrows, dim);
	projected.zero();

	for (i = 0; i < x.nrows; i++) {
		for (j = 0; j < dim; j++) {
			for (k = 0; k < x.ncols; k++)
				projected(i, j) += x(i, k) * pca_base(j, k);
		}
	}

	return 0;
}

