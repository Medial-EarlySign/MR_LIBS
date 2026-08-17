//Clustering functions

#include "medial_utilities.h"
#include "globalRNG.h"

int kmeans (double *x, int nrows, int ncols, int k, int *clusters, double *means) {

	int *counts = (int *) malloc(k*sizeof(int)) ;
	if (counts==NULL) {
		fprintf(stderr,"Allocation failed\n") ;
		return -1 ;
	}

	// Random assignment
	for (int i=0; i<nrows; i++)
		clusters[i] = globalRNG::rand()%k ;

	// Iterate
	int nchange = 1 ;
	int iter = 0 ;
	while (nchange) {
		fprintf(stderr,"Kmeans interation %d\n",++iter) ;

		// Get Means
		memset(means,0,static_cast<size_t>(ncols)*static_cast<size_t>(k)*sizeof(double)) ;
		memset(counts,0,k*sizeof(int)) ;

		for (int i=0; i<nrows; i++) {
			for (int j=0; j<ncols; j++)
				means[XIDX(clusters[i],j,ncols)] += x[XIDX(i,j,ncols)] ;
			counts[clusters[i]]++ ;
		}

		for (int i=0; i<k; i++) {
			for (int j=0; j<ncols; j++)
				means[XIDX(i,j,ncols)] /= counts[i] ;
		}

		// Move samples
		nchange = 0  ;
		for (int i=0; i<nrows; i++) {
			int new_cls ;
			double min_dist  ;

			for (int cls=0; cls<k; cls++) {
				double dist = 0 ;
				for (int j=0; j<ncols; j++)
					dist += (x[XIDX(i,j,ncols)] - means[XIDX(cls,j,ncols)]) * (x[XIDX(i,j,ncols)] - means[XIDX(cls,j,ncols)]) ;

				if (cls==0 || dist < min_dist) {
					min_dist = dist ;
					new_cls = cls ;
				}
			}

			if (new_cls != clusters[i]) {
				nchange++ ;
				clusters[i] = new_cls ;
			}
		}

		fprintf(stderr,"k-means : %d changes\n",nchange) ;
	}

	free(counts) ;
	return 0 ;
}


int get_closest(double *x,int nrows, int ncols, double *mean) {

	int idx ;
	double min_dist  ;

	for (int i=0; i<nrows; i++) {
		double dist = 0 ;

		for (int j=0; j<ncols; j++)
			dist += (x[XIDX(i,j,ncols)] - mean[j])*(x[XIDX(i,j,ncols)] - mean[j]);

		if (i==0 || dist < min_dist) {
			idx = i ;
			min_dist = dist ;
		}
	}

	return idx ;
}