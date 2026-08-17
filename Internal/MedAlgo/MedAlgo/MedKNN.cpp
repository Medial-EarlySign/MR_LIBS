#define _CRT_SECURE_NO_WARNINGS

#include <MedAlgo/MedAlgo/MedAlgo.h>
#include <MedAlgo/MedAlgo/MedLM.h>
#include <MedAlgo/MedAlgo/MedKNN.h>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

// KNN Model
// Learn just keeps the data.
//Predict runs knn_predict

MedKNN::MedKNN() {
	MedKNNParams inParams;
	inParams.k = 50;
	inParams.knnAv = KNN_1_DIST;
	inParams.knnMetr = KNN_L1;
	init(&inParams);

}

MedKNN::MedKNN(MedKNNParams& _in_params) {

	classifier_type = MODEL_KNN;
	transpose_for_learn = false;
	transpose_for_predict = false;

	init((void *)&_in_params);
}

MedKNN::MedKNN(void *_in_params) {

	classifier_type = MODEL_KNN;
	transpose_for_learn = false;
	transpose_for_predict = false;

	init(_in_params);
}

int MedKNN::init(void *_in_params) {

	MedKNNParams in_params = *(MedKNNParams *)_in_params;

	params = in_params;
	assert(in_params.k > 0);
	assert(in_params.knnAv >= 0 && in_params.knnAv < KNN_AVG_LAST);
	x.clear(); y.clear(); w.clear();
	nftrs = nsamples = 0;
	transpose_for_learn = false;
	transpose_for_predict = false;
	normalize_for_learn = true;
	normalize_for_predict = true;
	normalize_y_for_learn = false;

	return 0;
}

int MedKNN::set_params(map<string, string>& mapper) {

	for (auto entry : mapper) {
		string field = entry.first;
		//! [MedKNN::init]
		if (field == "k") params.k = stoi(entry.second);
		else if (field == "knnAv") params.knnAv = get_knn_averaging(entry.second);
		else if (field == "knnMetr") params.knnMetr = get_knn_metric(entry.second);
		else MLOG("Unknonw parameter \'%s\' for Lasso\n", field.c_str());
		//! [MedKNN::init]
	}

	return 0;
}

knnAveraging MedKNN::get_knn_averaging(string name) {

	boost::algorithm::to_lower(name);
	if (name == "dist_mean")
		return KNN_DIST_MEAN;
	else if (name == "1_dist")
		return KNN_1_DIST;
	else if (name == "weightedls")
		return KNN_WEIGHTEDLS;
	else
		return KNN_AVG_LAST;
}

knnMetric MedKNN::get_knn_metric(string name) {

	if (name == "l1")
		return KNN_L1;
	else if (name == "l2")
		return KNN_L2;
	else
		return KNN_METRIC_LAST;
}


int MedKNN::Learn(float *_x, float *_y, const float *_w, int _nsamples, int _nftrs) {
	nsamples = _nsamples;
	nftrs = _nftrs;
	x = vector<float>(_x, _x+ nftrs * nsamples);
	y = vector<float>(_y, _y+ nsamples);
	if (_w)
		w = vector<float>(_w, _w + nsamples);
	else
		w.resize(nsamples, 1.0);
	
	return(0);
}

//prototypes for the working functions
int order_ftrs(const float *x, int nsamples, int nftrs, const float *weights, int *order);
int knn_predict(const float *test_x, int ind, const float *learn_x, const float *learn_y, int nlearn, int nftrs, const float *weights, int *order, int *nbrs, double *dists, int k,
	knnAveraging knnAv, knnMetric knnMetr, float *nbrs_x, float *nbrs_y, float *nbrs_w, float *nbrs_b, float *nbrs_r, float *pred);

template <class T> void clear_mem(T *&order) {
	if (order != NULL) {
		free(order);
		order = NULL;
	}
}

int MedKNN::Predict(float *xPred, float *&preds, int pred_samples, int _nftrs) const {
	assert(preds);
	assert(_nftrs == nftrs);


	if ((params.knnAv == KNN_WEIGHTEDLS) && params.k < nftrs) {
		fprintf(stderr, "k (%d) must be larger than nftrs (%d) in KNN+LS\n", params.k, nftrs);
		return -1;
	}

	// OK, lets go ...
	fprintf(stderr, "Running knn : K = %d , Data = (%d + %d) x %d\n", params.k, nsamples, nsamples, nftrs);

	// Allocation
	int *order = NULL, *nbrs = NULL;
	double *dists = NULL;

	if ((order = (int *)malloc(nftrs * sizeof(int))) == NULL || (nbrs = (int *)malloc(params.k * sizeof(int))) == NULL ||
		(dists = (double *)malloc(params.k * sizeof(double))) == NULL) {
		clear_mem<int>(order);
		clear_mem<int>(nbrs);
		clear_mem<double>(dists);
		fprintf(stderr, "Allocation failed\n");
		return -1;
	}

	float *nbrs_x = NULL, *nbrs_y = NULL, *nbrs_w = NULL, *nbrs_b = NULL, *nbrs_r = NULL;
	if (params.knnAv == KNN_WEIGHTEDLS) {
		if ((nbrs_x = (float *)malloc(static_cast<size_t>(params.k) * static_cast<size_t>(nftrs) * sizeof(float))) == NULL || (nbrs_y = (float *)malloc(static_cast<size_t>(params.k) * sizeof(float))) == NULL ||
			(nbrs_w = (float *)malloc(params.k * sizeof(float))) == NULL || (nbrs_b = (float *)malloc(nftrs * sizeof(float))) == NULL ||
			(nbrs_r = (float *)malloc(nftrs * sizeof(float))) == NULL) {
			fprintf(stderr, "nbrs data allocation failed\n");
			clear_mem<int>(order);
			clear_mem<int>(nbrs);
			clear_mem<double>(dists);
			clear_mem<float>(nbrs_x);
			clear_mem<float>(nbrs_y);
			clear_mem<float>(nbrs_w);
			clear_mem<float>(nbrs_b);
			clear_mem<float>(nbrs_r);
			return -1;
		}
	}

	// Order features
	if (order_ftrs(x.data(), nsamples, nftrs, w.data(), order) == -1) {
		clear_mem<int>(order);
		clear_mem<int>(nbrs);
		clear_mem<double>(dists);
		clear_mem<float>(nbrs_x);
		clear_mem<float>(nbrs_y);
		clear_mem<float>(nbrs_w);
		clear_mem<float>(nbrs_b);
		clear_mem<float>(nbrs_r);
		return -1;
	}

	for (int i = 0; i < pred_samples; i++) {
		if (i % 1000 == 1)
			fprintf(stderr, "Predicting %d/%d\n", i, pred_samples);
		if (knn_predict(xPred, i, x.data(), y.data(), nsamples, nftrs, w.data(), order, nbrs,
			dists, params.k, params.knnAv, params.knnMetr, nbrs_x, nbrs_y, nbrs_w, nbrs_b, nbrs_r, &(preds[i])) == -1) {
			fprintf(stderr, "knn prediction failed\n");
			clear_mem<int>(order);
			clear_mem<int>(nbrs);
			clear_mem<double>(dists);
			clear_mem<float>(nbrs_x);
			clear_mem<float>(nbrs_y);
			clear_mem<float>(nbrs_w);
			clear_mem<float>(nbrs_b);
			clear_mem<float>(nbrs_r);
			return -1;
		}
	}

	free(order);
	free(dists);
	free(nbrs);

	if (params.knnAv == KNN_WEIGHTEDLS) {
		free(nbrs_x);
		free(nbrs_y);
		free(nbrs_w);
		free(nbrs_b);
		free(nbrs_r);
	}


	return(0);
}


#define VIDX(i,j,ncol) ((i)*(ncol)+(j))

// Normalize data - set means to zero.
void tnormalize_data(float *x, float *y, int nsamples, int nftrs, const double *avg, double yavg, double missing = -1)
{
	// Reduce average from each column 


	for (int i = 0; i < nsamples; i++) {
		for (int j = 0; j < nftrs; j++) {
			if (x[XIDX(j, i, nsamples)] == missing)
				x[XIDX(j, i, nsamples)] = 0;
			else
				x[XIDX(j, i, nsamples)] -= (float)avg[j];
		}

		y[i] -= (float)yavg;
	}
}


// Find Average of a matrix column
float calc_col_avg(const float *table, int col, int nrow, int ncol, float missing)
{
	float sum = 0;
	float cntr = 0;
	for (int i = 0; i < nrow; i++) {
		if (table[VIDX(i, col, ncol)] == missing) continue;
		cntr += 1.0;
		sum += table[VIDX(i, col, ncol)];
	}
	float result = sum / cntr;
	return result;
}



float calc_col_std(const float *table, int col, int nrow, int ncol, float avg, float missing)
{
	float result = 0;
	float cntr = 0;

	for (int i = 0; i < nrow; i++) {
		if (table[VIDX(i, col, ncol)] == missing) continue;
		cntr += 1.0;
		result += (table[VIDX(i, col, ncol)] - avg) * (table[VIDX(i, col, ncol)] - avg);
	}
	result = result / cntr;
	result = sqrt(result);
	return result;
}

// Calculate statistics of xtable with weights
void calc_xstats(const float *xtable, int npatient, int nvar, float *avg, float *std)
{
	for (int i = 0; i < nvar; i++) {
		avg[i] = calc_col_avg(xtable, i, npatient, nvar, -1.0);
		std[i] = calc_col_std(xtable, i, npatient, nvar, avg[i], -1.0);
		if (std[i] == 0)
			std[i] = 1;
	}
}
void calc_xstats(const float *xtable, int npatient, int nvar,
	double *avg, double *std, double missing)
{
	for (int i = 0; i < nvar; i++) {
		avg[i] = calc_col_avg(xtable, i, npatient, nvar, (float)missing);
		std[i] = calc_col_std(xtable, i, npatient, nvar, (float)avg[i], (float)missing);
		if (std[i] == 0)
			std[i] = 1.0;
	}
}


int order_ftrs(const float *x, int nsamples, int nftrs, const float *weights, int *order) {

	for (int i = 0; i < nftrs; i++)
		order[i] = i;


	float *avg = NULL, *std = NULL;
	if ((avg = (float *)malloc(nftrs * sizeof(float))) == NULL || (std = (float *)malloc(nftrs * sizeof(float))) == NULL) {
		fprintf(stderr, "Stats allocation failed\n");
		clear_mem<float>(avg);
		clear_mem<float>(std);
		return -1;
	}

	calc_xstats(x, nsamples, nftrs, avg, std);

	// sort by weights x std
	for (int i = 0; i < nftrs - 1; i++) {
		for (int j = i + 1; j < nftrs; j++) {
			if (weights[order[i]] * std[order[i]] < weights[order[j]] * std[order[j]]) {
				int temp = order[i];
				order[i] = order[j];
				order[j] = temp;
			}
		}
	}

	free(avg);
	free(std);

	return 0;
}

void find_nbrs(const float *test_x, int ind, int k, const float *learn_x, int nlearn, int nftrs, int *order, const float *ws, int *nbrs, double *dists, int norm, int test2learn) {
#define INF_DIST  (99999999.0)
	// Initialize with INF
	for (int i = 0; i < k; i++) {
		dists[i] = INF_DIST;
		nbrs[i] = -1;
	}

	double target_dist = INF_DIST;
	int target_nbr = 0;

	// Go over learn-table and find nearest neighbours
	for (int i = 0; i < nlearn; i++) {

		if (i == test2learn)
			continue;

		double dist = 0;
		for (int j = 0; j < nftrs; j++) {
			if (ws[order[j]] == 0)
				break;

			double d = static_cast<double>(ws[order[j]]) * static_cast<double>((learn_x[XIDX(i, order[j], nftrs)] - test_x[XIDX(ind, order[j], nftrs)]));
			if (norm == 2)
				dist += d * d;
			else
				dist += fabs(d);

			if (dist >= target_dist)
				break;
		}

		// Replace
		if (dist < target_dist) {
			nbrs[target_nbr] = i;
			target_dist = dists[target_nbr] = dist;

			// Find new target (most distance current neighbor)
			for (int j = 0; j < k; j++) {
				if (dists[j] > target_dist) {
					target_dist = dists[j];
					target_nbr = j;
				}
			}
		}
	}
}

double get_mean_dist(const float *test_x, int ind, const float *learn_x, int nlearn, int nftrs, int nrand, const float *ws, knnMetric knnMetr) {

	double sum = 0;

	for (int i = 0; i < nrand; i++) {
		int irand = (int)(nlearn * rand() / (RAND_MAX + 1.0));

		for (int j = 0; j < nftrs; j++) {
			double d = static_cast<double>(ws[j]) * static_cast<double>((learn_x[XIDX(irand, j, nftrs)] - test_x[XIDX(ind, j, nftrs)]));
			if (knnMetr == KNN_L2)
				sum += d * d;
			else
				sum += fabs(d);
		}
	}

	return sum / nrand;
}

void tcalc_xstats(const float *x, const float *w, int nsamples, int nftrs, double *avg, double *std, double missing = -1)
{

	memset(avg, 0, nftrs * sizeof(double));
	memset(std, 0, nftrs * sizeof(double));

	for (int j = 0; j < nftrs; j++) {
		double sum = 0;
		double norm = 0;
		for (int i = 0; i < nsamples; i++) {
			double val = x[XIDX(j, i, nsamples)];
			double weight = w[i];

			if (val != missing) {
				sum += val * weight;
				norm += weight;
			}
		}

		if (norm != 0) {
			avg[j] = sum / norm;

			sum = 0;
			for (int i = 0; i < nsamples; i++) {
				double val = x[XIDX(j, i, nsamples)];
				double weight = w[i];

				if (val != missing)
					sum += weight * (val - avg[j])*(val - avg[j]);
			}

			std[j] = sqrt(sum / norm);
			if (std[j] == 0)
				std[j] = 1.0;

		}
		else {
			avg[j] = 0.0;
			std[j] = 1.0;
		}
	}

	return;
}

// Correlation to label
double tget_corr(const float *x, const float *y, int nsamples, int ind, double missing = -1) {

	vector<double> vec1(nsamples), vec2(nsamples);

	int n = 0;
	for (int i = 0; i < nsamples; i++) {
		if (x[XIDX(ind, i, nsamples)] != missing) {
			vec1[n] = x[XIDX(ind, i, nsamples)];
			vec2[n++] = y[i];
		}
	}

	vec1.resize(n);
	vec2.resize(n);

	return medial::performance::pearson_corr_without_cleaning(vec1, vec2);
}

double nbrs_score(const int *nbrs,  double *dists, const float *y, int k, double mean_dist, knnAveraging knnAv) {

	double pred = 0;

	if (knnAv == KNN_DIST_MEAN) {
		double sumw = 0;
		for (int i = 0; i < k; i++) {
			if (dists[i] != -1) {
				double w = (dists[i] > mean_dist) ? 0 : (1 - sqrt(dists[i] / mean_dist));
				pred += y[nbrs[i]] * w;
				sumw += w;
			}
		}
		pred /= sumw;
	}
	else if (knnAv == KNN_1_DIST) {
		double sumw = 0;
		for (int i = 0; i < k; i++) {
			if (nbrs[i] != -1) {
				if (dists[i] < 1e-5)
					dists[i] = 1e-5;
				double w = 1 / (dists[i]);
				pred += y[nbrs[i]] * w;
				sumw += w;
			}
		}
		pred /= sumw;
	}

	return pred;
}

int tcalc_stats(float *x, float *y, float *w, int nsamples, int nftrs, double **avg, double **std, double *yavg, double missing = -1)
{
	if (((*avg) = (double *)malloc(nftrs * sizeof(double))) == NULL) {
		fprintf(stderr, "error : cannot allocate averages for %d\n", nftrs);
		return -1;
	}

	if (((*std) = (double *)malloc(nftrs * sizeof(double))) == NULL) {
		fprintf(stderr, "error : cannot allocate stds for %d\n", nftrs);
		free(*avg);
		return -1;
	}

	tcalc_xstats(x, w, nsamples, nftrs, *avg, *std, missing);

	double sum = 0;
	double norm = 0;
	for (int i = 0; i < nsamples; i++) {
		sum += static_cast<double>(y[i]) * static_cast<double>(w[i]);
		norm += w[i];
	}
	(*yavg) = sum / norm;

	return 0;
}

int calc_stats(float *xtable, float *ytable, int npatient, int nvar, double **avg, double **std, double *yavg, double missing = -1)
{
	if (((*avg) = (double *)malloc(nvar * sizeof(double))) == NULL) {
		fprintf(stderr, "error : cannot allocate averages for %d\n", nvar);
		return -1;
	}
	memset(*avg, 0, nvar * sizeof(double));

	if (((*std) = (double *)malloc(nvar * sizeof(double))) == NULL) {
		fprintf(stderr, "error : cannot allocate stds for %d\n", nvar);
		free(*avg);
		return -1;
	}
	memset(*std, 0, nvar * sizeof(double));

	calc_xstats(xtable, npatient, nvar, *avg, *std, missing);

	double ysum = 0;
	for (int i = 0; i < npatient; i++)
		ysum += ytable[i];
	(*yavg) = ysum / npatient;

	return 0;
}

#define NITER 200
#define EITER 0.00001
int nbrs_ls(const float *testx, int ind, const int *nbrs, const double *dists, const float *x, const float *y, int k, int nftrs, double mean_dist, float *localx, float *localy, float *localw,
	float *localb, float *localr, float *pred) {

	// Create
	int nrows = 0;
	for (int i = 0; i < k; i++) {
		if (nbrs[i] != -1)
			nrows++;
	}

	int irow = 0;
	for (int i = 0; i < k; i++) {
		if (nbrs[i] != -1) {
			localw[irow] = (dists[i] > mean_dist) ? 0 : (1 - (float)sqrt(dists[i] / mean_dist));
			localy[irow] = y[nbrs[i]];

			for (int j = 0; j < nftrs; j++)
				localx[XIDX(j, irow, nrows)] = x[XIDX(nbrs[i], j, nftrs)];
			irow++;
		}
	}

	if (nrows <= nftrs) {
		fprintf(stderr, "Couldn't find enough neihbors for nbrs-ls (found %d with %d features)\n", nrows, nftrs);
		return -1;
	}

	// Normalize
	double yavg, *xavg = NULL, *xstd = NULL;
	if (tcalc_stats(localx, localy, localw, nrows, nftrs, &xavg, &xstd, &yavg) == -1)
		return -1;

	tnormalize_data(localx, localy, nrows, nftrs, xavg, yavg);

	// Learn
	for (int i = 0; i < nftrs; i++) {
		double corr = tget_corr(localx, localy, nrows, i);
		localr[i] = (float)sqrt(fabs(corr));
	}

	float err;

	if (learn_lm(localx, localy, localw, nrows, nftrs, NITER, (float)EITER, localr, localw, localb, &err) == -1) {
		clear_mem<double>(xavg);
		clear_mem<double>(xstd);
		return -1;
	}

	// Predict
	*pred = (float)yavg;
	for (int j = 0; j < nftrs; j++)
		*pred += localb[j] * (testx[XIDX(ind, j, nftrs)] - (float)xavg[j]);

	free(xavg);
	free(xstd);

	return 0;
}

// Predict a single instance using KNN
#define NRAND 500

int knn_predict(const float *test_x, int ind, const float *learn_x, const float *learn_y, int nlearn, int nftrs, const float *weights, int *order, int *nbrs, double *dists, int k,
	knnAveraging knnAv, knnMetric knnMetr, float *nbrs_x, float *nbrs_y, float *nbrs_w, float *nbrs_b, float *nbrs_r, float *pred) {


	find_nbrs(test_x, ind, k, learn_x, nlearn, nftrs, order, weights, nbrs, dists, knnMetr, -1);
	double mean = get_mean_dist(test_x, ind, learn_x, nlearn, nftrs, NRAND, weights, knnMetr);

	if (knnAv == KNN_DIST_MEAN || knnAv == KNN_1_DIST) {
		*pred = (float)nbrs_score(nbrs, dists, learn_y, k, mean, knnAv);
		return 0;
	}
	else if (knnAv == KNN_WEIGHTEDLS) {
		return nbrs_ls(test_x, ind, nbrs, dists, learn_x, learn_y, k, nftrs, mean, nbrs_x, nbrs_y, nbrs_w, nbrs_b, nbrs_r, pred);
	}
	else
		return -1;
}

