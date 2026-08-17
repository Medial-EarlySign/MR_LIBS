#define _CRT_SECURE_NO_WARNINGS

#include <MedAlgo/MedAlgo/MedAlgo.h>
#include <MedAlgo/MedAlgo/MedLM.h>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

//============================================================================
// LASSO Model
//============================================================================
//..............................................................................

void init_default_lasso_params(MedLassoParams& _params) {
	_params.lambda = LASSO_LAMBDA;
	_params.num_iterations = LASSO_NITER;
}

int MedLasso::set_params(map<string, string>& mapper) {

	for (auto entry : mapper) {
		string field = entry.first;
		//! [MedLasso::init]
		if (field == "lambda") params.lambda = stod(entry.second);
		else if (field == "num_iterations") params.num_iterations = stoi(entry.second);
		else MLOG("Unknonw parameter \'%s\' for Lasso\n", field.c_str());
		//! [MedLasso::init]
	}

	return 0;
}

void MedLasso::init_defaults()
{
	classifier_type = MODEL_LASSO;
	transpose_for_learn = false;
	transpose_for_predict = false;
	normalize_for_learn = false;
	normalize_y_for_learn = false;
	normalize_for_predict = false;
	init_default_lasso_params(params);
}


void MedLasso::Initb() {
	b = vector<float>();
	b0 = 0;
}

//..............................................................................
int MedLasso::init(void *_in_params)
{
	init_defaults();
	MedLassoParams *in_params = (MedLassoParams *)_in_params;
	params.lambda = in_params->lambda;
	params.num_iterations = in_params->num_iterations;
	Initb();
	return 0;
}
//..............................................................................
MedLasso::MedLasso()
{
	init_defaults();
	params.lambda = LASSO_LAMBDA;
	params.num_iterations = LASSO_NITER;
	Initb();
}


//..............................................................................
MedLasso::MedLasso(MedLassoParams& _in_params)
{
	init((void *)&_in_params);
}

//..............................................................................
MedLasso::MedLasso(void *_in_params)
{
	init(_in_params);
}


//..............................................................................
int MedLasso::Learn(float *x, float *y, int nsamples, int nftrs) {
	vector<float> weights(nsamples, 1.0);
	return Learn(x, y, &(weights[0]), nsamples, nftrs);
}

double perform_lasso_iteration(double* xk_train, vector<double>& r, float bk, int nrow_train, double lambda) {
	double bk_hat = 0;
	for (int i = 0; i < nrow_train; i++)
		bk_hat += r[i] * xk_train[i];

	bk_hat /= nrow_train;

	bk_hat += bk;
	if (bk_hat - lambda > 0)
		bk_hat -= lambda;
	else if (bk_hat + lambda < 0)
		bk_hat += lambda;
	else
		bk_hat = 0;

	if (bk_hat != bk) {
		for (int i = 0; i < nrow_train; i++)
			r[i] += xk_train[i] * (bk - bk_hat);
	}
	return bk_hat;
}

void MedLasso::lasso_regression(vector<float>& b, int nrow_train, int n_ftrs, double lambda, int num_iterations) {
	vector<double> r(nrow_train);
	for (int i = 0; i < nrow_train; i++)
		r[i] = y1[i];

	for (int it = 0; it < num_iterations; it++)
		for (int k = 0; k < n_ftrs; k++)
			b[k] = (float)perform_lasso_iteration(trainx[k], r, b[k], nrow_train, lambda);
}


void MedLasso::initialize_vars(float *x_in, float *y_in, const float *w, vector<float>& b, int nrow_train, int n_ftrs) {
	trainx = (double **)malloc(n_ftrs * sizeof(double*));
	for (int j = 0; j < n_ftrs; j++)
		trainx[j] = (double *)malloc(nrow_train * sizeof(double));
	y1 = (double *)malloc(nrow_train * sizeof(double));
	
	b.resize(n_ftrs);
	for (int i = 0; i < n_ftrs; i++) b[i] = 0;

	for (int i = 0; i < nrow_train; i++)
		for (int j = 0; j < n_ftrs; j++)
			trainx[j][i] = static_cast<double>(sqrt(w[i])) * static_cast<double>(x_in[static_cast<size_t>(i) * static_cast<size_t>(n_ftrs) + static_cast<size_t>(j)]);
	for (int i = 0; i < nrow_train; i++)
		y1[i] = static_cast<double>(sqrt(w[i])) * static_cast<double>(y_in[i]);
}

//void initialize_test(double **&testx, float *test_in, int nrow_test, int n_ftrs) {
//	testx = (double**)malloc(n_ftrs * sizeof(double *));
//	for()
//	for (int i = 0; i < nrow_test; i++) {
//		testx[j] = (double *)malloc(nrow_test * sizeof(double));
//		for (int j = 0; j < n_ftrs; j++)
//			testx[i][j] = test_in[i * n_ftrs + j];
//	}
//}


//..............................................................................
int MedLasso::Learn(float *x, float *y, const float *w, int nsamples, int nftrs) {
	MedLasso::n_ftrs = nftrs;

	if (w == NULL)
		return (Learn(x, y, nsamples, nftrs));

	// Normalization
	vector<float> x_avg(nftrs), x_std(nftrs);
	float y_avg, y_std;
	normalize_x_and_y(x, y, w, nsamples, nftrs, x_avg, x_std, y_avg, y_std);

	initialize_vars(x, y, w, b, nsamples, n_ftrs);
	lasso_regression(b, nsamples, n_ftrs, params.lambda, params.num_iterations);
	denormalize_model(&(x_avg[0]), &(x_std[0]), y_avg, y_std);

	return 0;
}

//..............................................................................
int MedLasso::Predict(float *x, float *&preds, int nsamples, int nftrs) const {
	return Predict(x, preds, nsamples, nftrs, 0);
}

//..............................................................................
int MedLasso::Predict(float *x, float *&preds, int nsamples, int nftrs, int transposed_flag) const {

	if (preds == NULL)
		preds = new float[nsamples];

	memset(preds, 0, nsamples*sizeof(float));
	/*double **testx;
	initialize_test(testx, x, nsamples, nftrs);*/

	if (transposed_flag) {
		for (int j = 0; j < nftrs; j++) {
			for (int i = 0; i < nsamples; i++)
				preds[i] += b[j] * x[j * nsamples + i];
		}
	}
	else {
		for (int i = 0; i < nsamples; i++) {
			for (int j = 0; j < nftrs; j++)
				preds[i] += b[j] * x[i * nftrs + j];
		}
	}

	for (int i = 0; i < nsamples; i++)
		preds[i] += b0;

	return 0;
}

//..............................................................................
void MedLasso::normalize_x_and_y(float *x, float *y, const float *w, int nsamples, int nftrs, vector<float>& x_avg, vector<float>& x_std, float& y_avg, float& y_std) {

	// Get moments
	int n_clean;
	vector<float> v(nsamples);
	for (int i = 0; i < nftrs; i++) {
		for (int j = 0; j < nsamples; j++)
			v[j] = x[nftrs*j + i];
		medial::stats::get_mean_and_std(v, (float)-1.0, n_clean, x_avg[i], x_std[i]);
	}
	medial::stats::get_mean_and_std(y, w, nsamples, -1.0, y_avg, y_std, n_clean, false);

	// Normalize
	for (int j = 0; j < nsamples; j++) {
		float *xj = x + j*nftrs;
		for (int i = 0; i < nftrs; i++)
			xj[i] = (xj[i] - x_avg[i]) / x_std[i];
	}

	for (int j = 0; j < nsamples; j++)
		y[j] = (y[j] - y_avg) / y_std;

}
//..............................................................................
int MedLasso::denormalize_model(float *f_avg, float *f_std, float label_avg, float label_std)
{
	float new_b0;
	vector<float> new_b(n_ftrs);

	new_b0 = b0*label_std + label_avg;
	for (int j = 0; j < n_ftrs; j++) {
		new_b[j] = label_std*b[j] / f_std[j];
		new_b0 -= label_std*f_avg[j] * b[j] / f_std[j];
	}

	b0 = new_b0;
	for (int j = 0; j < n_ftrs; j++) 
		b[j] = new_b[j];

	transpose_for_predict = false;
	normalize_for_predict = false;
	return 0;
}

//..............................................................................
void MedLasso::print(FILE *fp, const string& prefix, int level) const {

	if (level == 0)
		fprintf(fp, "%s: MedLasso ()\n", prefix.c_str());
	else {
		fprintf(fp, "%s : Lasso Model : Nftrs = %d\n", prefix.c_str(), n_ftrs);
		fprintf(fp, "%s : Lasso Model b0 = %f\n", prefix.c_str(), b0);

		for (int i = 0; i < n_ftrs; i++)
			fprintf(fp, "%s : Lasso Model b[%d] = %f\n", prefix.c_str(), i, b[i]);
	}
}