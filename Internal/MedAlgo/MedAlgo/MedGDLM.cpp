#define _CRT_SECURE_NO_WARNINGS

#include <thread>
#include <MedAlgo/MedAlgo/MedAlgo.h>
#include <MedAlgo/MedAlgo/MedGDLM.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <MedUtils/MedUtils/MedGlobalRNG.h>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

using namespace Eigen;

//===============================================================================
// Linear Model - using Gradient Descent variants - supports Ridge and/or Lasso
//===============================================================================
//..............................................................................
void MedGDLM::init_defaults()
{
	classifier_type = MODEL_GD_LINEAR;
	transpose_for_learn = false;
	transpose_for_predict = false;
	normalize_for_learn = false;   //T
	normalize_y_for_learn = false; //T
	normalize_for_predict = false; //T

	params.max_iter = 500;
	params.stop_at_err = (float)1e-4;
	params.max_times_err_grows = 20;
	params.method = "logistic_sgd";
	params.batch_size = 512;
	params.rate = (float)0.01;
	params.rate_decay = (float)1.0;
	params.momentum = (float)0.95;
	params.last_is_bias = 0;
	params.l_ridge = 0;
	params.l_lasso = 0;
	params.ls_ridge.clear();
	params.ls_lasso.clear();
	params.nthreads = 0;
	params.err_freq = 10;
	params.verbose_learn = true;
	b0 = 0;
	b.clear();
	n_ftrs = 0;
}
//..............................................................................
int MedGDLM::init(void *_in_params)
{
	init_defaults();

	MedGDLMParams *in_params = (MedGDLMParams *)_in_params;

	params.max_iter = in_params->max_iter;
	params.stop_at_err = in_params->stop_at_err;
	params.max_times_err_grows = in_params->max_times_err_grows;
	params.method = in_params->method;
	params.batch_size = in_params->batch_size;
	params.rate = in_params->rate;
	params.rate_decay = in_params->rate_decay;
	params.momentum = in_params->momentum;
	params.last_is_bias = in_params->last_is_bias;
	params.l_ridge = in_params->l_ridge;
	params.l_lasso = in_params->l_lasso;
	params.ls_ridge = in_params->ls_ridge;
	params.ls_lasso = in_params->ls_lasso;
	params.nthreads = in_params->nthreads;
	params.verbose_learn = in_params->verbose_learn;

	return 0;
}
//..............................................................................

void MedGDLM::calc_feature_contribs(MedMat<float> &x, MedMat<float> &contribs) {
	contribs.resize(x.nrows, x.ncols + 1); // last is bo
	contribs.signals = x.signals;
	contribs.signals.push_back("b0");
	for (size_t i = 0; i < x.nrows; ++i) {
		for (size_t j = 0; j < x.ncols; ++j)
			contribs(i, j) = x(i, j)*b[j];
		contribs(i, x.ncols) = b0;
	}
}

//..............................................................................
int MedGDLM::set_params(map<string, string>& mapper) {

	vector<string> fields;

	for (auto entry : mapper) {
		string field = entry.first;
		//! [MedGDLM::init]
		if (field == "max_iter") params.max_iter = stoi(entry.second);
		else if (field == "stop_at_err") params.stop_at_err = stof(entry.second);
		else if (field == "max_times_err_grows") params.max_times_err_grows = stoi(entry.second);
		else if (field == "method") params.method = entry.second;
		else if (field == "batch_size") params.batch_size = stoi(entry.second);
		else if (field == "rate") params.rate = stof(entry.second);
		else if (field == "rate_decay") params.rate_decay = stof(entry.second);
		else if (field == "momentum") params.momentum = stof(entry.second);
		else if (field == "last_is_bias") params.last_is_bias = stoi(entry.second);
		else if (field == "l_ridge") params.l_ridge = stof(entry.second);
		else if (field == "l_lasso") params.l_lasso = stof(entry.second);
		else if (field == "ls_ridge") {
			boost::split(fields, entry.second, boost::is_any_of(","));
			for (string l : fields)
				params.ls_ridge.push_back(stof(l));
		}
		else if (field == "ls_lasso") {
			boost::split(fields, entry.second, boost::is_any_of(","));
			for (string l : fields)
				params.ls_lasso.push_back(stof(l));
		}
		else if (field == "nthreads") params.nthreads = stoi(entry.second);
		else if (field == "err_freq") params.err_freq = stoi(entry.second);
		else if (field == "print") params.print_model = stoi(entry.second);
		else if (field == "normalize") params.normalize = stoi(entry.second);
		else if (field == "verbose_learn") params.verbose_learn = stoi(entry.second) > 0;
		else MLOG("Unknown parameter \'%s\' for GDLM\n", field.c_str());
		//! [MedGDLM::init]
	}

	return 0;

}

//..............................................................................
void MedGDLM::calc_feature_importance(vector<float> &features_importance_scores,
	const string &general_params, const MedFeatures *features)
{
	for (auto coef : b) { features_importance_scores.push_back(abs(coef)); }
	features_importance_scores.push_back(abs(b0));
}

//..............................................................................
MedGDLM::MedGDLM()
{
	init_defaults();
}


//..............................................................................
MedGDLM::MedGDLM(MedGDLMParams& _in_params)
{
	init((void *)&_in_params);
}

//..............................................................................
MedGDLM::MedGDLM(void *_in_params)
{
	init(_in_params);
}


//..............................................................................
int MedGDLM::Learn(float *x, float *y, int nsamples, int nftrs) {

	vector<float> weights(nsamples, 1.0);
	return Learn(x, y, &(weights[0]), nsamples, nftrs);

}

//..............................................................................
int MedGDLM::Learn(float *x, float *y, const float *w, int nsamples, int nftrs) {

	if (w == NULL)
		return (Learn(x, y, nsamples, nftrs));

	// Learn
	n_ftrs = nftrs;
	b.resize(nftrs);

	int rc = 0;
	if (params.method == "full") {
		rc = Learn_full(x, y, w, nsamples, nftrs);
	}
	else if (params.method == "gd") {
		rc = Learn_gd(x, y, w, nsamples, nftrs);
	}
	else if (params.method == "sgd") {
		rc = Learn_sgd(x, y, w, nsamples, nftrs);
	}
	else if (params.method == "logistic_sgd") {
		rc = Learn_logistic_sgd(x, y, w, nsamples, nftrs);
	}
	else if (params.method == "parallel_logistic_sgd") {
		rc = Learn_logistic_sgd_threaded(x, y, w, nsamples, nftrs);
	}
	else
		rc = -1;

	if (rc >= 0 && params.print_model == 1) {
		for (int i = 0; i < b.size(); i++)
			MLOG("## gdlm model i %d b %f\n", i, b[i]);
		MLOG("## gdlm model i %d b0 %f\n", b.size(), b0);

	}

	if (rc < 0)
		MERR("GDLM:: Unsupported method %s\n", params.method.c_str());
	return rc; // no need to get here....
}

//..............................................................................
int MedGDLM::Predict(float *x, float *&preds, int nsamples, int nftrs) const {

	return Predict(x, preds, nsamples, nftrs, 0);
}

//..............................................................................
int MedGDLM::Predict(float *x, float *&preds, int nsamples, int nftrs, int transposed_flag) const {

	if (preds == NULL)
		preds = new float[nsamples];

	memset(preds, 0, nsamples * sizeof(float));
	set_eigen_threads();

	Map<const MatrixXf> bf(&b[0], nftrs, 1);
	Map<MatrixXf> pf(preds, nsamples, 1);
	if (transposed_flag) {
		Map<MatrixXf> xf(x, nftrs, nsamples);
		pf = xf * bf;
	}
	else {
		Map<MatrixXf> xf(x, nftrs, nsamples);
		pf = xf.transpose() * bf;
	}

	pf.array() += b0;

	if (params.method == "logistic_sgd" || params.method == "parallel_logistic_sgd") {

		//		MLOG("Predicting Logistic...\n");
		//		MLOG("(1) pf(0,0) = %g\n",pf(0,0));
		//		pf = -pf;
		//		MLOG("(2) pf(0,0) = %g\n",pf(0,0));
		pf = pf.array().exp();
		//		MLOG("(3) pf(0,0) = %g\n",pf(0,0));
		pf.array() += 1;
		//		MLOG("(4) pf(0,0) = %g\n",pf(0,0));
		pf = pf.array().inverse();
		//		MLOG("(5) pf(0,0) = %g\n",pf(0,0));

	}

	return 0;
}

/*
// keeping some research code here for now.... should be finalized or removed
//..............................................................................
int MedGDLM::predict_with_simple_but_why(float *x, float *&preds, int nsamples, int nftrs, vector<unsigned char> &is_imputed, unsigned char imputed_mask,
										 SimpleButWhyParams &bw_params, vector<SimpleButWhyRes> &bw_res)
{

	if (preds == NULL)
		preds = new float[nsamples];

	memset(preds, 0, nsamples * sizeof(float));
	bw_res.clear();
	bw_res.resize(nsamples);

	bool do_logistic = false;
	if (params.method == "logistic_sgd" || params.method == "parallel_logistic_sgd") do_logistic = true;

	for (int i = 0; i < nsamples; i++) {
		float score_i = 0;
		vector<float> contribs(nftrs);
		float *x_i = &x[i*nftrs];
		for (int j = 0; j < nftrs; j++) {
			contribs[j] = -b[j] * x_i[j]; // - sign since contribs are always positive in the direction of a positive outcome
			score_i -= contribs[j];
		}

		score_i += b0;

		if (do_logistic)
			score_i = 1.0f / (1.0f + exp(score_i));

		preds[i] = score_i;

		MLOG("i=%d score_i=%f\n", i, score_i);
		if (score_i >= bw_params.do_above_score || score_i < bw_params.do_below_score) {
			bw_res[i].get_from_contribs(bw_params, contribs);
		}
	}


	return 0;
}

*/
//..............................................................................
int MedGDLM::denormalize_model(float *f_avg, float *f_std, float label_avg, float label_std)
{
	float new_b0;
	vector<float> new_b(n_ftrs);

	new_b0 = b0 * label_std + label_avg;
	fill(new_b.begin(), new_b.end(), (float)0);
	for (int j = 0; j < n_ftrs; j++) {
		new_b[j] = label_std * b[j] / f_std[j];
		new_b0 -= label_std * f_avg[j] * b[j] / f_std[j];
	}

	b0 = new_b0;
	for (int j = 0; j < n_ftrs; j++)
		b[j] = new_b[j];

	//transpose_for_predict = false;
	//normalize_for_predict = false;
	return 0;
}


//..............................................................................
void MedGDLM::print(FILE *fp, const string& prefix, int level) const {

	if (level == 0)
		fprintf(fp, "%s: MedGDLM ()\n", prefix.c_str());
	else {
		fprintf(fp, "%s : GD_Linear Model : Nftrs = %d\n", prefix.c_str(), n_ftrs);
		fprintf(fp, "%s : GD_Linear Model b0 = %f\n", prefix.c_str(), b0);

		if (!model_features.empty()) {
			for (int i = 0; i < n_ftrs; i++)
				fprintf(fp, "%s : GD_Linear Model b[%d, %s] = %f\n", prefix.c_str(), i, model_features[i].c_str(), b[i]);
		}
		else {
			for (int i = 0; i < n_ftrs; i++)
				fprintf(fp, "%s : GD_Linear Model b[%d] = %f\n", prefix.c_str(), i, b[i]);
		}
	}
}

void MedGDLM::set_eigen_threads() const
{
	int ncores = std::thread::hardware_concurrency();

	if (params.nthreads <= 0)
		Eigen::setNbThreads(3 * ncores / 4);
	else
		Eigen::setNbThreads(params.nthreads);
}

//===============================================================================================
// Actual Math starts here ----> ......
//===============================================================================================
int MedGDLM::Learn_full(float *x, float *y, const float *w, int nsamples, int nftrs)
{
	if (params.l_lasso > 0) {
		MERR("MedGDLM::Learn : ERROR: methos=full can't learn model with lasso : try use the sgd model instead.\n");
		return -1;
	}

	for (float l : params.ls_lasso) {
		if (l > 0) {
			MERR("MedGDLM::Learn : ERROR: methos=full can't learn model with lasso : try use the sgd model instead.\n");
			return -1;
		}
	}

	if (params.last_is_bias != 0) {
		MERR("MedGDLM:: Learn : ERROR: method=full supports only last_is_bias==0 case... \n");
		return -1;
	}

	// Exapnd lambda ridge to a vector
	if (params.l_ridge > 0 && !params.ls_ridge.empty()) {
		MERR("MedGDLM:Either l_ridge or ls_ridge may be given.\n");
		return -1;
	}

	if (params.ls_ridge.empty())
		params.ls_ridge.resize(nftrs, params.l_ridge);
	Map<VectorXf> ridge(params.ls_ridge.data(), nftrs);

	set_eigen_threads();

	float fact_numeric = (float)1000;
	b0 = 0;
	b.resize(nftrs);
	fill(b.begin(), b.end(), (float)0);

	if (params.last_is_bias == 0) {

		VectorXf bf(nftrs + 1);
		Map<MatrixXf> xf(x, nftrs, nsamples);
		Map<VectorXf> yf(y, nsamples);
		MatrixXf xtx(nftrs, nftrs);
		VectorXf preb(nftrs, 1);

		float fact = sqrt(fact_numeric / ((float)nsamples*(float)nftrs));
		//float fact = 1;
		float sq_fact = fact * fact;
		xf *= fact;
		xtx = xf * xf.transpose();
		xtx.diagonal().array() += ridge.array();//*sq_fact;

		xtx.conservativeResize(nftrs + 1, nftrs + 1);
		xtx(nftrs, nftrs) = (float)nsamples*sq_fact;
		for (int i = 0; i < nftrs; i++) {
			xtx(i, nftrs) = fact * xf.row(i).array().sum();
			xtx(nftrs, i) = xtx(i, nftrs);
		}

		preb = xf * yf;
		preb *= fact;
		preb.conservativeResize(nftrs + 1);
		preb(nftrs) = sq_fact * yf.array().sum();
		bf = xtx.colPivHouseholderQr().solve(preb);
		for (int j = 0; j < nftrs; j++) {
			b[j] = bf(j);
		}
		b0 = bf(nftrs);
		return 0;
	}

	return 0;
}

//===========================================================================================================
int MedGDLM::Learn_gd(float *x, float *y, const float *w, int nsamples, int nftrs)
{
	set_eigen_threads();
	// currently completely ignoring w.... , will add soon....

	// initial guess - we start at 0.
	b0 = 0;
	b.resize(nftrs);
	fill(b.begin(), b.end(), (float)0);
	vector<float> prev_b = b;

	// preparing for iterations
	int niter = 0;
	int nerr = 0;
	float err = 10 * (float)nftrs;
	float prev_err = 2 * err;
	Map<MatrixXf> bf(&b[0], nftrs, 1);
	Map<MatrixXf> prev_bf(&prev_b[0], nftrs, 1);
	Map<MatrixXf> xf(x, nftrs, nsamples);
	Map<MatrixXf> yf(y, nsamples, 1);
	MatrixXf pf(nsamples, 1);
	MatrixXf grad(nftrs, 1);
	MatrixXf prev_grad(nftrs, 1);
	MatrixXf diff(nftrs, 1);
	MatrixXf sign(nftrs, 1);
	MatrixXf xtf = xf.transpose();
	float r = params.rate;

	// Exapnd lambdas to vectors
	if (params.ls_ridge.empty())
		params.ls_ridge.resize(nftrs, params.l_ridge / nftrs);
	Map<MatrixXf> ridge(params.ls_ridge.data(), 1, nftrs);

	if (params.ls_lasso.empty())
		params.ls_lasso.resize(nftrs, params.l_lasso / nftrs);
	Map<MatrixXf> lasso(params.ls_lasso.data(), 1, nftrs);


	float fact_grad = (float)1 / ((float)nsamples);
	prev_grad.array().setZero();

	for (int i = 0; i < nftrs; i++) bf(i, 0) = rand_1(); // random initialization

	// iterating
	if (params.verbose_learn)
		MLOG("Learn_gd:: err %f max_err %f , niter %d max_iter %d\n", err, params.stop_at_err, niter, params.max_iter);
	prev_bf = bf;
	while (err > params.stop_at_err && niter < params.max_iter && nerr < params.max_times_err_grows) {

		pf = xf.transpose() * bf - yf;
		//pf.array() += b0;
		//grad = xf*pf;
		grad = pf.transpose() * xtf;
		grad *= fact_grad; // normalizing gradient to be independent of sample size (to "gradient per sample" units)

		// ridge
		grad = grad + ridge.cwiseProduct(bf);

		if (niter > 0)
			grad = (1 - params.momentum)*grad + params.momentum*prev_grad;

		bf = bf - r * grad;

		// lasso
		for (int i = 0; i < nftrs; i++) {
			if (bf(0, i) > lasso(0, i))
				bf(0, i) -= lasso(0, i);
			else if (bf(0, i) < -lasso(0, i))
				bf(0, i) += lasso(0, i);
			if (bf(0, i) > -lasso(0, i) && bf(0, i) < lasso(0, i))
				bf(0, i) = 0;
		}

		diff = bf - prev_bf;
		err = (grad.norm() + diff.norm()) / (float)nftrs;

		if (err > prev_err*(float)3.01 + params.stop_at_err)
			nerr++;

		if (params.verbose_learn)
			MLOG("Learn_gd:: rate %g err %g prev_err %g max_err %g , niter %d max_iter %d\n", r, err, prev_err, params.stop_at_err, niter, params.max_iter);
		if (params.rate_decay < 1)
			r = r * params.rate_decay;

		prev_bf = bf;
		prev_err = err;
		prev_grad = grad;
		niter++;
	}

	return 0;
}

//===============================================================================================
int MedGDLM::Learn_sgd(float *x, float *y, const float *w, int nsamples, int nftrs)
{
	set_eigen_threads();

	// handling weights if needed
	vector<float> norm_wgts; // if weights are used we make sure their sum is nsamples
	if (w != NULL) {
		float sum_w = 0;
		for (int i = 0; i < nsamples; i++) sum_w += w[i];
		for (int i = 0; i < nsamples; i++) norm_wgts.push_back(w[i] * (float)nsamples / (sum_w + (float)1e-5));
	}

	// Exapnd lambdas to vectors
	if (params.ls_ridge.empty())
		params.ls_ridge.resize(nftrs, params.l_ridge / nftrs);
	Map<MatrixXf> ridge(params.ls_ridge.data(), 1, nftrs);

	if (params.ls_lasso.empty())
		params.ls_lasso.resize(nftrs, params.l_lasso / nftrs);
	Map<MatrixXf> lasso(params.ls_lasso.data(), 1, nftrs);

	// handling normalization if asked for
	MedMat<float> normalized_x, normalized_y;
	if (params.normalize) {
		normalized_x.load(x, nsamples, nftrs);
		normalized_x.normalize();
		normalized_y.load(y, nsamples, 1);
		normalized_y.normalize();
		x = normalized_x.data_ptr();
		y = normalized_y.data_ptr();
	}


	// initial guess - we start at 0.
	b0 = 0;
	b.resize(nftrs);
	fill(b.begin(), b.end(), (float)0);

	// preparing for iterations
	int niter = 0;
	float err = params.stop_at_err * 100;
	Map<MatrixXf> bf(&b[0], 1, nftrs);
	MatrixXf grad(1, nftrs);
	MatrixXf prev_grad(1, nftrs);

	Map<MatrixXf> x_all(x, nftrs, nsamples);
	MatrixXf xt = x_all.transpose();
	MatrixXf diff(1, nsamples);
	Map<MatrixXf> y_m(&y[0], 1, nsamples);

	float r = params.rate;
	vector<float> vpf(params.batch_size);

	int n_batches = nsamples / params.batch_size;
	if (n_batches*params.batch_size < nsamples)
		n_batches++;

	// iterating
	if (params.verbose_learn)
		MLOG("Learn_sgd:: err %f max_err %f , niter %d max_iter %d :: batch_size %d :: n_batches %d\n", err, params.stop_at_err, niter, params.max_iter, params.batch_size, n_batches);
	float momentum = params.momentum;
	prev_grad = grad;
	int first_time = 1;
	float bias_grad = 0, prev_bias_grad = 0;
	double prev_loss = 1e8;
	vector<float> preds(nsamples);
	float *ppreds = &preds[0];
	while (err > params.stop_at_err && niter < params.max_iter) {

		for (int bn = 0; bn < n_batches; bn++) {
			size_t from = (size_t)bn * (size_t)params.batch_size;
			int len = params.batch_size; // len is nsamples in batch
			if ((int)from + len > nsamples) len = nsamples - (int)from;
			Map<MatrixXf> xf(&x[from*(size_t)nftrs], nftrs, len);
			Map<MatrixXf> yf(&y[from], 1, len);
			Map<MatrixXf> pf(&vpf[0], 1, len);
			float fact_grad = (float)1 / (float)len;

			pf = bf * xf - yf;
			pf.array() += b0;
			if (w != NULL) for (size_t j = 0; j < len; j++) pf(0, j) = pf(0, j) * norm_wgts[from + j];
			grad = pf * xt.block(from, 0, len, nftrs);
			grad *= fact_grad; // normalizing gradient to be independent of sample size (to "gradient per sample" units)
			bias_grad = pf.array().sum();
			bias_grad *= fact_grad;

			// ridge
			grad = grad + ridge.cwiseProduct(bf);

			// momentum
			if (first_time) { prev_grad = grad; prev_bias_grad = bias_grad; first_time = 0; }
			grad *= (1 - momentum);
			grad = grad + momentum * prev_grad;
			bias_grad *= (1 - momentum);
			bias_grad = bias_grad + momentum * prev_bias_grad;

			// step
			bf = bf - r * grad;
			b0 = b0 - r * bias_grad;

			// lasso reguralizer step !! (using Tibrishani proximal gradient descent method)
			for (int i = 0; i < nftrs; i++) {
				float bound = r * lasso(0, i);
				if (bf(0, i) > bound) bf(0, i) -= bound;
				else if (bf(0, i) < -bound) bf(0, i) += bound;
				else bf(0, i) = 0;
			}

			prev_grad = grad;
			prev_bias_grad = bias_grad;
		}

		if (niter % params.err_freq == 0) {
			// after err_freq epochs of going through the data we check the stop criteria
			// for that we calculate the loss func and wait until changes are small

			// Evaluating so that we can get loss err and accuracy
			Predict(x, ppreds, nsamples, nftrs);

			Map<MatrixXf> p_m(&ppreds[0], 1, nsamples);

			int nacc = 0;
			double loss = 0;

			diff = y_m - p_m;
			if (w != NULL) for (int j = 0; j < nsamples; j++) diff(0, j) = diff(0, j) * norm_wgts[j];

			loss = diff.array().square().sum();

			for (int i = 0; i < nsamples; i++) {
				if ((preds[i] >= 0.5 && y[i] == 1) || (preds[i] < 0.5 && y[i] == 0)) nacc++;
			}

			loss /= (double)nsamples;

			// ridge
			loss += (ridge.array() * bf.array().square()).sum();

			// lasso
			loss += (lasso.array() * bf.array().abs()).sum();

			err = (float)(abs(prev_loss - loss) / (abs(prev_loss) + 1e-10));
			prev_loss = loss;

			// printing
			int n0bs = 0, n_small_bs = 0;

			for (int i = 0; i < nftrs; i++) {
				if (bf.array()(0, i) == 0) n0bs++;
				if (bf.array().abs()(0, i) < (float)1e-5) n_small_bs++;
			}
			if (params.verbose_learn)
				MLOG("Learn_linear_sgd:: rate %g err %g stop_err %g acc %g loss %g , niter %d max_iter %d n0 %d (%d) ns %d\n",
					r, err, params.stop_at_err, (double)nacc / (double)nsamples, loss, niter, params.max_iter, n0bs, nftrs - n0bs, n_small_bs);
		}

		// update rate with rate decay
		if (params.rate_decay < 1)
			r = r * params.rate_decay;

		niter++;
	}

	if (params.normalize) {
		denormalize_model(&normalized_x.avg[0], &normalized_x.std[0], normalized_y.avg[0], normalized_y.std[0]);
	}

	return 0;
}


//===============================================================================================
int MedGDLM::Learn_logistic_sgd(float *x, float *y, const float *w, int nsamples, int nftrs)
{
	if (params.last_is_bias) {
		MERR("logistic_sgd() : ERROR: last is bias not 0!!! This mode is NOT supported yet\n");
		MERR("logistic_sgd() : we ignore this and calc a b0 normally\n");
	}

	set_eigen_threads();
	// currently completely ignoring w.... , will add soon....
	vector<float> norm_wgts(nsamples, (float)1); // if weights are used we make sure their sum is nsamples
	if (w != NULL) {
		float sum_w = 0;
		for (int i = 0; i < nsamples; i++) sum_w += w[i];
		for (int i = 0; i < nsamples; i++) norm_wgts[i] = w[i] * (float)nsamples / (sum_w + (float)1e-5);
	}

	// check we are in a binary 0/1 problem
	for (int i = 0; i < nsamples; i++) {
		if (y[i] != 0 && y[i] != 1)
			MLOG("ERROR: i=%d y %f - only 0/1 values allowed\n", i, y[i]);
	}

	MedMat<float> normalized_x;
	if (params.normalize) {
		normalized_x.load(x, nsamples, nftrs);
		normalized_x.normalize();
		x = normalized_x.data_ptr();
	}

	// Exapnd lambdas to vectors
	if (params.ls_ridge.empty())
		params.ls_ridge.resize(nftrs, params.l_ridge);
	Map<MatrixXf> ridge(params.ls_ridge.data(), 1, nftrs);

	if (params.ls_lasso.empty())
		params.ls_lasso.resize(nftrs, params.l_lasso);
	Map<MatrixXf> lasso(params.ls_lasso.data(), 1, nftrs);

	// initial guess - we start at 0. (should we try random?)
	b0 = 0;
	b.resize(nftrs);
	fill(b.begin(), b.end(), (float)0);
	vector<float> prev_b = b;
	float prev_b0 = b0;


	// preparing for iterations
	int niter = 0;
	double err = params.stop_at_err * 100; // inital err
	Map<MatrixXf> bf(&b[0], 1, nftrs);
	Map<MatrixXf> prev_bf(&prev_b[0], 1, nftrs);
	MatrixXf grad(1, nftrs);

	MatrixXf dx(1, nftrs);
	MatrixXf prev_grad(1, nftrs);
	MatrixXf diff(1, nftrs);

	Map<MatrixXf> x_all(x, nftrs, nsamples);
	MatrixXf xt = x_all.transpose();

	float r = params.rate;
	vector<float> vpf(params.batch_size);

	int n_batches = nsamples / params.batch_size;
	if (n_batches*params.batch_size < nsamples)
		n_batches++;

	// iterating
	if (params.verbose_learn)
		MLOG("Learn_logistic_sgd:: err %f max_err %f , niter %d max_iter %d :: batch_size %d :: n_batches %d\n", err, params.stop_at_err, niter, params.max_iter, params.batch_size, n_batches);
	float momentum = params.momentum;
	grad = MatrixXf::Zero(1, nftrs);
	prev_grad = grad;
	float bias_grad = 0, prev_bias_grad = 0;

	prev_bf = bf;

	float dnorm = 0;
	double prev_loss = 1e8;
	////vector<float> xx, yy;
	vector<float> preds(nsamples);
	float *ppreds = &preds[0];

	while (err > params.stop_at_err && niter < params.max_iter) {
		for (int bn = 0; bn < n_batches; bn++) {

			size_t from = static_cast<size_t>(bn) * static_cast<size_t>(params.batch_size);
			int len = params.batch_size; // len is nsamples in batch
			if ((int)from + len > nsamples) len = nsamples - (int)from;

			// next patch is for randomizing the batch each round... currently "code in sleep"
			//xx.resize(params.batch_size*nftrs);
			//yy.resize(params.batch_size);
			//for (int i=0; i<params.batch_size; i++) {
			//	int ir = rand_N(nsamples-1);
			//	memcpy(&xx[i*nftrs],&x[ir*nftrs],nftrs*sizeof(float));
			//	yy[i] = y[ir];
			//}
			//len = params.batch_size;
			//			Map<MatrixXf> xf(&xx[0],nftrs,len);
			//			Map<MatrixXf> yf(&yy[0],1,len);

			Map<MatrixXf> xf(&x[from*(size_t)nftrs], nftrs, len);
			Map<MatrixXf> yf(&y[from], 1, len);
			Map<MatrixXf> pf(&vpf[0], 1, len);

			float fact_grad = (float)1.0 / (float)len;

			// for each sample we calc pf(i) = - 1 / (1 + exp (B*Xi + b0))
			pf = bf * xf;
			pf.array() += b0;
			pf = pf.array().exp();
			pf.array() += 1;
			pf = pf.array().inverse();
			pf = -pf;

			// we add +1 to pf(i), since y is 0/1 this trick transforms the y=0 gradient to the y=1 gradient that is
			// 1 / (1 + exp (-B*Xi-b0))
			pf = pf + yf;
			if (w != NULL) for (int j = 0; j < len; j++) pf(0, j) = pf(0, j) * norm_wgts[from + j];

			// summing current pf for the gradient of b0
			float bias_g = pf.array().sum();
			// multiplying by X to get the gradients for the Bj elements
			grad = pf * xt.block(from, 0, len, nftrs);

			// normalizing gradients to be independent of sample size (to "gradient per sample" units)
			// this allows for easier learning rate configuration
			grad.array() *= fact_grad;
			bias_grad = bias_g * fact_grad;

			if (grad.norm() > 1e10) { MLOG("grad norm too big\n"); }

			// ridge reguralizer
			grad = grad + ridge.cwiseProduct(bf);

			// initialize prev_grad in first iter
			if (niter == 0) {
				prev_grad = grad; prev_bias_grad = bias_grad;
			}

			// momentum
			grad *= (1 - momentum);
			grad = grad + momentum * prev_grad;

			bias_grad *= (1 - momentum);
			bias_grad = bias_grad + momentum * prev_bias_grad;

			// step
			bf = bf - r * grad;
			b0 = b0 - r * bias_grad;

			// lasso reguralizer step !! (using Tibrishani proximal gradient descent method)
			for (int i = 0; i < nftrs; i++) {
				float bound = r * lasso(0, i);
				if (bf(0, i) > bound) bf(0, i) -= bound;
				else if (bf(0, i) < -bound) bf(0, i) += bound;
				else bf(0, i) = 0;
			}

			// update prev
			prev_grad = grad;
			prev_bias_grad = bias_grad;
		}

		if (niter % params.err_freq == 0) {
			// after err_freq epochs of going through the data we check the stop criteria
			// for that we calculate the loss func and wait until changes are small

			// Evaluating so that we can get loss err and accuracy
			Predict(x, ppreds, nsamples, nftrs);

			int nacc = 0;
			double loss = 0;
			for (int i = 0; i < nsamples; i++) {
				if ((preds[i] >= 0.5 && y[i] == 1) || (preds[i] < 0.5 && y[i] == 0)) nacc++;

				if (y[i] == 1)
					loss += -norm_wgts[i] * log(max(1e-5, (double)preds[i]));
				else
					loss += -norm_wgts[i] * log(max(1e-5, 1.0 - (double)preds[i]));
			}

			loss /= (double)nsamples;

			// ridge
			loss += (ridge.array() * bf.array().square()).sum();

			// lasso
			loss += (lasso.array() * bf.array().abs()).sum();

			diff = bf - prev_bf;

			dnorm = sqrt(diff.array().square().sum() + (b0 - prev_b0)*(b0 - prev_b0)) / (float)nftrs;
			err = (float)abs(prev_loss - loss) / (abs(prev_loss) + 1e-10);
			prev_loss = loss;

			// printing
			int n0bs = 0, n_small_bs = 0;

			for (int i = 0; i < nftrs; i++) {
				if (bf.array()(0, i) == 0) n0bs++;
				if (bf.array().abs()(0, i) < (float)1e-5) n_small_bs++;
			}
			if (params.verbose_learn)
				MLOG("Learn_logistic_sgd:: rate %g err %g dnorm %g stop_err %g acc %g loss %g , niter %d max_iter %d n0 %d (%d) ns %d\n",
					r, err, dnorm, params.stop_at_err, (double)nacc / (double)nsamples, loss, niter, params.max_iter, n0bs, nftrs - n0bs, n_small_bs);
		}

		// update rate with rate decay
		if (params.rate_decay < 1)
			r = r * params.rate_decay;

		// update prev_bf , prev_b0
		prev_bf = bf;
		prev_b0 = b0;
		niter++;

	}

	if (params.normalize) {
		denormalize_model(&normalized_x.avg[0], &normalized_x.std[0], 0, 1);
	}
	if (params.verbose_learn)
		MLOG("Learn_logistic_sgd:: rate %g err %g dnorm %g max_err %g , niter %d max_iter %d\n",
			r, err, dnorm, params.stop_at_err, niter, params.max_iter);

	return 0;
}


//===============================================================================================
// general idea
// we split x into minibatches, 
// (1) repeat NMAX times:
// (2)   choose random N minibatches
// (3)   parallel(i) : calculate new w_i for each minibatch using gd and momentum
// (4)   average w_i to get new w

// we then choose a random set of N minibatches

int MedGDLM::Learn_logistic_sgd_threaded(float *x, float *y, const float *w, int nsamples, int nftrs)
{

	int print_every_iter = 1;
	set_eigen_threads();
	// currently completely ignoring w.... , will add soon....

	// initial guess - we start at 0.
	b0 = 0;
	b.resize(nftrs);
	fill(b.begin(), b.end(), (float)0);
	vector<float> prev_b = b;

	// preparing for iterations
	int nerr = 0;
	float err = params.stop_at_err * 100;
	float prev_err = 2 * err;
	Map<MatrixXf> bf(&b[0], 1, nftrs);
	Map<MatrixXf> prev_bf(&prev_b[0], 1, nftrs);
	MatrixXf grad(1, nftrs);
	MatrixXf dx(1, nftrs);
	MatrixXf pgrad(1, nftrs);
	MatrixXf diff(1, nftrs);

	Map<MatrixXf> x_all(x, nftrs, nsamples);
	MatrixXf xt = x_all.transpose();

	// Exapnd lambdas to vectors
	if (params.ls_ridge.empty())
		params.ls_ridge.resize(nftrs, params.l_ridge / nftrs);
	Map<MatrixXf> ridge(params.ls_ridge.data(), 1, nftrs);

	// setting the framework
	int n_threads = params.nthreads;
	int batch_size = params.batch_size;
	int n_batches_per_thread = 1;
	int n_batches = nsamples / batch_size;
	if (nsamples % batch_size != 0) n_batches++;
	int n_summed = n_threads * n_batches_per_thread;
	int n_rounds_per_epoch = nsamples / (n_summed * batch_size);
	if ((nsamples % (n_summed*batch_size)) != 0) n_rounds_per_epoch++;
	float momentum = params.momentum;
	float r = params.rate;

	if (params.verbose_learn)
		MLOG("parallel logistic sgd :: x %d x %d , n_th %d batch_size %d n_batches_per_thread %d n_batches %d n_summed %d n_rounds_per_epoch %d\n",
			nsamples, nftrs, n_threads, batch_size, n_batches_per_thread, n_batches, n_summed, n_rounds_per_epoch);
	if (params.verbose_learn)
		MLOG("parallel logistic sgd :: rate = %g , momentum = %g , min_err = %g , ridge = %g\n", r, momentum, params.stop_at_err, ridge.array().mean());

	vector<float> vpf(static_cast<size_t>(batch_size) * static_cast<size_t>(n_threads));
	int niter = 0;
	int go_on = 1;
	vector<vector<int>> i_b(n_threads);
	MatrixXf bfs(n_threads, nftrs); // a matrix to hold the threaded results, we will average it at the end of each round
	MatrixXf grads(n_threads, nftrs);
	MatrixXf normalizer(1, n_threads);
	MatrixXf prev_grad(n_threads, nftrs);

	for (int i = 0; i < n_threads; i++) normalizer(0, i) = (float)1 / (float)n_threads;

	float gnorm = 0, dnorm = 0, bnorm = 0, pgnorm = 0, mxg = 0;
	int k = 0;

	while (go_on && niter < params.max_iter) {

		for (int rn = 0; rn < n_rounds_per_epoch; rn++) {

			// first - preparing vector of random batches for this round
			for (int th = 0; th < n_threads; th++) {
				i_b[th].resize(n_batches_per_thread);
				for (int j = 0; j < n_batches_per_thread; j++) {
					i_b[th][j] = k % n_batches; k++; //rand_N(n_batches);
//					i_b[th][j] = rand_N(n_batches);
				}
			}

			// now parallelizing the actual work
#pragma omp parallel for
			for (int th = 0; th < n_threads; th++) {

				if (1) { //niter == 0) {
					prev_grad.row(th) = grad.row(0);
					//bfs.row(th) = bf.row(0);
				}
				grads.row(th) = grad.row(0);
				for (int j = 0; j < n_batches_per_thread; j++) {
					size_t from = static_cast<size_t>(i_b[th][j]) * static_cast<size_t>(batch_size);
					int to = min((int)from + batch_size, nsamples);
					int len = to - (int)from;
					Map<MatrixXf> xf(&x[from*(size_t)nftrs], nftrs, len);
					Map<MatrixXf> yf(&y[from], 1, len);
					Map<MatrixXf> pf(&vpf[th*batch_size], 1, len);
					float fact_grad = (float)1.0 / (float)len;

					//pf = bfs.row(th) * xf;
					pf = bf * xf;
					pf = -pf;
					pf = pf.array().exp();
					pf.array() += 1;
					pf = pf.array().inverse();
					pf = pf - yf;
					grads.row(th) = pf * xt.block(from, 0, len, nftrs);
					grads.row(th) *= fact_grad; // normalizing gradient to be independent of sample size (to "gradient per sample" units)

					if (grads.row(th).norm() > 1e10) { MLOG("grad norm too big\n"); }
					// ridge
					if (params.l_ridge > 0) {
						float bias = grads(th, nftrs - 1);
						grads.row(th) = grads.row(th) + ridge.cwiseProduct(bfs.row(th));
						if (params.last_is_bias)
							grads(th, nftrs - 1) = bias;
					}

					if (niter == 0 && j == 0) prev_grad.row(th) = grad.row(0);

					// momentum
					grads.row(th) *= (1 - momentum);
					grads.row(th) = grads.row(th) + momentum * prev_grad.row(th);

					// step
					//bfs.row(th) = bfs.row(th) - r*grads.row(th);

					prev_grad.row(0) = grads.row(th);
				}

			}

			// need to average grads and bfs to get our current situation
			//bf = normalizer*bfs;
			grad = normalizer * grads;
			//step
			bf = bf - r * grad;
		}

		// finished an epoch - getting error and deciding if to stop
		diff = bf - prev_bf;
		pgrad = grad - pgrad;
		dnorm = diff.squaredNorm() / (float)nftrs;
		pgnorm = pgrad.squaredNorm() / (float)nftrs;
		gnorm = grad.squaredNorm() / (float)nftrs;
		bnorm = bf.squaredNorm() / (float)nftrs;
		err = dnorm + pgnorm + (float)0.1*gnorm;
		mxg = max(abs(grad.maxCoeff()), abs(grad.minCoeff()));

		if (err > prev_err*(float)10.5 + params.stop_at_err)
			nerr++;

		float derr = prev_err - err;
		if (derr > 0 && derr < (float)1e-12)
			r = r * (float)0.99;

		if (print_every_iter && params.verbose_learn)
			MLOG("Learn_logistic_sgd (parallel):: rate %g err %g derr %g gnorm %g dnorm %g bnorm %g pgnorm %g mxg %g max_err %g , niter %d max_iter %d\n",
				r, err, derr, gnorm, dnorm, bnorm, pgnorm, mxg, params.stop_at_err, niter, params.max_iter);
		if (params.rate_decay < 1)
			r = r * params.rate_decay;

		prev_bf = bf;
		prev_err = err;
		pgrad = grad;
		niter++;
		if (err < params.stop_at_err) go_on = 0;
	}

	if (params.verbose_learn)
		MLOG("Learn_logistic_sgd (parallel):: rate %g err %g gnorm %g dnorm %g bnorm %g pgnorm %g mxg %g max_err %g , niter %d max_iter %d\n",
			r, err, gnorm, dnorm, bnorm, pgnorm, mxg, params.stop_at_err, niter, params.max_iter);

	return 0;
}