//
// MedBooster.cpp
//
// A general wrapper allowing several boosting strategies for predictors.
//

#include "MedBooster.h"

#include "Logger/Logger/Logger.h"
#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace boost;

//======================================================================================================================
// The format for params:
// boost#booster_type=...,loss_func=...,nrounds=...,shrinkage=....,internal=...#internal_params#....
// example:
// boost@booster_type=gradient,loss_func=square,nrounds=40,shrinkage=0.05,internal_model=linear_model@internal_params@rfactor=0.8
//
void MedBoosterParams::init_from_string(const string &init_str)
{
	vector<string> fields;
	//! [MedBoosterParams::init_from_string]
	boost::split(fields, init_str, boost::is_any_of("@"));

	for (int i=0; i<fields.size(); i++) {
		if (fields[i] == "boost") {
			vector<string> f;
			string curr = fields[++i];
			boost::split(f, curr, boost::is_any_of(",;="));
			for (int j=0; j<f.size(); j++) {
				//MLOG("j %d %s\n", j, f[j].c_str());
				if (f[j]=="booster_type")	{ booster_type = f[++j]; }
				if (f[j]=="loss_func")		{ loss_func = f[++j]; }
				if (f[j]=="nrounds")		{ nrounds = stoi(f[++j]); }
				if (f[j]=="shrinkage")		{ shrinkage = stof(f[++j]); }
				if (f[j]=="internal_model") { internal_booster_model = f[++j]; }
				if (f[j]=="print")			{ print_flag = 1; }
			}
		}

		if (fields[i] == "internal_params") {
			internal_booster_params = fields[++i];
		}
		//! [MedBoosterParams::init_from_string]
	}

}

//======================================================================================================================
int MedBooster::learn(MedMat<float> &x, MedMat<float> &y)
{
	if (params.booster_type == "gradient" && params.loss_func == "square") return learn_gradient_booster(x, y);

	MERR("MedBooster: booster_type=%s loss_fun=%s is not supported (yet)\n", params.booster_type.c_str(), params.loss_func.c_str());
	return -1;
}

//======================================================================================================================
int MedBooster::learn_gradient_booster(MedMat<float> &x, MedMat<float> &y)
{
	int alpha_dim = 2;		// for most cases linear multiplier + bias (softmax might have more)

	// init alpha
	alpha.resize(params.nrounds, alpha_dim);
	predictors.clear();

	// init residual label
	MedMat<float> residual = y;

	// init accumulated predictions (and history tracker, for debugging)
	MedMat<float> all_preds;
	vector<float> acc_preds;
	all_preds.resize(x.nrows, params.nrounds);

	// looping nrounds
	for (int round=0; round<params.nrounds; round++) {

		// create a new model and train it on current residual
		MedPredictor *model = train_internal_model(x, residual, params.internal_booster_model, params.internal_booster_params);
		if (model == NULL) {
			MERR("MedBooster::learn_gradient_booster: FAILED learning model in round %d\n", round);
			return -1;
		}

		// get predictions based on the model
		vector<float> preds;
		if (model->predict(x, preds) < 0) {
			MERR("MedBooster::learn_gradient_booster: FAILED predicting model in round %d\n", round);
			return -1;
		}
		for (int i=0; i<preds.size(); i++) all_preds(i, round) = preds[i];

		// get alphas for this round
		vector<float> a(alpha_dim);
		if (gradient_get_round_alphas(residual, preds, a) < 0) {
			MERR("MedBooster::learn_gradient_booster: FAILED getting alphas in round %d\n", round);
			return -1;
		}

		// update residual
		update_round_residuals(y, acc_preds, residual, preds, a);

		// keep preds, alphas, model
		predictors.push_back(model);
		for (int i=0; i<alpha_dim; i++) alpha(round, i) = a[i];
		for (int i=0; i<preds.size(); i++) all_preds(i, round) = preds[i];

		if (params.print_flag) {
			MLOG("MedBooster: gradient: round %d : x %d x %d : preds %d : alphas:", round, x.nrows, x.ncols, preds.size());
			for (int i=0; i<alpha_dim; i++) MLOG(" %f", alpha(round, i));
			double mse = 0;
			for (int i=0; i<preds.size(); i++) mse += static_cast<double>(residual(i, 0) - preds[i]) * static_cast<double>(residual(i, 0) - preds[i]);
			mse = mse/preds.size();
			MLOG(" square_err %f", mse);
			MLOG("\n");
		}

	}

	return 0;
}

//======================================================================================================================
MedPredictor *MedBooster::train_internal_model(MedMat<float> &x, MedMat<float> &y, string model_name, string model_params)
{
	MedPredictor *model;

	// create new model and initialize
	MLOG("MedBooster: init: %s :: with: %s\n", model_name.c_str(), model_params.c_str());
	model = MedPredictor::make_predictor(model_name);
	if (model == NULL) {
		MERR("MedBooster::train_internal_model: Cannot initialize a model of type \'%s\'\n", model_name.c_str());
		return NULL;
	}

	model->init_from_string(model_params);

	// actual training
	int rc = model->learn(x, y);

	if (rc < 0) {
		MERR("MedBooster::train_internal_model: failed training %s with params %s\n", model_name.c_str(), model_params.c_str());
		delete model;
		return NULL;
	}

	return model;
}

//======================================================================================================================
int MedBooster::gradient_get_round_alphas(MedMat<float> &residual, vector<float> &preds, vector<float> &a)
{
	if (params.loss_func == "square") return gradient_get_alphas_square_loss(residual, preds, a);
	MERR("MedBooster::gradient_get_round_alphas: loss_func %s is not supported in gradient boosting\n", params.loss_func.c_str());
	return -1;
}

//======================================================================================================================
int MedBooster::gradient_get_alphas_square_loss(MedMat<float> &residual, vector<float> &preds, vector<float> &a)
{
	// we are searching for a[0] and a[1] such that:
	// Sum (residual - (a[0]*preds[i] + a[1]))^2 is minimal
	// The solution is of course the solution for the single parameter regression problem of residual ~ shrinkage*preds[i]
	// we then use our shrinkage and move only with a shrinkage factor towards the minimum.

	double mx = 0, my = 0;
	int n = residual.nrows;

	for (int i=0; i<n; i++) {
		mx += preds[i];
		my += residual(i, 0);
	}

	mx = mx/n;
	my = my/n;

	double cov = 0, var = 0;

	for (int i=0; i<n; i++) {
		double dx = preds[i] - mx;
		cov += dx*residual(i, 0);
		var += dx * dx;
	}

	var = max(var, 1e-5);

	a[0] = (float)(cov/var);
	a[1] = (float)(my - a[0]*mx);

	MLOG("cov %f var %f mx %f my %f a[0] %f a[1] %f shrinkage %f\n", cov, var, mx, my, a[0], a[1], params.shrinkage);

	// DEBUG
	a[0] = 1;
	a[1] = 0;

	a[0] *= params.shrinkage;
	a[1] *= params.shrinkage;

	return 0;
}

//======================================================================================================================
int MedBooster::update_round_residuals(MedMat<float> &y, vector<float> &acc_preds, MedMat<float> &residual, vector<float> &preds, vector<float> &a)
{
	if (params.loss_func == "square") return update_round_residuals_single_linear(y, acc_preds, residual, preds, a);
	MERR("MedBooster::update_round_residuals: loss_func %s is not supported in gradient boosting\n", params.loss_func.c_str());
	return -1;
}


//======================================================================================================================
int MedBooster::update_round_residuals_single_linear(MedMat<float> &y, vector<float> &acc_preds, MedMat<float> &residual, vector<float> &preds, vector<float> &a)
{


	// alphas are already shrinked so not much to do

	int is_first = 0;
	if (acc_preds.size() == 0) {
		is_first=1; acc_preds.resize(residual.nrows, 0);
	}

	for (int i=0; i<residual.nrows; i++) {
		acc_preds[i] += a[0]*preds[i] + a[1];
		//residual(i, 0) = residual(i, 0) - a[0]*preds[i] - a[1];
	}

	// debug - testing some other function
	float momentum = (float)0.9;
	MedMat<float> r = residual;
	for (int i=0; i<residual.nrows; i++) {

		//residual(i, 0) = y(i, 0) - acc_preds[i]; // square func original gradient

		if (acc_preds[i] < 0) residual(i, 0) = y(i, 0);
		if (acc_preds[i] >=0 && acc_preds[i] <= 1) residual(i, 0) = y(i, 0) - acc_preds[i];
		if (acc_preds[i] > 1) residual(i, 0) = y(i, 0) - 1;

		if (!is_first)
			residual(i, 0) = momentum*r(i, 0) + ((float)1-momentum)*residual(i, 0);
	}

	return 0;
}

//======================================================================================================================
int MedBooster::predict(MedMat<float> &x, vector<float> &preds) const
{

	// first we get predictions for all rounds
	vector<vector<float>> round_preds(params.nrounds);

#pragma omp parallel for
	for (int i=0; i<params.nrounds; i++)
		predictors[i]->predict(x, round_preds[i]);

	if (params.booster_type == "gradient" && params.loss_func == "square") {
		
		// simple apply of alphas
#pragma omp parallel for
		for (int i=0; i<params.nrounds; i++)
			for (int j=0; j<round_preds[i].size(); j++)
				round_preds[i][j] = alpha(i, 0)*round_preds[i][j] + alpha(i, 1);

		// sum it up
		preds.clear();
		preds.resize(x.nrows, 0);
#pragma omp parallel for
		for (int j=0; j<params.nrounds; j++)
			for (int i=0; i<preds.size(); i++)
				preds[i] += round_preds[j][i];

	}
	else {
		MERR("MedBooster::predict: booster_type=%s , loss_func=%s not supported\n", params.booster_type.c_str(), params.loss_func.c_str());
		return -1;
	}

	return 0;
}


//======================================================================================================================
size_t MedBooster::get_size()
{
	MERR("MedBooster serialization not yet implemented...\n");
	return 0;
}

//======================================================================================================================
size_t MedBooster::serialize(unsigned char *blob)
{
	MERR("MedBooster serialization not yet implemented...\n");
	return 0;
}

//======================================================================================================================
size_t MedBooster::deserialize(unsigned char *blob)
{
	MERR("MedBooster serialization not yet implemented...\n");
	return 0;
}