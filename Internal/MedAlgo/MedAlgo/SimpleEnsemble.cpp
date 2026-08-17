//
// SimpleEnsemble : "simple_ensemble" , give 1 or more models to train, and ensemble them with given weights from the user.
//
// input example: 
// "predictor_1=gdlm;predictor_params_1={method=logistic_sgd;last_is_bias=0;stop_at_err=1e-4;batch_size=2048;momentum=0.95;rate=0.1};weight_1=0.5;predictor_2=lightgbm;predictor_params_2={..};weight_2=0.5
//
#include <MedAlgo/MedAlgo/SimpleEnsemble.h>
#include <iostream>
#include <string>

#include <Logger/Logger/Logger.h>
#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_APP
#define LOCAL_LEVEL LOG_DEF_LEVEL
extern MedLogger global_logger;


//==================================================================
int MedSimpleEnsemble::set_params(map<string, string>& mapper)
{
	for (auto entry : mapper) {
		string field = entry.first;

		if (field.rfind("predictor_params_", 0) == 0) {
			string s = field.substr(17, 1000);
			int n = stoi(field.substr(17, 1000));
			if (n > (int)predictor_params.size()) predictor_params.resize(n);
			predictor_params[n-1] = entry.second;
		}

		else if (field.rfind("predictor_",0) == 0) {
			int n = stoi(field.substr(10, 1000));
			if (n > (int)predictor_names.size()) predictor_names.resize(n);
			predictor_names[n-1] = entry.second;
		}

		else if (field.rfind("weight_", 0) == 0) {
			int n = stoi(field.substr(7, 1000));
			if (n > (int)weights.size()) weights.resize(n);
			weights[n-1] = stof(entry.second);
		}

		else if (field == "combiner") { combiner_name = entry.second; }
		else if (field == "combiner_params") { combiner_params = entry.second; }
		else if (field == "p_combine") { p_combine = stof(entry.second); }

		else MLOG("Unknown parameter \'%s\' for QRF\n", field.c_str());
		//! [MedQRF::init]

	}

	if (predictor_params.size() > predictor_names.size())
		MTHROW_AND_ERR("ERROR: MedSimpleEnsemble : got %d predictor params and only %d predictors\n", (int)predictor_params.size(), (int)predictor_names.size());
	if (predictor_params.size() < predictor_names.size()) predictor_params.resize(predictor_names.size());
	if (weights.size() == 0 && predictor_names.size() > 0) {
		float w = (float)1 / (float)predictor_names.size();
		weights.resize(predictor_names.size(), w);
	}

	if (weights.size() != predictor_names.size())
		MTHROW_AND_ERR("ERROR: MedSimpleEnsemble : got %d weights and only %d predictors\n", (int)weights.size(), (int)predictor_names.size());


	// initializing predictors
	predictors.resize(predictor_names.size());
	for (int i = 0; i < predictor_names.size(); i++) {
		MLOG("Ensemble init : predictor %d : %s : %f : %s\n", i, predictor_names[i].c_str(), weights[i], predictor_params[i].c_str());
		predictors[i] = MedPredictor::make_predictor(predictor_names[i], predictor_params[i]);
		if (predictors[i] == NULL)
			MTHROW_AND_ERR("ERROR: MedSimpleEnsemble : failed initializing predictor[%d]: %s with params: %s\n", i+1, predictor_names[i].c_str(), predictor_params[i].c_str());
	}

	// initializing combiner
	if (combiner_name != "") {
		MLOG("Ensemble init : combiner: %s : %s\n", combiner_name.c_str(), combiner_params.c_str());
		combiner = MedPredictor::make_predictor(combiner_name, combiner_params);
		if (combiner == NULL)
			MTHROW_AND_ERR("ERROR: MedSimpleEnsemble : failed initializing combiner: %s with params: %s\n", combiner_name.c_str(), combiner_params.c_str());
		if (p_combine <= 0 || p_combine >= 1)
			MTHROW_AND_ERR("ERROR: MedSimpleEnsemble : can't work with combiner AND p_combine %f\n", p_combine);
	}

	return 0;
}


//====================================================================================================
int MedSimpleEnsemble::learn(MedMat<float> &x, MedMat<float> &y, const vector<float> &wgts)
{
	MLOG(">>>>>>>>>>>>>>>>>>>>>>>>> SIMPLE ENSEMBLE LEARN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
	
	if (combiner == NULL) {
		for (auto &p : predictors) {
			// using copies as not sure learn is not touching x or y
			MedMat<float> x_copy = x;
			MedMat<float> y_copy = y;
			if (p->learn(x_copy, y_copy, wgts) != 0)
				MTHROW_AND_ERR("MedSimpleEnsemble : failed learning predictor\n");
		}
	}
	else {
		MedMat<float> x_learn, x_combine;
		vector<int> inds0, inds1, empty;
		x.random_split_mat_by_ids(x_combine, x_learn, p_combine, inds0, inds1);
		MLOG("MedSimpleEnsemble: Split in X matrix %d x %d to learn mat : %d x %d and combine mat: %d x %d\n", x.nrows, x.ncols, x_learn.nrows, x_learn.ncols, x_combine.nrows, x_combine.ncols);
		MedMat<float> y_learn, y_combine;
		y_learn = y;
		y_learn.get_sub_mat(inds1, empty);
		y_combine = y;
		y_combine.get_sub_mat(inds0, empty);
		MLOG("MedSimpleEnsemble: Split in Y matrix %d x %d to learn mat : %d x %d and combine mat: %d x %d\n", y.nrows, y.ncols, y_learn.nrows, y_learn.ncols, y_combine.nrows, y_combine.ncols);
		vector<float> w_learn, w_combine;
		if (wgts.size() > 0) {
			for (auto i : inds0) w_combine.push_back(wgts[i]);
			for (auto i : inds1) w_learn.push_back(wgts[i]);
		}

		for (auto &p : predictors) {
			// using copies as not sure learn is not touching x or y
			MedMat<float> x_copy = x_learn;
			MedMat<float> y_copy = y_learn;
			if (p->learn(x_copy, y_copy, w_learn) != 0)
				MTHROW_AND_ERR("MedSimpleEnsemble : failed learning predictor\n");
		}

		MLOG("MedSimpleEnsemble: learned all ensemble members, now learning the combiner.\n");
		MedMat<float> pre_combine;
		predict_pre_combine(x_combine, pre_combine);
		MLOG("Learn Combine...\n");
		if (combiner->learn(pre_combine, y_combine, w_combine) != 0)
			MTHROW_AND_ERR("MedSimpleEnsemble : failed learning predictor\n");

	}

	return 0;
}

//====================================================================================================
int MedSimpleEnsemble::predict_pre_combine(MedMat<float> &x, MedMat<float> &_preds) const
{
	MLOG(">>>>>>>>>>>>>>>>>>>>>>>>> SIMPLE ENSEMBLE PRE COMBINE PREDICT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
	vector<vector<float>> preds;

	int n_samples = (int)x.nrows;
	int n_predictors = (int)predictors.size();
	int n_preds_per_sample = predictors[0]->n_preds_per_sample();
	int n_predictions = n_preds_per_sample * n_samples;

	preds.resize(n_predictors);
	for (int i = 0; i < n_predictors; i++)
		preds[i].resize(n_predictions);

	for (int i = 0; i < n_predictors; i++) {
		predictors[i]->predict(x, preds[i]);
	}

	_preds.copy_header(x);
	_preds.ncols = static_cast<unsigned long long>(n_predictors) * static_cast<unsigned long long>(n_preds_per_sample);
	_preds.resize(n_samples, n_predictors * n_preds_per_sample);

	for (int j = 0; j < n_predictors; j++) {
		for (int i = 0; i < n_samples; i++) {
			for (int k = 0; k < n_preds_per_sample; k++) {
				_preds(i, j*n_preds_per_sample + k) = preds[j][i*n_preds_per_sample+k];
			}
		}
	}

	vector<string> sig_names;
	for (int j = 0; j < n_predictors; j++)
		for (int k = 0; k < n_preds_per_sample; k++)
			sig_names.push_back("Ensemble_predictor_" + to_string(j) + "_channel_" + to_string(k));
	_preds.set_signals(sig_names);

	return 0;
}

//====================================================================================================
int MedSimpleEnsemble::predict(MedMat<float> &x, vector<float> &combined_preds) const
{
	MLOG(">>>>>>>>>>>>>>>>>>>>>>>>> SIMPLE ENSEMBLE PREDICT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
	if (combiner == NULL) {
		vector<vector<float>> preds;

		int n_samples = (int)x.nrows;
		int n_predictors = (int)predictors.size();
		int n_preds_per_sample = predictors[0]->n_preds_per_sample();
		int n_predictions = n_preds_per_sample * n_samples;

		preds.resize(n_predictors);
		for (int i = 0; i < n_predictors; i++)
			preds[i].resize(n_predictions);

		for (int i = 0; i < n_predictors; i++) {
			predictors[i]->predict(x, preds[i]);
		}

		combined_preds.resize(n_predictions, 0);
		for (int j = 0; j < n_predictors; j++) {
			for (int i = 0; i < n_predictions; i++) {
				combined_preds[i] += preds[j][i] * weights[j];
			}
		}
	}
	else {
		MedMat<float> pre_combine;
		predict_pre_combine(x, pre_combine);
		combiner->predict(pre_combine, combined_preds);
	}

	return 0;
}