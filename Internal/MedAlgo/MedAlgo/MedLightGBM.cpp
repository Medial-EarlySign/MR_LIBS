#include "MedLightGBM.h"

#include <Logger/Logger/Logger.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <unordered_set>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL LOG_DEF_LEVEL
extern MedLogger global_logger;

//===============================================================================================
// MedLightGBM Helper Methods
//===============================================================================================

string MedLightGBM::build_c_api_param_string() const
{
	string init = params.defaults + ";" + params.user_params;
	map<string, string> init_map;
	MedSerialize::initialization_text_to_map(init, init_map);

	stringstream param_ss;
	for (const auto &pair : init_map)
	{
		param_ss << pair.first << "=" << pair.second << " ";
	}
	return param_ss.str();
}

int MedLightGBM::get_num_iterations() const
{
	string init = params.defaults + ";" + params.user_params;
	map<string, string> init_map;
	MedSerialize::initialization_text_to_map(init, init_map);

	if (init_map.find("num_iterations") != init_map.end())
		return stoi(init_map.at("num_iterations"));
	if (init_map.find("num_trees") != init_map.end())
		return stoi(init_map.at("num_trees"));
	return 100; // default fallback
}

//===============================================================================================
// Initialization
//===============================================================================================

MedLightGBM::MedLightGBM()
{
	classifier_type = MODEL_LIGHTGBM;
	normalize_for_learn = false;
	normalize_for_predict = false;
	normalize_y_for_learn = false;
	transpose_for_learn = false;
	transpose_for_predict = false;
	booster_handle_ = nullptr;

	init_from_string("");
	_mark_learn_done = false;
	prepared_single = false;
}

MedLightGBM::~MedLightGBM()
{
	if (booster_handle_ != nullptr)
	{
		LGBM_BoosterFree(booster_handle_);
		booster_handle_ = nullptr;
	}
}

int MedLightGBM::init(map<string, string> &initialization_map)
{
	return set_params(initialization_map);
}
int MedLightGBM::set_params(map<string, string> &initialization_map)
{
	for (const auto &pair : initialization_map)
	{
		// Append to user_params in the format "key=value;"
		params.user_params += pair.first + "=" + pair.second + ";";
	}
	return 0;
}

int MedLightGBM::init_from_string(string init_str)
{
	params.user_params += init_str;
	return 0;
}

//===============================================================================================
// Learn & Predict
//===============================================================================================

int MedLightGBM::Learn(float *x, float *y, const float *w, int nsamples, int nftrs)
{
	global_logger.log(LOG_MEDALGO, LOG_DEF_LEVEL, "Starting a LightGBM train session...\n");

	// 1. Get configuration
	string param_str = build_c_api_param_string();
	int num_iters = get_num_iterations();

	// 2. Create Dataset from Memory (Automatically parallelized in C)
	DatasetHandle train_data = nullptr;
	int ret = LGBM_DatasetCreateFromMat(x, C_API_DTYPE_FLOAT32, nsamples, nftrs, 1, param_str.c_str(), nullptr, &train_data);
	if (ret != 0)
		MTHROW_AND_ERR("LightGBM Dataset creation failed: %s", LGBM_GetLastError());

	// 3. Set Targets & Weights
	LGBM_DatasetSetField(train_data, "label", y, nsamples, C_API_DTYPE_FLOAT32);
	if (w != nullptr)
		LGBM_DatasetSetField(train_data, "weight", w, nsamples, C_API_DTYPE_FLOAT32);

	// 4. Clean up old booster if it exists
	if (booster_handle_ != nullptr)
		LGBM_BoosterFree(booster_handle_);

	// 5. Create new Booster
	ret = LGBM_BoosterCreate(train_data, param_str.c_str(), &booster_handle_);
	if (ret != 0)
		MTHROW_AND_ERR("LightGBM Booster creation failed: %s", LGBM_GetLastError());

	// 6. Train Loop
	int is_finished = 0;
	for (int iter = 0; iter < num_iters && !is_finished; ++iter)
	{
		LGBM_BoosterUpdateOneIter(booster_handle_, &is_finished);
	}

	LGBM_DatasetFree(train_data);

	_mark_learn_done = true;
	global_logger.log(LOG_MEDALGO, LOG_DEF_LEVEL, "Finished LightGBM training.\n");
	return 0;
}

int MedLightGBM::Predict(float *x, float *&preds, int nsamples, int nftrs) const
{
	if (booster_handle_ == nullptr)
		MTHROW_AND_ERR("Error: Predict called but booster_handle is null.\n");

	int num_preds_per_row = n_preds_per_sample();
	int64_t len_res = static_cast<int64_t>(nsamples) * num_preds_per_row;

	if (preds == nullptr)
		preds = new float[len_res];

	// C API requires double output buffer
	vector<double> out_result(len_res);
	int64_t out_len = 0;

	int ret = LGBM_BoosterPredictForMat(booster_handle_, x, C_API_DTYPE_FLOAT32, nsamples, nftrs, 1,
										C_API_PREDICT_NORMAL, 0, -1, build_c_api_param_string().c_str(),
										&out_len, out_result.data());

	if (ret != 0)
		MTHROW_AND_ERR("LightGBM Predict failed: %s", LGBM_GetLastError());

	// Cast double results to float
	for (int64_t i = 0; i < len_res; i++)
		preds[i] = static_cast<float>(out_result[i]);

	return 0;
}

//===============================================================================================
// SHAP and Feature Importance
//===============================================================================================

void MedLightGBM::calc_feature_contribs(MedMat<float> &x, MedMat<float> &contribs) const
{
	if (booster_handle_ == nullptr)
		MTHROW_AND_ERR("Error: SHAP Predict called but booster is null.\n");

	int nrows = x.nrows;
	int ncols = x.ncols;

	contribs.resize(nrows, ncols + 1);
	contribs.signals.insert(contribs.signals.end(), x.signals.begin(), x.signals.end());
	contribs.signals.push_back("b0");
	contribs.recordsMetadata.insert(contribs.recordsMetadata.end(), x.recordsMetadata.begin(), x.recordsMetadata.end());

	vector<double> out_result(static_cast<size_t>(ncols + 1) * static_cast<size_t>(nrows));
	int64_t out_len = 0;

	int ret = LGBM_BoosterPredictForMat(booster_handle_, x.data_ptr(), C_API_DTYPE_FLOAT32, nrows, ncols, 1,
										C_API_PREDICT_CONTRIB, 0, -1, build_c_api_param_string().c_str(),
										&out_len, out_result.data());

	if (ret != 0)
		MTHROW_AND_ERR("LightGBM SHAP Predict failed: %s", LGBM_GetLastError());

	float *contribs_ptr = contribs.data_ptr();
	for (int64_t i = 0; i < out_len; i++)
		contribs_ptr[i] = static_cast<float>(out_result[i]);
}

void MedLightGBM::calc_feature_importance(vector<float> &features_importance_scores, const string &general_params, const MedFeatures *features)
{
	if (!_mark_learn_done || booster_handle_ == nullptr)
		MTHROW_AND_ERR("ERROR:: Requested feature importance before running learn\n");

	map<string, string> params;
	MedSerialize::initialization_text_to_map(general_params, params);

	string importance_type = "gain";
	if (params.find("importance_type") != params.end())
		importance_type = params.at("importance_type");

	int imp_type = C_API_FEATURE_IMPORTANCE_GAIN;
	if (importance_type == "split")
		imp_type = C_API_FEATURE_IMPORTANCE_SPLIT;
	else if (importance_type == "shap")
	{
		// Execute SHAP branch exactly as it was
		MedMat<float> feat_mat, contribs_mat;
		if (features == NULL)
			MTHROW_AND_ERR("SHAP values feature importance requires features \n");
		features->get_as_matrix(feat_mat);
		calc_feature_contribs(feat_mat, contribs_mat);

		features_importance_scores.resize(contribs_mat.ncols);
#pragma omp parallel for
		for (int j = 0; j < contribs_mat.ncols; ++j)
		{
			float col_sum = 0;
			for (int i = 0; i < contribs_mat.nrows; ++i)
				col_sum += std::abs(contribs_mat.get(i, j));
			features_importance_scores[j] = col_sum / (float)contribs_mat.nrows;
		}
		return;
	}

	// Standard Native Importance API (Gain/Split)
	int n_ftrs = model_features.empty() ? features_count : (int)model_features.size();
	features_importance_scores.resize(n_ftrs);

	vector<double> out_results(n_ftrs);
	LGBM_BoosterFeatureImportance(booster_handle_, 0, imp_type, out_results.data());

	for (int i = 0; i < n_ftrs; i++)
		features_importance_scores[i] = static_cast<float>(out_results[i]);
}

//===============================================================================================
// Single Predictors & Helpers
//===============================================================================================

int MedLightGBM::n_preds_per_sample() const
{
	if (booster_handle_ == nullptr)
		return 1;
	int num_classes = 1;
	LGBM_BoosterGetNumClasses(booster_handle_, &num_classes);
	return num_classes;
}

void MedLightGBM::prepare_predict_single()
{
	if (!prepared_single)
	{
		num_preds = n_preds_per_sample();
		prepared_single = true;
	}
}

void MedLightGBM::predict_single(const vector<float> &x, vector<float> &preds) const
{
	int n_ftrs = (int)x.size();
	vector<double> out_result(num_preds);
	int64_t out_len = 0;

	LGBM_BoosterPredictForMat(booster_handle_, x.data(), C_API_DTYPE_FLOAT32, 1, n_ftrs, 1,
							  C_API_PREDICT_NORMAL, 0, -1, build_c_api_param_string().c_str(),
							  &out_len, out_result.data());

	preds.resize(num_preds);
	for (int64_t i = 0; i < num_preds; i++)
		preds[i] = static_cast<float>(out_result[i]);
}

void MedLightGBM::predict_single(const vector<double> &x, vector<double> &preds) const
{
	int n_ftrs = (int)x.size();
	preds.resize(num_preds);
	int64_t out_len = 0;

	LGBM_BoosterPredictForMat(booster_handle_, x.data(), C_API_DTYPE_FLOAT64, 1, n_ftrs, 1,
							  C_API_PREDICT_NORMAL, 0, -1, build_c_api_param_string().c_str(),
							  &out_len, preds.data());
}

//===============================================================================================
// Serialization
//===============================================================================================

void MedLightGBM::pre_serialization()
{
	model_as_string = "";
	if (_mark_learn_done && booster_handle_ != nullptr)
	{
		int64_t out_len = 0;
		// Probe required string buffer length
		LGBM_BoosterSaveModelToString(booster_handle_, 0, -1, 0, 0, &out_len, nullptr);

		vector<char> buffer(out_len + 1);
		LGBM_BoosterSaveModelToString(booster_handle_, 0, -1, 0, out_len + 1, &out_len, buffer.data());

		model_as_string = string(buffer.data());
	}
}

string MedLightGBM::get_model_str() {
	pre_serialization();
	return model_as_string;
}

void MedLightGBM::post_deserialization()
{
	init_from_string("");
	if (_mark_learn_done && !model_as_string.empty())
	{
		if (booster_handle_ != nullptr)
			LGBM_BoosterFree(booster_handle_);
		int out_num_iter = 0;
		LGBM_BoosterLoadModelFromString(model_as_string.c_str(), &out_num_iter, &booster_handle_);
		model_as_string = "";
	}
}

void MedLightGBM::export_predictor(const string &output_fname)
{
	pre_serialization();
	ofstream ofs(output_fname, std::ios::binary);
	if (!ofs)
		MTHROW_AND_ERR("MedLightGBM::export_predictor failed. couldn't write %s \n", output_fname.c_str());
	ofs << model_as_string;
	ofs.close();
}

void MedLightGBM::print(FILE *fp, const string &prefix, int level) const
{
	if (level == 0)
		fprintf(fp, "%s: MedLightGBM ()\n", prefix.c_str());
	else
	{
		// Probe string to print it directly
		int64_t out_len = 0;
		LGBM_BoosterSaveModelToString(booster_handle_, 0, -1, 0, 0, &out_len, nullptr);
		vector<char> buffer(out_len + 1);
		LGBM_BoosterSaveModelToString(booster_handle_, 0, -1, 0, out_len + 1, &out_len, buffer.data());

		fprintf(fp, "%s: MedLightGBM ()\n%s\n", prefix.c_str(), buffer.data());
	}
}