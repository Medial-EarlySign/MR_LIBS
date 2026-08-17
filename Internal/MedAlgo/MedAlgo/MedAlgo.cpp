//
// MedAlgo - unified wrappers for prediction algorithms
//

#include <MedAlgo/MedAlgo/MedAlgo.h>
#include <MedAlgo/MedAlgo/MedLM.h>
#include <MedAlgo/MedAlgo/MedGDLM.h>
#include <MedAlgo/MedAlgo/MedQRF.h>
#include <MedAlgo/MedAlgo/MedXGB.h>
#include <MedAlgo/MedAlgo/MedDeepBit.h>
#include "MedProcessTools/MedProcessTools/MedFeatures.h"
#include <MedAlgo/MedAlgo/MedLightGBM.h>
#include <MedAlgo/MedAlgo/MedLinearModel.h>
#include <MedAlgo/MedAlgo/MedBART.h>
#include <MedAlgo/MedAlgo/ExternalNN.h>
#include <MedAlgo/MedAlgo/SimpleEnsemble.h>
#include <MedAlgo/MedAlgo/MedMicNet.h>
#include <MedAlgo/MedAlgo/MedBooster.h>
#include <MedAlgo/MedAlgo/MedSpecificGroupModels.h>
#include <MedAlgo/MedAlgo/MedTQRF.h>
#include <MedAlgo/MedAlgo/MedSVM.h>
#include <MedAlgo/MedAlgo/MedBP.h>
#include <MedAlgo/MedAlgo/MedKNN.h>
#include <MedAlgo/MedAlgo/MedMultiClass.h>
#include <MedUtils/MedUtils/MedGenUtils.h>
#include <MedAlgo/MedAlgo/MedPredictorsByMissingValues.h>
#include <Eigen/Core>
#include <cmath>

#include <thread>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

unordered_map<int, string> predictor_type_to_name = {
	{ MODEL_LINEAR_MODEL , "linear_model"} ,
	{ MODEL_GD_LINEAR , "gdlm" },
	{ MODEL_QRF , "qrf" },
	{ MODEL_KNN , "knn" },
	{ MODEL_MARS , "mars" },
	{ MODEL_BP , "BP" },
	{ MODEL_MULTI_CLASS , "multi_class" },
	{ MODEL_XGB , "xgb" },
	{ MODEL_LASSO , "lasso" },
	{ MODEL_MIC_NET , "micNet" },
	{ MODEL_BOOSTER , "booster" },
	{ MODEL_DEEP_BIT , "deep_bit" },
	{ MODEL_LIGHTGBM , "lightgbm" },
	{ MODEL_SVM , "svm" },
	{ MODEL_LIGHTGBM , "lightgbm" },
	{ MODEL_LINEAR_SGD , "linear_sgd" },
	{ MODEL_SPECIFIC_GROUPS_MODELS, "multi_models" },
	{ MODEL_VW, "vw" },
	{ MODEL_TQRF, "tqrf"},
	{ MODEL_BART, "bart" },
	{ MODEL_EXTERNAL_NN, "external_nn" },
	{ MODEL_SIMPLE_ENSEMBLE, "simple_ensemble" },
	{ MODEL_BY_MISSING_VALUES_SUBSET, "by_missing_value_subset" }
};
//=======================================================================================
// MedPredictor
//=======================================================================================
// Model types
MedPredictorTypes predictor_name_to_type(const string& model_name) {
	for (auto it = predictor_type_to_name.begin(); it != predictor_type_to_name.end(); ++it)
		if (it->second == model_name) {
			return MedPredictorTypes(it->first);
		}
	string full_list = medial::io::get_list_op(predictor_type_to_name, "\n");
	MTHROW_AND_ERR("Unknown predictor name \"%s\" - Please choose one of: %s\n",
		model_name.c_str(), full_list.c_str());
}

// Initialization
//.......................................................................................
MedPredictor * MedPredictor::make_predictor(string model_type) {

	return make_predictor(predictor_name_to_type(model_type));
}

//.......................................................................................
MedPredictor * MedPredictor::make_predictor(string model_type, string init_string) {

	return make_predictor(predictor_name_to_type(model_type), init_string);
}

void *MedPredictor::new_polymorphic(string dname)
{
	CONDITIONAL_NEW_CLASS(dname, MedLM);
	CONDITIONAL_NEW_CLASS(dname, MedGDLM);
	CONDITIONAL_NEW_CLASS(dname, MedQRF);
	CONDITIONAL_NEW_CLASS(dname, MedKNN);
	CONDITIONAL_NEW_CLASS(dname, MedBP);
	CONDITIONAL_NEW_CLASS(dname, MedMultiClass);
	CONDITIONAL_NEW_CLASS(dname, MedXGB);
	CONDITIONAL_NEW_CLASS(dname, MedLasso);
	CONDITIONAL_NEW_CLASS(dname, MedMicNet);
	CONDITIONAL_NEW_CLASS(dname, MedBooster);
	CONDITIONAL_NEW_CLASS(dname, MedDeepBit);
	CONDITIONAL_NEW_CLASS(dname, MedLightGBM);
	CONDITIONAL_NEW_CLASS(dname, MedSpecificGroupModels);
	CONDITIONAL_NEW_CLASS(dname, MedSvm);
	CONDITIONAL_NEW_CLASS(dname, MedTQRF);
	CONDITIONAL_NEW_CLASS(dname, MedBART);
	CONDITIONAL_NEW_CLASS(dname, MedLinearModel);
	CONDITIONAL_NEW_CLASS(dname, MedExternalNN);
	CONDITIONAL_NEW_CLASS(dname, MedSimpleEnsemble);
	CONDITIONAL_NEW_CLASS(dname, MedPredictorsByMissingValues);
	MWARN("Warning in MedPredictor::new_polymorphic - Unsupported class %s\n", dname.c_str());
	return NULL;
}

//.......................................................................................
MedPredictor * MedPredictor::make_predictor(MedPredictorTypes model_type) {

	if (model_type == MODEL_LINEAR_MODEL)
		return new MedLM;
	else if (model_type == MODEL_GD_LINEAR)
		return new MedGDLM;
	else if (model_type == MODEL_QRF)
		return new MedQRF;
	else if (model_type == MODEL_KNN)
		return new MedKNN;
	else if (model_type == MODEL_BP)
		return new MedBP;
	else if (model_type == MODEL_MULTI_CLASS)
		return new MedMultiClass;
	else if (model_type == MODEL_XGB)
		return new MedXGB;
	else if (model_type == MODEL_LASSO)
		return new MedLasso;
	else if (model_type == MODEL_MIC_NET)
		return new MedMicNet;
	else if (model_type == MODEL_BOOSTER)
		return new MedBooster;
	else if (model_type == MODEL_DEEP_BIT)
		return new MedDeepBit;
	else if (model_type == MODEL_LIGHTGBM)
		return new MedLightGBM;
	else if (model_type == MODEL_SPECIFIC_GROUPS_MODELS)
		return new MedSpecificGroupModels;
	else if (model_type == MODEL_SVM)
		return new MedSvm;
	else if (model_type == MODEL_TQRF)
		return new MedTQRF;
	else if (model_type == MODEL_BART)
		return new MedBART;
	else if (model_type == MODEL_LINEAR_SGD)
		return new MedLinearModel;
	else if (model_type == MODEL_EXTERNAL_NN)
		return new MedExternalNN;
	else if (model_type == MODEL_SIMPLE_ENSEMBLE)
		return new MedSimpleEnsemble;
	else if (model_type == MODEL_BY_MISSING_VALUES_SUBSET)
		return new MedPredictorsByMissingValues;
	else
		return NULL;

}
MedPredictor * MedPredictor::make_predictor(MedPredictorTypes model_type, string init_string) {

	MedPredictor *newPred = make_predictor(model_type);
	MLOG("MedPredictor: making predictor %d with params %s\n", model_type, init_string.c_str());
	newPred->init_from_string(init_string);

	return newPred;
}
//.......................................................................................
int MedPredictor::init_from_string(string text) {

	MLOG_D("MedPredictor init from string [%s] (classifier type is %d)\n", text.c_str(), classifier_type);

	// parse text of the format "Name = Value ; Name = Value ; ..."

	if (classifier_type == MODEL_MIC_NET) {
		cerr << "But we are going to call mic net version directly\n";
		MedMicNet *mic = (MedMicNet *)this;
		cerr << "before\n";
		int rc = mic->init_from_string(text);
		cerr << "after " << rc << "\n";
		return rc;
	}

	if (classifier_type == MODEL_BOOSTER) {
		cerr << "But we are going to call booster version directly\n";
		MedBooster *med_b = (MedBooster *)this;
		cerr << "before\n";
		int rc = med_b->init_from_string(text);
		cerr << "after " << rc << "\n";
		return rc;
	}

	if (classifier_type == MODEL_LIGHTGBM) {
		MedLightGBM *med_light = (MedLightGBM *)this;
		return med_light->init_from_string(text);
	}
	// remove white spaces
	text.erase(remove_if(text.begin(), text.end(), ::isspace), text.end());

	map<string, string> init_map;
	if (MedSerialize::initialization_text_to_map(text, init_map) == -1)
		return -1;

	for (auto rec : init_map)
		MLOG_D("Initializing predictor with \'%s\' = \'%s\'\n", rec.first.c_str(), rec.second.c_str());

	init(init_map);

	return 0;
}

int MedPredictor::init(map<string, string>& mapper) {
	init_defaults();

	return set_params(mapper);
}

void MedPredictor::prepare_predict_single() {
	MWARN("WARN: Not impelemnted in %s\n", my_class_name().c_str());
}

//.......................................................................................
void MedPredictor::prepare_x_mat(MedMat<float> &x, const vector<float> &wgts, int &nsamples, int &nftrs, bool transpose_needed) const
{
	if ((transpose_needed && !x.transposed_flag) || (!transpose_needed && x.transposed_flag)) {
		//		MLOG("transposing matrix\n");
		x.transpose();
	}

	if (x.transposed_flag) {
		nsamples = x.ncols;
		nftrs = x.nrows;
	}
	else {
		nsamples = x.nrows;
		nftrs = x.ncols;
	}
}

string norm_feature_name(const string &feat_name) {
	return  feat_name.substr(0, 3) != "FTR" || feat_name.find_first_of('.') == string::npos ? feat_name :
		feat_name.substr(feat_name.find_first_of('.') + 1);
}

//.......................................................................................
int MedPredictor::learn(MedMat<float> &x, MedMat<float> &y, const vector<float> &wgts)
{
	if (!x.signals.empty())
		model_features = x.signals;
	else {
		model_features.clear();
		features_count = x.ncols;
	}
	int nsamples, nftrs;

	// patch for micNet
	if (classifier_type == MODEL_MIC_NET) {
		MedMicNet *mic = (MedMicNet *)this;
		cerr << "running micNet learn()\n";
		vector<float> w = wgts;
		return mic->learn(x, y, w);
	}
	// patch for booster
	if (classifier_type == MODEL_BOOSTER) {
		MedBooster *med_b = (MedBooster *)this;
		cerr << "running MedBooster learn()\n";
		return med_b->learn(x, y);
	}

	// ToDo : sanity check of sizes (nonzero, matching x,y dimensions)

	if (normalize_for_learn && !x.normalized_flag) {
		MERR("Learner Requires normalized matrix. Quitting\n");
		return -1;
	}

	prepare_x_mat(x, wgts, nsamples, nftrs, transpose_for_learn);
	if (normalize_y_for_learn && !y.normalized_flag)
		y.normalize();

	return Learn(x.data_ptr(), y.data_ptr(), VEC_DATA(wgts), nsamples, nftrs);
}

//.......................................................................................
int MedPredictor::learn(MedMat<float> &x, vector<float> &y, const vector<float> &wgts) {
	if (!x.signals.empty())
		model_features = x.signals;
	else {
		model_features.clear();
		features_count = x.ncols;
	}
	int nsamples, nftrs;

	if (normalize_for_learn && !x.normalized_flag) {
		MERR("Learner Requires normalized matrix. Quitting\n");
		return -1;
	}

	prepare_x_mat(x, wgts, nsamples, nftrs, transpose_for_learn);

	return Learn(x.data_ptr(), y.data(), VEC_DATA(wgts), nsamples, nftrs);
}

//.......................................................................................
int MedPredictor::learn(vector<float> &x, vector<float> &y, const vector<float> &w, int n_samples, int n_ftrs)
{
	features_count = n_ftrs;
	return(Learn(VEC_DATA(x), VEC_DATA(y), VEC_DATA(w), n_samples, n_ftrs));
}

//.......................................................................................
int MedPredictor::predict(MedMat<float> &x, vector<float> &preds) const {
	if (!model_features.empty()) {//test names of entered matrix:
		if (model_features.size() != x.ncols)
			MTHROW_AND_ERR("(1) Learned Feature model size was %d, request feature size for predict was %d\n",
			(int)model_features.size(), (int)x.ncols);

		if (!x.signals.empty()) //can compare names
			for (int feat_num = 0; feat_num < model_features.size(); ++feat_num)
				if (norm_feature_name(x.signals[feat_num]) != norm_feature_name(model_features[feat_num]))
					MTHROW_AND_ERR("Learned Features are the same. feat_num=%d. in learning was %s, now recieved %s\n",
					(int)model_features.size(), model_features[feat_num].c_str(), x.signals[feat_num].c_str());
	}
	else if (features_count > 0 && features_count != x.ncols)
		MTHROW_AND_ERR("(2) Learned Feature model size was %d, request feature size for predict was %d\n",
			features_count, (int)x.ncols);

	int nsamples, nftrs;
	vector<float> w;

	// patch for micNet
	if (classifier_type == MODEL_MIC_NET) {
		MedMicNet *mic = (MedMicNet *)this;
		cerr << "running micNet predict()\n";
		return mic->predict(x, preds);
	}
	// patch for booster
	if (classifier_type == MODEL_BOOSTER) {
		MedBooster *med_b = (MedBooster *)this;
		cerr << "running MedBooster predict()\n";
		return med_b->predict(x, preds);
	}

	if (classifier_type == MODEL_TQRF) {
		return ((MedTQRF *)this)->Predict(x, preds);
	}

	if (classifier_type == MODEL_EXTERNAL_NN) {
		return ((MedExternalNN *)this)->Predict(x, preds);
	}

	if (normalize_for_predict && !x.normalized_flag) {
		MERR("Predictor Requires normalized matrix. Quitting\n");
		return -1;
	}


	prepare_x_mat(x, w, nsamples, nftrs, transpose_for_predict);

	preds.resize(static_cast<size_t>(nsamples) * static_cast<size_t>(n_preds_per_sample()));
	float *_preds = &(preds[0]);

	//	MLOG("MedMat,vector call: preds size is %d n_preds_per_sample %d nsamples %d\n",preds.size(),n_preds_per_sample(),nsamples);
	return Predict(x.data_ptr(), _preds, nsamples, nftrs);
}

struct pred_thread_info {
	int id;
	int from_sample;
	int to_sample;
	float *preds;
	MedMat<float> *x;
	int nftrs;
	int n_preds_per_sample;
	int rc;
	int state;
};

void MedPredictor::predict_thread(void *p) const
//void MedPredictor::predict_thread()
{

	pred_thread_info *tp = (pred_thread_info *)p;

	//MLOG("Start thread %d : from: %d to: %d\n",tp->id,tp->from_sample,tp->to_sample);
	float *x = tp->x->data_ptr(tp->from_sample, 0);  //m[tp->from_sample * tp->nftrs]);
	float *preds = &(tp->preds[tp->from_sample * n_preds_per_sample()]);
	int nsamples = tp->to_sample - tp->from_sample + 1;
	int nftrs = tp->nftrs;

	tp->rc = Predict(x, preds, nsamples, nftrs);
	//MLOG("End thread %d : from: %d to: %d\n",tp->id,tp->from_sample,tp->to_sample);

	// signing job ended
	tp->state = 1;
}

//.......................................................................................
int MedPredictor::threaded_predict(MedMat<float> &x, vector<float> &preds, int nthreads) const {
	if (!model_features.empty()) {//test names of entered matrix:
		if (model_features.size() != x.ncols)
			MTHROW_AND_ERR("Learned Feature model size was %d, request feature size for predict was %d\n",
			(int)model_features.size(), (int)x.ncols);

		if (!x.signals.empty()) //can compare names
			for (int feat_num = 0; feat_num < model_features.size(); ++feat_num)
				if (norm_feature_name(x.signals[feat_num]) != norm_feature_name(model_features[feat_num]))
					MTHROW_AND_ERR("Learned Features are the same. feat_num=%d. in learning was %s, now recieved %s\n",
					(int)model_features.size(), model_features[feat_num].c_str(), x.signals[feat_num].c_str());
	}
	else if (features_count > 0 && features_count != x.ncols)
		MTHROW_AND_ERR("Learned Feature model size was %d, request feature size for predict was %d\n",
			features_count, (int)x.ncols);

	int nsamples, nftrs;
	vector<float> w;

	if (transpose_for_predict) {
		MERR("!!!!!! UNSUPORTED !!!! --> Currently threaded_predict does not support transposed matrices for predictions");
		return -1;
	}

	if (normalize_for_predict && !x.normalized_flag) {
		MERR("Predictor Requires normalized matrix. Quitting\n");
		return -1;
	}

	preds.resize(static_cast<size_t>(nsamples) * static_cast<size_t>(n_preds_per_sample()));

int th_nsamples = nsamples / nthreads;
	vector<pred_thread_info> tp(nthreads);
	for (int i = 0; i < nthreads; i++) {
		tp[i].id = i;
		tp[i].from_sample = i * th_nsamples;
		tp[i].to_sample = min((i + 1)*th_nsamples - 1, nsamples - 1);
		tp[i].preds = VEC_DATA(preds);
		tp[i].x = &x;
		tp[i].nftrs = nftrs;
		tp[i].rc = 0;
		tp[i].state = 0;
	}


	// sending threads
	vector<thread> th_handle(nthreads);
	for (int i = 0; i < nthreads; i++) {
		//		MLOG("Sending Thread %d\n",i);
		//		th_handle[i] = std::thread(&MedPredictor::predict_thread, (void *)&tp[i]);
		th_handle[i] = std::thread(&MedPredictor::predict_thread, this, (void *)&tp[i]);
	}

	int n_state = 0;
	while (n_state < nthreads) {
		this_thread::sleep_for(chrono::milliseconds(10));
		n_state = 0;
		for (int i = 0; i < nthreads; i++)
			n_state += tp[i].state;
	}
	for (int i = 0; i < nthreads; i++)
		th_handle[i].join();

	for (int i = 0; i < nthreads; i++)
		if (tp[i].rc != 0)
			return -1;
	return 0;

}


//.......................................................................................
int MedPredictor::predict(vector<float> &x, vector<float> &preds, int n_samples, int n_ftrs) const {
	if (!model_features.empty()) {
		if (model_features.size() != n_ftrs)
			MTHROW_AND_ERR("Learned Feature model size was %d, request feature size for predict was %d\n",
			(int)model_features.size(), (int)x.size());
	}
	else if (features_count > 0 && features_count != n_ftrs)
		MTHROW_AND_ERR("Learned Feature model size was %d, request feature size for predict was %d\n",
			features_count, n_ftrs);

	preds.resize(static_cast<size_t>(n_samples) * static_cast<size_t>(n_preds_per_sample()));
	float *_preds = &(preds[0]);
	return Predict(VEC_DATA(x), _preds, n_samples, n_ftrs);
}


// (De)Serialize
//.......................................................................................
size_t MedPredictor::get_predictor_size() {
	return sizeof(classifier_type) + get_size();
}

//.......................................................................................
size_t MedPredictor::predictor_serialize(unsigned char *blob) {

	size_t ptr = 0;
	memcpy(blob + ptr, &classifier_type, sizeof(MedPredictorTypes)); ptr += sizeof(MedPredictorTypes);
	ptr += serialize(blob + ptr);

	return ptr;
}

//.......................................................................................

void MedPredictor::print(FILE *fp, const string& prefix, int level) const {
	fprintf(fp, "%s: %s ()\n", prefix.c_str(), predictor_type_to_name[classifier_type].c_str());
}

int MedPredictor::learn(const MedFeatures& ftrs_data) {

	if (classifier_type == MODEL_EXTERNAL_NN) {
		return ((MedExternalNN *)this)->learn(ftrs_data);
	}
	vector<string> dummy_names;
	return learn(ftrs_data, dummy_names);
}

int MedPredictor::learn(const MedFeatures& ftrs_data, vector<string>& names) {
	//save model features names
	model_features.clear();
	for (auto it = ftrs_data.data.begin(); it != ftrs_data.data.end(); ++it)
		model_features.push_back(it->first);


	if (classifier_type == MODEL_TQRF) {
		return (((MedTQRF *)this)->Learn(ftrs_data));
	}

	// Build X
	MedMat<float> x;
	ftrs_data.get_as_matrix(x, names);

	MLOG("MedPredictor::learn() from MedFeatures, got train matrix of %d x %d\n", x.nrows, x.ncols);

	// Labels
	MedMat<float> y(x.nrows, 1);
	for (int i = 0; i < y.nrows; i++)
		y(i, 0) = ftrs_data.samples[i].outcome;

	// Weights
	if (ftrs_data.weights.size())
		return learn(x, y, ftrs_data.weights);
	else
		return learn(x, y);
}

int MedPredictor::learn_prob_calibration(MedMat<float> &x, vector<float> &y,
	vector<float> &min_range, vector<float> &max_range, vector<float> &map_prob,
	int min_bucket_size, float min_score_jump, float min_prob_jump, bool fix_prob_order) {
	// > min and <= max

	//add mapping from model score to probability based on big enough bins of score
	//get prediction for X:
	vector<float> preds;
	predict(x, preds);

	unordered_map<float, vector<int>> score_to_indexes;
	vector<float> unique_scores;
	for (size_t i = 0; i < preds.size(); ++i)
	{
		if (score_to_indexes.find(preds[i]) == score_to_indexes.end())
			unique_scores.push_back(preds[i]);
		score_to_indexes[preds[i]].push_back((int)i);
	}
	sort(unique_scores.begin(), unique_scores.end());
	int sz = (int)unique_scores.size();

	float curr_max = (float)INT32_MAX; //unbounded
	float curr_min = curr_max;
	int pred_sum = 0;
	int curr_cnt = 0;
	vector<int> bin_cnts;
	for (int i = sz - 1; i >= 0; --i)
	{
		//update values curr_cnt, pred_avg
		for (int ind : score_to_indexes[unique_scores[i]])
			pred_sum += int(y[ind] > 0);
		curr_cnt += (int)score_to_indexes[unique_scores[i]].size();

		if (curr_cnt > min_bucket_size && curr_max - unique_scores[i] > min_score_jump) {
			//flush buffer
			curr_min = unique_scores[i];
			max_range.push_back(curr_max);
			min_range.push_back(curr_min);
			map_prob.push_back(float(double(pred_sum) / curr_cnt));
			bin_cnts.push_back(curr_cnt);

			//init new buffer:
			curr_cnt = 0;
			curr_max = unique_scores[i];
			pred_sum = 0;
		}
	}
	if (curr_cnt > 0) {
		//flush last buffer
		curr_min = (float)INT32_MIN;
		max_range.push_back(curr_max);
		min_range.push_back(curr_min);
		map_prob.push_back(float(double(pred_sum) / curr_cnt));
		bin_cnts.push_back(curr_cnt);
	}

	//unite similar prob bins:
	vector<int> ind_to_unite;
	for (int i = (int)map_prob.size() - 1; i >= 1; --i)
		if (abs(map_prob[i] - map_prob[i - 1]) < min_prob_jump ||
			(fix_prob_order && map_prob[i] > map_prob[i - 1])) { //unite bins:
			ind_to_unite.push_back(i);
			int new_count = bin_cnts[i] + bin_cnts[i - 1];
			float new_prob = (map_prob[i] * bin_cnts[i] + map_prob[i - 1] * bin_cnts[i - 1]) / new_count;
			float max_th = max_range[i - 1];
			float min_th = min_range[i];
			min_range[i - 1] = min_th;
			max_range[i - 1] = max_th;
			map_prob[i - 1] = new_prob;
			bin_cnts[i - 1] = new_count;
		}

	//unite from end to start:
	for (int i = 0; i < ind_to_unite.size(); ++i)
	{
		int unite_index = ind_to_unite[i];
		//delete old records:
		min_range.erase(min_range.begin() + unite_index);
		max_range.erase(max_range.begin() + unite_index);
		map_prob.erase(map_prob.begin() + unite_index);
		bin_cnts.erase(bin_cnts.begin() + unite_index);
	}

	MLOG("Created %d bins for mapping prediction scores to probabilities\n", map_prob.size());
	for (size_t i = 0; i < map_prob.size(); ++i)
		MLOG_D("Range: [%2.4f, %2.4f] => %2.4f | %1.2f%%(%d / %d)\n",
			min_range[i], max_range[i], map_prob[i],
			100 * double(bin_cnts[i]) / y.size(), bin_cnts[i], (int)y.size());

	return 0;
}

int MedPredictor::convert_scores_to_prob(const vector<float> &preds, const vector<float> &min_range,
	const vector<float> &max_range, const vector<float> &map_prob, vector<float> &probs) const {
	probs.resize(preds.size());

	for (size_t i = 0; i < probs.size(); ++i)
	{
		//search for right range:
		int pos = 0;
		while (pos < map_prob.size() &&
			!((preds[i] > min_range[pos] || pos == map_prob.size() - 1) && (preds[i] <= max_range[pos] || pos == 0)))
			++pos;
		probs[i] = map_prob[pos];
	}

	return 0;
}

template<class T, class L> int MedPredictor::convert_scores_to_prob(const vector<T> &preds, const vector<double> &params, vector<L> &converted) const {
	converted.resize((int)preds.size());
	for (size_t i = 0; i < converted.size(); ++i)
	{
		double val = params[0];
		for (size_t k = 1; k < params.size(); ++k)
			val += params[k] * pow(double(preds[i]), double(k));
		val = 1 / (1 + exp(val));//Platt Scale technique for probability calibaration
		converted[i] = (L)val;
	}

	return 0;
}
template int MedPredictor::convert_scores_to_prob<double, double>(const vector<double> &preds, const vector<double> &params, vector<double> &converted) const;
template int MedPredictor::convert_scores_to_prob<double, float>(const vector<double> &preds, const vector<double> &params, vector<float> &converted) const;
template int MedPredictor::convert_scores_to_prob<float, double>(const vector<float> &preds, const vector<double> &params, vector<double> &converted) const;
template int MedPredictor::convert_scores_to_prob<float, float>(const vector<float> &preds, const vector<double> &params, vector<float> &converted) const;

int MedPredictor::learn_prob_calibration(MedMat<float> &x, vector<float> &y,
	int poly_rank, vector<double> &params, int min_bucket_size, float min_score_jump) {
	vector<float> min_range, max_range, map_prob;
	vector<float> preds;
	predict(x, preds);
	learn_prob_calibration(x, y, min_range, max_range, map_prob, min_bucket_size, min_score_jump);

	vector<float> probs;
	convert_scores_to_prob(preds, min_range, max_range, map_prob, probs);
	//probs is the new Y - lets learn A, B:
	MedLinearModel lm; //B is param[0], A is param[1]

	lm.loss_function = [](const vector<double> &prds, const vector<float> &y, const vector<float> *weights) {
		double res = 0;
		//L2 on 1 / (1 + exp(A*score + B)) vs Y. prds[i] = A*score+B: 1 / (1 + exp(prds))
		if (weights == NULL || weights->empty()) {
			for (size_t i = 0; i < y.size(); ++i)
			{
				double conv_prob = 1 / (1 + exp(prds[i]));
				res += (conv_prob - y[i]) * (conv_prob - y[i]);
			}
		}
		else {
			for (size_t i = 0; i < y.size(); ++i)
			{
				double conv_prob = 1 / (1 + exp(prds[i]));
				res += (conv_prob - y[i]) * (conv_prob - y[i]) * weights->at(i);
			}
		}
		res /= y.size();
		res = sqrt(res);
		return res;
	};
	lm.loss_function_step = [](const vector<double> &prds, const vector<float> &y, const vector<double> &params, const vector<float> *weights) {
		double res = 0;
		double reg_coef = 0;
		//L2 on 1 / (1 + exp(A*score + B)) vs Y. prds[i] = A*score+B: 1 / (1 + exp(prds))
		if (weights == NULL || weights->empty()) {
			for (size_t i = 0; i < y.size(); ++i)
			{
				double conv_prob = 1 / (1 + exp(prds[i]));
				res += (conv_prob - y[i]) * (conv_prob - y[i]);
			}
		}
		else {
			for (size_t i = 0; i < y.size(); ++i)
			{
				double conv_prob = 1 / (1 + exp(prds[i]));
				res += (conv_prob - y[i]) * (conv_prob - y[i]) * weights->at(i);
			}
		}
		res /= y.size();
		res = sqrt(res);
		//Reg A,B:
		if (reg_coef > 0)
			res += reg_coef * sqrt((params[0] * params[0] + params[1] * params[1]) / 2);
		return res;
	};
	lm.block_num = float(10 * sqrt(poly_rank + 1));
	lm.sample_count = 1000;
	lm.tot_steps = 500000;
	lm.learning_rate = 3 * 1e-1;

	vector<float> poly_preds_params(preds.size() * poly_rank);
	for (size_t j = 0; j < poly_rank; ++j)
		for (size_t i = 0; i < preds.size(); ++i)
			poly_preds_params[i * poly_rank + j] = (float)pow(preds[i], j + 1);

	lm.learn(poly_preds_params, probs, (int)preds.size(), poly_rank);
	vector<float> factors(poly_rank), mean_shifts(poly_rank);
	lm.get_normalization(mean_shifts, factors);

	//put normalizations inside params:
	params.resize(poly_rank + 1);
	params[0] = lm.model_params[0];
	for (size_t i = 1; i < params.size(); ++i) {
		params[i] = lm.model_params[i] / factors[i - 1];
		params[0] -= lm.model_params[i] * mean_shifts[i - 1] / factors[i - 1];
	}

	vector<double> converted((int)preds.size()), prior_score((int)preds.size());
	convert_scores_to_prob(preds, params, converted);
	int tot_pos = 0;
	for (size_t i = 0; i < y.size(); ++i)
		tot_pos += int(y[i] > 0);
	for (size_t i = 0; i < converted.size(); ++i)
		prior_score[i] = double(tot_pos) / y.size();

	double loss_model = _linear_loss_target_rmse(converted, probs, NULL);
	double loss_prior = _linear_loss_target_rmse(prior_score, probs, NULL);

	MLOG("Platt Scale prior=%2.5f. loss_model=%2.5f, loss_prior=%2.5f\n",
		double(tot_pos) / y.size(), loss_model, loss_prior);

	return 0;
}

int MedPredictor::predict(MedFeatures& ftrs_data) const {
	if (!model_features.empty()) {//test names of entered matrix:
		if (model_features.size() != ftrs_data.data.size())
			MTHROW_AND_ERR("Learned Feature model size was %d, request feature size for predict was %d\n",
			(int)model_features.size(), (int)ftrs_data.data.size());
		int feat_num = 0;
		for (auto it = ftrs_data.data.begin(); it != ftrs_data.data.end(); ++it)
			if (norm_feature_name(it->first) != norm_feature_name(model_features[feat_num++]))
				MTHROW_AND_ERR("Learned Features are the same. feat_num=%d. in learning was %s, now recieved %s\n",
				(int)model_features.size(), model_features[feat_num - 1].c_str(), it->first.c_str());
	}
	else if (features_count > 0 && features_count != ftrs_data.data.size())
		MTHROW_AND_ERR("Learned Feature model size was %d, request feature size for predict was %d\n",
			features_count, (int)ftrs_data.data.size());

	//split to bulks if too big to fit into memory:
	double max_samples = 0.95 *(INT_MAX / (double)ftrs_data.data.size());
	int samples_in_bucket = (int)max_samples;
	int smp_cnt = 0;
	if (!ftrs_data.data.empty())
		smp_cnt = (int)ftrs_data.data.begin()->second.size();

	if (smp_cnt <= samples_in_bucket) {
		// Build X
		MedMat<float> x;
		ftrs_data.get_as_matrix(x);

		// Predict
		vector<float> preds;
		if (predict(x, preds) < 0)
			return -1;

		int n = n_preds_per_sample();
		ftrs_data.samples.resize(preds.size() / n);
		for (int i = 0; i < x.nrows; i++) {
			ftrs_data.samples[i].prediction.resize(n);
			for (int j = 0; j < n; j++)
				ftrs_data.samples[i].prediction[j] = preds[i*n + j];
		}
	}
	else {
		MWARN("matrix is (%d X %zu) - too big to fit 4 bytes memory address - split into batches\n",
			smp_cnt, ftrs_data.data.size());

		vector<string> dummay_empty_sigs;
		vector<int> batch_ids(samples_in_bucket);
		int total_proc = 0;
		int n = n_preds_per_sample();
		if (ftrs_data.samples.size() < smp_cnt)
			ftrs_data.samples.resize(smp_cnt);
		MedProgress progress_pred("MedPredictor::predict", (int)ceil(double(ftrs_data.samples.size()) / samples_in_bucket), 30, 1);
		while (total_proc < ftrs_data.samples.size()) {
			//adjust batch size if needed:
			if (total_proc + samples_in_bucket > ftrs_data.samples.size()) {
				samples_in_bucket = (int)ftrs_data.samples.size() - total_proc;
				batch_ids.resize(samples_in_bucket);
			}
			for (int i = 0; i < samples_in_bucket; ++i)
				batch_ids[i] = total_proc + i; //continue for last id - total_proc

			// Build X
			MedMat<float> x;
			ftrs_data.get_as_matrix(x, dummay_empty_sigs, batch_ids);

			// Predict
			vector<float> preds;
			if (predict(x, preds) < 0)
				return -1;

			for (int i = 0; i < x.nrows; ++i) {
				ftrs_data.samples[total_proc + i].prediction.resize(n);
				for (int j = 0; j < n; ++j)
					ftrs_data.samples[total_proc + i].prediction[j] = preds[i*n + j];
			}

			total_proc += samples_in_bucket;
			progress_pred.update();
		}
	}

	return 0;
}

void MedPredictor::predict_single(const vector<float> &x, vector<float> &preds) const {
	MTHROW_AND_ERR("Error not implemented in %s\n", my_class_name().c_str());
}

void MedPredictor::predict_single(const vector<double> &x, vector<double> &preds) const {
	MTHROW_AND_ERR("Error not implemented in %s\n", my_class_name().c_str());
}

void MedPredictor::calc_feature_importance_shap(vector<float> &features_importance_scores, string &importance_type, const MedFeatures *features)
{
	MedMat<float> feat_mat, contribs_mat;
	if (features == NULL)
		MTHROW_AND_ERR("SHAP values feature importance requires features \n");

	features->get_as_matrix(feat_mat);
	calc_feature_contribs(feat_mat, contribs_mat);
	for (int j = 0; j < contribs_mat.ncols; ++j)
	{
		float col_sum = 0;

		for (int i = 0; i < contribs_mat.nrows; ++i)
		{
			col_sum += abs(contribs_mat.get(i, j));
		}
		features_importance_scores[j] = col_sum / (float)contribs_mat.nrows;
	}
}

void MedMicNet::prepare_predict_single() {
	//MWARN("Warning in MedMicNet::prepare_predict_single - no fast implementation provided\n");
	/*if (!is_prepared) {
		int N_tot_threads = omp_get_max_threads();
		model_per_thread.resize(N_tot_threads);
		for (size_t i = 0; i < model_per_thread.size(); ++i)
			model_per_thread[i] = mic;
		is_prepared = true;
	}*/
}

void MedMicNet::predict_single(const vector<float> &x, vector<float> &preds) const {
	//if (!is_prepared)
	//	MTHROW_AND_ERR("please call MedMicNet::prepare_predict_single()");

	//int n_th = omp_get_thread_num();
	//const micNet &threaded_net = model_per_thread[n_th];

	mic.predict_single(x, preds);
}

void convertXMat(const vector<vector<float>> x, MedMat<float> &xMat) {
	xMat.resize((int)x[0].size(), (int)x.size());
	for (size_t i = 0; i < x.size(); ++i)
	{
		vector<float> xx = x[i];
		for (size_t k = 0; k < xx.size(); ++k)
		{
			xMat((int)k, (int)i) = xx[k];
		}
	}
	xMat.missing_value = MED_MAT_MISSING_VALUE;
}

string medial::models::getParamsInfraModel(void *model) {
	MedPredictor *m = (MedPredictor *)model;
	MedQRFParams pr_qrf;
	MedLightGBMParams pr_lightGBM;
	MedXGBParams pr_xgb;
	map<string, string> empty_m;
	MedSpecificGroupModels *model_specific;
	MedSvm *svm;
	MedLinearModel *lm;
	MedGDLM *gdlm;
	MedLM *linear_m;
	char buff[2000];
	string l1_str, n_categ = "";
	string mono_str = "";

	switch (m->classifier_type) {
	case MODEL_QRF:
		pr_qrf = ((MedQRF *)model)->params;
		l1_str = to_string(pr_qrf.type);
		if (pr_qrf.type == QRF_BINARY_TREE)
			l1_str = "binary";
		else if (pr_qrf.type == QRF_REGRESSION_TREE)
			l1_str = "regression";
		else if (pr_qrf.type == QRF_CATEGORICAL_CHI2_TREE)
			l1_str = "categorial_chi2";
		else if (pr_qrf.type == QRF_CATEGORICAL_ENTROPY_TREE)
			l1_str = "categorial_entropy";
		if (pr_qrf.n_categ > 2)
			n_categ = " ;n_categ=" + to_string(pr_qrf.n_categ);
		snprintf(buff, 2000, "%s: ntrees=%d; maxq=%d; min_node=%d; ntry=%d; spread=%2.3f; type=%s; max_depth=%d; learn_nthreads=%d; predict_nthreads=%d; take_all_samples=%d%s",
			predictor_type_to_name[m->classifier_type].c_str(), pr_qrf.ntrees, pr_qrf.maxq, pr_qrf.min_node, pr_qrf.ntry, pr_qrf.spread,
			l1_str.c_str(), pr_qrf.max_depth, pr_qrf.learn_nthreads, pr_qrf.predict_nthreads, (int)pr_qrf.take_all_samples, n_categ.c_str());
		break;
	case MODEL_LIGHTGBM:
		pr_lightGBM = ((MedLightGBM *)model)->params;
		snprintf(buff, 2000, "%s: %s",
			predictor_type_to_name[m->classifier_type].c_str(), pr_lightGBM.user_params.c_str());
		break;
	case MODEL_XGB:
		pr_xgb = ((MedXGB *)model)->params;
		if (!pr_xgb.monotone_constraints.empty())
			mono_str = "monotone_constraints=" + pr_xgb.monotone_constraints;
		snprintf(buff, 2000, "%s: tree_method=%s; booster=%s; objective=%s; eta=%2.3f; alpha=%2.3f; lambda=%2.3f; gamma=%2.3f; max_depth=%d; colsample_bytree=%2.3f; colsample_bylevel=%2.3f; min_child_weight=%d; num_round=%d; subsample=%2.3f; %s",
			predictor_type_to_name[m->classifier_type].c_str(), pr_xgb.tree_method.c_str(), pr_xgb.booster.c_str(), pr_xgb.objective.c_str(), pr_xgb.eta,
			pr_xgb.alpha, pr_xgb.lambda, pr_xgb.gamma, pr_xgb.max_depth, pr_xgb.colsample_bytree, pr_xgb.colsample_bylevel, pr_xgb.min_child_weight, pr_xgb.num_round, pr_xgb.subsample, mono_str.c_str());
		break;
	case MODEL_SPECIFIC_GROUPS_MODELS:
		model_specific = ((MedSpecificGroupModels *)model);
		snprintf(buff, 2000, "%s: model=%s x %d",
			predictor_type_to_name[m->classifier_type].c_str(), predictor_type_to_name[model_specific->get_model(0)->classifier_type].c_str(),
			model_specific->model_cnt());
		break;
	case MODEL_SVM:
		svm = ((MedSvm *)model);
		snprintf(buff, 2000, "%s: kernal_type=%d; C=%2.3f; coef0=%2.3f; degree=%d; gamma=%2.3f; eps=%2.3f",
			predictor_type_to_name[m->classifier_type].c_str(), svm->params.kernel_type
			, svm->params.C, svm->params.coef0, svm->params.degree, svm->params.gamma, svm->params.eps);
		break;
	case MODEL_LINEAR_SGD:
		lm = ((MedLinearModel *)model);
		l1_str = lm->norm_l1 ? "(L1)" : "(L2)";
		snprintf(buff, 2000, "%s: name=%s; num_params=%d; block_num=%2.3f%s; learning_rate=%2.3f; sample_count=%d; tot_steps=%d",
			predictor_type_to_name[m->classifier_type].c_str(), lm->model_name.c_str(),
			(int)lm->model_params.size(), lm->block_num, l1_str.c_str(),
			lm->learning_rate, lm->sample_count, lm->tot_steps);
		break;
	case MODEL_GD_LINEAR:
		gdlm = (MedGDLM*)model;
		snprintf(buff, 2000, "%s: method=%s; batch_size=%d; l_lasso=%2.3f; l_ridge=%2.3f; rate=%2.3f; rate_decay=%2.3f; momentum=%2.3f; stop_at_err=%2.3f; max_iter=%d",
			predictor_type_to_name[m->classifier_type].c_str(), gdlm->params.method.c_str(), gdlm->params.batch_size,
			gdlm->params.l_lasso, gdlm->params.l_ridge, gdlm->params.rate, gdlm->params.rate_decay,
			gdlm->params.momentum, gdlm->params.stop_at_err, gdlm->params.max_iter);
		break;
	case MODEL_LINEAR_MODEL:
		linear_m = (MedLM*)model;
		snprintf(buff, 2000, "%s: niter=%d; eiter=%f; rfactor=%f",
			predictor_type_to_name[m->classifier_type].c_str(), linear_m->params.niter,
			linear_m->params.eiter, linear_m->params.rfactor);

		break;
	default:
		MTHROW_AND_ERR("Unsupported Type init for model %s:%d (getParams)\n", m->my_class_name().c_str(), m->classifier_type);
	}

	return string(buff);
}

void *medial::models::copyInfraModel(void *model, bool delete_old) {
	MedPredictor *m = (MedPredictor *)model;
	MedQRFParams pr_qrf;
	MedLightGBMParams pr_lightGBM;
	MedXGBParams pr_xgb;
	map<string, string> empty_m;
	MedSpecificGroupModels *model_specific;
	MedSvm *svm;
	MedLinearModel *lm;
	MedGDLM *gdlm;
	void *newM;
	vector<unsigned char> blob;

	switch (m->classifier_type) {
	case MODEL_QRF:
		pr_qrf = MedQRFParams(((MedQRF *)model)->params);
		newM = new MedQRF(pr_qrf);
		((MedQRF *)newM)->qf = QRF_Forest(); //Erase forest
		if (delete_old)
			delete ((MedQRF *)model);
		break;
	case MODEL_LIGHTGBM:
		pr_lightGBM = MedLightGBMParams(((MedLightGBM *)model)->params);
		if (delete_old)
			delete ((MedLightGBM *)model);
		newM = new MedLightGBM;
		((MedLightGBM *)newM)->params = pr_lightGBM;
		MedSerialize::initialization_text_to_map(pr_lightGBM.defaults + ";" + pr_lightGBM.user_params, empty_m);
		((MedLightGBM *)newM)->init(empty_m);
		break;
	case MODEL_XGB:
		pr_xgb = MedXGBParams(((MedXGB *)model)->params);
		if (delete_old)
			delete ((MedXGB *)model);
		newM = new MedXGB;
		((MedXGB *)newM)->params = pr_xgb;
		break;
	case MODEL_SPECIFIC_GROUPS_MODELS:
		model_specific = ((MedSpecificGroupModels *)model);
		newM = model_specific->clone();
		for (size_t i = 0; i < model_specific->model_cnt(); ++i)
			(*((MedQRF *)model_specific->get_model((int)i))).qf = QRF_Forest(); //to release memory

		break;
	case MODEL_SVM:
		svm = ((MedSvm *)model);
		newM = new MedSvm(svm->params);
		if (delete_old)
			delete (MedSvm *)model;
		break;
	case MODEL_LINEAR_SGD:
		lm = ((MedLinearModel *)model);
		newM = (MedLinearModel *)lm->clone();
		if (delete_old)
			delete (MedLinearModel *)model;
		break;
	case MODEL_GD_LINEAR:
		gdlm = (MedGDLM*)model;
		newM = new MedGDLM(gdlm->params);
		if (delete_old)
			delete (MedGDLM *)model;
		break;
	default:
		newM = MedPredictor::make_predictor(m->classifier_type);
		m->serialize_vec(blob);
		static_cast<MedPredictor *>(newM)->deserialize_vec(blob);
		//MTHROW_AND_ERR("Unsupported Type init for model %s:%d (copy)\n", m->my_class_name().c_str(), m->classifier_type);
	}

	return newM;
}

void medial::models::initInfraModel(void *&model) {
	void *newM = copyInfraModel(model, true);

	model = newM;
}

void medial::models::learnInfraModel(void *model, const vector<vector<float>> &xTrain, vector<float> &y, vector<float> &weights) {
	MedMat<float> xTrain_m;
	convertXMat(xTrain, xTrain_m);
	MedPredictor *m = (MedPredictor *)model;
	if (m->normalize_for_learn)
		xTrain_m.normalize();
	m->learn(xTrain_m, y, weights);
}

vector<float> medial::models::predictInfraModel(void *model, const vector<vector<float>> &xTest) {
	MedMat<float> xTest_m;
	convertXMat(xTest, xTest_m);
	MedPredictor *m = (MedPredictor *)model;
	if (m->normalize_for_predict)
		xTest_m.normalize();
	vector<float> preds;
	m->predict(xTest_m, preds);
	return preds;
}

void medial::models::get_pids_cv(MedPredictor *pred, MedFeatures &matrix, int nFolds,
	mt19937 &generator, vector<float> &preds) {
	int nSamples = (int)matrix.samples.size();
	unordered_map<int, int> pid_to_fold;
	preds.resize(nSamples);
	if (nFolds > 1) {
		uniform_int_distribution<> fold_dist(0, nFolds - 1);
		if (matrix.weights.empty()) {
			for (int i = 0; i < nSamples; i++)
				if (pid_to_fold.find(matrix.samples[i].id) == pid_to_fold.end())
					pid_to_fold[matrix.samples[i].id] = fold_dist(generator);
		}
		else {
			double tot_w = 0;
			for (size_t i = 0; i < matrix.weights.size(); ++i)
				tot_w += matrix.weights[i];
			tot_w = tot_w / nFolds + 1;
			unordered_map<int, double> pid_to_w;
			for (int i = 0; i < nSamples; i++)
				pid_to_w[matrix.samples[i].id] += matrix.weights[i];
			//split to folds untill each fold reaches tot_w
			vector<double> folds_w(nFolds);
			for (int i = 0; i < nSamples; i++)
				if (pid_to_fold.find(matrix.samples[i].id) == pid_to_fold.end()) {
					int propose = fold_dist(generator);
					while (folds_w[propose] >= tot_w)
						propose = fold_dist(generator);

					pid_to_fold[matrix.samples[i].id] = propose;
					folds_w[propose] += pid_to_w[matrix.samples[i].id];
				}
		}



		for (size_t iFold = 0; iFold < nFolds; ++iFold)
		{
			MedFeatures testMatrix, trainMatrix;
			medial::process::split_matrix(matrix, pid_to_fold, (int)iFold, trainMatrix, testMatrix);

			pred->learn(trainMatrix);
			pred->predict(testMatrix);

			int idx = 0;
			for (int i = 0; i < nSamples; i++)
				if (pid_to_fold[matrix.samples[i].id] == iFold)
					preds[i] = testMatrix.samples[idx++].prediction[0];
		}
	}
	else {
		MedFeatures testMatrix = matrix;
		pred->learn(testMatrix);
		pred->predict(testMatrix);

		for (int i = 0; i < nSamples; i++)
			preds[i] = testMatrix.samples[i].prediction[0];
	}

}

void medial::models::get_cv(MedPredictor *pred, MedFeatures &matrix, int nFolds,
	mt19937 &generator, vector<float> &preds) {
	int nSamples = (int)matrix.samples.size();
	vector<int> folds(nSamples);
	preds.resize(nSamples);

	if (nFolds > 1) {
		uniform_int_distribution<> fold_dist(0, nFolds - 1);
		if (matrix.weights.empty())
			for (int i = 0; i < nSamples; i++)
				folds[i] = fold_dist(generator);
		else {
			double tot_w = 0;
			for (size_t i = 0; i < matrix.weights.size(); ++i)
				tot_w += matrix.weights[i];
			tot_w = tot_w / nFolds + 1;
			//split to folds untill each fold reaches tot_w
			vector<double> folds_w(nFolds);
			for (int i = 0; i < nSamples; i++) {
				int propose = fold_dist(generator);
				while (folds_w[propose] >= tot_w)
					propose = fold_dist(generator);

				folds[i] = propose;
				folds_w[propose] += matrix.weights[i];
			}
		}

		for (size_t iFold = 0; iFold < nFolds; ++iFold)
		{
			MedFeatures testMatrix, trainMatrix;
			medial::process::split_matrix(matrix, folds, (int)iFold, trainMatrix, testMatrix);

			pred->learn(trainMatrix);
			pred->predict(testMatrix);

			int idx = 0;
			for (int i = 0; i < nSamples; i++)
				if (folds[i] == iFold)
					preds[i] = testMatrix.samples[idx++].prediction[0];
		}
	}
	else {
		MedFeatures testMatrix = matrix;
		pred->learn(testMatrix);
		pred->predict(testMatrix);

		for (int i = 0; i < nSamples; i++)
			preds[i] = testMatrix.samples[i].prediction[0];
	}

}

bool is_similar(float mean1, float lower1, float upper1, float std1,
	float mean2, float lower2, float upper2, float std2) {
	float simlar_range_ratio = (float)0.8;
	float similar_mean_ratio = (float)0.05;

	float min_range = min(upper1 - lower1, upper2 - lower2);
	float range = min(upper1, upper2) - max(lower1, lower2);
	if (range < 0)
		range = 0;
	if (min_range <= 0)
		range = 1;
	else
		range = range / min_range;
	float mean_diff_ratio = 0;
	float ratio_diff_range = abs(mean1);
	if (ratio_diff_range < std1) //if mean is closed to zero - take mean diff in std's ratio
		ratio_diff_range = std1;

	if (ratio_diff_range > 0)
		mean_diff_ratio = abs(mean1 - mean2) / ratio_diff_range;

	return (range >= simlar_range_ratio) && (mean_diff_ratio <= similar_mean_ratio)
		&& ((mean1 >= lower2 && mean1 <= upper2) ||
		(mean2 >= lower1 && mean2 <= upper1) ||
			(lower1 == lower2 && upper1 == upper2)); //means are inside CI of 95. like 2 stds in normal
}

void medial::process::compare_populations(const MedFeatures &population1, const MedFeatures &population2,
	const string &name1, const string &name2, const string &output_file,
	const string &predictor_type, const string &predictor_init, int nfolds, int max_learn) {
	if (population1.data.size() > population2.data.size())
		MTHROW_AND_ERR("population matrixes doesn't have same dimentions [%zu, %zu]\n",
			population1.data.size(), population2.data.size());
	vector<float> means1(population1.data.size()), means2(population1.data.size());
	vector<float> std1(population1.data.size()), std2(population1.data.size());
	vector<float> lower1(population1.data.size()), lower2(population1.data.size());
	vector<float> upper1(population1.data.size()), upper2(population1.data.size());
	vector<double> miss_cnt1(population1.data.size()), miss_cnt2(population1.data.size());

	vector<double> prc_vals = { 0.05, 0.95 };
	int feat_i = 0;
	int n_clean;
	for (auto it = population1.data.begin(); it != population1.data.end(); ++it)
	{
		if (population2.data.find(it->first) == population2.data.end())
			MTHROW_AND_ERR("population %s is missing feature %s\n",
				it->first.c_str(), name2.c_str());
		means1[feat_i] = medial::stats::mean(it->second, (float)MED_MAT_MISSING_VALUE, n_clean, &population1.weights);
		miss_cnt1[feat_i] = 100.0*((int)it->second.size() - n_clean) / ((double)it->second.size());
		means2[feat_i] = medial::stats::mean(population2.data.at(it->first), (float)MED_MAT_MISSING_VALUE, n_clean, &population2.weights);
		miss_cnt2[feat_i] = 100.0*((int)population2.data.at(it->first).size() - n_clean) / ((double)population2.data.at(it->first).size());
		std1[feat_i] = medial::stats::std(it->second, means1[feat_i], (float)MED_MAT_MISSING_VALUE, n_clean, &population1.weights);
		std2[feat_i] = medial::stats::std(population2.data.at(it->first), means2[feat_i], (float)MED_MAT_MISSING_VALUE, n_clean, &population2.weights);

		vector<float> prs;
		medial::process::prctils(it->second, prc_vals, prs, &population1.weights);
		lower1[feat_i] = prs[0];
		upper1[feat_i] = prs[1];
		prs.clear();
		medial::process::prctils(population2.data.at(it->first), prc_vals, prs, &population2.weights);
		lower2[feat_i] = prs[0];
		upper2[feat_i] = prs[1];
		++feat_i;
	}

	ofstream fw;
	char buffer_s[10000];
	if (!output_file.empty()) {
		fw.open(output_file);
		if (!fw.good())
			MTHROW_AND_ERR("Can't open %s for writing\n", output_file.c_str());
	}


	//try diffrentiate between populations:
	vector<float> features_scores(population1.data.size(), 0);
	MedFeatures new_data;
	vector<float> preds;
	vector<float> labels;
	if (!predictor_type.empty() && !predictor_init.empty()) {
		random_device rd;
		mt19937 gen(rd());
		new_data.attributes = population1.attributes;
		new_data.time_unit = population1.time_unit;
		new_data.samples.insert(new_data.samples.end(), population1.samples.begin(), population1.samples.end());
		new_data.samples.insert(new_data.samples.end(), population2.samples.begin(), population2.samples.end());
		//change outcome to be population label: is population 1?
		labels.resize(new_data.samples.size());
		for (size_t i = 0; i < new_data.samples.size(); ++i) {
			new_data.samples[i].outcome = i < population1.samples.size();
			labels[i] = new_data.samples[i].outcome;
		}
		new_data.init_pid_pos_len();
		if (!population1.weights.empty() || !population2.weights.empty()) {
			vector<float> temp_weights;
			const vector<float> *weights = &population1.weights;
			if (weights->empty()) {
				temp_weights.resize(population1.samples.size(), 1);
				weights = &temp_weights;
			}
			new_data.weights.insert(new_data.weights.end(), weights->begin(), weights->end());
			weights = &population2.weights;
			if (weights->empty()) {
				temp_weights.resize(population2.samples.size(), 1);
				weights = &temp_weights;
			}
			new_data.weights.insert(new_data.weights.end(), weights->begin(), weights->end());
		}
		for (auto it = population1.data.begin(); it != population1.data.end(); ++it)
		{
			new_data.data[it->first] = it->second;
			new_data.data[it->first].insert(new_data.data[it->first].end(),
				population2.data.at(it->first).begin(), population2.data.at(it->first).end());
		}

		//lets get auc on this problem:
		MedPredictor *predictor = MedPredictor::make_predictor(predictor_type, predictor_init);
		if (max_learn > 0 && new_data.samples.size() > max_learn) {
			double rt = double(max_learn) / new_data.samples.size();
			vector<int> sel;
			medial::process::down_sample(new_data, rt, false, &sel);
			medial::process::commit_selection(labels, sel);
		}
		//lets fix labels weight that cases will be less common
		medial::models::get_pids_cv(predictor, new_data, nfolds, gen, preds);
		try {
			predictor->calc_feature_importance(features_scores, "", NULL);
		}
		catch (exception &exp) {
			//handle not implemented calc_feature_importance
			features_scores.resize(population1.data.size());
			MERR("%s\n", exp.what());
		}
	}

	//print each feature dist in each population:
	snprintf(buffer_s, sizeof(buffer_s), "Comparing populations - %s population has %zu sampels, %s has %zu samples."
		" Features distributaions:\n", name1.c_str(), population1.samples.size(),
		name2.c_str(), population2.samples.size());

	if (!output_file.empty())
		fw << string(buffer_s);
	else
		MLOG("%s", string(buffer_s).c_str());

	feat_i = 0;
	int j = 0;
	for (auto it = population1.data.begin(); it != population1.data.end(); ++it) {
		bool res_sim = is_similar(means1[feat_i], lower1[feat_i], upper1[feat_i], std1[feat_i],
			means2[feat_i], lower2[feat_i], upper2[feat_i], std2[feat_i]);
		string desc_str = res_sim ? "GOOD" : "BAD";
		float ratio_diff_range = abs(means1[feat_i]);
		if (ratio_diff_range < std1[feat_i]) //if mean is closed to zero - take mean diff in std's ratio
			ratio_diff_range = std1[feat_i];
		snprintf(buffer_s, sizeof(buffer_s), "%s feature %s :: %s mean= %2.3f [ %2.3f - %2.3f ],std= %2.3f, miss_cnt=%2.3f%% | "
			"%s mean= %2.3f [ %2.3f - %2.3f ],std= %2.3f, miss_cnt=%2.3f%% | mean_diff_ratio=%2.3f%% | IMP %f\n", desc_str.c_str(), it->first.c_str(), name1.c_str(),
			means1[feat_i], lower1[feat_i], upper1[feat_i], std1[feat_i], miss_cnt1[feat_i], name2.c_str(),
			means2[feat_i], lower2[feat_i], upper2[feat_i], std2[feat_i], miss_cnt2[feat_i],
			100 * abs(means1[feat_i] - means2[feat_i]) / max(ratio_diff_range, (float)1e-10), features_scores[j++]);

		if (!output_file.empty())
			fw << string(buffer_s);
		else
			MLOG("%s", string(buffer_s).c_str());

		++feat_i;
	}

	//Mann-Whitney analysis on each column:
	snprintf(buffer_s, sizeof(buffer_s), "MANN_WHITNEY\tFEATURE_NAME\tMEAN_1\tSTD_1\tMISS_CNT_1\tMEAN_2\tSTD_2\tMISS_CNT_2\tP_VAL\tAUC\tFEATURE_IMPORTANCE\n");
	if (!output_file.empty())
		fw << string(buffer_s);
	else
		MLOG("%s", string(buffer_s).c_str());
	feat_i = 0;
	for (auto it = population1.data.begin(); it != population1.data.end(); ++it)
	{
		vector<float> test_auc, data_vec;
		const vector<float> &v1 = it->second;
		const vector<float> &v2 = population2.data.at(it->first);
		test_auc.reserve(v1.size() + v2.size());
		data_vec.reserve(v1.size() + v2.size());
		int control_cnt = 0, cases_cnt = 0;
		for (size_t i = 0; i < v1.size(); ++i)
			if (v1[i] != MED_MAT_MISSING_VALUE) {
				data_vec.push_back(v1[i]);
				test_auc.push_back(0);
				++control_cnt;
			}
		for (size_t i = 0; i < v2.size(); ++i)
			if (v2[i] != MED_MAT_MISSING_VALUE) {
				data_vec.push_back(v2[i]);
				test_auc.push_back(1);
				++cases_cnt;
			}

		if (control_cnt < 10 || cases_cnt < 10) {
			snprintf(buffer_s, sizeof(buffer_s), "MANN_WHITNEY\t%s\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\tMISSING\tMISSING\n",
				it->first.c_str(), means1[feat_i], std1[feat_i], miss_cnt1[feat_i], 
				means2[feat_i], std2[feat_i], miss_cnt2[feat_i], features_scores[feat_i]);
			if (!output_file.empty())
				fw << string(buffer_s);
			else
				MLOG("%s", string(buffer_s).c_str());
			++feat_i;
			continue;
		}
		float auc = medial::performance::auc_q(data_vec, test_auc);
		float p_val = 1 - 2 * abs(auc - 0.5);

		snprintf(buffer_s, sizeof(buffer_s), "MANN_WHITNEY\t%s\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\n",
			it->first.c_str(), means1[feat_i], std1[feat_i], miss_cnt1[feat_i], means2[feat_i], std2[feat_i], miss_cnt2[feat_i], p_val, auc, features_scores[feat_i]);

		if (!output_file.empty())
			fw << string(buffer_s);
		else
			MLOG("%s", string(buffer_s).c_str());
		++feat_i;
	}

	if (!predictor_type.empty() && !predictor_init.empty()) {
		float auc = medial::performance::auc_q(preds, labels, &new_data.weights);
		snprintf(buffer_s, sizeof(buffer_s),
			"predictor AUC with CV to diffrentiate between populations is %2.3f\n", auc);

		MLOG("%s", string(buffer_s).c_str());
		if (!output_file.empty())
			fw << string(buffer_s);
	}

	if (!output_file.empty())
		fw.close();
}