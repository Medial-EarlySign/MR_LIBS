#include "MPBootstrap.h"
#include "MedStat/MedStat/MedBootstrap.h"

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_APP
#define LOCAL_LEVEL LOG_DEF_LEVEL

/*************************************************/
MPStringBtResultMap::MPStringBtResultMap() { o = new std::map<std::string, std::map<std::string, float> >(); }
MPStringBtResultMap::MPStringBtResultMap(const MPStringBtResultMap& other)
{
	o_owned = other.o_owned;
	if (!other.o_owned) {
		o = other.o;
	}
	else {
		o = new map<string, std::map<std::string, float> >();
		*o = *other.o;
	}
};
MPStringBtResultMap::MPStringBtResultMap(std::map<std::string, std::map<std::string, float> >* ptr) { o_owned = true; o = ptr; };
MPStringBtResultMap::~MPStringBtResultMap() { if (o_owned) delete o; };
int MPStringBtResultMap::__len__() { return (int)o->size(); };
MPStringFloatMapAdaptor MPStringBtResultMap::__getitem__(const std::string& i) { return MPStringFloatMapAdaptor(&o->operator[](i)); };
void MPStringBtResultMap::__setitem__(const std::string &key, MPStringFloatMapAdaptor& val) {
	o->operator[](key) = *val.o;
}
std::vector<std::string> MPStringBtResultMap::keys() {
	std::vector<string> ret;
	ret.reserve(o->size());
	for (const auto& rec : *o) ret.push_back(rec.first);
	return ret;
}

MPStringBtResultMap& MPStringBtResultMap::operator=(const MPStringBtResultMap& other)
{
	if (&other == this)
		return *this;
	o_owned = other.o_owned;
	if (!o_owned) {
		o = other.o;
	}
	else {
		o = new std::map<std::string, std::map<std::string, float> >();
		*o = *(other.o);
	}
	return *this;
}
/*************************************************/

MPBootstrap::MPBootstrap() { o = new MedBootstrap(); };
MPBootstrap::~MPBootstrap() { delete o; };

int MPBootstrap::MEDPY_GET_sample_per_pid() { return o->sample_per_pid; };
void MPBootstrap::MEDPY_SET_sample_per_pid(int _sample_per_pid) { o->sample_per_pid = _sample_per_pid; };
int MPBootstrap::MEDPY_GET_nbootstrap() { return o->loopCnt; };
void MPBootstrap::MEDPY_SET_nbootstrap(int _nbootstrap) { o->loopCnt = _nbootstrap; }
int MPBootstrap::MEDPY_GET_ROC_score_min_samples() { return o->roc_Params.score_min_samples; };
void MPBootstrap::MEDPY_SET_ROC_score_min_samples(int _ROC_score_min_samples) { o->roc_Params.score_min_samples = _ROC_score_min_samples; }
int MPBootstrap::MEDPY_GET_ROC_score_bins() { return o->roc_Params.score_bins; };
void MPBootstrap::MEDPY_SET_ROC_score_bins(int _ROC_score_bins) { o->roc_Params.score_bins = _ROC_score_bins; }

float MPBootstrap::MEDPY_GET_ROC_max_diff_working_point() { return o->roc_Params.max_diff_working_point; };
void MPBootstrap::MEDPY_SET_ROC_max_diff_working_point(float _ROC_max_diff_working_point) { o->roc_Params.max_diff_working_point = _ROC_max_diff_working_point; }
float MPBootstrap::MEDPY_GET_ROC_score_resolution() { return o->roc_Params.score_resolution; };
void MPBootstrap::MEDPY_SET_ROC_score_resolution(float _ROC_score_resolution) { o->roc_Params.score_resolution = _ROC_score_resolution; }
bool MPBootstrap::MEDPY_GET_ROC_use_score_working_points() { return o->roc_Params.use_score_working_points; };
void MPBootstrap::MEDPY_SET_ROC_use_score_working_points(bool _ROC_use_score_working_points) { o->roc_Params.use_score_working_points = _ROC_use_score_working_points; }

std::vector<float> MPBootstrap::MEDPY_GET_ROC_working_point_FPR() { return o->roc_Params.working_point_FPR; };
void MPBootstrap::MEDPY_SET_ROC_working_point_FPR(const std::vector<float> &_ROC_working_point_FPR) { o->roc_Params.working_point_FPR = _ROC_working_point_FPR; }
std::vector<float> MPBootstrap::MEDPY_GET_ROC_working_point_PR() { return o->roc_Params.working_point_PR; };
void MPBootstrap::MEDPY_SET_ROC_working_point_PR(const std::vector<float> &_ROC_working_point_PR) { o->roc_Params.working_point_PR = _ROC_working_point_PR; }
std::vector<float> MPBootstrap::MEDPY_GET_ROC_working_point_SENS() { return o->roc_Params.working_point_SENS; };
void MPBootstrap::MEDPY_SET_ROC_working_point_SENS(const std::vector<float> &_ROC_working_point_SENS) { o->roc_Params.working_point_SENS = _ROC_working_point_SENS; }
std::vector<float> MPBootstrap::MEDPY_GET_ROC_working_point_Score() { return o->roc_Params.working_point_Score; };
void MPBootstrap::MEDPY_SET_ROC_working_point_Score(const std::vector<float> &_ROC_working_point_Score) { o->roc_Params.working_point_Score = _ROC_working_point_Score; }
std::vector<int> MPBootstrap::MEDPY_GET_ROC_working_point_TOPN() { return o->roc_Params.working_point_TOPN; };
void MPBootstrap::MEDPY_SET_ROC_working_point_TOPN(const std::vector<int> &_ROC_working_point_TOPN) { o->roc_Params.working_point_TOPN = _ROC_working_point_TOPN; }

void MPBootstrap::parse_cohort_line(const string &line) { o->parse_cohort_line(line); }

void MPBootstrap::get_cohort_from_arg(const string &single_cohort) { o->get_cohort_from_arg(single_cohort); }

void MPBootstrap::parse_cohort_file(const string &cohorts_path) { o->parse_cohort_file(cohorts_path); }

void MPBootstrap::init_from_str(const string &str) { o->init_from_string(str); }

void get_cohort_params_use(const map<string, vector<Filter_Param>> &filter_cohort, vector<string> &param_use) {
	param_use.clear();
	unordered_set<string> con_set;
	for (auto ii = filter_cohort.begin(); ii != filter_cohort.end(); ++ii)
		for (size_t i = 0; i < ii->second.size(); ++i)
			con_set.insert(ii->second[i].param_name);
	param_use.insert(param_use.end(), con_set.begin(), con_set.end());
}

MPStringBtResultMap MPBootstrap::bootstrap_cohort(MPSamples *samples, const string &rep_path, const string &json_model,
	const string &cohorts_path) {
	
	if (!cohorts_path.empty()) {
		o->filter_cohort.clear();
		o->parse_cohort_file(cohorts_path);
	}

	MedModel model;
	if (!json_model.empty())
		model.init_from_json_file(json_model);
	else {
		model.add_age();
		model.add_gender();
	}

	vector<int> pids;
	samples->o->get_ids(pids);

	MedPidRepository rep;
	MedFeatures empty_data;
    MedFeatures *features_data = &empty_data;
	if (!rep_path.empty()) { 
		model.load_repository(rep_path, pids, rep, true);

		if (model.learn(rep, samples->o, MED_MDL_LEARN_REP_PROCESSORS, MED_MDL_APPLY_FTR_PROCESSORS) < 0)
			MTHROW_AND_ERR("Model did not succeed to generate matrix for bootstrap filtering!\n");
		features_data = &model.features;
	}
	else {
		if (!json_model.empty() || !cohorts_path.empty())
			MTHROW_AND_ERR("can't pass json_model,cohorts and no repository\n");
		// Generate dummy features in empty_data
		samples->o->export_to_sample_vec(empty_data.samples);
	}

	if (samples->o->idSamples.size() > 0 && samples->o->idSamples[0].samples.size() >0 &&
        samples->o->idSamples[0].samples[0].attributes.find("weight") != samples->o->idSamples[0].samples[0].attributes.end()) {
			vector<float> weights;
			samples->o->get_attr_values("weight",weights);
			MLOG("Found weights\n");
			features_data->weights = weights; //store weights
	}

	std::map<std::string, std::map<std::string, float> > *res = new std::map<std::string, std::map<std::string, float> >();
	*res = o->bootstrap(*features_data);

	return MPStringBtResultMap(res);
}

MPStringFloatMapAdaptor MPBootstrap::_bootstrap(const std::vector<float> &preds, const std::vector<float> &labels) {
	if (preds.size() != labels.size()) {
		throw runtime_error(string("vectors are not in the same size: preds.size=") +
			std::to_string(preds.size()) + string(" labels.size=") + std::to_string(labels.size()));
	}
	MedSamples samples;
	samples.idSamples.resize(preds.size());
	for (int i = 0; i < (int)preds.size(); ++i)
	{
		samples.idSamples[i].id = i;
		samples.idSamples[i].samples.resize(1);
		MedSample &s = samples.idSamples[i].samples[0];
		s.id = i;
		s.outcome = labels[i];
		s.prediction = { preds[i] };
		s.outcomeTime = 20000101;
		s.time = s.outcomeTime;
	}

	std::map<std::string, std::vector<float> > additional_data;
	std::map<std::string, float>  *res = new std::map<std::string, float>();

	o->filter_cohort.clear();
	o->filter_cohort["All"] = {};
	*res = o->bootstrap(samples, additional_data)["All"];

	return MPStringFloatMapAdaptor(res);
}

map<string, float> calc_auc_per_pid(Lazy_Iterator *iterator, int thread_num,
	Measurement_Params *function_params) {
	map<string, float> res;

	unordered_map<int, pair<vector<float>, vector<float>>> pid_to_pred_label;
	unordered_map<int, int> pid_to_case_count, pid_to_control_count;
	float y, pred, pid;
	while (iterator->fetch_next_external(thread_num, y, pred, pid)) {
		pair<vector<float>, vector<float>> &pid_vec = pid_to_pred_label[pid];
		pid_vec.first.push_back(pred);
		pid_vec.second.push_back(y);
		if (y > 0)
			++pid_to_case_count[pid];
		else
			++pid_to_control_count[pid];
	}
	res["MACRO_AUC_PID"] = MED_MAT_MISSING_VALUE;
	res["MICRO_AUC_PID"] = MED_MAT_MISSING_VALUE;

	//Calc global AUC per pid and mean AUC per pid (MICRO/MACRO)
	//Macro = mean of AUC per pid
	//Micro - global auc for all pairs
	double macro_auc = 0, micro_auc = 0;
	int n = 0;
	unsigned long long int n_micto = 0;
	for (const auto &it : pid_to_pred_label)
		if (pid_to_case_count[it.first] > 0 && pid_to_control_count[it.first] > 0) {
			const pair<vector<float>, vector<float>> &pid_vec = it.second;
			float auc = medial::performance::auc_q(pid_vec.first, pid_vec.second);
			macro_auc += auc;
			n += 1;
			//Test all couples of cases-controls - It's the auc - multiply by num of cases*controls:
			unsigned long long int exp_count = static_cast<unsigned long long int>(pid_to_case_count[it.first]) * static_cast<unsigned long long int>(pid_to_control_count[it.first]);
			n_micto += exp_count;
			micro_auc += static_cast<double>(exp_count) * static_cast<double>(auc);
		}
	if (n > 0) {
		macro_auc /= n;
		res["MACRO_AUC_PID"] = macro_auc;
	}
	if (n_micto > 0) {
		micro_auc /= n_micto;
		res["MICRO_AUC_PID"] = micro_auc;
	}
	res["MACRO_NEXP"] = n;
	res["MICRO_NEXP"] = n_micto;

	return res;
}

MPStringFloatMapAdaptor MPBootstrap::_bootstrap_pid(const std::vector<float> &pids, const std::vector<float> &preds, const std::vector<float> &labels) {
	if (preds.size() != labels.size()) {
		throw runtime_error(string("vectors are not in the same size: preds.size=") +
			std::to_string(preds.size()) + string(" labels.size=") + std::to_string(labels.size()));
	}
	if (pids.size() != labels.size()) {
		throw runtime_error(string("vectors are not in the same size: pids.size=") +
			std::to_string(pids.size()) + string(" labels.size=") + std::to_string(labels.size()));
	}

	o->filter_cohort.clear();
	o->filter_cohort["All"] = {};

	MedFeatures bt_features;
	bt_features.samples.resize(pids.size());
	for (int i = 0; i < (int)preds.size(); ++i)
	{
		MedSample &s = bt_features.samples[i];
		s.id = pids[i];
		s.outcome = labels[i];
		s.prediction = { preds[i] };
		s.outcomeTime = 20000101;
		s.time = s.outcomeTime;
	}
	bt_features.init_pid_pos_len();

	std::map<std::string, float> *res = new std::map<std::string, float>();

	*res = o->bootstrap(bt_features)["All"];

	//Calc AUC Macro, Micro:
	vector<pair<MeasurementFunctions, Measurement_Params *>> backup_m = o->measurements_with_params;
	o->measurements_with_params.clear();
	o->measurements_with_params.push_back(pair<MeasurementFunctions, Measurement_Params *>(calc_auc_per_pid, NULL));
	//store pid in weight variable!
	bt_features.weights.resize(bt_features.samples.size());
	for (size_t i = 0; i < bt_features.samples.size(); ++i)
		bt_features.weights[i] = bt_features.samples[i].id;

	int before = o->sample_per_pid;
	o->sample_per_pid = 0;//take all
	std::map<std::string, float> extra_res = o->bootstrap(bt_features)["All"];
	o->sample_per_pid = before;
	o->measurements_with_params = backup_m;
	//Populate res:
	for (const auto &jt : extra_res)
		res->operator[](jt.first) = jt.second;

	return MPStringFloatMapAdaptor(res);
}


void MPStringBtResultMap::to_file(const std::string &file_path) {
	write_pivot_bootstrap_results(file_path, *o);
}