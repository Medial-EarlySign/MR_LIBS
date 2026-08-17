#define _CRT_SECURE_NO_WARNINGS

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_FEATCLEANER
#define LOCAL_LEVEL	LOG_DEF_LEVEL

#include "FeatureProcess.h"
#include "DoCalcFeatProcessor.h"
#include "PredictorImputer.h"
#include "ResampleWithMissingProcessor.h"
#include "DuplicateProcessor.h"
#include <omp.h>
#include <algorithm>
#include <regex>

//=======================================================================================
// Feature Processors
//=======================================================================================
// Processor types
FeatureProcessorTypes feature_processor_name_to_type(const string& processor_name)
{
	if (processor_name == "multi_processor" || processor_name == "multi")
		return FTR_PROCESS_MULTI;
	else if (processor_name == "basic_outlier_cleaner" || processor_name == "basic_cleaner" || processor_name == "basic_cln")
		return FTR_PROCESS_BASIC_OUTLIER_CLEANER;
	else if (processor_name == "normalizer")
		return FTR_PROCESS_NORMALIZER;
	else if (processor_name == "imputer")
		return FTR_PROCESS_IMPUTER;
	else if (processor_name == "iterative_imputer")
		return FTR_PROCESS_ITERATIVE_IMPUTER;
	else if (processor_name == "do_calc")
		return FTR_PROCESS_DO_CALC;
	else if (processor_name == "univariate_selector")
		return FTR_PROCESS_UNIVARIATE_SELECTOR;
	else if (processor_name == "mrmr" || processor_name == "mrmr_selector")
		return FTR_PROCESSOR_MRMR_SELECTOR;
	else if (processor_name == "lasso")
		return FTR_PROCESSOR_LASSO_SELECTOR;
	else if (processor_name == "remove_deg")
		return FTR_PROCESS_REMOVE_DGNRT_FTRS;
	else if (processor_name == "tags_selector")
		return FTR_PROCESSOR_TAGS_SELECTOR;
	else if (processor_name == "importance_selector")
		return FTR_PROCESSOR_IMPORTANCE_SELECTOR;
	else if (processor_name == "iterative_selector")
		return FTR_PROCESSOR_ITERATIVE_SELECTOR;
	else if (processor_name == "pca")
		return FTR_PROCESS_ENCODER_PCA;
	else if (processor_name == "one_hot")
		return FTR_PROCESS_ONE_HOT;
	else if (processor_name == "get_prob")
		return FTR_PROCESS_GET_PROB;
	else if (processor_name == "predictor_imputer")
		return FTR_PROCESS_PREDICTOR_IMPUTER;
	else if (processor_name == "multiplier")
		return FTR_PROCESS_MULTIPLIER;
	else if (processor_name == "resample_with_missing")
		return FTR_PROCESS_RESAMPLE_WITH_MISSING;
	else if (processor_name == "duplicate")
		return FTR_PROCESS_DUPLICATE;
	else if (processor_name == "missing_indicator")
		return FTR_PROCESS_MISSING_INDICATOR;
	else if (processor_name == "binning")
		return FTR_PROCESS_BINNING;
	else
		MTHROW_AND_ERR("feature_processor_name_to_type got unknown processor_name [%s]\n", processor_name.c_str());
}

// Initialization
//.......................................................................................
FeatureProcessor* FeatureProcessor::make_processor(string processor_name) {

	return make_processor(feature_processor_name_to_type(processor_name));
}

//.......................................................................................
FeatureProcessor * FeatureProcessor::make_processor(string processor_name, string init_string) {

	FeatureProcessorTypes type = feature_processor_name_to_type(processor_name);
	return make_processor(type, init_string);
}

//.......................................................................................
void *FeatureProcessor::new_polymorphic(string dname)
{
	CONDITIONAL_NEW_CLASS(dname, MultiFeatureProcessor);
	CONDITIONAL_NEW_CLASS(dname, FeatureBasicOutlierCleaner);
	CONDITIONAL_NEW_CLASS(dname, FeatureNormalizer);
	CONDITIONAL_NEW_CLASS(dname, FeatureImputer);
	CONDITIONAL_NEW_CLASS(dname, FeatureIterativeImputer);
	CONDITIONAL_NEW_CLASS(dname, DoCalcFeatProcessor);
	CONDITIONAL_NEW_CLASS(dname, UnivariateFeatureSelector);
	CONDITIONAL_NEW_CLASS(dname, MRMRFeatureSelector);
	CONDITIONAL_NEW_CLASS(dname, LassoSelector);
	CONDITIONAL_NEW_CLASS(dname, DgnrtFeatureRemvoer);
	CONDITIONAL_NEW_CLASS(dname, FeaturePCA);
	CONDITIONAL_NEW_CLASS(dname, TagFeatureSelector);
	CONDITIONAL_NEW_CLASS(dname, ImportanceFeatureSelector);
	CONDITIONAL_NEW_CLASS(dname, IterativeFeatureSelector);
	CONDITIONAL_NEW_CLASS(dname, OneHotFeatProcessor);
	CONDITIONAL_NEW_CLASS(dname, GetProbFeatProcessor);
	CONDITIONAL_NEW_CLASS(dname, PredictorImputer);
	CONDITIONAL_NEW_CLASS(dname, MultiplierProcessor);
	CONDITIONAL_NEW_CLASS(dname, ResampleMissingProcessor);
	CONDITIONAL_NEW_CLASS(dname, DuplicateProcessor);
	CONDITIONAL_NEW_CLASS(dname, MissingIndicatorProcessor);
	CONDITIONAL_NEW_CLASS(dname, BinningFeatProcessor);

	MTHROW_AND_ERR("Warning in FeatureProcessor::new_polymorphic - Unsupported class %s\n", dname.c_str());
	return NULL;
}

//.......................................................................................
FeatureProcessor * FeatureProcessor::make_processor(FeatureProcessorTypes processor_type) {

	if (processor_type == FTR_PROCESS_MULTI)
		return new MultiFeatureProcessor;
	else if (processor_type == FTR_PROCESS_BASIC_OUTLIER_CLEANER)
		return new FeatureBasicOutlierCleaner;
	else if (processor_type == FTR_PROCESS_NORMALIZER)
		return new FeatureNormalizer;
	else if (processor_type == FTR_PROCESS_IMPUTER)
		return new FeatureImputer;
	else if (processor_type == FTR_PROCESS_ITERATIVE_IMPUTER)
		return new FeatureIterativeImputer;
	else if (processor_type == FTR_PROCESS_DO_CALC)
		return new DoCalcFeatProcessor;
	else if (processor_type == FTR_PROCESS_UNIVARIATE_SELECTOR)
		return new UnivariateFeatureSelector;
	else if (processor_type == FTR_PROCESSOR_MRMR_SELECTOR)
		return new MRMRFeatureSelector;
	else if (processor_type == FTR_PROCESSOR_LASSO_SELECTOR)
		return new LassoSelector;
	else if (processor_type == FTR_PROCESS_REMOVE_DGNRT_FTRS)
		return new DgnrtFeatureRemvoer;
	else if (processor_type == FTR_PROCESS_ENCODER_PCA)
		return new FeaturePCA;
	else if (processor_type == FTR_PROCESSOR_TAGS_SELECTOR)
		return new TagFeatureSelector;
	else if (processor_type == FTR_PROCESSOR_IMPORTANCE_SELECTOR)
		return new ImportanceFeatureSelector;
	else if (processor_type == FTR_PROCESSOR_ITERATIVE_SELECTOR)
		return new IterativeFeatureSelector;
	else if (processor_type == FTR_PROCESS_ONE_HOT)
		return new OneHotFeatProcessor;
	else if (processor_type == FTR_PROCESS_GET_PROB)
		return new GetProbFeatProcessor;
	else if (processor_type == FTR_PROCESS_PREDICTOR_IMPUTER)
		return new PredictorImputer;
	else if (processor_type == FTR_PROCESS_MULTIPLIER)
		return new MultiplierProcessor;
	else if (processor_type == FTR_PROCESS_RESAMPLE_WITH_MISSING)
		return new ResampleMissingProcessor;
	else if (processor_type == FTR_PROCESS_DUPLICATE)
		return new DuplicateProcessor;
	else if (processor_type == FTR_PROCESS_MISSING_INDICATOR)
		return new MissingIndicatorProcessor;
	else if (processor_type == FTR_PROCESS_BINNING)
		return new BinningFeatProcessor;
	else
		MTHROW_AND_ERR("make_processor got unknown processor type [%d]\n", processor_type);

}

//.......................................................................................
FeatureProcessor * FeatureProcessor::make_processor(FeatureProcessorTypes processor_type, string init_string) {

	FeatureProcessor *newProcessor = make_processor(processor_type);
	if (newProcessor->init_from_string(init_string) < 0)
		MTHROW_AND_ERR("Cannot init FeatureProcessor of type %d with init string \'%s\'\n", processor_type, init_string.c_str());
	return newProcessor;
}

//.......................................................................................
int FeatureProcessor::learn(MedFeatures& features) {

	// All Ids - mark as an empty set
	unordered_set<int> temp;
	return Learn(features, temp);

}

//.......................................................................................
int FeatureProcessor::apply(MedFeatures& features, bool learning) {

	// All Ids - mark as an empty set
	unordered_set<int> temp;
	return _apply(features, temp, learning);
}

//.......................................................................................
int FeatureProcessor::apply(MedFeatures& features, unordered_set<int>& ids, bool learning) {
	return _apply(features, ids, learning);
}

//.......................................................................................
int FeatureProcessor::apply(MedFeatures& features, unordered_set<string>& req_features, bool learning) {

	// All Ids - mark as an empty set
	unordered_set<int> temp;
	return _conditional_apply(features, temp, req_features, learning);
}

//.......................................................................................
int FeatureProcessor::apply(MedFeatures& features, unordered_set<int>& ids, unordered_set<string>& req_features, bool learning) {
	return _conditional_apply(features, ids, req_features, learning);
}

//.......................................................................................
int FeatureProcessor::_conditional_apply(MedFeatures& features, unordered_set<int>& ids, unordered_set<string>& req_features, bool learning) {

	if (are_features_affected(req_features))
		return _apply(features, ids, learning);
	return 0;
}

//.......................................................................................
int FeatureProcessor::_apply(MedFeatures& features, unordered_set<int>& ids) {
	MTHROW_AND_ERR("FeatureProcessor Apply must be implemented");
}

//.......................................................................................

string FeatureProcessor::resolve_feature_name(MedFeatures& features, string substr) {
	string res;
#pragma omp critical
	{
		//resolve_feature_name - access features.data names in not thread safe manner
		res = features.resolve_name(substr);
	}
	return res;

}

// (De)Serialize
//.......................................................................................
size_t FeatureProcessor::get_processor_size() {
	return sizeof(processor_type) + get_size();
}

//.......................................................................................
size_t FeatureProcessor::processor_serialize(unsigned char *blob) {

	size_t ptr = 0;
	if (processor_type == FTR_PROCESS_LAST)
		MTHROW_AND_ERR("programmer error: trying to serialize a feature_processor with an undefined processor_type, must define it in init_defaults and call from ctor\n");
	memcpy(blob + ptr, &processor_type, sizeof(FeatureProcessorTypes)); ptr += sizeof(FeatureProcessorTypes);
	ptr += serialize(blob + ptr);

	return ptr;
}


//.......................................................................................
void FeatureProcessor::dprint(const string &pref, int fp_flag)
{
	if (fp_flag > 0) {
		MLOG("%s :: FP type %d(%s) : feature_name %s \n", pref.c_str(), processor_type, my_class_name().c_str(), feature_name.c_str());
	}
}
//=======================================================================================
// MultiFeatureProcessor
//=======================================================================================
int MultiFeatureProcessor::Learn(MedFeatures& features, unordered_set<int>& ids) {

	// Create processors
	if (processors.size() == 0 && duplicate) {
		vector<string> features_to_process;
		for (auto& rec : features.data) {
			string name = rec.first;
			if (tag == "" || features.tags[name].find(tag) != features.tags[name].end())
				features_to_process.push_back(name);
		}
		add_processors_set(members_type, features_to_process, init_string);
		string tp_name = "";
		if (!processors.empty())
			tp_name = processors.back()->my_class_name();
		MLOG("MultiFeautreProcessor - using duplicate to create %zu processors of type %d(%s)\n",
			features_to_process.size(), members_type, tp_name.c_str());
	}

	int RC = 0;

	// Allow nested parallelism if one processor
	if (processors.size() == 1) {
		use_parallel_learn = false;
		use_parallel_apply = false;
	}

	/*if (!use_parallel_learn && !processors.empty())
		MLOG("no threads for processor %s\n", processors.front()->my_class_name().c_str());*/

#pragma omp parallel for schedule(dynamic) if (use_parallel_learn && processors.size()>1)
	for (int j = 0; j < processors.size(); j++) {
		int rc = processors[j]->Learn(features, ids);
#pragma omp critical
		if (rc < 0) RC = -1;
	}

	return RC;
}

//.......................................................................................
int MultiFeatureProcessor::_apply(MedFeatures& features, unordered_set<int>& ids, bool learning) {
	int RC = 0;
	if (processors.size() == 1) {
		use_parallel_learn = false;
		use_parallel_apply = false;
	}
#pragma omp parallel for schedule(dynamic) if (use_parallel_apply && processors.size() > 1 && features.samples.size()>1)
	for (int j = 0; j < processors.size(); j++) {
		int rc = processors[j]->_apply(features, ids, learning);
#pragma omp critical
		if (rc < 0) RC = -1;
	}

	return RC;
}

//.......................................................................................
int MultiFeatureProcessor::_conditional_apply(MedFeatures& features, unordered_set<int>& ids, unordered_set<string>& req_features, bool learning) {

	int RC = 0;
#pragma omp parallel for schedule(dynamic) if (use_parallel_apply && processors.size() > 1 && features.samples.size()>1)
	for (int j = 0; j < processors.size(); j++) {
		int rc = processors[j]->_conditional_apply(features, ids, req_features, learning);
#pragma omp critical
		if (rc < 0) RC = -1;
	}

	return RC;
}

//.......................................................................................
void MultiFeatureProcessor::get_feature_names(vector<string>& all_feature_names) {
	all_feature_names.clear();
	for (auto p : processors) {
		vector<string> my_feature_names;
		p->get_feature_names(my_feature_names);
		all_feature_names.insert(all_feature_names.end(), my_feature_names.begin(), my_feature_names.end());
	}
}

// Add processors
//.......................................................................................
void  MultiFeatureProcessor::add_processors_set(FeatureProcessorTypes type, vector<string>& features) {

	for (string& feature : features) {
		FeatureProcessor *processor = FeatureProcessor::make_processor(type);
		processor->set_feature_name(feature);
		processors.push_back(processor);
	}
}

void  MultiFeatureProcessor::add_processors_set(FeatureProcessorTypes type, vector<string>& features, string init_string) {

	for (string& feature : features) {
		FeatureProcessor *processor = FeatureProcessor::make_processor(type, init_string);
		processor->set_feature_name(feature);
		processors.push_back(processor);
	}

}

// Filter according to a subset of features
//.......................................................................................
int  MultiFeatureProcessor::filter(unordered_set<string>& features) {

	int idx = 0;
	for (int i = 0; i < processors.size(); i++) {
		if (features.find(processors[i]->feature_name) != features.end())
			processors[idx++] = processors[i];
	}

	processors.resize(idx);
	return (int)processors.size();

}

// Copy
//.......................................................................................
void MultiFeatureProcessor::copy(FeatureProcessor *processor) {

	MultiFeatureProcessor *tempProcessor = dynamic_cast<MultiFeatureProcessor*>(processor);
	assert(tempProcessor != 0);

	*this = *tempProcessor;

	processors.resize(tempProcessor->processors.size());
	for (int i = 0; i < processors.size(); i++) {
		processors[i] = make_processor(tempProcessor->processors[i]->processor_type);
		processors[i]->copy(tempProcessor->processors[i]);
	}
}

// Clear
//.......................................................................................
void MultiFeatureProcessor::clear()
{
	for (auto pfp : processors) {
		if (pfp != NULL) {
			delete pfp;
			pfp = NULL;
		}
	}
	processors.clear();
}

/// check if a set of features is affected by the current processor
//.......................................................................................
bool MultiFeatureProcessor::are_features_affected(unordered_set<string>& out_req_features) {

	// Empty set == all features
	if (out_req_features.empty())
		return true;

	for (auto& processor : processors) {
		if (processor->are_features_affected(out_req_features))
			return true;
	}

	return false;
}

/// update sets of required as input according to set required as output to processor
//.......................................................................................
void MultiFeatureProcessor::update_req_features_vec(unordered_set<string>& out_req_features, unordered_set<string>& in_req_features) {

	in_req_features.clear();
	for (auto& processor : processors) {
		unordered_set<string> _in_req_features;
		processor->update_req_features_vec(out_req_features, _in_req_features);
		for (string ftr : _in_req_features)
			in_req_features.insert(ftr);
	}
}

// Init 
//.......................................................................................
int MultiFeatureProcessor::init(map<string, string>& mapper) {

	bool has_init_str = false, has_type = false;
	for (auto entry : mapper) {
		string field = entry.first;
		//! [MultiFeatureProcessor::init]
		if (field == "tag") tag = entry.second;
		if (field == "init_string") { init_string = entry.second;  has_init_str = true; }
		if (field == "members_type") { members_type = (FeatureProcessorTypes)med_stoi(entry.second); has_type = true; }
		if (field == "use_parallel_learn") use_parallel_learn = med_stoi(entry.second) > 0;
		if (field == "use_parallel_apply") use_parallel_apply = med_stoi(entry.second) > 0;
		//! [MultiFeatureProcessor::init]
	}
	if (has_init_str && has_type)
		duplicate = true;

	return 0;
}

string MultiFeatureProcessor::select_learn_matrix(const vector<string> &matrix_tags) const {
	if (processors.empty())
		return "";
	string res = processors.front()->select_learn_matrix(matrix_tags);
	for (size_t i = 1; i < processors.size(); ++i)
	{
		string r = processors[i]->select_learn_matrix(matrix_tags);
		if (r != res)
			MTHROW_AND_ERR("Error MultiFeatureProcessor::select_learn_matrix - can't select matrix. ascenders has different choises\n");
	}
	return res;
}

//.......................................................................................
void MultiFeatureProcessor::dprint(const string &pref, int fp_flag)
{
	if (fp_flag > 0) {
		MLOG("%s :: FP MULTI type %d : name %s \n", pref.c_str(), processor_type, feature_name.c_str());
		int ind = 0;
		for (auto& proc : processors) {
			proc->dprint("\t" + pref + "-in-MULTI[" + to_string(ind) + "]", fp_flag);
			++ind;
		}
	}
}

//=======================================================================================
// FeatureBasicOutlierCleaner
//=======================================================================================
// Init from map
//.......................................................................................
int FeatureBasicOutlierCleaner::init(map<string, string>& mapper)
{
	for (auto entry : mapper) {
		string field = entry.first;
		//! [FeatureBasicOutlierCleaner::init]
		if (field == "name") feature_name = entry.second;
		//! [FeatureBasicOutlierCleaner::init]
	}

	return MedValueCleaner::init(mapper);
}
//.......................................................................................
int FeatureBasicOutlierCleaner::Learn(MedFeatures& features, unordered_set<int>& ids) {

	if (params.type == VAL_CLNR_ITERATIVE)
		return iterativeLearn(features, ids);
	else if (params.type == VAL_CLNR_QUANTILE)
		return quantileLearn(features, ids);
	else {
		MERR("Unknown cleaning type %d\n", params.type);
		return -1;
	}
}

//.......................................................................................
int FeatureBasicOutlierCleaner::iterativeLearn(MedFeatures& features, unordered_set<int>& ids) {

	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);

	// Get all values
	vector<float> values;
	get_all_values(features, resolved_feature_name, ids, values, params.max_samples);

	// Get bounds
	int rc = get_iterative_min_max(values);
	if (num_samples_after_cleaning == 0)
		MWARN("FeatureBasicOutlierCleaner::iterativeLearn feature [%s] tried learning cleaning params from an empty vector\n", resolved_feature_name.c_str());
	return rc;
}

//.......................................................................................
int FeatureBasicOutlierCleaner::quantileLearn(MedFeatures& features, unordered_set<int>& ids) {

	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);

	// Get all values
	vector<float> values;
	get_all_values(features, resolved_feature_name, ids, values, params.max_samples);

	// Get bounds
	return get_quantile_min_max(values);
}

// Clean
//.......................................................................................
int FeatureBasicOutlierCleaner::_apply(MedFeatures& features, unordered_set<int>& ids) {

	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);

	// Clean
	bool empty = ids.empty();
	vector<float>& data = features.data[resolved_feature_name];
	for (unsigned int i = 0; i < features.samples.size(); i++) {
		if ((empty || ids.find(features.samples[i].id) != ids.end()) && data[i] != params.missing_value) {
			if (params.doRemove && (data[i] < removeMin - NUMERICAL_CORRECTION_EPS || data[i] > removeMax + NUMERICAL_CORRECTION_EPS))
				data[i] = params.missing_value;
			else if (params.doTrim) {
				if (data[i] < trimMin)
					data[i] = trimMin;
				else if (data[i] > trimMax)
					data[i] = trimMax;
			}
		}
	}
	return 0;
}

string FeatureNormalizer::select_learn_matrix(const vector<string> &matrix_tags) const {
	//see if it has more recent matrix - for example with resampling + imputations
	if (matrix_tags.empty())
		return "";
	if (matrix_tags.size() == 1)
		return matrix_tags.front();
	//select last one:
	return matrix_tags.back(); //won't create new tag
}

//=======================================================================================
// FeatureNormalizer
//=======================================================================================
//.......................................................................................
int FeatureNormalizer::Learn(MedFeatures& features, unordered_set<int>& ids) {

	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);

	// Get all values
	vector<float> values;
	get_all_values(features, resolved_feature_name, ids, values, max_samples);

	if (use_linear_transform) {
		vector<float> prc_vals;
		vector<float> prc_points = { 0, prctile_th, 1 - prctile_th, 1 };
		medial::stats::get_percentiles<float>(values, prc_points, prc_vals);
		min_x = prc_vals[1]; max_x = prc_vals[2];
		if (min_x == max_x) {
			MWARN("WARNING: min_x==max_x==%f for feature %s\n", min_x, resolved_feature_name.c_str());
			//use min, max
			min_x = prc_vals[0]; max_x = prc_vals[3];
			if (min_x == max_x)
				MWARN("WARNING: feature %s is redandant and all values are equal %f\n", resolved_feature_name.c_str(), min_x);
			max_x = min_x + 1;
		}

		return 0;
	}

	int n;
	medial::stats::get_mean_and_std(values, missing_value, n, mean, sd);

	// Handle constant vector
	if (sd == 0 && values.size()) {
		if (verbosity > 0)
			MWARN("Got constant (size=%d) vector in feature %s....\n", (int)values.size(), feature_name.c_str());
		sd = 1.0;
	}
	else  if (sd == 1)
		if (verbosity > 0)
			MLOG("got sd=1.0 in feature %s....\n", feature_name.c_str());

	if (sd == 0)
		MTHROW_AND_ERR("FeatureNormalizer learn sd: %f mean: %f size: %d", sd, mean, (int)values.size());

	//MLOG("FeatureNormalizer::Learn() done for feature %s , mean %f sd %f size %d\n", feature_name.c_str(), mean, sd, (int)values.size());

	return 0;
}

// Apply
//.......................................................................................
int FeatureNormalizer::_apply(MedFeatures& features, unordered_set<int>& ids) {

	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);

	// Attribute
#pragma omp critical
	{
		if (!resolution_only)
			features.attributes[resolved_feature_name].normalized = true;
		if (fillMissing)
			features.attributes[resolved_feature_name].imputed = true;
		features.attributes[resolved_feature_name].denorm_mean = mean;
		features.attributes[resolved_feature_name].denorm_sdv = sd;
	}

	// treat resolution
	float multiplier = 1;
	if (resolution > 0)
		multiplier = (float)pow(10, resolution);

	// Clean
	bool empty = ids.empty();
	vector<float>& data = features.data[resolved_feature_name];

	if (use_linear_transform) {
		double factor_m = 2 * max_val_prctile / (max_x - min_x);
		for (unsigned int i = 0; i < features.samples.size(); i++) {
			if ((empty || ids.find(features.samples[i].id) != ids.end())) {
				if (data[i] != missing_value) {
					float new_val = -max_val_prctile + (data[i] - min_x) *factor_m;
					if (new_val > max_val_for_triming)
						new_val = max_val_for_triming;
					if (new_val < -max_val_for_triming)
						new_val = -max_val_for_triming;
					data[i] = new_val;
				}
				else if (fillMissing)
					data[i] = 0;
			}
		}
		return 0;
	}

	for (unsigned int i = 0; i < features.samples.size(); i++) {
		if ((empty || ids.find(features.samples[i].id) != ids.end())) {
			if (data[i] != missing_value) {
				if (!resolution_only) {
					data[i] -= mean;
					if (normalizeSd)
						data[i] /= sd;
				}
				if (resolution > 0)
					data[i] = roundf(data[i] * multiplier) / multiplier;
				if (resolution_only && resolution_bin > 0)
					data[i] = int(data[i] / resolution_bin) * resolution_bin;
			}
			else if (fillMissing)
				data[i] = 0;
		}
		if (!isfinite(data[i]))
			MTHROW_AND_ERR("FeatureNormalizer sd: %f mean: %f", sd, mean);
	}

	//MLOG("FeatureNormalizer::Apply() done for feature %s , mean %f sd %f size %d flags: normalized %d imputed %d\n", 
	//	feature_name.c_str(), mean, sd, (int)data.size(), (int)features.attributes[resolved_feature_name].normalized, (int)features.attributes[resolved_feature_name].imputed);

	return 0;
}

// Reverse the Apply - Denorm
void FeatureNormalizer::reverse_apply(float &feature_value) const {
	// Clean
	if (feature_value != missing_value) {
		if (resolution > 0 && resolution_only)
			return; // no norm occoured

		if (use_linear_transform) {
			double inv_factor_m = static_cast<double>(max_x - min_x) / 2 * static_cast<double>(max_val_prctile);
			feature_value = (feature_value + max_val_prctile)*inv_factor_m + min_x;
			return;
		}

		if (normalizeSd)
			feature_value *= sd;
		feature_value += mean;
	}

}


// Init
//.......................................................................................
int FeatureNormalizer::init(map<string, string>& mapper) {
	for (auto entry : mapper) {
		string field = entry.first;
		//! [FeatureNormalizer::init]
		if (field == "missing_value") missing_value = stof(entry.second);
		else if (field == "normalizeSd") normalizeSd = (med_stoi(entry.second) != 0);
		else if (field == "resolution_only") resolution_only = (med_stoi(entry.second) != 0);
		else if (field == "fillMissing") fillMissing = (med_stoi(entry.second) != 0);
		else if (field == "max_samples") max_samples = med_stoi(entry.second);
		else if (field == "resolution") resolution = med_stoi(entry.second);
		else if (field == "resolution_bin") resolution_bin = med_stof(entry.second);
		else if (field == "signal") set_feature_name(entry.second);
		else if (field == "vorbosity") verbosity = med_stoi(entry.second);
		else if (field == "use_linear_transform") use_linear_transform = med_stoi(entry.second) > 0;
		else if (field == "max_val_prctile") max_val_prctile = med_stof(entry.second);
		else if (field == "prctile_th") prctile_th = med_stof(entry.second);
		else if (field == "max_val_for_triming") max_val_for_triming = med_stof(entry.second);
		else if (field != "names" && field != "fp_type" && field != "tag")
			MLOG("Unknonw parameter \'%s\' for FeatureNormalizer\n", field.c_str());
		//! [FeatureNormalizer::init]
	}

	return 0;
}

//=======================================================================================
// FeatureImputer
//=======================================================================================
void FeatureImputer::print()
{
	// Backword-compatability ..
	if (moment_type_vec.empty()) {
		moment_type_vec = { moment_type,moment_type };
		default_moment_vec = { default_moment ,default_moment };
		moments_vec = { moments, moments };
	}

	if (moment_type_vec[0] != moment_type_vec[1])
		MLOG("Imputer at learning stage:\n");

	if (moment_type_vec[0] == IMPUTE_MMNT_SAMPLE) {
		MLOG("Imputer: Feat: %s nHistograms: %d :: ", feature_name.c_str(), histograms.size());
		for (unsigned int i = 0; i < histograms.size(); i++) {
			for (auto& pair : histograms[i])
				MLOG("%d %f L %f", i, pair.first, pair.second);
		}
		MLOG("\n");
	}
	else {
		MLOG("Imputer: Feat: %s nMoments: %d :: ", feature_name.c_str(), moments_vec[0].size());
		for (auto moment : moments_vec[0])
			MLOG("%f ", moment);
		MLOG("\n");
	}

	if (moment_type_vec[0] != moment_type_vec[1]) {
		MLOG("Imputer at learning stage:\n");

		if (moment_type_vec[1] == IMPUTE_MMNT_SAMPLE) {
			MLOG("Imputer: Feat: %s nHistograms: %d :: ", feature_name.c_str(), histograms.size());
			for (unsigned int i = 0; i < histograms.size(); i++) {
				for (auto& pair : histograms[i])
					MLOG("%d %f L %f", i, pair.first, pair.second);
			}
			MLOG("\n");
		}
		else {
			MLOG("Imputer: Feat: %s nMoments: %d :: ", feature_name.c_str(), moments_vec[1].size());
			for (auto moment : moments_vec[1])
				MLOG("%f ", moment);
			MLOG("\n");
		}
	}
}

// Convert partial feature names to full names (including FTR_...)
//.......................................................................................
void FeatureImputer::check_stratas_name(MedFeatures& features, map <string, string> &strata_name_conversion)
{
	for (int i = 0; i < imputerStrata.nStratas(); i++) {
		if (strata_name_conversion.find(imputerStrata.stratas[i].name) != strata_name_conversion.end())
			// already mapped
			continue;
		strata_name_conversion[imputerStrata.stratas[i].name] = resolve_feature_name(features, imputerStrata.stratas[i].name);
	}
}

float FeatureImputer::round_to_closest(float val) const {
	int pos = medial::process::binary_search_position(existing_values, val);
	//pos is in range [0, v.size()] - position in sorted_vals where it's is smaller equals to sorted_vals[pos] but bigger than pos-1 
	if (pos >= existing_values.size())
		pos = (int)existing_values.size() - 1;

	if (existing_values[pos] == val)
		return val;
	//not equal - means sorted_vals[pos] > val
	if (pos == 0)
		return existing_values[0];
	//pos > 0  - mean val is not lower than lowest value in sorted_vals:
	float diff_lower = val - existing_values[pos - 1];
	float diff_upper = existing_values[pos] - val;
	if (diff_lower < diff_upper)
		return  existing_values[pos - 1];
	else
		return existing_values[pos];
}

// Learn
//.......................................................................................
int FeatureImputer::Learn(MedFeatures& features, unordered_set<int>& ids) {

	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);
	default_moment_vec = { missing_value, missing_value }; //initialize
	map <string, string> strata_name_conversion;
	check_stratas_name(features, strata_name_conversion);

	if (round_to_existing_value) {
		unordered_set<float> vals(features.data.at(resolved_feature_name).begin(), features.data.at(resolved_feature_name).end());
		existing_values.clear();
		if (vals.find(missing_value) != vals.end())
			vals.erase(missing_value);
		existing_values.insert(existing_values.end(), vals.begin(), vals.end());
		sort(existing_values.begin(), existing_values.end());
	}

	// Get all values
	vector<float> values;
	get_all_values(features, resolved_feature_name, ids, values, max_samples);
	// Get all strata values
	vector<vector<float> > strataValues(imputerStrata.nStratas());
	for (int i = 0; i < imputerStrata.nStratas(); i++) {
		string resolved_strata_name = resolve_feature_name(features, imputerStrata.stratas[i].name);
		get_all_values(features, resolved_strata_name, ids, strataValues[i], max_samples);
	}

	// Collect
	imputerStrata.getFactors();

	vector<vector<float> > stratifiedValues(imputerStrata.nValues());
	vector<float> all_existing_values;
	for (int j = 0; j < values.size(); j++) {
		if (values[j] != missing_value) {
			all_existing_values.push_back(values[j]);
			int index = 0;
			for (int i = 0; i < imputerStrata.nStratas(); i++)
				index += imputerStrata.factors[i] * imputerStrata.stratas[i].getIndex(strataValues[i][j], missing_value);
			stratifiedValues[index].push_back(values[j]);
		}
	}
	//for (int j = 0; j < stratifiedValues.size(); j++)
		//MLOG("collected %d %d\n", j, stratifiedValues[j].size());

	// Get moments
	moments_vec.resize(2);
	default_moment_vec.resize(2);

	for (int stage = 0; stage < 2; stage++) {
		// Is application stage same as learning stage ?
		if (stage == 1 && moment_type_vec[1] == moment_type_vec[0]) {
			if (moment_type_vec[0] != IMPUTE_MMNT_SAMPLE) {
				moments_vec[1] = moments_vec[0];
				default_moment_vec[1] = default_moment_vec[0];
			}
		}
		else {

			if (moment_type_vec[stage] == IMPUTE_MMNT_SAMPLE)
				histograms.resize(stratifiedValues.size());
			else
				moments_vec[stage].resize(stratifiedValues.size());

			strata_sizes.resize(stratifiedValues.size());
			int too_small_stratas = 0;
			for (unsigned int i = 0; i < stratifiedValues.size(); i++) {

				strata_sizes[i] = (int)stratifiedValues[i].size();
				if (strata_sizes[i] < min_samples) { // Not enough values to make valid imputation
					too_small_stratas++;
					if (moment_type_vec[stage] == IMPUTE_MMNT_SAMPLE)
						histograms[i].push_back({ missing_value,(float)1.0 });
					else
						moments_vec[stage][i] = missing_value;
				}
				else if (moment_type_vec[stage] == IMPUTE_MMNT_MEAN) {
					moments_vec[stage][i] = medial::stats::mean_without_cleaning(stratifiedValues[i]);
					if (round_to_existing_value)
						moments_vec[stage][i] = round_to_closest(moments_vec[stage][i]);
				}
				else if (moment_type_vec[stage] == IMPUTE_MMNT_MEDIAN) {
					if (stratifiedValues[i].size() > 0)
						moments_vec[stage][i] = medial::stats::median_without_cleaning(stratifiedValues[i]);
					else
						moments_vec[stage][i] = missing_value;
				}
				else if (moment_type_vec[stage] == IMPUTE_MMNT_COMMON) {
					if (stratifiedValues[i].size() > 0)
						moments_vec[stage][i] = medial::stats::most_common_without_cleaning(stratifiedValues[i]);
					else
						moments_vec[stage][i] = missing_value;
				}
				else if (moment_type_vec[stage] == IMPUTE_MMNT_SAMPLE) {
					medial::stats::get_histogram_without_cleaning(stratifiedValues[i], histograms[i]);
				}
				else MTHROW_AND_ERR("Unknown moment type %d for imputing %s\n", moment_type_vec[stage], feature_name.c_str());
			}

			// Small stratas ...
			if (all_existing_values.size() < min_samples) {
				if (verbose_learn)
					MLOG("WARNING: FeatureImputer::Learn found only %d < %d samples over all for [%s], will not learn to impute it\n",
						all_existing_values.size(), min_samples, feature_name.c_str());
				if (moment_type_vec[stage] == IMPUTE_MMNT_SAMPLE)
					default_histogram.push_back({ missing_value,(float)1.0 });
				else
					default_moment_vec[stage] = missing_value;
			}
			else {
				if (too_small_stratas > 0) {
					if (!leave_missing_for_small_stratas)
					{
						if (verbose_learn)
							MLOG("WARNING: FeatureImputer::Learn found less than %d samples for %d/%d stratas for [%s], will learn to impute them using all values\n",
								min_samples, too_small_stratas, stratifiedValues.size(), feature_name.c_str());
						if (moment_type_vec[stage] == IMPUTE_MMNT_MEAN) {
							default_moment_vec[stage] = medial::stats::mean_without_cleaning(all_existing_values);
							if (round_to_existing_value)
								default_moment_vec[stage] = round_to_closest(default_moment_vec[stage]);
						}
						else if (moment_type_vec[stage] == IMPUTE_MMNT_MEDIAN)
							default_moment_vec[stage] = medial::stats::median_without_cleaning(all_existing_values);
						else if (moment_type_vec[stage] == IMPUTE_MMNT_COMMON)
							default_moment_vec[stage] = medial::stats::most_common_without_cleaning(all_existing_values);
						else if (moment_type_vec[stage] == IMPUTE_MMNT_SAMPLE)
							medial::stats::get_histogram_without_cleaning(all_existing_values, default_histogram);
					}
					else {
						// leave_missing_for_small_stratas = true
						if (verbose_learn)
							MLOG("WARNING: FeatureImputer::Learn found less than %d samples for %d/%d stratas for [%s], will NOT impute them using all values\n",
								min_samples, too_small_stratas, stratifiedValues.size(), feature_name.c_str());
						if (moment_type_vec[stage] == IMPUTE_MMNT_SAMPLE)
							default_histogram.push_back({ missing_value,(float)1.0 });
						else
							default_moment_vec[stage] = missing_value;
					}
				}
			}
		}
	}

	//#pragma omp critical
	//	print();
	if (verbose_learn)
		print();
	return 0;
}

// Apply
//.......................................................................................
int FeatureImputer::_apply(MedFeatures& features, unordered_set<int>& ids, bool learning) {

	int stage = learning ? 0 : 1;

	// Backword-compatability ..
	if (moment_type_vec.empty()) {
		moment_type_vec = { moment_type,moment_type };
		default_moment_vec = { default_moment ,default_moment };
		moments_vec = { moments, moments };
	}

	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);

	map <string, string> strata_name_conversion;
	check_stratas_name(features, strata_name_conversion);
	// Attribute
#pragma omp critical
	features.attributes[resolved_feature_name].imputed = true;

	// Impute
	imputerStrata.getFactors();
	vector<float>& data = features.data[resolved_feature_name];
	vector<vector<float> *> strataData(imputerStrata.nStratas());
	for (int j = 0; j < imputerStrata.nStratas(); j++) {
		string resolved_strata_name = resolve_feature_name(features, imputerStrata.stratas[j].name);
		strataData[j] = &(features.data[resolved_strata_name]);
	}

	int missing_cnt = 0;
	for (unsigned int i = 0; i < features.samples.size(); i++) {
		if (data[i] == missing_value) {
			int index = 0;
			int missing_in_strata = 0;
			for (int j = 0; j < imputerStrata.nStratas(); j++) {
				float val = (*strataData[j])[i];
				index += imputerStrata.factors[j] * imputerStrata.stratas[j].getIndex(val, missing_value);
				if (val == missing_value) missing_in_strata = 1;
			}
			if (!missing_in_strata || impute_strata_with_missing) {
				if (moment_type_vec[stage] == IMPUTE_MMNT_SAMPLE) {
					if (strata_sizes[index] < min_samples)
						data[i] = medial::stats::sample_from_histogram(default_histogram);
					else
						data[i] = medial::stats::sample_from_histogram(histograms[index]);
				}
				else {
					if (strata_sizes[index] < min_samples)
						data[i] = default_moment_vec[stage];
					else
						data[i] = moments_vec[stage][index];
				}
				if (!isfinite(data[i]))
					MTHROW_AND_ERR("[%s] imputed illegal value for row %d moment_type %d index %d strata_sizes[index] %d %f\n",
						resolved_feature_name.c_str(), i, moment_type_vec[stage], index, strata_sizes[index], default_moment_vec[stage]);
				++missing_cnt;
			}
		}
	}

	if (verbose && missing_cnt > 0) {
		MLOG_D("FeatureImputer::%s:: with %d imputations out of %zu(%2.2f%%)\n",
			resolved_feature_name.c_str(), missing_cnt, data.size(), 100.0 * missing_cnt / double(data.size()));
	}

	return 0;
}

// Init : starta can be a vector, separated by ":"
//.......................................................................................
int FeatureImputer::init(map<string, string>& mapper) {
	vector<string> strata;

	for (auto entry : mapper) {
		string field = entry.first;
		//! [FeatureImputer::init]
		if (field == "name") feature_name = entry.second;
		else if (field == "min_samples") min_samples = med_stoi(entry.second);
		else if (field == "moment_type") {
			moment_type_vec.resize(2);
			moment_type_vec[0] = getMomentType(entry.second);
			moment_type_vec[1] = moment_type_vec[0];
		}
		else if (field == "learn_moment_type") {
			moment_type_vec.resize(2, IMPUTE_MMNT_MEAN);
			moment_type_vec[0] = getMomentType(entry.second);
		}
		else if (field == "apply_moment_type") {
			moment_type_vec.resize(2, IMPUTE_MMNT_MEAN);
			moment_type_vec[1] = getMomentType(entry.second);
		}
		else if (field == "max_samples") max_samples = med_stoi(entry.second);
		else if (field == "strata") {
			boost::split(strata, entry.second, boost::is_any_of(":"));
			for (string& stratum : strata) addStrata(stratum);
		}
		else if (field == "verbose")
			verbose = stoi(entry.second) > 0;
		else if (field == "verbose_learn")
			verbose_learn = stoi(entry.second) > 0;
		else if (field == "round_to_existing_value")
			round_to_existing_value = stoi(entry.second) > 0;
		else if (field == "leave_missing_for_small_stratas") leave_missing_for_small_stratas = med_stoi(entry.second);
		else if (field == "impute_strata_with_missing") impute_strata_with_missing = med_stoi(entry.second);
		else if (field != "names" && field != "fp_type" && field != "tag")
			MLOG("Unknown parameter \'%s\' for FeatureImputer\n", field.c_str());
		//! [FeatureImputer::init]
	}

	return 0;
}

//.......................................................................................
imputeMomentTypes FeatureImputer::getMomentType(string& entry) {

	boost::to_lower(entry);
	if (entry == "0" || entry == "mean")
		return IMPUTE_MMNT_MEAN;
	else if (entry == "1" || entry == "median")
		return IMPUTE_MMNT_MEDIAN;
	else if (entry == "2" || entry == "common")
		return IMPUTE_MMNT_COMMON;
	else if (entry == "3" || entry == "sample")
		return IMPUTE_MMNT_SAMPLE;
	else
		return IMPUTE_MMNT_LAST;
}

//.......................................................................................
void FeatureImputer::addStrata(string& init_string) {

	vector<string> fields;
	boost::split(fields, init_string, boost::is_any_of(","));

	if (fields.size() != 4)
		MLOG("Cannot initialize strata from \'%s\'. Ignoring\n", init_string.c_str());
	else
		addStrata(fields[0], stof(fields[3]), stof(fields[1]), stof(fields[2]));
}

/// update sets of required as input according to set required as output to processor
//.......................................................................................
void FeatureImputer::update_req_features_vec(unordered_set<string>& out_req_features, unordered_set<string>& in_req_features) {

	in_req_features = out_req_features;

	// Check if imputer is actually applied
	if (out_req_features.find(feature_name) != out_req_features.end()) {
		// Add signals required for imputation
		for (int i = 0; i < imputerStrata.nStratas(); i++)
			in_req_features.insert(imputerStrata.stratas[i].name);
	}
}

void FeatureImputer::dprint(const string &pref, int fp_flag) {
	if (fp_flag > 0) {
		MLOG("%s :: FP type %d(%s) : feature_name %s :: default_moment %f \n", pref.c_str(),
			processor_type, my_class_name().c_str(), feature_name.c_str(), default_moment);
	}
}

//=======================================================================================
// OneHotFeatProcessor
//=======================================================================================
int OneHotFeatProcessor::init(map<string, string>& mapper) {

	for (auto entry : mapper) {
		string field = entry.first;
		//! [OneHotFeatProcessor::init]
		if (field == "name") feature_name = entry.second;
		else if (field == "prefix") index_feature_prefix = entry.second;
		else if (field == "remove_origin") rem_origin = (med_stoi(entry.second) != 0);
		else if (field == "add_other") add_other = (med_stoi(entry.second) != 0);
		else if (field == "allow_other") allow_other = (med_stoi(entry.second) != 0);
		else if (field == "remove_last") remove_last = (med_stoi(entry.second) != 0);
		else if (field == "max_values") max_values = med_stoi(entry.second);
		else if (field == "regex_list") boost::split(regex_list, entry.second, boost::is_any_of(","));
		else if (field == "regex_list_names") boost::split(regex_list_names, entry.second, boost::is_any_of(","));
		else if (field == "other_suffix") other_suffix = entry.second;
		else if (field != "names" && field != "fp_type" && field != "tag")
			MLOG("Unknown parameter \'%s\' for OneHotFeatProcessor\n", field.c_str());
		//! [OneHotFeatProcessor::init]
	}

	// Set output names
	if (index_feature_prefix == "")
		index_feature_prefix = feature_name;

	if (regex_list.size() != regex_list_names.size())
		MTHROW_AND_ERR("regex_list with size: %zu, regex_list_names with size: %zu. Should be same size \n", regex_list.size(), regex_list_names.size());

	return 0;
}

string OneHotFeatProcessor::get_feature_name(float value, const string &out_prefix, unordered_map<float, string> &value2Name, float missing_value) {
	stringstream s;

	s << out_prefix << ".";

	if (value2Name.empty())
		s << to_string(value);
	else if (value == missing_value)
		s << "MISSING_VALUE";
	else {
		if (value2Name.find(value) == value2Name.end())
			MTHROW_AND_ERR("Value %f missing from dictionary for OneHot for feature %s\n", value, feature_name.c_str());
		s << value2Name[value];
	}

	return s.str();
}

int OneHotFeatProcessor::Learn(MedFeatures& features, unordered_set<int>& ids) {

	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);
	string out_prefix = resolved_feature_name;
	boost::replace_first(out_prefix, feature_name, index_feature_prefix);

	set<string> feature_names_s;
	if (regex_list.size() > 0)
	{
		// multilabel mode - with regex
		for (int i = 0; i < regex_list.size(); i++)
		{
			std::regex regf(regex_list[i]);
			for (auto& value2Name : features.attributes[resolved_feature_name].value2Name)
			{
				if (std::regex_match(value2Name.second, regf)) {
					unordered_map<float, string> val2name_tmp = { { value2Name.first, regex_list_names[i]} };
					string name = get_feature_name(value2Name.first, out_prefix, val2name_tmp, features.medf_missing_value);
					value2feature[value2Name.first].push_back(name);
					feature_names_s.insert(name);
				}
			}
		}
	}
	else
	{
		// Get all values
		vector<float> values;
		get_all_values(features, resolved_feature_name, ids, values, 0);

		// Build value2feature
		set<float> all_values(values.begin(), values.end());
		if (all_values.size() > max_values)
			MTHROW_AND_ERR("Found %zd different values for %s. More than allowed %d\n", all_values.size(), feature_name.c_str(), max_values);

		string feature_name;

		for (float value : all_values) {
			string name = get_feature_name(value, out_prefix, features.attributes[resolved_feature_name].value2Name, features.medf_missing_value);
			value2feature[value].push_back(name);
			feature_names_s.insert(name);
		}
	}

	// Remove last one

	if (add_other) {
		other_feature_name = out_prefix + "." + index_feature_prefix + "_" + other_suffix;

		if (feature_names_s.find(other_feature_name) != feature_names_s.end())
			MTHROW_AND_ERR("Feature name %s cannot be used for other-value in oneHot for %s. Change using other_suffix\n", other_feature_name.c_str(), feature_name.c_str());
		feature_names_s.insert(other_feature_name);

	}
	// Remove last one
	if (remove_last && !value2feature.empty())
		removed_feature_name = *feature_names_s.begin();
	return 0;
}

int OneHotFeatProcessor::_apply(MedFeatures& features, unordered_set<int>& ids) {

	// Prepare new Features
	int samples_size = (int)features.samples.size();
	for (auto& feature_name : value2feature) {
		for (string& name : feature_name.second)
		{
			if (name != removed_feature_name)
#pragma omp critical
			{
				features.data[name].clear();
				features.data[name].resize(samples_size, 0.0);
				// Attributes
				features.attributes[name].normalized = false;
				features.attributes[name].imputed = true;
			}
		}
	}

	if (add_other) {
#pragma omp critical
		{
			features.data[other_feature_name].clear();
			features.data[other_feature_name].resize(samples_size, 0.0);
			// Attributes
			features.attributes[other_feature_name].normalized = false;
			features.attributes[other_feature_name].imputed = true;
		}
	}


	// Fill it up
	for (int i = 0; i < samples_size; i++) {
		if (ids.empty() || ids.find(features.samples[i].id) != ids.end()) {
			float num_value = features.data[resolved_feature_name][i];
			if (value2feature.find(num_value) != value2feature.end()) {
				for (string& name : value2feature[num_value])
				{
					if (name != removed_feature_name)
						features.data[name][i] = 1.0;
				}
			}
			else {
				if (add_other)
					features.data[other_feature_name][i] = 1.0;
				else if (!allow_other)
					MTHROW_AND_ERR("Unknown value %f for feature %s. Consider using allow_other \n", num_value, feature_name.c_str());
			}
		}
	}

	// Remove original, if required
#pragma omp critical
	if (rem_origin) {
		features.data.erase(resolved_feature_name);
		features.attributes.erase(resolved_feature_name);
	}

	return 0;
}

/// check if a set of features is affected by the current processor
//.......................................................................................
bool OneHotFeatProcessor::are_features_affected(unordered_set<string>& out_req_features) {

	// If empty = all features are required
	if (out_req_features.empty())
		return true;

	// Otherwise - check in generated features
	for (auto& feature_names : value2feature) {
		for (auto& feature_name : feature_names.second) {
			if (out_req_features.find(feature_name) != out_req_features.end())
				return true;
		}
	}

	if (add_other &&out_req_features.find(other_feature_name) != out_req_features.end())
		return true;

	return false;
}

/// update sets of required as input according to set required as output to processor
//.......................................................................................
void OneHotFeatProcessor::update_req_features_vec(unordered_set<string>& out_req_features, unordered_set<string>& in_req_features) {

	// If empty, keep as is
	if (out_req_features.empty())
		in_req_features.clear();
	else {
		in_req_features = out_req_features;
		// If active, than add original 
		if (are_features_affected(out_req_features))
			in_req_features.insert(resolved_feature_name);
	}
}


//=======================================================================================
// GetProbFeatProcessor
//=======================================================================================
//.......................................................................................
int GetProbFeatProcessor::Learn(MedFeatures& features, unordered_set<int>& ids) {

	// Sanity
	if (!target_labels.empty() && all_labels)
		MTHROW_AND_ERR("GetProbFeatProcessor Error: both all_labels and target_labels given\n");

	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);

	// Fill target labels
	if (all_labels) {
		unordered_set<float> all_labels_set;
		for (auto& sample : features.samples)
			all_labels_set.insert(sample.outcome);

		int idx = 0;
		for (float label : all_labels_set)
			target_labels[label] = idx++;


	}

	// Get all values
	vector<float> values;
	get_all_values(features, resolved_feature_name, ids, values, (int)features.samples.size());

	// Learn Probs
	int nlabels = target_labels.empty() ? 1 : (int)target_labels.size();
	map<float, int> nums;
	vector<map<float, int>> pos_nums(nlabels);
	int overall_num = 0;
	vector<int> overall_pos_num(nlabels);

	if (target_labels.empty()) { // Binary outcome

		for (unsigned int i = 0; i < values.size(); i++) {
			if (values[i] != missing_value) {
				nums[values[i]] ++;
				overall_num++;

				if (features.samples[i].outcome) {
					pos_nums[0][values[i]] ++;
					overall_pos_num[0]++;
				}
			}
		}
	}
	else { // Multi-categorical
		for (unsigned int i = 0; i < values.size(); i++) {
			if (values[i] != missing_value) {
				nums[values[i]] ++;
				overall_num++;

				float outcome = features.samples[i].outcome;
				if (target_labels.find(outcome) != target_labels.end()) {
					pos_nums[target_labels[outcome]][values[i]] ++;
					overall_pos_num[target_labels[outcome]]++;
				}
			}
		}

		for (auto& rec : target_labels)
			feature_names[rec.first] = resolved_feature_name + "_" + to_string(rec.first);
	}

	if (overall_num == 0)
		MTHROW_AND_ERR("Cannot learn Get-Prob feature processor on an empty vector for %s\n", feature_name.c_str());

	overall_prob.resize(nlabels);
	probs.resize(nlabels);
	for (int i = 0; i < nlabels; i++) {
		overall_prob[i] = (overall_pos_num[i] + 0.0) / overall_num;
		for (auto& rec : nums)
			if (rec.second >= min_obs)
				probs[i][rec.first] = (pos_nums[i][rec.first] + overall_count * overall_prob[i]) / (nums[rec.first] + overall_count);
			else
				probs[i][rec.first] = overall_prob[i];
	}

	return 0;
}

// Apply
//.......................................................................................
int GetProbFeatProcessor::_apply(MedFeatures& features, unordered_set<int>& ids) {

	//cerr << "Apply\n";
	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);

	// Transform
	bool empty = ids.empty();
	vector<float>& data = features.data[resolved_feature_name];

	if (target_labels.empty()) { // Single outcome. inplace	
		for (unsigned int i = 0; i < features.samples.size(); i++) {
			if ((empty || ids.find(features.samples[i].id) != ids.end())) {
				if (data[i] == missing_value || probs[0].find(data[i]) == probs[0].end())
					data[i] = overall_prob[0];
				else
					data[i] = probs[0][data[i]];
			}
		}
	}
	else { // Multiple outcomes. new features

		// Prepare new Features
		int samples_size = (int)features.samples.size();
		for (auto& rec : feature_names) {
			string feature_name = rec.second;
#pragma omp critical
			{
				features.data[feature_name].clear();
				features.data[feature_name].resize(samples_size, 0.0);
				// Attributes
				features.attributes[feature_name].normalized = false;
				features.attributes[feature_name].imputed = true;
			}
		}

		// Fill
		for (unsigned int i = 0; i < features.samples.size(); i++) {
			if ((empty || ids.find(features.samples[i].id) != ids.end())) {
				if (data[i] == missing_value || probs[0].find(data[i]) == probs[0].end()) {
					for (auto& rec : feature_names)
						features.data[rec.second][i] = overall_prob[target_labels[rec.first]];
				}
				else {
					for (auto& rec : feature_names)
						features.data[rec.second][i] = probs[target_labels[rec.first]][data[i]];
				}
			}
		}

		// Remove original, if required
#pragma omp critical
		if (remove_origin) {
			features.data.erase(resolved_feature_name);
			features.attributes.erase(resolved_feature_name);
		}
	}


	return 0;
}

// Init
//.......................................................................................
int GetProbFeatProcessor::init(map<string, string>& mapper) {
	for (auto entry : mapper) {
		string field = entry.first;
		//! [GetProbFeatProcessor::init]
		if (field == "name") feature_name = entry.second;
		else if (field == "missing_value") missing_value = stof(entry.second);
		else if (field == "overall_count") overall_count = med_stoi(entry.second);
		else if (field == "min_obs") min_obs = med_stoi(entry.second);
		else if (field == "remove_origin") remove_origin = (med_stoi(entry.second) != 0);
		else if (field == "target_labels") {
			vector<string> labels;
			boost::split(labels, entry.second, boost::is_any_of(","));
			for (int i = 0; i < (int)labels.size(); i++)
				target_labels[stof(labels[i])] = i;
		}
		else if (field == "all_labels") all_labels = (med_stoi(entry.second) != 0);
		else if (field != "names" && field != "fp_type" && field != "tag")
			MLOG("Unknonw parameter \'%s\' for GetProbFeatProcessor\n", field.c_str());
		//! [GetProbFeatProcessor::init]
	}

	return 0;
}


//=======================================================================================
// MissingIndicatorProcessor
//=======================================================================================

int MissingIndicatorProcessor::init(map<string, string>& mapper) {
	bool replace_value_exists = false;
	for (auto entry : mapper) {
		string field = entry.first;
		//! [GetProbFeatProcessor::init]
		if (field == "name") feature_name = entry.second;
		else if (field == "missing_value") missing_value = stof(entry.second);
		else if (field == "replace_value") {
			replace_value = stof(entry.second);
			replace_value_exists = true;
		}
		else if (field != "names" && field != "fp_type" && field != "tag")
			MLOG("Unknonw parameter \'%s\' for FeatureMissingIndicator\n", field.c_str());
	}
	if (!replace_value_exists)
		replace_value = missing_value;

	return 0;
}


int MissingIndicatorProcessor::_apply(MedFeatures& features, unordered_set<int>& ids)
{
	// Resolve
	resolved_feature_name = resolve_feature_name(features, feature_name);

	new_feature_name = resolved_feature_name + "." + name;

#pragma omp critical
	{
		features.data[new_feature_name].reserve(features.samples.size());
		// Attributes
		features.attributes[new_feature_name].normalized = false;
		features.attributes[new_feature_name].imputed = true;
	}

	for (float &val : features.data[resolved_feature_name])
	{
		if (val == missing_value) {
			features.data[new_feature_name].push_back(1.);
			if (val != replace_value)
#pragma omp critical
				val = replace_value;
		}
		else
			features.data[new_feature_name].push_back(0.);
	}
	return 0;
}

void MissingIndicatorProcessor::update_req_features_vec(unordered_set<string>& out_req_features, unordered_set<string>& in_req_features)
{
	// If empty, keep as is
	if (out_req_features.empty())
		in_req_features.clear();
	else {
		in_req_features = out_req_features;
		if (out_req_features.find(new_feature_name) != out_req_features.end())
			in_req_features.insert(resolved_feature_name);
	}
}

//=======================================================================================
// Utilities
//=======================================================================================
//.......................................................................................
void get_all_values(MedFeatures& features, string& signalName, unordered_set<int>& ids, vector<float>& values, int max_sample) {

	values.clear();
	if (ids.empty()) {

		int jump = 1;
		int size = (int)features.data[signalName].size();
		if (max_sample > 0 && max_sample < size)
			jump = size / max_sample;

		vector<float>& dataVec = features.data[signalName];
		for (int i = 0; i < size; i += jump)
			values.push_back(dataVec[i]);

	}
	else {
		for (unsigned int i = 0; i < features.samples.size(); i++) {
			if (ids.find(features.samples[i].id) != ids.end())
				values.push_back(features.data[signalName][i]);
		}
	}
}

//.......................................................................................
void get_all_outcomes(MedFeatures& features, unordered_set<int>& ids, vector<float>& values, int max_sample) {

	values.clear();
	if (ids.empty()) {

		int jump = 1;
		int size = (int)features.samples.size();
		if (max_sample > 0 && max_sample < size)
			jump = size / max_sample;

		for (int i = 0; i < size; i += jump)
			values.push_back(features.samples[i].outcome);
		//values = features.data[signalName];

	}
	else {
		for (unsigned int i = 0; i < features.samples.size(); i++) {
			if (ids.find(features.samples[i].id) != ids.end())
				values.push_back(features.samples[i].outcome);
		}
	}
}

//.......................................................................................
void smearBins(vector<int>& bins, int nBins, int reqNbins) {

	float f = (float)nBins / (float)reqNbins;
	vector<vector<int> > newBins(nBins);
	for (int iBin = 0; iBin < reqNbins; iBin++) {
		int OrigBin = (int)(iBin * f);
		newBins[OrigBin].push_back(iBin);
	}

	for (int i = 0; i < bins.size(); i++) {
		int origBin = bins[i];
		int nNewBins = (int)newBins[origBin].size();
		bins[i] = newBins[origBin][static_cast<size_t>(nNewBins) * (static_cast<size_t>(rand()) / static_cast<size_t>(RAND_MAX))];
	}
}
