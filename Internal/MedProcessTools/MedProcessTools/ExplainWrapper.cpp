#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <omp.h>
#include "ExplainWrapper.h"
#include <MedAlgo/MedAlgo/MedXGB.h>
#include <MedStat/MedStat/MedStat.h>
#include "medial_utilities/medial_utilities/globalRNG.h"
#include <MedAlgo/MedAlgo/MedLightGBM.h>
#include <MedAlgo/MedAlgo/MedQRF.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/optional/optional.hpp>
#include <regex>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MED_MODEL
#define LOCAL_LEVEL LOG_DEF_LEVEL

ExplainFilters::ExplainFilters() {
	max_count = 0;
	sum_ratio = 1;
	sort_mode = 0;
}

int ExplainFilters::init(map<string, string> &map) {
	for (auto it = map.begin(); it != map.end(); ++it)
	{
		if (it->first == "max_count")
			max_count = med_stoi(it->second);
		else if (it->first == "sum_ratio")
			sum_ratio = med_stof(it->second);
		else if (it->first == "sort_mode")
			sort_mode = med_stoi(it->second);
		else
			MTHROW_AND_ERR("Error in ExplainFilters::init - Unknown param \"%s\"\n", it->first.c_str());
	}

	if (sum_ratio < 0 || sum_ratio > 1)
		MTHROW_AND_ERR("Error in ExplainFilters::init - sum_ratio should be in [0,1]\n");
	return 0;
}

void ExplainFilters::filter(map<string, float> &explain_list) const {
	vector<pair<string, float>> sorted;
	sorted.reserve(explain_list.size());
	for (const auto &it : explain_list)
		if (sort_mode == 0 || (sort_mode > 0 && it.second > 0) || (sort_mode < 0 && it.second < 0))
			sorted.push_back(pair<string, float>(it.first, it.second));
	if (sort_mode == 0)
		sort(sorted.begin(), sorted.end(), [](const pair<string, float>&pr1, const pair<string, float>&pr2)
	{ return abs(pr1.second) > abs(pr2.second); });
	else if (sort_mode > 0)
		sort(sorted.begin(), sorted.end(), [](const pair<string, float>&pr1, const pair<string, float>&pr2)
	{ return pr1.second > pr2.second; });
	else if (sort_mode < 0)
		sort(sorted.begin(), sorted.end(), [](const pair<string, float>&pr1, const pair<string, float>&pr2)
	{ return pr1.second < pr2.second; });

	//filter ratio:
	if (sum_ratio < 1) {
		float tot = 0, curr_sum = 0;
		for (const auto &it : sorted)
			tot += abs(it.second);
		int stop_at = 0;
		while (stop_at < sorted.size() && curr_sum / tot > sum_ratio) {
			curr_sum += abs(sorted[stop_at].second);
			++stop_at;
		}
		sorted.resize(stop_at);
	}
	//filter max_count:
	if (max_count > 0 && sorted.size() > max_count)
		sorted.resize(max_count);
	//commit selections:
	map<string, float> filterd;
	for (const auto &it : sorted)
		filterd[it.first] = it.second;
	explain_list = std::move(filterd);
}

ExplainProcessings::ExplainProcessings() {
	group_by_sum = false;
	learn_cov_matrix = false;
	use_mutual_information = false;
	//default bin for mutual information
	mutual_inf_bin_setting.binCnt = 50;
	mutual_inf_bin_setting.min_bin_count = 100;
	mutual_inf_bin_setting.split_method = BinSplitMethod::IterativeMerge;
}

int ExplainProcessings::init(map<string, string> &map) {
	for (auto it = map.begin(); it != map.end(); ++it)
	{
		if (it->first == "group_by_sum")
			group_by_sum = med_stoi(it->second) > 0;
		else if (it->first == "learn_cov_matrix") {
			learn_cov_matrix = med_stoi(it->second) > 0;
			if (!learn_cov_matrix) {
				if (abs_cov_features.nrows > 0)
					MLOG("INFO :: Clear Covariance Matrix\n");
				abs_cov_features.clear();
			}
		}
		else if (it->first == "cov_features") {
			abs_cov_features.read_from_csv_file(it->second, 1);
			for (int i = 0; i < abs_cov_features.nrows; i++)
				for (int j = 0; j < abs_cov_features.ncols; j++)
					abs_cov_features(i, j) = abs(abs_cov_features(i, j));
		}
		else if (it->first == "grouping")
			grouping = it->second;
		else if (it->first == "zero_missing")
			zero_missing = stoi(it->second);
		else if (it->first == "normalize_vals")
			normalize_vals = stoi(it->second);
		else if (it->first == "keep_b0")
			keep_b0 = med_stoi(it->second) > 0;
		else if (it->first == "iterative")
			iterative = med_stoi(it->second) > 0;
		else if (it->first == "iteration_cnt")
			iteration_cnt = med_stoi(it->second);
		else if (it->first == "use_mutual_information")
			use_mutual_information = med_stoi(it->second) > 0;
		else if (it->first == "mutual_inf_bin_setting")
			mutual_inf_bin_setting.init_from_string(it->second);
		else if (it->first == "use_max_cov")
			use_max_cov = med_stoi(it->second) > 0;
		else
			MTHROW_AND_ERR("Error in ExplainProcessings::init - Unknown param \"%s\"\n", it->first.c_str());
	}

	return 0;
}

void ExplainProcessings::post_deserialization()
{
	groupName2Inds.clear();
	for (int i = 0; i < groupNames.size(); i++)
		groupName2Inds[groupNames[i]] = group2Inds[i];
}

double joint_dist_entropy(const vector<float> &v1, const vector<float> &v2) {
	//assume they are binned already:
	unordered_map<float, int> val_to_ind_x, val_to_ind_y;
	for (float v : v1)
	{
		if (val_to_ind_x.find(v) == val_to_ind_x.end()) {
			int curr_sz = (int)val_to_ind_x.size();
			val_to_ind_x[v] = curr_sz;
		}
	}
	for (float v : v2)
	{
		if (val_to_ind_y.find(v) == val_to_ind_y.end()) {
			int curr_sz = (int)val_to_ind_y.size();
			val_to_ind_y[v] = curr_sz;
		}
	}

	vector<int> bins_x(v1.size()), bins_y(v2.size());
	for (size_t i = 0; i < bins_x.size(); ++i)
	{
		bins_x[i] = val_to_ind_x.at(v1[i]);
		bins_y[i] = val_to_ind_y.at(v2[i]);
	}
	unordered_map<int, int> joint_bins; //from bin to count
	int v2_bins = (int)val_to_ind_y.size();
	for (size_t i = 0; i < bins_x.size(); ++i)
		++joint_bins[bins_x[i] * v2_bins + bins_y[i]];

	double res = 0;
	int total = (int)v1.size();
	for (auto &it : joint_bins)
	{
		double prob = double(it.second) / total;
		res += -prob * log(prob) / log(2.0);
	}

	return res;
}

void ExplainProcessings::learn(const MedFeatures &train_mat) {
	if (learn_cov_matrix) {
		//int feat_cnt = (int)train_mat.data.size();
		//cov_features.resize(feat_cnt, feat_cnt);
		if (use_mutual_information) {
			MLOG("Calc Mutual Information mat\n"); //ranges from 0 to inf. should transform into [0,1] := 1/(1+e-x) - 0.5
			abs_cov_features.resize((int)train_mat.data.size(), (int)train_mat.data.size());
			auto it_f1 = train_mat.data.begin();
			//vector<vector<float>> binned(train_mat.data.size());
			vector<vector<float>> binned(train_mat.data.size());
			vector<const vector<float> *> original(train_mat.data.size());
			for (int i = 0; i < train_mat.data.size(); ++i) {
				original[i] = &it_f1->second;
				++it_f1;
			}

			vector<int> empt;
			MedProgress prog_bin("binning_features", (int)original.size(), 30, 1);
#pragma omp parallel for schedule(dynamic) 
			for (int i = 0; i < original.size(); ++i) {
				vector<float> f = *original[i];
				medial::process::split_feature_to_bins(mutual_inf_bin_setting, f, empt, f);
#pragma omp critical 
				binned[i] = std::move(f);
				prog_bin.update();
			}

			for (size_t i = 0; i < train_mat.data.size(); ++i)
				abs_cov_features(i, i) = 1; //should be infinity - full

			MedProgress prog_mi("mutual_informaion_calc", (int)original.size(), 30, 1);
#pragma omp parallel for schedule(dynamic) 
			for (int i = 0; i < (int)train_mat.data.size(); ++i)
			{
				for (size_t j = i + 1; j < train_mat.data.size(); ++j)
				{
					int n;
					float mi = medial::performance::mutual_information(binned[i], binned[j], n);
					//calcualte the joint dist entropy - this is the divider - when 2 features are excatly determnien from one another it will result in 1 after division.
					double den = joint_dist_entropy(binned[i], binned[j]);

					//mi = 1 / (1 + exp(-mi)) - 0.5; //transform 
					if (den > 0)
						mi = mi / den;
#pragma omp critical 
					{
						abs_cov_features(i, j) = mi;
						abs_cov_features(j, i) = mi; //symmetric
					}
				}
				prog_mi.update();
			}
		}
		else {
			MLOG("Calc Covariance mat\n");
			MedMat<float> x_mat;
			train_mat.get_as_matrix(x_mat);
			x_mat.normalize();
			//0 - no transpose, 1 - A_Transpose * B, 2 - A * B_Transpose, 3 - both transpose
			fast_multiply_medmat_transpose(x_mat, x_mat, abs_cov_features, 1, 1.0 / x_mat.nrows);

			for (int i = 0; i < abs_cov_features.nrows; i++)
				for (int j = 0; j < abs_cov_features.ncols; j++)
					abs_cov_features(i, j) = abs(abs_cov_features(i, j));
		}
		//// debug
		//vector<string> f_names;
		//train_mat.get_feature_names(f_names);
		//for (int i = 0; i < cov_features.nrows; i++)
		//	for (int j = 0; j < cov_features.ncols; j++)
		//		MLOG("COV_DEBUG: (%d) %s (%d) %s :: %f\n", i, f_names[i].c_str(), j, f_names[j].c_str(), cov_features(i, j));

	}

	// Should we do post-porcessing covariance fix ?
	if (abs_cov_features.nrows && (!iterative))
		postprocessing_cov = true;

	// Fix covariance matrix for working with groups
	// The ad-hoc approach is to take the maximal inter-groups covariance coefficient
	if (abs_cov_features.nrows && group2Inds.size()) {
		if (group_by_sum) {
			int nGroups = (int)groupNames.size();
			int nFeatures = abs_cov_features.nrows;
			MedMat<float> fixed_cov_abs(nGroups, nFeatures); //cov matrix with groups X features connections, zero inside groups:
			for (int i = 0; i < group2Inds.size(); i++) {
				const vector<int> &all_inds = group2Inds[i];
				vector<bool> mask_grp(nFeatures);
				for (int j : all_inds)
					mask_grp[j] = true;

				for (int j2 = 0; j2 < nFeatures; ++j2) {
					float w = 1;
					if (!mask_grp[j2]) {
						//take max for feature in the group
						float max_alp = 0;
						for (int k : all_inds)
							if (abs_cov_features(k, j2) > max_alp)
								max_alp = abs_cov_features(k, j2);
						w = max_alp;
					}

					fixed_cov_abs(i, j2) = w;
				}
			}
			abs_cov_features = fixed_cov_abs;
		}
	}

	post_deserialization();
}

float ExplainProcessings::get_group_normalized_contrib(const vector<int> &group_inds, vector<float> &contribs, float total_normalization_factor) const
{
	float group_normalization_factor = (float)1e-8;

	for (auto i : group_inds) group_normalization_factor += abs(contribs[i]);

	vector<int> group_mask(contribs.size(), 0);
	for (auto i : group_inds) group_mask[i] = 1;

	vector<float> alphas(contribs.size());

	for (int i = 0; i < group_mask.size(); i++) {
		if (group_mask[i])
			alphas[i] = 1.0f;
		else {
			alphas[i] = 0.0f;
			for (int j : group_inds)
				if (abs_cov_features(j, i) > alphas[i])
					alphas[i] = abs_cov_features(j, i);
			//alphas[i] += abs_cov_features(j, i) * abs(contribs[j]);
		//alphas[i] /= group_normalization_factor;
		}
	}

	float group_contrib = 0.0f;
	for (int i = 0; i < contribs.size(); i++)
		group_contrib += alphas[i] * contribs[i];

	group_contrib /= total_normalization_factor;
	return group_contrib;
}

void ExplainProcessings::process(map<string, float> &explain_list) const {
	unordered_set<string> skip_bias_names = { "b0", "Prior_Score" };
	if (!keep_b0)
		for (auto &s : skip_bias_names) explain_list.erase(s);

	if ((!postprocessing_cov) && !group_by_sum && normalize_vals <= 0)
		return;

	MedMat<float> orig_explain((int)explain_list.size(), 1);
	int k = 0;
	for (auto &e : explain_list) orig_explain(k++, 0) = e.second;


	float normalization_factor = 1.0;
	if (normalize_vals > 0) {

		normalization_factor = (float)1e-8; // starting with a small epsilon so that we never divide by 0 later
		for (auto &e : explain_list) normalization_factor += abs(e.second);
		//MLOG("====> DEBUG normalization_factor %f\n", normalization_factor);
	}

	//first do covarinace if has:
	if (postprocessing_cov) {
		if (abs_cov_features.ncols != explain_list.size() && abs_cov_features.ncols != (int)explain_list.size() - 1)
			MTHROW_AND_ERR("Error in ExplainProcessings::process - processing covarince agg. wrong sizes. cov_features.ncols=%lld, "
				"explain_list.size()=%zu\n", abs_cov_features.ncols, explain_list.size());



		if (group_by_sum) { //if has groups

			//do greedy - from top to down: 
			vector<bool> seen_idx(groupNames.size());
			vector<float> groups_vals(groupNames.size()), group_val_curr(groupNames.size());
			map<string, float> group_explain;
			vector<float *> pointer_vals(groupNames.size());
			for (int i = 0; i < groupNames.size(); ++i) {
				float group_contrib = 0;
				for (size_t j = 0; j < abs_cov_features.ncols; ++j)
					group_contrib += abs_cov_features(i, j) * orig_explain(j, 0);
				group_contrib /= (float)1.0 / normalization_factor;
				group_val_curr[i] = group_contrib;
				group_explain[groupNames[i]] = group_contrib;
				pointer_vals[i] = &group_explain.at(groupNames[i]);
			}

			//iterate groups greedy and substract the most contributing group

			//comment/erase this code if you want the old behaviour - without the fix
			for (int i = 0; i < groupNames.size(); ++i)
			{

				//find max contrib in new group_val_curr:
				float max_contrib = -1, max_contrib_abs = -1;
				int max_contrib_idx = -1;
				for (int j = 0; j < groupNames.size(); ++j)
					if (!seen_idx[j] && max_contrib_abs < abs(group_val_curr[j])) {
						max_contrib = group_val_curr[j];
						max_contrib_abs = abs(max_contrib);
						max_contrib_idx = j;
					}
				//update top value to fixed contribution with cov in explain_list
				//float contrib_before_fix = group_val_curr[max_contrib_idx];
				*pointer_vals[max_contrib_idx] = max_contrib;

				//remove contrib from all others using contrib_before_fix (from all other groups):
				for (int j = 0; j < groupNames.size(); ++j)
					for (int ind_grp2 : group2Inds[j])  //all group indexes that needs to be canceled in curretn group 
						group_val_curr[j] -= 2 * abs_cov_features(max_contrib_idx, ind_grp2) * orig_explain(ind_grp2, 0) * normalization_factor;

				//zero  mark group that won't appear again
				seen_idx[max_contrib_idx] = true;
			}


			explain_list = std::move(group_explain);
		}
		else { //no grouping
			MedMat<float> fixed_with_cov(abs_cov_features.ncols, 1);

			//do greedy - from top to down: 
			vector<bool> seen_idx(fixed_with_cov.ncols);
			vector<float *> pointer_vals(explain_list.size());
			int ind_i = 0;
			for (auto it = explain_list.begin(); it != explain_list.end(); ++it)
			{
				pointer_vals[ind_i] = &it->second;
				++ind_i;
			}

			fast_multiply_medmat(abs_cov_features, orig_explain, fixed_with_cov, (float)1.0 / normalization_factor);
			for (int i = 0; i < explain_list.size(); ++i)
			{

				//find max contrib in new fixed_with_cov:
				float max_contrib = -1, max_contrib_abs = -1;
				int max_contrib_idx = -1;
				for (int j = 0; j < fixed_with_cov.ncols; ++j)
					if (!seen_idx[j] && max_contrib_abs < abs(fixed_with_cov(j, 0))) {
						max_contrib = fixed_with_cov(j, 0);
						max_contrib_abs = abs(max_contrib);
						max_contrib_idx = j;
					}
				//update top value to fixed contribution with cov in explain_list
				float contrib_before_fix = *pointer_vals[max_contrib_idx];
				*pointer_vals[max_contrib_idx] = max_contrib;

				//remove contrib from all others using contrib_before_fix and abs_cov_features in curr_original:
				for (int j = 0; j < fixed_with_cov.ncols; ++j)
					fixed_with_cov(j, 0) -= 2 * contrib_before_fix * abs_cov_features(max_contrib_idx, j) * normalization_factor;

				//zero and mark feature curr_original to zero - that won't appear again
				seen_idx[max_contrib_idx] = true;
				fixed_with_cov(max_contrib_idx, 0) = 0;
			}
		}

		return; // ! -> since we treat group_by_sum differently in this case

	}

	//sum features in groups
	if (group_by_sum) {
		if (group2Inds.empty())
			MTHROW_AND_ERR("Error in ExplainProcessings::process - asked for group_by_sum but haven't provide groups in grouping\n");
		map<string, float> group_explain;
		for (size_t i = 0; i < group2Inds.size(); ++i)
		{
			const string &grp_name = groupNames[i];
			float contrib = 0.0f;
			for (int ind : group2Inds[i])
				contrib += orig_explain(ind, 0);
			group_explain[grp_name] = contrib;

		}

		normalization_factor = 0;
		if (normalize_vals > 0) {
			for (const auto &e : group_explain) normalization_factor += abs(e.second);

			if (normalization_factor > 0)
				for (auto &e : group_explain) e.second /= normalization_factor;
		}

		explain_list = std::move(group_explain);
	}
	else {
		normalization_factor = 0;
		if (normalize_vals > 0) {
			for (const auto &e : explain_list) normalization_factor += abs(e.second);

			if (normalization_factor > 0)
				for (auto &e : explain_list) e.second /= normalization_factor;
		}
	}
}

void ExplainProcessings::process(map<string, float> &explain_list, unsigned char *missing_value_mask) const
{
	process(explain_list);
	if (zero_missing == 0 || missing_value_mask == NULL) 	return;
	//check if has groups - has, zero missings must be done before. skip it here:
	bool has_groups = false;
	for (const vector<int> &v : group2Inds)
	{
		has_groups = v.size() > 1;
		if (has_groups)
			break;
	}


	if (!has_groups) {
		unordered_set<string> skip_bias_names = { "b0", "Prior_Score" };
		for (auto &s : skip_bias_names) explain_list.erase(s);

		// now zero all missing
		int k = 0;
		for (auto &e : explain_list) {
			//MLOG("feat[%d] : %s : %6.4f : mask = %x\n", k, e.first.c_str(), e.second, missing_value_mask[k]);
			if (missing_value_mask[k++] & MedFeatures::imputed_mask)
				e.second = 0;
		}

	}
	else {

		for (auto &g : groupName2Inds) {

			int is_empty = 1;
			for (auto v : g.second)
				if (!(missing_value_mask[v] & MedFeatures::imputed_mask)) {
					is_empty = 0;
					break;
				}

			if (is_empty)
				explain_list[g.first] = 0;
		}

	}

}

int ModelExplainer::init(map<string, string> &mapper) {
	map<string, string> left_to_parse;
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		if (it->first == "processing")
			processing.init_from_string(it->second);
		else if (it->first == "filters")
			filters.init_from_string(it->second);
		else if (it->first == "attr_name")
			global_explain_params.attr_name = it->second;
		else if (it->first == "use_split")
			use_split = stoi(it->second);
		else if (it->first == "use_p")
			use_p = stof(it->second);
		else if (it->first == "store_as_json")
			global_explain_params.store_as_json = med_stoi(it->second) > 0;
		else if (it->first == "denorm_features")
			global_explain_params.denorm_features = med_stoi(it->second) > 0;
		else if (it->first == "pp_type") {} //ignore
		else {
			left_to_parse[it->first] = it->second;
		}
	}
	_init(left_to_parse);

	return 0;
}

int ModelExplainer::update(map<string, string> &mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		if (it->first == "rename_group") {
			vector<string> tokens;
			boost::split(tokens, it->second, boost::is_any_of("|"));
			for (const string & token : tokens)
			{
				vector<string> kv;
				boost::split(kv, token, boost::is_any_of(":"));
				if (kv.size() != 2)
					MTHROW_AND_ERR("Error - bad format - should have 2 tokens with \":\" delimeter. Instead got \"%s\"\n",
						token.c_str());

				//change name:
				bool found = false;
				for (size_t i = 0; i < processing.groupNames.size() && !found; ++i)
					if (processing.groupNames[i] == kv[0]) {
						processing.groupNames[i] = kv[1];
						found = true;
					}
				if (!found)
					MWARN("WARN: couldn't find %s\n", kv[0].c_str());
				if (processing.groupName2Inds.find(kv[0]) == processing.groupName2Inds.end())
					MWARN("WARN: couldn't find %s in map\n", kv[0].c_str());
				else {
					processing.groupName2Inds[kv[1]] = processing.groupName2Inds[kv[0]];
					processing.groupName2Inds.erase(kv[0]);
				}
			}
		}
		else
			MWARN("Unknown argument %s\n", it->first.c_str());
	}
	return 0;
}

void ModelExplainer::get_input_fields(vector<Effected_Field> &fields) const {
	fields.push_back(Effected_Field(Field_Type::FEATURE, ""));
}
void ModelExplainer::get_output_fields(vector<Effected_Field> &fields) const {
	string group_name = global_explain_params.attr_name;
	if (global_explain_params.attr_name.empty()) //default name
		group_name = my_class_name();

	if (global_explain_params.store_as_json) {
		fields.push_back(Effected_Field(Field_Type::STRING_ATTRIBUTE, group_name));
	}
	else {
		for (size_t i = 0; i < processing.groupNames.size(); ++i) {
			fields.push_back(Effected_Field(Field_Type::NUMERIC_ATTRIBUTE,
				group_name + "::" + processing.groupNames[i]));
		}
	}
}

void ModelExplainer::init_post_processor(MedModel& model) {
	original_predictor = model.predictor;
	//Find Norm Processors:
	feats_to_norm.clear();
	if (global_explain_params.store_as_json && global_explain_params.denorm_features) {
		vector<const FeatureProcessor *> p_norms;
		for (const FeatureProcessor *fp : model.feature_processors) {
			if (fp->processor_type == FTR_PROCESS_MULTI) {
				for (const FeatureProcessor *fp_i : static_cast<const MultiFeatureProcessor *>(fp)->processors)
					if (fp_i->processor_type == FTR_PROCESS_NORMALIZER)
						p_norms.push_back(fp_i);
			}
			if (fp->processor_type == FTR_PROCESS_NORMALIZER)
				p_norms.push_back(fp);
		}

		//Map name to feature_processor
		for (const FeatureProcessor *fp : p_norms)
		{
			const FeatureNormalizer *norm_fp = static_cast<const FeatureNormalizer *>(fp);
			feats_to_norm[norm_fp->resolved_feature_name] = norm_fp;
		}
	}
}

void ModelExplainer::explain(MedFeatures &matrix) const {
	vector<map<string, float>> explain_reasons; //for each sample, reasons and scores
	explain(matrix, explain_reasons);

	if (explain_reasons.size() != matrix.samples.size())
		MTHROW_AND_ERR("Error in ModelExplainer::explain - explain returned musmatch number of samples %zu, and requested %zu\n",
			explain_reasons.size(), matrix.samples.size());

	MedMat<unsigned char> masks_mat;
	if (processing.zero_missing)
		matrix.get_masks_as_mat(masks_mat);
	//process:
#pragma omp parallel for
	for (int i = 0; i < (int)explain_reasons.size(); ++i) {
		if (processing.zero_missing)
			processing.process(explain_reasons[i], masks_mat.data_ptr(i, 0));
		else
			processing.process(explain_reasons[i]);
	}
	//filter:
#pragma omp parallel for
	for (int i = 0; i < explain_reasons.size(); ++i)
		filters.filter(explain_reasons[i]);

	string group_name = global_explain_params.attr_name;
	if (global_explain_params.attr_name.empty()) //default name
		group_name = my_class_name();

	if (global_explain_params.store_as_json) {
		vector<string> full_feat_names;
		matrix.get_feature_names(full_feat_names);
		vector<const vector<float> *> p_data(full_feat_names.size());
		for (size_t i = 0; i < full_feat_names.size(); ++i)
			p_data[i] = &matrix.data.at(full_feat_names[i]);

#pragma omp parallel for if (explain_reasons.size() > 2)
		for (int i = 0; i < explain_reasons.size(); ++i) {
			nlohmann::ordered_json full_res;
			string global_explainer_section = "explainer_output";
			full_res[global_explainer_section] = nlohmann::ordered_json::array();
			vector<pair<string, float>> sorted_exp(explain_reasons[i].size());
			int idx = 0;
			double tot_val = 0;
			for (auto it = explain_reasons[i].begin(); it != explain_reasons[i].end(); ++it) {
				sorted_exp[idx].first = it->first;
				sorted_exp[idx].second = it->second;
				++idx;
				tot_val += abs(it->second);
			}
			sort(sorted_exp.begin(), sorted_exp.end(), [](const pair<string, float> &a, const pair<string, float> &b)
			{ return abs(a.second) > abs(b.second); });
			for (const auto &pt : sorted_exp) {
				nlohmann::ordered_json group_json;
				nlohmann::ordered_json child_elements = nlohmann::ordered_json::array();
				//Fill with group features: Feature_Name, Feature_Value
				const vector<int> &grp_idx = processing.groupName2Inds.at(pt.first);
				// TODO: sort by importance

				for (int feat_idx : grp_idx)
				{
					const string &feat_name = full_feat_names[feat_idx];
					float feat_value = p_data[feat_idx]->at(i);
					// Save feature values without normalizations if requested
					if (global_explain_params.denorm_features && feats_to_norm.find(feat_name) != feats_to_norm.end()) {
						const FeatureNormalizer *normalizer = feats_to_norm.at(feat_name);
						normalizer->reverse_apply(feat_value);
					}
					nlohmann::ordered_json child_e;
					child_e["feature_name"] = feat_name;
					child_e["feature_value"] = feat_value;
					child_elements.push_back(child_e);
				}

				group_json["contributor_name"] = pt.first;
				group_json["contributor_value"] = pt.second;
				group_json["contributor_percentage"] = 100 * abs(pt.second) / tot_val;
				group_json["contributor_elements"] = child_elements;

				full_res[global_explainer_section].push_back(group_json);
			}

			matrix.samples[i].str_attributes[group_name] = full_res.dump(1, '\t');
		}
	}
	else {
#pragma omp parallel for
		for (int i = 0; i < explain_reasons.size(); ++i) {
			for (auto it = explain_reasons[i].begin(); it != explain_reasons[i].end(); ++it)
				matrix.samples[i].attributes[group_name + "::" + it->first] = it->second;
		}
	}
}

///format TAB delim, 2 tokens: [Feature_name [TAB] group_name]
void ExplainProcessings::read_feature_grouping(const string &file_name, const vector<string>& features,
	vector<vector<int>>& group2index, vector<string>& group_names, bool verbose) {
	// Features
	int nftrs = (int)features.size();
	map<string, vector<int>> groups;
	vector<bool> grouped_ftrs(nftrs);
	unordered_set<string> trends_set = { "slope", "std", "last_delta", "win_delta", "max_diff", "range_width", "sum" };
	unordered_set<string> time_set = { "last_time", "last_time2", "first_time", "time_since_last_change" }; // "time_inside", "time_len", "time_diff", "time_covered"

	string range_names = "^(last_nth_time_len_[0-9]+_|last_nth_[0-9]+_|time_inside_|time_diff_start_|time_covered_|time_diff_[0-9]+_|ever_|latest_|current_)";

	if (file_name == "BY_SIGNAL") {
		for (int i = 0; i < nftrs; ++i)
		{
			vector<string> tokens;
			boost::split(tokens, features[i], boost::is_any_of("."));
			string word = tokens[0];
			if (tokens.size() > 1 && boost::starts_with(tokens[0], "FTR_"))
				word = tokens[1];

			groups[word].push_back(i);
			grouped_ftrs[i] = true;
		}
	}
	else if (file_name == "BY_SIGNAL_CATEG") {
		std::regex last_nth_reg(range_names);
		std::regex tm_val_ch_no_default_regex("t[0-9]+v[0-9]+");
		for (int i = 0; i < nftrs; ++i)
		{
			vector<string> tokens;
			boost::split(tokens, features[i], boost::is_any_of("."));
			string word = tokens[0];
			int idx = 0;
			if (tokens.size() > 1 && boost::starts_with(tokens[0], "FTR_")) {
				word = tokens[1];
				idx = 1;
			}

			if (idx + 1 < tokens.size()) {
				if (boost::starts_with(tokens[idx + 1], "category_") ||
					std::regex_search(tokens[idx + 1], last_nth_reg)) {
					boost::replace_all(tokens[idx + 1], "category_set_count_", "");
					boost::replace_all(tokens[idx + 1], "category_set_sum_", "");
					boost::replace_all(tokens[idx + 1], "category_set_first_time_", "");
					boost::replace_all(tokens[idx + 1], "category_set_first_", "");
					boost::replace_all(tokens[idx + 1], "category_dep_set_", "");
					boost::replace_all(tokens[idx + 1], "category_dep_count_", "");
					boost::replace_all(tokens[idx + 1], "category_set_", "");
					boost::replace_all(tokens[idx + 1], "category_intake_", "");
					tokens[idx + 1] = std::regex_replace(tokens[idx + 1], last_nth_reg, "");
					word += "." + tokens[idx + 1];
				}
				++idx;
				++idx;
				//skip another last one if last one is "tXvY" format where X,Y are numbers:
				int skip_count = 1;
				if (std::regex_search(tokens.back(), tm_val_ch_no_default_regex))
					++skip_count;
				//Add more tokens till last one - last one in ".win_X_Y"
				while (idx + skip_count < tokens.size()) {
					word += "." + tokens[idx];
					++idx;
				}
			}

			groups[word].push_back(i);
			grouped_ftrs[i] = true;
		}
	}
	else if (file_name == "BY_SIGNAL_CATEG_TREND") {
		std::regex last_nth_reg(range_names);
		std::regex tm_val_ch_no_default_regex("t[0-9]+v[0-9]+");
		for (int i = 0; i < nftrs; ++i)
		{
			vector<string> tokens;
			boost::split(tokens, features[i], boost::is_any_of("."));
			string word = tokens[0];
			int idx = 0;
			if (tokens.size() > 1 && boost::starts_with(tokens[0], "FTR_")) {
				word = tokens[1];
				idx = 1;
			}
			bool categ = false;
			if (idx + 1 < tokens.size()) {
				if (boost::starts_with(tokens[idx + 1], "category_")
					|| std::regex_search(tokens[idx + 1], last_nth_reg)) {
					boost::replace_all(tokens[idx + 1], "category_set_count_", "");
					boost::replace_all(tokens[idx + 1], "category_set_sum_", "");
					boost::replace_all(tokens[idx + 1], "category_set_first_time_", "");
					boost::replace_all(tokens[idx + 1], "category_set_first_", "");
					boost::replace_all(tokens[idx + 1], "category_dep_set_", "");
					boost::replace_all(tokens[idx + 1], "category_dep_count_", "");
					boost::replace_all(tokens[idx + 1], "category_set_", "");
					boost::replace_all(tokens[idx + 1], "category_intake_", "");
					tokens[idx + 1] = std::regex_replace(tokens[idx + 1], last_nth_reg, "");
					word += "." + tokens[idx + 1];
					categ = true;
				}
				if (categ) {
					++idx;
					++idx;
					int skip_count = 1;
					if (std::regex_search(tokens.back(), tm_val_ch_no_default_regex))
						++skip_count;
					//Add more tokens till last one - last one in ".win_X_Y"
					while (idx + skip_count < tokens.size()) {
						word += "." + tokens[idx];
						++idx;
					}
				}
			}
			//check if TREND: slope, std, last_delta, win_delta, max_diff

			if (!categ) {
				string tp = "_Values";
				if (idx + 1 < tokens.size() && trends_set.find(tokens[idx + 1]) != trends_set.end())
					tp = "_Trends";
				if (idx + 1 < tokens.size() && time_set.find(tokens[idx + 1]) != time_set.end())
					tp = "_Time";
				if (idx > 0)
					word += tp;
			}

			groups[word].push_back(i);
			grouped_ftrs[i] = true;
		}
	}
	else {
		// Read Grouping
		ifstream inf(file_name);
		if (!inf.is_open())
			MTHROW_AND_ERR("Cannot open \'%s\' for reading\n", file_name.c_str());

		string curr_line;
		vector<string> fields;

		while (getline(inf, curr_line)) {
			boost::split(fields, curr_line, boost::is_any_of("\t"));
			if (fields.size() != 2)
				MTHROW_AND_ERR("Cannot parse line \'%s\' from %s\n", curr_line.c_str(), file_name.c_str());
			int feat_pos = find_in_feature_names(features, fields[0]);
			if (grouped_ftrs[feat_pos])
				MTHROW_AND_ERR("Features %s given twice\n", fields[0].c_str());

			grouped_ftrs[feat_pos] = true;
			groups[fields[1]].push_back(feat_pos);
		}
	}
	// Arrange
	for (auto& rec : groups) {
		group_names.push_back(rec.first);
		group2index.push_back(rec.second);
	}

	for (int i = 0; i < nftrs; i++) {
		if (!grouped_ftrs[i]) {
			group_names.push_back(features[i]);
			group2index.push_back({ i });
		}
	}
	if (verbose)
		MLOG("Grouping: %d features into %d groups\n", nftrs, (int)group_names.size());
}

void ModelExplainer::Learn(const MedFeatures &train_mat) {
	if (original_predictor == NULL)
		MTHROW_AND_ERR("Error ModelExplainer::Learn - please call init_post_processor before learn\n");
	if (!processing.grouping.empty()) {
		vector<string> features_nms;
		train_mat.get_feature_names(features_nms);
		ExplainProcessings::read_feature_grouping(processing.grouping, features_nms, processing.group2Inds, processing.groupNames);
	}
	else {
		int icol = 0;
		for (auto& rec : train_mat.data) {
			processing.group2Inds.push_back({ icol++ });
			processing.groupNames.push_back(rec.first);
		}
	}
	processing.learn(train_mat);
	_learn(train_mat);
}

void ModelExplainer::dprint(const string &pref) const {
	string predictor_nm = "";
	if (original_predictor != NULL)
		predictor_nm = original_predictor->my_class_name();
	string filters_str = "", processing_str = "";
	char buffer[5000];
	snprintf(buffer, sizeof(buffer), "group_by_sum=%d, learn_cov_matrix=%d, normalize_vals=%d, zero_missing=%d, grouping=%s",
		int(processing.group_by_sum), int(processing.learn_cov_matrix), processing.normalize_vals
		, processing.zero_missing, processing.grouping.c_str());
	processing_str = string(buffer);
	snprintf(buffer, sizeof(buffer), "sort_mode=%d, max_count=%d, sum_ratio=%2.3f",
		filters.sort_mode, filters.max_count, filters.sum_ratio);
	filters_str = string(buffer);
	MLOG("%s :: ModelExplainer type %d(%s), original_predictor=%s, attr_name=%s, processing={%s}, filters={%s}\n",
		pref.c_str(), processor_type, my_class_name().c_str(), predictor_nm.c_str(), global_explain_params.attr_name.c_str(),
		processing_str.c_str(), filters_str.c_str());
}

bool comp_score_str(const pair<string, float> &pr1, const pair<string, float> &pr2) {
	return abs(pr1.second) > abs(pr2.second); //bigger is better in absolute
}

void ModelExplainer::print_explain(MedSample &smp, int sort_mode) {
	vector<pair<string, float>> ranked;
	for (auto it = smp.attributes.begin(); it != smp.attributes.end(); ++it)
		if (boost::starts_with(it->first, "ModelExplainer::"))
			ranked.push_back(pair<string, float>(it->first, it->second));
	if (sort_mode == 0)
		sort(ranked.begin(), ranked.end(), comp_score_str);
	else if (sort_mode > 0)
		sort(ranked.begin(), ranked.end(), [](const pair<string, float>&pr1, const pair<string, float>&pr2) { return pr1.second > pr2.second; });
	else
		sort(ranked.begin(), ranked.end(), [](const pair<string, float>&pr1, const pair<string, float>&pr2) { return pr1.second < pr2.second; });

	for (size_t i = 0; i < ranked.size(); ++i)
		MLOG("%s = %f\n", ranked[i].first.c_str(), ranked[i].second);
	if (!smp.prediction.empty())
		MLOG("ModelExplainer::Prediction_Raw_Score = %f\n", smp.prediction[0]);
}

int get_max_rec(const vector<QRF_ResNode> &nodes, int idx) {
	const QRF_ResNode &curr = nodes[idx];
	if (curr.is_leaf)
		return 0; //reached leaf
	int max_d = get_max_rec(nodes, curr.left) + 1;
	int max_right = get_max_rec(nodes, curr.right) + 1;
	if (max_right > max_d)
		max_d = max_right;

	return max_d;
}

int get_tree_max_depth(const vector<QRF_ResNode> &nodes) {
	if (nodes.empty())
		return 0;
	int max_d = get_max_rec(nodes, 0) + 1;
	return max_d;
}

void get_tree_nnodes(vector<int> left_children, vector<int> right_children, int& nInternal, int& nLeaves) {

	nInternal = nLeaves = 0;

	for (size_t i = 0; i < left_children.size(); i++) {
		if (right_children[i] > nInternal)
			nInternal = right_children[i];
		if ((~(right_children[i])) > nLeaves)
			nLeaves = ~(right_children[i]);
		if (left_children[i] > nInternal)
			nInternal = left_children[i];
		if ((~(left_children[i])) > nLeaves)
			nLeaves = ~(left_children[i]);
	}

	nInternal++; // Add 0
	return;
}

int get_max_rec(vector<int> left_children, vector<int> right_children, int idx) {
	int max_right = (right_children[idx] < 0) ? 1 : get_max_rec(left_children, right_children, right_children[idx]) + 1;
	int max_left = (left_children[idx] < 0) ? 1 : get_max_rec(left_children, right_children, left_children[idx]) + 1;

	return((max_right > max_left) ? max_right : max_left);
}

int get_tree_max_depth(vector<int> left_children, vector<int> right_children) {

	if (left_children.empty())
		return 0;
	return get_max_rec(left_children, right_children, 0) + 1;

}

//all conversion functions
bool TreeExplainer::convert_qrf_trees() {
	MLOG("Converting QRF to generic ensemble trees\n");
	int num_outputs = original_predictor->n_preds_per_sample();
	const QRF_Forest &forest = static_cast<MedQRF *>(original_predictor)->qf;
	if (forest.n_categ == 2)
		num_outputs = 1;
	const vector<float> &all_vals = forest.sorted_values;
	if (all_vals.empty() && forest.n_categ != 2) {
		MWARN("Can't convert QRF. please retrain with keep_all_values to be able to convert categorical trees with n_categ > 2\n");
		return false;
	}
	int max_nodes = 0, max_depth = 0;
	for (size_t i = 0; i < forest.qtrees.size(); ++i)
	{
		if (max_nodes < forest.qtrees[i].qnodes.size())
			max_nodes = (int)forest.qtrees[i].qnodes.size();
		int mm = get_tree_max_depth(forest.qtrees[i].qnodes);
		if (mm > max_depth)
			max_depth = mm;
	}

	++max_nodes; //this is tree seperator index for model
	generic_tree_model.allocate((int)forest.qtrees.size(), max_nodes, num_outputs);
	int pos_in_model = 0;
	generic_tree_model.max_depth = max_depth;
	generic_tree_model.tree_limit = (int)forest.qtrees.size();
	generic_tree_model.max_nodes = max_nodes;
	generic_tree_model.num_outputs = num_outputs;
	generic_tree_model.base_offset = 0; //no bias
	for (size_t i = 0; i < forest.qtrees.size(); ++i) {
		//convert each tree:
		const vector<QRF_ResNode> &tr = forest.qtrees[i].qnodes;
		for (size_t j = 0; j < tr.size(); ++j)
		{
			generic_tree_model.children_left[pos_in_model + j] = tr[j].left;
			generic_tree_model.children_right[pos_in_model + j] = tr[j].right;
			generic_tree_model.children_default[pos_in_model + j] = tr[j].left; //smaller than is left
																				//generic_tree_model.children_default[pos_in_model + j] = -1; //no default - will fail

			generic_tree_model.features[pos_in_model + j] = int(tr[j].ifeat);
			if (tr[j].is_leaf)
			{
				generic_tree_model.children_right[pos_in_model + j] = -1;//mark leaf
				generic_tree_model.children_left[pos_in_model + j] = -1;//mark leaf
				generic_tree_model.children_default[pos_in_model + j] = -1;//mark leaf
			}
			generic_tree_model.thresholds[pos_in_model + j] = tr[j].split_val;
			generic_tree_model.node_sample_weights[pos_in_model + j] = float(tr[j].n_size); //no support for trained in weights for now
																							//if n_categ ==2:
			if (forest.n_categ == 2) {
				if (!tr[j].counts.empty()) {
					if (tr[j].n_size != 0)
						generic_tree_model.values[pos_in_model + j] = tr[j].counts[1] / float(tr[j].n_size);
					else
						MTHROW_AND_ERR("Error node leaf has 0 obs\n");
				}
				else
					generic_tree_model.values[pos_in_model + j] = tr[j].pred;
			}
			else {
				vector<float> scores(num_outputs);
				tr[j].get_scores(forest.mode, forest.get_counts_flag, forest.n_categ, scores);
				//convert values for each prediction:
				for (size_t k = 0; k < scores.size(); ++k)
					generic_tree_model.values[static_cast<size_t>(pos_in_model) * static_cast<size_t>(num_outputs) + j * static_cast<size_t>(num_outputs) + k] = scores[k];
			}
		}

		pos_in_model += max_nodes;
	}

	return true;
}

void parse_tree(ptree& subtree, TreeEnsemble& generic_tree_model, int pos_in_model) {

	int j = subtree.get<int>("nodeid");
	generic_tree_model.node_sample_weights[pos_in_model + j] = subtree.get<float>("cover");
	if (subtree.count("children") > 0) {
		int lc = generic_tree_model.children_left[pos_in_model + j] = subtree.get<int>("yes");
		int rc = generic_tree_model.children_right[pos_in_model + j] = subtree.get<int>("no");
		generic_tree_model.children_default[pos_in_model + j] = subtree.get<int>("missing");

		generic_tree_model.features[pos_in_model + j] = med_stoi(subtree.get<string>("split").substr(1));
		generic_tree_model.thresholds[pos_in_model + j] = subtree.get<float>("split_condition");

		for (ptree::value_type &child : subtree.get_child("children"))
			parse_tree(child.second, generic_tree_model, pos_in_model);

		generic_tree_model.values[pos_in_model + j] = (generic_tree_model.values[pos_in_model + lc] * generic_tree_model.node_sample_weights[pos_in_model + lc] +
			generic_tree_model.values[pos_in_model + rc] * generic_tree_model.node_sample_weights[pos_in_model + rc]) / generic_tree_model.node_sample_weights[pos_in_model + j];
	}
	else {
		generic_tree_model.values[pos_in_model + j] = subtree.get<float>("leaf");
		generic_tree_model.children_left[pos_in_model + j] = -1;
		generic_tree_model.children_right[pos_in_model + j] = -1;
		generic_tree_model.children_default[pos_in_model + j] = -1;
	}
}

void analyze_tree(ptree& subtree, int &nnodes, int& depth) {

	int nodeId = subtree.get<int>("nodeid");
	if (nodeId > nnodes)
		nnodes = nodeId;

	if (subtree.count("depth") > 0) {
		int nodeDepth = subtree.get<int>("depth");
		if (nodeDepth + 2 > depth)
			depth = nodeDepth + 2;

		if (subtree.count("children") > 0) {
			for (ptree::value_type &child : subtree.get_child("children"))
				analyze_tree(child.second, nnodes, depth);
		}
	}
}

bool TreeExplainer::convert_xgb_trees() {

	MLOG("Converting XGB to generic ensemble trees\n");

	int num_outputs = original_predictor->n_preds_per_sample();
	if (num_outputs > 1) {
		MLOG("multiple categries not implemented in converting xgboost to generic-tree. Will use xgboost implementation of TreeSHAP");
		return false;
	}

	// Export to Json
	const char **trees = NULL;
	int nTrees;
	static_cast<MedXGB *>(original_predictor)->get_json(&trees, nTrees, "json");
	if (trees == NULL)
		MTHROW_AND_ERR("Error TreeExplainer::convert_xgb_trees - can't retriev xgboost model\n");

	// Analyze treees
	int max_nodes = 0, max_depth = 0;
	for (int iTree = 0; iTree < nTrees; iTree++) {

		std::stringstream ss;
		ss << trees[iTree];
		ptree pt;
		read_json(ss, pt);

		int nnodes = 0, depth = 0;
		analyze_tree(pt, nnodes, depth);
		if (nnodes > max_nodes)
			max_nodes = nnodes;
		if (depth > max_depth)
			max_depth = depth;

	}

	// Parse trees
	++max_nodes; //this is tree seperator index for model
	generic_tree_model.allocate(nTrees, max_nodes, num_outputs);

	int pos_in_model = 0;
	generic_tree_model.max_depth = max_depth;
	generic_tree_model.tree_limit = nTrees;
	generic_tree_model.max_nodes = max_nodes;
	generic_tree_model.num_outputs = num_outputs;
	generic_tree_model.base_offset = 0; //no bias

	for (int iTree = 0; iTree < nTrees; ++iTree) {
		//convert each tree:
		std::stringstream ss;
		ss << trees[iTree];
		ptree pt;
		read_json(ss, pt);
		parse_tree(pt, generic_tree_model, pos_in_model);

		pos_in_model += max_nodes;
	}

	return true;
}

bool TreeExplainer::convert_lightgbm_trees() {
	MLOG("Converting LightGBM to generic ensemble trees\n");

	int num_outputs = original_predictor->n_preds_per_sample();
	if (num_outputs > 1) {
		MLOG("multiple categries not implemented in converting light-gbm to generic-tree. Will use light-gbm implementation of TreeSHAP");
		return false;
	}

	vector<vector<int>> split_features, decision_types, left_children, right_children, leaf_counts, internal_counts;
	vector<vector<double>> thresholds, leaf_values;

	// Export to string
	string trees;
	trees = static_cast<MedLightGBM *>(original_predictor)->get_model_str();

	// Parse string
	vector<string> lines, fields, values;
	boost::split(lines, trees, boost::is_any_of("\n"));
	int iTree = -1;
	for (string& line : lines) {
		if (line != "") {
			boost::split(fields, line, boost::is_any_of("="));
			if (fields[0] == "Tree") {
				int _tree = stoi(fields[1]);
				if (_tree != iTree + 1)
					MTHROW_AND_ERR("Missing tree #%d in light-gbm\n", iTree + 1);
				iTree = _tree;
			}
			else if (fields[0] == "split_feature") {
				boost::split(values, fields[1], boost::is_any_of(" "));
				vector<int> _values(values.size());
				for (size_t i = 0; i < values.size(); i++)
					_values[i] = stoi(values[i]);
				split_features.push_back(_values);
			}
			else if (fields[0] == "threshold") {
				boost::split(values, fields[1], boost::is_any_of(" "));
				vector<double> _values(values.size());
				for (size_t i = 0; i < values.size(); i++)
					_values[i] = stof(values[i]);
				thresholds.push_back(_values);
			}
			else if (fields[0] == "decision_type") {
				boost::split(values, fields[1], boost::is_any_of(" "));
				vector<int> _values(values.size());
				for (size_t i = 0; i < values.size(); i++)
					_values[i] = stoi(values[i]);
				decision_types.push_back(_values);

				for (size_t i = 0; i < values.size(); i++) {
					if (_values[i] & 1) {
						MLOG("Categorical Decision not implemented yet in generic-tree. Will use light-gbm implementation of TreeSHAP\n");
						return false;
					}
				}
			}
			else if (fields[0] == "left_child") {
				boost::split(values, fields[1], boost::is_any_of(" "));
				vector<int> _values(values.size());
				for (size_t i = 0; i < values.size(); i++)
					_values[i] = stoi(values[i]);
				left_children.push_back(_values);
			}
			else if (fields[0] == "right_child") {
				boost::split(values, fields[1], boost::is_any_of(" "));
				vector<int> _values(values.size());
				for (size_t i = 0; i < values.size(); i++)
					_values[i] = stoi(values[i]);
				right_children.push_back(_values);
			}
			else if (fields[0] == "leaf_value") {
				boost::split(values, fields[1], boost::is_any_of(" "));
				vector<double> _values(values.size());
				for (size_t i = 0; i < values.size(); i++)
					_values[i] = stof(values[i]);
				leaf_values.push_back(_values);
			}
			else if (fields[0] == "leaf_count") {
				boost::split(values, fields[1], boost::is_any_of(" "));
				vector<int> _values(values.size());
				for (size_t i = 0; i < values.size(); i++)
					_values[i] = stoi(values[i]);
				leaf_counts.push_back(_values);
			}
			else if (fields[0] == "internal_count") {
				boost::split(values, fields[1], boost::is_any_of(" "));
				vector<int> _values(values.size());
				for (size_t i = 0; i < values.size(); i++)
					_values[i] = stoi(values[i]);
				internal_counts.push_back(_values);
			}
		}
	}

	// Build generic tree
	int nTrees = iTree + 1;
	int max_nodes = 0, max_depth = 0;
	vector<int> nLeaves(nTrees), nInternal(nTrees);
	for (int i = 0; i < nTrees; ++i)
	{
		get_tree_nnodes(left_children[i], right_children[i], nInternal[i], nLeaves[i]);
		int nn = nInternal[i] + nLeaves[i];
		if (nn > max_nodes)
			max_nodes = nn;
		int mm = get_tree_max_depth(left_children[i], right_children[i]);
		if (mm > max_depth)
			max_depth = mm;
	}

	++max_nodes; //this is tree seperator index for model
	generic_tree_model.allocate(nTrees, max_nodes, num_outputs);

	int pos_in_model = 0;
	generic_tree_model.max_depth = max_depth;
	generic_tree_model.tree_limit = nTrees;
	generic_tree_model.max_nodes = max_nodes;
	generic_tree_model.num_outputs = num_outputs;
	generic_tree_model.base_offset = 0; //no bias

	for (int i = 0; i < nTrees; ++i) {
		//convert each tree:
		// Internal nodes
		tfloat tot_weights = 0, tot_values = 0;
		for (size_t j = 0; j < split_features[i].size(); ++j)
		{
			int left = (left_children[i][j] > 0) ? left_children[i][j] : (nInternal[i] + (~(left_children[i][j])));
			int right = (right_children[i][j] > 0) ? right_children[i][j] : (nInternal[i] + (~(right_children[i][j])));
			generic_tree_model.children_left[pos_in_model + j] = left;
			generic_tree_model.children_right[pos_in_model + j] = right;
			generic_tree_model.children_default[pos_in_model + j] = (decision_types[i][j] & 2) ? left : right;

			generic_tree_model.features[pos_in_model + j] = split_features[i][j];
			generic_tree_model.thresholds[pos_in_model + j] = thresholds[i][j];
			generic_tree_model.node_sample_weights[pos_in_model + j] = internal_counts[i][j];
		}

		// Leaves
		for (size_t j = 0; j < leaf_values[i].size(); ++j) {
			generic_tree_model.children_left[pos_in_model + nInternal[i] + j] = -1;
			generic_tree_model.children_right[pos_in_model + nInternal[i] + j] = -1;
			generic_tree_model.children_default[pos_in_model + nInternal[i] + j] = -1;

			generic_tree_model.values[pos_in_model * num_outputs + nInternal[i] + j * num_outputs] = leaf_values[i][j];
			generic_tree_model.node_sample_weights[pos_in_model + nInternal[i] + j] = leaf_counts[i][j];

			tot_weights += leaf_counts[i][j];
			tot_values += leaf_values[i][j];
		}

		pos_in_model += max_nodes;
		generic_tree_model.values[pos_in_model] = tot_values / tot_weights;
	}

	return true;
}

bool TreeExplainer::try_convert_trees() {
	//convert QRF, BART to generic_tree_model structure:
	if (original_predictor->classifier_type == MODEL_QRF)
		return convert_qrf_trees();
	else if (original_predictor->classifier_type == MODEL_LIGHTGBM)
		return convert_lightgbm_trees();
	else if (original_predictor->classifier_type == MODEL_XGB)
		return convert_xgb_trees();

	return false;
}

TreeExplainerMode TreeExplainer::get_mode() const {
	if (proxy_predictor != NULL)
		return TreeExplainerMode::PROXY_IMPL;
	else if (generic_tree_model.is_allocate)
		return TreeExplainerMode::CONVERTED_TREES_IMPL;
	else if (original_predictor != NULL)
		return TreeExplainerMode::ORIGINAL_IMPL;
	else
		MTHROW_AND_ERR("Error TreeExplainer::get_mode() - unspecified mode. have you called init with predicotr first?");
}

void TreeExplainer::_init(map<string, string> &mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		if (it->first == "proxy_model_type")
			proxy_model_type = it->second;
		else if (it->first == "proxy_model_init")
			proxy_model_init = it->second;
		else if (it->first == "interaction_shap")
			interaction_shap = stoi(it->second) > 0;
		else if (it->first == "approximate")
			approximate = stoi(it->second) > 0;
		else if (it->first == "missing_value")
			missing_value = med_stof(it->second);
		else if (it->first == "verbose")
			verbose = stoi(it->second) > 0;
		else
			MTHROW_AND_ERR("Error in TreeExplainer::init - Unsupported parameter \"%s\"\n", it->first.c_str());
	}
}

void TreeExplainer::post_deserialization() {
	if (original_predictor != NULL) {
		if (original_predictor->classifier_type == MODEL_XGB) {
			const int PRED_CONTRIBS = 4, APPROX_CONTRIBS = 8, INTERACTION_SHAP = 16;
			if (interaction_shap)
				static_cast<MedXGB *>(original_predictor)->feat_contrib_flags = PRED_CONTRIBS | INTERACTION_SHAP;
			if (approximate)
				static_cast<MedXGB *>(original_predictor)->feat_contrib_flags |= APPROX_CONTRIBS;
		}
		try_convert_trees();
	}

}

void TreeExplainer::init_post_processor(MedModel& model) {
	ModelExplainer::init_post_processor(model);
	post_deserialization();
}

void TreeExplainer::_learn(const MedFeatures &train_mat) {

	if (try_convert_trees())
		return; //success in convert to trees

	// Iterative mode only when working with coverted trees
	if (processing.iterative || approximate)
		MTHROW_AND_ERR("Cannot work in iterative mode when convertions failed. Please remove\n");

	// Cannot convert - use internal implementation or proxy
	if (processing.group2Inds.size() != train_mat.data.size() && processing.group_by_sum == 0) {
		processing.group_by_sum = 1;
		MWARN("Warning in TreeExplainer::Learn - no support for grouping in tree_shap not by sum. setting {group_by_sum:=1}\n");
	}

	if (original_predictor->classifier_type == MODEL_XGB) {
		const int PRED_CONTRIBS = 4, APPROX_CONTRIBS = 8, INTERACTION_SHAP = 16;
		if (interaction_shap)
			static_cast<MedXGB *>(original_predictor)->feat_contrib_flags = PRED_CONTRIBS | INTERACTION_SHAP;
		if (approximate)
			static_cast<MedXGB *>(original_predictor)->feat_contrib_flags |= APPROX_CONTRIBS;

		return; // no need to learn - will use XGB
	}
	if (original_predictor->classifier_type == MODEL_LIGHTGBM)
		return; // no need to learn - will use LigthGBM

	//Train XGboost model on model output.
	proxy_predictor = MedPredictor::make_predictor(proxy_model_type, proxy_model_init);
	//learn regression on input - TODO

	vector<float> labels_reg(train_mat.samples.size());
	MedMat<float> train_m;
	train_mat.get_as_matrix(train_m);
	if (proxy_predictor->transpose_for_predict != (train_m.transposed_flag > 0))
		train_m.transpose();
	original_predictor->predict(train_m, labels_reg);
	if (proxy_predictor->transpose_for_learn != (train_m.transposed_flag > 0))
		train_m.transpose();
	//proxy_predictor->prepare_x_mat(train_m);
	proxy_predictor->learn(train_m, labels_reg);
}

void conv_to_vec(const MedMat<float> &feat_contrib, vector<map<string, float>> &sample_explain_reasons) {
	sample_explain_reasons.resize(feat_contrib.nrows);
	for (int i = 0; i < sample_explain_reasons.size(); ++i)
	{
		map<string, float> &curr_sample_res = sample_explain_reasons[i];
		for (int j = 0; j < feat_contrib.ncols; ++j)
			curr_sample_res[feat_contrib.signals[j]] = feat_contrib(i, j);
	}
}

void TreeExplainer::explain(const MedFeatures &matrix, vector<map<string, float>> &sample_explain_reasons) const {
	TreeExplainerMode md = get_mode();
	MedMat<float> x_mat, feat_res;
	matrix.get_as_matrix(x_mat);

	vector<tfloat> shap_res; ///of size: Sample_Count, Features_count + 1(for bias/prior score), outputs_count
	ExplanationDataset data_set;
	vector<double> x, y, R;
	unique_ptr<bool[]> x_missing, R_missing;
	int M = x_mat.ncols;
	int num_Exp = M;
	int num_outputs = original_predictor->n_preds_per_sample();
	int sel_output_channel = 0;
	if (num_outputs > 1)
		MWARN("Warning in TreeExplainer::explain - has several prediction channels(%d) - will explain only %d\n",
			num_outputs, sel_output_channel);

	string bias_name = "Prior_Score";
	if (md == CONVERTED_TREES_IMPL) {
		x.resize(x_mat.size());
		y.resize(matrix.samples.size());

		x_missing = unique_ptr<bool[]>(new bool[x_mat.size()]);
		for (size_t i = 0; i < x_mat.size(); ++i)
		{
			x[i] = (double)x_mat.get_vec()[i];
			//x_missing.get()[i] = x_mat.m[i] == missing_value;
			x_missing.get()[i] = false; //In QRF no missing values
		}
		for (size_t i = 0; i < matrix.samples.size(); ++i)
			y[i] = (double)matrix.samples[i].outcome;
		int num_X = (int)y.size();


		//TODO: init R
		//R.resize(x_mat.m.size());
		tfloat *R_p = NULL; // R.data()
		R_missing = NULL;
		//R_missing = unique_ptr<bool>(new bool[x_mat.m.size()]);
		int num_R = 0;

		if (!processing.group_by_sum)
			num_Exp = (int)processing.group2Inds.size();

		data_set = ExplanationDataset(x.data(), x_missing.get(), y.data(), R_p, R_missing.get(), num_X, M, num_R, num_Exp);
		shap_res.assign(num_X * (num_Exp + 1)* num_outputs, 0);
	}

	int tree_dep = FEATURE_DEPENDENCE::tree_path_dependent; //global is not supported in python - so not completed yet. indepent is usefull for complex transform, but can't be run with interaction
	int tranform = MODEL_TRANSFORM::identity; //this will explain raw score, the rest are use to explain loss/probability or some tranformation, based on model return function

	switch (md)
	{
	case ORIGINAL_IMPL:
		original_predictor->calc_feature_contribs(x_mat, feat_res);
		conv_to_vec(feat_res, sample_explain_reasons);
		break;
	case PROXY_IMPL:
		proxy_predictor->calc_feature_contribs(x_mat, feat_res);
		conv_to_vec(feat_res, sample_explain_reasons);
		break;
	case CONVERTED_TREES_IMPL:
		if (!approximate) {
			// Build sets
			vector<string> names(num_Exp);
			vector<unsigned> feature_sets(x_mat.ncols);
			if (processing.group_by_sum) {
				for (unsigned i = 0; i < num_Exp; i++) {
					feature_sets[i] = i;
					names[i] = x_mat.signals[i];
				}
			}
			else {
				for (unsigned i = 0; i < num_Exp; i++) {
					for (unsigned j : processing.group2Inds[i])
						feature_sets[j] = i;
					names[i] = processing.groupNames[i];
				}
			}
			if (processing.iterative)
				iterative_tree_shap(generic_tree_model, data_set, shap_res.data(), tree_dep, tranform,
					interaction_shap, feature_sets.data(), verbose, names, processing.abs_cov_features, processing.iteration_cnt, processing.use_max_cov);
			else
				dense_tree_shap(generic_tree_model, data_set, shap_res.data(), tree_dep, tranform, interaction_shap, feature_sets.data());

			sample_explain_reasons.resize(matrix.samples.size());
#pragma omp parallel for
			for (int i = 0; i < sample_explain_reasons.size(); ++i)
			{
				map<string, float> &curr_exp = sample_explain_reasons[i];
				tfloat *curr_res_exp = &shap_res[i * (num_Exp + 1)  * num_outputs];
				//do only for sel_output_channel - the rest isn't supported yet
				for (size_t j = 0; j < num_Exp; ++j)
				{
					string &feat_name = names[j];
					curr_exp[feat_name] = (float)curr_res_exp[j * num_outputs + sel_output_channel];
				}
				curr_exp[bias_name] = (float)curr_res_exp[num_Exp * num_outputs + sel_output_channel];
			}
		}
		else {
			dense_tree_saabas(shap_res.data(), generic_tree_model, data_set);

			sample_explain_reasons.resize(matrix.samples.size());
#pragma omp parallel for
			for (int i = 0; i < sample_explain_reasons.size(); ++i)
			{
				map<string, float> &curr_exp = sample_explain_reasons[i];
				tfloat *curr_res_exp = &shap_res[i * (M + 1) * num_outputs];
				//do only for sel_output_channel - the rest isn't supported yet
				for (size_t j = 0; j < M; ++j)
				{
					string &feat_name = x_mat.signals[j];
					curr_exp[feat_name] = (float)curr_res_exp[j * num_outputs + sel_output_channel];
				}
				curr_exp[bias_name] = (float)curr_res_exp[M * num_outputs + sel_output_channel];
			}
		}

		break;
	default:
		MTHROW_AND_ERR("Error TreeExplainer::explain - Unsuppotrted mode %d\n", md);
	}
}

TreeExplainer::~TreeExplainer() {
	//TODO: use uniqe_ptr and than can remove those destructors..
	if (proxy_predictor != NULL) {
		delete proxy_predictor;
		proxy_predictor = NULL;
	}
	generic_tree_model.free();
}

MissingShapExplainer::~MissingShapExplainer() {
	//TODO: use uniqe_ptr and than can remove those destructors..
	if (retrain_predictor != NULL && !no_relearn) {
		delete retrain_predictor;
		retrain_predictor = NULL;
	}
}

template<typename T> int msn_count(const T *vals, int sz, T val) {
	int res = 0;
	for (size_t i = 0; i < sz; ++i)
		res += int(vals[i] == val);
	return res;
}

MissingShapExplainer::MissingShapExplainer() {
	processor_type = FTR_POSTPROCESS_MISSING_SHAP;
	max_test = 500;
	missing_value = MED_MAT_MISSING_VALUE;
	sample_masks_with_repeats = false;
	select_from_all = (float)0.8;
	uniform_rand = false;
	use_shuffle = false;
	add_new_data = 0;
	predictor_args = "";
	predictor_type = "";
	verbose_learn = true;
	no_relearn = false;
	avg_bias_score = 0;
	max_weight = 0;
	subsample_train = 0;
	limit_mask_size = 0;

	use_minimal_set = false;
	sort_params_a = 1;
	sort_params_b = 1;
	sort_params_k1 = 2;
	sort_params_k2 = 2;
	max_set_size = 10;
	override_score_bias = MED_MAT_MISSING_VALUE;
	verbose_apply = "";
	split_to_test = 0;
}

void MissingShapExplainer::_init(map<string, string> &mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		if (it->first == "missing_value")
			missing_value = med_stof(it->second);
		else if (it->first == "max_test")
			max_test = med_stoi(it->second);
		else if (it->first == "no_relearn")
			no_relearn = med_stoi(it->second) > 0;
		else if (it->first == "sample_masks_with_repeats")
			sample_masks_with_repeats = med_stoi(it->second) > 0;
		else if (it->first == "uniform_rand")
			uniform_rand = med_stoi(it->second) > 0;
		else if (it->first == "use_shuffle")
			use_shuffle = med_stoi(it->second) > 0;
		else if (it->first == "select_from_all")
			select_from_all = med_stof(it->second);
		else if (it->first == "add_new_data")
			add_new_data = med_stoi(it->second);
		else if (it->first == "predictor_type")
			predictor_type = it->second;
		else if (it->first == "predictor_args")
			predictor_args = it->second;
		else if (it->first == "verbose_learn")
			verbose_learn = stoi(it->second) > 0;
		else if (it->first == "max_weight")
			max_weight = med_stof(it->second);
		else if (it->first == "use_minimal_set")
			use_minimal_set = med_stoi(it->second) > 0;
		else if (it->first == "sort_params_a")
			sort_params_a = med_stof(it->second);
		else if (it->first == "sort_params_b")
			sort_params_b = med_stof(it->second);
		else if (it->first == "sort_params_k1")
			sort_params_k1 = med_stof(it->second);
		else if (it->first == "sort_params_k2")
			sort_params_k2 = med_stof(it->second);
		else if (it->first == "max_set_size")
			max_set_size = med_stoi(it->second);
		else if (it->first == "override_score_bias")
			override_score_bias = med_stof(it->second);
		else if (it->first == "subsample_train")
			subsample_train = med_stoi(it->second);
		else if (it->first == "limit_mask_size")
			limit_mask_size = med_stoi(it->second);
		else if (it->first == "verbose_apply")
			verbose_apply = it->second;
		else if (it->first == "split_to_test")
			split_to_test = med_stof(it->second);
		else
			MTHROW_AND_ERR("Error SHAPExplainer::init - Unknown param \"%s\"\n", it->first.c_str());
	}

	if (split_to_test >= 1)
		MTHROW_AND_ERR("Error - MissingShapExplainer::init - split_to_test should be < 1\n");

	if (sort_params_k1 < 1 || sort_params_k2 < 1)
		MTHROW_AND_ERR("Error - MissingShapExplainer::init - sort_params_k1,sort_params_k2 should be >= 1\n");

	if (uniform_rand && limit_mask_size > 0)
		MTHROW_AND_ERR("Error in MissingShapExplainer::_init - can't use uniform_rand and limit_mask_size > 0\n");
}

float get_avg_preds(const MedFeatures &train_mat, MedPredictor *original_predictor) {
	if (train_mat.samples.empty())
		MTHROW_AND_ERR("Error get_avg_preds learn matrix is empty\n");
	vector<float> preds_orig(train_mat.samples.size());
	float avg_bias_score = 0;
	if (train_mat.samples.front().prediction.empty()) {
		MedMat<float> mat_x;
		train_mat.get_as_matrix(mat_x);
		if (original_predictor->transpose_for_predict != (mat_x.transposed_flag > 0))
			mat_x.transpose();
		original_predictor->predict(mat_x, preds_orig);
	}
	else {
		for (size_t i = 0; i < preds_orig.size(); ++i)
			preds_orig[i] = train_mat.samples[i].prediction[0];
	}
	for (size_t i = 0; i < preds_orig.size(); ++i)
		avg_bias_score += preds_orig[i];
	avg_bias_score /= preds_orig.size();
	return avg_bias_score;
}

void MissingShapExplainer::_learn(const MedFeatures &train_mat) {
	avg_bias_score = get_avg_preds(train_mat, original_predictor);
	if (no_relearn) {
		retrain_predictor = original_predictor;
		return;
	}

	if (limit_mask_size >= processing.group2Inds.size()) {
		MWARN("WARNING: limit_mask_size=%d which is bigger than number of groups/features(%zu)\n",
			limit_mask_size, processing.group2Inds.size());
		limit_mask_size = (int)processing.group2Inds.size(); //problem with arguments
	}

	if (predictor_type.empty())
		retrain_predictor = (MedPredictor *)medial::models::copyInfraModel(original_predictor, false);
	else
		retrain_predictor = MedPredictor::make_predictor(predictor_type, predictor_args);

	mt19937 gen(globalRNG::rand());
	MedMat<float> x_mat;
	int nftrs_grp = (int)processing.group2Inds.size();
	int nftrs = (int)train_mat.data.size();

	vector<int> missing_hist(nftrs + 1), added_missing_hist(nftrs + 1), added_grp_hist(nftrs_grp + 1);
	vector<float> labels;
	//manipulate train_mat if needed:
	int train_mat_size = (int)train_mat.samples.size();
	MedFeatures collected_test;
	MedMat<float> test_mat;
	if (verbose_learn && split_to_test > 0) {
		//split train_mat and populate collected_test:
		uniform_int_distribution<> rnd_selection(0, train_mat_size - 1);
		int test_size = int(split_to_test * train_mat_size);
		if (test_size > train_mat_size)
			MTHROW_AND_ERR("Error MissingShapExplainer::_learn - test size is biger than train size\n");
		if (test_size <= 0)
			MTHROW_AND_ERR("Error MissingShapExplainer::_learn - test size is empty\n");
		vector<int> sel_idx;
		vector<bool> seen_row(train_mat_size);
		for (size_t i = 0; i < test_size; ++i)
		{
			int rnd_sel = rnd_selection(gen);
			while (seen_row[rnd_sel])
				rnd_sel = rnd_selection(gen);
			seen_row[rnd_sel] = true;
			sel_idx.push_back(rnd_sel);
		}
		collected_test = train_mat;
		MedFeatures train_split = train_mat;
		medial::process::filter_row_indexes(collected_test, sel_idx);
		medial::process::filter_row_indexes(train_split, sel_idx, true);
		if (collected_test.samples.front().prediction.empty())
			original_predictor->predict(collected_test);

		collected_test.get_as_matrix(test_mat);
		//set masks on collected_test:
		for (size_t i = 0; i < collected_test.samples.size(); ++i)
		{
			vector<bool> curr_mask(nftrs_grp);
			for (int j = 0; j < nftrs_grp; ++j) {
				bool has_missing = false;
				for (size_t k = 0; k < processing.group2Inds[j].size() && !has_missing; ++k)
					has_missing = test_mat(i, processing.group2Inds[j][k]) == missing_value;
				curr_mask[j] = !has_missing;
			}
			medial::shapley::generate_mask_(curr_mask, nftrs_grp, gen, uniform_rand, 0.5, use_shuffle, limit_mask_size);
			//commit mask:
			for (int j = 0; j < nftrs_grp; ++j)
				if (!curr_mask[j]) {
					for (size_t k = 0; k < processing.group2Inds[j].size(); ++k)
						test_mat(i, processing.group2Inds[j][k]) = missing_value;
				}
		}

		train_mat_size = (int)train_split.samples.size();
		train_split.get_as_matrix(x_mat);

		labels.resize(train_mat_size);
		if (!train_split.samples.front().prediction.empty())
			for (size_t i = 0; i < labels.size(); ++i)
				labels[i] = train_split.samples[i].prediction[0];
		else {
			MedMat<float> tt;
			train_split.get_as_matrix(tt);
			original_predictor->predict(tt, labels);
		}

	}
	else {
		train_mat.get_as_matrix(x_mat);

		labels.resize(train_mat_size);
		if (!train_mat.samples.front().prediction.empty())
			for (size_t i = 0; i < labels.size(); ++i)
				labels[i] = train_mat.samples[i].prediction[0];
		else {
			MedMat<float> tt;
			train_mat.get_as_matrix(tt);
			original_predictor->predict(tt, labels);
		}
	}


	vector<int> miss_cnts(train_mat_size + add_new_data);
	vector<float>weights(train_mat_size + add_new_data, 1);
	vector<int> mask_group_sizes(train_mat_size + add_new_data); //stores for each sample - how many missings in groups manner:
	for (size_t i = 0; i < train_mat_size; ++i)
	{
		//check how many groups missings:
		int grp_misses = 0;
		for (int j = 0; j < nftrs_grp; ++j) {
			bool has_missing = false;
			for (size_t k = 0; k < processing.group2Inds[j].size() && !has_missing; ++k)
				has_missing = x_mat(i, processing.group2Inds[j][k]) == missing_value;
			grp_misses += int(has_missing);
		}
		mask_group_sizes[i] = grp_misses;
	}

	if (add_new_data > 0) {
		//processing.group2Inds.size()
		vector<float> rows_m(add_new_data * nftrs);
		unordered_set<vector<bool>> seen_mask;
		uniform_int_distribution<> rnd_row(0, train_mat_size - 1);
		double log_max_opts = log(add_new_data) / log(2.0);
		if (log_max_opts >= nftrs_grp) {
			if (!sample_masks_with_repeats)
				MWARN("Warning: you have request to sample masks without repeats, but it can't be done. setting sample with repeats\n");
			sample_masks_with_repeats = true;
		}
		if (verbose_learn)
			MLOG("Adding %d Data points (has %d features with %d groups)\n", add_new_data, nftrs, nftrs_grp);
		MedProgress add_progress("Add_Train_Data", add_new_data, 30, 1);
		for (size_t i = 0; i < add_new_data; ++i)
		{
			float *curr_row = &rows_m[i *  nftrs];
			//select row:
			int row_sel = rnd_row(gen);

			vector<bool> curr_mask; curr_mask.resize(nftrs_grp);
			for (int j = 0; j < nftrs_grp; ++j) {
				bool has_missing = false;
				for (size_t k = 0; k < processing.group2Inds[j].size() && !has_missing; ++k)
					has_missing = x_mat(row_sel, processing.group2Inds[j][k]) == missing_value;
				curr_mask[j] = !has_missing;
			}

			medial::shapley::generate_mask_(curr_mask, nftrs_grp, gen, uniform_rand, 0.5, use_shuffle, limit_mask_size);
			while (!sample_masks_with_repeats && seen_mask.find(curr_mask) != seen_mask.end())
				medial::shapley::generate_mask_(curr_mask, nftrs_grp, gen, uniform_rand, 0.5, use_shuffle, limit_mask_size);
			if (!sample_masks_with_repeats)
				seen_mask.insert(curr_mask);

			//commit mask to curr_row
			int msn_cnt = 0;
			for (int j = 0; j < nftrs_grp; ++j)
			{
				if (curr_mask[j]) {
					for (size_t k = 0; k < processing.group2Inds[j].size(); ++k)
						curr_row[processing.group2Inds[j][k]] = x_mat(row_sel, processing.group2Inds[j][k]);
				}
				else {
					for (size_t k = 0; k < processing.group2Inds[j].size(); ++k)
						curr_row[processing.group2Inds[j][k]] = missing_value;
				}
				msn_cnt += int(!curr_mask[j]); //how many missings
			}
			labels.push_back(labels[row_sel]);
			++added_grp_hist[msn_cnt];
			add_progress.update();
			mask_group_sizes[train_mat_size + i] = msn_cnt;
		}
		x_mat.add_rows(rows_m);
	}

	// Add data with missing values according to sample masks
	vector<int> grp_missing_hist_all(nftrs_grp + 1);

	for (int i = 0; i < x_mat.nrows; ++i) {
		miss_cnts[i] = msn_count<float>(x_mat.data_ptr(i, 0), nftrs, missing_value);
		++missing_hist[miss_cnts[i]];
		if (i >= train_mat_size)
			++added_missing_hist[miss_cnts[i]];
		++grp_missing_hist_all[mask_group_sizes[i]];
	}
	for (size_t i = 0; i < x_mat.nrows; ++i) {
		float curr_mask_w = x_mat.nrows / float(grp_missing_hist_all[mask_group_sizes[i]]);
		weights[i] = curr_mask_w;
	}
	if (max_weight > 0) {
		float min_weight = 0;
		if (!weights.empty())
			min_weight = weights[0];
		for (size_t i = 1; i < weights.size(); ++i)
			if (weights[i] < min_weight)
				min_weight = weights[i];
		//normalize be max:
		if (min_weight > 0)
			for (size_t i = 1; i < weights.size(); ++i) {
				weights[i] /= min_weight;
				if (weights[i] > max_weight)
					weights[i] = max_weight;
			}
	}
	if (verbose_learn) {
		medial::print::print_hist_vec(miss_cnts, "missing_values_cnt percentiles [0 - " + to_string(nftrs) + "] (with added samples - no groups)", "%d");
		medial::print::print_hist_vec(mask_group_sizes, "mask_group_sizes percentiles [0 - " + to_string(nftrs_grp) + "] (with added samples - for groups)", "%d");
		medial::print::print_hist_vec(added_missing_hist, "selected counts in hist of missing_values_cnt (only for added - no groups)", "%d");
		if (added_grp_hist.size() < 50)
			medial::print::print_vec(added_grp_hist, "grp hist (only for added - on groups)", "%d");
		else
			medial::print::print_hist_vec(added_grp_hist, "hist of added_grp_hist (only for added - on groups)", "%d");
		medial::print::print_hist_vec(weights, "weights for learn", "%2.4f");
	}
	if (original_predictor->transpose_for_learn != (x_mat.transposed_flag > 0))
		x_mat.transpose();
	//reweight train_mat:
	if (predictor_type.empty() && !predictor_args.empty())
		retrain_predictor->init_from_string(predictor_args);

	if (subsample_train > 0 && subsample_train < train_mat_size) {
		//do subsampling:
		MLOG("INFO:: MissingShapExplainer::_learn - subsampling original train matrix");
		unordered_set<int> selected_idx;

		uniform_int_distribution<> rnd_opts(0, train_mat_size - 1);
		for (size_t i = 0; i < subsample_train; ++i)
		{
			int sel_idx = rnd_opts(gen);
			while (selected_idx.find(sel_idx) != selected_idx.end())
				sel_idx = rnd_opts(gen);
			selected_idx.insert(sel_idx);
		}
		//add all rest:
		vector<int> empty_is_all;
		vector<int> selected_idx_vec(selected_idx.begin(), selected_idx.end());
		for (int i = train_mat_size; i < x_mat.nrows; ++i)
			selected_idx_vec.push_back(i);
		sort(selected_idx_vec.begin(), selected_idx_vec.end());

		//commit selection xmat and labels, weights:
		vector<float> new_weights(selected_idx_vec.size()), new_labels(selected_idx_vec.size());
		x_mat.get_sub_mat(selected_idx_vec, empty_is_all);
		for (size_t i = 0; i < selected_idx_vec.size(); ++i)
		{
			new_labels[i] = labels[selected_idx_vec[i]];
			new_weights[i] = weights[selected_idx_vec[i]];
		}
		labels = std::move(new_labels);
		weights = std::move(new_weights);
	}

	retrain_predictor->learn(x_mat, labels, weights);
	//test pref:
	if (verbose_learn) {
		vector<float> train_p;
		retrain_predictor->predict(x_mat, train_p);
		float rmse = medial::performance::rmse_without_cleaning(train_p, labels, &weights);
		float rmse_no_weights = medial::performance::rmse_without_cleaning(train_p, labels);
		float mean_pred, std_labels;
		medial::stats::get_mean_and_std_without_cleaning(labels, mean_pred, std_labels);
		float r_square = MED_MAT_MISSING_VALUE;
		float r_square_no = MED_MAT_MISSING_VALUE;
		if (std_labels > 0) {
			r_square = 1 - (rmse / std_labels);
			r_square_no = 1 - (rmse_no_weights / std_labels);
		}
		MLOG("RMSE=%2.4f, RMSE(no weights)=%2.4f on train for model, R_Square=%2.3f, R_Square(no weights)=%2.3f\n",
			rmse, rmse_no_weights, r_square, r_square_no);

		if (split_to_test > 0) {
			//test also on test (collected_test):
			vector<float> test_p;
			retrain_predictor->predict(test_mat, test_p);
			vector<float> labels_test(collected_test.samples.size()), stable_pred(collected_test.samples.size(), mean_pred);
			for (size_t i = 0; i < labels_test.size(); ++i)
				labels_test[i] = collected_test.samples[i].prediction[0];

			float rmse_test = medial::performance::rmse_without_cleaning(test_p, labels_test);
			float rmse_test_st = medial::performance::rmse_without_cleaning(stable_pred, labels_test);
			float r_square_2 = -1;
			if (rmse_test_st > 0)
				r_square_2 = 1 - (rmse_test / rmse_test_st);
			MLOG("RMSE=%2.4f on test for model, rmse_for_prior = %2.4f, R_Square=%2.3f\n",
				rmse_test, rmse_test_st, r_square_2);
		}
	}
}

void MissingShapExplainer::explain(const MedFeatures &matrix, vector<map<string, float>> &sample_explain_reasons) const {
	MedPredictor *predictor = retrain_predictor;
	if (no_relearn)
		predictor = original_predictor;
	const vector<vector<int>> *group_inds = &processing.group2Inds;
	const vector<string> *group_names = &processing.groupNames;
	vector<vector<int>> group_inds_loc;
	vector<string> group_names_loc;
	if (processing.group_by_sum) {
		MWARN("WARN :: MissingShapExplainer called with group_by_sum and it has it's own logic\n");
		int icol = 0;
		for (auto& rec : matrix.data) {
			group_inds_loc.push_back({ icol++ });
			group_names_loc.push_back(rec.first);
		}
		group_inds = &group_inds_loc;
		group_names = &group_names_loc;
	}

	sample_explain_reasons.resize(matrix.samples.size());
	string bias_name = "Prior_Score";
	vector<float> preds_orig(matrix.samples.size());
	if (matrix.samples.front().prediction.empty()) {
		MedMat<float> mat_x;
		matrix.get_as_matrix(mat_x);
		if (predictor->transpose_for_predict != (mat_x.transposed_flag > 0))
			mat_x.transpose();
		predictor->predict(mat_x, preds_orig);
	}
	else {
		for (size_t i = 0; i < preds_orig.size(); ++i)
			preds_orig[i] = matrix.samples[i].prediction[0];
	}
	int N_TOTAL_TH = omp_get_max_threads();

	bool outer_parallel = matrix.samples.size() >= N_TOTAL_TH;
	MedProgress progress("MissingShapley", (int)matrix.samples.size(), 15);
	//make thread safe create copy of predictor
	vector<MedPredictor *> pred_threads(1);
	vector<mt19937> gen_threads(1);
	random_device rd;
	if (outer_parallel) {
		pred_threads.resize(N_TOTAL_TH);
		gen_threads.resize(N_TOTAL_TH);
		size_t sz_pred = predictor->get_size();
		unsigned char *blob_pred = new unsigned char[sz_pred];
		predictor->serialize(blob_pred);
		for (size_t i = 0; i < pred_threads.size(); ++i)
		{
			pred_threads[i] = (MedPredictor *)medial::models::copyInfraModel(predictor, false);
			pred_threads[i]->deserialize(blob_pred);
			gen_threads[i] = mt19937(rd());
			if (use_minimal_set)
				pred_threads[i]->prepare_predict_single();
		}
		delete[]blob_pred;
	}
	else
		gen_threads[0] = mt19937(rd());

	float use_bias = avg_bias_score;
	if (override_score_bias != MED_MAT_MISSING_VALUE)
		use_bias = override_score_bias;
	vector<const vector<float> *> data_pointer(matrix.data.size());
	vector<string> feat_names;
	matrix.get_feature_names(feat_names);
	int ind_i = 0;
	ofstream fw_apply;
	if (!verbose_apply.empty()) {
		fw_apply.open(verbose_apply);
		if (!fw_apply.good())
			MWARN("WARN : can't open file %s for verbose_apply\n");
	}

	for (auto it = matrix.data.begin(); it != matrix.data.end(); ++it)
	{
		data_pointer[ind_i] = &it->second;
		++ind_i;
	}

#pragma omp parallel for if (outer_parallel)
	for (int i = 0; i < matrix.samples.size(); ++i)
	{
		int th_n;
		vector<float> features_coeff;
		vector<float> score_history;
		float pred_shap = 0;
		MedPredictor *curr_p = predictor;
		if (outer_parallel) {
			th_n = omp_get_thread_num();
			curr_p = pred_threads[th_n];
		}
		else
			th_n = 0;

		if (!use_minimal_set)
			medial::shapley::explain_shapley(matrix, (int)i, max_test, curr_p, missing_value, *group_inds, *group_names,
				features_coeff, gen_threads[th_n], sample_masks_with_repeats, select_from_all,
				uniform_rand, use_shuffle, global_logger.levels[LOCAL_SECTION] < LOG_DEF_LEVEL && !outer_parallel);
		else {
			medial::shapley::explain_minimal_set(matrix, (int)i, 1, curr_p, missing_value,
				*group_inds, features_coeff, score_history, max_set_size, use_bias, sort_params_a, sort_params_b,
				sort_params_k1, sort_params_k2, global_logger.levels[LOCAL_SECTION] < LOG_DEF_LEVEL && !outer_parallel);

			if (!verbose_apply.empty()) {
#pragma omp critical 
				{
					char buffer_out[8000];
					//debug prints:
					snprintf(buffer_out, sizeof(buffer_out), "pid %d, time %d, score %2.5f (%zu) baseline %2.5f:\n",
						matrix.samples[i].id, matrix.samples[i].time, matrix.samples[i].prediction[0], score_history.size(),
						use_bias);
					fw_apply << string(buffer_out);
					for (int j = 0; j < score_history.size(); ++j)
					{
						//remove 0 - find from 1 to max_set in abs:
						int search_term = j + 1;
						int grp_idx = -1;
						for (int k = 0; k < features_coeff.size() && grp_idx < 0; ++k)
							if (int(abs(features_coeff[k])) == search_term)
								grp_idx = k;

						if (grp_idx < 0) {
							//snprintf(buffer_out, sizeof(buffer_out), "Done\n");
							//fw_apply << string(buffer_out);
							break;
						}

						string contrib_str = "POSITIVE";
						if (features_coeff[grp_idx] < 0)
							contrib_str = "NEGATIVE";
						int first_idx_grp = group_inds->at(grp_idx)[0];
						snprintf(buffer_out, sizeof(buffer_out), "\t%d. Group %s(%s=%f) :: After_Score= %2.5f :: %s\n",
							search_term, group_names->at(grp_idx).c_str(), feat_names[first_idx_grp].c_str(),
							data_pointer[first_idx_grp]->at(i), score_history[j], contrib_str.c_str());
						fw_apply << string(buffer_out);
					}
				}
			}

			//reverse order in features_coeff:
			int ind_score_hist = 0;
			vector<pair<int, float>> tp(features_coeff.size());
			for (int j = 0; j < tp.size(); ++j)
			{
				tp[j].first = j;
				tp[j].second = features_coeff[j];
			}
			sort(tp.begin(), tp.end(), [](const pair<int, float>&a, const pair<int, float> &b) {
				return abs(a.second) < abs(b.second); }); //0 are ignored
			for (size_t j = 0; j < tp.size(); ++j)
			{
				if (tp[j].second == 0)
					continue;
				bool positive_contrib = tp[j].second > 0;
				features_coeff[tp[j].first] = float((int)features_coeff.size() + 1 - abs(tp[j].second));
				double diff = abs(score_history[ind_score_hist] - (ind_score_hist > 0 ? score_history[ind_score_hist - 1] : use_bias));
				if (diff > 1)
					diff = 0.99999;
				features_coeff[tp[j].first] += diff;
				if (!positive_contrib)
					features_coeff[tp[j].first] = -features_coeff[tp[j].first];
				++ind_score_hist;
			}
		}

		for (size_t j = 0; j < features_coeff.size(); ++j)
			pred_shap += features_coeff[j];

#pragma omp critical 
		{
			map<string, float> &curr_res = sample_explain_reasons[i];
			for (size_t j = 0; j < group_names->size(); ++j)
				curr_res[group_names->at(j)] = features_coeff[j];
			//Add prior to score:
			//curr_res[bias_name] = preds_orig[i] - pred_shap; //that will sum to current score
			curr_res[bias_name] = avg_bias_score; //that will sum to current score
		}

		progress.update();
	}
	if (outer_parallel)
		for (size_t i = 0; i < pred_threads.size(); ++i)
			delete pred_threads[i];

	if (!verbose_apply.empty())
		fw_apply.close();
}

void ShapleyExplainer::_init(map<string, string> &mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		if (it->first == "gen_type")
			gen_type = GeneratorType_fromStr(it->second);
		else if (it->first == "generator_args")
			generator_args = it->second;
		else if (it->first == "missing_value")
			missing_value = med_stof(it->second);
		else if (it->first == "n_masks")
			n_masks = med_stoi(it->second);
		else if (it->first == "sampling_args")
			sampling_args = it->second;
		else if (it->first == "use_random_sampling")
			use_random_sampling = med_stoi(it->second) > 0;
		else
			MTHROW_AND_ERR("Error in ShapleyExplainer::init - Unsupported param \"%s\"\n", it->first.c_str());
	}
	init_sampler(); //from args
}

void ShapleyExplainer::init_sampler(bool with_sampler) {
	switch (gen_type)
	{
	case GIBBS:
		if (with_sampler) {
			_gibbs.init_from_string(generator_args);
			_sampler = unique_ptr<SamplesGenerator<float>>(new GibbsSamplesGenerator<float>(_gibbs, true));
		}
		_gibbs_sample_params.init_from_string(sampling_args);
		sampler_sampling_args = &_gibbs_sample_params;
		break;
	case GAN:
		if (with_sampler) {
			_sampler = unique_ptr<SamplesGenerator<float>>(new MaskedGAN<float>);
			static_cast<MaskedGAN<float> *>(_sampler.get())->read_from_text_file(generator_args);
			static_cast<MaskedGAN<float> *>(_sampler.get())->mg_params.init_from_string(sampling_args);
		}
		break;
	case MISSING:
		if (with_sampler)
			_sampler = unique_ptr<SamplesGenerator<float>>(new MissingsSamplesGenerator<float>(missing_value));
		break;
	case RANDOM_DIST:
		if (with_sampler)
			_sampler = unique_ptr<SamplesGenerator<float>>(new RandomSamplesGenerator<float>(0, 5));
		sampler_sampling_args = &n_masks;
		break;
	default:
		MTHROW_AND_ERR("Error in ShapleyExplainer::init_sampler() - Unsupported Type %d\n", gen_type);
	}
}

void ShapleyExplainer::_learn(const MedFeatures &train_mat) {
	_sampler->learn(train_mat.data);
	avg_bias_score = get_avg_preds(train_mat, original_predictor);
}

void ShapleyExplainer::explain(const MedFeatures &matrix, vector<map<string, float>> &sample_explain_reasons) const {
	sample_explain_reasons.resize(matrix.samples.size());
	string bias_name = "Prior_Score";
	vector<float> preds_orig(matrix.samples.size());
	if (matrix.samples.front().prediction.empty()) {
		MedMat<float> mat_x;
		matrix.get_as_matrix(mat_x);
		if (original_predictor->transpose_for_predict != (mat_x.transposed_flag > 0))
			mat_x.transpose();
		original_predictor->predict(mat_x, preds_orig);
	}
	else {
		for (size_t i = 0; i < preds_orig.size(); ++i)
			preds_orig[i] = matrix.samples[i].prediction[0];
	}

	const vector<vector<int>> *group_inds = &processing.group2Inds;
	const vector<string> *group_names = &processing.groupNames;
	vector<vector<int>> group_inds_loc;
	vector<string> group_names_loc;
	if (processing.group_by_sum) {
		int icol = 0;
		for (auto& rec : matrix.data) {
			group_inds_loc.push_back({ icol++ });
			group_names_loc.push_back(rec.first);
		}
		group_inds = &group_inds_loc;
		group_names = &group_names_loc;
	}

	int MAX_Threads = omp_get_max_threads();
	//copy sample for each thread:
	random_device rd;
	vector<mt19937> gen_thread(MAX_Threads);
	vector<MedPredictor *> predictor_cp(MAX_Threads);
	for (size_t i = 0; i < gen_thread.size(); ++i)
		gen_thread[i] = mt19937(globalRNG::rand());
	_sampler->prepare(sampler_sampling_args);
	size_t sz_pred = original_predictor->get_size();
	unsigned char *blob_pred = new unsigned char[sz_pred];
	original_predictor->serialize(blob_pred);
	for (size_t i = 0; i < predictor_cp.size(); ++i) {
		predictor_cp[i] = (MedPredictor *)medial::models::copyInfraModel(original_predictor, false);
		predictor_cp[i]->deserialize(blob_pred);
	}
	delete[]blob_pred;

	MedProgress progress("ShapleyExplainer", (int)matrix.samples.size(), 15);
#pragma omp parallel for if (matrix.samples.size() >= 2)
	for (int i = 0; i < matrix.samples.size(); ++i)
	{
		int n_th = omp_get_thread_num();
		vector<float> features_coeff;
		float pred_shap = 0;
		medial::shapley::explain_shapley(matrix, (int)i, n_masks, predictor_cp[n_th]
			, *group_inds, *group_names, *_sampler, gen_thread[n_th], 1, sampler_sampling_args, features_coeff,
			use_random_sampling, global_logger.levels[LOCAL_SECTION] < LOCAL_LEVEL &&
			(!(matrix.samples.size() >= 2) || omp_get_thread_num() == 1));

		for (size_t j = 0; j < features_coeff.size(); ++j)
			pred_shap += features_coeff[j];

#pragma omp critical 
		{
			map<string, float> &curr_res = sample_explain_reasons[i];
			for (size_t j = 0; j < group_names->size(); ++j)
				curr_res[group_names->at(j)] = features_coeff[j];
			//Add prior to score:
			//curr_res[bias_name] = preds_orig[i] - pred_shap; //that will sum to current score
			curr_res[bias_name] = avg_bias_score;
		}

		progress.update();
	}
	for (size_t i = 0; i < predictor_cp.size(); ++i)
		delete predictor_cp[i];
}

void ShapleyExplainer::post_deserialization() {
	init_sampler(false);
}

void ShapleyExplainer::load_GIBBS(MedPredictor *original_pred, const GibbsSampler<float> &gibbs, const GibbsSamplingParams &sampling_args) {
	this->original_predictor = original_pred;
	_gibbs = gibbs;
	_gibbs_sample_params = sampling_args;

	sampler_sampling_args = &_gibbs_sample_params;
	_sampler = unique_ptr<SamplesGenerator<float>>(new GibbsSamplesGenerator<float>(_gibbs, true));

	gen_type = GeneratorType::GIBBS;
}

void ShapleyExplainer::load_GAN(MedPredictor *original_pred, const string &gan_path) {
	this->original_predictor = original_pred;
	_sampler = unique_ptr<SamplesGenerator<float>>(new MaskedGAN<float>);
	static_cast<MaskedGAN<float> *>(_sampler.get())->read_from_text_file(gan_path);
	static_cast<MaskedGAN<float> *>(_sampler.get())->mg_params.init_from_string(sampling_args);

	gen_type = GeneratorType::GAN;
}

void ShapleyExplainer::load_MISSING(MedPredictor *original_pred) {
	this->original_predictor = original_pred;
	_sampler = unique_ptr<SamplesGenerator<float>>(new MissingsSamplesGenerator<float>(missing_value));
	gen_type = GeneratorType::MISSING;
}

void ShapleyExplainer::load_sampler(MedPredictor *original_pred, unique_ptr<SamplesGenerator<float>> &&generator) {
	this->original_predictor = original_pred;
	_sampler = std::move(generator);
}

void ShapleyExplainer::dprint(const string &pref) const {
	string predictor_nm = "";
	if (original_predictor != NULL)
		predictor_nm = original_predictor->my_class_name();
	string filters_str = "", processing_str = "";
	char buffer[5000];
	snprintf(buffer, sizeof(buffer), "group_by_sum=%d, learn_cov_matrix=%d, normalize_vals=%d, zero_missing=%d, grouping=%s",
		int(processing.group_by_sum), int(processing.learn_cov_matrix), processing.normalize_vals
		, processing.zero_missing, processing.grouping.c_str());
	processing_str = string(buffer);
	snprintf(buffer, sizeof(buffer), "sort_mode=%d, max_count=%d, sum_ratio=%2.3f",
		filters.sort_mode, filters.max_count, filters.sum_ratio);
	filters_str = string(buffer);

	MLOG("%s :: ModelExplainer type %d(%s), original_predictor=%s, gen_type=%s, attr_name=%s, processing={%s}, filters={%s}\n",
		pref.c_str(), processor_type, my_class_name().c_str(), predictor_nm.c_str(),
		GeneratorType_toStr(gen_type).c_str(), global_explain_params.attr_name.c_str(),
		processing_str.c_str(), filters_str.c_str());
}

void LimeExplainer::_init(map<string, string> &mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		if (it->first == "gen_type")
			gen_type = GeneratorType_fromStr(it->second);
		else if (it->first == "generator_args")
			generator_args = it->second;
		else if (it->first == "missing_value")
			missing_value = med_stof(it->second);
		else if (it->first == "sampling_args")
			sampling_args = it->second;
		else if (it->first == "p_mask")
			p_mask = med_stof(it->second);
		else if (it->first == "weight")
			weighting = get_weight_method(it->second);
		else if (it->first == "n_masks")
			n_masks = med_stoi(it->second);
		else
			MTHROW_AND_ERR("Error in LimeExplainer::init - Unsupported param \"%s\"\n", it->first.c_str());
	}
	init_sampler(); //from args
}

medial::shapley::LimeWeightMethod LimeExplainer::get_weight_method(string method_s) {

	boost::to_lower(method_s);
	if (method_s == "lime") return medial::shapley::LimeWeightLime;
	if (method_s == "unif" || method_s == "uniform") return medial::shapley::LimeWeightUniform;
	if (method_s == "shap" || method_s == "shapley") return medial::shapley::LimeWeightShap;
	if (method_s == "sum" || method_s == "shap_sum" || method_s == "shapley_sum") return medial::shapley::LimeWeightSum;

	MTHROW_AND_ERR("Unknown weighting method %s for LIME explainer\n", method_s.c_str());
	return medial::shapley::LimeWeightLast;
}

void LimeExplainer::init_sampler(bool with_sampler) {
	switch (gen_type)
	{
	case GIBBS:
		if (with_sampler) {
			_gibbs.init_from_string(generator_args);
			_sampler = unique_ptr<SamplesGenerator<float>>(new GibbsSamplesGenerator<float>(_gibbs, true));
		}
		_gibbs_sample_params.init_from_string(sampling_args);
		sampler_sampling_args = &_gibbs_sample_params;
		break;
	case GAN:
		if (with_sampler) {
			_sampler = unique_ptr<SamplesGenerator<float>>(new MaskedGAN<float>);
			static_cast<MaskedGAN<float> *>(_sampler.get())->read_from_text_file(generator_args);
			static_cast<MaskedGAN<float> *>(_sampler.get())->mg_params.init_from_string(sampling_args);
		}
		break;
	case MISSING:
		if (with_sampler)
			_sampler = unique_ptr<SamplesGenerator<float>>(new MissingsSamplesGenerator<float>(missing_value));
		break;
	default:
		MTHROW_AND_ERR("Error in LimeExplainer::init_sampler() - Unsupported Type %d\n", gen_type);
	}
}

void LimeExplainer::load_GIBBS(MedPredictor *original_pred, const GibbsSampler<float> &gibbs, const GibbsSamplingParams &sampling_args) {
	this->original_predictor = original_pred;
	_gibbs = gibbs;
	_gibbs_sample_params = sampling_args;

	sampler_sampling_args = &_gibbs_sample_params;
	_sampler = unique_ptr<SamplesGenerator<float>>(new GibbsSamplesGenerator<float>(_gibbs, true));

	gen_type = GeneratorType::GIBBS;
}

void LimeExplainer::load_GAN(MedPredictor *original_pred, const string &gan_path) {
	this->original_predictor = original_pred;
	_sampler = unique_ptr<SamplesGenerator<float>>(new MaskedGAN<float>);
	static_cast<MaskedGAN<float> *>(_sampler.get())->read_from_text_file(gan_path);
	static_cast<MaskedGAN<float> *>(_sampler.get())->mg_params.init_from_string(sampling_args);

	gen_type = GeneratorType::GAN;
}

void LimeExplainer::load_MISSING(MedPredictor *original_pred) {
	this->original_predictor = original_pred;
	_sampler = unique_ptr<SamplesGenerator<float>>(new MissingsSamplesGenerator<float>(missing_value));
	gen_type = GeneratorType::MISSING;
}

void LimeExplainer::load_sampler(MedPredictor *original_pred, unique_ptr<SamplesGenerator<float>> &&generator) {
	this->original_predictor = original_pred;
	_sampler = std::move(generator);
}

void LimeExplainer::post_deserialization() {
	init_sampler(false);
}

void LimeExplainer::_learn(const MedFeatures &train_mat) {
	_sampler->learn(train_mat.data);
}

void LimeExplainer::explain(const MedFeatures &matrix, vector<map<string, float>> &sample_explain_reasons) const {
	vector<vector<float>> alphas;

	const vector<vector<int>> *group_inds = &processing.group2Inds;
	const vector<string> *group_names = &processing.groupNames;
	vector<vector<int>> group_inds_loc;
	vector<string> group_names_loc;
	if (processing.group_by_sum) {
		int icol = 0;
		for (auto& rec : matrix.data) {
			group_inds_loc.push_back({ icol++ });
			group_names_loc.push_back(rec.first);
		}
		group_inds = &group_inds_loc;
		group_names = &group_names_loc;
	}

	if (processing.iterative)
		medial::shapley::get_iterative_shapley_lime_params(matrix, original_predictor, _sampler.get(), p_mask, n_masks, weighting, missing_value,
			sampler_sampling_args, *group_inds, *group_names, processing.abs_cov_features, processing.iteration_cnt, alphas, processing.use_max_cov);
	else
		medial::shapley::get_shapley_lime_params(matrix, original_predictor, _sampler.get(), p_mask, n_masks, weighting, missing_value,
			sampler_sampling_args, *group_inds, *group_names, alphas);

	sample_explain_reasons.resize(matrix.samples.size());

#pragma omp parallel for
	for (int i = 0; i < sample_explain_reasons.size(); ++i)
	{
		map<string, float> &curr = sample_explain_reasons[i];
		const vector<float> &curr_res = alphas[i];
		for (size_t k = 0; k < group_names->size(); ++k)
			curr[group_names->at(k)] = curr_res[k];
	}
}

void LimeExplainer::dprint(const string &pref) const {
	string predictor_nm = "";
	if (original_predictor != NULL)
		predictor_nm = original_predictor->my_class_name();
	string filters_str = "", processing_str = "";
	char buffer[5000];
	snprintf(buffer, sizeof(buffer), "group_by_sum=%d, learn_cov_matrix=%d, normalize_vals=%d, zero_missing=%d, grouping=%s",
		int(processing.group_by_sum), int(processing.learn_cov_matrix), processing.normalize_vals
		, processing.zero_missing, processing.grouping.c_str());
	processing_str = string(buffer);
	snprintf(buffer, sizeof(buffer), "sort_mode=%d, max_count=%d, sum_ratio=%2.3f",
		filters.sort_mode, filters.max_count, filters.sum_ratio);
	filters_str = string(buffer);

	MLOG("%s :: ModelExplainer type %d(%s), original_predictor=%s, gen_type=%s, attr_name=%s, processing={%s}, filters={%s}\n",
		pref.c_str(), processor_type, my_class_name().c_str(), predictor_nm.c_str(),
		GeneratorType_toStr(gen_type).c_str(), global_explain_params.attr_name.c_str(),
		processing_str.c_str(), filters_str.c_str());
}

void LinearExplainer::_init(map<string, string> &mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		//no arguments so far
		MTHROW_AND_ERR("Error in LinearExplainer::init - Unsupported param \"%s\"\n", it->first.c_str());
	}
}

void LinearExplainer::_learn(const MedFeatures &train_mat) {
	avg_bias_score = get_avg_preds(train_mat, original_predictor);
}

void LinearExplainer::explain(const MedFeatures &matrix, vector<map<string, float>> &sample_explain_reasons) const {
	sample_explain_reasons.resize(matrix.data.begin()->second.size());
	string bias_name = "Prior_Score";
	vector<float> preds_orig(matrix.samples.size());
	if (matrix.samples.front().prediction.empty()) {
		MedMat<float> mat_x;
		matrix.get_as_matrix(mat_x);
		if (original_predictor->transpose_for_predict != (mat_x.transposed_flag > 0))
			mat_x.transpose();
		original_predictor->predict(mat_x, preds_orig);
	}
	else {
		for (size_t i = 0; i < preds_orig.size(); ++i)
			preds_orig[i] = matrix.samples[i].prediction[0];
	}

	const vector<vector<int>> *group_inds = &processing.group2Inds;
	const vector<string> *group_names = &processing.groupNames;
	vector<vector<int>> group_inds_loc;
	vector<string> group_names_loc;
	if (processing.group_by_sum) {
		int icol = 0;
		for (auto& rec : matrix.data) {
			group_inds_loc.push_back({ icol++ });
			group_names_loc.push_back(rec.first);
		}
		group_inds = &group_inds_loc;
		group_names = &group_names_loc;
	}

	MedProgress progress("LinearExplainer", (int)matrix.samples.size(), 15);

	vector<float> x(matrix.samples.size() * matrix.data.size());
	for (int i = 0; i < matrix.samples.size(); ++i) {
		int j = 0;
		for (auto it = matrix.data.begin(); it != matrix.data.end(); ++it) {
			x[i* matrix.data.size() + j] = it->second[i];
			++j;
		}
	}


	vector<vector<float>> all_features_coeff(group_names->size());
	//no parallel - will happen in predict
	for (int i = 0; i < group_names->size(); ++i)
	{
		//put zeros in mask i
		vector<float> masked_x = x;
		for (int ind : group_inds->at(i))
			for (size_t j = 0; j < matrix.samples.size(); ++j)
				masked_x[j * matrix.data.size() + ind] = 0;
		vector<float> preds_masked;
		original_predictor->predict(masked_x, preds_masked, (int)matrix.samples.size(), (int)matrix.data.size());

		//commit:
#pragma omp critical
		{
			all_features_coeff[i].resize(matrix.samples.size());
			for (size_t j = 0; j < matrix.samples.size(); ++j)
				all_features_coeff[i][j] = preds_orig[j] - preds_masked[j];
		}

		progress.update();
	}

	//commit to memory:
#pragma omp parallel for
	for (int i = 0; i < sample_explain_reasons.size(); ++i)
	{
		map<string, float> &curr_res = sample_explain_reasons[i];

		float pred_shap = 0;
		for (size_t j = 0; j < group_names->size(); ++j)
			pred_shap += all_features_coeff[j][i];

		for (size_t j = 0; j < group_names->size(); ++j)
			curr_res[group_names->at(j)] = all_features_coeff[j][i];
		//Add prior to score:
		//curr_res[bias_name] = preds_orig[i] - pred_shap; //that will sum to current score
		curr_res[bias_name] = avg_bias_score; //that will sum to current score
	}
}

void KNN_Explainer::_init(map<string, string> &mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		if (it->first == "fraction")
			fraction = med_stof(it->second);
		else if (it->first == "numClusters")
			numClusters = med_stoi(it->second);
		else if (it->first == "chosenThreshold")
			chosenThreshold = med_stof(it->second);
		else if (it->first == "thresholdQ")
			thresholdQ = med_stof(it->second);
		else
			MTHROW_AND_ERR("Error in KNN_Explainer::init - Unsupported param \"%s\"\n", it->first.c_str());
	}
}
void KNN_Explainer::_learn(const MedFeatures &train_mat) {
	if (numClusters == -1)numClusters = (int)train_mat.samples.size();
	if (numClusters > train_mat.samples.size()) {
		MWARN("Warning in KNN_Explainer::Learn - numClusters reduced to size of training \"%d>>%zu\"\n", numClusters, train_mat.samples.size());
		numClusters = (int)train_mat.samples.size();
	}

	MedMat<float> centers(numClusters, (int)train_mat.data.size());

	// get the features and normalize them
	MedFeatures normalizedFeatures = train_mat;
	MedMat<float> normalizedMatrix;
	normalizedFeatures.get_as_matrix(normalizedMatrix);

	normalizedMatrix.normalize();
	normalizedFeatures.set_as_matrix(normalizedMatrix);
	// keep normalization params for future use in apply
	average = normalizedMatrix.avg;
	std = normalizedMatrix.std;
	/* we will test on kmeans later because it is unstable
		//represent features by limitted number of clusters
		vector<int>clusters;
		MedMat<float>dists;
		KMeans(normalizedMatrix, numClusters, centers, clusters, dists);
		vector <float> weights(centers.nrows,0);
		for (int i=0;i<clusters.size();i++){
			weights[clusters[i]]+=1./clusters.size();
		}
		*/

		// random sample of space
	vector <int>krand;
	for (int k = 0; k < normalizedMatrix.nrows; k++)
		krand.push_back(k);
	shuffle(krand.begin(), krand.end(), default_random_engine(5246245));

	for (int i = 0; i < numClusters; i++)
		for (int col = 0; col < normalizedMatrix.ncols; col++)
			centers(i, col) = normalizedMatrix(krand[i], col);
	centers.signals = normalizedMatrix.signals;
	vector<float> weights(centers.nrows, 1);


	//keep the features for the apply phase
	trainingMap.set_as_matrix(centers);
	trainingMap.weights = weights;
	trainingMap.samples.resize(numClusters);
	trainingMap.init_pid_pos_len();

	// compute the thershold according to quantile
	MedFeatures myMat = train_mat;// train_mat is constant
	this->original_predictor->predict(myMat);

	//assign predictions to the sampled  features
	for (int i = 0; i < numClusters; i++)
		trainingMap.samples[i].prediction = vector <float>(1, myMat.samples[krand[i]].prediction[0]);

	// compute the thershold according to quantile
	if (chosenThreshold == MED_MAT_MISSING_VALUE) {
		if (thresholdQ != MED_MAT_MISSING_VALUE) {
			vector <float> predictions = {};
			vector <float> w(train_mat.samples.size(), 1);
			for (int k = 0; k < train_mat.samples.size(); k++)
				predictions.push_back(myMat.samples[k].prediction[0]);

			chosenThreshold = medial::stats::get_quantile(predictions, w, 1 - thresholdQ);
		}
	}
}

void KNN_Explainer::explain(const MedFeatures &matrix, vector<map<string, float>> &sample_explain_reasons) const
{
	MedFeatures explainedFeatures = matrix;
	MedMat<float> explainedMatrix, explainedMatrixCopy;

	vector <string> featureNames;
	trainingMap.get_feature_names(featureNames);

	// check if grouping required if not prepare knn groups from single features
	vector <vector<int>> knnGroups; // will hold the given groups from processing or single features group if not given
	vector<string> knnGroupNames;
	if ((processing.group2Inds.size() > 0) && (processing.group_by_sum == 0)) {
		knnGroups = processing.group2Inds;
		knnGroupNames = processing.groupNames;
	}
	else
		for (int col = 0; col < trainingMap.data.size(); col++) {
			knnGroups.push_back(vector<int>{col});
			knnGroupNames.push_back(featureNames[col]);
		}
	//normalize the explained features
	explainedFeatures.get_as_matrix(explainedMatrix);
	explainedFeatures.get_as_matrix(explainedMatrixCopy);// keep it to handle the missing
	MedMat <float> trainingCentersMatrix;
	trainingMap.get_as_matrix(trainingCentersMatrix);
	sample_explain_reasons = {};

	explainedMatrix.normalize(average, std, 1);
	for (int row = 0; row < explainedMatrix.nrows; row++)
		for (int col = 0; col < explainedMatrix.ncols; col++)
			if (explainedMatrixCopy.get(row, col) == MED_MAT_MISSING_VALUE)
				explainedMatrix.set(row, col) = MED_MAT_MISSING_VALUE;
	//for each sample compute the explanation
	vector <float> thisRow;
	for (int row = 0; row < explainedMatrix.nrows; row++) {
		explainedMatrix.get_row(row, thisRow);
		sample_explain_reasons.push_back({});
		computeExplanation(thisRow, sample_explain_reasons[row], knnGroups, knnGroupNames);
	}
}
void KNN_Explainer::computeExplanation(vector<float> thisRow, map<string, float> &sample_explain_reasons, vector <vector<int>> knnGroups, vector<string> knnGroupNames) const
// do the calculation for a single sample after normalization
{

	MedMat<float> centers; //matrix taken from features and holds the centers of clusters
	trainingMap.get_as_matrix(centers);
	MedMat<float> pDistance(centers.nrows, centers.ncols);//initialized to 0
	MedMat<float> gDistance(centers.nrows, (int)knnGroups.size());
	vector<float>totalDistance(centers.nrows, 0);
#define SQR(x)  ((x)*(x))
	for (int row = 0; row < centers.nrows; row++) {
		for (int col = 0; col < centers.ncols; col++)
			if (thisRow[col] != MED_MAT_MISSING_VALUE) {
				pDistance(row, col) = SQR(centers.get(row, col) - thisRow[col]);
				totalDistance[row] += pDistance(row, col);
			}
		for (int group = 0; group < knnGroupNames.size(); group++) {
			gDistance(row, group) = totalDistance[row];
			for (auto inGroup : knnGroups[group])
				if (thisRow[inGroup] != MED_MAT_MISSING_VALUE)
					gDistance(row, group) -= pDistance(row, inGroup);

		}
	}
	vector<float> thresholds(centers.nrows, 0);
	vector <float> colVector;
	float totalThreshold = medial::stats::get_quantile(totalDistance, trainingMap.weights, fraction);
	for (int group = 0; group < knnGroupNames.size(); group++) {
		gDistance.get_col(group, colVector);
		thresholds[group] = medial::stats::get_quantile(colVector, trainingMap.weights, fraction);
	}
	double sumWeights = 0;
	double pCol;
	double pTotal = 0;

	for (int row = 0; row < pDistance.nrows; row++)
		if (totalDistance[row] < totalThreshold) {
			float thisPred = trainingMap.samples[row].prediction[0];
			sumWeights += trainingMap.weights[row];
			if (chosenThreshold != MED_MAT_MISSING_VALUE)thisPred = thisPred > chosenThreshold;// threshold the predictions if needed
			pTotal += trainingMap.weights[row] * thisPred;
			//cout <<row<<" "<< trainingMap.samples[row].prediction[0] << "\n";
		}
	pTotal /= sumWeights;


	for (int group = 0; group < knnGroupNames.size(); group++) {
		pCol = 0;
		sumWeights = 0;
		for (int row = 0; row < gDistance.nrows; row++)
			if (gDistance.get(row, group) < thresholds[group]) {
				float thisPred = trainingMap.samples[row].prediction[0];
				if (chosenThreshold != MED_MAT_MISSING_VALUE)thisPred = thisPred > chosenThreshold;// threshold the predictions if needed
				pCol += trainingMap.weights[row] * thisPred;
				sumWeights += trainingMap.weights[row];
			}
		pCol /= sumWeights;
		sample_explain_reasons.insert(pair<string, float>(knnGroupNames[group], float(log((pTotal + 1e-10) / (pCol + 1e-10) / (1 - pTotal + 1e-10)*(1 - pCol + 1e-10)))));


	}




}

//########################Iterative##################################
void IterativeSetExplainer::_init(map<string, string> &mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		if (it->first == "gen_type")
			gen_type = GeneratorType_fromStr(it->second);
		else if (it->first == "generator_args")
			generator_args = it->second;
		else if (it->first == "missing_value")
			missing_value = med_stof(it->second);
		else if (it->first == "n_masks")
			n_masks = med_stoi(it->second);
		else if (it->first == "sampling_args")
			sampling_args = it->second;
		else if (it->first == "use_random_sampling")
			use_random_sampling = med_stoi(it->second) > 0;
		else if (it->first == "sort_params_a")
			sort_params_a = med_stof(it->second);
		else if (it->first == "sort_params_b")
			sort_params_b = med_stof(it->second);
		else if (it->first == "sort_params_k1")
			sort_params_k1 = med_stof(it->second);
		else if (it->first == "sort_params_k2")
			sort_params_k2 = med_stof(it->second);
		else if (it->first == "max_set_size")
			max_set_size = med_stoi(it->second);
		else
			MTHROW_AND_ERR("Error in IterativeSetExplainer::init - Unsupported param \"%s\"\n", it->first.c_str());
	}
	init_sampler(); //from args
}

void IterativeSetExplainer::init_sampler(bool with_sampler) {
	switch (gen_type)
	{
	case GIBBS:
		if (with_sampler) {
			_gibbs.init_from_string(generator_args);
			_sampler = unique_ptr<SamplesGenerator<float>>(new GibbsSamplesGenerator<float>(_gibbs, true));
		}
		_gibbs_sample_params.init_from_string(sampling_args);
		sampler_sampling_args = &_gibbs_sample_params;
		break;
	case GAN:
		if (with_sampler) {
			_sampler = unique_ptr<SamplesGenerator<float>>(new MaskedGAN<float>);
			static_cast<MaskedGAN<float> *>(_sampler.get())->read_from_text_file(generator_args);
			static_cast<MaskedGAN<float> *>(_sampler.get())->mg_params.init_from_string(sampling_args);
		}
		break;
	case MISSING:
		if (with_sampler)
			_sampler = unique_ptr<SamplesGenerator<float>>(new MissingsSamplesGenerator<float>(missing_value));
		break;
	case RANDOM_DIST:
		if (with_sampler)
			_sampler = unique_ptr<SamplesGenerator<float>>(new RandomSamplesGenerator<float>(0, 5));
		sampler_sampling_args = &n_masks;
		break;
	default:
		MTHROW_AND_ERR("Error in ShapleyExplainer::init_sampler() - Unsupported Type %d\n", gen_type);
	}
}

void IterativeSetExplainer::_learn(const MedFeatures &train_mat) {
	_sampler->learn(train_mat.data);
	avg_bias_score = get_avg_preds(train_mat, original_predictor);
}

void IterativeSetExplainer::explain(const MedFeatures &matrix, vector<map<string, float>> &sample_explain_reasons) const {
	sample_explain_reasons.resize(matrix.samples.size());
	string bias_name = "Prior_Score";
	vector<float> preds_orig(matrix.samples.size());
	if (matrix.samples.front().prediction.empty()) {
		MedMat<float> mat_x;
		matrix.get_as_matrix(mat_x);
		if (original_predictor->transpose_for_predict != (mat_x.transposed_flag > 0))
			mat_x.transpose();
		original_predictor->predict(mat_x, preds_orig);
	}
	else {
		for (size_t i = 0; i < preds_orig.size(); ++i)
			preds_orig[i] = matrix.samples[i].prediction[0];
	}

	const vector<vector<int>> *group_inds = &processing.group2Inds;
	const vector<string> *group_names = &processing.groupNames;
	vector<vector<int>> group_inds_loc;
	vector<string> group_names_loc;
	if (processing.group_by_sum) {
		int icol = 0;
		for (auto& rec : matrix.data) {
			group_inds_loc.push_back({ icol++ });
			group_names_loc.push_back(rec.first);
		}
		group_inds = &group_inds_loc;
		group_names = &group_names_loc;
	}

	int MAX_Threads = omp_get_max_threads();
	//copy sample for each thread:
	random_device rd;
	vector<mt19937> gen_thread(MAX_Threads);
	vector<MedPredictor *> predictor_cp(MAX_Threads);
	for (size_t i = 0; i < gen_thread.size(); ++i)
		gen_thread[i] = mt19937(globalRNG::rand());
	_sampler->prepare(sampler_sampling_args);
	size_t sz_pred = original_predictor->get_size();
	unsigned char *blob_pred = new unsigned char[sz_pred];
	original_predictor->serialize(blob_pred);
	for (size_t i = 0; i < predictor_cp.size(); ++i) {
		predictor_cp[i] = (MedPredictor *)medial::models::copyInfraModel(original_predictor, false);
		predictor_cp[i]->deserialize(blob_pred);
	}
	delete[]blob_pred;

	MedProgress progress("IterativeSetExplainer", (int)matrix.samples.size(), 15, 1);
#pragma omp parallel for if (matrix.samples.size() >= 2)
	for (int i = 0; i < matrix.samples.size(); ++i)
	{
		int n_th = omp_get_thread_num();
		vector<float> features_coeff, score_history;
		float pred_shap = 0;
		float use_bias = avg_bias_score;
		medial::shapley::explain_minimal_set(matrix, (int)i, n_masks, predictor_cp[n_th], missing_value,
			*group_inds, *_sampler, gen_thread[n_th], sampler_sampling_args, features_coeff, score_history, max_set_size,
			use_bias, sort_params_a, sort_params_b, sort_params_k1, sort_params_k2
			, global_logger.levels[LOCAL_SECTION] < LOCAL_LEVEL &&
			(!(matrix.samples.size() >= 2) || omp_get_thread_num() == 1));

		//reverse order in features_coeff:
		int ind_score_hist = 0;
		vector<pair<int, float>> tp(features_coeff.size());
		for (int j = 0; j < tp.size(); ++j)
		{
			tp[j].first = j;
			tp[j].second = features_coeff[j];
		}
		sort(tp.begin(), tp.end(), [](const pair<int, float>&a, const pair<int, float> &b) {
			return abs(a.second) < abs(b.second); }); //0 are ignored
		for (size_t j = 0; j < tp.size(); ++j)
		{
			if (tp[j].second == 0)
				continue;
			bool positive_contrib = tp[j].second > 0;
			features_coeff[tp[j].first] = float((int)features_coeff.size() + 1 - abs(tp[j].second));
			double diff = abs(score_history[ind_score_hist] - (ind_score_hist > 0 ? score_history[ind_score_hist - 1] : use_bias));
			if (diff > 1)
				diff = 0.99999;
			features_coeff[tp[j].first] += diff;
			if (!positive_contrib)
				features_coeff[tp[j].first] = -features_coeff[tp[j].first];
			++ind_score_hist;
		}

		for (size_t j = 0; j < features_coeff.size(); ++j)
			pred_shap += features_coeff[j];

#pragma omp critical 
		{
			map<string, float> &curr_res = sample_explain_reasons[i];
			for (size_t j = 0; j < group_names->size(); ++j)
				curr_res[group_names->at(j)] = features_coeff[j];
			//Add prior to score:
			//curr_res[bias_name] = preds_orig[i] - pred_shap; //that will sum to current score
			curr_res[bias_name] = avg_bias_score;
		}

		progress.update();
	}
	for (size_t i = 0; i < predictor_cp.size(); ++i)
		delete predictor_cp[i];
}

void IterativeSetExplainer::post_deserialization() {
	init_sampler(false);
}

void IterativeSetExplainer::load_GIBBS(MedPredictor *original_pred, const GibbsSampler<float> &gibbs, const GibbsSamplingParams &sampling_args) {
	this->original_predictor = original_pred;
	_gibbs = gibbs;
	_gibbs_sample_params = sampling_args;

	sampler_sampling_args = &_gibbs_sample_params;
	_sampler = unique_ptr<SamplesGenerator<float>>(new GibbsSamplesGenerator<float>(_gibbs, true));

	gen_type = GeneratorType::GIBBS;
}

void IterativeSetExplainer::load_GAN(MedPredictor *original_pred, const string &gan_path) {
	this->original_predictor = original_pred;
	_sampler = unique_ptr<SamplesGenerator<float>>(new MaskedGAN<float>);
	static_cast<MaskedGAN<float> *>(_sampler.get())->read_from_text_file(gan_path);
	static_cast<MaskedGAN<float> *>(_sampler.get())->mg_params.init_from_string(sampling_args);

	gen_type = GeneratorType::GAN;
}

void IterativeSetExplainer::load_MISSING(MedPredictor *original_pred) {
	this->original_predictor = original_pred;
	_sampler = unique_ptr<SamplesGenerator<float>>(new MissingsSamplesGenerator<float>(missing_value));
	gen_type = GeneratorType::MISSING;
}

void IterativeSetExplainer::load_sampler(MedPredictor *original_pred, unique_ptr<SamplesGenerator<float>> &&generator) {
	this->original_predictor = original_pred;
	_sampler = std::move(generator);
}

void IterativeSetExplainer::dprint(const string &pref) const {
	string predictor_nm = "";
	if (original_predictor != NULL)
		predictor_nm = original_predictor->my_class_name();
	string filters_str = "", processing_str = "";
	char buffer[5000];
	snprintf(buffer, sizeof(buffer), "group_by_sum=%d, learn_cov_matrix=%d, normalize_vals=%d, zero_missing=%d, grouping=%s",
		int(processing.group_by_sum), int(processing.learn_cov_matrix), processing.normalize_vals
		, processing.zero_missing, processing.grouping.c_str());
	processing_str = string(buffer);
	snprintf(buffer, sizeof(buffer), "sort_mode=%d, max_count=%d, sum_ratio=%2.3f",
		filters.sort_mode, filters.max_count, filters.sum_ratio);
	filters_str = string(buffer);

	MLOG("%s :: ModelExplainer type %d(%s), original_predictor=%s, gen_type=%s, attr_name=%s, processing={%s}, filters={%s}\n",
		pref.c_str(), processor_type, my_class_name().c_str(), predictor_nm.c_str(),
		GeneratorType_toStr(gen_type).c_str(), global_explain_params.attr_name.c_str(),
		processing_str.c_str(), filters_str.c_str());
}