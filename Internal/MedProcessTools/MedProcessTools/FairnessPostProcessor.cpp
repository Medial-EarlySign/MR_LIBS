#include "FairnessPostProcessor.h"
#include <MedStat/MedStat/MedBootstrap.h>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <cmath>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MED_MODEL
#define LOCAL_LEVEL LOG_DEF_LEVEL

unordered_map<string, int> map_target_type = {
	{"SENS", Fairness_Target_Type::SENS  },
	{ "SPEC", Fairness_Target_Type::SPEC }
};

void Cutoff_Constraint::set_type(const string &t) {
	string type_str = boost::to_lower_copy(t);
	if (type_str == "score")
		type = Cutoff_Type::Score;
	else if (type_str == "pr")
		type = Cutoff_Type::PR;
	else if (type_str == "sens")
		type = Cutoff_Type::Sens;
	else
		MTHROW_AND_ERR("Error Cutoff_Constraint::set_type - unknown type %s\n", t.c_str());
}

void FairnessPostProcessor::parse_constrains(const string &s) {
	constraints.clear();
	vector<string> tokens;
	boost::split(tokens, s, boost::is_any_of(","));

	for (const string &token : tokens)
	{
		vector<string> parts;
		boost::split(parts, token, boost::is_any_of(":"));
		if (parts.size() != 2)
			MTHROW_AND_ERR("Error FairnessPostProcessor::parse_constrains - bad tokens format. Expecting \":\".\nGot %s\n",
				token.c_str());
		string type_str = parts[0];
		float val_ = med_stof(parts[1]);
		Cutoff_Constraint cc;
		cc.set_type(type_str);
		cc.value = val_;
		constraints.push_back(std::move(cc));
	}
}

int FairnessPostProcessor::init(map<string, string> &mapper) {
	for (auto &it : mapper)
	{
		if (it.first == "feature_name")
			feature_name = it.second;
		else if (it.first == "model_json")
			model_json = it.second;
		else if (it.first == "reference_group_val")
			reference_group_val = med_stof(it.second);
		else if (it.first == "resulotion")
			resulotion = med_stof(it.second);
		else if (it.first == "fairness_target_type") {
			string s = boost::to_upper_copy(it.second);
			if (map_target_type.find(s) == map_target_type.end())
				MTHROW_AND_ERR("Error: unable to find type %s\n", it.second.c_str());
			fairness_target_type = Fairness_Target_Type(map_target_type[s]);
		}
		else if (it.first == "constraints") {
			parse_constrains(it.second);
		}
		else if (it.first == "allow_distance_score")
			allow_distance_score = stod(it.second);
		else if (it.first == "allow_distance_target")
			allow_distance_target = stod(it.second);
		else if (it.first == "allow_distance_cutoff_constraint")
			allow_distance_cutoff_constraint = stod(it.second);
		else if (it.first == "score_bin_count")
			score_bin_count = med_stoi(it.second);
		else if (it.first == "score_resulotion")
			score_resulotion = med_stof(it.second);
		else if (it.first == "pp_type") {} //ignore
		else
			MTHROW_AND_ERR("Error - FairnessPostProcessor::init - unknown param %s\n", it.first.c_str());
	}

	if (model_json.empty())
		MTHROW_AND_ERR("Error - FairnessPostProcessor::init - Must provide model_json\n");
	if (feature_name.empty())
		MTHROW_AND_ERR("Error - FairnessPostProcessor::init - Must provide feature_name\n");
	if (constraints.empty())
		MTHROW_AND_ERR("Error - FairnessPostProcessor::init - Must provide constraints\n");
	if (reference_group_val == MED_MAT_MISSING_VALUE)
		MTHROW_AND_ERR("Error - FairnessPostProcessor::init - Must provide reference_group_val\n");

	return 0;
}

void FairnessPostProcessor::get_input_fields(vector<Effected_Field> &fields) const {
	fields.push_back(Effected_Field(Field_Type::PREDICTION, "0"));
}
void FairnessPostProcessor::get_output_fields(vector<Effected_Field> &fields) const {
	fields.push_back(Effected_Field(Field_Type::PREDICTION, "0"));
}

float fetch_score_from_string(const string &best_val) {
	vector<string> tokens;
	boost::split(tokens, best_val, boost::is_any_of("@"));
	string score_part = tokens[1];
	tokens.clear();
	boost::split(tokens, score_part, boost::is_any_of("_"));
	float res = med_stof(tokens[1]);
	return res;
}
//fetch score
float fetch_bt_num(const map<string, float> &bt,
	const string &search_prefix, float search_num, float max_diff = 0.5) {
	//SHOULD be search_prefix@SCORE_value -> find closet and fetch value
	float min_th = -1;
	string best_val = "";
	for (const auto &it : bt)
	{
		string measure = it.first;
		if (!boost::starts_with(measure, search_prefix))
			continue;
		if (!boost::ends_with(measure, "_Mean"))
			continue;
		//correct measure class:
		float diff = abs(it.second - search_num);
		if (min_th < 0 || diff < min_th) {
			min_th = diff;
			best_val = it.first;
		}
	}
	if (best_val.empty())
		MTHROW_AND_ERR("Error - not found %s\n", search_prefix.c_str());
	if (min_th <0 || min_th > max_diff)
		MTHROW_AND_ERR("Error - not found %s\nfound best in diff %f",
			search_prefix.c_str(), min_th);

	//now parse best_val
	vector<string> tokens;
	boost::split(tokens, best_val, boost::is_any_of("@"));
	string score_part = tokens[1];
	tokens.clear();
	boost::split(tokens, score_part, boost::is_any_of("_"));
	float res = med_stof(tokens[1]);
	return res;
}

void fetch_bt_num_by_score(const map<string, float> &bt,
	const string &search_prefix, double resulotion, bool sorted_inc,
	vector<float> &scores, vector<float> &targets, vector<float> &pr,
	vector<float> &sens) {
	scores.clear();
	targets.clear();
	pr.clear();
	sens.clear();
	//SHOULD be search_prefix@SCORE_value -> find closet and fetch value
	float tar_next = -1;
	string best_val = "";
	for (const auto &it : bt)
	{
		string measure = it.first;
		if (!boost::starts_with(measure, search_prefix))
			continue;
		if (!boost::ends_with(measure, "_Mean"))
			continue;
		//correct measure class - sorted from low to high by score.
		float target = it.second;
		//check if closet to this or (this + res):
		float target_round = int(target / resulotion)*resulotion;
		float diff = abs(target - target_round);
		if (targets.empty() || targets.back() != target_round) {
			//moved to new target bin:
			float diff_next = abs(target_round - tar_next);
			if (tar_next < 0 || diff < diff_next)  //this is new first and best then so far
				best_val = it.first;
			scores.push_back(fetch_score_from_string(best_val));
			targets.push_back(target_round);

			vector<string> tokens;
			boost::split(tokens, it.first, boost::is_any_of("@"));
			string measure_pr = "PR@" + tokens[1];
			if (bt.find(measure_pr) == bt.end())
				pr.push_back(MED_MAT_MISSING_VALUE);
			else
				pr.push_back(bt.at(measure_pr));
			measure_pr = "SENS@" + tokens[1];
			if (bt.find(measure_pr) == bt.end())
				sens.push_back(MED_MAT_MISSING_VALUE);
			else
				sens.push_back(bt.at(measure_pr));
		}
		//update diff_next and best_val for next:
		best_val = it.first;
		tar_next = target;

	}
}

int get_closet_idx(const vector<float> &search_vec, float search, float max_diff = 1) {
	float min_diff = -1;
	int idx = -1;
	for (int i = 0; i < search_vec.size(); ++i)
	{
		float diff = abs(search_vec[i] - search);
		if (min_diff < 0 || diff < min_diff) {
			min_diff = diff;
			idx = i;
		}
	}

	if (idx < 0)
		MTHROW_AND_ERR("Error Can't find value - empty\n");
	if (min_diff > max_diff)
		idx = -1;
	//MTHROW_AND_ERR("Error value is too far - %f\n", min_diff);

	return idx;
}

float get_closet(const vector<float> &search_vec, const vector<float> &val_vec,
	float search, float max_diff = 1) {
	int idx = get_closet_idx(search_vec, search, max_diff);
	if (idx < 0)
		return MED_MAT_MISSING_VALUE;
	return val_vec[idx];
}

void FairnessPostProcessor::Learn(const MedFeatures &matrix) {
	group_feature_gen_model.init_from_json_file(model_json);

	MedSamples samples;
	samples.import_from_sample_vec(matrix.samples);
	int tot_sample_size = (int)matrix.samples.size();

	MedPidRepository rep_reader;
	unordered_set<string> req_sigs;
	vector<string> rsigs;
	vector<int> pids;
	samples.get_ids(pids);
	if (rep_reader.init(p_rep->config_fname) < 0)
		MTHROW_AND_ERR("ERROR could not read repository %s\n", p_rep->config_fname.c_str());
	group_feature_gen_model.fit_for_repository(rep_reader);
	group_feature_gen_model.get_required_signal_names(req_sigs);
	for (auto &s : req_sigs) rsigs.push_back(s);
	if (rep_reader.read_all(p_rep->config_fname, pids, rsigs) < 0)
		MTHROW_AND_ERR("ERROR could not read repository %s\n", p_rep->config_fname.c_str());
	group_feature_gen_model.learn(rep_reader, samples, MED_MDL_LEARN_REP_PROCESSORS, MED_MDL_APPLY_FTR_PROCESSORS);

	//Now can generate group and eval resolved_name:
	vector<string> all_n;
	group_feature_gen_model.features.get_feature_names(all_n);
	resolved_name = all_n[find_in_feature_names(all_n, feature_name)];
	vector<float> &grp_data_vec = group_feature_gen_model.features.data.at(resolved_name);

	//Find cutoffs using ref group and aggregate into other groups to compare fairness:
	unordered_map<float, vector<MedSample>> group_to_res;
	for (size_t i = 0; i < group_feature_gen_model.features.samples.size(); ++i)
	{
		const MedSample &smp = group_feature_gen_model.features.samples[i];
		group_to_res[grp_data_vec[i]].push_back(smp);
	}
	//sort from small to high by score
	for (auto &it : group_to_res)
		sort(it.second.begin(), it.second.end(), [](const MedSample &a, const MedSample &b) {
		return a.prediction[0] < b.prediction[0]; });
	if (group_to_res.find(reference_group_val) == group_to_res.end())
		MTHROW_AND_ERR("Error - FairnessPostProcessor::Learn - unable to find ref value %f in groups\n", reference_group_val);

	//calc target for each group and score by Bootstrap
	MedBootstrap bt;
	bt.roc_Params.use_score_working_points = true;
	bt.roc_Params.score_bins = score_bin_count;
	bt.roc_Params.score_resolution = score_resulotion;
	map<string, vector<float>> empty_info;
	unordered_map<float, map<string, float>> group_to_bootstrap;
	//map<string, float> global_bt;
	for (auto &it : group_to_res)
	{
		MedSamples grp_samples;
		grp_samples.import_from_sample_vec(it.second);
		group_to_bootstrap[it.first] = std::move(bt.bootstrap(grp_samples, empty_info).at("All"));
	}

	const map<string, float> &ref_bt = group_to_bootstrap.at(reference_group_val);
	string search_pre; bool is_sorted_asc;
	switch (fairness_target_type)
	{
	case Fairness_Target_Type::SENS:
		search_pre = "SENS";
		is_sorted_asc = false;
		break;
	case Fairness_Target_Type::SPEC:
		search_pre = "SPEC";
		is_sorted_asc = true;
		break;
	default:
		MTHROW_AND_ERR("Error - unsupported\n");
	}
	//calc for all:
	unordered_map<float, vector<float>> group_to_scores;
	unordered_map<float, vector<float>> group_to_targets;
	unordered_map<float, vector<float>> group_to_PR;
	unordered_map<float, vector<float>> group_to_SENS; //might be duplicate of group_to_targets when target is SENS and not SPEC
	for (const auto &it : group_to_bootstrap)
	{
		vector<float> &ref_cutoffs = group_to_scores[it.first];
		vector<float> &ref_targets = group_to_targets[it.first];
		vector<float> &ref_PR = group_to_PR[it.first];
		vector<float> &ref_SENS = group_to_SENS[it.first];
		fetch_bt_num_by_score(it.second, search_pre, resulotion,
			is_sorted_asc, ref_cutoffs, ref_targets, ref_PR, ref_SENS);
	}

	//prints fairness in constraints checks:
	const vector<float> &ref_scores = group_to_scores.at(reference_group_val);
	const vector<float> &ref_targets = group_to_targets.at(reference_group_val);
	const vector<float> &ref_sens = group_to_SENS.at(reference_group_val);
	const vector<float> &ref_pr = group_to_PR.at(reference_group_val);
	int ref_size = (int)group_to_res.at(reference_group_val).size();

	unordered_map<float, vector<float>> group_to_score_vals;
	for (const auto &c : constraints) {
		float cutoff_search = c.value;
		switch (c.type)
		{
		case Cutoff_Type::PR:
			cutoff_search = fetch_bt_num(ref_bt, "PR", c.value, allow_distance_cutoff_constraint);
			break;
		case Cutoff_Type::Sens:
			cutoff_search = fetch_bt_num(ref_bt, "SENS", c.value, allow_distance_cutoff_constraint);
			break;
		case Cutoff_Type::Score:
			cutoff_search = c.value;
			break;
		default:
			MTHROW_AND_ERR("Error - unsupported\n");
		}

		//print fairness for all groups vs reference:
		float reference_target = get_closet(ref_scores, ref_targets, cutoff_search, allow_distance_score);
		if (reference_target == MED_MAT_MISSING_VALUE)
			MTHROW_AND_ERR("Error - can't find cutoff. score resulotion is too low for ref?\n");
		for (const auto &it : group_to_scores)
		{
			if (it.first == reference_group_val)
				continue;
			int closet_idx = get_closet_idx(it.second, cutoff_search, allow_distance_score);
			if (closet_idx < 0)
				MTHROW_AND_ERR("Error - can't find cutoff. score resulotion is too low for other group?\n");
			float other_target = group_to_targets.at(it.first)[closet_idx];
			float other_score_cutoff = it.second[closet_idx];
			MLOG("Compare %s in group_value=%f to reference val=%f :: [%f <=> %f]. cutoffs [%f <=> %f]\n",
				search_pre.c_str(), it.first, reference_group_val,
				other_target, reference_target, other_score_cutoff, cutoff_search);
		}

		bool first_time = true;
		for (const auto &it : group_to_targets) {
			if (it.first == reference_group_val)
				continue;
			const vector<float> &other_group_targets = it.second;
			float min_diff = -1; int idx = -1;
			for (int i = 0; i < ref_targets.size(); ++i)
			{
				//TODO: iterate together - same order and faster
				int other_idx = get_closet_idx(other_group_targets, ref_targets[i], allow_distance_target);// other_group_targets[other_idx] ~= ref_targets[i]
				if (other_idx < 0)
					continue; //can't reach this target
				//calculate all options for constraint: SENS, PR, score (and also target):
				float sens_ref_i = ref_sens[i];
				float score_ref_i = ref_scores[i];

				float pr_other_i = group_to_PR.at(it.first)[other_idx];
				float pr_ref_i = ref_pr[i];
				//Calc new PR for "fixed" scores:
				double new_pr_size = static_cast<double>(pr_other_i) * static_cast<double>(group_to_res.at(it.first).size());
				new_pr_size += static_cast<double>(pr_ref_i) * static_cast<double>(ref_size);
				new_pr_size /= tot_sample_size;

				float val_to_cmp = new_pr_size;
				switch (c.type)
				{
				case Cutoff_Type::PR:
					val_to_cmp = new_pr_size;
					break;
				case Cutoff_Type::Sens:
					val_to_cmp = sens_ref_i;
					break;
				case Cutoff_Type::Score:
					val_to_cmp = score_ref_i;
					break;
				default:
					MTHROW_AND_ERR("Error - unsupported\n");
				}
				//find closet to c.value by: (PR: new_pr_size, SENS: sens_ref_i, Score: score_ref_i) => val_to_cmp
				float diff = abs(c.value - val_to_cmp);
				if (min_diff < 0 || diff < min_diff) {
					min_diff = diff;
					idx = i;
				}
			}

			//idx is the "best" match for constraint:
			int other_idx = get_closet_idx(other_group_targets, ref_targets[idx], allow_distance_target);// other_group_targets[other_idx] ~= ref_targets[idx]
			if (other_idx < 0)
				MTHROW_AND_ERR("Error - can't find optimal target.\n");
			float score_other_i = group_to_scores.at(it.first)[other_idx];
			float score_ref_i = ref_scores[idx];
			//need to transform score_other_i into score_ref_i for this constraint
			if (first_time) {
				group_to_score_vals[reference_group_val].push_back(score_ref_i);
				first_time = false;
			}
			group_to_score_vals[it.first].push_back(score_other_i);
		}
	}

	//Now fix biases - find b0 and factor to each constraint by using group_to_score_vals - reorder for ranges:
	vector<float> &ref_score_vals = group_to_score_vals.at(reference_group_val);
	//sort to get indexes:
	vector<pair<int, float>> pair_order(ref_score_vals.size());
	for (int i = 0; i < ref_score_vals.size(); ++i)
	{
		pair_order[i].first = i;
		pair_order[i].second = ref_score_vals[i];
	}
	sort(pair_order.begin(), pair_order.end(), [](const pair<int, float> &a, const pair<int, float> &b) {
		return a.second < b.second; });
	//use order of pair_order - first value:

	for (const auto &it : group_to_score_vals)
	{
		vector<float> &grp_score_cutoffs = group_to_score_cutoffs_ranges[it.first];
		grp_score_cutoffs.resize(pair_order.size());
		for (size_t i = 0; i < pair_order.size(); ++i)
		{
			int ii = pair_order[i].first;
			grp_score_cutoffs[i] = it.second[ii];
		}
	}
	//TODO: improve and select better trasformation to optimize something 
	// in the middle and not just fix bias by "factor" or addition

	//manipulate scores for each group by reference. use group_to_score_cutoffs_ranges and reference_group_val:
	const vector<float> &reference_cutoffs = group_to_score_cutoffs_ranges.at(reference_group_val);

	//calc factors:
	for (auto &it : group_to_score_cutoffs_ranges)
	{
		vector<float> &factor = group_to_factors[it.first];
		vector<float> &bias = group_to_bias[it.first];
		const vector<float> &grp_cutoffs = it.second;
		factor.resize(grp_cutoffs.size(), 1);
		bias.resize(grp_cutoffs.size());
		for (size_t i = 0; i < factor.size(); ++i)
		{
			if (grp_cutoffs[i] > 0) {
				factor[i] = reference_cutoffs[i] / grp_cutoffs[i];
				bias[i] = 0;
				if (i > 0) {
					double diff_y = reference_cutoffs[i] - reference_cutoffs[i - 1];
					double diff_x = grp_cutoffs[i] - grp_cutoffs[i - 1];
					if (diff_x > 0)
						factor[i] = diff_y / diff_x;
					else
						MTHROW_AND_ERR("Error can't adjust for bias\n");
					bias[i] = reference_cutoffs[i] - factor[i] * grp_cutoffs[i];
				}
			}
			else
				MWARN("WARN: can't fix with factors\n");
		}
	}

}

void FairnessPostProcessor::Apply(MedFeatures &matrix) {
	//Calc groups for matrix:
	MedSamples samples;
	samples.import_from_sample_vec(matrix.samples);
	//TODO: in order for it to run smoothly in AlgoMarker - the need signals needs to be updated as required for model to run
	//medial::medmodel::apply(group_feature_gen_model, samples, p_rep->config_fname,
	//	MedModelStage::MED_MDL_APPLY_FTR_PROCESSORS);
	medial::print::medmodel_logging(false);
	if (!feature_gen_init) {
		group_feature_gen_model.init_model_for_apply(*p_rep,
			MedModelStage::MED_MDL_APPLY_FTR_GENERATORS, MED_MDL_APPLY_FTR_PROCESSORS);
		feature_gen_init = true;

	}
	group_feature_gen_model.no_init_apply(*p_rep, samples, MedModelStage::MED_MDL_APPLY_FTR_GENERATORS, MedModelStage::MED_MDL_APPLY_FTR_PROCESSORS);
	medial::print::medmodel_logging(true);
	vector<float> &group_vec = group_feature_gen_model.features.data.at(resolved_name);

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < matrix.samples.size(); ++i)
	{
		MedSample &s = matrix.samples[i];
		if (group_vec[i] == reference_group_val)
			continue;
		if (group_to_score_cutoffs_ranges.find(group_vec[i]) == group_to_score_cutoffs_ranges.end())
			MTHROW_AND_ERR("Error FairnessPostProcessor::Apply - can't find group %f\n", group_vec[i]);

		const vector<float> &grp_cutoffs = group_to_score_cutoffs_ranges.at(group_vec[i]);
		float &score = s.prediction[0];
		//manipulate group group_vec[i] grp_cutoffs to reference_cutoffs
		//find index in grp_cutoffs:
		int pos = medial::process::binary_search_position(grp_cutoffs, score);
		if (pos >= grp_cutoffs.size())
			pos = (int)grp_cutoffs.size() - 1;
		float factor = group_to_factors.at(group_vec[i])[pos];
		float bias = group_to_bias.at(group_vec[i])[pos];
		score = score * factor + bias;
	}
}

void FairnessPostProcessor::dprint(const string &pref) const {
	MLOG("%s :: %s :: on group %s, reference group val=%f\n",
		pref.c_str(), my_class_name().c_str(), feature_name.c_str(), reference_group_val);
	for (const auto &it : group_to_score_cutoffs_ranges)
	{
		const vector<float> &cutoff = it.second;
		const vector<float> &factors = group_to_factors.at(it.first);
		const vector<float> &bias = group_to_bias.at(it.first);
		float group_val = it.first;
		string reference = "";
		if (group_val == reference_group_val)
			reference = "[Reference group]";
		for (size_t i = 0; i < cutoff.size(); ++i)
			MLOG("Group_val:=%f%s :: %zu :: cutoff %f :: factor %f :: bias %f\n",
				group_val, reference.c_str(), i, cutoff[i], factors[i], bias[i]);


	}

}