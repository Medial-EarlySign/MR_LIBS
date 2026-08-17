#include "TQRF.h"
#include <MedStat/MedStat/MedPerformance.h>
#include <MedUtils/MedUtils/MedGlobalRNG.h>
#include <MedStat/MedStat/MedStat.h>
#include <math.h>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL

#include <Eigen/Core>
#include <Eigen/Dense>
using namespace Eigen;


//================================================================================================
// TQRF_Params
//================================================================================================
//------------------------------------------------------------------------------------------------
int TQRF_Params::init(map<string, string>& map)
{
	for (auto &f : map) {

		//MLOG("%s => %s %d\n", f.first.c_str(), f.second.c_str(), (f.first == "time_slice_size"));

		// general
		if (f.first == "samples_time_unit") samples_time_unit = f.second;
		if (f.first == "ncateg") ncateg = stoi(f.second);

		// time slices
		else if (f.first == "time_slice_unit") time_slice_unit = f.second;
		else if (f.first == "time_slice_size") time_slice_size = stoi(f.second);
		else if (f.first == "n_time_slices") n_time_slices = stoi(f.second);
		else if (f.first == "time_slices") {
			vector<string> fields;
			boost::split(fields, f.second, boost::is_any_of(","));
			for (auto &ts : fields)
				time_slices.push_back(stoi(ts));
			n_time_slices = (int)time_slices.size();
		}
		else if (f.first == "time_slices_wgts") {
			vector<string> fields;
			boost::split(fields, f.second, boost::is_any_of(","));
			for (auto &ts : fields)
				time_slices_wgts.push_back(stof(ts));
		}
		else if (f.first == "censor_cases") censor_cases = stoi(f.second);

		// quantization
		else if (f.first == "max_q") max_q = stoi(f.second);
		//else if (f.first == "max_q_sample") max_q_sample = stoi(f.second);
		else if (f.first == "qpoints_per_split") qpoints_per_split = stoi(f.second);

		// trees and stopping criteria
		else if (f.first == "tree_type") tree_type = f.second;
		else if (f.first == "ntrees") ntrees = stoi(f.second);
		else if (f.first == "max_depth") max_depth = stoi(f.second);
		else if (f.first == "min_node_last_slice") min_node_last_slice = stoi(f.second);
		else if (f.first == "min_node") min_node = stoi(f.second);
		else if (f.first == "random_split_prob") random_split_prob = stof(f.second);

		// speedup by subsample control
		else if (f.first == "max_node_test_samples") max_node_test_samples = stoi(f.second);

		// bagging control
		else if (f.first == "single_sample_per_pid") single_sample_per_pid = stoi(f.second);
		else if (f.first == "bag_with_repeats") bag_with_repeats = stoi(f.second);
		else if (f.first == "bag_prob") bag_prob = stof(f.second);
		else if (f.first == "bag_ratio") bag_ratio = stof(f.second);
		else if (f.first == "bag_feat") bag_feat = stof(f.second);

		// feature sampling
		else if (f.first == "ntry") ntry = stoi(f.second);
		else if (f.first == "ntry_prob") ntry_prob = stof(f.second);


		// missing value
		else if (f.first == "missing_val") missing_val = stof(f.second);
		else if (f.first == "missing_method_str") {
			missing_method_str = f.second;
			missing_method = TQRF_Forest::get_missing_value_method(missing_method_str);
		}

		// categorial features
		else if (f.first == "nvals_for_categorial") nvals_for_categorial = stoi(f.second);
		else if (f.first == "categorial_str") { boost::split(categorial_str, f.second, boost::is_any_of(",")); }
		else if (f.first == "categorial_tags") { boost::split(categorial_tags, f.second, boost::is_any_of(",")); }

		// weights
		else if (f.first == "case_wgt") case_wgt = stof(f.second);

		// boosting
		else if (f.first == "nrounds") nrounds = stoi(f.second);
		else if (f.first == "min_p") min_p = stof(f.second);
		else if (f.first == "max_p") max_p = stof(f.second);
		else if (f.first == "alpha") alpha = stof(f.second);
		else if (f.first == "wgts_pow") wgts_pow = stof(f.second);

		// lists
		else if (f.first == "tuning_size") tuning_size = stof(f.second);
		else if (f.first == "tune_max_depth") tune_max_depth = stoi(f.second);
		else if (f.first == "tune_min_node_size") tune_min_node_size = stoi(f.second);

		// gradient descent
		else if (f.first == "gd_rate")	gd_rate = stof(f.second);
		else if (f.first == "gd_batch")	gd_batch = stoi(f.second);
		else if (f.first == "gd_momentum")	gd_momentum = stof(f.second);
		else if (f.first == "gd_lambda")	gd_lambda = stof(f.second);
		else if (f.first == "gd_epochs")	gd_epochs = stoi(f.second);


		// sanities
		else if (f.first == "test_for_inf") test_for_inf = stoi(f.second);
		else if (f.first == "test_for_missing") test_for_missing = stoi(f.second);

		// prediction configuration
		else if (f.first == "only_this_categ") only_this_categ = stoi(f.second);
		else if (f.first == "predict_from_slice") predict_from_slice = stoi(f.second);
		else if (f.first == "predict_to_slice") predict_to_slice = stoi(f.second);
		else if (f.first == "predict_sum_times") predict_sum_times = stoi(f.second);

		// debug
		else if (f.first == "verbosity") verbosity = stoi(f.second);
		else if (f.first == "ids_to_print") ids_to_print = stoi(f.second);
		else if (f.first == "debug") debug = stoi(f.second);

		else {
			MERR("TQRF_Params::init(): No such parameter \'%s\'\n", f.first.c_str());
			return -1;
		}
	}

	tree_type_i = TQRF_Forest::get_tree_type(tree_type);
	samples_time_unit_i = med_time_converter.string_to_type(samples_time_unit);
	time_slice_unit_i = med_time_converter.string_to_type(time_slice_unit);

	if (predict_from_slice >= 0 && 
		((predict_to_slice - predict_from_slice + 1) > n_time_slices || predict_to_slice < 0 || predict_from_slice < 0 || predict_from_slice >= n_time_slices || predict_to_slice >= n_time_slices) ) {
		MTHROW_AND_ERR("TQRF: Conflicting params: n_time_slices=%d <==> predict_from_slice=%d predict_to_slice=%d\n", n_time_slices, predict_from_slice, predict_to_slice);
	}

	if (time_slices_wgts.size() == 0) {
		time_slices_wgts.resize(time_slices.size(), (float)1);
	}
	else {
		if (time_slices_wgts.size() != time_slices.size())
			MTHROW_AND_ERR("TQRF: Conflicting params sizes: time_slices_wgts & time_slices\n");
		float s = 0;
		for (auto w : time_slices_wgts) s+=w;
		s = (float)time_slices_wgts.size()/s;
		for (auto &w : time_slices_wgts) w = w * s;

	}

	return 0;
}

//================================================================================================
// Quantized_Feat
//================================================================================================
//------------------------------------------------------------------------------------------------
// quantize_feat(int i_feat, TQRF_Params &params) :
// assumes: nfeat, orig_data, ncateg, y are initialized and qx, q_to_val initialized to right size.
// actually doing the quantization:
// (1) prepare a value list and sort it.
// (2) count the number of different values.
// (3??) in case of categorial and adjacent values with a single same category - unite them. (Not relevant at the moment due to time slicing)
// (4) if left with less than max_q values : we finished
// (5) Otherwise: we split given the split strategy (currently - even splits. Future Idea: better subsampling of tails/information rich areas)
//
// Our basic convention is that missing value for each feature is mapped to q=0.
//------------------------------------------------------------------------------------------------
int Quantized_Feat::quantize_feat(int i_feat, TQRF_Params &params)
{
	vector<pair<float,int>> curr_data;
	const vector<float> *data;
	data = orig_data[i_feat];

	//if (params.verbosity > 0) MLOG("Quantized_Feat::quantize_feat :: feat %d %s :: nvals %d\n", i_feat, feat_names[i_feat].c_str(), data->size());

	curr_data.resize(data->size());
	for (int i=0; i<(int)data->size(); i++) {
		if ((*data)[i] != params.missing_val)
		curr_data[i] ={ (*data)[i], i };
	}


	if (params.verbosity > 2) {
		for (int i=0; i<params.ids_to_print; i++)
			MLOG("Quantuized Feat (unsorted) :: feat %d :: i %d :: curr_data %f %d\n", i_feat, i, curr_data[i].first, curr_data[i].second);
	}

	// sorting for easier quantization later
	sort(curr_data.begin(), curr_data.end());

	if (params.verbosity > 2) {
		for (int i=0; i<params.ids_to_print; i++)
			MLOG("Quantuized Feat (sorted) :: feat %d :: i %d :: curr_data %f %d\n", i_feat, i, curr_data[i].first, curr_data[i].second);
	}


	// count diff values
	vector<pair<float, int>> v_counts;
	float prev = curr_data[0].first;
	int n = 1; // 0 is kept for missing values
	for (auto v : curr_data) {
		if (v.first == prev)
			n++;
		else {
			v_counts.push_back({ prev,n });
			prev = v.first;
			n = 1;
		}
	}
	v_counts.push_back({ prev,n }); // last count

	if (params.verbosity > 0) MLOG("Quantized_Feat::quantize_feat :: feat %d %s :: ndiff %d (max_q %d)\n", i_feat, feat_names[i_feat].c_str(), v_counts.size(), params.max_q);

	// Step2 of calculating which features are categorial
	if (v_counts.size() < params.nvals_for_categorial) is_categorial_feat[i_feat] = 1;

	// prepare q_to_val
	// the convention is that if a value is in the range (q_to_val[i-1],q_to_val[i]] then its quantized value will be i
	if (v_counts.size() < params.max_q) {
		q_to_val[i_feat] = { params.missing_val }; // 0 place is special : always there to signal us the missing val cases
		for (auto &v : v_counts)
			q_to_val[i_feat].push_back(v.first);
	}
	else {
		// need more work to be done, we have too many different values... 
		// our algorithm is simply trying to make the i-th qval such that there are (i+1)/max_q * len , of the elements
		// up to some fixes that are caused due to integer numbers
		int delta = (int)((float)curr_data.size()/((float)params.max_q-1));
		int j = 0;
		int len = (int)curr_data.size();
		q_to_val[i_feat] ={ params.missing_val };
		while (j < len-1) {
			int prev_j = j;
			j = j + delta;
			if (j >= len) j = len - 1;
			float q = curr_data[j].first;
			while ((j < len-1) && (curr_data[j+1].first == q)) j++;
			if (params.verbosity > 2) MLOG("%s : %f : j %d size %d\n", feat_names[i_feat].c_str(), q, j, j-prev_j);
			q_to_val[i_feat].push_back(q);
		}
	}

	if (params.verbosity > 1) {
		MLOG("Quantized_Feat::quantize_feat :: %d %s :: q_to_val size %d :: ", i_feat, feat_names[i_feat].c_str(), q_to_val[i_feat].size());
		for (auto q : q_to_val[i_feat])
			MLOG(" %f :", q);
		MLOG("\n");
	}

	// now q_to_val is ready and all that is left is to actually use it to create qx
	int q_i = 1; // missing values will be filled later
	int q_size = (int)q_to_val[i_feat].size();
	qx[i_feat].resize(data->size());
	for (auto &v : curr_data) {
		while (q_i < q_size && v.first > q_to_val[i_feat][q_i])
			q_i++;
		qx[i_feat][v.second] = (short)q_i;
	}

	// fill in missing vals
	for (int i=0; i<(int)data->size(); i++)
		if ((*data)[i] == params.missing_val)
			qx[i_feat][i] = 0;

	return 0;
}

//------------------------------------------------------------------------------------------------
// Quantized_Feat::init(const MedFeatures &medf, TQRF_Params &params) :
// initializations needed for all trees together, mainly data quantization , time slices, and
// filling in all the needed variables in Quantized_Feat
//------------------------------------------------------------------------------------------------
int Quantized_Feat::init(const MedFeatures &medf, TQRF_Params &params)
{
	// filling in needed variables
	orig_medf = &medf;
	orig_data.clear();
	feat_names.clear();
	for (auto &df : medf.data) {
		orig_data.push_back(&df.second);
		feat_names.push_back(df.first);
		if (params.verbosity > 2) MLOG("Quantized_Feat:: %s :: %d elements :: %f %f %f %f ....\n", feat_names.back().c_str(), orig_data.back()->size(),
			(*orig_data.back())[0], (*orig_data.back())[1], (*orig_data.back())[2], (*orig_data.back())[3]);
	}


	nfeat = (int)(medf.data.size());

	// init y and get ncateg if needed
	y.resize(medf.samples.size());
	y_i.resize(medf.samples.size());

#pragma omp parallel for
	for (int i=0; i<medf.samples.size(); i++) {
		y[i] = medf.samples[i].outcome;
		y_i[i] = (int)y[i];
	}

	if (params.tree_type_i != TQRF_TREE_REGRESSION) {
		float maxy = *max_element(y.begin(), y.end());
		ncateg = (int)maxy + 1;
		// TBD :: sanity test that the y's in this case are all in the 0,ncateg-1 integers range ... currently assuming this
	}
	else
		ncateg = 0;


	// time slices
	if (init_lists(medf, params) < 0)
		return -1;

	// time slices
	if (init_time_slices(medf, params) < 0)
		return -1;

	// categorial features Step1 (Step2 will be done in quantize_feat)
	is_categorial_feat.clear();
	is_categorial_feat.resize(nfeat, 0);
	for (int i=0; i<nfeat; i++) {
		for (int j=0; j<params.categorial_str.size(); j++) {
			if (feat_names[i].find(params.categorial_str[j]) != string::npos) {
				is_categorial_feat[i] = 1;
				break;
			}
		}

		for (int j=0; j<params.categorial_tags.size(); j++) {
			if (medf.tags.at(feat_names[i]).find(params.categorial_tags[j]) != medf.tags.at(feat_names[i]).end()) {
				is_categorial_feat[i] = 1;
				break;
			}
		}
	}	


	// init q arrays to right size
	qx.resize(nfeat);
	q_to_val.resize(nfeat);

	int rc = 0;
	// quantization of features
#pragma omp parallel for
	for (int i=0; i<nfeat; i++)
		if (quantize_feat(i, params) < 0)
			rc = -1;

	medf.get_feature_names(feature_names);


	wgts = orig_medf->weights;

	if (wgts.size() == 0) {
		wgts.resize(y.size(), 1);
		if (params.case_wgt != 1) {
			for (int i=0; i<y.size(); i++)
				if (y[i]) wgts[i] = params.case_wgt;
		}
		orig_wgts = wgts;
		probs.resize(y.size());
		w_to_sum.resize(y.size());
		sum_over_trees.resize(y.size(), vector<float>(params.ncateg, 0));

	}

	// rc
	return rc;
}

//------------------------------------------------------------------------------------------------
int Quantized_Feat::init_lists(const MedFeatures &medf, TQRF_Params &params)
{
	lists.resize(2); // currently we may have at 0 the tree training ids and at 1 the weights tuning list
	if (params.tuning_size <= 0) {
		for (int i=0; i<medf.samples.size(); i++)
			lists[0].push_back(i);
		return 0;
	}

	// going over ids and splitting them to lists based on the pid
	unordered_map<int,int> pid2list;
	for (int i=0; i<medf.samples.size(); i++) {
		int pid = medf.samples[i].id;
		if (pid2list.find(pid) == pid2list.end()) {
			if (rand_1() < params.tuning_size)
				pid2list[pid] = 1;
			else
				pid2list[pid] = 0;
		}
		lists[pid2list[pid]].push_back(i);
	}

	shuffle(lists[0].begin(), lists[0].end(), globalRNG::get_engine());

	MLOG("TQRF: split with tuning_size=%f : list0: %d list1: %d overall: %d\n", params.tuning_size, lists[0].size(), lists[1].size(), medf.samples.size());

	return 0;
}

//------------------------------------------------------------------------------------------------
int Quantized_Feat::init_pre_bagging(TQRF_Params &params)
{	
	int _n_categ = 2;
	time_categ_pids.resize(params.n_time_slices, vector<vector<int>>(_n_categ));
	time_categ_idx.resize(params.n_time_slices, vector<vector<int>>(_n_categ));
	categ_pids.resize(_n_categ);
	categ_idx.resize(_n_categ);

	vector<vector<vector<int>>> empty_time_categ_idx(params.n_time_slices, vector<vector<int>>(_n_categ));

	unordered_set<int> pid_used;

	for (int l=0; l<lists[0].size(); l++) {
		int i = lists[0][l];
		const MedSample &s = orig_medf->samples[i];

		// currently we split categories to 0 and the others, sampling the others as they are (later)
		// in theory it is possible to improve this and build a mechanism to control the sampling of every category
		// in every time slice.

		int c = 0;
		if (params.tree_type_i == TQRF_TREE_REGRESSION || y[i] != 0) c=1;

		//if (y[i]!=0 && y[i]!=1) MLOG("ERROR !!!!!!!!!!!!! i %d c %d y %f\n", i, c , y[i]);

		int t = last_time_slice[i];
		int pid = s.id;

		if (pid_used.find(pid) == pid_used.end()) { time_categ_pids[t][c].push_back(pid); categ_pids[c].push_back(pid); }

		time_categ_idx[t][c].push_back(i);
		categ_idx[c].push_back(i);

		if (pid2time_categ_idx.find(pid) == pid2time_categ_idx.end()) pid2time_categ_idx[pid] = empty_time_categ_idx;
		pid2time_categ_idx[pid][t][c].push_back(i);
		if (c == 0 && t > 0) pid2time_categ_idx[pid][0][0].push_back(i); // making life easier for controls handling
		pid_used.insert(pid);
	}

	for (int c=0; c<_n_categ; c++) {
		MLOG("TQRF: samples [c=%d p %6d i %6d] ::", c, categ_pids[c].size(), categ_idx[c].size());
		for (int t=0; t<params.n_time_slices; t++)
			MLOG(" [t=%d] pids %6d idx %6d :", t, time_categ_pids[t][c].size(), time_categ_idx[t][c].size());
		MLOG("\n");
	}


	// for convinience we set the 0 vector for t=0 to be all the controls
	time_categ_pids[0][0] = categ_pids[0];
	time_categ_idx[0][0] = categ_idx[0];

	return 0;
}

//------------------------------------------------------------------------------------------------
int Quantized_Feat::init_time_slices(const MedFeatures &medf, TQRF_Params &params)
{
	if (params.tree_type_i != TQRF_TREE_REGRESSION) {
		if (params.time_slices.size() == 0) {
			if (params.time_slice_size > 0)
				for (int i=0; i<params.n_time_slices; i++)
					params.time_slices.push_back((i+1)*params.time_slice_size);
			else
				params.time_slices.push_back(TQRF_MAX_TIME_SLICE); // single infinite cell
		}
	}
	else {
		// regression tree case - currently using a single infinity time slice
		params.time_slices.push_back(TQRF_MAX_TIME_SLICE); // single infinite cell
	}

	n_time_slices = (int)params.time_slices.size();

	if (params.verbosity > 0) {
		MLOG("time_slices (%d) : ", params.time_slices.size());
		for (auto t : params.time_slices)
			MLOG("%d ", t);
		MLOG("\n");
	}

	// slices for each sample
	// we assume that '0' outcome is the control, for binary and multicategory cases
	// The regression case for time slices is much more complex, and needs more input from the user: currently a project to the future
	last_time_slice.resize(medf.samples.size());
#pragma omp parallel for
	for (int i=0; i<medf.samples.size(); i++) {
		const MedSample &s = medf.samples[i];
		int d1 = med_time_converter.convert_times(params.samples_time_unit_i, params.time_slice_unit_i, (int)s.outcomeTime);
		int d2 = med_time_converter.convert_times(params.samples_time_unit_i, params.time_slice_unit_i, (int)s.time);
		int time_diff = d1-d2;
		int j = 0;
		for (j=0; j<params.time_slices.size(); j++)
			if (time_diff < params.time_slices[j])
				break;
		if (y[i] == 0 || j<params.time_slices.size()) {
			last_time_slice[i] = min(j, (int)params.time_slices.size()-1);
			//if (params.debug) if (y[i] && j==1) { y[i] = -1; y_i[i] = -1; } // DEBUG
		}
		else {
			// interesting case in which we have y[i] != 0 that ended AFTER the LAST time slice.
			// in this case we have to treat the sample as a 0 outcome (!)
			last_time_slice[i] = (int)params.time_slices.size()-1;
			y[i] = 0; 
			y_i[i] = 0;
			//MLOG(">>>>> sample %d : pid %d otime %d outcome %f time %d : d1 %d d2 %d diff %d : last_time_slice %d y %f (%d)\n", i, s.id, s.outcomeTime, s.outcome, s.time, d1, d2, time_diff, last_time_slice[i], y[i], y[i] != s.outcome);

		}

		if (params.verbosity > 0 && i < params.ids_to_print)
			MLOG("sample %d : pid %d otime %d outcome %f time %d : d1 %d d2 %d diff %d : last_time_slice %d y %f (%d)\n", i, s.id, s.outcomeTime, s.outcome, s.time, d1, d2, time_diff, last_time_slice[i], y[i], y[i] != s.outcome);
	}

	// getting stats for each time slice
	vector<int> cnt[2];
	cnt[0].resize(n_time_slices, 0);
	cnt[1].resize(n_time_slices, 0);
	for (int i=0; i<last_time_slice.size(); i++) {
		int j;
		for (j=0; j<last_time_slice[i]; j++)
			cnt[0][j]++;
		if (y[i] == 0)
			cnt[0][j]++;
		else
			cnt[1][j]++;
	}
	for (int j=0; j<n_time_slices; j++) {
		if (params.verbosity > 0)
			MLOG("Slice %d : %d : 0: %d 1: %d p: %f\n", j, params.time_slices[j], cnt[0][j], cnt[1][j], (float)cnt[1][j]/(1+cnt[1][j]+cnt[0][j]));
		if (cnt[0][j] < MIN_ELEMENTS_IN_TIME_SLICE || cnt[1][j] < MIN_ELEMENTS_IN_TIME_SLICE) {
			MWARN("TQRF: WARNING : time slice %d (%d) too small or non variable :0: %d 1: %d p: %f\n", j, params.time_slices[j], cnt[0][j], cnt[1][j], (float)cnt[1][j]/(1+cnt[0][j]));
		}
	}

	init_pre_bagging(params);
	return 0;
}


//================================================================================================
// TQRF_Forest
//================================================================================================
//------------------------------------------------------------------------------------------------
int TQRF_Forest::n_preds_per_sample() const
{
	if (params.tree_type_i == TQRF_TREE_REGRESSION) {
		MTHROW_AND_ERR("TQRF Regression trees not available yet...\n");
		return -1;
	}
	//else {
	// categorial cases
	int n = params.ncateg;
	if (params.only_this_categ >= 0) n=1;

	int m = params.n_time_slices;

	if (params.predict_from_slice >=0 && params.predict_from_slice < params.n_time_slices &&
		params.predict_to_slice >= params.predict_from_slice && params.predict_to_slice < params.n_time_slices)
		m = params.predict_to_slice - params.predict_from_slice + 1;

	if (params.predict_sum_times)
		m = 1;

	return n*m;
}

//------------------------------------------------------------------------------------------------
int TQRF_Forest::get_tree_type(const string &str)
{
	if (boost::iequals(str, "entropy")) return TQRF_TREE_ENTROPY;
	if (boost::iequals(str, "likelihood")) return TQRF_TREE_LIKELIHOOD;
	if (boost::iequals(str, "wlikelihood") || boost::iequals(str, "weighted_likelihood")) return TQRF_TREE_WEIGHTED_LIKELIHOOD;
	if (boost::iequals(str, "devel")) return TQRF_TREE_DEV;
	if (boost::iequals(str, "regression")) return TQRF_TREE_REGRESSION;

	return TQRF_TREE_UNDEFINED;
}

//------------------------------------------------------------------------------------------------
int TQRF_Forest::get_missing_value_method(const string &str)
{
	if (boost::iequals(str, "mean")) return TQRF_MISSING_VALUE_MEAN;
	if (boost::iequals(str, "median")) return TQRF_MISSING_VALUE_MEDIAN;
	if (boost::iequals(str, "larger")) return TQRF_MISSING_VALUE_LARGER_NODE;
	if (boost::iequals(str, "left")) return TQRF_MISSING_VALUE_LEFT;
	if (boost::iequals(str, "right")) return TQRF_MISSING_VALUE_RIGHT;
	if (boost::iequals(str, "rand")) return TQRF_MISSING_VALUE_RAND_ALL;
	if (boost::iequals(str, "rand_each_sample")) return TQRF_MISSING_VALUE_RAND_EACH_SAMPLE;

	return TQRF_MISSING_VALUE_NOTHING;
}


//------------------------------------------------------------------------------------------------
void TQRF_Forest::init_tables(Quantized_Feat &qfeat)
{
	if (params.tree_type_i == TQRF_TREE_ENTROPY) {

		// initializing a precomputed n * log(n) table for the sake of computing entropy scores
		int max_samples = (int)qfeat.y.size()+1;
		params.log_table.resize(max_samples);
		params.log_table[0] = 0; // we will always use it in an nlogn manner hence the 0 rather than -inf
		for (int i = 1; i < max_samples; i++)
			params.log_table[i] = (double)i * log((double)i)/log(2.0);

	}
	if ((params.tree_type_i == TQRF_TREE_LIKELIHOOD) || (params.tree_type_i == TQRF_TREE_WEIGHTED_LIKELIHOOD)) {

		MLOG("TQRF: Preparing sum log table\n");
		// initializing a precomputed sigma(log(n)) table for the sake of computing log likelihood scores
		int max_samples = (int)qfeat.y.size()+1;
		params.log_table.resize(max_samples);
		params.log_table[0] = 0; 
		for (int i = 1; i < max_samples; i++)
			params.log_table[i] = params.log_table[i-1] + log((double)i);
	}
}

//------------------------------------------------------------------------------------------------
int TQRF_Forest::Train(const MedFeatures &medf, const MedMat<float> &Y)
{
	if (params.nrounds > 1)
		return Train_AdaBoost(medf, Y);

	MLOG("====================================TQRF==================================\n");
	MLOG("TQRF_Forest: Running with params: %s\n", params.init_string.c_str());
	MLOG("TQRF_Forest: Train: medf : %d x %d \n", medf.data.size(), medf.samples.size());

	MedTimer timer;

	timer.start();
	// first - quantifying data
	Quantized_Feat qfeat;
	qfeat.init(medf, params);

	// additional initializations of needed lookup tables (which will be kept in params)
	// for example: log tables for entropy scores etc...
	init_tables(qfeat);

	timer.take_curr_time();
	MLOG("TQRF_Forest: Init qfeat and tables time: %f sec\n", timer.diff_sec());

	MLOG("TQRF_Forest: Starting run on %d trees\n", params.ntrees);

	timer.start();
	// creating the trees and parallelize train on each tree
	trees.resize(params.ntrees);
#pragma omp parallel for
	for (int i=0; i<params.ntrees; i++) {
		trees[i].id = i;
		trees[i].tree_type = params.tree_type_i;
		trees[i].init(qfeat, params);
		trees[i].Train();
		if (params.verbosity > 0) MLOG("TQRF: Trained Tree %d : type %d : indexes %d : feats %d : n_nodes %d\n", trees[i].id, trees[i].tree_type, trees[i].indexes.size(), trees[i].i_feats.size(), trees[i].nodes.size());
	}

	timer.take_curr_time();
	MLOG("TQRF_Forest: %d Trees training time : %f sec \n", params.ntrees, timer.diff_sec());

	print_average_bagging(qfeat.n_time_slices, qfeat.ncateg);

	//// DEBUG - Saving all indexes
	if (0) {
		AllIndexes a_i;
		a_i.init_all_indexes(trees);
		a_i.write_to_file("all_indexes.bin");
	}

	if (params.tuning_size > 0)	tune_betas(qfeat);

	//prepare for predict
	if (alphas.size() != trees.size()) alphas.resize(trees.size(), 1);

	return 0;
}

//------------------------------------------------------------------------------------------------
void TQRF_Forest::print_average_bagging(int _n_time_slices, int _n_categ)
{
	// calculating average bagging per tree
	vector<vector<float>> avg_bag;
	avg_bag.resize(_n_time_slices, vector<float>(_n_categ, 0));
	for (int i=0; i<params.ntrees; i++) {
		if (trees[i].nodes.size() > 1)
			for (int t=0; t<_n_time_slices; t++)
				for (int c=0; c<_n_categ; c++) {
					avg_bag[t][c] += trees[i].nodes[0].time_categ_count[t][c];
					//MLOG("node 0: Tree %d , t %d c %d count %f\n", i, t, c, trees[i].nodes[0].time_categ_count[t][c]);
				}
	}
	for (int t=0; t<_n_time_slices; t++)
		for (int c=0; c<_n_categ; c++)
			avg_bag[t][c] /= params.ntrees;

	MLOG("TQRF_Forest: average bagging report :");
	for (int t=0; t<_n_time_slices; t++)
		for (int c=0; c<_n_categ; c++)
			MLOG(" [t=%d][c=%d] %d :", t, c, (int)avg_bag[t][c]);
	MLOG("\n");
}

//------------------------------------------------------------------------------------------------
int TQRF_Forest::Train_AdaBoost(const MedFeatures &medf, const MedMat<float> &Y)
{
	MLOG("==========================TQRF AdaBoost Mode ==================================\n");
	MLOG("TQRF_Forest: Running with params: %s\n", params.init_string.c_str());
	MLOG("TQRF_Forest: Train: medf : %d x %d \n", medf.data.size(), medf.samples.size());

	MedTimer timer;

	timer.start();
	// first - quantifying data
	Quantized_Feat qfeat;
	params.case_wgt = 1; // currently we always start with all weights equal to 1 in adaboost mode

	qfeat.init(medf, params);

	// additional initializations of needed lookup tables (which will be kept in params)
	// for example: log tables for entropy scores etc...
	init_tables(qfeat);

	// allocating all trees
	trees.resize(static_cast<size_t>(params.ntrees) * static_cast<size_t>(params.nrounds));
	alphas.resize(static_cast<size_t>(params.ntrees) * static_cast<size_t>(params.nrounds), 0);

	// allocating zero counts for each sample
	// for each count we only need to keep the counts for its last slice
	vector<vector<float>> sample_slice_counts;
	sample_slice_counts.resize(qfeat.y.size(), vector<float>(qfeat.ncateg, 0));

	MedMat<float> x;
	qfeat.orig_medf->get_as_matrix(x, {},qfeat.lists[0]);

	timer.take_curr_time();
	MLOG("TQRF_Forest: Init qfeat and tables time: %f sec\n", timer.diff_sec());

	MLOG("TQRF_Forest: Starting run of %d adaboost rounds\n", params.nrounds);

	timer.start();
	for (int round=0; round<params.nrounds; round++) {

		int from_tree = round*params.ntrees;
		int to_tree = from_tree + params.ntrees;

		// weights are ready - at this stage we need to train ntrees
#pragma omp parallel for
		for (int i=from_tree; i<to_tree; i++) {
			trees[i].id = i;
			trees[i].tree_type = params.tree_type_i;
			trees[i].init(qfeat, params);
			trees[i].Train();
			if (params.verbosity > 0) MLOG("TQRF: Trained Tree %d : type %d : indexes %d : feats %d : n_nodes %d\n", trees[i].id, trees[i].tree_type, trees[i].indexes.size(), trees[i].i_feats.size(), trees[i].nodes.size());
		}

		// now updating our sample counts
		update_counts(sample_slice_counts, x, qfeat, 0, round);

		timer.take_curr_time();
		MLOG("TQRF AdaBoost: finished round %d : %f sec : %d/%d trees so far\n", round, timer.diff_sec(), to_tree, trees.size());
		//for (int i=0; i<20; i++) MLOG("(%d) y_i %d last %d counts %f,%f probs %f wsum %f wgt %f \n", i, qfeat.y_i[i], qfeat.last_time_slice[i], sample_slice_counts[i][0], sample_slice_counts[i][1], probs[i], wsum, qfeat.wgts[i]);
	}

	timer.take_curr_time();
	MLOG("TQRF_Forest: %d Trees training time : %f sec \n", trees.size(), timer.diff_sec());

	print_average_bagging(qfeat.n_time_slices, qfeat.ncateg);

	if (params.tuning_size > 0)	tune_betas(qfeat);

	return 0;
}

//---------------------------------------------------------------------------------------------------------------------------------
int TQRF_Forest::update_counts(vector<vector<float>> &sample_counts, MedMat<float> &x, Quantized_Feat &qf, int zero_counts, int round)
{
	int nsamples = x.nrows;
	int from_tree = round*params.ntrees;
	int to_tree = from_tree + params.ntrees;

	assert(nsamples == qf.lists[0].size());

#pragma omp parallel for
	for (int i=0; i<nsamples; i++) {
		int idx = qf.lists[0][i];
		int i_t = qf.last_time_slice[idx];

		// summing counts over trees and only on the relevant time point for our idx
		fill(qf.sum_over_trees[i].begin(), qf.sum_over_trees[i].end(), (float)0);		
		for (int i_tree = from_tree; i_tree < to_tree; i_tree++) {
			TQRF_Node *cnode = trees[i_tree].Get_Node(x, i, params.missing_val);	
			for (int c=0; c<params.ncateg; c++) {
				qf.sum_over_trees[i][c] += cnode->time_categ_count[i_t][c];
			}
		}
		float csum = 0;
		for (int c=0; c<params.ncateg; c++)
			csum += qf.sum_over_trees[i][c];

		qf.probs[i] = (csum > 0) ? qf.sum_over_trees[i][qf.y_i[idx]]/csum : 0;

		if (qf.probs[i] < params.min_p) qf.probs[i] = params.min_p;
		else if (qf.probs[i] > params.max_p) qf.probs[i] = params.max_p;

		//qf.wgts[i] = -log(qf.probs[i]);
		qf.w_to_sum[i] = (float)pow(-log(1-qf.probs[i]), params.wgts_pow); // local weights used for calculating alpha
		//qf.w_to_sum[i] = (float)-log(1-qf.probs[i]); // local weights used for calculating alpha
		//qf.w_to_sum[i] = (float)1/(1-qf.probs[i]); // local weights used for calculating alpha
	}

	// update alpha
	double sumw = 0;
	if (params.alpha <= 0) {
		// in this case we want alpha to be estimated using w_to_sum average
		for (int i=0; i<nsamples; i++) sumw += qf.w_to_sum[i];
		sumw /= (float)nsamples;
		if (round == 0) qf.alpha0 = (float)(sumw*sumw);
		for (int i=round*params.ntrees; i<(round+1)*params.ntrees; i++) 
			alphas[i] = (float)(sumw*sumw)/qf.alpha0;
	} else {
		for (int i=round*params.ntrees; i<(round+1)*params.ntrees; i++) {
			if (round == 0)
				alphas[i] = 1;
			else
				alphas[i] = alphas[i-params.ntrees] * params.alpha;
		}
	}


#pragma omp parallel for
	for (int i=0; i< nsamples; i++) {
		float csum = 0;
		int idx = qf.lists[0][i];
		for (int c=0; c<params.ncateg; c++) {
			//if (i<10) MLOG("counts before(%d) i %d c %d: %f,%f : i_t %d : sum_over_trees(%d-%d) %f,%f : alpha %f : ", params.ncateg, i, c, sample_counts[i][0], sample_counts[i][1], i_t, from_tree, to_tree, sum_over_trees[i_t][0], sum_over_trees[i_t][1], params.alpha);
			if (zero_counts)
				sample_counts[i][c] = qf.sum_over_trees[i][c];
			else {
				//sample_counts[i][c] = (1-params.alpha)*sample_counts[i][c] + params.alpha*sum_over_trees[i_t][c];
				sample_counts[i][c] = sample_counts[i][c] + alphas[from_tree]*qf.sum_over_trees[i][c];
			}

			csum += sample_counts[i][c];
		}
		qf.probs[i] = (csum > 0) ? sample_counts[i][qf.y_i[idx]]/csum : 0;

		if (qf.probs[i] < params.min_p) qf.probs[i] = params.min_p;
		else if (qf.probs[i] > params.max_p) qf.probs[i] = params.max_p;

		qf.wgts[idx] = pow(-log(qf.probs[i]), params.wgts_pow);
		//qf.wgts[idx] = -log(qf.probs[i]);
		//qf.wgts[idx] = 1/(qf.probs[i]);
		qf.orig_wgts[idx] = qf.wgts[idx];
		//if (i<10) MLOG("counts after : %f,%f\n", sample_counts[i][0], sample_counts[i][1]);

	}
	
	float wsum = 0;
	float sum_p = 0;
	for (int i=0; i<nsamples; i++) {
		int idx = qf.lists[0][i];
		wsum += qf.wgts[idx];
		sum_p += qf.probs[i];
	}

	wsum = (float)nsamples/wsum;
	sum_p /= (float)nsamples;

	for (int i=0; i<nsamples; i++) {
		int idx = qf.lists[0][i];
		qf.wgts[idx] = qf.wgts[idx] * wsum;
	}

	MLOG("ROUND %d [%d-%d] %d samples : avg prob: %f wsum: %f alpha %f\n", round, from_tree, to_tree, nsamples, sum_p, wsum , alphas[from_tree]);

	return 0;

}

//------------------------------------------------------------------------------------------------
int TQRF_Forest::Predict(MedMat<float> &x, vector<float> &preds) const
{
	MLOG("TQRF_Forest: type %d : Running predict on matrix of %d x %d\n", params.tree_type_i, x.nrows, x.ncols);

	if (params.tree_type_i == TQRF_TREE_REGRESSION) {
		MTHROW_AND_ERR("TQRF Regression Trees not available yet...\n");
		return -1;
	}

	if (params.tree_type_i == TQRF_TREE_ENTROPY || params.tree_type_i == TQRF_TREE_LIKELIHOOD  || params.tree_type_i == TQRF_TREE_WEIGHTED_LIKELIHOOD || params.tree_type_i == TQRF_TREE_DEV) {
		return Predict_Categorial(x, preds);
	}

	return -1;
}

//------------------------------------------------------------------------------------------------
int TQRF_Forest::Predict_Categorial(MedMat<float> &x, vector<float> &preds) const
{
	MLOG("TQRF_Forest : Predict_Categorial\n");
	MedTimer timer;

	timer.start();

	int n_per_sample = n_preds_per_sample();
	// get preds to the right size (so we can thread it all)
	preds.resize(x.nrows * n_per_sample, (float)-1);

	//if (alphas.size() != trees.size()) alphas.resize(trees.size(), 1); //done in Learn

#pragma omp parallel for
	for (int i=0; i<x.nrows; i++) {
		vector<vector<float>> sum_over_trees;
		sum_over_trees.resize(params.n_time_slices, vector<float>(params.ncateg, 0));
		for (int i_tree = 0; i_tree < trees.size(); i_tree++) {
			int beta_idx;
			const TQRF_Node *cnode = trees[i_tree].Get_Node_for_predict(x, i, params.missing_val, beta_idx);
			float beta = (beta_idx < 0) ? 1 : betas[beta_idx];
			for (int t=0; t<params.n_time_slices; t++)
				for (int c=0; c<params.ncateg; c++)
					sum_over_trees[t][c] += alphas[i_tree] * beta * cnode->time_categ_count[t][c];
		}

		vector<float> sum_t(params.n_time_slices, 0);
		for (int t=0; t<params.n_time_slices; t++)
			for (int c=0; c<params.ncateg; c++)
				sum_t[t] += sum_over_trees[t][c];

		vector<vector<float>> probs_t;
		probs_t.resize(params.n_time_slices, vector<float>(params.ncateg, 0));
		for (int t=0; t<params.n_time_slices; t++)
			for (int c=0; c<params.ncateg; c++) {
				if (sum_t[t] > 0)
					probs_t[t][c] = sum_over_trees[t][c]/sum_t[t];
			}

		// we are now ready to actually report the needed predicions
		int from_t = 0, to_t = params.n_time_slices-1;
		if (params.predict_from_slice >= 0) {
			from_t = params.predict_from_slice;
			to_t = params.predict_to_slice;
		}
		int from_c = 0, to_c = params.ncateg-1;
		if (params.only_this_categ >= 0) {
			from_c = params.only_this_categ;
			to_c = params.only_this_categ;
		}

		// abit km explanations:
		// when looking at 2 adjacent time slices , we have counts N0_c for t0 (and category c) and N1_c for t1
		// we have the following connection:
		// N0_c = SUM(N1_C) + Censored
		// Hence the survival (category 0) after each time is:
		// S0 = N0_0 / SUM(N0_c) , S1 = S0 * (N1_0 / SUM(N1_c))
		// The probability to get class 'c' during each of these times is
		// P0_c = N0_c / SUM(N0_c) , P1_c = S0 * (N1_c / SUM(N1_c)
		// Those are our estimates with this model

		vector<float> survival(params.n_time_slices);
		survival[0] = (float)1.0;
		for (int t=1; t<params.n_time_slices; t++)
			survival[t] = survival[t-1] * probs_t[t-1][0];
		int index = i*n_per_sample;
		//assert(from_t ==0 && to_t == 0 && from_c == 1 && to_c == 1); // DEBUG
		if (params.predict_sum_times) {
			for (int c=from_c; c<=to_c; c++) {
				preds[index] = 0;
				for (int t=from_t; t<=to_t; t++)
					preds[index] += survival[t] * probs_t[t][c];
				index++;
			}
		}
		else {
			for (int t=from_t; t<=to_t; t++)
				for (int c=from_c; c<=to_c; c++) {
					preds[index++] = survival[t] * probs_t[t][c];
				}
		}

	}

	timer.take_curr_time();
	MLOG("TQRF_Forest : Predict_Categorial : Finished : got %d predictions (n_per_sample %d) : time: %f sec\n", preds.size(), n_per_sample, timer.diff_sec());

	return 0;
}


//================================================================================================
// TQRF_Split_Stat family of classes
//================================================================================================
TQRF_Split_Stat *TQRF_Split_Stat::make_tqrf_split_stat(int tree_type)
{
	if (tree_type == TQRF_TREE_LIKELIHOOD)
		return new TQRF_Split_Likelihood;
	if (tree_type == TQRF_TREE_WEIGHTED_LIKELIHOOD)
		return new TQRF_Split_Weighted_Likelihood;
	if (tree_type == TQRF_TREE_ENTROPY)
		return new TQRF_Split_Entropy;
	if (tree_type == TQRF_TREE_REGRESSION)
		return new TQRF_Split_Regression;
	if (tree_type == TQRF_TREE_DEV)
		return new TQRF_Split_Dev;
	 
	return NULL;
}

//--------------------------------------------------------------------------------------------------------------------
// another source for randomization and speedup is calculating the split scores not at all quantized points
// but only on a subset of them.
// This method selects a random set of qpoints to work on, and returns them:
// an empty qpoints, means : test all points
// otherwise it will contain the q values sorted.
//--------------------------------------------------------------------------------------------------------------------
int TQRF_Split_Stat::get_q_test_points(int feat_max_q, TQRF_Params &params, vector<int> &_qpoints)
{
	int n = feat_max_q - 2; // possible split points are 1,2,...,feat_max_q-2 : 
							// 0 is not a split point: it is missing values , and feat_max_q-1 means we take it all to the left side, so there's no split


	if (params.qpoints_per_split == 0 || params.qpoints_per_split >= n) {
		_qpoints.clear();
		return 0;
	}

	_qpoints.resize(n);
	
	for (int i=0; i<n; i++)
		_qpoints[i] = i+1;

	for (int i=0; i<params.qpoints_per_split; i++)
		swap(_qpoints[i], _qpoints[i + rand_N(n-i)]);

	_qpoints.resize(params.qpoints_per_split);
	sort(_qpoints.begin(), _qpoints.end());
	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
int TQRF_Split_Categorial::init(Quantized_Feat &qf, TQRF_Params &params)
{
	// initializing counts[t][q][c] , sums[t][c]
	ncateg = qf.ncateg;
	maxq = params.max_q;
	nslices = qf.n_time_slices;

	counts.resize(nslices, vector<vector<int>>(maxq , vector<int>(ncateg)));
	sums.resize(nslices, vector<int>(ncateg));
	sums_t.resize(nslices);

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
// preparing counts for the current i_feat and node
int TQRF_Split_Categorial::prep_histograms(int i_feat, TQRF_Node &node, vector<int> &indexes, Quantized_Feat &qf, TQRF_Params &params)
{
	int feat_maxq = (int)qf.q_to_val[i_feat].size();

	// zero counts (is it needed at all??)
	for (int t=0; t<nslices; t++)
		for (int q=0; q<feat_maxq; q++)
			for (int c=0; c<ncateg; c++)
				counts[t][q][c] = 0;
	for (int t=0; t<nslices; t++) {
		sums_t[t] = 0;
		for (int c=0; c<ncateg; c++)
			sums[t][c] = 0;
	}

	// now going over each sample in the node (sometimes we have to sample)
	// and adding counts for each category.
	

	vector<short> &feat_qvals = qf.qx[i_feat];
	for (int i=node.from_idx; i<=node.to_idx; i++) {
		int idx = indexes[i];
		int q = feat_qvals[idx];
		int c = qf.y_i[idx];
		int t = qf.last_time_slice[idx];

		counts[t][q][c]++;
		// in case c is 0 we need now to also add a ++ for all cells [0...(t-1)][q][0] . However, we will do those in one swip
		// at the end which should be more efficient.
		if (q > 0) sums[t][c]++; // not summing q=0 cases: missing values are separate
	}

	// now we consider the cases of fewer qpoints 

	// get qpoints
	get_q_test_points(feat_maxq, params, qpoints);

	counts_q = feat_maxq;

	if (qpoints.size() != 0) {

		// checking fewer q points - given orderes in qpoints
		// we squeeze all q values to qpoints.size() values
		int prev = 0;
		for (int i=0; i<qpoints.size(); i++) {
			if (prev > i) {
				for (int t=0; t<nslices; t++)
					for (int c=0; c<ncateg; c++)
						counts[t][i][c] = counts[t][prev][c];
			}

			for (int j=prev+1; j<qpoints[i]; j++) {
				for (int t=0; t<nslices; t++)
					for (int c=0; c<ncateg; c++)
						counts[t][i][c] += counts[t][j][c];
				prev = qpoints[i];
			}
		}
		counts_q = (int)qpoints.size() + 1; // +1 for q=0
	}

	// summing all 0 counts in one swip
	for (int t=nslices-2; t>=0; t--) {
		int to_categ = (params.censor_cases) ? 1 : ncateg;
		for (int c=0; c<to_categ; c++)
			sums[t][0] += sums[t+1][c];
		for (int q=1; q<counts_q; q++)
			for (int c=0; c<to_categ; c++)
				counts[t][q][0] += counts[t+1][q][c];
	}

	total_sum = 0;
	for (int t=0; t<nslices; t++) {
		for (int c=0; c<ncateg; c++)
			sums_t[t] += sums[t][c];
		total_sum += sums_t[t];
	}

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
void TQRF_Split_Categorial::print_histograms()
{
	MLOG("nslices %d ncateg %d maxq %d counts_q %d\n", nslices, ncateg, maxq, counts_q);

	for (int q=0; q<counts_q; q++) {
		MLOG("counts q[%d] :", q);
		for (int c=0; c<ncateg; c++)
			for (int t=0; t<nslices; t++)
				MLOG(" c[%1d] t[%1d] %d :", c, t, counts[t][q][c]);
		MLOG("\n");
	}


	MLOG("sums :");
	for (int c=0; c<ncateg; c++)
		for (int t=0; t<nslices; t++)
			MLOG(" c[%1d] t[%1d] %d :", c, t, sums[t][c]);
	MLOG("\n");
}

//--------------------------------------------------------------------------------------------------------------------
int TQRF_Split_Weighted_Categorial::init(Quantized_Feat &qf, TQRF_Params &params)
{
	// initializing counts[t][q][c] , sums[t][c]
	ncateg = qf.ncateg;
	maxq = params.max_q;
	nslices = qf.n_time_slices;

	counts.resize(nslices, vector<vector<float>>(maxq, vector<float>(ncateg)));
	sums.resize(nslices, vector<float>(ncateg));
	sums_t.resize(nslices);

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
// preparing counts for the current i_feat and node
int TQRF_Split_Weighted_Categorial::prep_histograms(int i_feat, TQRF_Node &node, vector<int> &indexes, Quantized_Feat &qf, TQRF_Params &params)
{
	int feat_maxq = (int)qf.q_to_val[i_feat].size();

	// zero counts (is it needed at all??)
	for (int t=0; t<nslices; t++)
		for (int q=0; q<feat_maxq; q++)
			for (int c=0; c<ncateg; c++)
				counts[t][q][c] = 0;
	for (int t=0; t<nslices; t++) {
		sums_t[t] = 0;
		for (int c=0; c<ncateg; c++)
			sums[t][c] = 0;
	}


	// now going over each sample in the node (sometimes we have to sample)
	// and adding counts for each category.

	vector<short> &feat_qvals = qf.qx[i_feat];
	for (int i=node.from_idx; i<=node.to_idx; i++) {
		int idx = indexes[i];
		int q = feat_qvals[idx];
		int c = qf.y_i[idx];
		int t = qf.last_time_slice[idx];

		counts[t][q][c]+=qf.wgts[idx];
		// in case c is 0 we need now to also add a ++ for all cells [0...(t-1)][q][0] . However, we will do those in one swip
		// at the end which should be more efficient.
		if (q > 0) sums[t][c]+=qf.wgts[idx]; // not summing q=0 cases: missing values are separate
	}

	// now we consider the cases of fewer qpoints 

	// get qpoints
	get_q_test_points(feat_maxq, params, qpoints);

	counts_q = feat_maxq;

	if (qpoints.size() != 0) {

		// checking fewer q points - given orderes in qpoints
		// we squeeze all q values to qpoints.size() values
		int prev = 0;
		for (int i=0; i<qpoints.size(); i++) {
			if (prev > i) {
				for (int t=0; t<nslices; t++)
					for (int c=0; c<ncateg; c++)
						counts[t][i][c] = counts[t][prev][c];
			}

			for (int j=prev+1; j<qpoints[i]; j++) {
				for (int t=0; t<nslices; t++)
					for (int c=0; c<ncateg; c++)
						counts[t][i][c] += counts[t][j][c];
				prev = qpoints[i];
			}
		}
		counts_q = (int)qpoints.size() + 1; // +1 for q=0
	}

	// summing all 0 counts in one swip
	for (int t=nslices-2; t>=0; t--) {
		int to_categ = (params.censor_cases) ? 1 : ncateg;
		for (int c=0; c<to_categ; c++)
			sums[t][0] += sums[t+1][c];
		for (int q=1; q<counts_q; q++)
			for (int c=0; c<to_categ; c++)
				counts[t][q][0] += counts[t+1][q][c];
	}

	total_sum = 0;
	for (int t=0; t<nslices; t++) {
		for (int c=0; c<ncateg; c++)
			sums_t[t] += sums[t][c];
		total_sum += sums_t[t];
	}


	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
void TQRF_Split_Weighted_Categorial::print_histograms()
{
	MLOG("nslices %d ncateg %d maxq %d counts_q %d\n", nslices, ncateg, maxq, counts_q);

	for (int q=0; q<counts_q; q++) {
		MLOG("counts q[%d] :", q);
		for (int c=0; c<ncateg; c++)
			for (int t=0; t<nslices; t++)
				MLOG(" c[%1d] t[%1d] %f :", c, t, counts[t][q][c]);
		MLOG("\n");
	}


	MLOG("sums :");
	for (int c=0; c<ncateg; c++)
		for (int t=0; t<nslices; t++)
			MLOG(" c[%1d] t[%1d] %f :", c, t, sums[t][c]);
	MLOG("\n");
}


//--------------------------------------------------------------------------------------------------------------------
int TQRF_Split_Entropy::get_best_split(TQRF_Params &params, int &best_q, double &best_score)
{
	// the scenario is that we have counts and sums ready and squeezed to the qpoints we want to test
	// we need to go over each of the possible split points, and if it is valid in the sense of 
	// number of samples left on each side (we always test this WITHOUT the missing vals) we get the score

	best_q = -1;			// signing no split point found
	best_score = -1;

	//vector<int> left_sums(ncateg, 0), right_sums(ncateg, 0);
	vector<vector<int>> left_sums, right_sums;

	left_sums.resize(nslices, vector<int>(ncateg, 0));
	right_sums.resize(nslices, vector<int>(ncateg, 0));
	vector<int> lsum(nslices), rsum(nslices);
	vector<double> H(nslices);

	// below is a version for a single time slice only... to begin with debugging and build code
	// will be changed to sum over slices later
	//if (nslices != 1)
	//	MTHROW_AND_ERR("ERROR: Running on debug version that supports only a single time slice !!!!!!\n");

	if (params.verbosity > 2) MLOG("TQRF_Split_Entropy::get_best_split counts_q=%d ncateg=%d\n", counts_q, ncateg);

	for (int q=1; q<counts_q-1; q++) {

		fill(lsum.begin(), lsum.end(), 0);
		fill(rsum.begin(), rsum.end(), 0);

		for (int t=0; t<nslices; t++) {
			for (int c=0; c<ncateg; c++) {
				left_sums[t][c] += counts[t][q][c];
				right_sums[t][c] = sums[t][c] - left_sums[t][c];
				lsum[t] += left_sums[t][c];
				rsum[t] += right_sums[t][c];
			}
		}

		if (params.verbosity > 2) {
			for (int t=0; t<nslices; t++) {
				MLOG("[T slice = %d] Left  q %d : lsum %d :", t, q, lsum[t]);
				for (int c=0; c<ncateg; c++)
					MLOG("c[%d] %d :", c, left_sums[t][c]);
				MLOG(" Right q %d : rsum %d :", q, rsum[t]);
				for (int c=0; c<ncateg; c++)
					MLOG("c[%d] %d :", c, right_sums[t][c]);
				MLOG("\n");
			}
		}

		if (lsum[0] >= params.min_node && rsum[0] >= params.min_node) {

			double H_tot = 0;
			fill(H.begin(), H.end(), 0);

			for (int t=0; t<nslices; t++) {

				// add left and right side entropy
				for (int c=0; c<ncateg; c++) {
					H[t] += params.log_table[left_sums[t][c]];
					H[t] += params.log_table[right_sums[t][c]];
					H[t] -= params.log_table[left_sums[t][c]+right_sums[t][c]];
				}

				// subtract overall sum entropy
				H[t] -= params.log_table[lsum[t]];
				H[t] -= params.log_table[rsum[t]];

				H[t] += params.log_table[lsum[t]+rsum[t]];

				H[t] /= (double)(lsum[t]+rsum[t])/1000.0; // actual information gain (x1000 for better resolution)

				H_tot += H[t] * (double)sums_t[t]/(double)total_sum;
			}

			if (params.verbosity > 2) {
				MLOG("      q %d : H %f best_score %f best_q %d\n", q, H_tot, best_score, best_q);
			}

			//if (best_score < 0 || (H > 0 && H < best_score)) {
			if (H_tot > best_score) {
				best_score = H_tot;
				best_q = q;
			}
		}

	}

	if (qpoints.size() > 0 && best_q>0) best_q = qpoints[best_q-1];

	return 0;
}


//--------------------------------------------------------------------------------------------------------------------
int TQRF_Split_Likelihood::get_best_split(TQRF_Params &params, int &best_q, double &best_score)
{
	// the scenario is that we have counts and sums ready and squeezed to the qpoints we want to test
	// we need to go over each of the possible split points, and if it is valid in the sense of 
	// number of samples left on each side (we always test this WITHOUT the missing vals) we get the score

	best_q = -1;			// signing no split point found
	best_score = -1e30;


	// this development version is calculating the likelihood score for getting the exact split



	//vector<int> left_sums(ncateg, 0), right_sums(ncateg, 0);
	vector<vector<int>> left_sums, right_sums;

	//nslices = 1; // DEBUG

	left_sums.resize(nslices, vector<int>(ncateg, 0));
	right_sums.resize(nslices, vector<int>(ncateg, 0));
	vector<int> lsum(nslices), rsum(nslices);
	vector<double> L(nslices);

	if (params.verbosity > 2) MLOG("TQRF_Split_Dev::get_best_split counts_q=%d ncateg=%d\n", counts_q, ncateg);

	for (int q=1; q<counts_q-1; q++) {

		fill(lsum.begin(), lsum.end(), 0);
		fill(rsum.begin(), rsum.end(), 0);

		// getting the slices counts for left and right
		for (int t=0; t<nslices; t++) {
			for (int c=0; c<ncateg; c++) {
				left_sums[t][c] += counts[t][q][c];
				right_sums[t][c] = sums[t][c] - left_sums[t][c];
				lsum[t] += left_sums[t][c];
				rsum[t] += right_sums[t][c];
			}
		}

		if (params.verbosity > 2) {
			for (int t=0; t<nslices; t++) {
				MLOG("[T slice = %d] Left  q %d : lsum %d :", t, q, lsum[t]);
				for (int c=0; c<ncateg; c++)
					MLOG("c[%d] %d :", c, left_sums[t][c]);
				MLOG("Right q %d : rsum %d :", q, rsum[t]);
				for (int c=0; c<ncateg; c++)
					MLOG("c[%d] %d :", c, right_sums[t][c]);
				MLOG("\n");
			}
		}


		if (lsum[0] >= params.min_node && rsum[0] >= params.min_node) {

			double L_tot = 0;
			fill(L.begin(), L.end(), 0);

			for (int t=0; t<nslices; t++) {

				// add left and right side entropy
				for (int c=0; c<ncateg; c++) {
					L[t] += params.log_table[sums[t][c]];
					L[t] -= params.log_table[left_sums[t][c]];
					L[t] -= params.log_table[right_sums[t][c]];
				}
				L[t] += params.log_table[lsum[t]];
				L[t] += params.log_table[rsum[t]];
				L[t] -= params.log_table[sums_t[t]];

				L_tot += L[t] * params.time_slices_wgts[t];
			}

			//L_tot = L[0];
			// prob=exp(L_tot) but we want the minimal probability so we need to use -L_tot
			L_tot = -L_tot;

			if (params.verbosity > 2) {
				MLOG("      q %d : L_tot %f best_score %f best_q %d\n", q, L_tot, best_score, best_q);
			}

			//if (best_score < 0 || (H > 0 && H < best_score)) {
			if (L_tot > best_score) {
				best_score = L_tot;
				best_q = q;
			}
		}

	}

	//if (qpoints.size() > 0 && best_q>0) best_q = qpoints[best_q-1]; // TBD - qpoints
	//nslices = params.time_slices.size(); // DEBUG

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
int TQRF_Split_Weighted_Likelihood::get_best_split(TQRF_Params &params, int &best_q, double &best_score)
{
	// the scenario is that we have counts and sums ready and squeezed to the qpoints we want to test
	// we need to go over each of the possible split points, and if it is valid in the sense of 
	// number of samples left on each side (we always test this WITHOUT the missing vals) we get the score

	best_q = -1;			// signing no split point found
	best_score = -1e30;


	// this development version is calculating the likelihood score for getting the exact split



	//vector<int> left_sums(ncateg, 0), right_sums(ncateg, 0);
	vector<vector<float>> left_sums, right_sums;

	//nslices = 1; // DEBUG

	left_sums.resize(nslices, vector<float>(ncateg, 0));
	right_sums.resize(nslices, vector<float>(ncateg, 0));
	vector<float> lsum(nslices), rsum(nslices);
	vector<double> L(nslices);

	if (params.verbosity > 2) MLOG("TQRF_Split_Dev::get_best_split counts_q=%d ncateg=%d\n", counts_q, ncateg);

	for (int q=1; q<counts_q-1; q++) {

		fill(lsum.begin(), lsum.end(), (float)0);
		fill(rsum.begin(), rsum.end(), (float)0);

		// getting the slices counts for left and right
		for (int t=0; t<nslices; t++) {
			for (int c=0; c<ncateg; c++) {
				left_sums[t][c] += counts[t][q][c];
				right_sums[t][c] = sums[t][c] - left_sums[t][c];
				lsum[t] += left_sums[t][c];
				rsum[t] += right_sums[t][c];
			}
		}

		if (params.verbosity > 2) {
			for (int t=0; t<nslices; t++) {
				MLOG("[T slice = %d] Left  q %d : lsum %d :", t, q, lsum[t]);
				for (int c=0; c<ncateg; c++)
					MLOG("c[%d] %d :", c, left_sums[t][c]);
				MLOG("Right q %d : rsum %d :", q, rsum[t]);
				for (int c=0; c<ncateg; c++)
					MLOG("c[%d] %d :", c, right_sums[t][c]);
				MLOG("\n");
			}
		}


		if (lsum[0] >= (float)params.min_node && rsum[0] >= (float)params.min_node) {

			double L_tot = 0;
			fill(L.begin(), L.end(), 0);

			for (int t=0; t<nslices; t++) {

				// add left and right side entropy
				for (int c=0; c<ncateg; c++) {
					L[t] += params.log_table[(int)sums[t][c]];
					L[t] -= params.log_table[(int)left_sums[t][c]];
					L[t] -= params.log_table[(int)right_sums[t][c]];
				}
				L[t] += params.log_table[(int)lsum[t]];
				L[t] += params.log_table[(int)rsum[t]];
				L[t] -= params.log_table[(int)sums_t[t]];

				L_tot += L[t] * params.time_slices_wgts[t];
			}

			//L_tot = L[0];
			// prob=exp(L_tot) but we want the minimal probability so we need to use -L_tot
			L_tot = -L_tot;

			if (params.verbosity > 2) {
				MLOG("      q %d : L_tot %f best_score %f best_q %d\n", q, L_tot, best_score, best_q);
			}

			//if (best_score < 0 || (H > 0 && H < best_score)) {
			if (L_tot > best_score) {
				best_score = L_tot;
				best_q = q;
			}
		}

	}

	//if (qpoints.size() > 0 && best_q>0) best_q = qpoints[best_q-1]; // TBD - qpoints
	//nslices = params.time_slices.size(); // DEBUG

	return 0;
}

//================================================================================================
// TQRF_Tree
//================================================================================================
//--------------------------------------------------------------------------------------------------------------------
// major stage in algorithm:
// we finished the work on deciding if and how to split our node, and need to actually do it.
// list of issues handled in this stage:
// (1) close work on our node, wheather needed a split or not.
// (2) add info from the splitting tqs into our node (distributions etc)
// (3) split node if needed, create the new nodes.
// (4) update indexes as needed
// (5) decide what to do with missing values !!
int TQRF_Tree::node_splitter(int i_curr_node, int i_best, int q_best)
{
	// 
	// finish the work on current node
	//

	TQRF_Node *cnode = &nodes[i_curr_node];

	if (_params->verbosity > 1) MLOG("TQRF: node_splitter : Tree %d : node %d / %d : %d - %d : i_best %d : q_best %d : feat %s\n", 
		id, i_curr_node, nodes.size(), cnode->from_idx, cnode->to_idx, i_best, q_best, (i_best<0)? "NONE" :_qfeat->feature_names[i_best].c_str());
	if (i_best >= 0) {

		// We found a point to split the node
		TQRF_Node Left, Right;

		cnode->i_feat = i_best;
		cnode->bound = _qfeat->q_to_val[i_best][q_best];

		// need to calc node sizes, and general average in order to decide for missing value strategy
		int n_missing = 0, n_left = 0, n_right = 0;
		float sum_vals = 0;

		bool do_random_split = (rand_1() < _params->random_split_prob);
		if (do_random_split) cnode->i_feat = -1;

		for (int i=cnode->from_idx; i<=cnode->to_idx; i++) {
			int idx = indexes[i];
			int q = _qfeat->qx[i_best][idx];
			if (q > 0) {
				if (q <= q_best)
					n_left++;
				else
					n_right++;
				sum_vals += _qfeat->q_to_val[i_best][q];
			}
			else
				n_missing++;
		}
		if (_params->verbosity > 1) MLOG("TQRF: node_splitter : Tree %d : node %d : n_missing %d n_left %d n_right %d\n",
											id, i_curr_node, n_missing, n_left, n_right);

		// decide missing val strategy
		if (do_random_split) cnode->missing_direction = TQRF_MISSING_DIRECTION_RAND_EACH_SAMPLE;
		else if (_params->missing_method == TQRF_MISSING_VALUE_LEFT) cnode->missing_direction = TQRF_MISSING_DIRECTION_LEFT;
		else if (_params->missing_method == TQRF_MISSING_VALUE_RAND_ALL) {
			if (rand_1() < 0.5)
				cnode->missing_direction = TQRF_MISSING_DIRECTION_LEFT;
			else
				cnode->missing_direction = TQRF_MISSING_DIRECTION_RIGHT;
		}
		else if (_params->missing_method == TQRF_MISSING_VALUE_LARGER_NODE || _params->missing_method == TQRF_MISSING_VALUE_MEDIAN) {
			if (n_left >= n_right)
				cnode->missing_direction = TQRF_MISSING_DIRECTION_LEFT;
			else
				cnode->missing_direction = TQRF_MISSING_DIRECTION_RIGHT;
		}
		else if (_params->missing_method == TQRF_MISSING_VALUE_MEAN) {
			float node_avg = sum_vals / ((float)(n_right + n_left) + (float)1e-3);
			if (node_avg <= cnode->bound)
				cnode->missing_direction = TQRF_MISSING_DIRECTION_LEFT;
			else
				cnode->missing_direction = TQRF_MISSING_DIRECTION_RIGHT;
		}
		else if (_params->missing_method == TQRF_MISSING_VALUE_RAND_EACH_SAMPLE)
			cnode->missing_direction = TQRF_MISSING_DIRECTION_RAND_EACH_SAMPLE;


		if (_params->verbosity > 1) MLOG("TQRF: node_splitter : Tree %d : node %d : missing direction %d\n", id, cnode->node_idx, cnode->missing_direction);

		// making the split , first we rearange indexes
		vector<int> left_inds, right_inds;
		//left_inds.reserve(cnode.size());
		//right_inds.reserve(cnode.size());
		for (int i=cnode->from_idx; i<=cnode->to_idx; i++) {
			int idx = indexes[i];
			int q = _qfeat->qx[i_best][idx];
			if (do_random_split) {
				if (rand_1() < (float)0.5)
					left_inds.push_back(idx);
				else
					right_inds.push_back(idx);
			}
			else if (q > 0) {
				if (q <= q_best)
					left_inds.push_back(idx);
				else
					right_inds.push_back(idx);
			}
			else {
				if (cnode->missing_direction == TQRF_MISSING_DIRECTION_LEFT) left_inds.push_back(idx);
				else if (cnode->missing_direction == TQRF_MISSING_DIRECTION_RIGHT) right_inds.push_back(idx);
				else { // if (cnode.missing_direction == TQRF_MISSING_DIRECTION_RAND_EACH_SAMPLE) case ...
					if (rand_1() < (float)0.5)
						left_inds.push_back(idx);
					else
						right_inds.push_back(idx);
				}
			}
		}

		int curr_i = cnode->from_idx;
		for (auto idx : left_inds) indexes[curr_i++] = idx;
		for (auto idx : right_inds) indexes[curr_i++] = idx;
		Left.from_idx = cnode->from_idx;
		Left.to_idx = cnode->from_idx + (int)left_inds.size() - 1;
		Right.from_idx = Left.to_idx + 1;
		Right.to_idx = cnode->to_idx;

		Left.depth = cnode->depth + 1;
		Right.depth = cnode->depth + 1;

		// we may be running in threads over nodes... hence we make sure the following part is protected
#pragma omp critical
		{
			int n_nodes = (int)nodes.size();

			Left.node_idx = n_nodes;
			Right.node_idx = n_nodes+1;

			cnode->left_node = n_nodes;
			cnode->right_node = n_nodes+1;
			cnode->is_terminal = 0;

			nodes.push_back(Left);
			nodes.push_back(Right);

			cnode = &nodes[i_curr_node]; // reassigning as it may have changed in the push !!!
			if (_params->verbosity > 1) MLOG("TQRF: Tree %d Node %d ( s %d d %d ) : split: feat %d : q %d : qval %f : left %d (%d) , right %d (%d)\n", 
				id, cnode->node_idx, cnode->size(), cnode->depth, i_best, q_best, cnode->bound, Left.node_idx, Left.size(), Right.node_idx, Right.size());
		}
	}

	// we now have to finalize the work on cnode (current node) no matter if it was split or not.
	// we need to make sure it has the needed counts etc in order to be able to give predictions
	// plus we change its state
	float nsize = 0;
	if (tree_type == TQRF_TREE_ENTROPY || tree_type == TQRF_TREE_LIKELIHOOD || tree_type == TQRF_TREE_DEV) {
		nsize = prep_node_counts(i_curr_node, 0);
	}
	else if (tree_type == TQRF_TREE_WEIGHTED_LIKELIHOOD) {
		nsize = prep_node_counts(i_curr_node, 1);
	}
	else {
		MTHROW_AND_ERR("TQRF::node_splitter(): tree_type %d doesn't know how to finalize nodes yet !!... sayonara...\n", tree_type);
	}
	if (i_best < 0 || nodes[i_curr_node].depth >= _params->max_depth || nsize <= _params->min_node) {// || nodes[i_curr_node].size() <= _params->min_node) {
		nodes[i_curr_node].is_terminal = 1;
		if (_params->verbosity > 1) MLOG("TQRF: node_splitter : Tree %d : node %d : depth %d size %d : terminal %d\n", id, i_curr_node, nodes[i_curr_node].depth, nodes[i_curr_node].size(), nodes[i_curr_node].is_terminal);
	}

	nodes[i_curr_node].state = TQRF_Node_State_Done;

	if (_params->verbosity > 1) MLOG("TQRF: node_splitter : Tree %d : node %d : size %d : terminal %d : Done\n", id, i_curr_node, nodes[i_curr_node].size(), nodes[i_curr_node].is_terminal);

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
float TQRF_Tree::prep_node_counts(int i_curr_node, int use_wgts_flag)
{
	// we go over the y values for the indexes in our node and collect them for each time slice
	nodes[i_curr_node].time_categ_count.clear();
	nodes[i_curr_node].time_categ_count.resize(_qfeat->n_time_slices, vector<float>(_qfeat->ncateg, 0));
	if (use_wgts_flag) {
		for (int i=nodes[i_curr_node].from_idx; i<=nodes[i_curr_node].to_idx; i++) {
			int idx = indexes[i];
			int c = _qfeat->y_i[idx];
			int t = _qfeat->last_time_slice[idx];
			nodes[i_curr_node].time_categ_count[t][c]+=_qfeat->orig_wgts[idx];
		}

		int auto_adjust_weights = 0;

		if (auto_adjust_weights) {

			// we muliply the weights such that we have a balanced 0/1 weighting.
			vector<float> cnts(2, 0);
			for (int i=nodes[i_curr_node].from_idx; i<=nodes[i_curr_node].to_idx; i++) {
				int idx = indexes[i];
				if (_qfeat->y_i[idx])
					cnts[1] += _qfeat->wgts[idx];
				else
					cnts[0] += _qfeat->wgts[idx];
			}

			if (cnts[1] > 0 && cnts[0] > 0) {
				float s = cnts[0] + cnts[1];
				float fact0 = s/((float)2*cnts[0]);
				float fact1 = s/((float)2*cnts[1]);
				for (int i=nodes[i_curr_node].from_idx; i<=nodes[i_curr_node].to_idx; i++) {
					int idx = indexes[i];
					if (_qfeat->y_i[idx])
						_qfeat->wgts[idx] *= fact1;
					else
						_qfeat->wgts[idx] *= fact0;
				}
			}

		}

	}
	else {
		for (int i=nodes[i_curr_node].from_idx; i<=nodes[i_curr_node].to_idx; i++) {
			int idx = indexes[i];
			int c = _qfeat->y_i[idx];
			int t = _qfeat->last_time_slice[idx];
			nodes[i_curr_node].time_categ_count[t][c]++;
		}
	}


	// reverse adding 0 (=control) categs
	int to_categ = (_params->censor_cases == 1) ? 1 : _qfeat->ncateg;
	for (int t=_qfeat->n_time_slices-2; t>=0; t--) {
		for (int c=0; c<to_categ; c++)
			nodes[i_curr_node].time_categ_count[t][0] += nodes[i_curr_node].time_categ_count[t+1][c];
	}

	float sum = 0;
	for (int c=0; c<to_categ; c++) sum += nodes[i_curr_node].time_categ_count[0][c];



	if (_params->verbosity > 1) {
		vector<int> c_for_chi;
		MLOG("TQRF: Tree %d Node %d cnts[t][c] ::", id, nodes[i_curr_node].node_idx);
		for (int t=0; t<_qfeat->n_time_slices; t++) {
			float sum = 0;
			for (int c=0; c<_qfeat->ncateg; c++) sum += nodes[i_curr_node].time_categ_count[t][c];
			for (int c=0; c<_qfeat->ncateg; c++) {
				float ratio = (sum > 0) ? (float)nodes[i_curr_node].time_categ_count[t][c]/(float)sum : 0;
				MLOG(" [%d][%d] %f %4.3f :", t, c, nodes[i_curr_node].time_categ_count[t][c], ratio);
				c_for_chi.push_back((int)nodes[i_curr_node].time_categ_count[t][c]);
			}
		}
		vector<double> exp_for_chi;
		double chi_score = medial::stats::chi2_n_x_m(c_for_chi, _qfeat->ncateg, _qfeat->n_time_slices, exp_for_chi);
		MLOG(" chi_score %6.3f\n", chi_score);
	}

	return sum;
}

//--------------------------------------------------------------------------------------------------------------------
int TQRF_Tree::get_bagged_indexes()
{
	if (_params->verbosity > 0) MLOG("Tree %d %s : bagging : params: bag_prob %f bag_ratio %f single_per_pid %d bag_with_repeats %d\n", id, _params->tree_type.c_str(), _params->bag_prob, _params->bag_ratio, _params->single_sample_per_pid, _params->bag_with_repeats);

	if (tree_type != TQRF_TREE_REGRESSION) {

		// the general idea is to calculate how many 1's and 0's we need to choose for time slice 0 (marked n0, n1)
		// we will then choose for each other time slice the minimum between n0 and the bag_prob * number of pids/samples in the slice.

		int _n_categs = 2;
		// calculate the bagging probabilities for 0 and 1
		vector<vector<int>> time_categ_cnts(_params->n_time_slices, vector<int>(_n_categs,0));
		vector<vector<float>> time_categ_probs(_params->n_time_slices, vector<float>(_n_categs,0));
		
		if (_params->bag_ratio < 0) {
			for (int t=0; t<_params->n_time_slices; t++) {
				time_categ_probs[t][0] = _params->bag_prob;
				time_categ_probs[t][1] = _params->bag_prob;
			}
		}
		else {

			// solving for time 0 : calculating n0 , n1
			int n0 = 0, n1 = 0;
			float p0, p1;
			if (_params->single_sample_per_pid) {
				n0 = (int)_qfeat->categ_pids[0].size();
				n1 = (int)_qfeat->time_categ_pids[0][1].size();
			}
			else {
				n0 = (int)_qfeat->categ_idx[0].size();
				n1 = (int)_qfeat->time_categ_idx[0][1].size();
			}

			//
			// following calculations use the fact that we want : bag_ratio = (n0*p0)/(n1*p1)
			//
			if (n0 > n1) {
				p1 = _params->bag_prob;
				p0 = _params->bag_ratio * (float)(n1+1) * p1 / (float)(n0+1);
			}
			else {
				p0 = _params->bag_prob;
				p1 = p0 * (float)(n0+1) / (_params->bag_ratio * (float)(n1+1));
			}

			p0 = min(p0, (float)1);
			p1 = min(p1, (float)1);

			time_categ_probs[0][0] = p0; // will be used for controls
			time_categ_probs[0][1] = p1; // will be used for cases in time slice 0

			n1 = (int)(p1*(float)n1);
			// now handling the c=1 cases for all time slices greater than 0
			float p_t;
			int n_t;
			for (int t=1; t<_params->n_time_slices; t++) {
				p_t = 0;
				n_t = (_params->single_sample_per_pid) ? (int)(_qfeat->time_categ_pids[t][1].size()) : (int)(_qfeat->time_categ_idx[t][1].size());
				if (n_t > 0)
					p_t = (float)n1/(float)n_t;
				time_categ_probs[t][1] = min(_params->bag_prob, p_t);
			}
		}

		indexes.clear();

		// choosing the controls (c=0) -> always using t=0 for those
		bag_chooser(time_categ_probs[0][0], 0, 0, /* OUT APPEND */ indexes);
		int nc0 = (int)indexes.size();
		if (_params->verbosity > 0) MLOG("Tree %d : bagging : 0 : %d chosen : first %d last %d : p %f\n", id, nc0, indexes[0], indexes[nc0-1], time_categ_probs[0][0]);

		// choosing the cases (c=1) for all the different time slices
		int nc_bef = nc0;
		for (int t=0; t<_params->n_time_slices; t++) {
			bag_chooser(time_categ_probs[t][1], t, 1, indexes);
			int nc1 = (int)indexes.size() - nc_bef;
			float actual_ratio = (float)nc0/(nc1+1);
			if (_params->verbosity > 0) MLOG("Tree %d : bagging : 1 : t=%d : %d chosen : first %d last %d : p %f : ratio %f\n", id, t, nc1, indexes[nc0], indexes[nc1+nc0-1], time_categ_probs[t][1], actual_ratio);
			nc_bef = (int)indexes.size();
		}

	}
	else {
		// in regression all samples are with j==1
		bag_chooser(_params->bag_prob, 0, 1, indexes);
		if (_params->verbosity > 0) MLOG("Tree %d :: bagging : regression : indexes %d\n", id, indexes.size());
	}

	// sanity print
	if (_params->verbosity > 0) {
		vector<int> cnts ={ 0,0 };
		vector<int> tcnts ={ 0,0 };
		for (auto idx : indexes) {
			if (_qfeat->y[idx] == 0)
				cnts[0]++;
			else
				cnts[1]++;
			if (_qfeat->last_time_slice[idx] == 0)
				tcnts[0]++;
			else
				tcnts[1]++;
		}
		MLOG("Tree %d :: bagging : counting by Y values: 0: %d 1: %d :: times 0: %d >0: %d\n", id, cnts[0], cnts[1], tcnts[0], tcnts[1]);
	}

	// now choosing the features to be used in this tree (in case feature bagging is requested)
	i_feats.clear();
	for (int i=0; i<_qfeat->nfeat; i++)
		if (rand_1() <= _params->bag_feat)
			i_feats.push_back(i);
	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
int TQRF_Tree::bag_chooser(float p, int _t , int _c, /* OUT APPEND */ vector<int> &_indexes)
{
	int	n_pids = (int)_qfeat->time_categ_pids[_t][_c].size();
	int	n_idx = (int)_qfeat->time_categ_idx[_t][_c].size();

	if (_params->single_sample_per_pid) {

		if (_params->bag_with_repeats) {
			unordered_map<int, int> pid_idx;
			int n_choose = (int)(p*(float)n_pids);
			for (int i=0; i<n_choose; i++) {
				int rand_pid = _qfeat->time_categ_pids[_t][_c][rand_N(n_pids)];
				if (pid_idx.find(rand_pid) == pid_idx.end()) {
					int len_pid = (int)_qfeat->pid2time_categ_idx[rand_pid][_t][_c].size();
					int rand_idx = rand_N(len_pid);
					pid_idx[rand_pid] = _qfeat->pid2time_categ_idx[rand_pid][_t][_c][rand_idx];
				}
				_indexes.push_back(pid_idx[rand_pid]); // making sure to choose the SAME sample per id
			}
		}
		else {
			for (int i=0; i<n_pids; i++)
				if (rand_1() < p) {
					int pid = _qfeat->time_categ_pids[_t][_c][i];
					int len_pid = (int)_qfeat->pid2time_categ_idx[pid][_t][_c].size();
					int rand_idx = rand_N(len_pid);
					_indexes.push_back(_qfeat->pid2time_categ_idx[pid][_t][_c][rand_idx]);
				}
		}

	}
	else {
		if (_params->bag_with_repeats) {

			int n_choose = (int)(p*(float)n_idx);
			for (int i=0; i<n_choose; i++)
				_indexes.push_back(_qfeat->time_categ_idx[_t][_c][rand_N(n_idx)]);

		}
		else {
			for (int i=0; i<n_idx; i++)
				if (rand_1() < p)
					_indexes.push_back(_qfeat->time_categ_idx[_t][_c][i]);
		}
	}

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
// simple initiations that needs to be done for the first node into the tree, and in general just right before
// we start the learning process.
//--------------------------------------------------------------------------------------------------------------------
int TQRF_Tree::init_root_node()
{
	nodes.clear();
	TQRF_Node root;

	root.from_idx = 0;
	root.to_idx = (int)indexes.size()-1;

	root.state = TQRF_Node_State_Initiated;
	root.depth = 0;

	root.node_idx = 0;
	nodes.push_back(root);

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
// returns the next node with state TQRF_Node_State_Initiated
int TQRF_Tree::get_next_node(int curr_node)
{
	for (int i=max(curr_node+1, 0); i<nodes.size(); i++)
		if (nodes[i].state == TQRF_Node_State_Initiated) {
			nodes[i].state = TQRF_Node_State_In_Progress;
			return i;
		}
	return -1; // none found
}

//--------------------------------------------------------------------------------------------------------------------
// getting a list of features to test, based on tree parameters
int TQRF_Tree::get_feats_to_test(vector<int> &feats_to_test)
{
	// first we need to know how many features we need to choose
	int n_to_choose = max(_params->ntry, (int)(_params->ntry_prob * (float)i_feats.size()));

	if (n_to_choose > (int)i_feats.size())
		n_to_choose = (int)i_feats.size();

	// we now go through n_to_choose steps of random swapping on i_feats
	feats_to_test.clear();

	if (_params->verbosity > 1) MLOG("TQRF: get_feats_to_test: n_to_choose %d/%d\n", n_to_choose, i_feats.size());
	for (int i=0; i<n_to_choose; i++) {
		int n = (int)i_feats.size() - i - 1;
		int j = rand_N(n) + i;
		int f = i_feats[i];
		i_feats[i] = i_feats[j];
		i_feats[j] = f;
		feats_to_test.push_back(i_feats[i]);
	}
	if (_params->verbosity > 1) {
		MLOG("TQRF: Tree %d : chose %d features to split by : ", id, feats_to_test.size());
		for (auto f : feats_to_test) MLOG(" %d", f);
		MLOG("\n");
	}

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------
void TQRF_Tree::free_split_stats(vector<TQRF_Split_Stat *> &tqs)
{
	for (auto &t : tqs) {
		delete t;
	}

	tqs.clear();
}

//--------------------------------------------------------------------------------------------------------------------
int TQRF_Tree::init_split_stats(vector<TQRF_Split_Stat *> &tqs)
{
	free_split_stats(tqs);
	tqs.resize(i_feats.size());

	for (int i=0; i<tqs.size(); i++) {
		tqs[i] = TQRF_Split_Stat::make_tqrf_split_stat(tree_type);
		if (tqs[i] == NULL)
			return -1;
		tqs[i]->init((*_qfeat), (*_params));
	}


	return 0;
}




//--------------------------------------------------------------------------------------------------------------------
// non threaded version
// threading a specific tree is much more complex.... we have to do it over nodes.
//--------------------------------------------------------------------------------------------------------------------
int TQRF_Tree::Train()
{
	if (_params->verbosity > 1) MLOG("Tree %d  Train() before bagging\n", id);

	// creating the bag
	get_bagged_indexes();

	if (_params->verbosity > 1) MLOG("After get_bagged_indexes()\n");
	// initializing tree and root
	init_root_node();
	if (_params->verbosity > 1) MLOG("After init_root_node()\n");

	//bool go_on = true;
	int i_curr_node = -1;

	vector<int> feats_to_test;
	vector<TQRF_Split_Stat *> tqs;

	init_split_stats(tqs);
	if (_params->verbosity > 1) MLOG("After init_split_stats()\n");

	vector<pair<int, double>> best_q;

	while (1) {

		if ((i_curr_node = get_next_node(i_curr_node)) < 0)
			break; // finished work on this tree - no more nodes to work on.

		if (_params->verbosity > 1) MLOG("TQRF: Tree %d Working on =================>>>>> node %d\n", id, i_curr_node);


		// getting a list of features to test, based on tree parameters
		get_feats_to_test(feats_to_test);

		if (feats_to_test.size() > 0)
			best_q.resize(feats_to_test.size()); //, { -1,-1 });

												 
		// optional "easy" threading right here !
		for (int i=0; i<feats_to_test.size(); i++) {
			best_q[i] ={ -1, -1.0 };
			int i_f = feats_to_test[i];
			if (_params->verbosity > 2) MLOG("TQRF: Tree %d node %d feat[%d] = %d : %s before histogram\n", id, i_curr_node, i, i_f, _qfeat->feature_names[i_f].c_str());

			tqs[i]->prep_histograms(i_f, nodes[i_curr_node], indexes, (*_qfeat), (*_params));
			if (_params->verbosity > 2) MLOG("TQRF: Tree %d node %d feat[%d] = %d : %s after histogram\n", id, i_curr_node, i, i_f, _qfeat->feature_names[i_f].c_str());
			if (_params->verbosity > 2) tqs[i]->print_histograms();
			
			if (nodes[i_curr_node].depth <= _params->max_depth)
				tqs[i]->get_best_split((*_params), best_q[i].first, best_q[i].second);
			if (_params->verbosity > 2) MLOG("TQRF: Tree %d node %d feat[%d] = %d : after get_best_split %s : %d %f : cut off val %f\n", id, i_curr_node, i, i_f, _qfeat->feature_names[i_f].c_str(), best_q[i].first, best_q[i].second, (best_q[i].first < 0) ? -1 : _qfeat->q_to_val[i_f][best_q[i].first]);
		}

		// choose best choice : scores are ALWAYS for maximum
		int i_best = -1;
		int q_best = -1;
		double q_best_score = -1e10;

		for (int i=0; i<feats_to_test.size(); i++) {
			if (_params->verbosity > 2) MLOG("TQRF: after features scan : %d : feat %d %s : q %d score %f\n", i, feats_to_test[i], _qfeat->feature_names[feats_to_test[i]].c_str(), best_q[i].first, best_q[i].second);
			if (best_q[i].first > 0 && best_q[i].second > q_best_score) {
				q_best_score = best_q[i].second;
				q_best = best_q[i].first;
				i_best = feats_to_test[i];
			}
		}
		if (_params->verbosity > 1) MLOG("TQRF: Tree %d Node %d : best feature %d %s , q %d qval %f score %f\n", id, nodes[i_curr_node].node_idx, i_best, (i_best < 0) ? "" : _qfeat->feature_names[i_best].c_str(), q_best, (i_best < 0) ? 0 : _qfeat->q_to_val[i_best][q_best], q_best_score);

		node_splitter(i_curr_node, i_best, q_best);
	}

	// need to free split stats
	free_split_stats(tqs);

	return 0;
}

const TQRF_Node *TQRF_Tree::Get_Node_for_predict(MedMat<float> &x, int i_row, float missing_val, int &beta_idx) const {
	int curr_node = 0;
	const TQRF_Node *cnode;

	float *row = x.data_ptr(i_row, 0);
	cnode = &nodes[curr_node];
	//	while (cnode->is_terminal == 0) {
	float v;
	beta_idx = cnode->beta_idx;
	while (1) {

		if (cnode->beta_idx >= 0) beta_idx = cnode->beta_idx;
		if (cnode->is_terminal)
			break;

		if (cnode->i_feat >= 0)
			v = row[cnode->i_feat];
		else
			v = missing_val;

		if (v == missing_val) {
			// applying missing val strategy:
			if (cnode->missing_direction == TQRF_MISSING_DIRECTION_LEFT) curr_node = cnode->left_node;
			else if (cnode->missing_direction == TQRF_MISSING_DIRECTION_RIGHT) curr_node = cnode->right_node;
			else if (cnode->missing_direction == TQRF_MISSING_DIRECTION_RAND_EACH_SAMPLE) {
				if (rand_1() <= 0.5)
					curr_node = cnode->left_node;
				else
					curr_node = cnode->right_node;
			}

		}
		else {
			if (v <= cnode->bound)
				curr_node = cnode->left_node;
			else
				curr_node = cnode->right_node;
		}

		cnode = &nodes[curr_node];

	}

	return cnode;
}

//--------------------------------------------------------------------------------------------------------------------
// get a feature matrix and does the travel on the tree to the terminal node for a specific row
//--------------------------------------------------------------------------------------------------------------------
TQRF_Node *TQRF_Tree::Get_Node(MedMat<float> &x, int i_row, float missing_val)
{
	int curr_node = 0;
	TQRF_Node *cnode;

	float *row = x.data_ptr(i_row, 0);
	cnode = &nodes[curr_node];
	//	while (cnode->is_terminal == 0) {
	float v;
	int beta_idx = cnode->beta_idx;
	while (1) {

		if (cnode->beta_idx >= 0) beta_idx = cnode->beta_idx;
		if (cnode->is_terminal)
			break;

		if (cnode->i_feat >= 0)
			v = row[cnode->i_feat];
		else
			v = missing_val;

		if (v == missing_val) {
			// applying missing val strategy:
			if (cnode->missing_direction == TQRF_MISSING_DIRECTION_LEFT) curr_node = cnode->left_node;
			else if (cnode->missing_direction == TQRF_MISSING_DIRECTION_RIGHT) curr_node = cnode->right_node;
			else if (cnode->missing_direction == TQRF_MISSING_DIRECTION_RAND_EACH_SAMPLE) {
				if (rand_1() <= 0.5)
					curr_node = cnode->left_node;
				else
					curr_node = cnode->right_node;
			}

		}
		else {
			if (v <= cnode->bound)
				curr_node = cnode->left_node;
			else
				curr_node = cnode->right_node;
		}

		cnode = &nodes[curr_node];

	}

	cnode->beta_idx = beta_idx;
	return cnode;
}

//===============================================================================================================
// Tuning betas
//===============================================================================================================
int TQRF_Forest::tune_betas(Quantized_Feat &qfeat)
{
	// general plan:
	// (1) prepare a list of nodes that will have a beta we will optimize.
	// (2) prepare a map from end nodes to beta number.
	// (3) optional: choose a subset to optimize on. (TBD)
	// (4) build a matrix for beta optimization.
	// (5) solve betas using gd.
	// (6) write back betas into correct position in trees.

	MLOG("TQRF:: Tuning betas on %d samples with params: min_size %d , max_depth %d\n", qfeat.lists[1].size(), params.tune_min_node_size, params.tune_max_depth);

	//
	// (1) prepare a list of nodes that will have a beta we will optimize.
	//
	vector<TreeNodeIdx> betas_locs;
	for (int i=0; i<trees.size(); i++) {
		//MLOG("tree %d :: %d nodes\n", i, trees[i].nodes.size());
		queue<int> q;
		q.push(0);
		// search tree only to the needed amount.
		while (!q.empty()) {
			int j = q.front();
			q.pop();
			TQRF_Node *cnode = &(trees[i].nodes[j]);
			//MLOG("i=%d j=%d cnode %d\n", i, j, cnode->node_idx);
			if (cnode->is_terminal || cnode->depth >= params.tune_max_depth) betas_locs.push_back(TreeNodeIdx(i, j));
			else {
				int csize = (int)cnode->get_size();
				int lsize = (int)trees[i].nodes[cnode->left_node].get_size();
				int rsize = (int)trees[i].nodes[cnode->right_node].get_size();
				if (csize >= params.tune_min_node_size && (lsize < params.tune_min_node_size || rsize < params.tune_min_node_size)) {
					betas_locs.push_back(TreeNodeIdx(i, j));
					cnode->beta_idx = (int)betas_locs.size();
				}
				if (lsize >= params.tune_min_node_size) q.push(cnode->left_node);
				if (rsize >= params.tune_min_node_size) q.push(cnode->right_node);
			}
		}
	}

	MLOG("TQRF:: tune betas : got %d betas in %d trees\n", betas_locs.size(), trees.size());

	//
	// (2) prepare a map from terminal nodes to beta number.
	//
#pragma omp parallel for
	for (int i=0; i<betas_locs.size(); i++) {
		
		int i_t = betas_locs[i].i_tree;
		int i_n = betas_locs[i].i_node;
		// for each loc we find all terminal nodes below it and mark them
		queue<int> q;
		q.push(i_n);
		while (!q.empty()) {
			int c_n = q.front();
			q.pop();
			TQRF_Node *cnode = &trees[i_t].nodes[c_n];
			if (cnode->is_terminal) cnode->beta_idx = i;
			else {
				if (trees[i_t].nodes[cnode->left_node].beta_idx < 0) q.push(cnode->left_node);
				if (trees[i_t].nodes[cnode->right_node].beta_idx < 0) q.push(cnode->right_node);
			}
		}

	}

	MLOG("TQRF:: tune betas : placed %d betas indexes in trees\n", betas_locs.size());

	//
	// at this stage each terminal node contains the index of the beta that matches it.
	// We are now ready to start the gradient descent to solve for the betas.
	//

	// 4.1 : set the set of indexes we work with (into a new vector, allowing easy changes in the future)
	vector<int> gd_idx = qfeat.lists[1];

	shuffle(gd_idx.begin(), gd_idx.end(), globalRNG::get_engine());

	MedMat<float> x;
	qfeat.orig_medf->get_as_matrix(x, {}, gd_idx);

	// 4.2 : we prepare the Ckj and Skj matrices
	int n_betas = (int)betas_locs.size();
	int n_samples = (int)gd_idx.size();
	MedMat<float> C(n_samples, n_betas), S(n_samples, n_betas);
	C.zero();
	S.zero();

	MLOG("TQRF:: tune betas : building C,S matrices of size %d x %d\n", n_samples, n_betas);

	if (alphas.size() == 0) alphas.resize(trees.size(), 1);
	//alphas.clear();
	//alphas.resize(trees.size(), 1); // we anyway recalculate all the weightings... no need for these now..

#pragma omp parallel for
	for (int i=0; i<gd_idx.size(); i++) {
		int i_c = qfeat.y_i[gd_idx[i]];
		int t = qfeat.last_time_slice[gd_idx[i]];
		for (int i_t=0; i_t<trees.size(); i_t++) {
			TQRF_Node *cnode = trees[i_t].Get_Node(x, i, params.missing_val);
			int k = cnode->beta_idx;
			C(i, k) = alphas[i_t]*cnode->time_categ_count[t][i_c];
			for (int c=0; c<qfeat.ncateg; c++)
				S(i, k) += alphas[i_t]*cnode->time_categ_count[t][c];
		}
	}

	MLOG("TQRF:: tune betas : Finished building C,S matrices. Movinf to gd process\n");

	solve_betas_gd(C, S, betas);
	return 0;
}

//
// The following is a gd solver for the problem of finding b's that minimize the loss:
// 
// Sum(j) { -log ( Sum(k) { Bk^2 * Ckj } / Sum(k) {Bk^2 * Skj} ) } + lambda * Sum(j) {(Bk-1)^2)}
//
//
int TQRF_Forest::solve_betas_gd(MedMat<float>& C, MedMat<float>& S, vector<float> &b)
{
	int n_betas = C.ncols;
	int n_samples = C.nrows;


	int n_batches = n_samples/params.gd_batch;
	if (n_batches*params.gd_batch < n_samples)
		n_batches++;

	b.resize(n_betas, 1);

	MLOG("TQRF:: solve_betas: n_betas %d n_samples %d n_batches %d batch_size %d rate %f momentum %f lambda %f ephocs %d\n", 
		n_betas, n_samples, n_batches, params.gd_batch, params.gd_rate, params.gd_momentum, params.gd_lambda, params.gd_epochs);



	int n_print = 0;
	for (int i=0; i<n_print; i++) {
		MLOG("i %d :: ",i);
		for (int j=0; j<n_betas; j++)
			MLOG("[%d] C %f S %f : ", j, C(i, j), S(i, j));
		MLOG("\n");
	}

	Map<MatrixXf> bf(&b[0], n_betas, 1);
	Map<MatrixXf> Cf(C.data_ptr(), n_betas, n_samples);
	Map<MatrixXf> Sf(S.data_ptr(), n_betas, n_samples);
	MatrixXf probs(1, n_samples);
	MatrixXf grad(n_betas, 1);
	MatrixXf grads(n_betas, params.gd_batch);
	MatrixXf gradc(n_betas, params.gd_batch);
	MatrixXf prev_grad(n_betas, 1);
	MatrixXf b_sq(n_betas, 1);

	//MLOG("Min C mat: %f , Min S mat %f\n", Cf.minCoeff(), Sf.minCoeff());

	//double loss = (float)1e8;
	double prev_loss = (float)1e8;


	int first_time = 1;
	int niter = 0;
	MatrixXf c1;
	while (1) {
		
		// do an epoch

		for (int bn=0; bn<n_batches; bn++) {
			int from = bn*params.gd_batch;
			int len = params.gd_batch;		// len is nsamples in batch
			if (from+len > n_samples) len = n_samples - from;

			Map<MatrixXf> cf(C.data_ptr(from, 0), n_betas, len);
			Map<MatrixXf> sf(S.data_ptr(from, 0), n_betas, len);
			Map<MatrixXf> gradcf(&gradc(0, 0), n_betas, len);
			Map<MatrixXf> gradsf(&grads(0, 0), n_betas, len);
			float fact_grad = (float)1/(float)len;

			//MLOG("niter %d bn %d from %d len %d fact_grad %f\n", niter, bn, from, len, fact_grad);
			//cout << "grad: " << endl << grad << endl;
			//cout << "bf: " << endl << bf << endl;


			b_sq = bf.array()*bf.array();

			c1 = (b_sq.transpose() * cf);
			c1 = c1.array() + (float)1e-10;
			gradcf = (bf.asDiagonal() * cf) * (c1.asDiagonal().inverse());
			gradsf = (bf.asDiagonal() * sf) * ((b_sq.transpose() * sf).asDiagonal().inverse());
			grad = gradsf.rowwise().sum() - gradcf.rowwise().sum();
			grad *= fact_grad; // normalizing gradient to be independent of sample size (to "gradient per sample" units)

			if (params.gd_lambda > 0)
				grad = grad.array() + params.gd_lambda * (bf.array() - 1);

			// momentum
			if (first_time) { prev_grad = grad; first_time = 0; }

			grad *= (1 - params.gd_momentum);
			grad = grad + params.gd_momentum * prev_grad;

			// step
			bf = bf - params.gd_rate*grad;

			// normalizing
			float fnorm = (float)n_betas/b_sq.sum();
			bf = fnorm * bf;

			prev_grad = grad;
		}

		// calculate loss and stop criteria
		if (1) { // (some niter test)
			MLOG("Finished epoch %d\n", niter);
			float epsilon = (float)0.000001;
			float stop_err = (float)0.00001;
			// calculating loss
			b_sq = bf.array()*bf.array();

			probs = ((b_sq.transpose() * Cf).array()) *((b_sq.transpose() * Sf).array().inverse());
			probs = probs.array().abs() + epsilon;
			//MLOG("epoch %d : max probs %f min probs %f\n", niter, probs.maxCoeff(), probs.minCoeff());
			double loss = -(probs.array().log().sum());
			loss /= (float)n_samples;

			double diff = prev_loss - loss;
			double rel_diff = diff / loss;
			double sum_b_sq = b_sq.array().sum();
			
			MLOG("TQRF: solve_betas: epoch %d : loss %f : diff %f : rel_diff %f : sum_b_sq %f\n", niter, loss, diff, rel_diff, sum_b_sq);

			prev_loss = loss;
			//if ((params.gd_epochs > 0 && niter >= params.gd_epochs) || (niter>= params.gd_epochs && rel_diff < stop_err)) break;
			if (niter>= params.gd_epochs && rel_diff < stop_err) break;

		}

		// update rate with rate decay
		//if (params.rate_decay < 1)
		//	r = r * params.rate_decay;

		niter++;

	}
	

	// in our final touch we return the squared b's as our answer 
	bf = bf.array()*bf.array();
	bf = ((float)n_betas/bf.sum())*bf;
	cout << "final bf: " << endl << bf.transpose() << endl;

	return 0;
}

