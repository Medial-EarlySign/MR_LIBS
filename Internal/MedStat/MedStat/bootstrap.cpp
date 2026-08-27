#include "bootstrap.h"
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <fenv.h>
#include "MedMat/MedMat/MedMatConstants.h"

// next saves the need to include InfraMed.h ... 
#define GENDER_MALE	1 
#ifndef  __unix__
#pragma float_control( except, on )
#endif

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDSTAT
#define LOCAL_LEVEL	LOG_DEF_LEVEL
//#define WARN_SKIP_WP
//#define USE_MIN_THREADS

#pragma region Helper Functions
float meanArr(const vector<float> &arr) {
	/*
	Compute mean while ignoring missing values
	*/
	double res = 0;
	int cnt = 0;
	for (size_t i = 0; i < arr.size(); ++i)
		if (arr[i] != MED_MAT_MISSING_VALUE) {
			res += arr[i];
			++cnt;
		}
	if (cnt > 0)
		res /= cnt;
	else
		res = MED_MAT_MISSING_VALUE;
	return (float)res;
}
float stdArr(const vector<float> &arr, float meanVal) {
	/*
	Compute standard deviation while ignoring missing values.
	Accepts pre-computed mean as an argument
	*/
	double res = 0;
	int cnt = 0;
	for (size_t i = 0; i < arr.size(); ++i)
		if (arr[i] != MED_MAT_MISSING_VALUE) {
			res += static_cast<double>(arr[i] - meanVal) * static_cast<double>(arr[i] - meanVal);
			++cnt;
		}
	if (cnt > 0) {
		res /= cnt;
		res = sqrt(res);
	}
	else
		res = MED_MAT_MISSING_VALUE;
	return (float)res;
}
template<class T> string print_obj(T obj, string format) {
	// TODO
	char res[50];
	snprintf(res, sizeof(res), format.c_str(), obj);
	return string(res);
}
string printVec(const vector<float> &v, int from, int to) {
	/*
	Convert a vector of floats to a string
	*/
	string res;
	if (from < 0)
		from = 0;
	for (size_t i = from; i < to && i < v.size(); ++i)
	{
		res += to_string(v[i]) + ", ";
	}
	return res;
}

random_device Lazy_Iterator::rd;

void get_random_seed(int seed, size_t i, random_device &rd, mt19937 &output) {
	if (seed == 0)
		output = mt19937(rd());
	else
		output = mt19937(seed + i);
}


void Lazy_Iterator::init(
	const vector<int>   *p_pids,   // note: vector can contain multiple entries for the same patient id 
	const vector<float> *p_preds,  // predictions 
	const vector<float> *p_y,      // ground truth label
	const vector<float> *p_w,      // optional weights [can affect metrics computation]
	float p_sample_ratio, 
	int p_sample_per_pid, 
	int max_loops, 
	int seed) 
{
	/*
	TODO:

	Reduce memory consumption by avoiding allocation of full vector 
	when randomly drawing  blah blah blah
	*/

	// How many samples to take for each patients, 0 - means no sampling (take all sample for patient)
	sample_per_pid = p_sample_per_pid;
	//sample_ratio = p_sample_ratio;
	sample_ratio = 1.0; //no support for smaller size for now - need to fix Std for smaller sizes
	// there are two modes of sampling
	// true:     no sampling at patient level
	// false:    randomly draw a patient (with replacement)
	sample_all_no_sampling = false;
	pids = p_pids;
	weights = (p_w == NULL || p_w->empty()) ? NULL : p_w->data();
	y = p_y->data();
	preds = p_preds->data();
	
	// thread-realted stuff
	maxThreadCount = max_loops;
	vec_size.resize(maxThreadCount);
	vec_y.resize(maxThreadCount);
	vec_preds.resize(maxThreadCount);
	vec_weights.resize(maxThreadCount);

	vec_size.back() = (int)p_pids->size();
	vec_y.back() = y;
	vec_preds.back() = preds;

	vec_weights.back() = weights;
	num_categories = p_preds->size() / p_y->size();

	unordered_map<int, vector<int>> pid_to_inds;
	for (size_t i = 0; i < pids->size(); ++i)
		pid_to_inds[(*pids)[i]].push_back(int(i));

	pid_index_to_indexes.resize((int)pid_to_inds.size());
	ind_to_pid.resize((int)pid_index_to_indexes.size());
	min_pid_start = INT_MAX;
	int max_pid_start = 0;
	int cnt_i = 0;
	int max_samples = 0;
	for (auto it = pid_to_inds.begin(); it != pid_to_inds.end(); ++it)
	{
		ind_to_pid[cnt_i] = it->first;
		if (max_samples < it->second.size())
			max_samples = (int)it->second.size();
		pid_index_to_indexes[cnt_i].swap(it->second);
		++cnt_i;
		if (it->first < min_pid_start)
			min_pid_start = it->first;
		if (it->first > max_pid_start)
			max_pid_start = it->first;
	}
	cohort_size = int(sample_ratio * pid_index_to_indexes.size());
	//init:
	rd_gen.resize(maxThreadCount);
	for (size_t i = 0; i < maxThreadCount; ++i)
		get_random_seed(seed, i, rd, rd_gen[i]);
	rand_pids = uniform_int_distribution<>(0, (int)pid_index_to_indexes.size() - 1);
	internal_random.resize(max_samples + 1);
	for (int i = 1; i <= max_samples; ++i)
		internal_random[i] = uniform_int_distribution<>(0, i - 1);
	//MLOG_D("created %d random gens\n", (int)internal_random.size());

	current_pos.resize(maxThreadCount, 0);
	inner_pos.resize(maxThreadCount, 0);
	sel_pid_index.resize(maxThreadCount, -1);

}

Lazy_Iterator::Lazy_Iterator(const vector<int> *p_pids, const vector<float> *p_preds,
	const vector<float> *p_y, const vector<float> *p_w, float p_sample_ratio, int p_sample_per_pid, int max_loops, int seed, const vector<int> *p_preds_order)
{
	init(p_pids, p_preds, p_y, p_w, p_sample_ratio, p_sample_per_pid, max_loops, seed);
	preds_order = p_preds_order->data();
	vec_preds_order.resize(maxThreadCount);
	vec_preds_order.back() = preds_order;
}

void Lazy_Iterator::set_thread_sample_all(int thread_num) {
#pragma omp critical 
	{
		vec_size[thread_num] = (int)pids->size();
		vec_y[thread_num] = y;
		vec_preds[thread_num] = preds;
		vec_weights[thread_num] = weights;
	}
}

void Lazy_Iterator::set_static(const vector<float> *p_y, const vector<float> *p_preds, const vector<float> *p_w, const vector<int> *p_preds_order, int thread_num) {
#pragma omp critical 
	{
		vec_size[thread_num] = (int)p_y->size();
		vec_y[thread_num] = p_y->data();
		vec_preds[thread_num] = p_preds->data();
		vec_weights[thread_num] = (p_w == NULL || p_w->empty()) ? NULL : p_w->data();
		vec_preds_order[thread_num] = p_preds_order->data();


	}
}
bool Lazy_Iterator::fetch_next(int thread, float &ret_y, float &ret_pred, float &weight)
{
	const float* ret_pred_pointer = NULL;
	const int * preds_order;
	bool res = this->fetch_next(thread, ret_y, ret_pred_pointer, weight, preds_order);
	if (ret_pred_pointer != NULL)
		ret_pred = *ret_pred_pointer;
	else
		ret_pred = MED_MAT_MISSING_VALUE;
	return res;
}

bool Lazy_Iterator::fetch_next(int thread, float &ret_y, const float* &ret_pred, float &weight, const int * &ret_preds_order) {
	if (sample_per_pid > 0) {
		//choose pid:
		if (current_pos[thread] >= sample_per_pid * cohort_size)
			return false;
		int selected_pid_index = int(current_pos[thread] / sample_per_pid);
		if (!sample_all_no_sampling)
			selected_pid_index = rand_pids(rd_gen[thread]);

		vector<int> *inds = &pid_index_to_indexes[selected_pid_index];
		uniform_int_distribution<> *rnd_num = &internal_random[inds->size()];

		//If has weights - not sampling by weights. can be done by calculating cum sum array
		// of weights. randomizing real number from 0 to sum_of_all_weights and using binary search to find the index.
		int selected_index = (*inds)[(*rnd_num)(rd_gen[thread])];
		ret_y = y[selected_index];
		ret_pred = &preds[selected_index*num_categories];
		ret_preds_order = &preds_order[selected_index*num_categories];
		weight = weights == NULL ? -1 : weights[selected_index];
		++current_pos[thread];
		return true;
	}
	else { //taking all samples for pid when selected, sample_ratio is less than 1
		if (sample_all_no_sampling) {
			if (current_pos[thread] >= vec_size[thread])
				return false;
			//iterate on all!:
			ret_y = vec_y[thread][current_pos[thread]];
			ret_pred = &vec_preds[thread][current_pos[thread] * num_categories];
			ret_preds_order = &vec_preds_order[thread][current_pos[thread] * num_categories];
			weight = vec_weights[thread] == NULL ? -1 : vec_weights[thread][current_pos[thread]];
			++current_pos[thread];
			return true;
		}
		if (current_pos[thread] >= cohort_size)
			return  false;
		if (sel_pid_index[thread] < 0)
		{
			int selected_pid_index = rand_pids(rd_gen[thread]);
			sel_pid_index[thread] = selected_pid_index;
			inner_pos[thread] = 0;
		}
		vector<int> *inds = &pid_index_to_indexes[sel_pid_index[thread]];
		int final_index = (*inds)[inner_pos[thread]];
		ret_y = y[final_index];
		ret_pred = &preds[final_index * num_categories];

		ret_preds_order = &preds_order[final_index * num_categories];
		weight = weights == NULL ? -1 : weights[final_index];
		//take all inds:
		++inner_pos[thread];
		if (inner_pos[thread] >= inds->size()) {
			sel_pid_index[thread] = -1;
			++current_pos[thread]; //mark pid as done
		}
		return true;
	}
}

bool Lazy_Iterator::fetch_next_external(int thread, float &ret_y, float &ret_pred, float &weight) {
	bool ret_val = fetch_next(thread, ret_y, ret_pred, weight);
	return ret_val;
}

bool Lazy_Iterator::fetch_next_external(int thread, float &ret_y, float &ret_pred, float &weight, const int *&preds_order) {
	const float *pred_ret = &ret_pred;
	bool ret_val = fetch_next(thread, ret_y, pred_ret, weight, preds_order);
	return ret_val;
}

void Lazy_Iterator::restart_iterator(int thread) {

	if (sample_ratio < 1) {
#pragma omp critical 
		{
			current_pos[thread] = 0;
			inner_pos[thread] = 0;
			sel_pid_index[thread] = -1;
		}
	}
	else {
#pragma omp critical 
		current_pos[thread] = 0;
	}

}

Lazy_Iterator::~Lazy_Iterator() {} //do nothing. nothing to clear

template<typename T> inline int binary_search_position(const T *begin, const T *end, T val, bool reversed = false) {
	int maxSize = (int)(end - begin) + 1;
	int mid = int((maxSize - 1) / 2);
	if (maxSize <= 2) {
		if (!reversed) {
			if (val <= *begin) {
				return 0;
			}
			else if (val <= *end) {
				return 1;
			}
			else {
				return maxSize;
			}
		}
		else {
			if (val >= *begin) {
				return 0;
			}
			else if (val >= *end) {
				return 1;
			}
			else {
				return maxSize;
			}
		}
	}

	if (!reversed) {
		if (val <= begin[mid]) {
			return binary_search_position(begin, begin + mid, val, reversed);
		}
		else {
			return mid + binary_search_position(begin + mid, end, val, reversed);
		}
	}
	else {
		if (val >= begin[mid]) {
			return binary_search_position(begin, begin + mid, val, reversed);
		}
		else {
			return mid + binary_search_position(begin + mid, end, val, reversed);
		}
	}
}

template<typename T> inline int binary_search_position_last(const T *begin, const T *end, T val, bool reversed = false) {
	int maxSize = (int)(end - begin) + 1;
	int mid = int((maxSize - 1) / 2);
	if (maxSize <= 2) {
		if (!reversed) {
			if (val < *begin) {
				return 0;
			}
			else if (val < *end) {
				return 1;
			}
			else {
				return maxSize;
			}
		}
		else {
			if (val > *begin) {
				return 0;
			}
			else if (val > *end) {
				return 1;
			}
			else {
				return maxSize;
			}
		}
	}

	if (!reversed) {
		if (val < begin[mid]) {
			return binary_search_position_last(begin, begin + mid, val, reversed);
		}
		else {
			return mid + binary_search_position_last(begin + mid, end, val, reversed);
		}
	}
	else {
		if (val > begin[mid]) {
			return binary_search_position_last(begin, begin + mid, val, reversed);
		}
		else {
			return mid + binary_search_position_last(begin + mid, end, val, reversed);
		}
	}
}

Mem_Iterator::Mem_Iterator(const vector<int> &pids, const vector<int> &cohort_indexes, float p_sample_ratio, int p_sample_per_pid, int seed) {
	sample_per_pid = p_sample_per_pid;
	sample_ratio = p_sample_ratio;
	random_device rd;
	get_random_seed(seed, 0, rd, _rd_gen);
	
	if (cohort_indexes.empty())
		MTHROW_AND_ERR("Error in Mem_Iterator::Mem_Iterator - empty cohort_indexes\n");
	if (pids.size() <= cohort_indexes.front())
		MTHROW_AND_ERR("Error in Mem_Iterator::Mem_Iterator - got index in cohort_indexes[%zu]=%d which is bigger than pids %zu\n",
		(size_t)0, cohort_indexes.front(), pids.size());
	int min_pid = pids[cohort_indexes.front()], max_pid = pids[cohort_indexes.front()];
	for (size_t i = 1; i < cohort_indexes.size(); ++i)
	{
		int ii = cohort_indexes[i];
		if (ii >= pids.size())
			MTHROW_AND_ERR("Error in Mem_Iterator::Mem_Iterator - got index in cohort_indexes[%zu]=%d which is bigger than pids %zu\n",
				i, cohort_indexes[i], pids.size());
		if (pids[ii] < min_pid)
			min_pid = pids[ii];
		if (pids[ii] > max_pid)
			max_pid = pids[ii];
	}

	pid_to_inds.resize(max_pid - min_pid + 1);
	ind_to_pid.reserve(max_pid - min_pid + 1);
	for (int ii : cohort_indexes) {
		int pid_ind_val = pids[ii] - min_pid;
		if (pid_to_inds[pid_ind_val].empty())
			ind_to_pid.push_back(pid_ind_val);
		pid_to_inds[pid_ind_val].push_back(int(ii));
	}

	cohort_size = int(sample_ratio * ind_to_pid.size());
	//choose pids:


	cohort_idx = cohort_indexes;
	tot_rec_cnt = (int)pids.size();
}

void Mem_Iterator::fetch_selection(mt19937 &rd_gen, vector<int> &indexes) const {
	indexes.clear();
	if (sample_per_pid > 0)
		indexes.reserve(static_cast<size_t>(cohort_size) * static_cast<size_t>(sample_per_pid));
	else
		indexes.reserve(tot_rec_cnt);
	uniform_int_distribution<> rand_pids(0, (int)ind_to_pid.size() - 1);
	for (size_t k = 0; k < cohort_size; ++k)
	{
		int ind_pid = rand_pids(rd_gen);
		int pid_idx_sel = ind_to_pid[ind_pid];
		//if (pid_idx_sel >= pid_to_inds.size())
		//	MTHROW_AND_ERR("Error Mem_Iterator::fetch_selection - pid_idx_sel(%d) >= pid_to_inds.size(%zu)\n",
		//		pid_idx_sel, pid_to_inds.size());
		const vector<int> &ind_vec = pid_to_inds[pid_idx_sel]; //the indexes of current pid:
		//if (ind_vec.empty())
		//	MTHROW_AND_ERR("Error Mem_Iterator::fetch_selection - ind_vec is empty\n");
		//subsample if needed:
		if (sample_per_pid == 0)
			indexes.insert(indexes.end(), ind_vec.begin(), ind_vec.end());
		else {
			uniform_int_distribution<> sel_rnd(0, (int)ind_vec.size() - 1);
			//with repeats:
			for (size_t i = 0; i < sample_per_pid; ++i)
			{
				int rnd_sel = sel_rnd(rd_gen);
				indexes.push_back(ind_vec[rnd_sel]);
			}
		}
	}
}

void Mem_Iterator::fetch_selection(vector<int> &indexes) const {
	mt19937 rd_cop = _rd_gen;
	fetch_selection(rd_cop, indexes);
}

void Mem_Iterator::fetch_selection_external(vector<int> &indexes) const {
	fetch_selection(indexes);
}

void Mem_Iterator::fetch_selection_external(mt19937 &rd_gen, vector<int> &indexes) const {
	fetch_selection(rd_gen, indexes);
}

#pragma endregion

int get_checksum(const vector<int> &pids) {
	int checksum = 0;
	for (int pid : pids)
		checksum = (checksum + pid) & 0xFFFF;
	return checksum;
}

void prepare_for_bootstrap(const vector<int> &pids,
	const map<string, vector<float>> &additional_info, FilterCohortFunc &filter_cohort
	, void *cohort_params, float sample_ratio, int sample_per_pid, int seed, vector<int> &indexes) {
	vector<int>  filtered_indexes;
	filtered_indexes.clear();
	filtered_indexes.reserve((int)pids.size());

	for (size_t j = 0; j < pids.size(); ++j)
		if (filter_cohort(additional_info, (int)j, cohort_params)) {
			filtered_indexes.push_back((int)j);
		}

	Mem_Iterator iterator(pids, filtered_indexes, sample_ratio, sample_per_pid, seed);
	iterator.fetch_selection(indexes);
}

map<string, float> booststrap_analyze_cohort(const vector<float> &preds, const vector<int> &preds_order, const vector<float> &y,
	const vector<int> &pids, float sample_ratio, int sample_per_pid, int loopCnt,
	const vector<MeasurementFunctions> &meas_functions, const vector<Measurement_Params *> &function_params,
	ProcessMeasurementParamFunc process_measurments_params,
	const map<string, vector<float>> &additional_info, const vector<float> &y_full,
	const vector<int> &pids_full, const vector<float> *weights, const vector<int> &filter_indexes, FilterCohortFunc cohort_def,
	void *cohort_params, int &warn_cnt, const string &cohort_name, int seed) {
	//this function called after filter cohort
	//for each pid - randomize x sample from all it's tests. do loop_times
	float ci_bound = (float)0.95;
	int max_warns = 5;

	//initialize measurement params per cohort:
	//time_t st = time(NULL);

	for (size_t i = 0; i < function_params.size(); ++i)
		if (process_measurments_params != NULL && function_params[i] != NULL) {
			ROC_And_Filter_Params static_mem;
			ROC_And_Filter_Params* prm = &static_mem;
			ROC_Params *only_roc = dynamic_cast<ROC_Params*> (function_params[i]);
			if (only_roc == NULL)
				continue;

			prm->roc_params = only_roc;
			prm->filter = (vector<Filter_Param> *)cohort_params;
			process_measurments_params(additional_info, y, pids, prm,
				filter_indexes, y_full, pids_full);
		}
	//MLOG_D("took %2.1f sec to process_measurments_params\n", (float)difftime(time(NULL), st));

#ifdef USE_MIN_THREADS
	Lazy_Iterator iterator(&pids, &preds, &y, weights, sample_ratio, sample_per_pid, omp_get_max_threads(), seed, &preds_order); //for Obs
#else
	Lazy_Iterator iterator(&pids, &preds, &y, weights, sample_ratio, sample_per_pid, loopCnt + 1, seed, &preds_order); //for Obs
#endif
	//MLOG_D("took %2.1f sec till allocate mem\n", (float)difftime(time(NULL), st));

	Mem_Iterator mem_iter;
	if (sample_per_pid == 0) {
		vector<int> empty_all(pids.size());
		for (size_t i = 0; i < pids.size(); ++i)
			empty_all[i] = (int)i;
		mem_iter = Mem_Iterator(pids, empty_all, sample_ratio, sample_per_pid, seed);
	}
	map<string, vector<float>> all_measures;
	iterator.sample_all_no_sampling = true;
	//iterator.sample_per_pid = 0; //take all samples in Obs
	//iterator.sample_ratio = 1; //take all pids
#ifdef USE_MIN_THREADS
	int main_thread = 0;
#else
	int main_thread = loopCnt;
#endif
	for (size_t k = 0; k < meas_functions.size(); ++k)
	{
		if (k > 0)
			iterator.restart_iterator(main_thread);
		map<string, float> batch_measures = meas_functions[k](&iterator, main_thread, function_params[k]);
		for (auto jt = batch_measures.begin(); jt != batch_measures.end(); ++jt)
			all_measures[jt->first + "_Obs"].push_back(jt->second);
	}
#ifdef USE_MIN_THREADS
	iterator.restart_iterator(0);
#endif
	//If True will create in memory selection of indexes. If false will do it lazy.
	//In some cenarios it might be faster to use "lazy" or "memory" 
	bool allow_use_memory_iter = false;

	if (sample_per_pid > 0) {
		//save results for all cohort:
		iterator.sample_all_no_sampling = false;
		//iterator.sample_per_pid = sample_per_pid;
		//iterator.sample_ratio = sample_ratio;

		MedProgress done_cnt("bootstrap_progress", loopCnt, 30, 1);
		Lazy_Iterator *iter_for_omp = &iterator;
#pragma omp parallel for schedule(static)
		for (int i = 0; i < loopCnt; ++i)
		{
#ifdef USE_MIN_THREADS
			int th_num = omp_get_thread_num();
#else
			int th_num = i;
#endif
			for (size_t k = 0; k < meas_functions.size(); ++k)
			{
#ifdef USE_MIN_THREADS
				iterator.restart_iterator(th_num);
#else
				if (k > 0)
					iterator.restart_iterator(th_num);
#endif
				map<string, float> batch_measures = meas_functions[k](iter_for_omp, th_num, function_params[k]);
#pragma omp critical
				for (auto jt = batch_measures.begin(); jt != batch_measures.end(); ++jt)
					all_measures[jt->first].push_back(jt->second);
			}
			done_cnt.update();
		}
	}
	else {
		//old implementition with memory:
		iterator.sample_all_no_sampling = allow_use_memory_iter;

#ifdef USE_MIN_THREADS
		int max_rnd_gen = omp_get_thread_num();
#else
		int max_rnd_gen = loopCnt;
#endif
		vector<mt19937> rd_gen(max_rnd_gen);
		random_device rd;
		for (size_t i = 0; i < rd_gen.size(); ++i)
			get_random_seed(seed, i, rd, rd_gen[i]);

		//other sampling - sample pids and take all thier data:
		//now sample cohort 

		MedProgress done_cnt("bootstrap_progress", loopCnt, 30, 1);
#pragma omp parallel for schedule(dynamic,1)
		for (int i = 0; i < loopCnt; ++i)
		{
#ifdef USE_MIN_THREADS
			int th_num = omp_get_thread_num();
#else
			int th_num = i;
#endif
			//create preds, y for all seleceted pids:
			vector<float> selected_preds, selected_y, selected_weights;
			vector<int> selected_preds_order;
			if (allow_use_memory_iter) {
				vector<int> idx;
				mem_iter.fetch_selection(rd_gen[th_num], idx);
				selected_preds.resize(idx.size() * iterator.num_categories);
				selected_y.resize(idx.size());
				selected_preds_order.resize(idx.size() * iterator.num_categories);
				if (weights != NULL && !weights->empty())
					selected_weights.resize(idx.size());
				for (size_t k = 0; k < idx.size(); ++k)
				{
					int ind = idx[k];
					for (size_t j = 0; j < iterator.num_categories; j++)
					{
						selected_preds[k * iterator.num_categories + j] = preds[ind * iterator.num_categories + j];
						selected_preds_order[k * iterator.num_categories + j] = preds_order[ind * iterator.num_categories + j];
					}

					selected_y[k] = y[ind];
					if (weights != NULL && !weights->empty())
						selected_weights[k] = weights->at(ind);
				}

				iterator.set_static(&selected_y, &selected_preds, &selected_weights, &selected_preds_order, i);
			}

			//calc measures for sample:
			for (size_t k = 0; k < meas_functions.size(); ++k)
			{
				map<string, float> batch_measures;
#ifdef USE_MIN_THREADS
				iterator.restart_iterator(th_num);
#else
				if (k > 0)
					iterator.restart_iterator(th_num);
#endif
				batch_measures = meas_functions[k](&iterator, i, function_params[k]);
				if (batch_measures.empty()) {
					if (warn_cnt < max_warns) {
#pragma omp atomic
						++warn_cnt;
						MWARN("bootstrap warning: in cohort \"%s\" - no measurements\n", cohort_name.c_str());
						if (warn_cnt == max_warns)
							//cancel warning is parameters:
#pragma omp critical
							for (size_t kk = 0; kk < meas_functions.size(); ++kk)
								function_params[kk]->show_warns = false;
					}

				}
#pragma omp critical
				for (auto jt = batch_measures.begin(); jt != batch_measures.end(); ++jt)
					all_measures[jt->first].push_back(jt->second);
			}

			done_cnt.update();
		}
	}

	//now calc - mean, std , CI0.95_lower, CI0.95_upper for each measurement in all exp
	map<string, float> all_final_measures;
	for (auto it = all_measures.begin(); it != all_measures.end(); ++it)
	{
		vector<float> &measures = it->second;
		if (measures.empty()) {
			if (it->first.size() > 4 && it->first.substr(it->first.size() - 4) == "_Obs")
				all_final_measures[it->first] = MED_MAT_MISSING_VALUE;
			else {
				all_final_measures[it->first + "_Mean"] = MED_MAT_MISSING_VALUE;
				all_final_measures[it->first + "_Std"] = MED_MAT_MISSING_VALUE;
				all_final_measures[it->first + "_CI.Lower.95"] = MED_MAT_MISSING_VALUE;
				all_final_measures[it->first + "_CI.Upper.95"] = MED_MAT_MISSING_VALUE;
			}
			continue;
		}
		sort(measures.begin(), measures.end());
		float meanVal = meanArr(measures);
		float stdVal = stdArr(measures, meanVal);
		int last_idx = 0;
		if (measures.front() == MED_MAT_MISSING_VALUE)
			last_idx = binary_search_position_last(measures.data(), measures.data() + (int)measures.size() - 1, (float)MED_MAT_MISSING_VALUE);
		int lower_ct_idx = last_idx + (int)round(((1 - ci_bound) / 2) * (measures.size() - last_idx));
		float lower_ci = MED_MAT_MISSING_VALUE;
		if (lower_ct_idx < measures.size())
			lower_ci = measures[lower_ct_idx];

		int max_pos = last_idx + (int)round((ci_bound + (1 - ci_bound) / 2) * (measures.size() - last_idx));
		if (max_pos >= measures.size())
			max_pos = (int)measures.size() - 1;
		float upper_ci = measures[max_pos];

		if (it->first.size() > 4 && it->first.substr(it->first.size() - 4) == "_Obs")
			all_final_measures[it->first] = meanVal;
		else {
			all_final_measures[it->first + "_Mean"] = meanVal;
			all_final_measures[it->first + "_Std"] = stdVal;
			all_final_measures[it->first + "_CI.Lower.95"] = lower_ci;
			all_final_measures[it->first + "_CI.Upper.95"] = upper_ci;
		}
	}
	all_final_measures["Checksum"] = (float)get_checksum(pids);
	//MLOG_D("took %2.1f sec to cohort\n", (float)difftime(time(NULL), st));

	return all_final_measures;
}

map<string, map<string, float>> booststrap_analyze(
	const vector<float> &preds, 
	const vector<int> &preds_order, 
	const vector<float> &y, const vector<float> *weights, const vector<int> &pids,
	const map<string, vector<float>> &additional_info, const map<string, FilterCohortFunc> &filter_cohort
	, const vector<MeasurementFunctions> &meas_functions, const map<string, void *> *cohort_params,
	const vector<Measurement_Params *> *function_params, ProcessMeasurementParamFunc process_measurments_params,
	PreprocessScoresFunc preprocess_scores, Measurement_Params *preprocess_scores_params, float sample_ratio, int sample_per_pid,
	int loopCnt, int seed, bool binary_outcome) {
#if defined(__unix__)
	//feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
	
	// sanity check
	if (pids.size() != y.size()) {
		cerr << "bootstrap sizes aren't equal pids=" << pids.size() << " y=" << y.size() << endl;
		throw invalid_argument("bootstrap sizes aren't equal");
	}
	
	// allocate vector to keep parameters of function that implement different metrics 
	vector<Measurement_Params *> params((int)meas_functions.size());

	if (function_params == NULL)
		for (size_t i = 0; i < params.size(); ++i)
			params[i] = NULL;

	MLOG_D("Started Bootstarp Analysis on %d samples with %d cohorts\n", preds.size(), filter_cohort.size());
	time_t start = time(NULL);
	
	// preprocess predictions (if requested)
	const vector<float> *final_preds = &preds;
	vector<float> copy_preds;
	if (preprocess_scores != NULL) {
		copy_preds = vector<float>(preds);
		preprocess_scores(copy_preds, preprocess_scores_params);
		final_preds = &copy_preds;
	}

	map<string, map<string, float>> all_cohorts_measurments;
	vector<float> preds_c, y_c, weights_c;
	vector<int> pids_c, filtered_indexes, preds_order_c;
	vector<int> class_sz;
	pids_c.reserve((int)y.size());
	preds_c.reserve((int)preds.size());
	preds_order_c.reserve((int)preds.size());
	filtered_indexes.reserve((int)y.size());
	y_c.reserve((int)y.size());

	int warn_cnt = 0;
	if (weights != NULL && !weights->empty())
		weights_c.reserve(weights->size());
	
	// In cases of multiclass predictions, predictions are arranged like
    // class_1_pred[0], class_2_pred[0], class_1_pred[1], class_2_pred[1] ...
	size_t num_categories = preds.size() / y.size();
	
	// Loop over cohorts
	MedProgress progress_bt("Full_bootstrap", (int)filter_cohort.size(), 60, 1);
	for (auto it = filter_cohort.begin(); it != filter_cohort.end(); ++it)
	{
		void *c_params = NULL;
		// fetch cohort parameters (if exist)
		if (cohort_params != NULL && (*cohort_params).find(it->first) != (*cohort_params).end())
			c_params = (*cohort_params).at(it->first);

		//===============
		// Compose cohort
		//===============

		pids_c.clear();
		preds_c.clear();
		y_c.clear();
		preds_order_c.clear();
		weights_c.clear();
		filtered_indexes.clear();
		
		// initialize to zeros
		class_sz.assign(2, 0);
		
		// loop over label/prediction pairs
		for (size_t j = 0; j < y.size(); ++j)
			// decide whether current sample belogs to current cohort
			if (it->second(additional_info, (int)j, c_params)) {
				bool has_legal_w = true;
				
				// check whether weight of current sample is valid
				if (weights != NULL && !weights->empty()) {
					if (weights->at(j) > 0)
						weights_c.push_back(weights->at(j));
					else
						has_legal_w = false;
				}

				// apparently pids_c/y_c/preds_c contain only pids/labels/predictions of samples with _valid_ weights
				if (has_legal_w) {

					pids_c.push_back(pids[j]);
					y_c.push_back(y[j]);
					
					for (size_t k = 0; k < num_categories; k++)
					{
						preds_c.push_back((*final_preds)[j*num_categories + k]);
						preds_order_c.push_back(preds_order[j*num_categories + k]);
					}

					filtered_indexes.push_back((int)j);
					++class_sz[y[j] > 0];
				}

			}

		//===============
		// Analyse cohort
		//===============
		string cohort_name = it->first;

		// Skip cohort if too small
		if (y_c.size() < 10) {
			MWARN("WARN: Cohort [%s] is too small - has %d samples. Skipping\n", cohort_name.c_str(), int(y_c.size()));
			progress_bt.update();
			continue;
		}

		// Additional sanity checks for binary classification case 
		if (binary_outcome) {
			if ((class_sz[0] < 1 || class_sz[1] < 1)) {
				MWARN("WARN: Cohort [%s] is too small - has %d samples with labels = [%d, %d]. Skipping\n",
					cohort_name.c_str(), int(y_c.size()), class_sz[0], class_sz[1]);
				continue;
			}
			else MLOG("Cohort [%s] - has %d samples with labels = [%d, %d]\n",
				cohort_name.c_str(), int(y_c.size()), class_sz[0], class_sz[1]);
		}

		vector<float> *weights_p = NULL;
		if (!weights_c.empty())
			weights_p = &weights_c;

		map<string, float> cohort_measurments = booststrap_analyze_cohort(
			preds_c, 
			preds_order_c, 
			y_c, 
			pids_c,
			sample_ratio, 
			sample_per_pid, 
			loopCnt, 
			meas_functions,
			function_params != NULL ? *function_params : params,
			process_measurments_params, 
			additional_info, 
			y, 
			pids, 
			weights_p, 
			filtered_indexes, 
			it->second, 
			c_params, 
			warn_cnt, cohort_name, seed);

		all_cohorts_measurments[cohort_name] = cohort_measurments;

		progress_bt.update();
	}
	MLOG_D("Finished Bootstarp Analysis. took %2.1f seconds\n", difftime(time(NULL), start));
	return all_cohorts_measurments;
}

void write_bootstrap_results(const string &file_name, const map<string, map<string, float>> &all_cohorts_measurments, const string& run_id) {
	string delimeter = "\t";
	if (all_cohorts_measurments.empty())
		throw invalid_argument("all_cohorts_measurments can't be empty");
	unordered_set<string> all_columns_uniq;
	for (auto jt = all_cohorts_measurments.begin(); jt != all_cohorts_measurments.end(); ++jt)
		for (auto it = jt->second.begin(); it != jt->second.end(); ++it)
			all_columns_uniq.insert(it->first);
	vector<string> all_columns(all_columns_uniq.begin(), all_columns_uniq.end());
	sort(all_columns.begin(), all_columns.end());
	ofstream fw(file_name);
	if (!fw.good())
		MTHROW_AND_ERR("IO Error: can't write \"%s\"\n", file_name.c_str());

	fw << "Cohort_Description";
	for (size_t i = 0; i < all_columns.size(); ++i)
		fw << delimeter << all_columns[i];
	if (!run_id.empty())
		fw << delimeter << "run_id";
	fw << endl;

	for (auto it = all_cohorts_measurments.begin(); it != all_cohorts_measurments.end(); ++it)
	{
		string cohort_name = it->first;
		map<string, float> cohort_values = it->second;
		fw << cohort_name;
		for (size_t i = 0; i < all_columns.size(); ++i)
			fw << delimeter <<
			(cohort_values.find(all_columns[i]) != cohort_values.end() ? cohort_values.at(all_columns[i]) : MED_MAT_MISSING_VALUE);
		if (!run_id.empty())
			fw << delimeter << run_id;
		fw << endl;
	}

	fw.close();
}
void read_bootstrap_results(const string &file_name, map<string, map<string, float>> &all_cohorts_measurments) {
	string delimeter = "\t";
	ifstream of(file_name);
	if (!of.good())
		MTHROW_AND_ERR("IO Error: can't read \"%s\"\n", file_name.c_str());
	string line, header;
	getline(of, header); //read header
	vector<string> column_names;
	boost::split(column_names, header, boost::is_any_of(delimeter));
	int cohort_name_ind = (int)distance(column_names.begin(), find(column_names.begin(), column_names.end(), "Cohort_Description"));
	if (cohort_name_ind > column_names.size())
		MTHROW_AND_ERR("Couldn't find \"Cohort_Description\" in bootstrap header\n");

	while (getline(of, line)) {
		vector<string> tokens;
		boost::split(tokens, line, boost::is_any_of(delimeter));
		if (tokens.size() != column_names.size())
			MTHROW_AND_ERR("Bad bootstrap format! header has %d columns. got line with %d fields. line=\"%s\"\n",
			(int)column_names.size(), (int)tokens.size(), line.c_str());
		string name = tokens[cohort_name_ind];
		map<string, float> cohort_values;
		for (size_t i = 0; i < tokens.size(); ++i)
		{
			if (i == cohort_name_ind)
				continue;
			cohort_values[column_names[i]] = stof(tokens[i]);
		}
		all_cohorts_measurments[name] = cohort_values;
	}
	of.close();
}

void write_pivot_bootstrap_results(const string &file_name, const map<string, map<string, float>> &all_cohorts_measurments, const string& run_id) {
	string delimeter = "\t";
	if (all_cohorts_measurments.empty())
		throw invalid_argument("all_cohorts_measurments can't be empty");
	map<string, float> flat_map;
	for (auto jt = all_cohorts_measurments.begin(); jt != all_cohorts_measurments.end(); ++jt) {
		char buff[1000];
		for (auto it = jt->second.begin(); it != jt->second.end(); ++it) {
			snprintf(buff, sizeof(buff), "%s%s%s", jt->first.c_str(), delimeter.c_str(), it->first.c_str());
			flat_map[string(buff)] = it->second;
		}
	}

	ofstream fw(file_name);
	if (!fw.good())
		MTHROW_AND_ERR("IO Error: can't write \"%s\"\n", file_name.c_str());

	fw << "Cohort" << delimeter << "Measurement" << delimeter << "Value" << endl;
	for (auto it = flat_map.begin(); it != flat_map.end(); ++it)
	{
		string cohort_measure_name = it->first;
		float value = it->second;
		fw << cohort_measure_name << delimeter << value << "\n";
	}
	if (!run_id.empty())
		for (auto jt = all_cohorts_measurments.begin(); jt != all_cohorts_measurments.end(); ++jt)
			fw << jt->first << delimeter << "run_id" << delimeter << run_id << "\n";

	fw.flush();
	fw.close();
}
void read_pivot_bootstrap_results(const string &file_name, map<string, map<string, float>> &all_cohorts_measurments) {
	string delimeter = "\t";
	map<string, float> flat_map;

	ifstream fr(file_name);
	if (!fr.good())
		MTHROW_AND_ERR("IO Error: can't read \"%s\"\n", file_name.c_str());
	string line;
	getline(fr, line); //skip header
	while (getline(fr, line)) {
		mes_trim(line);
		if (line.empty())
			continue;
		vector<string> tokens;
		boost::split(tokens, line, boost::is_any_of(delimeter));
		if (tokens.size() != 3)
			MTHROW_AND_ERR("format error in line \"%s\"\n", line.c_str());
		string &cohort_name = tokens[0];
		string &measure_name = tokens[1];
		if (measure_name == "run_id")
			continue;
		float value = stof(tokens[2]);
		all_cohorts_measurments[cohort_name][measure_name] = value;
	}

	fr.close();
}

#pragma region Measurements Functions

map<string, float> calc_npos_nneg(Lazy_Iterator *iterator, int thread_num, Measurement_Params *function_params) {
	map<string, float> res;

	map<float, int> cnts;
	float y, w, pred;
	while (iterator->fetch_next(thread_num, y, pred, w))
		cnts[y] += w != -1 ? w : 1;

	res["NPOS"] = (float)cnts[(float)1.0];
	res["NNEG"] = (float)cnts[(float)0];

	return res;
}

float get_auc(float tot_cnt, float tot_true_labels, unordered_map<float, vector<float>>& pred_to_labels, unordered_map<float, vector<float>>& pred_to_weights) {

	double tot_false_labels = tot_cnt - tot_true_labels;
	if (tot_true_labels == 0 || tot_false_labels == 0)
		MTHROW_AND_ERR("Error in bootstrap::get_auc - only falses or positives exists in cohort");

	vector<float >pred_threshold = vector<float>((int)pred_to_labels.size());
	unordered_map<float, vector<float>>::iterator it = pred_to_labels.begin();
	for (size_t i = 0; i < pred_threshold.size(); ++i)
	{
		pred_threshold[i] = it->first;
		++it;
	}
	sort(pred_threshold.begin(), pred_threshold.end());
	//From up to down sort:
	double t_cnt = 0;
	double f_cnt = 0;
	vector<float> true_rate = vector<float>((int)pred_to_labels.size());
	vector<float> false_rate = vector<float>((int)pred_to_labels.size());
	int st_size = (int)pred_threshold.size() - 1;
	if (pred_to_weights.empty())
		for (int i = st_size; i >= 0; --i)
		{
			vector<float> *y_vals = &pred_to_labels[pred_threshold[i]];
			//calc AUC status for this step:
			for (float y : *y_vals)
			{
				bool true_label = y > 0;
				t_cnt += int(true_label);
				f_cnt += int(!true_label);
			}
			true_rate[st_size - i] = float(t_cnt / tot_true_labels);
			false_rate[st_size - i] = float(f_cnt / tot_false_labels);
		}
	else
		for (int i = st_size; i >= 0; --i)
		{
			vector<float> *y_vals = &pred_to_labels[pred_threshold[i]];
			vector<float> *w_vals = &pred_to_weights[pred_threshold[i]];
			//calc AUC status for this step:
			for (int y_ind = 0; y_ind < y_vals->size(); ++y_ind)
			{
				bool true_label = (*y_vals)[y_ind] > 0;
				t_cnt += int(true_label) * (*w_vals)[y_ind];
				f_cnt += int(!true_label)* (*w_vals)[y_ind];
			}
			true_rate[st_size - i] = float(t_cnt / tot_true_labels);
			false_rate[st_size - i] = float(f_cnt / tot_false_labels);
		}

	float auc = false_rate[0] * true_rate[0] / 2;
	for (size_t i = 1; i < true_rate.size(); ++i)
		auc += (false_rate[i] - false_rate[i - 1]) * (true_rate[i - 1] + true_rate[i]) / 2;

	return auc;
}

map<string, float> calc_only_auc(Lazy_Iterator *iterator, int thread_num, Measurement_Params *function_params) {
	map<string, float> res;

	unordered_map<float, vector<float>> pred_to_labels;
	unordered_map<float, vector<float>> pred_to_weights;
	double tot_true_labels = 0;
	float y, pred, weight;
	double tot_cnt = 0;
	while (iterator->fetch_next(thread_num, y, pred, weight)) {
		pred_to_labels[pred].push_back(y);
		if (weight != -1)
			pred_to_weights[pred].push_back(weight);
		else
			weight = 1;
		tot_true_labels += int(y > 0) * weight;
		tot_cnt += weight;
	}

	res["AUC"] = get_auc(tot_cnt, tot_true_labels, pred_to_labels, pred_to_weights);
	return res;
}

map<string, float> calc_multi_class(Lazy_Iterator *iterator, int thread_num, Measurement_Params *function_params) {

	map<string, float> res;
	float y, weight, w;
	const float *pred;
	const int *preds_order;

	Multiclass_Params* params = (Multiclass_Params *)function_params;
	int n_top_n = (int)params->top_n.size();

	// Sanity
	if (iterator->num_categories != params->n_categ)
		MTHROW_AND_ERR("n_categ and predictions inconsistency\n");

	vector<float> dist_vals(iterator->num_categories);
	float avg_dist_until_correct = 0, avg_weighted_dist_until_correct = 0, avg_weighted_preds_dist_until_correct = 0;
	float sum_weighted_preds_dist = 0, sum_correct_location = 0;
	vector<float> avg_weighted_dist_top_n(n_top_n, 0), avg_dist_top_n(n_top_n, 0), avg_weighted_preds_dist_top_n(n_top_n, 0), sum_accuracy_top_n(n_top_n, 0);
	float total_weights = 0;

	// For AUCs
	vector<unordered_map<float, vector<float>>> pred_to_labels(params->n_categ);
	vector<unordered_map<float, vector<float>>> pred_to_weights(params->n_categ);
	vector<float> tot_true_labels(params->n_categ);
	float tot_count = 0.0;

	while (iterator->fetch_next(thread_num, y, pred, weight, preds_order)) {

		// Sanity
		if (y > params->n_categ)
			MTHROW_AND_ERR("Found label %d while n_categ = %d\n", (int)y, params->n_categ);

		if (weight == -1)
			w = 1;
		else
			w = weight;

		// AUC per category
		if (params->do_class_auc) {
			for (int i = 0; i < iterator->num_categories; i++) {
				int outcome = ((int)y == i) ? 1 : 0;
				pred_to_labels[i][pred[i]].push_back(outcome);
				pred_to_weights[i][pred[i]].push_back(w);
				tot_true_labels[i] += outcome * w;
			}
			tot_count += w;
		}

		// Position of true class
		int i_equal = -1;
		//MLOG("y = %f pred = %f Order: \n", y, *pred);
		for (int i = 0; i < iterator->num_categories; i++)
		{
			//MLOG("preds_order[%d] %d \n", i, preds_order[i]);
			if (preds_order[i] >= params->n_categ)
				MTHROW_AND_ERR("Found predict %d while n_categ = %d\n", (int)y, params->n_categ);
			dist_vals[i] = params->dist_matrix[(int)y][preds_order[i]];
			if (y == preds_order[i])
				i_equal = i;
		}

		// Measures limited by position of true class
		float sum_den_weighted_until_y = 0, sum_weighted_preds_until_y = 0, sum_weighted_until_y = 0, sum_avg_until_y = 0;
		for (int i = 0; i <= i_equal; i++)
		{
			sum_avg_until_y += dist_vals[i];
			sum_weighted_until_y += (i_equal - i + 1)*dist_vals[i];
			sum_den_weighted_until_y += (i_equal - i + 1);
			sum_weighted_preds_until_y += *(pred + preds_order[i]) *dist_vals[i];
		}

		avg_dist_until_correct += w * (sum_avg_until_y / (i_equal + 1));
		avg_weighted_dist_until_correct += w * (sum_weighted_until_y / sum_den_weighted_until_y);
		avg_weighted_preds_dist_until_correct += w * sum_weighted_preds_until_y;

		// Measures limited by top_n
		for (int in = 0; in < n_top_n; in++) {
			int top_n = params->top_n[in];
			float sum_avg = 0, sum_weighted = 0, sum_den_weighted = 0, sum_weighted_preds = 0;
			for (int i = 0; i < top_n; i++) {
				sum_avg += dist_vals[i];
				sum_weighted += (top_n - i + 1)*dist_vals[i];
				sum_den_weighted += (top_n - i + 1);
				sum_weighted_preds += *(pred + preds_order[i]) * dist_vals[i];
			}

			sum_accuracy_top_n[in] += w * (i_equal < top_n);
			avg_dist_top_n[in] += w * (sum_avg / top_n);
			avg_weighted_dist_top_n[in] += w * (sum_weighted / sum_den_weighted);
			avg_weighted_preds_dist_top_n[in] += w * sum_weighted_preds;
		}

		float sum_weighted_preds_dist_tmp = 0;
		for (int i = 0; i < iterator->num_categories; i++)
			sum_weighted_preds_dist_tmp += dist_vals[i] * *(pred + preds_order[i]);
		sum_weighted_preds_dist += w * params->dist_weights[(int)y] * sum_weighted_preds_dist_tmp;

		sum_correct_location += w * (i_equal + 1); // start from one
		total_weights += w;
	}

	for (int in = 0; in < n_top_n; in++) {
		int n = params->top_n[in];
		res["AVG_" + params->dist_name + "_TOP_" + to_string(n)] = avg_dist_top_n[in] / total_weights;
		res["AVG_WEIGHTED_" + params->dist_name + "_TOP_" + to_string(n)] = avg_weighted_dist_top_n[in] / total_weights;
		res["AVG_WEIGHTED_PREDS_" + params->dist_name + "_TOP_" + to_string(n)] = avg_weighted_preds_dist_top_n[in] / total_weights;
		res["ACCURACY_TOP_" + to_string(n)] = sum_accuracy_top_n[in] / total_weights;
	}

	res["AVG_" + params->dist_name + "_UNTIL_CORRECT"] = avg_dist_until_correct / total_weights;
	res["AVG_WEIGHTED_" + params->dist_name + "_UNTIL_CORRECT"] = avg_weighted_dist_until_correct / total_weights;
	res["AVG_WEIGHTED_PREDS_" + params->dist_name + "_UNTIL_CORRECT"] = avg_weighted_preds_dist_until_correct / total_weights;
	res["AVG_WEIGHTED_PREDS_" + params->dist_name] = sum_weighted_preds_dist / total_weights;
	res["AVG_CORRECT_LOCATION"] = sum_correct_location / total_weights;

	// AUC per category
	if (params->do_class_auc) {
		for (int i = 0; i < params->n_categ; i++) {
			if (tot_true_labels[i] > 0)
				res["CLASS_" + to_string(i) + "_AUC"] = get_auc(tot_count, tot_true_labels[i], pred_to_labels[i], pred_to_weights[i]);
		}
	}

	return res;
}

map<string, float> calc_roc_measures_with_inc(Lazy_Iterator *iterator, int thread_num, Measurement_Params *function_params) {
	map<string, float> res;
	
	bool censor_removed = true;
	bool trunc_max = false;

	ROC_Params *params = (ROC_Params *)function_params;
	float max_diff_in_wp = params->max_diff_working_point;
	float max_diff_in_wp_top_n = max_diff_in_wp;
	int max_qunt_vals = params->min_score_quants_to_force_score_wp;

	vector<int> topN_points = params->working_point_TOPN; //Working Top N points:
	sort(topN_points.begin(), topN_points.end());
	vector<float> fpr_points = params->working_point_FPR; //Working FPR points:
	sort(fpr_points.begin(), fpr_points.end());
	for (size_t i = 0; i < fpr_points.size(); ++i)
		fpr_points[i] /= 100.0;
	vector<float> sens_points = params->working_point_SENS; //Working SENS points:
	sort(sens_points.begin(), sens_points.end());
	for (size_t i = 0; i < sens_points.size(); ++i)
		sens_points[i] /= 100.0;
	vector<float> pr_points = params->working_point_PR; //Working PR points:
	sort(pr_points.begin(), pr_points.end());
	for (size_t i = 0; i < pr_points.size(); ++i)
		pr_points[i] /= 100.0;
	vector<float> score_points = params->working_point_Score; // Working Score points
	sort(score_points.begin(), score_points.end());
	reverse(score_points.begin(), score_points.end());

	unordered_map<float, vector<float>> thresholds_labels;
	unordered_map<float, vector<float>> thresholds_weights;
	vector<float> unique_scores;
	float y, pred, weight;
	while (iterator->fetch_next(thread_num, y, pred, weight)) {
		thresholds_labels[pred].push_back(y);
		if (weight > 0)
			thresholds_weights[pred].push_back(weight);
	}

	unique_scores.resize((int)thresholds_labels.size());
	int ind_p = 0;
	for (auto it = thresholds_labels.begin(); it != thresholds_labels.end(); ++it)
	{
		unique_scores[ind_p] = it->first;
		++ind_p;
	}
	sort(unique_scores.begin(), unique_scores.end());

	//calc measures on each bucket of scores as possible threshold:
	double t_sum = 0, f_sum = 0, tt_cnt = 0;
	double f_cnt = 0;
	double t_cnt = 0;
	vector<float> true_rate((int)unique_scores.size());
	vector<float> false_rate((int)unique_scores.size());
	int st_size = (int)unique_scores.size() - 1;
	if (thresholds_weights.empty())
		for (int i = st_size; i >= 0; --i)
		{
			vector<float> *labels = &thresholds_labels[unique_scores[i]];
			for (float y : *labels)
			{
				float true_label = params->fix_label_to_binary ? y > 0 : y;
				t_sum += true_label; /// counts also false positives weights if no fix_label_to_binary
				tt_cnt += true_label > 0 ? true_label : 0;
				if (!censor_removed)
					f_sum += (1 - true_label);
				else
					f_sum += int(true_label <= 0);
				f_cnt += int(true_label <= 0);
				t_cnt += int(true_label > 0);
			}
			true_rate[st_size - i] = float(t_sum);
			false_rate[st_size - i] = float(f_sum);
		}
	else
		for (int i = st_size; i >= 0; --i)
		{
			vector<float> *labels = &thresholds_labels[unique_scores[i]];
			vector<float> *weights = &thresholds_weights[unique_scores[i]];
			if (labels->size() != weights->size())
				MTHROW_AND_ERR("Error in bootstrap: labels(%zu) and weights(%zu) not in same size\n",
					labels->size(), weights->size());
			for (int y_i = 0; y_i < labels->size(); ++y_i)
			{
				float true_label = params->fix_label_to_binary ? (*labels)[y_i] > 0 : (*labels)[y_i];
				t_sum += static_cast<double>(true_label) * static_cast<double>((*weights)[y_i]);
				tt_cnt += static_cast<double>((true_label > 0 ? true_label : 0)) * static_cast<double>((*weights)[y_i]);
				if (!censor_removed)
					f_sum += static_cast<double>(1 - true_label) * static_cast<double>((*weights)[y_i]);
				else
					f_sum += int(true_label <= 0) * (*weights)[y_i];
				f_cnt += int(true_label <= 0) * (*weights)[y_i];
				t_cnt += int(true_label > 0) * (*weights)[y_i];
			}
			true_rate[st_size - i] = float(t_sum);
			false_rate[st_size - i] = float(f_sum);
		}

	if (f_cnt <= 0 || t_sum <= 0) {
		if (params->show_warns) {
			if (t_sum <= 0)
				MWARN("no positives exists in cohort\n");
			else
				MWARN("no falses exists in cohort\n");
		}
		return res;
	}
	if (params->show_warns) {
		int last_idx = -1;
		for (size_t i = 0; i < true_rate.size(); ++i) {
			if (true_rate[i] < 0)
				last_idx = (int)i;
		}

		if (last_idx > -1 && false_rate[last_idx] / f_sum >= 0.005)
			MWARN("true positive has negative values - outcome fix is too aggresive (index=%d, fpr=%2.1f%%, sens=%2.1f%%)\n",
				last_idx, 100 * false_rate[last_idx] / f_sum, 100 * true_rate[last_idx] / float(!trunc_max ? t_sum : tt_cnt));
	}
	for (size_t i = 0; i < true_rate.size(); ++i)
		if (true_rate[i] < 0)
			true_rate[i] = 0;

	for (size_t i = 0; i < true_rate.size(); ++i) {
		true_rate[i] /= float(!trunc_max ? t_sum : tt_cnt);
		false_rate[i] /= float(f_sum);
	}
	//calc maesures based on true_rate and false_rate
	double auc = false_rate[0] * true_rate[0] / 2; //"auc" on expectitions:
	for (size_t i = 1; i < true_rate.size(); ++i)
		auc += (false_rate[i] - false_rate[i - 1]) * (true_rate[i - 1] + true_rate[i]) / 2;

	max_diff_in_wp_top_n = max_diff_in_wp_top_n*(t_sum+f_sum) * 0.1;
	// Partial aucs, if required
	vector<float> part_aucs(params->working_point_auc.size());
	if (params->working_point_auc.size()) {
		part_aucs[0] = false_rate[0] * true_rate[0] / 2;
		size_t working_point = 0;
		for (size_t i = 1; i < true_rate.size(); ++i) {
			if (false_rate[i] > params->working_point_auc[working_point]) {
				// Move to next partial auc
				working_point++;
				if (working_point == params->working_point_auc.size())
					break;
				part_aucs[working_point] = part_aucs[working_point - 1];
			}
			part_aucs[working_point] += (false_rate[i] - false_rate[i - 1]) * (true_rate[i - 1] + true_rate[i]) / 2;
		}
	}


	bool use_wp = unique_scores.size() > max_qunt_vals && !params->use_score_working_points; //change all working points
	int curr_wp_fpr_ind = 0, curr_wp_sens_ind = 0, curr_wp_pr_ind = 0, curr_wp_score_ind = 0, curr_wp_topn_ind = 0;
	int i = 0;

	float ppv_c, pr_prev, ppv_prev, pr_c, score_c, score_prev, npv_c, npv_prev, or_prev, or_c, rr_prev, rr_c;
	if (use_wp) {
		//fpr points:
		i = 1;
		while (i < true_rate.size() && curr_wp_fpr_ind < fpr_points.size())
		{
			if (curr_wp_fpr_ind < fpr_points.size() &&
				false_rate[i] >= fpr_points[curr_wp_fpr_ind]) { //passed work_point - take 2 last points for measure - by distance from wp

				float prev_diff = fpr_points[curr_wp_fpr_ind] - false_rate[i - 1];
				float curr_diff = false_rate[i] - fpr_points[curr_wp_fpr_ind];
				float tot_diff = prev_diff + curr_diff;
				if (tot_diff <= 0) {
					curr_diff = 1;
					tot_diff = 2; //take prev - first apeareance
					prev_diff = 1;
				}
				if (prev_diff > max_diff_in_wp || curr_diff > max_diff_in_wp) {
					res[format_working_point("SCORE@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("SENS@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("PR@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("PPV@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("NPV@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("OR@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("LIFT@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("RR@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
#ifdef  WARN_SKIP_WP
					MWARN("SKIP WORKING POINT FPR=%f, prev_FPR=%f, next_FPR=%f, prev_score=%f, next_score=%f\n",
						fpr_points[curr_wp_fpr_ind], false_rate[i - 1], false_rate[i],
						pred_threshold[st_size - (i - 1)], pred_threshold[st_size - i]);
#endif
					++curr_wp_fpr_ind;
					continue; //skip working point - diff is too big
				}
				res[format_working_point("SCORE@FPR", fpr_points[curr_wp_fpr_ind])] = unique_scores[st_size - i] * (prev_diff / tot_diff) +
					unique_scores[st_size - (i - 1)] * (curr_diff / tot_diff);
				res[format_working_point("SENS@FPR", fpr_points[curr_wp_fpr_ind])] = 100 * (true_rate[i] * (prev_diff / tot_diff) +
					true_rate[i - 1] * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					ppv_c = float(params->incidence_fix*true_rate[i] / (params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]));
					if (true_rate[i - 1] > 0 || false_rate[i - 1] > 0)
						ppv_prev = float(params->incidence_fix*true_rate[i - 1] / (params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]));
					else
						ppv_prev = ppv_c;
				}
				else {
					ppv_c = float((true_rate[i] * t_sum) /
						((true_rate[i] * t_sum) + (false_rate[i] * f_sum)));
					if (true_rate[i - 1] > 0 || false_rate[i - 1] > 0)
						ppv_prev = float((true_rate[i - 1] * t_sum) /
						((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)));
					else
						ppv_prev = ppv_c;
				}
				float ppv = ppv_c * (prev_diff / tot_diff) + ppv_prev * (curr_diff / tot_diff);
				res[format_working_point("PPV@FPR", fpr_points[curr_wp_fpr_ind])] = 100 * ppv;
				if (params->incidence_fix > 0) {
					pr_c = float(params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]);
					pr_prev = float(params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]);
				}
				else {
					pr_c = float(((true_rate[i] * t_sum) + (false_rate[i] * f_sum)) /
						(t_sum + f_sum));
					pr_prev = float(((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)) /
						(t_sum + f_sum));
				}
				res[format_working_point("PR@FPR", fpr_points[curr_wp_fpr_ind])] = 100 * (pr_c* (prev_diff / tot_diff) + pr_prev * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					npv_prev = float(((1 - false_rate[i - 1]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i - 1]) *  params->incidence_fix) + ((1 - false_rate[i - 1]) *  (1 - params->incidence_fix))));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i]) *  params->incidence_fix) + ((1 - false_rate[i]) *  (1 - params->incidence_fix))));
					else
						npv_c = npv_prev;
				}
				else {
					npv_prev = float(((1 - false_rate[i - 1]) * f_sum) /
						(((1 - true_rate[i - 1]) * t_sum) + ((1 - false_rate[i - 1]) * f_sum)));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) * f_sum) /
						(((1 - true_rate[i]) * t_sum) + ((1 - false_rate[i]) * f_sum)));
					else
						npv_c = npv_prev;
				}
				res[format_working_point("NPV@FPR", fpr_points[curr_wp_fpr_ind])] = 100 * (npv_c * (prev_diff / tot_diff) + npv_prev * (curr_diff / tot_diff));
				if (params->incidence_fix > 0)
					res[format_working_point("LIFT@FPR", fpr_points[curr_wp_fpr_ind])] = float(ppv / params->incidence_fix);
				else
					res[format_working_point("LIFT@FPR", fpr_points[curr_wp_fpr_ind])] = float(ppv /
					(t_sum / (t_sum + f_sum))); //lift of prevalance when there is no inc

				if (false_rate[i] > 0 && false_rate[i] < 1 && true_rate[i] < 1)
					or_c = float(
					(true_rate[i] / false_rate[i]) / ((1 - true_rate[i]) / (1 - false_rate[i])));
				else
					or_c = MED_MAT_MISSING_VALUE;
				if (false_rate[i - 1] > 0 && false_rate[i - 1] < 1 && true_rate[i - 1] < 1)
					or_prev = float(
					(true_rate[i - 1] / false_rate[i - 1]) / ((1 - true_rate[i - 1]) / (1 - false_rate[i - 1])));
				else
					or_prev = MED_MAT_MISSING_VALUE;
				if (or_c != MED_MAT_MISSING_VALUE && or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@FPR", fpr_points[curr_wp_fpr_ind])] = (or_c * (prev_diff / tot_diff) +
						or_prev * (curr_diff / tot_diff));
				else if (or_c != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@FPR", fpr_points[curr_wp_fpr_ind])] = or_c;
				else if (or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@FPR", fpr_points[curr_wp_fpr_ind])] = or_prev;
				else
					res[format_working_point("OR@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;

				if (params->incidence_fix > 0) {
					if (true_rate[i - 1] < 1)
						rr_prev = float(ppv_prev + ppv_prev * (1 - params->incidence_fix)* (1 - false_rate[i - 1]) /
						(params->incidence_fix * (1 - true_rate[i - 1])));
					else
						rr_prev = MED_MAT_MISSING_VALUE;

					if (true_rate[i] < 1)
						rr_c = float(ppv_c + ppv_c * (1 - params->incidence_fix)* (1 - false_rate[i]) /
						(params->incidence_fix * (1 - true_rate[i])));
					else
						rr_c = MED_MAT_MISSING_VALUE;
					if (rr_c != MED_MAT_MISSING_VALUE && rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@FPR", fpr_points[curr_wp_fpr_ind])] = (rr_c * (prev_diff / tot_diff) +
							rr_prev * (curr_diff / tot_diff));
					else if (rr_c != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@FPR", fpr_points[curr_wp_fpr_ind])] = rr_c;
					else if (rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@FPR", fpr_points[curr_wp_fpr_ind])] = rr_prev;
					else
						res[format_working_point("RR@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
				}
				else {
					if (true_rate[i] < 1 || ppv == MED_MAT_MISSING_VALUE) {
						float FOR = float(((1.0 - true_rate[i]) * t_sum) /
							((1 - true_rate[i]) * t_sum + (1 - false_rate[i]) * f_sum));
						res[format_working_point("RR@FPR", fpr_points[curr_wp_fpr_ind])] =
							float(ppv / FOR);
					}
					else
						res[format_working_point("RR@FPR", fpr_points[curr_wp_fpr_ind])] = MED_MAT_MISSING_VALUE;
				}

				++curr_wp_fpr_ind;
				continue;
			}
			++i;
		}

		//handle sens points:
		i = 1; //first point is always before
		while (i < true_rate.size() && curr_wp_sens_ind < sens_points.size())
		{
			if (curr_wp_sens_ind < sens_points.size() &&
				true_rate[i] >= sens_points[curr_wp_sens_ind]) { //passed work_point - take 2 last points for measure - by distance from wp

				float prev_diff = sens_points[curr_wp_sens_ind] - true_rate[i - 1];
				float curr_diff = true_rate[i] - sens_points[curr_wp_sens_ind];
				float tot_diff = prev_diff + curr_diff;
				if (tot_diff <= 0) {
					curr_diff = 1;
					tot_diff = 2; //take prev - first apeareance
					prev_diff = 1;
				}
				if (prev_diff > max_diff_in_wp || curr_diff > max_diff_in_wp) {
					res[format_working_point("SCORE@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("FPR@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("SPEC@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("PR@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("PPV@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("NPV@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("OR@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("RR@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("LIFT@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
#ifdef  WARN_SKIP_WP
					MWARN("SKIP WORKING POINT SENS=%f, prev_SENS=%f, next_SENS=%f, prev_score=%f, next_score=%f\n",
						sens_points[curr_wp_sens_ind], true_rate[i - 1], true_rate[i],
						pred_threshold[st_size - (i - 1)], pred_threshold[st_size - i]);
#endif
					++curr_wp_sens_ind;
					continue; //skip working point - diff is too big
				}
				res[format_working_point("SCORE@SENS", sens_points[curr_wp_sens_ind])] = unique_scores[st_size - i] * (prev_diff / tot_diff) +
					unique_scores[st_size - (i - 1)] * (curr_diff / tot_diff);
				res[format_working_point("FPR@SENS", sens_points[curr_wp_sens_ind])] = 100 * (false_rate[i] * (prev_diff / tot_diff) +
					false_rate[i - 1] * (curr_diff / tot_diff));
				res[format_working_point("SPEC@SENS", sens_points[curr_wp_sens_ind])] = 100 * ((1 - false_rate[i]) * (prev_diff / tot_diff) +
					(1 - false_rate[i - 1]) * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					ppv_c = float(params->incidence_fix*true_rate[i] / (params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]));
					if (true_rate[i - 1] > 0 || false_rate[i - 1] > 0)
						ppv_prev = float(params->incidence_fix*true_rate[i - 1] / (params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]));
					else
						ppv_prev = ppv_c;
				}
				else {
					ppv_c = float((true_rate[i] * t_sum) /
						((true_rate[i] * t_sum) + (false_rate[i] * f_sum)));
					if (true_rate[i - 1] > 0 || false_rate[i - 1] > 0)
						ppv_prev = float((true_rate[i - 1] * t_sum) /
						((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)));
					else
						ppv_prev = ppv_c;
				}
				float ppv = ppv_c * (prev_diff / tot_diff) + ppv_prev * (curr_diff / tot_diff);
				res[format_working_point("PPV@SENS", sens_points[curr_wp_sens_ind])] = 100 * ppv;
				if (params->incidence_fix > 0) {
					pr_c = float(params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]);
					pr_prev = float(params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]);
				}
				else {
					pr_c = float(((true_rate[i] * t_sum) + (false_rate[i] * f_sum)) /
						(t_sum + f_sum));
					pr_prev = float(((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)) /
						(t_sum + f_sum));
				}
				res[format_working_point("PR@SENS", sens_points[curr_wp_sens_ind])] = 100 * (pr_c* (prev_diff / tot_diff) + pr_prev * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					npv_prev = float(((1 - false_rate[i - 1]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i - 1]) *  params->incidence_fix) + ((1 - false_rate[i - 1]) *  (1 - params->incidence_fix))));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i]) *  params->incidence_fix) + ((1 - false_rate[i]) *  (1 - params->incidence_fix))));
					else
						npv_c = npv_prev;
				}
				else {
					npv_prev = float(((1 - false_rate[i - 1]) * f_sum) /
						(((1 - true_rate[i - 1]) * t_sum) + ((1 - false_rate[i - 1]) * f_sum)));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) * f_sum) /
						(((1 - true_rate[i]) * t_sum) + ((1 - false_rate[i]) * f_sum)));
					else
						npv_c = npv_prev;
				}
				res[format_working_point("NPV@SENS", sens_points[curr_wp_sens_ind])] = 100 * (npv_c * (prev_diff / tot_diff) + npv_prev * (curr_diff / tot_diff));
				if (params->incidence_fix > 0)
					res[format_working_point("LIFT@SENS", sens_points[curr_wp_sens_ind])] = float(ppv / params->incidence_fix);
				else
					res[format_working_point("LIFT@SENS", sens_points[curr_wp_sens_ind])] = float(ppv /
					(t_sum / (t_sum + f_sum))); //lift of prevalance when there is no inc

				if (false_rate[i] > 0 && false_rate[i] < 1 && true_rate[i] < 1)
					or_c = float(
					(true_rate[i] / false_rate[i]) / ((1 - true_rate[i]) / (1 - false_rate[i])));
				else
					or_c = MED_MAT_MISSING_VALUE;
				if (false_rate[i - 1] > 0 && false_rate[i - 1] < 1 && true_rate[i - 1] < 1)
					or_prev = float(
					(true_rate[i - 1] / false_rate[i - 1]) / ((1 - true_rate[i - 1]) / (1 - false_rate[i - 1])));
				else
					or_prev = MED_MAT_MISSING_VALUE;
				if (or_c != MED_MAT_MISSING_VALUE && or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@SENS", sens_points[curr_wp_sens_ind])] = (or_c * (prev_diff / tot_diff) +
						or_prev * (curr_diff / tot_diff));
				else if (or_c != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@SENS", sens_points[curr_wp_sens_ind])] = or_c;
				else if (or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@SENS", sens_points[curr_wp_sens_ind])] = or_prev;
				else
					res[format_working_point("OR@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;

				if (params->incidence_fix > 0) {
					if (true_rate[i - 1] < 1)
						rr_prev = float(ppv_prev + ppv_prev * (1 - params->incidence_fix)* (1 - false_rate[i - 1]) /
						(params->incidence_fix * (1 - true_rate[i - 1])));
					else
						rr_prev = MED_MAT_MISSING_VALUE;

					if (true_rate[i] < 1)
						rr_c = float(ppv_c + ppv_c * (1 - params->incidence_fix)* (1 - false_rate[i]) /
						(params->incidence_fix * (1 - true_rate[i])));
					else
						rr_c = MED_MAT_MISSING_VALUE;
					if (rr_c != MED_MAT_MISSING_VALUE && rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@SENS", sens_points[curr_wp_sens_ind])] = (rr_c * (prev_diff / tot_diff) +
							rr_prev * (curr_diff / tot_diff));
					else if (rr_c != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@SENS", sens_points[curr_wp_sens_ind])] = rr_c;
					else if (rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@SENS", sens_points[curr_wp_sens_ind])] = rr_prev;
					else
						res[format_working_point("RR@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
				}
				else {
					if (true_rate[i] < 1 || ppv == MED_MAT_MISSING_VALUE) {
						float FOR = float(((1.0 - true_rate[i]) * t_sum) /
							((1 - true_rate[i]) * t_sum + (1 - false_rate[i]) * f_sum));
						res[format_working_point("RR@SENS", sens_points[curr_wp_sens_ind])] =
							float(ppv / FOR);
					}
					else
						res[format_working_point("RR@SENS", sens_points[curr_wp_sens_ind])] = MED_MAT_MISSING_VALUE;
				}

				++curr_wp_sens_ind;
				continue;
			}
			++i;
		}

		//handle pr points:
		i = 1; //first point is always before
		while (i < true_rate.size() && curr_wp_pr_ind < pr_points.size())
		{
			if (params->incidence_fix > 0)
				pr_c = float(params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]);
			else
				pr_c = float(((true_rate[i] * t_sum) + (false_rate[i] * f_sum)) /
				(t_sum + f_sum));

			if (curr_wp_pr_ind < pr_points.size() && pr_c >= pr_points[curr_wp_pr_ind]) { //passed work_point - take 2 last points for measure - by distance from wp
				if (params->incidence_fix > 0)
					pr_prev = float(params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]);
				else
					pr_prev = float(((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)) /
					(t_sum + f_sum));

				float prev_diff = pr_points[curr_wp_pr_ind] - pr_prev;
				float curr_diff = pr_c - pr_points[curr_wp_pr_ind];
				float tot_diff = prev_diff + curr_diff;
				if (tot_diff <= 0) {
					curr_diff = 1;
					tot_diff = 2; //take prev - first apeareance
					prev_diff = 1;
				}
				if (prev_diff > max_diff_in_wp || curr_diff > max_diff_in_wp) {
					res[format_working_point("SCORE@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("FPR@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("SPEC@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("SENS@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("PPV@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("NPV@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("OR@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("RR@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point("LIFT@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
#ifdef  WARN_SKIP_WP
					MWARN("SKIP WORKING POINT PR=%f, prev_PR=%f, next_PR=%f, prev_score=%f, next_score=%f\n",
						pr_points[curr_wp_pr_ind], pr_prev, pr_c,
						pred_threshold[st_size - (i - 1)], pred_threshold[st_size - i]);
#endif //  WARN_SKIP_WP
					++curr_wp_pr_ind;
					continue; //skip working point - diff is too big
				}
				res[format_working_point("SCORE@PR", pr_points[curr_wp_pr_ind])] = unique_scores[st_size - i] * (prev_diff / tot_diff) +
					unique_scores[st_size - (i - 1)] * (curr_diff / tot_diff);
				res[format_working_point("FPR@PR", pr_points[curr_wp_pr_ind])] = 100 * (false_rate[i] * (prev_diff / tot_diff) +
					false_rate[i - 1] * (curr_diff / tot_diff));
				res[format_working_point("SPEC@PR", pr_points[curr_wp_pr_ind])] = 100 * ((1 - false_rate[i]) * (prev_diff / tot_diff) +
					(1 - false_rate[i - 1]) * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					ppv_c = float(params->incidence_fix*true_rate[i] / (params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]));
					if (false_rate[i - 1] > 0 || true_rate[i - 1] > 0)
						ppv_prev = float(params->incidence_fix*true_rate[i - 1] / (params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]));
					else
						ppv_prev = ppv_c;
				}
				else {
					ppv_c = float((true_rate[i] * t_sum) /
						((true_rate[i] * t_sum) + (false_rate[i] * f_sum)));
					if (false_rate[i - 1] > 0 || true_rate[i - 1] > 0)
						ppv_prev = float((true_rate[i - 1] * t_sum) /
						((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)));
					else
						ppv_prev = ppv_c;
				}
				float ppv = ppv_c * (prev_diff / tot_diff) + ppv_prev * (curr_diff / tot_diff);
				res[format_working_point("PPV@PR", pr_points[curr_wp_pr_ind])] = 100 * ppv;
				res[format_working_point("SENS@PR", pr_points[curr_wp_pr_ind])] = 100 * (true_rate[i] * (prev_diff / tot_diff) + true_rate[i - 1] * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					npv_prev = float(((1 - false_rate[i - 1]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i - 1]) *  params->incidence_fix) + ((1 - false_rate[i - 1]) *  (1 - params->incidence_fix))));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i]) *  params->incidence_fix) + ((1 - false_rate[i]) *  (1 - params->incidence_fix))));
					else
						npv_c = npv_prev;
				}
				else {
					npv_prev = float(((1 - false_rate[i - 1]) * f_sum) /
						(((1 - true_rate[i - 1]) * t_sum) + ((1 - false_rate[i - 1]) * f_sum)));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) * f_sum) /
						(((1 - true_rate[i]) * t_sum) + ((1 - false_rate[i]) * f_sum)));
					else
						npv_c = npv_prev;
				}
				res[format_working_point("NPV@PR", pr_points[curr_wp_pr_ind])] = 100 * (npv_c * (prev_diff / tot_diff) + npv_prev * (curr_diff / tot_diff));
				if (params->incidence_fix > 0)
					res[format_working_point("LIFT@PR", pr_points[curr_wp_pr_ind])] = float(ppv / params->incidence_fix);
				else
					res[format_working_point("LIFT@PR", pr_points[curr_wp_pr_ind])] = float(ppv /
					(t_sum / (t_sum + f_sum))); //lift of prevalance when there is no inc
				if (false_rate[i] > 0 && false_rate[i] < 1 && true_rate[i] < 1)
					or_c = float(
					(true_rate[i] / false_rate[i]) / ((1 - true_rate[i]) / (1 - false_rate[i])));
				else
					or_c = MED_MAT_MISSING_VALUE;
				if (false_rate[i - 1] > 0 && false_rate[i - 1] < 1 && true_rate[i - 1] < 1)
					or_prev = float(
					(true_rate[i - 1] / false_rate[i - 1]) / ((1 - true_rate[i - 1]) / (1 - false_rate[i - 1])));
				else
					or_prev = MED_MAT_MISSING_VALUE;
				if (or_c != MED_MAT_MISSING_VALUE && or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@PR", pr_points[curr_wp_pr_ind])] = (or_c * (prev_diff / tot_diff) +
						or_prev * (curr_diff / tot_diff));
				else if (or_c != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@PR", pr_points[curr_wp_pr_ind])] = or_c;
				else if (or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@PR", pr_points[curr_wp_pr_ind])] = or_prev;
				else
					res[format_working_point("OR@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;

				if (params->incidence_fix > 0) {
					if (true_rate[i - 1] < 1)
						rr_prev = float(ppv_prev + ppv_prev * (1 - params->incidence_fix)* (1 - false_rate[i - 1]) /
						(params->incidence_fix * (1 - true_rate[i - 1])));
					else
						rr_prev = MED_MAT_MISSING_VALUE;

					if (true_rate[i] < 1)
						rr_c = float(ppv_c + ppv_c * (1 - params->incidence_fix)* (1 - false_rate[i]) /
						(params->incidence_fix * (1 - true_rate[i])));
					else
						rr_c = MED_MAT_MISSING_VALUE;
					if (rr_c != MED_MAT_MISSING_VALUE && rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@PR", pr_points[curr_wp_pr_ind])] = (rr_c * (prev_diff / tot_diff) +
							rr_prev * (curr_diff / tot_diff));
					else if (rr_c != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@PR", pr_points[curr_wp_pr_ind])] = rr_c;
					else if (rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@PR", pr_points[curr_wp_pr_ind])] = rr_prev;
					else
						res[format_working_point("RR@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
				}
				else {
					if (true_rate[i] < 1 || ppv == MED_MAT_MISSING_VALUE) {
						float FOR = float(((1.0 - true_rate[i]) * t_sum) /
							((1 - true_rate[i]) * t_sum + (1 - false_rate[i]) * f_sum));
						res[format_working_point("RR@PR", pr_points[curr_wp_pr_ind])] =
							float(ppv / FOR);
					}
					else

						res[format_working_point("RR@PR", pr_points[curr_wp_pr_ind])] = MED_MAT_MISSING_VALUE;
				}

				++curr_wp_pr_ind;
				continue;
			}
			++i;
		}

		//handle score points:
		i = 1; //first point is always before
		while (i < true_rate.size() && curr_wp_score_ind < score_points.size())
		{
			score_c = unique_scores[true_rate.size() - i - 1];
			if (curr_wp_score_ind < score_points.size() && score_c <= score_points[curr_wp_score_ind]) { //passed work_point - take 2 last points for measure - by distance from wp
				score_prev = unique_scores[true_rate.size() - i];

				float prev_diff = score_prev - score_points[curr_wp_score_ind];
				float curr_diff = score_points[curr_wp_score_ind] - score_c;
				float tot_diff = prev_diff + curr_diff;
				if (tot_diff <= 0) {
					curr_diff = 1;
					tot_diff = 2; //take prev - first apeareance
					prev_diff = 1;
				}
				if (prev_diff > max_diff_in_wp || curr_diff > max_diff_in_wp) {
					res[format_working_point("PR@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
					res[format_working_point("FPR@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
					res[format_working_point("SPEC@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
					res[format_working_point("SENS@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
					res[format_working_point("PPV@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
					res[format_working_point("NPV@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
					res[format_working_point("OR@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
					res[format_working_point("RR@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
					res[format_working_point("LIFT@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
#ifdef  WARN_SKIP_WP
					MWARN("SKIP WORKING POINT Score=%f, prev_Score%f, next_Score=%f\n",
						score_points[curr_wp_score_ind], score_prev, score_c);
#endif //  WARN_SKIP_WP
					++curr_wp_score_ind;
					continue; //skip working point - diff is too big
				}

				if (params->incidence_fix > 0) {
					pr_c = float(params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]);
					pr_prev = float(params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]);
				}
				else {
					pr_c = float(((true_rate[i] * t_sum) + (false_rate[i] * f_sum)) /
						(t_sum + f_sum));
					pr_prev = float(((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)) /
						(t_sum + f_sum));
				}
				res[format_working_point("PR@SCORE", score_points[curr_wp_score_ind], false)] = 100 * (pr_c* (prev_diff / tot_diff) + pr_prev * (curr_diff / tot_diff));

				res[format_working_point("FPR@SCORE", score_points[curr_wp_score_ind], false)] = 100 * (false_rate[i] * (prev_diff / tot_diff) + false_rate[i - 1] * (curr_diff / tot_diff));
				res[format_working_point("SPEC@SCORE", score_points[curr_wp_score_ind], false)] = 100 * ((1 - false_rate[i]) * (prev_diff / tot_diff) + (1 - false_rate[i - 1]) * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					ppv_c = float(params->incidence_fix*true_rate[i] / (params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]));
					if (false_rate[i - 1] > 0 || true_rate[i - 1] > 0)
						ppv_prev = float(params->incidence_fix*true_rate[i - 1] / (params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]));
					else
						ppv_prev = ppv_c;
				}
				else {
					ppv_c = float((true_rate[i] * t_sum) /
						((true_rate[i] * t_sum) + (false_rate[i] * f_sum)));
					if (false_rate[i - 1] > 0 || true_rate[i - 1] > 0)
						ppv_prev = float((true_rate[i - 1] * t_sum) /
						((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)));
					else
						ppv_prev = ppv_c;
				}
				float ppv = ppv_c * (prev_diff / tot_diff) + ppv_prev * (curr_diff / tot_diff);
				res[format_working_point("PPV@SCORE", score_points[curr_wp_score_ind], false)] = 100 * ppv;
				res[format_working_point("SENS@SCORE", score_points[curr_wp_score_ind], false)] = 100 * (true_rate[i] * (prev_diff / tot_diff) + true_rate[i - 1] * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					npv_prev = float(((1 - false_rate[i - 1]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i - 1]) *  params->incidence_fix) + ((1 - false_rate[i - 1]) *  (1 - params->incidence_fix))));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i]) *  params->incidence_fix) + ((1 - false_rate[i]) *  (1 - params->incidence_fix))));
					else
						npv_c = npv_prev;
				}
				else {
					npv_prev = float(((1 - false_rate[i - 1]) * f_sum) /
						(((1 - true_rate[i - 1]) * t_sum) + ((1 - false_rate[i - 1]) * f_sum)));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) * f_sum) /
						(((1 - true_rate[i]) * t_sum) + ((1 - false_rate[i]) * f_sum)));
					else
						npv_c = npv_prev;
				}
				res[format_working_point("NPV@SCORE", score_points[curr_wp_score_ind], false)] = 100 * (npv_c * (prev_diff / tot_diff) + npv_prev * (curr_diff / tot_diff));
				if (params->incidence_fix > 0)
					res[format_working_point("LIFT@SCORE", score_points[curr_wp_score_ind], false)] = float(ppv / params->incidence_fix);
				else
					res[format_working_point("LIFT@SCORE", score_points[curr_wp_score_ind], false)] = float(ppv /
					(t_sum / (t_sum + f_sum))); //lift of prevalance when there is no inc
				if (false_rate[i] > 0 && false_rate[i] < 1 && true_rate[i] < 1)
					or_c = float(
					(true_rate[i] / false_rate[i]) / ((1 - true_rate[i]) / (1 - false_rate[i])));
				else
					or_c = MED_MAT_MISSING_VALUE;
				if (false_rate[i - 1] > 0 && false_rate[i - 1] < 1 && true_rate[i - 1] < 1)
					or_prev = float(
					(true_rate[i - 1] / false_rate[i - 1]) / ((1 - true_rate[i - 1]) / (1 - false_rate[i - 1])));
				else
					or_prev = MED_MAT_MISSING_VALUE;
				if (or_c != MED_MAT_MISSING_VALUE && or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@SCORE", score_points[curr_wp_score_ind], false)] = (or_c * (prev_diff / tot_diff) +
						or_prev * (curr_diff / tot_diff));
				else if (or_c != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@SCORE", score_points[curr_wp_score_ind], false)] = or_c;
				else if (or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point("OR@SCORE", score_points[curr_wp_score_ind], false)] = or_prev;
				else
					res[format_working_point("OR@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;

				if (params->incidence_fix > 0) {
					if (true_rate[i - 1] < 1)
						rr_prev = float(ppv_prev + ppv_prev * (1 - params->incidence_fix)* (1 - false_rate[i - 1]) /
						(params->incidence_fix * (1 - true_rate[i - 1])));
					else
						rr_prev = MED_MAT_MISSING_VALUE;

					if (true_rate[i] < 1)
						rr_c = float(ppv_c + ppv_c * (1 - params->incidence_fix)* (1 - false_rate[i]) /
						(params->incidence_fix * (1 - true_rate[i])));
					else
						rr_c = MED_MAT_MISSING_VALUE;
					if (rr_c != MED_MAT_MISSING_VALUE && rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@SCORE", score_points[curr_wp_score_ind], false)] = (rr_c * (prev_diff / tot_diff) +
							rr_prev * (curr_diff / tot_diff));
					else if (rr_c != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@SCORE", score_points[curr_wp_score_ind], false)] = rr_c;
					else if (rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point("RR@SCORE", score_points[curr_wp_score_ind], false)] = rr_prev;
					else
						res[format_working_point("RR@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
				}
				else {
					if (true_rate[i] < 1 || ppv == MED_MAT_MISSING_VALUE) {
						float FOR = float(((1.0 - true_rate[i]) * t_sum) /
							((1 - true_rate[i]) * t_sum + (1 - false_rate[i]) * f_sum));
						res[format_working_point("RR@SCORE", score_points[curr_wp_score_ind], false)] =
							float(ppv / FOR);
					}
					else

						res[format_working_point("RR@SCORE", score_points[curr_wp_score_ind], false)] = MED_MAT_MISSING_VALUE;
				}

				++curr_wp_score_ind;
				continue;
			}
			++i;
		}

		//top N points:
		i = 1;
		int current_N_prev = 0;
		int current_N = (true_rate[0] * float(!trunc_max ? t_sum : tt_cnt)) + (false_rate[0] * f_sum);
		while (i < true_rate.size() && curr_wp_topn_ind < topN_points.size())
		{
			current_N = (true_rate[i] * float(!trunc_max ? t_sum : tt_cnt)) + (false_rate[i] * f_sum);
			if (curr_wp_topn_ind < topN_points.size() &&
				current_N >= topN_points[curr_wp_topn_ind]) { //passed top N point

				float prev_diff = topN_points[curr_wp_topn_ind] - current_N_prev;
				float curr_diff = current_N - topN_points[curr_wp_topn_ind];
				float tot_diff = prev_diff + curr_diff;
				if (tot_diff <= 0) {
					curr_diff = 1;
					tot_diff = 2; //take prev - first apeareance
					prev_diff = 1;
				}
				if (prev_diff > max_diff_in_wp_top_n || curr_diff > max_diff_in_wp_top_n) {
					res[format_working_point_topn("SCORE@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point_topn("FPR@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point_topn("SENS@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point_topn("POS@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point_topn("PR@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point_topn("PPV@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point_topn("NPV@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point_topn("OR@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point_topn("LIFT@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
					res[format_working_point_topn("RR@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
#ifdef  WARN_SKIP_WP
					MWARN("SKIP WORKING POINT FPR=%f, prev_FPR=%f, next_FPR=%f, prev_score=%f, next_score=%f\n",
						topN_points[curr_wp_topn_ind], false_rate[i - 1], false_rate[i],
						pred_threshold[st_size - (i - 1)], pred_threshold[st_size - i]);
#endif
					++curr_wp_topn_ind;
					continue; //skip working point - diff is too big
				}
				res[format_working_point_topn("SCORE@TOPN", topN_points[curr_wp_topn_ind])] = unique_scores[st_size - i] * (prev_diff / tot_diff) +
					unique_scores[st_size - (i - 1)] * (curr_diff / tot_diff);
				res[format_working_point_topn("SENS@TOPN", topN_points[curr_wp_topn_ind])] = 100 * (true_rate[i] * (prev_diff / tot_diff) +
					true_rate[i - 1] * (curr_diff / tot_diff));
				res[format_working_point_topn("POS@TOPN", topN_points[curr_wp_topn_ind])] = t_sum * (true_rate[i] * (prev_diff / tot_diff) +
					true_rate[i - 1] * (curr_diff / tot_diff));
				res[format_working_point_topn("FPR@TOPN", topN_points[curr_wp_topn_ind])] = 100 * (false_rate[i] * (prev_diff / tot_diff) +
					false_rate[i - 1] * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					ppv_c = float(params->incidence_fix*true_rate[i] / (params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]));
					if (true_rate[i - 1] > 0 || false_rate[i - 1] > 0)
						ppv_prev = float(params->incidence_fix*true_rate[i - 1] / (params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]));
					else
						ppv_prev = ppv_c;
				}
				else {
					ppv_c = float((true_rate[i] * t_sum) /
						((true_rate[i] * t_sum) + (false_rate[i] * f_sum)));
					if (true_rate[i - 1] > 0 || false_rate[i - 1] > 0)
						ppv_prev = float((true_rate[i - 1] * t_sum) /
						((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)));
					else
						ppv_prev = ppv_c;
				}
				float ppv = ppv_c * (prev_diff / tot_diff) + ppv_prev * (curr_diff / tot_diff);
				res[format_working_point_topn("PPV@TOPN", topN_points[curr_wp_topn_ind])] = 100 * ppv;
				if (params->incidence_fix > 0) {
					pr_c = float(params->incidence_fix*true_rate[i] + (1 - params->incidence_fix)*false_rate[i]);
					pr_prev = float(params->incidence_fix*true_rate[i - 1] + (1 - params->incidence_fix)*false_rate[i - 1]);
				}
				else {
					pr_c = float(((true_rate[i] * t_sum) + (false_rate[i] * f_sum)) /
						(t_sum + f_sum));
					pr_prev = float(((true_rate[i - 1] * t_sum) + (false_rate[i - 1] * f_sum)) /
						(t_sum + f_sum));
				}
				res[format_working_point_topn("PR@TOPN", topN_points[curr_wp_topn_ind])] = 100 * (pr_c* (prev_diff / tot_diff) + pr_prev * (curr_diff / tot_diff));
				if (params->incidence_fix > 0) {
					npv_prev = float(((1 - false_rate[i - 1]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i - 1]) *  params->incidence_fix) + ((1 - false_rate[i - 1]) *  (1 - params->incidence_fix))));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) *  (1 - params->incidence_fix)) /
						(((1 - true_rate[i]) *  params->incidence_fix) + ((1 - false_rate[i]) *  (1 - params->incidence_fix))));
					else
						npv_c = npv_prev;
				}
				else {
					npv_prev = float(((1 - false_rate[i - 1]) * f_sum) /
						(((1 - true_rate[i - 1]) * t_sum) + ((1 - false_rate[i - 1]) * f_sum)));
					if (true_rate[i] < 1 || false_rate[i] < 1)
						npv_c = float(((1 - false_rate[i]) * f_sum) /
						(((1 - true_rate[i]) * t_sum) + ((1 - false_rate[i]) * f_sum)));
					else
						npv_c = npv_prev;
				}
				res[format_working_point_topn("NPV@TOPN", topN_points[curr_wp_topn_ind])] = 100 * (npv_c * (prev_diff / tot_diff) + npv_prev * (curr_diff / tot_diff));
				if (params->incidence_fix > 0)
					res[format_working_point_topn("LIFT@TOPN", topN_points[curr_wp_topn_ind])] = float(ppv / params->incidence_fix);
				else
					res[format_working_point_topn("LIFT@TOPN", topN_points[curr_wp_topn_ind])] = float(ppv /
					(t_sum / (t_sum + f_sum))); //lift of prevalance when there is no inc

				if (false_rate[i] > 0 && false_rate[i] < 1 && true_rate[i] < 1)
					or_c = float(
					(true_rate[i] / false_rate[i]) / ((1 - true_rate[i]) / (1 - false_rate[i])));
				else
					or_c = MED_MAT_MISSING_VALUE;
				if (false_rate[i - 1] > 0 && false_rate[i - 1] < 1 && true_rate[i - 1] < 1)
					or_prev = float(
					(true_rate[i - 1] / false_rate[i - 1]) / ((1 - true_rate[i - 1]) / (1 - false_rate[i - 1])));
				else
					or_prev = MED_MAT_MISSING_VALUE;
				if (or_c != MED_MAT_MISSING_VALUE && or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point_topn("OR@TOPN", topN_points[curr_wp_topn_ind])] = (or_c * (prev_diff / tot_diff) +
						or_prev * (curr_diff / tot_diff));
				else if (or_c != MED_MAT_MISSING_VALUE)
					res[format_working_point_topn("OR@TOPN", topN_points[curr_wp_topn_ind])] = or_c;
				else if (or_prev != MED_MAT_MISSING_VALUE)
					res[format_working_point_topn("OR@TOPN", topN_points[curr_wp_topn_ind])] = or_prev;
				else
					res[format_working_point_topn("OR@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;

				if (params->incidence_fix > 0) {
					if (true_rate[i - 1] < 1)
						rr_prev = float(ppv_prev + ppv_prev * (1 - params->incidence_fix)* (1 - false_rate[i - 1]) /
						(params->incidence_fix * (1 - true_rate[i - 1])));
					else
						rr_prev = MED_MAT_MISSING_VALUE;

					if (true_rate[i] < 1)
						rr_c = float(ppv_c + ppv_c * (1 - params->incidence_fix)* (1 - false_rate[i]) /
						(params->incidence_fix * (1 - true_rate[i])));
					else
						rr_c = MED_MAT_MISSING_VALUE;
					if (rr_c != MED_MAT_MISSING_VALUE && rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point_topn("RR@TOPN", topN_points[curr_wp_topn_ind])] = (rr_c * (prev_diff / tot_diff) +
							rr_prev * (curr_diff / tot_diff));
					else if (rr_c != MED_MAT_MISSING_VALUE)
						res[format_working_point_topn("RR@TOPN", topN_points[curr_wp_topn_ind])] = rr_c;
					else if (rr_prev != MED_MAT_MISSING_VALUE)
						res[format_working_point_topn("RR@TOPN", topN_points[curr_wp_topn_ind])] = rr_prev;
					else
						res[format_working_point_topn("RR@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
				}
				else {
					if (true_rate[i] < 1 || ppv == MED_MAT_MISSING_VALUE) {
						float FOR = float(((1.0 - true_rate[i]) * t_sum) /
							((1 - true_rate[i]) * t_sum + (1 - false_rate[i]) * f_sum));
						res[format_working_point_topn("RR@TOPN", topN_points[curr_wp_topn_ind])] =
							float(ppv / FOR);
					}
					else
						res[format_working_point_topn("RR@TOPN", topN_points[curr_wp_topn_ind])] = MED_MAT_MISSING_VALUE;
				}

				++curr_wp_topn_ind;
				continue;
			}
			++i;
			current_N_prev = current_N;
		}

	}
	else {
		float score_working_point;
		for (i = 0; i < true_rate.size(); ++i)
		{
			score_working_point = unique_scores[st_size - i];
			res[format_working_point("SENS@SCORE", score_working_point, false)] = 100 * true_rate[i];
			res[format_working_point("SPEC@SCORE", score_working_point, false)] = 100 * (1 - false_rate[i]);
			float ppv = MED_MAT_MISSING_VALUE;
			if (true_rate[i] > 0 || false_rate[i] > 0) {
				if (params->incidence_fix > 0)
					ppv = float((true_rate[i] * params->incidence_fix) /
					(params->incidence_fix*(true_rate[i]) +
						(false_rate[i] * (1 - params->incidence_fix))));
				else
					ppv = float((true_rate[i] * t_sum) /
					((true_rate[i] * t_sum) + (false_rate[i] * f_sum)));
				res[format_working_point("PPV@SCORE", score_working_point, false)] = 100 * ppv;
			}
			else
				res[format_working_point("PPV@SCORE", score_working_point, false)] = MED_MAT_MISSING_VALUE;

			if (params->incidence_fix > 0) {
				res[format_working_point("PR@SCORE", score_working_point, false)] = float(100 * ((true_rate[i] * params->incidence_fix) + (false_rate[i] * (1 - params->incidence_fix))));
			}
			else {
				res[format_working_point("PR@SCORE", score_working_point, false)] = float(100 * ((true_rate[i] * t_sum) + (false_rate[i] * f_sum)) /
					(t_sum + f_sum));
			}
			if (true_rate[i] < 1 || false_rate[i] < 1) {
				if (params->incidence_fix > 0) {
					res[format_working_point("NPV@SCORE", score_working_point, false)] = float(100 * ((1 - false_rate[i]) * (1 - params->incidence_fix)) /
						(((1 - true_rate[i]) * params->incidence_fix) + ((1 - false_rate[i]) *  (1 - params->incidence_fix))));
				}
				else {
					res[format_working_point("NPV@SCORE", score_working_point, false)] = float(100 * ((1 - false_rate[i]) * f_sum) /
						(((1 - true_rate[i]) * t_sum) + ((1 - false_rate[i]) * f_sum)));
				}
			}
			else
				res[format_working_point("NPV@SCORE", score_working_point, false)] = MED_MAT_MISSING_VALUE;
			if (params->incidence_fix > 0) {
				if (true_rate[i] > 0 || false_rate[i] > 0 || ppv == MED_MAT_MISSING_VALUE)
					res[format_working_point("LIFT@SCORE", score_working_point, false)] = float(ppv / params->incidence_fix);
				else
					res[format_working_point("LIFT@SCORE", score_working_point, false)] = MED_MAT_MISSING_VALUE;
			}
			else {
				if (true_rate[i] > 0 || false_rate[i] > 0 || ppv == MED_MAT_MISSING_VALUE)
					res[format_working_point("LIFT@SCORE", score_working_point, false)] = float(ppv /
					(t_sum / (t_sum + f_sum)));
				else
					res[format_working_point("LIFT@SCORE", score_working_point, false)] = MED_MAT_MISSING_VALUE;
			}
			if (false_rate[i] > 0 && false_rate[i] < 1 && true_rate[i] < 1)
				res[format_working_point("OR@SCORE", score_working_point, false)] = float(
				(true_rate[i] / false_rate[i]) / ((1 - true_rate[i]) / (1 - false_rate[i])));
			else
				res[format_working_point("OR@SCORE", score_working_point, false)] = MED_MAT_MISSING_VALUE;

			if (params->incidence_fix > 0) {
				if (true_rate[i] < 1 || ppv == MED_MAT_MISSING_VALUE)
					res[format_working_point("RR@SCORE", score_working_point, false)] = float((ppv + ppv * (1 - params->incidence_fix)* (1 - false_rate[i]) /
					(params->incidence_fix * (1 - true_rate[i]))));
				else
					res[format_working_point("RR@SCORE", score_working_point, false)] = MED_MAT_MISSING_VALUE;
			}
			else {
				if (true_rate[i] < 1 || ppv == MED_MAT_MISSING_VALUE) {
					float FOR = float(((1.0 - true_rate[i]) * t_sum) /
						((1 - true_rate[i]) * t_sum + (1 - false_rate[i]) * f_sum));
					res[format_working_point("RR@SCORE", score_working_point, false)] =
						float(ppv / FOR);
				}
				else
					res[format_working_point("RR@SCORE", score_working_point, false)] = MED_MAT_MISSING_VALUE;
			}
		}
	}


	res["AUC"] = float(auc);
	if (!part_aucs.empty()) {
		for (unsigned int i = 0; i < part_aucs.size(); i++) {
			float fpr = params->working_point_auc[i];
			res[format_working_point("PART_AUC", fpr, false)] = part_aucs[i] / fpr;
		}
	}

	if (abs(t_cnt - t_sum) > 0.01) {
		res["NEG_SUM"] = float(f_sum);
		res["POS_SUM"] = float(t_sum);
		res["POS_CNT"] = float(t_cnt);
		res["NEG_CNT"] = float(f_cnt);
	}
	else {
		res["NNEG"] = float(f_sum);
		res["NPOS"] = float(t_sum);
	}

	return res;
}

map<string, float> calc_kandel_tau(Lazy_Iterator *iterator, int thread_num, Measurement_Params *function_params) {
	map<string, float> res;

	double tau = 0, cnt = 0;
	float y, pred, weight;
	//vector<float> scores, labels;
	unordered_map<float, vector<float>> label_to_scores;
	unordered_map<float, vector<float>> label_to_weights;
	while (iterator->fetch_next(thread_num, y, pred, weight)) {
		label_to_scores[y].push_back(pred);
		if (weight != -1)
			label_to_weights[y].push_back(pred);
	}

	for (auto it = label_to_scores.begin(); it != label_to_scores.end(); ++it)
		sort(it->second.begin(), it->second.end());

	if (label_to_weights.empty())
		for (auto it = label_to_scores.begin(); it != label_to_scores.end(); ++it)
		{
			auto bg = it;
			++bg;
			vector<float> *preds = &it->second;
			int pred_i_bigger;
			double pred_i_smaller;
			for (auto jt = bg; jt != label_to_scores.end(); ++jt)
			{
				vector<float> *preds_comp = &jt->second;
				double p_size = (double)preds_comp->size();
				for (float pred : *preds)
				{
					pred_i_bigger = binary_search_position(preds_comp->data(), preds_comp->data() + preds_comp->size() - 1, pred);
					pred_i_smaller = p_size - binary_search_position_last(preds_comp->data(), preds_comp->data() + preds_comp->size() - 1, pred);
					if (it->first > jt->first)
						//tau += pred_i_bigger;
						tau += pred_i_bigger - pred_i_smaller;
					else
						//tau += pred_i_smaller;
						tau += pred_i_smaller - pred_i_bigger;
				}
				cnt += p_size * preds->size();
			}
		}
	else {
		vector<double> group_weights(label_to_scores.size());
		vector<vector<double>> group_weights_cumsum(label_to_scores.size());
		int iter = 0;
		for (auto it = label_to_scores.begin(); it != label_to_scores.end(); ++it) {
			vector<float> *weights = &label_to_weights[it->first];
			for (size_t i = 0; i < it->second.size(); ++i) {
				group_weights[iter] += (*weights)[i];
				group_weights_cumsum[iter].push_back((*weights)[i]);
			}
			++iter;
		}
		//make cumsum:
		for (size_t i = 0; i < group_weights_cumsum.size(); ++i)
			for (size_t j = 1; j < group_weights_cumsum[i].size(); ++j)
				group_weights_cumsum[i][j] += group_weights_cumsum[i][j - 1];

		iter = 0;
		for (auto it = label_to_scores.begin(); it != label_to_scores.end(); ++it)
		{
			auto bg = it;
			++bg;
			vector<float> *preds = &it->second;

			double pred_i_bigger;
			double pred_i_smaller;
			int pred_i_bigger_i;
			int pred_i_smaller_i;
			int inside_group_idx = iter + 1;
			for (auto jt = bg; jt != label_to_scores.end(); ++jt)
			{
				vector<float> *preds_comp = &jt->second;
				//double p_size = (double)preds_comp->size();
				double p_size = group_weights[inside_group_idx];
				for (float pred : *preds)
				{
					pred_i_bigger_i = binary_search_position(preds_comp->data(), preds_comp->data() + preds_comp->size() - 1, pred);
					pred_i_smaller_i = binary_search_position_last(preds_comp->data(), preds_comp->data() + preds_comp->size() - 1, pred);
					if (pred_i_bigger_i < group_weights_cumsum[inside_group_idx].size())
						pred_i_bigger = group_weights_cumsum[inside_group_idx][pred_i_bigger_i];
					else
						pred_i_bigger = p_size;
					if (pred_i_smaller_i < group_weights_cumsum[inside_group_idx].size())
						pred_i_smaller = group_weights_cumsum[inside_group_idx][pred_i_smaller_i];
					else
						pred_i_smaller = p_size;

					if (it->first > jt->first)
						//tau += pred_i_bigger;
						tau += pred_i_bigger - (p_size - pred_i_smaller);
					else
						//tau += pred_i_smaller;
						tau += (p_size - pred_i_smaller) - pred_i_bigger;
				}
				cnt += p_size * preds->size();
				++inside_group_idx;
			}
			++iter;
		}
	}

	if (cnt > 1) {
		tau /= cnt;
		res["Kendall-Tau"] = (float)tau;
	}

	return res;
}

map<string, float> calc_harrell_c_statistic(Lazy_Iterator *iterator, int thread_num, Measurement_Params *function_params) {
	//Encoding:
	//Case/Control => effect outcome/y sign. positive is case, negative controls. Can't handle event in time zero.
	//Time to event => abs value of outcome/y
	//Score => the prediction

	map<string, float> res;
	double tau = 0, cnt = 0, case_cnt = 0, cntrl_cnt = 0;
	float y, pred, weight;
	//vector<float> scores, labels;
	map<float, vector<float>> cases_to_scores, controls_to_scores;
	while (iterator->fetch_next(thread_num, y, pred, weight)) {
		if (y > 0)
			cases_to_scores[y].push_back(pred);
		else
			controls_to_scores[-y].push_back(pred);
		if (weight != -1)
			MTHROW_AND_ERR("Error - not implemented with weights\n");
		case_cnt += int(y > 0);
		cntrl_cnt += int(y < 0);
	}

	//sort scores inside
	for (auto it = cases_to_scores.begin(); it != cases_to_scores.end(); ++it)
		sort(it->second.begin(), it->second.end());
	for (auto it = controls_to_scores.begin(); it != controls_to_scores.end(); ++it)
		sort(it->second.begin(), it->second.end());

	for (auto it = cases_to_scores.begin(); it != cases_to_scores.end(); ++it)
	{
		auto bg = it;
		++bg;
		const vector<float> &preds = it->second;
		int pred_i_bigger;
		double pred_i_smaller;
		//1. compare within cases
		for (auto jt = bg; jt != cases_to_scores.end(); ++jt)
		{
			const vector<float> &preds_comp = jt->second;
			//preds should get higher scores than preds_comp. preds has lower outcome (shorter time to event) than pred_comp
			double p_size = (double)preds_comp.size();
			for (float pred : preds)
			{
				pred_i_bigger = binary_search_position(preds_comp.data(), preds_comp.data() + preds_comp.size() - 1, pred);
				pred_i_smaller = p_size - binary_search_position_last(preds_comp.data(), preds_comp.data() + preds_comp.size() - 1, pred);

				tau += pred_i_bigger; //count only bigger as good
				cnt += pred_i_bigger + pred_i_smaller; //count total compares - also smaller
			}
		}

		//2. Compare case to control with farther time to event(censor)
		for (auto jt = controls_to_scores.begin(); jt != controls_to_scores.end(); ++jt) {
			if (jt->first < it->first)
				continue;
			const vector<float> &preds_comp = jt->second;
			double p_size = (double)preds_comp.size();
			//Here the controls has farther censor time than the cases
			//pred should have higher scores then preds_comp:
			for (float pred : preds)
			{
				pred_i_bigger = binary_search_position(preds_comp.data(), preds_comp.data() + preds_comp.size() - 1, pred);
				pred_i_smaller = p_size - binary_search_position_last(preds_comp.data(), preds_comp.data() + preds_comp.size() - 1, pred);

				tau += pred_i_bigger; //count only bigger as good
				cnt += pred_i_bigger + pred_i_smaller; //count total compares - also smaller
			}
		}
	}


	if (cnt > 1) {
		tau /= cnt;
		res["Harrell-C-Statistic"] = (float)tau;
		res["NPOS"] = case_cnt;
		res["NNEG"] = cntrl_cnt;
	}

	return res;
}

inline void update_vec_idx(vector<bool> &idx_status,
	double &total_on, const vector<float> &weights, const vector<int> &to_check) {
	//update idx_status by to_check idx - turn off whats needed and update total_on:
	for (int i : to_check)
	{
		if (idx_status[i]) {
			idx_status[i] = false;
			if (!weights.empty())
				total_on -= weights[i];
			else
				--total_on;
		}
	}
}

map<string, float> calc_regression(Lazy_Iterator *iterator, int thread_num, Measurement_Params *function_params) {
	map<string, float> res;

	Regression_Params* params = (Regression_Params *)function_params;

	float y, w, pred;
	bool has_weights = false;
	double tot_loss = 0, sum_outcome = 0, tot_count = 0, second_moment = 0,
		tot_mse = 0;
	map<float, float> per_score;
	map<float, double> per_diff;
	unordered_map<float, float> per_score_sec;
	unordered_map<float, double> per_score_size;
	map<float, double> outcome_counts;
	unordered_map<float, vector<int>> outcome_idx, pred_idx;
	vector<float> weights;
	int idx = 0;
	double logloss = 0, logloss_sz = 0;
	while (iterator->fetch_next(thread_num, y, pred, w))
	{
		has_weights = has_weights || w >= 0;

		double diff = abs(pred - y);
		if (has_weights) {
			tot_loss += diff * diff * w;
			sum_outcome += static_cast<double>(y) * static_cast<double>(w);
			tot_count += w;
			second_moment += static_cast<double>(y) * static_cast<double>(y) * static_cast<double>(w);

			per_score[pred] += y * w;
			per_score_sec[pred] += y * y * w;
			per_score_size[pred] += w;
			outcome_counts[y] += w;
			per_diff[diff] += w;

			weights.push_back(w);
			if (params->do_logloss && y >= 0 && y <= 1) {
				logloss += w * (y*log(min(params->epsilon, (double)pred)) + (1 - y)*log(min(params->epsilon, 1 - (double)pred)));
				logloss_sz += w;
			}
		}
		else {
			tot_loss += diff * diff;
			sum_outcome += y;
			++tot_count;
			second_moment += static_cast<double>(y) * static_cast<double>(y);

			per_score[pred] += y;
			per_score_sec[pred] += y * y;
			++per_score_size[pred];
			++outcome_counts[y];
			++per_diff[diff];

			if (params->do_logloss && y >= 0 && y <= 1) {
				logloss += y * log(min(params->epsilon, (double)pred)) + (1 - y)*log(min(params->epsilon, 1 - (double)pred));
				++logloss_sz;
			}
		}

		outcome_idx[y].push_back(idx);
		pred_idx[pred].push_back(idx);
		++idx;
	}
	if (tot_count <= 0)
		MTHROW_AND_ERR("Error - empty test\n");
	tot_loss /= tot_count;
	tot_mse = tot_loss;
	tot_loss = sqrt(tot_loss);

	double prior = sum_outcome / tot_count;
	//cum sum - will count falses:
	/*double tot_sum_y = 0;
	for (auto &it : outcome_counts)
	{
	double curr_w = it.second;
	it.second = tot_sum_y;
	tot_sum_y += curr_w;
	}*/

	map<float, double>::iterator it_outcome = outcome_counts.begin();
	double total_y_below_th = 0; ///< can be called False
	double total_pred_below_th = 0; ///< can be called Negative
	vector<bool> true_positive(idx, true);
	//float lowest_score = per_score.begin()->first;
	double tot_true_positive = tot_count;
	//first update - unmark true_positive that thier outcome is lower than lowest score:
	//double max_tot_count = tot_true_positive;

	for (const auto &it : per_score)
	{
		if (per_score_size.find(it.first) == per_score_size.end()
			|| per_score_size.at(it.first) <= 0)
			continue;

		while (it_outcome != outcome_counts.end() && it_outcome->first < it.first) {
			total_y_below_th += it_outcome->second;
			//update true_positive with outcomes that are not true now
			const vector<int> &to_check = outcome_idx.at(it_outcome->first);
			update_vec_idx(true_positive, tot_true_positive, weights, to_check);
			++it_outcome;
		}
		//Update true positive init (by outcome with lower than cutoff):

		double total_y_equal_above_th = tot_count - total_y_below_th; ///< can be called True
		double total_pred_equal_above_th = tot_count - total_pred_below_th; ///< can be called Positive

																			//Update False & Positive: False & Positive := Positive - True&Positive (disjoint groups)
		double tot_false_positive = total_pred_equal_above_th - tot_true_positive;
		//double tot_true_negative = total_y_equal_above_th - tot_true_positive;
		//double tot_false_negative = total_y_below_th - tot_false_positive;

		double real_prob = MED_MAT_MISSING_VALUE;
		double diff = MED_MAT_MISSING_VALUE;
		double real_obs_std = MED_MAT_MISSING_VALUE;
		if (per_score_size.at(it.first) > 0) {
			real_prob = it.second / per_score_size.at(it.first);
			diff = abs(it.first - real_prob);
			if (per_score_sec.at(it.first) / per_score_size.at(it.first) > real_prob * real_prob)
				real_obs_std = sqrt(per_score_sec.at(it.first) / per_score_size.at(it.first) - real_prob * real_prob);
		}

		res[format_working_point("MEAN_OBSERVED_VALUE@PREDICTED_VALUE", it.first, false)] = real_prob;
		res[format_working_point("MEAN_DIFF@PREDICTED_VALUE", it.first, false)] = diff;
		res[format_working_point("MEAN_POPULATION_SIZE@PREDICTED_VALUE", it.first, false)] = per_score_size.at(it.first);
		res[format_working_point("STD_OBSERVED_VALUE@PREDICTED_VALUE", it.first, false)] = real_obs_std;

		res[format_working_point("SENS@PREDICTED_VALUE", it.first, false)] = MED_MAT_MISSING_VALUE;
		res[format_working_point("FPR@PREDICTED_VALUE", it.first, false)] = MED_MAT_MISSING_VALUE;
		res[format_working_point("PPV@PREDICTED_VALUE", it.first, false)] = MED_MAT_MISSING_VALUE;
		if (total_y_equal_above_th > 0)
			res[format_working_point("SENS@PREDICTED_VALUE", it.first, false)] = 100 * tot_true_positive / total_y_equal_above_th;
		if (total_pred_equal_above_th > 0)
			res[format_working_point("PPV@PREDICTED_VALUE", it.first, false)] = 100 * tot_true_positive / total_pred_equal_above_th;
		if (total_y_below_th > 0)
			res[format_working_point("FPR@PREDICTED_VALUE", it.first, false)] = 100 * tot_false_positive / total_y_below_th;

		//Update True & Positive: based on outcome_idx.at(it.first) if exists, pred_idx.at(it.first) indexes that are now "excluded"
		if (outcome_idx.find(it.first) != outcome_idx.end()) {
			const vector<int> &to_check = outcome_idx.at(it.first);
			update_vec_idx(true_positive, tot_true_positive, weights, to_check);
		}
		const vector<int> &to_check = pred_idx.at(it.first);
		update_vec_idx(true_positive, tot_true_positive, weights, to_check);
		//update total_pred_below_th
		total_pred_below_th += per_score_size.at(it.first);
	}


	double loss_prior = second_moment - 2 * prior * sum_outcome + prior * prior * tot_count;
	loss_prior /= tot_count;
	float R2 = 1 - (tot_mse / loss_prior);

	//calc calibration index
	res["RMSE"] = tot_loss;
	res["R2"] = R2;
	if (params->do_logloss) {
		res["LOGLOSS"] = MED_MAT_MISSING_VALUE;
		res["LOGLOSS_SIZE"] = logloss_sz;
		res["TOTAL_SIZE"] = tot_count;
		if (logloss_sz > 0)
			res["LOGLOSS"] = logloss;
	}

	//Add mesaurements of cutoffs: use per_diff, prior, tot_count
	sort(params->coverage_quantile_percentages.begin(), params->coverage_quantile_percentages.end());
	auto it_diff = per_diff.begin();
	double coverage_weight = 0;
	for (float cutoff : params->coverage_quantile_percentages)
	{
		double score_threshold = cutoff / 100 * abs(prior);
		//stop at diff > score_threshold. Advance till that
		while (it_diff != per_diff.end()) {
			float curr_diff = it_diff->first;
			double curr_count_w = it_diff->second;
			if (curr_diff > score_threshold)
				break;
			//below threshold - count this and advance
			coverage_weight += curr_count_w;
			++it_diff;
		}

		//now we are passed the threshold. let's calc coverage:
		float coverage_res = 100 * coverage_weight / tot_count;

		/**
		Counts how much in percentage [0-100] in the data points are "covered" (the L1 error is within threshold)
		The threshold is determined by percentages for mean outcome (the prior)
		*/
		res[format_working_point("COVERAGE@DIFF_THRESHOLD", cutoff, false)] = coverage_res;
	}

	return res;
}

#pragma endregion

#pragma region Cohort Functions
bool time_range_filter(float outcome, int min_time, int max_time, int time, int outcome_time) {
	if (med_time.YearsMonths2Days.empty())
		med_time.init_time_tables();
	int diff_days = (med_time.convert_times(global_default_time_unit, MedTime::Days, outcome_time) -
		med_time.convert_times(global_default_time_unit, MedTime::Days, time));
	return ((outcome > 0 && diff_days >= min_time && diff_days <= max_time) ||
		(outcome <= 0 && diff_days > max_time));
}
bool time_range_filter(float outcome, float min_time, float max_time, float diff_days) {
	return ((outcome > 0 && diff_days >= min_time && diff_days <= max_time) ||
		(outcome <= 0 && diff_days >= max_time));
}

bool filter_range_param(const map<string, vector<float>> &record_info, int index, void *cohort_params) {
	Filter_Param *param = (Filter_Param *)cohort_params; //can't be null
	if (param->param_name != "Time-Window")
		return record_info.at(param->param_name)[index] >= param->min_range &&
		record_info.at(param->param_name)[index] <= param->max_range;
	else
		return time_range_filter(record_info.at("Label")[index] > 0, param->min_range,
			param->max_range, record_info.at(param->param_name)[index]);
}

bool filter_range_params(const map<string, vector<float>> &record_info, int index, void *cohort_params) {
	vector<Filter_Param> *param = (vector<Filter_Param> *)cohort_params; //can't be null
	bool res = true;
	int i = 0;
	while (res && i < (*param).size()) {
		if ((*param)[i].param_name != "Time-Window")
			res = record_info.at((*param)[i].param_name)[index] >= (*param)[i].min_range &&
			record_info.at((*param)[i].param_name)[index] <= (*param)[i].max_range;
		else
			res = time_range_filter(record_info.at("Label")[index] > 0, (*param)[i].min_range,
			(*param)[i].max_range, record_info.at((*param)[i].param_name)[index]);
		++i;
	}
	return res;
}
#pragma endregion

#pragma region Process Measurement Param Functions
void count_stats(int bin_counts, const vector<float> &y, const map<string, vector<float>> &additional_info
	, const vector<int> &filtered_indexes, const ROC_Params *params,
	vector<vector<double>> &male_counts, vector<vector<double>> &female_counts) {

	male_counts.resize(bin_counts);
	female_counts.resize(bin_counts);
	for (size_t i = 0; i < male_counts.size(); ++i)
		male_counts[i].resize(2);
	for (size_t i = 0; i < female_counts.size(); ++i)
		female_counts[i].resize(2);
	int min_age = (int)params->inc_stats.min_age;
	int max_age = (int)params->inc_stats.max_age;
	//if filtered_indexes is empty pass on all y. otherwise traverse over indexes:
	if (filtered_indexes.empty()) {
		for (size_t i = 0; i < y.size(); ++i)
		{
			if (additional_info.at("Age")[i] < min_age ||
				additional_info.at("Age")[i] >= max_age + params->inc_stats.age_bin_years)
				continue; //skip out of range or already case
			int age_index = (int)floor((additional_info.at("Age")[i] - min_age) /
				params->inc_stats.age_bin_years);
			if (age_index >= bin_counts)
				age_index = bin_counts - 1;

			if (additional_info.at("Gender")[i] == GENDER_MALE)  //Male
				++male_counts[age_index][y[i] > 0];
			else //Female
				++female_counts[age_index][y[i] > 0];
		}
	}
	else {
		for (size_t ii = 0; ii < filtered_indexes.size(); ++ii)
		{
			int i = filtered_indexes[ii];
			if (additional_info.at("Age")[i] < min_age ||
				additional_info.at("Age")[i] >= max_age + params->inc_stats.age_bin_years)
				continue; //skip out of range or already case
			int age_index = (int)floor((additional_info.at("Age")[i] - min_age) /
				params->inc_stats.age_bin_years);
			if (age_index >= bin_counts)
				age_index = bin_counts - 1;

			if (additional_info.at("Gender")[i] == GENDER_MALE)  //Male
				++male_counts[age_index][y[i] > 0];
			else //Female
				++female_counts[age_index][y[i] > 0];
		}
	}
}

void bin_calc_inc(const vector<double> &general_counts,
	const vector<double> &filtered_counts, const vector<double> &all_counts
	, double &tot_population, double &incidence_fix) {
	//Assume general_counts - doesn't have zeros. checked before

	if (all_counts[1] <= 0) {
		tot_population += filtered_counts[0] + filtered_counts[1]; //1 should be zero also
		return; //adds zero contribution to incidence - (like adding 0 incidence rate)
	}

	double general_or = general_counts[1] / general_counts[0];
	tot_population += filtered_counts[0] + filtered_counts[1];
	double current_bin_inc = 1;

	if (all_counts[0] > 0) {
		double all_or = all_counts[1] / all_counts[0];
		if (filtered_counts[0] > 0) {
			double filtered_or = filtered_counts[1] / filtered_counts[0];
			double or_ratio = filtered_or / all_or;
			double general_or_fix = or_ratio * general_or;
			current_bin_inc = general_or_fix / (1.0 + general_or_fix);
		}
	}
	else //Maybe skip those records?
		current_bin_inc = general_or / (1.0 + general_or);

	//weighted average
	incidence_fix += (filtered_counts[0] + filtered_counts[1]) * current_bin_inc;

}

void fix_cohort_sample_incidence(const map<string, vector<float>> &additional_info,
	const vector<float> &y, const vector<int> &pids, Measurement_Params *function_params,
	const vector<int> &filtered_indexes, const vector<float> &y_full, const vector<int> &pids_full) {
	ROC_And_Filter_Params *pr_full = (ROC_And_Filter_Params *)function_params;
	ROC_Params *params = pr_full->roc_params;
	vector<Filter_Param> *cohort_filt = pr_full->filter;

	if (params->inc_stats.sorted_outcome_labels.empty())
		return; //no inc file
	//calculating the "fixed" incidence in the cohort giving the true inc. in the general population
	// select cohort - and multiply in the given original incidence
	if (params->inc_stats.sorted_outcome_labels.size() != 2)
		MTHROW_AND_ERR("Category outcome aren't supported for now\n");
	if (additional_info.find("Age") == additional_info.end() || additional_info.find("Gender") == additional_info.end())
		MTHROW_AND_ERR("Age or Gender Signals are missings\n");

	int min_age = (int)params->inc_stats.min_age;
	int max_age = (int)params->inc_stats.max_age;
	int bin_counts = (int)floor((max_age - min_age) / params->inc_stats.age_bin_years);
	if (bin_counts * params->inc_stats.age_bin_years >
		(max_age - min_age) + 0.5)
		++bin_counts; //has at least 0.5 years for last bin to create it
	if (params->inc_stats.male_labels_count_per_age.size() != bin_counts)
		MTHROW_AND_ERR("Male vector has %d members. and need to have %d members\n",
		(int)params->inc_stats.male_labels_count_per_age.size(), bin_counts);
	for (int i = 0; i < bin_counts; ++i) {
		if (params->inc_stats.male_labels_count_per_age[i][0] <= 0)
			MTHROW_AND_ERR("Males Age bin %d can't have 0 controls\n",
				int(params->inc_stats.min_age + i * params->inc_stats.age_bin_years));
		if (params->inc_stats.female_labels_count_per_age[i][0] <= 0)
			MTHROW_AND_ERR("Females Age bin %d can't have 0 controls\n",
				int(params->inc_stats.min_age + i * params->inc_stats.age_bin_years));
	}


	vector<vector<double>> filtered_male_counts, filtered_female_counts;
	vector<vector<double>> all_male_counts, all_female_counts;
	count_stats(bin_counts, y_full, additional_info, filtered_indexes, params,
		filtered_male_counts, filtered_female_counts);
	//always filter Time-Window:
	vector<int> baseline_all;
	Filter_Param *time_win_cond = NULL;
	for (auto i = 0; i < cohort_filt->size() && time_win_cond == NULL; ++i)
		if ((*cohort_filt)[i].param_name == "Time-Window")
			time_win_cond = &(*cohort_filt)[i];
	if (time_win_cond != NULL)
		for (size_t i = 0; i < y_full.size(); ++i)
			if (filter_range_param(additional_info, (int)i, time_win_cond))
				baseline_all.push_back((int)i);
	count_stats(bin_counts, y_full, additional_info, baseline_all, params,
		all_male_counts, all_female_counts);

	//Lets calc the ratio for the incidence in the filter:
	params->incidence_fix = 0;
	double tot_population = 0;
	//test for problems:
	for (int i = 0; i < bin_counts; ++i) {
		if (all_male_counts[i][1] <= 0)
			MWARN("Warning fix_cohort_sample_incidence :: incidence - Males Age %d is empty of cases and will be counted as incidence rate of 0.0\n",
				int(params->inc_stats.min_age + i * params->inc_stats.age_bin_years));
		if (all_female_counts[i][1] <= 0)
			MWARN("Warning fix_cohort_sample_incidence :: incidence - Females Age %d is empty of cases and will be counted as incidence rate of 0.0\n",
				int(params->inc_stats.min_age + i * params->inc_stats.age_bin_years));
		if (all_male_counts[i][0] <= 0)
			MWARN("Warning fix_cohort_sample_incidence :: incidence - Males Age %d is empty of controls and will be counted as incidence rate of 1.0\n",
				int(params->inc_stats.min_age + i * params->inc_stats.age_bin_years));
		if (all_female_counts[i][0] <= 0)
			MWARN("Warning fix_cohort_sample_incidence :: incidence - Females Age %d is empty of controls and will be counted as incidence rate of 1.0\n",
				int(params->inc_stats.min_age + i * params->inc_stats.age_bin_years));
	}
	//recalc new ratio of #1/(#1+#0) and fix stats
	for (int i = 0; i < bin_counts; ++i)
	{
		//MALE calc
		bin_calc_inc(params->inc_stats.male_labels_count_per_age[i], filtered_male_counts[i],
			all_male_counts[i], tot_population, params->incidence_fix);

		//FEMALE calc
		bin_calc_inc(params->inc_stats.female_labels_count_per_age[i], filtered_female_counts[i],
			all_female_counts[i], tot_population, params->incidence_fix);
	}

	if (tot_population > 0)
		params->incidence_fix /= tot_population;

	MLOG_D("Running fix_cohort_sample_incidence and got %2.4f%% mean incidence\n",
		100 * params->incidence_fix);
}

void fix_cohort_sample_incidence_old(const map<string, vector<float>> &additional_info,
	const vector<float> &y, const vector<int> &pids, Measurement_Params *function_params,
	const vector<int> &filtered_indexes, const vector<float> &y_full, const vector<int> &pids_full) {
	ROC_And_Filter_Params *pr_full = (ROC_And_Filter_Params *)function_params;
	ROC_Params *params = pr_full->roc_params;
	if (params->inc_stats.sorted_outcome_labels.empty())
		return; //no inc file
				//calculating the "fixed" incidence in the cohort giving the true inc. in the general population
				// select cohort - and multiply in the given original incidence
	if (params->inc_stats.sorted_outcome_labels.size() != 2)
		MTHROW_AND_ERR("Category outcome aren't supported for now\n");
	if (additional_info.find("Age") == additional_info.end() || additional_info.find("Gender") == additional_info.end())
		MTHROW_AND_ERR("Age or Gender Signals are missings\n");

	int bin_counts = (int)floor((params->inc_stats.max_age - params->inc_stats.min_age) / params->inc_stats.age_bin_years);
	if (bin_counts * params->inc_stats.age_bin_years >
		(params->inc_stats.max_age - params->inc_stats.min_age) + 0.5)
		++bin_counts; //has at least 0.5 years for last bin to create it
	if (params->inc_stats.male_labels_count_per_age.size() != bin_counts)
		MTHROW_AND_ERR("Male vector has %d members. and need to have %d members\n",
		(int)params->inc_stats.male_labels_count_per_age.size(), bin_counts);

	vector<vector<double>> filtered_male_counts, filtered_female_counts;
	count_stats(bin_counts, y_full, additional_info, filtered_indexes, params,
		filtered_male_counts, filtered_female_counts);

	params->incidence_fix = 0;
	double tot_controls = 0;
	//recalc new ratio of #1/(#1+#0) and fix stats
	for (size_t i = 0; i < bin_counts; ++i)
	{
		//Males:
		if (filtered_male_counts[i][0] > 0) {
			double general_inc = params->inc_stats.male_labels_count_per_age[i][1] /
				(params->inc_stats.male_labels_count_per_age[i][1] +
					params->inc_stats.male_labels_count_per_age[i][0]);
			tot_controls += filtered_male_counts[i][0];
			params->incidence_fix += filtered_male_counts[i][0] * general_inc;
		}
		//Females:
		if (filtered_female_counts[i][0] > 0) {
			double general_inc = params->inc_stats.female_labels_count_per_age[i][1] /
				(params->inc_stats.female_labels_count_per_age[i][1] +
					params->inc_stats.female_labels_count_per_age[i][0]);
			tot_controls += filtered_female_counts[i][0];
			params->incidence_fix += filtered_female_counts[i][0] * general_inc;
		}
	}

	if (tot_controls > 0)
		params->incidence_fix /= tot_controls;

	MLOG_D("Running fix_cohort_sample_incidence and got %2.4f%% mean incidence\n",
		100 * params->incidence_fix);
}
#pragma endregion

#pragma region Process Scores Functions
void _simple_find(const vector<pair<int, int>> &vec, int &found_pos, int search_pos) {
	found_pos = -1;
	for (int j = (int)vec.size() - 1; j >= 0 && found_pos == -1; --j)
		if (vec[j].second >= search_pos && vec[j].first <= search_pos)
			found_pos = j;
}

void merge_down(vector<int> &ind_to_size, vector<vector<pair<int, int>>> &size_to_ind, set<int> &sizes,
	const pair<int, int> *index_to_merge) {
	pair<int, int> *merge_into = NULL;
	int to_merge_size = ind_to_size[index_to_merge->first - 1];
	int erase_index = -1;
	//remove index_to_merge.first - 1:
	_simple_find(size_to_ind[to_merge_size], erase_index, index_to_merge->first - 1);
	if (erase_index == -1)
		MTHROW_AND_ERR("down: Bug couldn't found merge_into\n");
	merge_into = &size_to_ind[to_merge_size][erase_index];

	int new_size = *sizes.begin() + to_merge_size;
	sizes.insert(new_size);
	//update in min,max:
	ind_to_size[merge_into->first] = new_size;
	ind_to_size[index_to_merge->second] = new_size;
	ind_to_size[merge_into->second] = new_size;
	ind_to_size[index_to_merge->first] = new_size;
	//erase old one
	int first_pos = merge_into->first;
	int second_pos = index_to_merge->second;
	//already popd merged_into element
	size_to_ind[to_merge_size].erase(size_to_ind[to_merge_size].begin() + erase_index);

	//insert new union
	if (size_to_ind.size() <= new_size) {
		//MWARN("Warn - size_to_ind reach end\n");
		size_to_ind.resize(new_size + 1);
	}
	size_to_ind[new_size].push_back(pair<int, int>(first_pos, second_pos));
}
void merge_up(vector<int> &ind_to_size, vector<vector<pair<int, int>>> &size_to_ind, set<int> &sizes,
	const pair<int, int> *index_to_merge) {
	//merge with +1
	pair<int, int> *merge_into = NULL;
	int to_merge_size = ind_to_size[index_to_merge->second + 1];
	int erase_index = -1;
	//remove index_to_merge.second + 1:
	_simple_find(size_to_ind[to_merge_size], erase_index, index_to_merge->second + 1);
	if (erase_index == -1)
		MTHROW_AND_ERR("up: Bug couldn't found merge_into\n");
	merge_into = &size_to_ind[to_merge_size][erase_index];

	int new_size = *sizes.begin() + to_merge_size;
	sizes.insert(new_size);
	//update in min,max:
	ind_to_size[index_to_merge->first] = new_size;
	ind_to_size[merge_into->second] = new_size;
	ind_to_size[index_to_merge->second] = new_size;
	ind_to_size[merge_into->first] = new_size;
	//erase old one:
	int first_pos = index_to_merge->first;
	int second_pos = merge_into->second;
	//already popd merged_into element
	size_to_ind[to_merge_size].erase(size_to_ind[to_merge_size].begin() + erase_index);

	//insert new union set:
	if (size_to_ind.size() <= new_size) {
		//MWARN("Warn - size_to_ind reach end\n");
		size_to_ind.resize(new_size + 1);
	}
	size_to_ind[new_size].push_back(pair<int, int>(first_pos, second_pos));
}

void preprocess_bin_scores(vector<float> &preds, Measurement_Params *function_params) {
	ROC_Params params;
	if (function_params != NULL)
		params = *(ROC_Params *)function_params;
	else
		return;

	if (params.use_score_working_points)
		return;

	if (params.score_resolution != 0)
		for (size_t i = 0; i < preds.size(); ++i)
			preds[i] = (float)round((double)preds[i] / params.score_resolution) *
			params.score_resolution;

	unordered_map<float, vector<int>> thresholds_indexes;
	vector<float> unique_scores;
	for (size_t i = 0; i < preds.size(); ++i)
		thresholds_indexes[preds[i]].push_back((int)i);
	unique_scores.resize((int)thresholds_indexes.size());
	int ind_p = 0, min_size = -1;
	for (auto it = thresholds_indexes.begin(); it != thresholds_indexes.end(); ++it)
	{
		unique_scores[ind_p] = it->first;
		++ind_p;
		if (min_size == -1 || min_size > it->second.size())
			min_size = (int)it->second.size();
	}
	sort(unique_scores.begin(), unique_scores.end());
	int bin_size_last = (int)thresholds_indexes.size();
	if (params.score_bins > 0 && bin_size_last < 10) {
		if (params.score_resolution != 0) {
			if (params.show_warns) {
				MWARN("Warnning Bootstrap:: requested specific working points, but score vector"
					" is highly quantitize(%d). try canceling preprocess_score by "
					"score_resolution, score_bins. Will use score working points\n",
					bin_size_last);
			}
		}
		else {
			if (params.show_warns)
				MWARN("Warnning Bootstrap:: requested specific working points, but score vector"
					" is highly quantitize(%d). Will use score working points\n",
					bin_size_last);
		}
	}

	if ((params.score_bins > 0 && bin_size_last > params.score_bins) ||
		(params.score_min_samples > 0 && min_size < params.score_min_samples)) {
		int c = 0;
		vector<vector<pair<int, int>>> size_to_ind(preds.size()); //size, group, index_min_max
		vector<int> ind_to_size(bin_size_last);
		set<int> sizes;
		for (auto it = unique_scores.begin(); it != unique_scores.end(); ++it)
		{
			size_to_ind[(int)thresholds_indexes[*it].size()].push_back(pair<int, int>(c, c));
			ind_to_size[c] = (int)thresholds_indexes[*it].size();
			++c;
			sizes.insert((int)thresholds_indexes[*it].size());
		}

		while (((params.score_bins > 0 && bin_size_last > params.score_bins)
			|| (params.score_min_samples > 0 && *sizes.begin() < params.score_min_samples)) &&
			bin_size_last > 1) {
			min_size = *sizes.begin();
			if (size_to_ind[min_size].empty())
				MTHROW_AND_ERR("Bug couldn't found min_size=%d\n", min_size);

			pair<int, int> index_to_merge = size_to_ind[min_size].back();
			size_to_ind[min_size].pop_back(); //now popback
			//merge index_to_merge with index_to_merge+-1. and update size_to_ind, ind_to_size, sizes
			if (index_to_merge.second == unique_scores.size() - 1)
				merge_down(ind_to_size, size_to_ind, sizes, &index_to_merge);
			else if (index_to_merge.first == 0)
				merge_up(ind_to_size, size_to_ind, sizes, &index_to_merge);
			else {
				//MLOG("DEBUG: %d,%d\n", index_to_merge.first, index_to_merge.second);
				if (ind_to_size[index_to_merge.second + 1] < ind_to_size[index_to_merge.first - 1])
					merge_up(ind_to_size, size_to_ind, sizes, &index_to_merge);
				else
					merge_down(ind_to_size, size_to_ind, sizes, &index_to_merge);
			}

			while (size_to_ind[min_size].empty()) {//erase if left empty after merge
				sizes.erase(sizes.begin());
				min_size = *sizes.begin();
			}
			--bin_size_last;
		}

		//update thresholds_indexes based on: size_to_ind groups -
		//merge all indexes in each group to first index in thresholds_indexes. "mean" other scores to unique_scores
		unordered_set<float> u_scores;
		for (auto it = sizes.begin(); it != sizes.end(); ++it)
		{
			for (size_t k = 0; k < size_to_ind[*it].size(); ++k)
			{ //merge from first => second
				pair<int, int> *merge = &size_to_ind[*it][k];
				double mean_score = 0, tot_cnt = 0;
				vector<int> merged_inds;
				for (int ii = merge->first; ii <= merge->second; ++ii) {
					mean_score += static_cast<double>(unique_scores[ii]) * static_cast<double>(thresholds_indexes[unique_scores[ii]].size());
					tot_cnt += thresholds_indexes[unique_scores[ii]].size();
					merged_inds.insert(merged_inds.end(),
						thresholds_indexes[unique_scores[ii]].begin(), thresholds_indexes[unique_scores[ii]].end());
				}
				mean_score /= tot_cnt;
				//update all preds to mean_score in merged_inds:
				for (int ind : merged_inds)
					preds[ind] = (float)mean_score;
				u_scores.insert((float)mean_score);
			}
		}
		if (u_scores.size() < 10) {
			if (params.show_warns)
				MWARN("Warnning Bootstrap:: requested specific working points, but score vector"
					" is highly quantitize(%d). try canceling preprocess_score by "
					"score_resolution, score_bins. Will use score working points\n",
					(int)u_scores.size());
		}
	}

	MLOG_D("Preprocess_bin_scores Done - left with %d bins!\n", bin_size_last);
}
#pragma endregion

#pragma region Parameter Functions
Measurement_Params::Measurement_Params() {
	show_warns = true;
}
int Filter_Param::init_from_string(string init_string) {
	if (init_string.find(':') == string::npos)
		MTHROW_AND_ERR("Wrong format given \"%s\". expected format is \"PARAM_NAME:min_range,max_range\"\n",
			init_string.c_str());
	param_name = init_string.substr(0, init_string.find(':'));
	string rest = init_string.substr(init_string.find(':') + 1);
	if (rest.find(',') == string::npos)
		MTHROW_AND_ERR("Wrong format given \"%s\". expected format is \"PARAM_NAME:min_range,max_range\"\n",
			init_string.c_str());
	min_range = stof(rest.substr(0, rest.find(',')));
	max_range = stof(rest.substr(rest.find(',') + 1));
	return 0;
}
Filter_Param::Filter_Param(const string &init_string) {
	init_from_string(init_string);
}
int Filter_Param::init(map<string, string>& map) {
	for (auto it = map.begin(); it != map.end(); ++it)
	{
		//! [Filter_Param::init]
		if (it->first == "param_name")
			param_name = it->second;
		else if (it->first == "min_range")
			min_range = stof(it->second);
		else if (it->first == "max_range")
			max_range = stof(it->second);
		//! [Filter_Param::init]
		else
			MTHROW_AND_ERR("Error in Filter_Param::init - unknown parameter \"%s\"\n",
				it->first.c_str());
	}
	return 0;
}
void Incident_Stats::write_to_text_file(const string &text_file) {
	ofstream fw(text_file);
	if (!fw.good())
		MTHROW_AND_ERR("IO Error: can't write \"%s\"\n", text_file.c_str());
	string delim = "\t";
	fw << "AGE_BIN" << delim << age_bin_years << endl;
	fw << "AGE_MIN" << delim << min_age << endl;
	fw << "AGE_MAX" << delim << max_age << endl;
	for (size_t i = 0; i < sorted_outcome_labels.size(); ++i)
		fw << "OUTCOME_VALUE" << delim << sorted_outcome_labels[i] << "\n";
	fw.flush();
	for (size_t i = 0; i < male_labels_count_per_age.size(); ++i)
		for (size_t j = 0; j < male_labels_count_per_age[i].size(); ++j)
			fw << "STATS_ROW" << delim << "MALE" << delim << min_age + i * age_bin_years
			<< delim << sorted_outcome_labels[j] << delim << male_labels_count_per_age[i][j] << "\n";
	for (size_t i = 0; i < female_labels_count_per_age.size(); ++i)
		for (size_t j = 0; j < female_labels_count_per_age[i].size(); ++j)
			fw << "STATS_ROW" << delim << "FEMALE" << delim << min_age + i * age_bin_years
			<< delim << sorted_outcome_labels[j] << delim << female_labels_count_per_age[i][j] << "\n";
	fw.flush();
	fw.close();
}
void Incident_Stats::read_from_text_file(const string &text_file) {
	MLOG("Loading Incidence file %s\n", text_file.c_str());
	ifstream of(text_file);
	if (!of.good())
		MTHROW_AND_ERR("IO Error: can't read \"%s\"\n", text_file.c_str());
	string line;
	vector<vector<bool>> gender_read(2); // males, females
	while (getline(of, line)) {
		if (line.empty() || boost::starts_with(line, "#"))
			continue;
		vector<string> tokens;
		boost::split(tokens, line, boost::is_any_of("\t"));
		if (tokens.size() < 2)
			MTHROW_AND_ERR("Format Error: got line: \"%s\"\n", line.c_str());
		string command = tokens[0];
		if (command == "AGE_BIN")
			age_bin_years = stoi(tokens[1]);
		else if (command == "AGE_MIN") {
			min_age = stof(tokens[1]);
		}
		else if (command == "AGE_MAX")
			max_age = stof(tokens[1]);
		else if (command == "OUTCOME_VALUE")
			sorted_outcome_labels.push_back(stof(tokens[1]));
		else if (command == "STATS_ROW") {
			if (tokens.size() != 5)
				MTHROW_AND_ERR("Unknown lines format \"%s\"\n", line.c_str());
			float age = stof(tokens[2]);

			if (age < min_age || age> max_age) {
				MWARN("Warning:: skip age because out of range in line \"%s\"\n", line.c_str());
				continue;
			}
			int age_bin = (int)floor((age - min_age) / age_bin_years);
			int max_bins = (int)floor((max_age - min_age) / age_bin_years);
			if (max_bins * age_bin_years > (max_age - min_age) + 0.5)
				++max_bins;
			if (age_bin >= max_bins)
				age_bin = max_bins - 1;
			if (male_labels_count_per_age.empty()) {
				male_labels_count_per_age.resize(max_bins);
				gender_read[0].resize(max_bins);
				for (size_t i = 0; i < male_labels_count_per_age.size(); ++i)
					male_labels_count_per_age[i].resize((int)sorted_outcome_labels.size());
			}
			if (female_labels_count_per_age.empty()) {
				female_labels_count_per_age.resize(max_bins);
				gender_read[1].resize(max_bins);
				for (size_t i = 0; i < female_labels_count_per_age.size(); ++i)
					female_labels_count_per_age[i].resize((int)sorted_outcome_labels.size());
			}
			float outcome_val = stof(tokens[3]);
			int outcome_ind = (int)distance(sorted_outcome_labels.begin(),
				find(sorted_outcome_labels.begin(), sorted_outcome_labels.end(), outcome_val));
			if (outcome_ind > sorted_outcome_labels.size())
				MTHROW_AND_ERR("Couldn't find outcome_value=%2.3f\n", outcome_val);
			if (tokens[1] == "MALE") {
				male_labels_count_per_age[age_bin][outcome_ind] = stof(tokens[4]);
				gender_read[0][age_bin] = true;
			}
			else if (tokens[1] == "FEMALE") {
				female_labels_count_per_age[age_bin][outcome_ind] = stof(tokens[4]);
				gender_read[1][age_bin] = true;
			}
			else
				MTHROW_AND_ERR("Unknown gender \"%s\"\n", tokens[1].c_str());
		}
		else
			MTHROW_AND_ERR("Unknown command \"%s\"\n", command.c_str());
	}
	sort(sorted_outcome_labels.begin(), sorted_outcome_labels.end());
	of.close();
	//validate read all:
	int max_bins = (int)floor((max_age - min_age) / age_bin_years);
	for (size_t i = 0; i < max_bins; ++i)
	{
		if (!gender_read[0][i])
			MTHROW_AND_ERR("Error in reading inc stats file. missing bin num %zu of age %d (%f,%d) for males\n",
				i, int(min_age + i * age_bin_years), min_age, age_bin_years);
		if (!gender_read[1][i])
			MTHROW_AND_ERR("Error in reading inc stats file. missing bin num %zu of age %d for females\n",
				i, int(min_age + i * age_bin_years));
	}
}
void parse_vector(const string &value, vector<float> &output_vec) {
	vector<string> vec;
	boost::split(vec, value, boost::is_any_of(","));
	output_vec.resize((int)vec.size());
	for (size_t i = 0; i < vec.size(); ++i)
		output_vec[i] = stof(vec[i]);
}
void parse_vector(const string &value, vector<int> &output_vec) {
	vector<string> vec;
	boost::split(vec, value, boost::is_any_of(","));
	output_vec.resize((int)vec.size());
	for (size_t i = 0; i < vec.size(); ++i)
		output_vec[i] = stoi(vec[i]);
}
int ROC_Params::init(map<string, string>& map) {
	for (auto it = map.begin(); it != map.end(); ++it)
	{
		const string &param_name = boost::to_lower_copy(it->first);
		const string &param_value = it->second;

		//! [ROC_Params::init]
		if (param_name == "max_diff_working_point")
			max_diff_working_point = stof(param_value);
		else if (param_name == "use_score_working_points")
			use_score_working_points = stoi(param_value) > 0;
		else if (param_name == "fix_label_to_binary")
			fix_label_to_binary = stoi(param_value) > 0;
		else if (param_name == "score_bins")
			score_bins = stoi(param_value);
		else if (param_name == "score_min_samples")
			score_min_samples = stoi(param_value);
		else if (param_name == "score_resolution")
			score_resolution = stof(param_value);
		else if (param_name == "inc_stats_text")
			inc_stats.read_from_text_file(param_value);
		else if (param_name == "inc_stats_bin")
			inc_stats.read_from_file(param_value);
		else if (param_name == "working_point_fpr")
			parse_vector(param_value, working_point_FPR);
		else if (param_name == "working_point_topn")
			parse_vector(param_value, working_point_TOPN);
		else if (param_name == "working_point_pr")
			parse_vector(param_value, working_point_PR);
		else if (param_name == "working_point_score")
			parse_vector(param_value, working_point_Score);
		else if (param_name == "working_point_sens")
			parse_vector(param_value, working_point_SENS);
		else if (param_name == "working_point_auc")
			parse_vector(param_value, working_point_auc);
		else if (param_name == "show_warns")
			show_warns = med_stoi(param_value) > 0;
		else if (param_name == "min_score_quants_to_force_score_wp")
			min_score_quants_to_force_score_wp = med_stoi(param_value);
		//! [ROC_Params::init]
		else
			MTHROW_AND_ERR("Unknown paramter \"%s\" for ROC_Params\n", param_name.c_str());
	}
	return 0;
}
ROC_Params::ROC_Params(const string &init_string) : ROC_Params() {
	init_from_string(init_string);
}

void Multiclass_Params::read_dist_matrix_from_file(const string& fileName) {

	ifstream inf(fileName);
	if (!inf)
		MTHROW_AND_ERR("Cannot open file \'%s\' for reading\n", fileName.c_str());

	string line;
	vector<string> fields;
	while (getline(inf, line)) {
		boost::split(fields, line, boost::is_any_of(","));
		vector<float> row;
		for (string& s : fields)
			row.push_back(stof(s));

		if ((!dist_matrix.empty()) && (row.size() != dist_matrix.back().size()))
			MTHROW_AND_ERR("Row size inconsistency in distance matrix from \'%s\'\n", fileName.c_str());
		dist_matrix.push_back(row);
	}

	if ((!dist_matrix.empty()) && (dist_matrix.size() != dist_matrix[0].size()))
		MTHROW_AND_ERR("Distance matrix is not square in \'%s\'\n", fileName.c_str());

	inf.close();
}

int Multiclass_Params::init(map<string, string>& map) {

	for (auto it = map.begin(); it != map.end(); ++it)
	{
		const string &param_name = boost::to_lower_copy(it->first);
		const string &param_value = it->second;

		//! [Multiclass_Params::init]
		if (param_name == "top_n") {
			vector<string> fields;
			boost::split(fields, param_value, boost::is_any_of(","));
			for (string& _n : fields)
				top_n.push_back(stoi(_n));
			sort(top_n.begin(), top_n.end());
		}
		else if (param_name == "n_categ")
			n_categ = stoi(param_value);
		else if (param_name == "dist") {
			dist_name = param_value;
			boost::to_upper(dist_name);
		}
		else if (param_name == "dist_matrix") {
			dist_file = param_value;
			read_dist_matrix_from_file(dist_file);
		}
		else if (param_name == "do_class_auc")
			do_class_auc = (param_value == "1" || param_value == "y" || param_value == "Y");
		//! [Multiclass_Params::init]
		else
			MTHROW_AND_ERR("Unknown paramter \"%s\" for Multiclass_Params\n", param_name.c_str());
	}

	// Distance
	// fill distance matrix by name
	if (dist_matrix.empty()) {
		if (dist_name == "JACCARD")
			medial::performance::get_jaccard_matrix(n_categ, dist_matrix);
		else if (dist_name == "UNIFORM") {
			dist_matrix.assign(n_categ, vector<float>(n_categ, 1.0));
			for (int i = 0; i < n_categ; i++)
				dist_matrix[i][i] = 0;
		}
	}

	if (dist_matrix.empty())
		MTHROW_AND_ERR("Cannot perform multi-class analysis without distance matrix (try JACCARD/UNIFORM)");

	// Update n_categ according to matrix
	if (n_categ != 1 && n_categ != dist_matrix.size())
		MTHROW_AND_ERR("n_categ (%d) and distance matrix (%d) are inconsistent\n", n_categ, (int)dist_matrix.size());
	n_categ = (int)dist_matrix.size();

	dist_weights.reserve(n_categ);
	for (size_t i = 0; i < n_categ; i++) {
		float sum = 0;
		for (size_t k = 0; k < n_categ; k++)
			sum += dist_matrix[i][k];
		dist_weights[i] = (float)(n_categ) / sum;
	}

	return 0;
}

Multiclass_Params::Multiclass_Params(const string &init_string) : Multiclass_Params() {
	init_from_string(init_string);
}

int Regression_Params::init(map<string, string>& mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		//! [Regression_Params::init]
		if (it->first == "do_logloss")
			do_logloss = med_stoi(it->second) > 0;
		else if (it->first == "epsilon")
			epsilon = stod(it->second);
		else if (it->first == "coverage_quantile_percentages")
			parse_vector(it->second, coverage_quantile_percentages);
		//! [Regression_Params::init]
		else
			MTHROW_AND_ERR("Error in Regression_Params::init - unknown param %s\n", it->first.c_str());
	}
	return 0;
}
#pragma endregion
