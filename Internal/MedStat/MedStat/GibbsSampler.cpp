#include "GibbsSampler.h"
#include <MedAlgo/MedAlgo/MedAlgo.h>
#include <MedAlgo/MedAlgo/BinSplitOptimizer.h>
#include <MedMat/MedMat/MedMatConstants.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <regex>
#include "medial_utilities/medial_utilities/globalRNG.h"

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDSTAT
#define LOCAL_LEVEL LOG_DEF_LEVEL

Gibbs_Params::Gibbs_Params() {
	kmeans = 0;
	selection_count = 0;
	max_iters = 500;
	selection_ratio = (float)1.0;

	select_with_repeats = false;
	calibration_save_ratio = (float)0.2;
	calibration_string = "calibration_type=isotonic_regression;verbose=0";

	predictor_type = "lightgbm";
	predictor_args = "objective=multiclass;metric=multi_logloss;verbose=0;num_threads=0;"
		"num_trees=100;learning_rate=0.05;lambda_l2=0;metric_freq=50";
	num_class_setup = "num_class";
	bin_settings.init_from_string("split_method=iterative_merge;min_bin_count=100;binCnt=100");
}

GibbsSamplingParams::GibbsSamplingParams() {
	burn_in_count = 1000;
	jump_between_samples = 10;
	samples_count = 1;
	find_real_value_bin = true;
}

int Gibbs_Params::init(map<string, string>& map) {

	for (auto it = map.begin(); it != map.end(); ++it)
	{
		if (it->first == "kmeans")
			kmeans = med_stoi(it->second);
		else if (it->first == "max_iters")
			max_iters = med_stoi(it->second);
		else if (it->first == "selection_ratio")
			selection_ratio = med_stof(it->second);
		else if (it->first == "selection_count")
			selection_count = med_stoi(it->second);
		else if (it->first == "select_with_repeats")
			select_with_repeats = med_stoi(it->second) > 0;
		else if (it->first == "predictor_type")
			predictor_type = it->second;
		else if (it->first == "predictor_args")
			predictor_args = it->second;
		else if (it->first == "num_class_setup")
			num_class_setup = it->second;
		else if (it->first == "bin_settings")
			bin_settings.init_from_string(it->second);
		else if (it->first == "calibration_save_ratio")
			calibration_save_ratio = med_stof(it->second);
		else if (it->first == "calibration_string")
			calibration_string = it->second;
		else
			MTHROW_AND_ERR("Error in Gibbs_Params::init - no parameter \"%s\"\n", it->first.c_str());
	}

	return 0;
}

int GibbsSamplingParams::init(map<string, string>& map) {
	for (auto it = map.begin(); it != map.end(); ++it)
	{
		if (it->first == "burn_in_count")
			burn_in_count = med_stoi(it->second);
		else if (it->first == "jump_between_samples")
			jump_between_samples = med_stoi(it->second);
		else if (it->first == "samples_count")
			samples_count = med_stoi(it->second);
		else if (it->first == "find_real_value_bin")
			find_real_value_bin = med_stoi(it->second) > 0;
		else
			MTHROW_AND_ERR("Error in GibbsSamplingParams::init - no parameter \"%s\"\n", it->first.c_str());
	}
	return 0;
}

template<typename T> PredictorOrEmpty<T>::PredictorOrEmpty() {
	predictor = NULL;
}

template<typename T> PredictorOrEmpty<T>::~PredictorOrEmpty() {
	if (predictor != NULL)
		delete predictor;
	predictor = NULL;
}

template<typename T> int GibbsSampler<T>::init(map<string, string>& map) {
	return params.init(map);
}

template<typename T> T PredictorOrEmpty<T>::get_sample(vector<T> &x, mt19937 &gen) {
	if (!sample_cohort.empty()) {
		uniform_int_distribution<> rnd_gen(0, (int)sample_cohort.size() - 1);
		int sel = rnd_gen(gen);
		return sample_cohort[sel];
	}
	else if (!cluster_centers.empty()) {
		//find closet cluster:
		int close_idx = -1;
		double min_diff = -1;
		int k = (int)cluster_centers.size() / input_size;

		for (size_t i = 0; i < k; ++i)
		{
			if (clusters_y[i].empty())
				continue; // skip empty clusters
			double curr_dif = 0;
			for (size_t j = 0; j < input_size; ++j)
				curr_dif += pow(cluster_centers[i * input_size + j] - x[j], 2);
			if (close_idx == -1 || min_diff > curr_dif) {
				min_diff = curr_dif;
				close_idx = (int)i;
			}
		}
		//the relevant cluster is close_idx - let's randomize y value from it:
		const vector<float> &sample_from = clusters_y[close_idx];
		uniform_int_distribution<> rnd_gen(0, (int)sample_from.size() - 1);
		int sel = rnd_gen(gen);
		return sample_from[sel];
	}
	else if (predictor != NULL) {
		vector<T> prd; //for each class:
		//predictor->predict(x, prd, 1, (int)x.size());
		predictor->predict_single(x, prd);
		if (!calibrators.empty()) {
			//need to use calibrator for all predictions:
			for (size_t i = 0; i < prd.size(); ++i)
				prd[i] = calibrators[i].Apply(prd[i]);
		}
		double tot_num = 0;
		for (size_t i = 0; i < prd.size(); ++i)
			tot_num += prd[i];
		uniform_real_distribution<> real_dist(0, tot_num);
		double sel = real_dist(gen);

		//now select correspond bin value:
		tot_num = 0;
		int sel_idx = 0;
		while (sel_idx < prd.size() && tot_num + prd[sel_idx] < sel) {
			tot_num += prd[sel_idx];
			++sel_idx;
		}

		return bin_vals[sel_idx];
	}

	MTHROW_AND_ERR("Error PredictorOrEmpty - not initialized");
}

template<typename T> GibbsSampler<T>::GibbsSampler() {
	_gen = mt19937(globalRNG::rand());
	done_prepare = false;
}

template<typename T> void GibbsSampler<T>::learn_gibbs(const map<string, vector<T>> &cohort_data) {
	vector<string> all(cohort_data.size());
	int ii = 0;
	for (const auto &it : cohort_data)
	{
		all[ii] = it.first;
		++ii;
	}
	learn_gibbs(cohort_data, all, false);
}

template<typename T> void GibbsSampler<T>::learn_gibbs(const map<string, vector<T>> &cohort_data, const vector<string> &learn_features, bool skip_missing) {
	mt19937 gen(globalRNG::rand());
	if (params.selection_ratio > 1) {
		MWARN("Warning - GibbsSampler::learn_gibbs - params.selection_ratio is bigger than 1 - setting to 1");
		params.selection_ratio = 1;
	}

	vector<string> all_names; all_names.reserve(cohort_data.size());
	for (auto it = cohort_data.begin(); it != cohort_data.end(); ++it)
		all_names.push_back(it->first);
	all_feat_names = all_names;
	impute_feat_names = learn_features;
	int cohort_size = (int)cohort_data.begin()->second.size(); //assume not empty


	feats_predictors.resize(learn_features.size());
	uniqu_value_bins.resize(learn_features.size());
	int pred_num_feats = (int)cohort_data.size() - 1;
	for (size_t i = 0; i < learn_features.size(); ++i)
		feats_predictors[i].input_size = pred_num_feats;
	if (pred_num_feats == 0) {
		for (size_t i = 0; i < learn_features.size(); ++i) {
			//just test for values as distribution mean, variance
			feats_predictors[i].sample_cohort = cohort_data.at(learn_features[i]);
		}
	}
	else {
		MedProgress progress("Learn_Gibbs", (int)learn_features.size(), 30, 1);

		int train_sz = int(cohort_size * params.selection_ratio);
		if (params.selection_count > 0 && train_sz > params.selection_count)
			train_sz = params.selection_count;

		if (params.kmeans > 0) {
			uniform_int_distribution<> rnd_num(0, cohort_size - 1);
#pragma omp parallel for
			for (int i = 0; i < all_names.size(); ++i)
			{
				// Find unique values of feature i
				unordered_set<float> uniq_vals;
				for (size_t k = 0; k < cohort_data.at(all_names[i]).size(); ++k)
					uniq_vals.insert(cohort_data.at(all_names[i])[k]);
#pragma omp critical
				{
					uniqu_value_bins[i].insert(uniqu_value_bins[i].end(), uniq_vals.begin(), uniq_vals.end());
					sort(uniqu_value_bins[i].begin(), uniqu_value_bins[i].end());
				}

				vector<int> clusters;
				vector<float> train_vec(static_cast<size_t>(train_sz) * static_cast<size_t>(pred_num_feats)), label_vec(train_sz);
				vector<bool> seen;
				if (!params.select_with_repeats)
					seen.resize(cohort_size);
				vector<int> sel_ls(train_sz);
				for (size_t ii = 0; ii < train_sz; ++ii) {
					int random_idx = rnd_num(gen);
					if (!params.select_with_repeats) { //if need to validate no repeats - do it
						while (seen[random_idx])
							random_idx = rnd_num(gen);
						seen[random_idx] = true;
					}
					for (size_t jj = 0; jj < pred_num_feats; ++jj) {
						int fixed_idx = (int)jj + int(jj >= i); //skip current
						train_vec[ii* pred_num_feats + jj] = float(cohort_data.at(all_names[fixed_idx])[random_idx]);
					}
					label_vec[ii] = float(cohort_data.at(all_names[i])[random_idx]);
					sel_ls[ii] = random_idx;
				}

				//seperate the X space to k clusters:
				int k = params.kmeans;
				if (INT_MAX / train_sz < k) {
					k = INT_MAX / train_sz - 1;
					MWARN("Warning: k=%d for kMeans is too large for that sample size of %d shrinking k to %d\n",
						params.kmeans, train_sz, k);
				}
				//vector<float> centers(k * pred_num_feats);
#pragma omp critical 
				{
					feats_predictors[i].cluster_centers.resize(static_cast<size_t>(k) * static_cast<size_t>(pred_num_feats));
					feats_predictors[i].clusters_y.resize(k);
				}
				vector<float> dists(static_cast<size_t>(k) * static_cast<size_t>(train_sz));
				clusters.resize(train_sz);
				//MLOG("Running kMeans for %s (%zu / %zu)\n", all_names[i].c_str(), i + 1, all_names.size());
				KMeans(train_vec.data(), train_sz, pred_num_feats, k, params.max_iters,
					feats_predictors[i].cluster_centers.data(), clusters.data(), dists.data(), false);
				//calc feats_predictors[i].clusters_y:
#pragma omp critical 
				for (size_t j = 0; j < train_sz; ++j)
					feats_predictors[i].clusters_y[clusters[j]].push_back(float(cohort_data.at(all_names[i])[j]));

			}
		}
		else {
			for (int i = 0; i < learn_features.size(); ++i)
			{
				int feat_idx = -1;
				//find index:
				for (size_t j = 0; j < all_names.size(); ++j)
					if (all_names[j] == learn_features[i])
						feat_idx = (int)j;
				if (feat_idx < 0)
					MTHROW_AND_ERR("Error in learn_gibbs can't find %s\n",
						learn_features[i].c_str());
				vector<T> full_vals = cohort_data.at(learn_features[i]);
				vector<int> sel_idx;
				if (skip_missing) {
					full_vals.clear();
					for (size_t j = 0; j < cohort_data.at(learn_features[i]).size(); ++j)
						if (cohort_data.at(learn_features[i])[j] != MED_MAT_MISSING_VALUE) {
							full_vals.push_back(cohort_data.at(learn_features[i])[j]);
							sel_idx.push_back((int)j);
						}
				}
				train_sz = (int)full_vals.size();
				cohort_size = train_sz;
				train_sz = int(train_sz * params.selection_ratio);
				if (params.selection_count > 0 && train_sz > params.selection_count)
					train_sz = params.selection_count;

				uniform_int_distribution<> rnd_num(0, cohort_size - 1);
				unordered_set<float> uniq_vals;
				for (size_t k = 0; k < full_vals.size(); ++k)
					uniq_vals.insert(full_vals[k]);
				uniqu_value_bins[i].insert(uniqu_value_bins[i].end(), uniq_vals.begin(), uniq_vals.end());
				sort(uniqu_value_bins[i].begin(), uniqu_value_bins[i].end());
				vector<int> clusters;
				vector<float> train_vec(static_cast<size_t>(train_sz) * static_cast<size_t>(pred_num_feats)), label_vec(train_sz);
				vector<bool> seen;
				if (!params.select_with_repeats)
					seen.resize(cohort_size);
				vector<int> sel_ls(train_sz);
				for (size_t ii = 0; ii < train_sz; ++ii) {
					int random_idx = rnd_num(gen);
					if (!params.select_with_repeats) { //if need to validate no repeats - do it
						while (seen[random_idx])
							random_idx = rnd_num(gen);
						seen[random_idx] = true;
					}
					if (!sel_idx.empty())
						random_idx = sel_idx[random_idx];
					for (size_t jj = 0; jj < pred_num_feats; ++jj) {
						int fixed_idx = (int)jj + int(jj >= feat_idx); //skip current
						train_vec[ii* pred_num_feats + jj] = float(cohort_data.at(all_names[fixed_idx])[random_idx]);
					}
					label_vec[ii] = float(cohort_data.at(learn_features[i])[random_idx]);
					sel_ls[ii] = random_idx;
				}

				//use predictors to train on train_vec and predcit on label_vec:
				//do binning for label_vec:
				vector<int> empt;
				medial::process::split_feature_to_bins(params.bin_settings, label_vec, empt, label_vec);
				//count num of classes:
				unordered_set<float> seen_val;
				for (size_t ii = 0; ii < label_vec.size(); ++ii)
					seen_val.insert(label_vec[ii]);
				int class_num = (int)seen_val.size();
				string predictor_init = params.predictor_args;
				//set num classes if needed:
				string empty_str = "";
				if (!params.num_class_setup.empty()) {
					//std::regex rgx(params.num_class_setup + "=[^;]+");
					//predictor_init = std::regex_replace(predictor_init, rgx, empty_str);
					//boost::replace_all(predictor_init, ";;", ";");
					predictor_init += ";" + params.num_class_setup + "=" + to_string(class_num);
					//change predictor_init
				}
				//init predictor
				MedPredictor *train_pred = MedPredictor::make_predictor(params.predictor_type, predictor_init);

				vector<float> sorted_vals(seen_val.begin(), seen_val.end());
				sort(sorted_vals.begin(), sorted_vals.end());
				MLOG("Feature %s has %d categories\n", learn_features[i].c_str(), class_num);

				feats_predictors[i].bin_vals.insert(feats_predictors[i].bin_vals.end(), sorted_vals.begin(), sorted_vals.end());
				//learn predictor
				//change labels to be 0 to K-1:
				unordered_map<float, int> map_categ;
				//calc by order:
				for (size_t ii = 0; ii < sorted_vals.size(); ++ii)
					map_categ[sorted_vals[ii]] = (int)ii;
				//commit:
				for (size_t ii = 0; ii < label_vec.size(); ++ii)
					label_vec[ii] = (float)map_categ.at(label_vec[ii]);

				//split to train and train_calibration if needed:
				if (params.calibration_save_ratio > 0) {
#pragma omp critical
					feats_predictors[i].calibrators.resize(seen_val.size());
					for (size_t kk = 0; kk < feats_predictors[i].calibrators.size(); ++kk)
						feats_predictors[i].calibrators[kk].init_from_string(params.calibration_string);

					int calib_ratio = params.calibration_save_ratio * (int)label_vec.size();
					int train_ratio = (int)label_vec.size() - calib_ratio;
					uniform_int_distribution<> sel_rnd(0, (int)label_vec.size() - 1);
					vector<bool> seen_sel(label_vec.size());
					vector<float> pred_train_vec(static_cast<size_t>(train_ratio) * static_cast<size_t>(pred_num_feats)), pred_label_vec(train_ratio);
					vector<vector<MedSample>> pred_calib_train(seen_val.size());
					MedFeatures pred_calib_mat;
					for (const string &name_feat : all_names)
						pred_calib_mat.attributes[name_feat].denorm_mean = 0;
					pred_calib_mat.samples.resize(calib_ratio);

					for (size_t j = 0; j < calib_ratio; ++j)
					{
						int sel_idx = sel_rnd(gen);
						while (seen_sel[sel_idx])
							sel_idx = sel_rnd(gen);
						seen_sel[sel_idx] = true;

						int categ = label_vec[sel_idx];
						MedSample smp; smp.id = (int)j;

						for (size_t k = 0; k < pred_calib_train.size(); ++k)
						{
							smp.outcome = k == categ; // set outcome := 1 (as case) only for categ
							pred_calib_train[k].push_back(smp);
						}
						pred_calib_mat.samples[j].id = (int)j;
						pred_calib_mat.samples[j].outcome = 0; //doesn't matter for prediction only
						for (size_t k = 0; k < pred_num_feats; ++k) {
							int fixed_idx = (int)k + int(k >= feat_idx); //skip current
							pred_calib_mat.data[all_names[fixed_idx]].push_back(train_vec[static_cast<size_t>(sel_idx) * static_cast<size_t>(pred_num_feats) + k]);
						}
					}

					//build pred:
					int idx_train = 0;
					for (size_t j = 0; j < seen_sel.size(); ++j)
					{
						if (seen_sel[j])
							continue;
						for (size_t k = 0; k < pred_num_feats; ++k)
							pred_train_vec[static_cast<size_t>(idx_train) * pred_num_feats + k] = train_vec[j * pred_num_feats + k];
						pred_label_vec[idx_train] = label_vec[j];
						++idx_train;
					}

					train_pred->learn(pred_train_vec, pred_label_vec, train_ratio, pred_num_feats);
					//get predictions for pred_calib_train to learn calibrator:
					train_pred->predict(pred_calib_mat);
					//get predictions into pred_calib_train:
					for (size_t j = 0; j < feats_predictors[i].calibrators.size(); ++j)
						for (size_t k = 0; k < pred_calib_train[j].size(); ++k)
							pred_calib_train[j][k].prediction = { pred_calib_mat.samples[k].prediction[j] };

					//Learn calibrators - for each pred bin:
					for (size_t k = 0; k < feats_predictors[i].calibrators.size(); ++k)
						feats_predictors[i].calibrators[k].Learn(pred_calib_train[k]);
				}
				else
					train_pred->learn(train_vec, label_vec, (int)label_vec.size(), pred_num_feats);
				feats_predictors[i].predictor = train_pred;

				progress.update();
			}
		}
	}
}

template<typename T> void GibbsSampler<T>::get_samples(map<string, vector<T>> &results, const GibbsSamplingParams &sampling_params, mt19937 &rnd_gen
	, const vector<bool> *mask, const vector<T> *mask_values, bool print_progress) {

	vector<bool> mask_f(all_feat_names.size());
	vector<T> mask_values_f(all_feat_names.size());
	if (mask == NULL)
		mask = &mask_f;
	if (mask_values == NULL) //and with init values
		mask_values = &mask_values_f;
	if (all_feat_names.empty())
		MTHROW_AND_ERR("Error in GibbsSampler<T>::get_samples - all_feat_names can't be empty\n");
	//fix mask values and sample gibbs for the rest by cohort_data as statistical cohort for univariate marginal dist
	int sample_loop = sampling_params.burn_in_count + (sampling_params.samples_count - 1) * sampling_params.jump_between_samples + 1;

	const vector<string> &all_names = all_feat_names;

	vector<T> current_sample(all_feat_names.size());
	for (size_t i = 0; i < mask->size(); ++i)
	{
		if (mask->at(i))
			current_sample[i] = mask_values->at(i);
		else
			current_sample[i] = mask_values->at(i); //init value - not fixed to be this value
	}
	vector<int> idx_iter; idx_iter.reserve(mask->size());
	vector<int> feat_idx;  feat_idx.reserve(mask->size());
	for (int i = 0; i < mask->size(); ++i)
		if (!mask->at(i)) {
			idx_iter.push_back(i);
			//find feature i index:
			int learn_ind = -1;
			for (size_t jj = 0; jj < impute_feat_names.size() && learn_ind < 0; ++jj)
				if (all_feat_names[i] == impute_feat_names[jj])
					learn_ind = (int)jj;
			if (learn_ind < 0)
				MTHROW_AND_ERR("Error GibbsSampler<T>::get_samples - Can't find feature %s in learned features\n", all_feat_names[i].c_str());
			feat_idx.push_back(learn_ind);
		}
	int pred_num_feats = (int)all_feat_names.size() - 1;

	//can parallel for random init of initiale values (just burn in)
	MedProgress progress("GibbsSampler::get_samples", sample_loop, 30, 10);
	for (size_t i = 0; i < sample_loop; ++i)
	{
		//create sample - iterate over all variables not in mask:
		int curr_feat_i = 0;
		for (int f_idx : idx_iter)
		{
			vector<T> curr_x(pred_num_feats);
			for (size_t k = 0; k < curr_x.size(); ++k)
			{
				int fixxed_idx = (int)k + int(k >= f_idx);
				curr_x[k] = current_sample[fixxed_idx];
			}
			T val = feats_predictors[feat_idx[curr_feat_i]].get_sample(curr_x, rnd_gen); //based on dist (or predictor - value bin dist)

			current_sample[f_idx] = val; //update current pos variable
			++curr_feat_i;
		}

		if (i >= sampling_params.burn_in_count && ((i - sampling_params.burn_in_count) % sampling_params.jump_between_samples) == 0) {
			//collect sample to result:
			for (size_t k = 0; k < all_names.size(); ++k) {
				T val = current_sample[k];
				//find best bin if needed:
				if (sampling_params.find_real_value_bin && !mask->at(k)) {
					//find index in impute_feat_names
					int learn_ind = -1;
					for (size_t jj = 0; jj < impute_feat_names.size() && learn_ind < 0; ++jj)
						if (all_feat_names[k] == impute_feat_names[jj])
							learn_ind = (int)jj;
					if (learn_ind < 0)
						MTHROW_AND_ERR("Error GibbsSampler<T>::get_samples - Can't find feature %s in learned features\n", all_feat_names[k].c_str());
					int pos = medial::process::binary_search_position(uniqu_value_bins[learn_ind].data(), uniqu_value_bins[learn_ind].data() + uniqu_value_bins[learn_ind].size() - 1, val);
					if (pos == 0)
						val = uniqu_value_bins[learn_ind][0];
					else {
						if (pos >= uniqu_value_bins[learn_ind].size())
							val = uniqu_value_bins[learn_ind].back();
						else {
							T diff_next = abs(val - uniqu_value_bins[learn_ind][pos]);
							T diff_prev = abs(val - uniqu_value_bins[learn_ind][pos - 1]);
							if (diff_prev < diff_next)
								val = uniqu_value_bins[learn_ind][pos - 1];
							else
								val = uniqu_value_bins[learn_ind][pos];
						}
					}
				}
				results[all_names[k]].push_back(val);
			}
		}

		if (print_progress)
			progress.update();
	}


}

template<typename T> void GibbsSampler<T>::get_samples(map<string, vector<T>> &results, const GibbsSamplingParams &sampling_params,
	const vector<bool> *mask, const vector<T> *mask_values, bool print_progress) {
	prepare_predictors();
	get_samples(results, sampling_params, _gen, mask, mask_values, print_progress);
}

template<typename T> void GibbsSampler<T>::prepare_predictors() {
	if (!done_prepare) {
		for (size_t i = 0; i < feats_predictors.size(); ++i)
		{
			if (feats_predictors[i].predictor != NULL)
#pragma omp critical
				feats_predictors[i].predictor->prepare_predict_single();
		}
		done_prepare = true;
	}
}

template<typename T> void GibbsSampler<T>::get_parallel_samples(map<string, vector<T>> &results,
	const GibbsSamplingParams &sampling_params, const vector<bool> *mask, const vector<T> *mask_values) {
	random_device rd;
	int worker_num = 0; //0 means max number of threads
	//int worker_num = sampling_params.samples_count; // 1 for each sample

	vector<T> mask_values_f(all_feat_names.size());
	vector<bool> mask_f(all_feat_names.size());
	if (mask == NULL)
		mask = &mask_f;
	if (mask_values == NULL) //and with init values
		mask_values = &mask_values_f;
	if (all_feat_names.empty())
		MTHROW_AND_ERR("Error in GibbsSampler<T>::get_parallel_samples - all_feat_names can't be empty\n");
	int N_tot_threads = omp_get_max_threads();
	if (worker_num > 0)
		N_tot_threads = worker_num;
	vector<mt19937> rnd_gens(N_tot_threads);
	for (size_t i = 0; i < rnd_gens.size(); ++i)
		rnd_gens[i] = mt19937(globalRNG::rand());

	GibbsSamplingParams per_thread_params = sampling_params;
	per_thread_params.samples_count = (int)ceil(float(sampling_params.samples_count) / N_tot_threads);
	prepare_predictors();

#pragma omp parallel for
	for (int i = 0; i < N_tot_threads; ++i)
	{
		int n_th = omp_get_thread_num();
		mt19937 &gen = rnd_gens[n_th];

		vector<T> mask_vals(all_feat_names.size());
		for (size_t i = 0; i < mask_vals.size(); ++i)
			if (!mask->at(i))
				mask_vals[i] = mask_values->at(i);
			else
				mask_vals[i] = mask_values->at(i);
		map<string, vector<T>> res;

		get_samples(res, per_thread_params, gen, mask, &mask_vals, n_th == 0);

#pragma omp critical
		for (auto it = res.begin(); it != res.end(); ++it)
			results[it->first].insert(results[it->first].end(), it->second.begin(), it->second.end());
	}
}

template<typename T> void GibbsSampler<T>::filter_samples(const map<string, vector<float>> &cohort_data,
	map<string, vector<T>> &results, const string &predictor_type, const string &predictor_args, float filter_sens) {
	mt19937 gen(globalRNG::rand());

	MedFeatures new_data;
	for (auto it = cohort_data.begin(); it != cohort_data.end(); ++it)
		new_data.attributes[it->first].normalized = false;

	int cohort_size = (int)cohort_data.begin()->second.size();
	new_data.samples.resize(cohort_size + results.begin()->second.size());
	for (size_t i = 0; i < new_data.samples.size(); ++i) {
		new_data.samples[i].id = (int)i;
		new_data.samples[i].outcome = (float)int(i < cohort_size);
	}
	//change outcome to be population label: is population 1?
	vector<float> labels(new_data.samples.size());
	for (size_t i = 0; i < new_data.samples.size(); ++i)
		labels[i] = new_data.samples[i].outcome;
	new_data.init_pid_pos_len();
	for (auto it = cohort_data.begin(); it != cohort_data.end(); ++it)
	{
		new_data.data[it->first] = it->second;
		for (size_t i = 0; i < results.at(it->first).size(); ++i)
			new_data.data[it->first].push_back(float(results.at(it->first)[i]));
	}

	//lets get auc on this problem:
	MedPredictor *predictor = MedPredictor::make_predictor(predictor_type, predictor_args);
	//lets fix labels weight that cases will be less common
	vector<float> preds;
	medial::models::get_pids_cv(predictor, new_data, 5, gen, preds);

	float auc = medial::performance::auc_q(preds, labels, &new_data.weights);
	MLOG("predictor AUC with CV to diffrentiate between populations is %2.3f\n", auc);

	//do filter: take FPR on SENS
	unordered_map<float, vector<int>> pred_idx;
	vector<float> sorted_preds;
	for (size_t i = 0; i < preds.size(); ++i) {
		if (pred_idx.find(preds[i]) == pred_idx.end())
			sorted_preds.push_back(preds[i]);
		pred_idx[preds[i]].push_back((int)i);
	}
	sort(sorted_preds.begin(), sorted_preds.end());

	double t_cnt = 0;
	double f_cnt = 0;
	double tot_true_labels = cohort_size;
	double tot_false_labels = results.begin()->second.size();
	vector<float> true_rate = vector<float>((int)sorted_preds.size());
	vector<float> false_rate = vector<float>((int)sorted_preds.size());
	int st_size = (int)sorted_preds.size() - 1;
	for (int i = st_size; i >= 0; --i)
	{
		const vector<int> &indexes = pred_idx[sorted_preds[i]];
		//calc AUC status for this step:
		for (int ind : indexes)
		{
			bool true_label = labels[ind] > 0;
			t_cnt += int(true_label);
			f_cnt += int(!true_label);
		}
		true_rate[st_size - i] = float(t_cnt / tot_true_labels);
		false_rate[st_size - i] = float(f_cnt / tot_false_labels);
	}

	//stop on SENS point:
	int stop_idx = 0;
	while (stop_idx < true_rate.size() && true_rate[stop_idx] < filter_sens)
		++stop_idx;
	if (stop_idx >= true_rate.size())
		--stop_idx;
	stop_idx = st_size - stop_idx;
	//collect all indexes above that score
	vector<int> filter_sel;
	for (int i = st_size; i >= stop_idx; --i)
	{
		const vector<int> &indexes = pred_idx[sorted_preds[i]];
		for (int ind : indexes)
			if (!labels[ind])
				filter_sel.push_back(ind - cohort_size);
	}

	//commit selection:
	map<string, vector<T>> filterd;
	for (auto it = results.begin(); it != results.end(); ++it) {
		filterd[it->first].resize(filter_sel.size());
		for (size_t i = 0; i < filter_sel.size(); ++i)
			filterd[it->first][i] = (it->second[filter_sel[i]]);
	}

	results.swap(filterd);

}

template<typename T> GibbsSampler<T>::~GibbsSampler() {}

template class PredictorOrEmpty<float>;
template class PredictorOrEmpty<double>;

template class GibbsSampler<float>;
template class GibbsSampler<double>;