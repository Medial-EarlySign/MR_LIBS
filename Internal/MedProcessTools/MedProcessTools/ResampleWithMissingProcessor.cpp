#include "ResampleWithMissingProcessor.h"
#include "ExplainWrapper.h"

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MED_MODEL
#define LOCAL_LEVEL	LOG_DEF_LEVEL

void ResampleMissingProcessor::init_defaults() {
	missing_value = MED_MAT_MISSING_VALUE;
	add_new_data = 0;
	sample_masks_with_repeats = true;
	uniform_rand = false;
	use_shuffle = true;
	subsample_train = 0;
	limit_mask_size = 0;
	uniform_rand_p = (float)0.5;
	verbose = true;
	processor_type = FTR_PROCESS_RESAMPLE_WITH_MISSING;
	grouping = "";
	duplicate_only_with_missing = false;
}

int ResampleMissingProcessor::init(map<string, string>& mapper) {
	for (const auto &it : mapper)
	{
		//! [ResampleMissingProcessor::init]
		if (it.first == "missing_value")
			missing_value = med_stof(it.second);
		else if (it.first == "add_new_data")
			add_new_data = med_stoi(it.second);
		else if (it.first == "selected_tags")
			boost::split(selected_tags, it.second, boost::is_any_of(","));
		else if (it.first == "removed_tags")
			boost::split(removed_tags, it.second, boost::is_any_of(","));
		else if (it.first == "sample_masks_with_repeats")
			sample_masks_with_repeats = med_stoi(it.second) > 0;
		else if (it.first == "uniform_rand")
			uniform_rand = med_stoi(it.second) > 0;
		else if (it.first == "use_shuffle")
			use_shuffle = med_stoi(it.second) > 0;
		else if (it.first == "subsample_train")
			subsample_train = med_stoi(it.second);
		else if (it.first == "limit_mask_size")
			limit_mask_size = med_stoi(it.second);
		else if (it.first == "uniform_rand_p")
			uniform_rand_p = med_stof(it.second);
		else if (it.first == "duplicate_only_with_missing")
			duplicate_only_with_missing = med_stoi(it.second) > 0;
		else if (it.first == "verbose")
			verbose = med_stoi(it.second) > 0;
		else if (it.first == "grouping")
			grouping = it.second;
		else if (it.first == "fp_type" || it.first == "use_parallel_learn" || it.first == "use_parallel_apply") {}
		else
			MTHROW_AND_ERR("Error in ResampleMissingProcessor::init - unsupported argument %s\n", it.first.c_str());
		//! [ResampleMissingProcessor::init]
	}

	return 0;
}

void ResampleMissingProcessor::dprint(const string &pref, int fp_flag) {
	string res = this->object_json();
	boost::replace_all(res, "\n", " ");
	MLOG("%s :: %s\n", pref.c_str(), res.c_str());
}

int _count_msn(const float *vals, int sz, float val) {
	int res = 0;
	for (size_t i = 0; i < sz; ++i)
		res += int(vals[i] == val);
	return res;
}

string ResampleMissingProcessor::select_learn_matrix(const vector<string> &matrix_tags) const {
	//creates new tag
	return my_class_name();
}

int ResampleMissingProcessor::Learn(MedFeatures& features, unordered_set<int>& ids) {
	vector<string> features_nms;
	features.get_feature_names(features_nms);
	if (!grouping.empty())
		ExplainProcessings::read_feature_grouping(grouping, features_nms, group2Inds, groupNames);
	//use group2Inds, groupNames

	if (limit_mask_size >= group2Inds.size()) {
		MWARN("WARNING: limit_mask_size=%d which is bigger than number of groups/features(%zu)\n",
			limit_mask_size, group2Inds.size());
		limit_mask_size = (int)group2Inds.size(); //problem with arguments
	}

	mt19937 gen(globalRNG::rand());
	int nftrs_grp = (int)group2Inds.size();
	int nftrs = (int)features.data.size();
	int train_mat_size = (int)features.samples.size();

	unordered_set<string> whitelist(selected_tags.begin(), selected_tags.end());
	unordered_set<string> blacklist(removed_tags.begin(), removed_tags.end());
	vector<bool> allowed_selection(nftrs);
	for (size_t i = 0; i < nftrs; ++i)
	{
		const unordered_set<string> &candidate_tags = features.tags.at(features_nms[i]);
		bool allowed = whitelist.empty(); //by default allow to select all to generate missing value if whitelist is empty
		//find in whitelist if needed:
		string allow_reason = "";
		for (const string &c : candidate_tags)
		{
			allowed = whitelist.find(c) != whitelist.end();
			if (allowed) {
				allow_reason = c;
				break;
			}
		}
		//only if allowed check for blacklist, otherwise not needed:
		if (allowed) {
			string black_reason = "";
			for (const string &c : candidate_tags)
			{
				allowed = blacklist.find(c) == blacklist.end(); //allow only if not in blacklist
				if (!allowed) {
					black_reason = c;
					break;
				}
			}
			//check if not allowed now and whitelist was not empty, means tag was in both sets directly - raise error:
			if (!(allowed) && !whitelist.empty())
				MTHROW_AND_ERR("Error ResampleMissingProcessor::Learn - feature %s is both in whitelist selected_tags (tag=%s) "
					" and blacklist remove_tags (tag=%s) directly\n",
					features_nms[i].c_str(), allow_reason.c_str(), black_reason.c_str());
		}
		allowed_selection[i] = allowed;
	}
	vector<int> skip_grp;
	for (int i = 0; i < nftrs_grp; ++i) {
		bool all_not_allowed = true;
		for (size_t k = 0; k < group2Inds[i].size() && all_not_allowed; ++k)
			all_not_allowed = !allowed_selection[group2Inds[i][k]];
		if (all_not_allowed)
			skip_grp.push_back(i);
	}
	int allowed_grp_cnt = nftrs_grp - (int)skip_grp.size();
	if (nftrs_grp != nftrs && (!whitelist.empty() || !blacklist.empty()))
		MLOG("INFO :: has %d groups, and can erase values in %d groups after whitelist, blacklist\n",
			nftrs_grp, allowed_grp_cnt);
	unordered_set<int> skip_grp_set(skip_grp.begin(), skip_grp.end());

	vector<int> missing_hist(nftrs + 1), added_missing_hist(nftrs + 1), added_grp_hist(nftrs_grp + 1);
	MedMat<float> x_mat;
	vector<int> original_samples_id(features.samples.size());
	features.get_as_matrix(x_mat);
	for (int i = 0; i < original_samples_id.size(); ++i)
		original_samples_id[i] = i;

	vector<int> miss_cnts(train_mat_size + add_new_data);
	vector<int> mask_group_sizes(train_mat_size + add_new_data); //stores for each sample - how many missings in groups manner:
	for (size_t i = 0; i < train_mat_size; ++i)
	{
		//check how many groups missings:
		int grp_misses = 0;
		for (int j = 0; j < nftrs_grp; ++j) {
			bool has_missing = false;
			for (size_t k = 0; k < group2Inds[j].size() && !has_missing; ++k)
				has_missing = x_mat(i, group2Inds[j][k]) == missing_value;
			grp_misses += int(has_missing);
		}
		mask_group_sizes[i] = grp_misses;
	}

	if (add_new_data > 0) {
		original_samples_id.reserve(original_samples_id.size() + add_new_data);
		vector<float> rows_m(static_cast<size_t>(add_new_data) * static_cast<size_t>(nftrs));
		unordered_set<vector<bool>> seen_mask;
		
		double log_max_opts = log(add_new_data) / log(2.0);
		if (log_max_opts >= nftrs_grp) {
			if (!sample_masks_with_repeats)
				MWARN("Warning: you have request to sample masks without repeats, but it can't be done. setting sample with repeats\n");
			sample_masks_with_repeats = true;
		}
		if (verbose)
			MLOG("Adding %d Data points (has %d features with %d groups)\n", add_new_data, nftrs, nftrs_grp);
		vector<int> allowed_to_duplicate_list;
		int max_opts = train_mat_size;
		if (duplicate_only_with_missing) {
			//check who can be duplicated
			for (int i = 0; i < train_mat_size; ++i)
			{
				bool has_missing = false;
				for (int j = 0; j < x_mat.ncols && !has_missing; ++j)
					has_missing = x_mat(i, j) == missing_value;
				if (has_missing)
					allowed_to_duplicate_list.push_back(i);
			}
			if (allowed_to_duplicate_list.size() == train_mat_size)
				allowed_to_duplicate_list.clear(); //all are allowed
			else
				max_opts = (int)allowed_to_duplicate_list.size();
		}

		uniform_int_distribution<> rnd_row(0, max_opts - 1);
		MedProgress add_progress("Add_Train_Data", add_new_data, 30, 1);
		for (size_t i = 0; i < add_new_data; ++i)
		{
			float *curr_row = &rows_m[i *  nftrs];
			//select row:
			int row_sel = rnd_row(gen);
			if (!allowed_to_duplicate_list.empty()) //fetch real index if has limitation. row_sel is index in allowed_to_duplicate_list
				row_sel = allowed_to_duplicate_list[row_sel];

			//True means - use mat value (don't override with missing values). in this stage it means you can select it
			vector<bool> curr_mask(nftrs_grp);

			for (int j = 0; j < nftrs_grp; ++j) {
				bool has_missing = false;
				for (size_t k = 0; k < group2Inds[j].size() && !has_missing; ++k)
					has_missing = x_mat(row_sel, group2Inds[j][k]) == missing_value;
				curr_mask[j] = !has_missing && (skip_grp_set.find(j) == skip_grp_set.end());
			}

			medial::shapley::generate_mask_(curr_mask, nftrs_grp, gen, uniform_rand, uniform_rand_p, use_shuffle, limit_mask_size);
			while (!sample_masks_with_repeats && seen_mask.find(curr_mask) != seen_mask.end())
				medial::shapley::generate_mask_(curr_mask, nftrs_grp, gen, uniform_rand, uniform_rand_p, use_shuffle, limit_mask_size);
			if (!sample_masks_with_repeats)
				seen_mask.insert(curr_mask);
			//fix mask and return skip_grp to True to keep values as they are:
			for (int g : skip_grp)
				curr_mask[g] = true;

			//commit mask to curr_row
			int msn_cnt = 0;
			for (int j = 0; j < nftrs_grp; ++j)
			{
				if (curr_mask[j]) {
					for (size_t k = 0; k < group2Inds[j].size(); ++k)
						curr_row[group2Inds[j][k]] = x_mat(row_sel, group2Inds[j][k]);
				}
				else {
					for (size_t k = 0; k < group2Inds[j].size(); ++k)
						curr_row[group2Inds[j][k]] = missing_value;
				}
				msn_cnt += int(!curr_mask[j]); //how many missings
			}
			original_samples_id.push_back(row_sel);
			++added_grp_hist[msn_cnt];
			add_progress.update();
			mask_group_sizes[train_mat_size + i] = msn_cnt;
		}
		x_mat.add_rows(rows_m);
	}

	// Add data with missing values according to sample masks
	vector<int> grp_missing_hist_all(nftrs_grp + 1);

	for (int i = 0; i < x_mat.nrows; ++i) {
		miss_cnts[i] = _count_msn(x_mat.data_ptr(i, 0), nftrs, missing_value);
		++missing_hist[miss_cnts[i]];
		if (i >= train_mat_size)
			++added_missing_hist[miss_cnts[i]];
		++grp_missing_hist_all[mask_group_sizes[i]];
	}
	if (verbose) {
		medial::print::print_hist_vec(miss_cnts, "missing_values_cnt percentiles [0 - " + to_string(nftrs) + "] (with added samples - no groups)", "%d");
		medial::print::print_hist_vec(mask_group_sizes, "mask_group_sizes percentiles [0 - " + to_string(nftrs_grp) + "] (with added samples - for groups)", "%d");
		medial::print::print_hist_vec(added_missing_hist, "selected counts in hist of missing_values_cnt (only for added - no groups)", "%d");
		if (added_grp_hist.size() < 50)
			medial::print::print_vec(added_grp_hist, "grp hist (only for added - on groups)", "%d");
		else
			medial::print::print_hist_vec(added_grp_hist, "hist of added_grp_hist (only for added - on groups)", "%d");
	}

	if (subsample_train > 0 && subsample_train < train_mat_size) {
		//do subsampling:
		MLOG("INFO:: ResampleMissingProcessor::Learn - subsampling original train matrix\n");
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
		vector<int> new_samples_ids(selected_idx_vec.size());
		x_mat.get_sub_mat(selected_idx_vec, empty_is_all);
		for (size_t i = 0; i < selected_idx_vec.size(); ++i)
			new_samples_ids[i] = original_samples_id[selected_idx_vec[i]];
		original_samples_id = std::move(new_samples_ids);
	}
	//use x_mat, original_samples_id to manipulate features => data and samples:
	vector<MedSample> new_smps(original_samples_id.size());
	map<string, vector<float>> new_data;
	for (size_t i = 0; i < original_samples_id.size(); ++i)
		new_smps[i] = features.samples[original_samples_id[i]];
	for (size_t i = 0; i < features_nms.size(); ++i)
	{
		vector<float> &vec = new_data[features_nms[i]];
		vec.resize(x_mat.nrows);
		for (size_t j = 0; j < x_mat.nrows; ++j)
			vec[j] = x_mat(j, i);
	}

	features.samples = std::move(new_smps);
	features.data = std::move(new_data);
	features.init_pid_pos_len();

	return 0;
}