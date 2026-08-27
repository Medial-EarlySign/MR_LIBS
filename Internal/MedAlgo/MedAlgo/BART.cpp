#include "BART.h"
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <limits>
#include <boost/math/distributions/inverse_gamma.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/weibull_distribution.hpp>
//#include "MedStat/MedStat/MedPerformance.h"

//#define DEBUG_VERBOSE

bool bart_tree::select_split(bart_node *current_node, const vector<float> &x, int nftrs,
	vector<vector<int>> &split_obx_indexes) {
	//choose legal split that has at least TBD obs in the split
	current_node->split_index = -1;
	vector<float> &sorted_keys = feature_to_sorted_vals[current_node->feature_number];
	const unordered_map<float, int> &val_to_index = feature_to_val_index[current_node->feature_number];

	int min_limit = 0, max_limit = (int)sorted_keys.size() - 1;
	//traverse branch to find strict limit:
	const bart_node *p = current_node->parent;
	const bart_node *prev_child = current_node;
	while (p != NULL)
	{
		if (p->feature_number == current_node->feature_number) {
			if (p->childs[1] == prev_child && p->split_index > min_limit)
				min_limit = p->split_index;
			if (p->childs[0] == prev_child && p->split_index < max_limit)
				max_limit = p->split_index;
		}
		prev_child = p;
		p = p->parent;
	}

	//check where we can split - has enought samples between min=>max and node:
	vector<int> cumsum_samples(max_limit - min_limit + 1);
	for (size_t k = 0; k < current_node->observation_indexes.size(); ++k) {
		int ind = current_node->observation_indexes[k];
		float search_term = x[ind*nftrs + current_node->feature_number];
		/*int pos = medial::process::binary_search_index(sorted_keys.data(),
		sorted_keys.data() + (int)sorted_keys.size() - 1, search_term); *///the pos of search_term in sorted_keys
		int pos = val_to_index.at(search_term);
		++cumsum_samples[pos - min_limit];
	}
	for (size_t i = min_limit + 1; i <= max_limit; ++i)
		cumsum_samples[i - min_limit] += cumsum_samples[i - min_limit - 1];

	//list valid split indexes:
	int final_min = -1, final_max = -1;
	for (size_t i = min_limit; i <= max_limit; ++i)
		if (cumsum_samples[i - min_limit] >= params.min_obs_in_node) {
			final_min = (int)i;
			break;
		}
	if (final_min == -1)
		return false;
	int total_count = (int)current_node->observation_indexes.size();
	for (int i = max_limit - 1; i > final_min; --i)
		if (total_count - cumsum_samples[i - min_limit] >= params.min_obs_in_node) {
			final_max = i;
			break;
		}
	if (final_max == -1)
		return false;

	//choose between final_min, final_max
	uniform_int_distribution<> choose_split(final_min, final_max);
	current_node->split_index = choose_split(_rnd_gen);
	current_node->num_split_options = final_max - final_min + 1;
	current_node->mark_change = true;
	//commit on observation_indexes:
	split_obx_indexes.resize(2);
	split_obx_indexes[0].reserve(current_node->observation_indexes.size());
	split_obx_indexes[1].reserve(current_node->observation_indexes.size());
	for (size_t k = 0; k < current_node->observation_indexes.size(); ++k) {
		int ind = current_node->observation_indexes[k];
		split_obx_indexes[x[ind*nftrs + current_node->feature_number] > sorted_keys[current_node->split_index]].push_back(ind);
	}
	return true;
}

bool bart_tree::has_split(const bart_node *current_node, const vector<float> &x, int nftrs,
	int feature_num) const {
	const vector<float> &sorted_keys = feature_to_sorted_vals[feature_num];
	const unordered_map<float, int> &val_to_index = feature_to_val_index[feature_num];

	int min_limit = 0, max_limit = (int)sorted_keys.size() - 1;
	//traverse branch to find strict limit:
	const bart_node *p = current_node->parent;
	const bart_node *prev_child = current_node;
	while (p != NULL)
	{
		if (p->feature_number == feature_num) {
			if (p->childs[1] == prev_child && p->split_index > min_limit)
				min_limit = p->split_index;
			if (p->childs[0] == prev_child && p->split_index < max_limit)
				max_limit = p->split_index;
		}
		prev_child = p;
		p = p->parent;
	}
	if (min_limit == max_limit)
		return false;

	//check where we can split - has enought samples between min=>max and node:
	vector<int> cumsum_samples(max_limit - min_limit + 1);
	for (size_t k = 0; k < current_node->observation_indexes.size(); ++k) {
		int ind = current_node->observation_indexes[k];
		float search_term = x[ind*nftrs + feature_num];
		/*int pos = medial::process::binary_search_index(sorted_keys.data(),
		sorted_keys.data() + (int)sorted_keys.size() - 1, search_term); *///the pos of search_term in sorted_keys
		int pos = val_to_index.at(search_term);
		++cumsum_samples[pos - min_limit];
	}
	for (size_t i = min_limit + 1; i <= max_limit; ++i)
		cumsum_samples[i - min_limit] += cumsum_samples[i - min_limit - 1];

	//list valid split indexes:
	int final_min = -1, final_max = -1;
	for (size_t i = min_limit; i <= max_limit; ++i)
		if (cumsum_samples[i - min_limit] >= params.min_obs_in_node) {
			final_min = (int)i;
			break;
		}
	if (final_min == -1)
		return false;
	int total_count = (int)current_node->observation_indexes.size();
	for (int i = max_limit - 1; i > final_min; --i)
		if (total_count - cumsum_samples[i - min_limit] >= params.min_obs_in_node) {
			final_max = i;
			break;
		}
	if (final_max == -1)
		return false;

	return true; //has final_min, final_max
}

int bart_tree::clear_tree_mem(bart_node *node) {
	if (node == NULL)
		return 0; //nothing to do - reached beyond leaf

	int s = clear_tree_mem(node->childs[0]);
	s += clear_tree_mem(node->childs[1]);
	if (node->parent != NULL) {
		if (node->parent->childs[0] == node)
			node->parent->childs[0] = NULL;
		else
			node->parent->childs[1] = NULL;
	}
	delete node;
	++s;
	return s;
}

void bart_node::print_tree(const vector<vector<float>> &feature_sorted_values) const {
	string addtional = "", additional2 = "";
	if (childs[0] != NULL)
		addtional = to_string(feature_sorted_values[feature_number][split_index]);
	else
		additional2 = ", node_value=" + to_string(node_value);
	if (parent == NULL)
		printf("root in %p, childs=[%p, %p], feature_num=%d, split_index=%d(%s), obs_count=%zu\n", this,
			childs[0], childs[1], feature_number, split_index, addtional.c_str(),
			observation_indexes.size());
	else
		printf("node in %p, parent=%p, childs=[%p, %p], feature_num=%d, split_index=%d(%s), obs_count=%zu%s\n",
			this, this->parent, childs[0], childs[1], feature_number, split_index, addtional.c_str(),
			observation_indexes.size(), additional2.c_str());
	if (childs[0] != NULL) {
		childs[0]->print_tree(feature_sorted_values);
		childs[1]->print_tree(feature_sorted_values);
	}
}

void bart_node::validate_tree(const vector<vector<float>> &feature_sorted_values, const vector<float> &x, int nftrs) const {
	if (childs[0] != NULL) {
		if (childs[0]->parent != this) {
			print_tree(feature_sorted_values);
			printf("broken tree. in node=%p, has child=%p, and child[0]->parent=%p\n", this, childs[0],
				childs[0]->parent);
			throw logic_error("broken tree");
		}
		if (childs[1]->parent != this) {
			print_tree(feature_sorted_values);
			printf("broken tree. in node=%p, has child=%p, and child[1]->parent=%p\n", this, childs[1],
				childs[1]->parent);
			throw logic_error("broken tree");
		}
		//test for split condition:
		int f_num = feature_number;
		int spl_num = split_index;
		const vector<float> &split_vals = feature_sorted_values[f_num];
		vector<vector<int>> split_new(2);
		for (int ind : observation_indexes)
			split_new[x[ind*nftrs + f_num] > split_vals[spl_num]].push_back(ind);
		for (size_t k = 0; k < 2; ++k)
		{
			if (split_new[k].size() != childs[k]->observation_indexes.size()) {
				print_tree(feature_sorted_values);
				printf("broken observation_indexes. in node=%p, has child[%zu]=%p, "
					"count=%zu, right_count=%zu\n", this, k, childs[k],
					childs[k]->observation_indexes.size(), split_new[k].size());
				throw logic_error("broken observation_indexes");
			}
			for (size_t i = 0; i < split_new[k].size(); ++i)
				if (split_new[k][i] != childs[k]->observation_indexes[i]) {
					print_tree(feature_sorted_values);
					printf("broken observation_indexes. in node=%p, has child[%zu]=%p, index=%zu"
						"found_index_value=%d, right_index_value=%d\n", this, k, childs[k], i,
						childs[k]->observation_indexes[i], split_new[k][i]);
					throw logic_error("broken observation_indexes");
				}
		}
		childs[0]->validate_tree(feature_sorted_values, x, nftrs);
		childs[1]->validate_tree(feature_sorted_values, x, nftrs);
	}
}

void bart_tree::next_gen_tree(const vector<float> &x, const vector<float> &y) {
	//get tree
	int nsamples = (int)y.size();
	int nftrs = (int)x.size() / nsamples;
	uniform_real_distribution<> random_number;

	//first init
	if (feature_to_sorted_vals.empty()) {
		feature_to_sorted_vals.resize(nftrs);
		feature_to_val_index.resize(nftrs);
		for (size_t i = 0; i < nftrs; ++i) {
			unordered_set<float> uniq_vals;
			for (int k = 0; k < y.size(); ++k)
				if (uniq_vals.find(x[static_cast<size_t>(k)*static_cast<size_t>(nftrs) + i]) == uniq_vals.end()) {
					feature_to_sorted_vals[i].push_back(x[static_cast<size_t>(k)*static_cast<size_t>(nftrs) + i]);
					uniq_vals.insert(x[static_cast<size_t>(k)*static_cast<size_t>(nftrs) + i]);
				}
		}
		for (size_t i = 0; i < nftrs; ++i)
			sort(feature_to_sorted_vals[i].begin(), feature_to_sorted_vals[i].end());
		//populated feature_to_val_index:
		for (size_t i = 0; i < nftrs; ++i) {
			const vector<float> &sorted_vals = feature_to_sorted_vals[i];
			for (int k = 0; k < sorted_vals.size(); ++k)
				feature_to_val_index[i][sorted_vals[k]] = k;
		}

	}

	//Handle root==NULL - start of tree:
	if (root == NULL) {
		root = new bart_node;
		root->feature_number = -1; //no split
		root->node_value = 0; //default value
		root->observation_indexes.resize(nsamples);
		for (size_t i = 0; i < nsamples; ++i)
			root->observation_indexes[i] = (int)i;

		tree_loglikelihood = -1; //no need to calc exactly always update
	}

	//select action:
	if (params.data_prior_type == regression_mean_shift)
		calc_likelihood(x, nftrs, y); //calc again - sigma, sigma_mu have changed!
	double current_tree_likelihood = tree_loglikelihood;
	tree_change_details change;
	int stuck_val = 0;
	for (size_t i = 0; i < action_priors.size(); ++i)
		stuck_val |= 1 << i;

	int selected_action;
	while (change.changed_nodes_before.empty()) {
		double action_sm = 0;
		int seen_action = 0;
		selected_action = (int)action_priors.size() - 1;
		double sel_num = random_number(_rnd_gen);
		for (size_t i = 0; i < action_priors.size(); ++i)
		{
			action_sm += action_priors[i];
			if (action_sm >= sel_num) {//passed - selected
				selected_action = (int)i;
				break;
			}
		}
		//printf("choosing action = %d\n", selected_action);
		switch (selected_action)
		{
		case 0:
			change = do_grow(x, y);
			break;
		case 1:
			change = do_prune(x, y);
			break;
		case 2:
			change = do_change(x, y);
			break;
		case 3:
			change = do_swap(x, y);
			break;
		default:
			throw logic_error("Unsupported action\n");
		}
		seen_action |= 1 << selected_action;
		if (seen_action == stuck_val) {
			printf("Warning: Stucked - no way to continue\n");
			return;
		}
		//printf("choose action = %d, change_size=%zu\n", selected_action, change.changed_nodes_before.size());
	}
#ifdef DEBUG_VERBOSE
	printf("last_action=%d\n", change.action);
	root->validate_tree(feature_to_sorted_vals, x, nftrs);
#endif //  DEBUG_VERBOSE

	calc_likelihood(x, nftrs, y);
	double calc_log_transition = 0; //TODO:
	if (change.action == 0) { //grow
		calc_log_transition += log(action_priors[1] * change.changed_nodes_after[0]->num_feature_options
			* change.changed_nodes_after[0]->num_split_options * change.num_node_selection);
		vector<bart_node *> prune_opt_for_count;
		get_avalible_prune(x, nftrs, prune_opt_for_count);
		calc_log_transition -= log(action_priors[0]) + log(prune_opt_for_count.size());
	}
	else if (change.action == 1) {//prune
		calc_log_transition -= log(action_priors[0] * change.num_node_selection);
		vector<bart_node *> grow_opt_for_count;
		get_avalible_grow(x, nftrs, grow_opt_for_count);
		calc_log_transition += log(action_priors[1] * grow_opt_for_count.size()
			*change.changed_nodes_before[0]->num_feature_options *
			change.changed_nodes_before[0]->num_split_options);
	}
	else if (change.action == 2) {//change
								  //only change in split options for new rule:
		calc_log_transition = log(change.changed_nodes_before[0]->num_split_options) -
			log(change.changed_nodes_after[0]->num_split_options);

	}
	else if (change.action == 3) {//swap
								  //nothing to do - cancels out
	}
	else
		throw logic_error("unsupported");

	double new_likelihood = tree_loglikelihood;
	double l_ratio = 0;
	double rnd_sel = random_number(_rnd_gen);
	if (rnd_sel > 0)
		rnd_sel = -log(rnd_sel);
	else
		rnd_sel = numeric_limits<double>().lowest();
	//if new_likelihood < current_tree_likelihood - it's improve - so not needed to calc
	if (current_tree_likelihood > 0 && new_likelihood > current_tree_likelihood)
		l_ratio = new_likelihood - current_tree_likelihood + calc_log_transition;
	//#ifdef DEBUG_VERBOSE
	/*printf(", action=%d, l_ratio=%2.3f, new_likelihood=%2.1f, prevoious=%2.1f, calc_log_transition=%2.3f\n",
	change.action, l_ratio, new_likelihood, current_tree_likelihood, calc_log_transition);*/
	//#endif
	if (rnd_sel >= l_ratio)
		commit_change(change);
	else {
		rollback_change(change);
		tree_loglikelihood = current_tree_likelihood;
	}
}

void bart_node::list_all_nodes(vector<bart_node *> &all_nodes) {
	all_nodes.push_back(this);
	if (childs[0] == NULL)
		return; //has no childs - add itself and end:	
	childs[0]->list_all_nodes(all_nodes);
	childs[1]->list_all_nodes(all_nodes);
}

//commit change
void bart_tree::commit_change(const tree_change_details &change) {
	//clear memory of old state which was existed:
	for (size_t i = 0; i < change.changed_nodes_before.size(); ++i)
		if (change.changed_nodes_before[i] != NULL) {
			//printf("commit_change: delete %p\n", change.changed_nodes_before[i]);
			delete change.changed_nodes_before[i];
		}
#ifdef DEBUG_VERBOSE
	printf("AFTER COMMIT with action=%d\n", change.action);
	root->print_tree(feature_to_sorted_vals);
#endif
}

//rollback change - dont accept new tree move
void bart_tree::rollback_change(const tree_change_details &change) {
	//update pointers to leaves, parents of leaves and all nodes:
#ifdef DEBUG_VERBOSE
	printf("BEFORE ROLLBACK with action=%d\n", change.action);
	root->print_tree(feature_to_sorted_vals);
#endif
	//first do rollback - than delete:
	for (size_t i = 0; i < change.changed_nodes_before.size(); ++i) {
		if (change.changed_nodes_after[i] == NULL || change.changed_nodes_before[i] == NULL)
			continue;

		if (change.changed_nodes_before[i]->parent == NULL) //change was in root
			root = change.changed_nodes_before[i];
		bart_node *parent_node = change.changed_nodes_after[i]->parent;

		if (parent_node != NULL) { //If wasn't root - has parent - connect it's parent to child
			if (parent_node->childs[0] == change.changed_nodes_after[i])
				parent_node->childs[0] = change.changed_nodes_before[i];
			else
				parent_node->childs[1] = change.changed_nodes_before[i];
		}
		/*if (change.action == 3) { //swap: connect also childs to parents
		if (change.changed_nodes_before[i]->childs[0] != NULL) {
		change.changed_nodes_before[i]->childs[0]->parent = change.changed_nodes_before[i];
		change.changed_nodes_before[i]->childs[1]->parent = change.changed_nodes_before[i];
		}
		}*/
	}

	//delete
	for (size_t i = 0; i < change.changed_nodes_before.size(); ++i) {
		if (change.changed_nodes_after[i] == NULL)  //mark for deletion, but doing rollback - so not deletion
			continue;
		//printf("rollback_change_2: delete %p\n", change.changed_nodes_after[i]);
		delete change.changed_nodes_after[i]; //delete current object
	}
#ifdef DEBUG_VERBOSE
	printf("AFTER ROLLBACK with action=%d\n", change.action);
	root->print_tree(feature_to_sorted_vals);
#endif
}

void bart_tree::get_avalible_change(const vector<float> &x, int nftrs, vector<bart_node *> &good_idx) {
	//not in bottom and has split:
	good_idx.clear();
	vector<bart_node *> all_nodes;
	root->list_all_nodes(all_nodes);
	for (size_t i = 0; i < all_nodes.size(); ++i)
	{
		if (all_nodes[i]->childs[0] == NULL)
			continue; //it's bottom node
					  //check that it has split feature:
		bool has_sp = false;
		for (size_t f = 0; f < nftrs && !has_sp; ++f)
			has_sp = has_split(all_nodes[i], x, nftrs, (int)f);
		if (has_sp)
			good_idx.push_back(all_nodes[i]);
	}
}

void bart_tree::get_avalible_feats_change(const bart_node *selected_node, const vector<float> &x, int nftrs, vector<int> &good_idx) {
	//not in bottom and has split:
	good_idx.clear();
	if (selected_node->childs[0] == NULL)
		return;
	//check that it has split feature:
	for (size_t f = 0; f < nftrs; ++f)
		if (has_split(selected_node, x, nftrs, (int)f))
			good_idx.push_back((int)f);
}

void bart_tree::propogate_change_down(bart_node *current_node, const vector<float> &x, int nftrs
	, vector<bart_node *> &list_nodes_after) {
	if (current_node->childs[0] == NULL)
		return; //nothing to do - reach leaf
				//find new split:
	int feature_num = current_node->feature_number;
	vector<float> &sorted_keys = feature_to_sorted_vals[feature_num];
	int split_index = current_node->split_index;
	vector<vector<int>> split_obx_indexes(2);
	split_obx_indexes[0].reserve(current_node->observation_indexes.size());
	split_obx_indexes[1].reserve(current_node->observation_indexes.size());
	for (size_t k = 0; k < current_node->observation_indexes.size(); ++k) {
		int ind = current_node->observation_indexes[k];
		split_obx_indexes[x[ind*nftrs + feature_num] > sorted_keys[split_index]].push_back(ind);
	}

	if (split_obx_indexes[0].size() < params.min_obs_in_node || split_obx_indexes[1].size()
		< params.min_obs_in_node) {
		//break tree here - don't split
		//clear memory for tress:
		int era_cnt = clear_tree_mem(current_node->childs[0]);
		for (size_t i = 0; i < era_cnt; ++i)
			list_nodes_after.push_back(NULL);
		era_cnt = clear_tree_mem(current_node->childs[1]);
		for (size_t i = 0; i < era_cnt; ++i)
			list_nodes_after.push_back(NULL);
		/* done by clear_tree and record changes in change
		current_node->childs[0] = NULL;
		current_node->childs[1] = NULL; */
		return; //reached end
	}
	current_node->childs[0]->observation_indexes.swap(split_obx_indexes[0]);
	current_node->childs[1]->observation_indexes.swap(split_obx_indexes[1]);
	current_node->childs[0]->mark_change = true;
	current_node->childs[1]->mark_change = true;
	list_nodes_after.push_back(current_node->childs[0]);
	propogate_change_down(current_node->childs[0], x, nftrs, list_nodes_after);
	list_nodes_after.push_back(current_node->childs[1]);
	propogate_change_down(current_node->childs[1], x, nftrs, list_nodes_after);
}

tree_change_details bart_tree::do_change(const vector<float> &x, const vector<float> &y) {
	tree_change_details change;
	change.action = 2;
	int nsamples = (int)y.size();
	int nftrs = (int)x.size() / nsamples;

	vector<bart_node *> avalible_nodes;
	get_avalible_change(x, nftrs, avalible_nodes);
	if (avalible_nodes.empty())
		return change; //not possible
	uniform_int_distribution<> select_node(0, (int)avalible_nodes.size() - 1);
	change.num_node_selection = (int)avalible_nodes.size();
	bart_node *selected_node = avalible_nodes[select_node(_rnd_gen)];
	vector<int> avalible_feats;
	get_avalible_feats_change(selected_node, x, nftrs, avalible_feats);
	if (avalible_feats.empty())
		return change; //not possible
					   //GET availble split var not in bottom:
	uniform_int_distribution<> select_feat(0, (int)avalible_feats.size() - 1);

	bart_node *new_node;
	selected_node->deep_clone(new_node);
	new_node->feature_number = avalible_feats[select_feat(_rnd_gen)];
	new_node->num_feature_options = (int)avalible_feats.size();
	vector<vector<int>> new_splits;
	if (select_split(new_node, x, nftrs, new_splits)) {
		selected_node->list_all_nodes(change.changed_nodes_before);

		new_node->childs[0]->observation_indexes.swap(new_splits[0]);
		new_node->childs[1]->observation_indexes.swap(new_splits[1]);
		new_node->childs[0]->mark_change = true;
		new_node->childs[1]->mark_change = true;

		change.changed_nodes_after = { new_node, new_node->childs[0] };
		propogate_change_down(new_node->childs[0], x, nftrs, change.changed_nodes_after);
		change.changed_nodes_after.push_back(new_node->childs[1]);
		propogate_change_down(new_node->childs[1], x, nftrs, change.changed_nodes_after);

		if (change.changed_nodes_before.size() != change.changed_nodes_after.size())
		{
			printf("bug in change mark changes: before sie=%zu, after_size=%zu\n",
				change.changed_nodes_before.size(), change.changed_nodes_after.size());
			throw logic_error("bug");
		}
		//new_node->list_all_nodes(change.changed_nodes_after);
	}
	else {
#ifdef DEBUG_VERBOSE
		printf("do_change: delete %p\n", new_node);
#endif
		delete new_node;
		return change;
	}

	if (new_node->parent == NULL)
		root = new_node;
	else  //change in tree structure - parent of change will point to new:
		if (new_node->parent->childs[0] == selected_node)
			new_node->parent->childs[0] = new_node;
		else
			new_node->parent->childs[1] = new_node;

	return change;
}

void bart_tree::get_avalible_grow(const vector<float> &x, int nftrs, vector<bart_node *> &good_idx) const {
	//not in bottom and has split:
	good_idx.clear();
	vector<bart_node *> all_nodes;
	root->list_all_nodes(all_nodes);
	for (size_t i = 0; i < all_nodes.size(); ++i)
	{
		if (all_nodes[i]->childs[0] != NULL)
			continue; //it's not bottom node
					  //check that it has split feature:
		bool has_sp = false;
		for (size_t f = 0; f < nftrs && !has_sp; ++f)
			has_sp = has_split(all_nodes[i], x, nftrs, (int)f);
		if (has_sp)
			good_idx.push_back(all_nodes[i]);
	}
}

void bart_tree::get_avalible_feats_grow(const bart_node *selected_node, const vector<float> &x, int nftrs, vector<int> &good_idx) {
	good_idx.clear();
	if (selected_node->childs[0] != NULL)
		return; //it's not bottom node
				//check that it has split feature:
	for (size_t f = 0; f < nftrs; ++f)
		if (has_split(selected_node, x, nftrs, (int)f))
			good_idx.push_back((int)f);
}

tree_change_details bart_tree::do_grow(const vector<float> &x, const vector<float> &y) {
	tree_change_details change;
	change.action = 0;
	int nsamples = (int)y.size();
	int nftrs = (int)x.size() / nsamples;

	vector<bart_node *> avalible_nodes;
	get_avalible_grow(x, nftrs, avalible_nodes);
	if (avalible_nodes.empty())
		return change; //not possible
	uniform_int_distribution<> select_node(0, (int)avalible_nodes.size() - 1);
	change.num_node_selection = (int)avalible_nodes.size();
	//GET availble split var not in bottom:
	bart_node *selected_node = avalible_nodes[select_node(_rnd_gen)];
	vector<int> avalible_feats;
	get_avalible_feats_grow(selected_node, x, nftrs, avalible_feats);

	uniform_int_distribution<> select_feat(0, (int)avalible_feats.size() - 1);
	//split selected_node which is leaf
	selected_node->feature_number = avalible_feats[select_feat(_rnd_gen)];
	selected_node->num_feature_options = (int)avalible_feats.size();
	vector<vector<int>> obs_list;
	if (select_split(selected_node, x, nftrs, obs_list)) {
		bart_node *old_node = new bart_node;
		old_node->observation_indexes = selected_node->observation_indexes;
		old_node->parent = selected_node->parent;
		selected_node->childs[0] = new bart_node;
		selected_node->childs[1] = new bart_node;

		selected_node->childs[0]->parent = selected_node;
		selected_node->childs[1]->parent = selected_node;
		selected_node->childs[0]->observation_indexes.swap(obs_list[0]);
		selected_node->childs[1]->observation_indexes.swap(obs_list[1]);

		change.changed_nodes_before.push_back(old_node);
		change.changed_nodes_after.push_back(selected_node);

		change.changed_nodes_before.push_back(NULL);
		change.changed_nodes_after.push_back(selected_node->childs[0]);
		change.changed_nodes_before.push_back(NULL);
		change.changed_nodes_after.push_back(selected_node->childs[1]);
	}
	else
		return change;

	return change;
}

void bart_tree::get_avalible_prune(const vector<float> &x, int nftrs, vector<bart_node *> &good_idx) {
	good_idx.clear();
	vector<bart_node *> all_nodes;
	root->list_all_nodes(all_nodes);
	for (size_t i = 0; i < all_nodes.size(); ++i)
	{
		if (all_nodes[i]->childs[0] == NULL)
			continue; //it's bottom node
					  //check that it is parent of bottom node:
		if (all_nodes[i]->childs[0]->childs[0] == NULL &&
			all_nodes[i]->childs[1]->childs[0] == NULL)
			good_idx.push_back(all_nodes[i]);
	}
}

tree_change_details bart_tree::do_prune(const vector<float> &x, const vector<float> &y) {
	tree_change_details change;
	change.action = 1;
	int nsamples = (int)y.size();
	int nftrs = (int)x.size() / nsamples;

	vector<bart_node *> avalible_nodes;
	get_avalible_prune(x, nftrs, avalible_nodes);
	if (avalible_nodes.empty())
		return change; //not possible
	uniform_int_distribution<> select_node(0, (int)avalible_nodes.size() - 1);
	change.num_node_selection = (int)avalible_nodes.size();
	//GET availble split var not in bottom:
	bart_node *selected_node = avalible_nodes[select_node(_rnd_gen)];

	bart_node *old_node = new bart_node(*selected_node);
	old_node->childs[0]->parent = old_node;
	old_node->childs[1]->parent = old_node;
	//delete childs
	selected_node->childs[0] = NULL;
	selected_node->childs[1] = NULL;
	selected_node->mark_change = true;
	selected_node->feature_number = -1;
	selected_node->split_index = -1;

	change.changed_nodes_before.push_back(old_node);
	change.changed_nodes_after.push_back(selected_node);

	change.changed_nodes_before.push_back(old_node->childs[0]);
	change.changed_nodes_before.push_back(old_node->childs[1]);
	change.changed_nodes_after.push_back(NULL);
	change.changed_nodes_after.push_back(NULL);

	return change;
}

//will list the child => has unique parent
void bart_tree::get_avalible_swap(const vector<float> &x, int nftrs, vector<bart_node *> &good_idx) {
	good_idx.clear();
	vector<bart_node *> all_nodes;
	root->list_all_nodes(all_nodes);
	for (size_t i = 0; i < all_nodes.size(); ++i)
	{
		if (all_nodes[i]->childs[0] == NULL || all_nodes[i]->parent == NULL)
			continue; //it's bottom node or root so no parent

					  //check that it is parent of bottom node:
		good_idx.push_back(all_nodes[i]);
	}
}

tree_change_details bart_tree::do_swap(const vector<float> &x, const vector<float> &y) {
	tree_change_details change;
	change.action = 3;
	int nsamples = (int)y.size();
	int nftrs = (int)x.size() / nsamples;

	vector<bart_node *> avalible_nodes;
	get_avalible_swap(x, nftrs, avalible_nodes);
	if (avalible_nodes.empty())
		return change; //not possible
	uniform_int_distribution<> select_node(0, (int)avalible_nodes.size() - 1);
	change.num_node_selection = (int)avalible_nodes.size();
	//GET availble split var not in bottom:
	bart_node *selected_child = avalible_nodes[select_node(_rnd_gen)];
	int child_idx = 0;
	if (selected_child->parent->childs[0] == selected_child)
		child_idx = 0;

	else
		child_idx = 1;
	bart_node *other_child = selected_child->parent->childs[1 - child_idx];
	bool change_both = selected_child->feature_number == other_child->feature_number &&
		selected_child->split_index == other_child->split_index;


	int child_rule_feature_number = selected_child->feature_number;
	int child_rule_split_index = selected_child->split_index;
	//copy and list all copt node before change
	bart_node *parent_swap;
	selected_child->parent->deep_clone(parent_swap);
	parent_swap->list_all_nodes(change.changed_nodes_before);

	//swap - child parent rule:
	selected_child->feature_number = selected_child->parent->feature_number;
	selected_child->split_index = selected_child->parent->split_index;
	selected_child->parent->feature_number = child_rule_feature_number;
	selected_child->parent->split_index = child_rule_split_index;
	if (change_both) {
		other_child->feature_number = selected_child->feature_number;
		other_child->split_index = selected_child->split_index;
	}
	//propogate and update new swap state:
	selected_child->parent->mark_change = true;
	change.changed_nodes_after.push_back(selected_child->parent);
	propogate_change_down(selected_child->parent, x, nftrs, change.changed_nodes_after);

	return change;
}

float bart_tree::score_leaf(const vector<float> &y, const vector<int> &obs_indexes) {
	//returns mean value
	if (obs_indexes.empty())
		return 0;
	double mean_resp = 0;
	for (int ind : obs_indexes)
		mean_resp += y[ind];
	mean_resp /= obs_indexes.size();

	double posterior_mean, posterior_var;
	posterior_var = 1 / (1 / params.sigsq_mu + obs_indexes.size() / sigma);
	posterior_mean = (params.mean_mu / params.sigsq_mu + obs_indexes.size()*mean_resp / sigma) * posterior_var;
	//return posterior_mean;

	normal_distribution<> nrm_dist(posterior_mean, sqrt(posterior_var));
	float res = (float)nrm_dist(_rnd_gen);

	return res;
}

void bart_tree::predict(const vector<float> &x, int nSamples, vector<float> &scores) const {
	int nftrs = (int)x.size() / nSamples;
	scores.resize(nSamples);
	if (root == NULL)
		return; //empty tree, version 0 - all scores will be set to 0
#pragma omp parallel for
	for (int i = 0; i < nSamples; ++i)
	{
		int start_pos = (int)i * nftrs;
		bart_node *p = root;
		while (p->childs[0] != NULL) { //while not leaf, propogate down
			const vector<float> &sotrted_vals = feature_to_sorted_vals[p->feature_number];
			if (x[start_pos + p->feature_number] <= sotrted_vals[p->split_index])
				p = p->childs[0];
			else
				p = p->childs[1];
		}
		scores[i] = p->node_value;
	}
}

void bart_tree::predict_on_train(const vector<float> &x, int nSamples, vector<float> &scores) const {
	scores.resize(nSamples);
	if (root == NULL)
		return;
	vector<bart_node *> all_nodes;
	root->list_all_nodes(all_nodes);
	for (size_t i = 0; i < all_nodes.size(); ++i)
	{
		if (all_nodes[i]->childs[0] != NULL)
			continue; //it's not bottom node
					  //fetch results for leaf:
		for (int ind : all_nodes[i]->observation_indexes)
			scores[ind] = all_nodes[i]->node_value;
	}
}

int bart_node::depth() {
	int dep = 0;
	bart_node *p = this;
	while (p != NULL) {
		++dep;
		p = p->parent;
	}
	return dep;
}

float bart_node::variance(const vector<float> &y) {
	float var = 0;
	if (observation_indexes.empty())
		return 0;

	for (int ind : observation_indexes)
		var += (y[ind] - node_value) * (y[ind] - node_value);
	var = (var / observation_indexes.size()) * ((int)observation_indexes.size() - 1);
	return var;
}


double bart_tree::node_data_likelihood(const vector<bart_node *> &leaf_node, const vector<float> &x, int nftrs,
	const vector<float> &y) {
	//if ()

	double sum_p = 0;
	double a = 1.0 / params.sigsq_mu;
	for (size_t i = 0; i < leaf_node.size(); ++i)
	{
		int nl = (int)leaf_node[i]->observation_indexes.size();
		double leaf_l2 = 0, mean_resp = 0;
		for (int ind : leaf_node[i]->observation_indexes)
			mean_resp += y[ind];
		mean_resp /= nl;
		for (int ind : leaf_node[i]->observation_indexes)
			leaf_l2 += (y[ind] - mean_resp)* (y[ind] - mean_resp);

		sum_p += 0.5*(log(a) - log(a + nl / sigma));
		sum_p -= 0.5*leaf_l2 / sigma;
		sum_p -= 0.5*mean_resp*mean_resp*nl*a / (sigma * a + nl);
		/*printf("mean_resp=%lf, leaf_l2=%lf, sigma2=%lf, 1arg=%lf, 2arg=%lf, 3args=%lf\n", mean_resp, leaf_l2,
		sigma, 0.5*(log(a) - log(a + nl / sigma)), 0.5*leaf_l2 / sigma,
		0.5*mean_resp*mean_resp*nl*a / (sigma * a + nl));*/
		/*sum_p += -0.5*nl / log(sigma);
		sum_p += 0.5*(log(sigma) - log(sigma + nl * params.sigsq_mu));
		sum_p += -0.5 / sigma*(leaf_l2 + mean_resp*mean_resp / nl - (mean_resp*mean_resp) / (nl + sigma / params.sigsq_mu));
		*/
	}
	float res = (float)-sum_p;

	return res;
}

void bart_tree::calc_likelihood(const vector<float> &x, int nftrs, const vector<float> &y) {
	//traverse tree and calc leaf node_values and tree likelihood:
	vector<bart_node *> all_nodes;
	root->list_all_nodes(all_nodes);

	//calc for each node:
	double tree_structure_prior = 0;
	double data_likelihood;
	vector<bart_node *> leaf_nodes;
	leaf_nodes.reserve(all_nodes.size());
	for (size_t i = 0; i < all_nodes.size(); ++i)
	{
		if (all_nodes[i]->childs[0] != NULL) {//not leaf 
			tree_structure_prior += -log(params.alpha) + params.beta * log(1 + all_nodes[i]->depth());
			tree_structure_prior += log(all_nodes[i]->num_feature_options) + log(all_nodes[i]->num_split_options);
			continue; //not leaf or no change
		}
		tree_structure_prior += -log(1 - params.alpha * pow(1 + all_nodes[i]->depth(), -params.beta));

		if (all_nodes[i]->mark_change)
			all_nodes[i]->node_value = score_leaf(y, all_nodes[i]->observation_indexes);
		all_nodes[i]->mark_change = false; //mark done
										   //calc data likelihood:
		leaf_nodes.push_back(all_nodes[i]);
	}
	data_likelihood = node_data_likelihood(leaf_nodes, x, nftrs, y);
	tree_loglikelihood = tree_structure_prior + data_likelihood;
	//#ifdef DEBUG_VERBOSE
	/*printf("tree_loglikelihood=%2.1f, data_likelihood=%2.1f, sigma=%2.5f",
	tree_loglikelihood, data_likelihood, sigma);*/
	//#endif // DEBUG_VERBOSE

}

void bart_node::deep_clone(bart_node *&target) {
	target = new bart_node(*this);
	if (childs[0] != NULL) {
		childs[0]->deep_clone(target->childs[0]);
		childs[1]->deep_clone(target->childs[1]);
		target->childs[0]->parent = target;
		target->childs[1]->parent = target;
	}
}

void bart_tree::clone_tree(bart_tree &tree) {
	tree.params = params;
	tree.action_priors = action_priors;
	tree.feature_to_sorted_vals = feature_to_sorted_vals;
	tree.feature_to_val_index = feature_to_val_index;
	tree.tree_loglikelihood = tree_loglikelihood;
	tree._rnd_gen = _rnd_gen;
	//deep copy tree
	root->deep_clone(tree.root);
}

void add_vectors(vector<float> &target, vector<float> &add, bool minus_sign = false) {
	for (size_t i = 0; i < target.size(); ++i)
		target[i] += minus_sign ? -add[i] : add[i];
}

void BART::transform_y(vector<float> &y) {
	if (tree_params.data_prior_type == bart_data_prior_type::classification)
		return; //no transformation

	trans_y_b = y[0];
	trans_y_max = y[0];
	for (size_t i = 1; i < y.size(); ++i)
	{
		if (y[i] < trans_y_b)
			trans_y_b = y[i];
		if (y[i] > trans_y_max)
			trans_y_max = y[i];
	}
	if (trans_y_b == trans_y_max)
		throw logic_error("y vector is equal in all elements!");

	for (size_t i = 0; i < y.size(); ++i)
		y[i] = ((y[i] - trans_y_b) / (trans_y_max - trans_y_b)) - 0.5;
}

void BART::untransform_y(vector<float> &y) const {
	if (tree_params.data_prior_type == bart_data_prior_type::classification)
		return; //no transformation
	for (size_t i = 0; i < y.size(); ++i)
		y[i] = (y[i] + 0.5) * (trans_y_max - trans_y_b) + trans_y_b;
}

void BART::init_hyper_parameters(const vector<float> &residuals) {
	int nSamples = (int)residuals.size();
	double sample_var_y = 0;
	if (tree_params.data_prior_type == regression_mean_shift) {
		double mean_y = 0;
		for (size_t i = 0; i < residuals.size(); ++i) {
			sample_var_y += static_cast<double>(residuals[i]) * static_cast<double>(residuals[i]);
			mean_y += residuals[i];
		}
		mean_y /= nSamples;
		sample_var_y /= nSamples;
		sample_var_y = (sample_var_y - mean_y * mean_y);
		tree_params.set_regression(ntrees, sample_var_y);
	}
	else
		tree_params.set_classification(ntrees);
	printf("running with %d restarts, %d trees, alpha=%2.2f, beta=%2.2f, k=%2.1f"
		", min_obs_in_node=%d, nu=%2.1f, lambda=%f\n",
		restart_count, ntrees, tree_params.alpha, tree_params.beta,
		tree_params.k, tree_params.min_obs_in_node, tree_params.nu,
		tree_params.lambda);
}

void BART::update_sigma_param(boost::mt19937 &rng, const vector<float> &residuals, double &sigma) {
	if (tree_params.data_prior_type == regression_mean_shift) {
		//update sigsq for loglikelohood based on current residuals
		double sse = 0;
		for (double e : residuals)
			sse += e * e;
		//we're sampling from sigsq ~ InvGamma((nu + n) / 2, 1/2 * (sum_i error^2_i + lambda * nu))
		//which is equivalent to sampling (1 / sigsq) ~ Gamma((nu + n) / 2, 2 / (sum_i error^2_i + lambda * nu))
		boost::random::weibull_distribution<> inv_gam((tree_params.nu + residuals.size()) / 2,
			2 / (sse + tree_params.nu*tree_params.lambda));

		boost::variate_generator<boost::mt19937&,
			boost::random::weibull_distribution<> > var_gamma(rng, inv_gam);
		sigma = var_gamma();
	}
}

void BART::update_latent_z_params(boost::random::random_number_generator<boost::mt19937> &rng_gen,
	const vector<float> &x, const vector<float> &y, const vector<bart_tree> &forest_trees,
	vector<float> &residuals) {
	int nSamples = (int)y.size();
	if (tree_params.data_prior_type == classification) {
		//prepare labels for classification:
		vector<float> full_pred(y.size());
		for (size_t k = 0; k < ntrees; ++k)
		{
			vector<float> current_tree_scores;
			forest_trees[k].predict_on_train(x, nSamples, current_tree_scores);
			add_vectors(full_pred, current_tree_scores, false);
		}

		vector<float> z_vec(y.size());
		boost::math::normal_distribution<> norm_dist(0, 1);
		for (size_t k = 0; k < y.size(); ++k)
		{
			//based on y, full_pred
			double u = rng_gen(1000000) / (double)1000000;
			double normal_c = boost::math::cdf(norm_dist, (y[k] > 0) * full_pred[k] - (y[k] <= 0) * full_pred[k]);
			normal_c = (1 - u) *normal_c + u;
			if (y[k] > 0) {
				z_vec[k] = full_pred[k] + boost::math::quantile(norm_dist, normal_c);
			}
			else {
				z_vec[k] = full_pred[k] - boost::math::quantile(norm_dist, normal_c);
			}
		}

		//calc residuals again:
		for (size_t k = 0; k < residuals.size(); ++k)
			residuals[k] = z_vec[k] - full_pred[k];

		//printf("Z_vec, AUC=%2.3f\n", get_preds_auc_q(z_vec, y));
	}
}

void BART::learn(const vector<float> &x, const vector<float> &y) {
	int nSamples = (int)y.size();
	nftrs = int(x.size() / y.size());
	vector<float> residuals(y.begin(), y.end());

	//Build all trees together in Monte-Carlo-Markov-Chain manner using Metropolis-Hasting:
	double best_forest_likelihood = 0;
	vector<bart_tree> best_forest(ntrees);

	vector<vector<float>> feature_to_sorted_vals(nftrs);
	vector<unordered_map<float, int>> feature_to_val_index(nftrs);
	for (size_t i = 0; i < nftrs; ++i) {
		unordered_set<float> uniq_vals;
		for (int k = 0; k < y.size(); ++k)
			if (uniq_vals.find(x[static_cast<size_t>(k)*static_cast<size_t>(nftrs) + i]) == uniq_vals.end()) {
				feature_to_sorted_vals[i].push_back(x[static_cast<size_t>(k)*static_cast<size_t>(nftrs) + i]);
				uniq_vals.insert(x[static_cast<size_t>(k)*static_cast<size_t>(nftrs) + i]);
			}
	}
	for (size_t i = 0; i < nftrs; ++i)
		sort(feature_to_sorted_vals[i].begin(), feature_to_sorted_vals[i].end());
	//populated feature_to_val_index:
	for (size_t i = 0; i < nftrs; ++i) {
		const vector<float> &sorted_vals = feature_to_sorted_vals[i];
		for (int k = 0; k < sorted_vals.size(); ++k)
			feature_to_val_index[i][sorted_vals[k]] = k;
	}

	vector<int> more_than;
	for (size_t i = 0; i < nftrs; ++i)
		if (feature_to_sorted_vals[i].size() > 1000)
			more_than.push_back((int)i);
	if (!more_than.empty())
		printf("Warning has %zu features with more than 1000 unique value."
			" you may bin the feature before learn to speedup\n", more_than.size());
	for (size_t i = 0; i < ntrees; ++i)
		_trees[i].feature_to_sorted_vals = feature_to_sorted_vals;
	for (size_t i = 0; i < ntrees; ++i)
		_trees[i].feature_to_val_index = feature_to_val_index;

	bool best_is_last = false;
	int progress = 0;
	time_t start_t = time(NULL);
	time_t last_print_tm = start_t;
	int tot_count = restart_count * iter_count* ntrees;
	transform_y(residuals); //first transform of y values if needed!

							//initialize hyper parameters:
	init_hyper_parameters(residuals);
	for (int loop = 0; loop < restart_count; ++loop)
	{
		random_device rd;
		boost::mt19937 rng(rd());
		boost::random::random_number_generator<boost::mt19937> rng_gen(rng);
		//printf("doing restart %d\n", loop);
		vector<bart_tree> forest_trees(ntrees);
		for (size_t i = 0; i < ntrees; ++i)
			forest_trees[i].params = tree_params;
		double sigma = 1;
		update_sigma_param(rng, {}, sigma); //first time call with empty vector
		for (size_t i = 0; i < iter_count; ++i)
		{
			update_latent_z_params(rng_gen, x, y, forest_trees, residuals);
			//printf("\tdoing iteration %zu\n", i);
			for (size_t k = 0; k < ntrees; ++k)
			{
				//printf("\t\tdoing iteration %zu with tree %zu out of %d\n", i, k, ntrees);
				vector<float> current_tree_scores, new_tree_scores;
				//get current tree scores to remove from residuals:
				forest_trees[k].predict_on_train(x, nSamples, current_tree_scores);
				//subtract _tree[k] scores from current residuals:
				add_vectors(residuals, current_tree_scores, false);
				//do iteration for current tree:
				forest_trees[k].set_sigma(sigma);
				forest_trees[k].next_gen_tree(x, residuals);
				//printf("doing iteration %zu with tree %zu out of %d - after raise\n", i, k, ntrees);
				//calc new scores and add to residuals:
				forest_trees[k].predict_on_train(x, nSamples, new_tree_scores);
				add_vectors(residuals, new_tree_scores, true);

				++progress;
				int duration = (int)difftime(time(NULL), last_print_tm);
				if (duration > 30) {
					last_print_tm = time(NULL);
					int time_elapsed = (int)difftime(time(NULL), start_t);
					int estimate_time = int(double(tot_count - progress) / double(progress) * double(time_elapsed));
					char buffer[1000];
					snprintf(buffer, sizeof(buffer), "Processed %d out of %d(%2.2f%%) time elapsed: %2.1f Minutes, "
						"estimate time to finish %2.1f Minutes",
						progress, (int)tot_count, 100.0*(progress / float(tot_count)), float(time_elapsed) / 60,
						float(estimate_time) / 60.0);
					cout << string(buffer) << endl;
				}
			}

			best_is_last = false;
			if (i >= burn_count) {
				//TODO: use restarts
				//search for best prior till now
				double forest_prior = 0; //log_likelihood should minimize
				for (size_t k = 0; k < ntrees; ++k)
					if (forest_trees[k].tree_loglikelihood < 0) {
						printf("ERROR: wrong parameters reaches tree_prior 0 in tree=%zu\n", k);
						throw logic_error("ERROR: wrong parameters reaches tree_prior 0");
					}
					else
						forest_prior += forest_trees[k].tree_loglikelihood;

				if (i == burn_count || forest_prior < best_forest_likelihood) {
					best_forest_likelihood = forest_prior;
					for (size_t j = 0; j < best_forest.size(); ++j) {
						best_forest[j].clear_tree_mem(best_forest[j].root);
						forest_trees[j].clone_tree(best_forest[j]);
					}
					best_is_last = i == iter_count - 1;
				}

			}

			update_sigma_param(rng, residuals, sigma);
		}
		if (!best_is_last) { //clear memory for loop if can
			for (size_t j = 0; j < best_forest.size(); ++j)
				forest_trees[j].clear_tree_mem(forest_trees[j].root);
		}

	}

	//save best_forest, no neep for deep copy:
	for (size_t i = 0; i < ntrees; ++i)
	{
		_trees[i].action_priors = best_forest[i].action_priors;
		_trees[i].feature_to_sorted_vals = best_forest[i].feature_to_sorted_vals;
		_trees[i].feature_to_val_index = best_forest[i].feature_to_val_index;
		_trees[i].params = best_forest[i].params;
		_trees[i].tree_loglikelihood = best_forest[i].tree_loglikelihood;
		_trees[i]._rnd_gen = best_forest[i]._rnd_gen;
		_trees[i].root = best_forest[i].root;
#ifdef DEBUG_VERBOSE
		printf("tree %zu with loglikelihood=%lf:\n", i, _trees[i].tree_loglikelihood);
		_trees[i].root->print_tree(_trees[i].feature_to_sorted_vals);
#endif // DEBUG_VERBOSE
	}

	int duration_sec = (int)difftime(time(NULL), start_t);
	printf("Done learining in %d seconds\n", duration_sec);
}

void BART::predict(const vector<float> &x, int nSamples, vector<float> &scores) const {
	if (static_cast<size_t>(nftrs) * static_cast<size_t>(nSamples) != x.size())
		throw invalid_argument("x vector is in wrong size."
			" should be nsamples observation X number_of_features\n");
	scores.resize(nSamples);
	for (size_t t = 0; t < _trees.size(); ++t)
	{
		vector<float> tree_scores;
		_trees[t].predict(x, nSamples, tree_scores);
		add_vectors(scores, tree_scores, false);
	}
	untransform_y(scores);
}
