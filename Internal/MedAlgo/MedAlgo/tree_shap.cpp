#include "tree_shap.h"
#include <omp.h>
#include "medial_utilities/medial_utilities/globalRNG.h"
#include <MedAlgo/MedAlgo/MedLM.h>
#include <Eigen/Core>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL LOG_DEF_LEVEL

bool comp_score_flt_str(const pair<float, string> &pr1, const pair<float, string> &pr2) {
	return abs(pr1.first) > abs(pr2.first); //bigger is better
}

TreeEnsemble::TreeEnsemble() {
	is_allocate = false;
}

TreeEnsemble::TreeEnsemble(int *children_left, int *children_right, int *children_default, int *features,
	tfloat *thresholds, tfloat *values, tfloat *node_sample_weights,
	unsigned max_depth, unsigned tree_limit, tfloat base_offset,
	unsigned max_nodes, unsigned num_outputs) :
	children_left(children_left), children_right(children_right),
	children_default(children_default), features(features), thresholds(thresholds),
	values(values), node_sample_weights(node_sample_weights),
	max_depth(max_depth), tree_limit(tree_limit),
	base_offset(base_offset), max_nodes(max_nodes), num_outputs(num_outputs), is_allocate(true) {}

void TreeEnsemble::get_tree(TreeEnsemble &tree, const unsigned i) const {
	const unsigned d = i * max_nodes;

	tree.children_left = children_left + d;
	tree.children_right = children_right + d;
	tree.children_default = children_default + d;
	tree.features = features + d;
	tree.thresholds = thresholds + d;
	tree.values = values + d * num_outputs;
	tree.node_sample_weights = node_sample_weights + d;
	tree.max_depth = max_depth;
	tree.tree_limit = 1;
	tree.base_offset = base_offset;
	tree.max_nodes = max_nodes;
	tree.num_outputs = num_outputs;
	tree.is_allocate = true;
}

void TreeEnsemble::allocate(unsigned tree_limit_in, unsigned max_nodes_in, unsigned num_outputs_in) {
	tree_limit = tree_limit_in;
	max_nodes = max_nodes_in;
	num_outputs = num_outputs_in;
	children_left = new int[tree_limit * max_nodes];
	children_right = new int[tree_limit * max_nodes];
	children_default = new int[tree_limit * max_nodes];
	features = new int[tree_limit * max_nodes];
	thresholds = new tfloat[tree_limit * max_nodes];
	values = new tfloat[tree_limit * max_nodes * num_outputs];
	node_sample_weights = new tfloat[tree_limit * max_nodes];
	is_allocate = true;
}

void TreeEnsemble::free() {
	if (is_allocate) {
		delete[] children_left;
		delete[] children_right;
		delete[] children_default;
		delete[] features;
		delete[] thresholds;
		delete[] values;
		delete[] node_sample_weights;
		children_left = NULL;
		children_right = NULL;
		children_default = NULL;
		features = NULL;
		thresholds = NULL;
		values = NULL;
		node_sample_weights = NULL;
	}
	is_allocate = false;
}

/* Adjust a model, conditioning upon a sample and mask */
void TreeEnsemble::create_adjusted_tree(ExplanationDataset& instance, const int *mask, unsigned *feature_sets, TreeEnsemble& adjusted) {

	adjusted.allocate(tree_limit, max_nodes, num_outputs);
	adjusted.base_offset = base_offset;
	adjusted.max_depth = max_depth;
	adjusted.max_nodes = max_nodes;
	adjusted.num_outputs = num_outputs;
	adjusted.tree_limit = tree_limit;
	adjusted.is_allocate = true;

	fill_adjusted_tree(0, instance, mask, feature_sets, adjusted);

}

void TreeEnsemble::calc_feature_contribs_conditional(MedMat<float> &mat_x_in, unordered_map<string, float> contiditional_variables, MedMat<float> &mat_x_out, MedMat<float> &mat_contribs)
{
	//create adjusted tree by setting a row according to the mask
	vector<tfloat> features_vec(mat_x_in.get_ncols(), 0);
	vector<pair<int, float>> ind_to_val_vec;
	vector<int> mask(mat_x_in.get_ncols(), 0);

	for (auto & contiditional_variable : contiditional_variables)
	{
		int i = find_in_feature_names(mat_x_in.signals, contiditional_variable.first);
		features_vec[i] = contiditional_variable.second;
		ind_to_val_vec.push_back({ i,contiditional_variable.second });
		mask[i] = 1;
	}
	// prepare features_sets (used for grouping)
	vector<unsigned> feature_sets(mat_x_in.ncols);
	for (size_t i = 0; i < feature_sets.size(); i++)
		feature_sets[i] = (int)i;

	unique_ptr<bool[]> x_missing = unique_ptr<bool[]>(new bool[mat_x_in.ncols]);
	for (size_t i = 0; i < mat_x_in.ncols; ++i)
	{
		x_missing.get()[i] = false;
	}

	tfloat *R_p = NULL; // R.data()
	unique_ptr<bool[]> R_missing = NULL;
	//R_missing = unique_ptr<bool>(new bool[x_mat.m.size()]);
	int num_R = 0;
	double y = 0;
	int num_X = 1; // (int)y.size();
	int M = mat_x_in.ncols;

	// adjust model according to features_vec
	TreeEnsemble tree;
	vector<TreeEnsemble> adjusted_trees;
	ExplanationDataset instance(features_vec.data(), x_missing.get(), &y, R_p, R_missing.get(), num_X, M, num_R, mat_x_in.ncols);
	adjusted_trees.resize(this->tree_limit);

	for (unsigned j = 0; j < this->tree_limit; ++j) {
		this->get_tree(tree, j);
		tree.create_adjusted_tree(instance, mask.data(), feature_sets.data(), adjusted_trees[j]);
	}

	// run over rows, 
	mat_x_out.ncols = mat_x_in.ncols;
	mat_contribs.ncols = mat_x_in.ncols + 1; // +1 for bias 
	for (int i = 0; i < mat_x_in.get_nrows(); i++)
	{
		vector<float> row;
		mat_x_in.get_row(i, row);
		vector<double> row_tmp(row.begin(), row.end());

		// skip those who do not meet condtions
		bool skip_row = false;
		for (auto & ind_to_val : ind_to_val_vec)
		{
			if (row[ind_to_val.first] != ind_to_val.second)
			{
				skip_row = true;
				break;
			}
		}
		if (skip_row)
		{
			continue;
		}
		//get condtional shap values
		// Get conditioned SHAP values
		vector<tfloat> instance_contrib(mat_x_in.ncols + 1, 0);
		for (auto &adjusted_tree : adjusted_trees)
		{
			// vector<tfloat> instance_temp_contrib(mat_x_in.ncols);
			ExplanationDataset instance_eval(row_tmp.data(), x_missing.get(), &y, R_p, R_missing.get(), num_X, M, num_R, mat_x_in.ncols);
			tree_shap(adjusted_tree, instance_eval, instance_contrib.data(), 0, 0, feature_sets.data());
		}
		mat_x_out.add_rows(row);
		//		mat_x_out.recordsMetadata.push_back(mat_x_in.recordsMetadata[i]);
		mat_contribs.add_rows(instance_contrib);
		//mat_contribs.recordsMetadata.push_back(mat_x_in.recordsMetadata[i]);
	}

	// free trees
	for (auto &adjusted_tree : adjusted_trees)
	{
		adjusted_tree.free();
	}
}


inline void copy_node(TreeEnsemble * const origTree, int orig_index, TreeEnsemble& newTree, int new_index) {

	newTree.children_left[new_index] = origTree->children_left[orig_index];
	newTree.children_default[new_index] = origTree->children_default[orig_index];
	newTree.children_right[new_index] = origTree->children_right[orig_index];
	newTree.values[new_index] = origTree->values[orig_index];
	newTree.node_sample_weights[new_index] = origTree->node_sample_weights[orig_index];
	newTree.features[new_index] = origTree->features[orig_index];
	newTree.thresholds[new_index] = origTree->thresholds[orig_index];;
}

void TreeEnsemble::fill_adjusted_tree(int node_index, ExplanationDataset& instance, const int *mask, unsigned *feature_sets, TreeEnsemble& adjusted) {

	if (children_right[node_index] < 0) {
		// leaf node
		copy_node(this, node_index, adjusted, node_index);
	}
	else {
		// internal node
		const unsigned split_index = features[node_index];
		const unsigned split_set = feature_sets[split_index];

		fill_adjusted_tree(children_left[node_index], instance, mask, feature_sets, adjusted);
		fill_adjusted_tree(children_right[node_index], instance, mask, feature_sets, adjusted);


		if (mask[split_set] == 0) {
			// Not conditioned upon
			copy_node(this, node_index, adjusted, node_index);
			adjusted.node_sample_weights[node_index] = adjusted.node_sample_weights[children_left[node_index]] + adjusted.node_sample_weights[children_right[node_index]];
			adjusted.values[node_index] = (adjusted.values[children_left[node_index]] * adjusted.node_sample_weights[children_left[node_index]] +
				adjusted.values[children_right[node_index]] * adjusted.node_sample_weights[children_right[node_index]]) / adjusted.node_sample_weights[node_index];
		}
		else {
			// Conditioned upon
			// find which branch is "hot" (meaning x would follow it)
			unsigned hot_index = 0;
			if (instance.X_missing[split_index]) {
				hot_index = children_default[node_index];
			}
			else if (instance.X[split_index] <= thresholds[node_index]) {
				hot_index = children_left[node_index];
			}
			else {
				hot_index = children_right[node_index];
			}

			// Override with hot-index.
			copy_node(&adjusted, hot_index, adjusted, node_index);
		}
	}
}

tfloat TreeEnsemble::predict(ExplanationDataset& instance, int node_index) {

	if (children_right[node_index] < 0)
		return values[node_index];
	else {
		const unsigned split_index = features[node_index];
		unsigned hot_index = 0;
		if (instance.X_missing[split_index]) {
			hot_index = children_default[node_index];
		}
		else if (instance.X[split_index] <= thresholds[node_index]) {
			hot_index = children_left[node_index];
		}
		else {
			hot_index = children_right[node_index];
		}
		return this->predict(instance, hot_index);
	}
}

ExplanationDataset::ExplanationDataset() {}

ExplanationDataset::ExplanationDataset(tfloat *X, bool *X_missing, tfloat *y, tfloat *R, bool *R_missing, unsigned num_X,
	unsigned M, unsigned num_R) :
	X(X), X_missing(X_missing), y(y), R(R), R_missing(R_missing), num_X(num_X), M(M), num_R(num_R), num_Exp(M) {}
ExplanationDataset::ExplanationDataset(tfloat *X, bool *X_missing, tfloat *y, tfloat *R, bool *R_missing, unsigned num_X,
	unsigned M, unsigned num_R, unsigned num_Exp) :
	X(X), X_missing(X_missing), y(y), R(R), R_missing(R_missing), num_X(num_X), M(M), num_R(num_R), num_Exp(num_Exp) {}

void ExplanationDataset::get_x_instance(ExplanationDataset &instance, const unsigned i) const {
	instance.M = M;
	instance.X = X + i * M;
	instance.X_missing = X_missing + i * M;
	instance.num_X = 1;
	instance.num_Exp = num_Exp;
}

PathElement::PathElement() {}
PathElement::PathElement(int i, tfloat z, tfloat o, tfloat w) :
	feature_index(i), zero_fraction(z), one_fraction(o), pweight(w) {}


inline tfloat logistic_transform(const tfloat margin, const tfloat y) {
	return 1 / (1 + exp(-margin));
}

inline tfloat logistic_nlogloss_transform(const tfloat margin, const tfloat y) {
	return log(1 + exp(margin)) - y * margin; // y is in {0, 1}
}

inline tfloat squared_loss_transform(const tfloat margin, const tfloat y) {
	return (margin - y) * (margin - y);
}


inline tfloat *tree_predict(unsigned i, const TreeEnsemble &trees, const tfloat *x, const bool *x_missing) {
	const unsigned offset = i * trees.max_nodes;
	unsigned node = 0;
	while (true) {
		const unsigned pos = offset + node;
		const unsigned feature = trees.features[pos];

		// we hit a leaf so return a pointer to the values
		if (trees.children_left[pos] < 0) {
			return trees.values + pos * trees.num_outputs;
		}

		// otherwise we are at an internal node and need to recurse
		if (x_missing[feature]) {
			node = trees.children_default[pos];
		}
		else if (x[feature] <= trees.thresholds[pos]) {
			node = trees.children_left[pos];
		}
		else {
			node = trees.children_right[pos];
		}
	}
}

inline void dense_tree_predict(tfloat *out, const TreeEnsemble &trees, const ExplanationDataset &data, unsigned model_transform) {
	tfloat *row_out = out;
	const tfloat *x = data.X;
	const bool *x_missing = data.X_missing;

	// see what transform (if any) we have
	tfloat(*transform)(const tfloat margin, const tfloat y) = NULL;
	switch (model_transform) {
	case MODEL_TRANSFORM::logistic:
		transform = logistic_transform;
		break;

	case MODEL_TRANSFORM::logistic_nlogloss:
		transform = logistic_nlogloss_transform;
		break;

	case MODEL_TRANSFORM::squared_loss:
		transform = squared_loss_transform;
		break;
	}

	for (unsigned i = 0; i < data.num_X; ++i) {

		// add the base offset
		for (unsigned k = 0; k < trees.num_outputs; ++k) {
			row_out[k] += trees.base_offset;
		}

		// add the leaf values from each tree
		for (unsigned j = 0; j < trees.tree_limit; ++j) {
			const tfloat *leaf_value = tree_predict(j, trees, x, x_missing);

			for (unsigned k = 0; k < trees.num_outputs; ++k) {
				row_out[k] += leaf_value[k];
			}
		}

		// apply any needed transform
		if (transform != NULL) {
			const tfloat y_i = data.y == NULL ? 0 : data.y[i];
			for (unsigned k = 0; k < trees.num_outputs; ++k) {
				row_out[k] = transform(row_out[k], y_i);
			}
		}

		x += data.M;
		x_missing += data.M;
		row_out += trees.num_outputs;
	}
}

inline void tree_update_weights(unsigned i, TreeEnsemble &trees, const tfloat *x, const bool *x_missing) {
	const unsigned offset = i * trees.max_nodes;
	unsigned node = 0;
	while (true) {
		const unsigned pos = offset + node;
		const unsigned feature = trees.features[pos];

		// Record that a sample passed through this node
		trees.node_sample_weights[pos] += 1.0;

		// we hit a leaf so return a pointer to the values
		if (trees.children_left[pos] < 0) break;

		// otherwise we are at an internal node and need to recurse
		if (x_missing[feature]) {
			node = trees.children_default[pos];
		}
		else if (x[feature] <= trees.thresholds[pos]) {
			node = trees.children_left[pos];
		}
		else {
			node = trees.children_right[pos];
		}
	}
}

inline void dense_tree_update_weights(TreeEnsemble &trees, const ExplanationDataset &data) {
	const tfloat *x = data.X;
	const bool *x_missing = data.X_missing;

	for (unsigned i = 0; i < data.num_X; ++i) {

		// add the leaf values from each tree
		for (unsigned j = 0; j < trees.tree_limit; ++j) {
			tree_update_weights(j, trees, x, x_missing);
		}

		x += data.M;
		x_missing += data.M;
	}
}

inline void tree_saabas(tfloat *out, const TreeEnsemble &tree, const ExplanationDataset &data) {
	unsigned curr_node = 0;
	unsigned next_node = 0;
	while (true) {

		// we hit a leaf and are done
		if (tree.children_left[curr_node] < 0) return;

		// otherwise we are at an internal node and need to recurse
		const unsigned feature = tree.features[curr_node];
		if (data.X_missing[feature]) {
			next_node = tree.children_default[curr_node];
		}
		else if (data.X[feature] <= tree.thresholds[curr_node]) {
			next_node = tree.children_left[curr_node];
		}
		else {
			next_node = tree.children_right[curr_node];
		}

		// assign credit to this feature as the difference in values at the current node vs. the next node
		for (unsigned i = 0; i < tree.num_outputs; ++i) {
			out[feature * tree.num_outputs + i] += tree.values[next_node * tree.num_outputs + i] - tree.values[curr_node * tree.num_outputs + i];
		}

		curr_node = next_node;
	}
}

/**
* This runs Tree SHAP with a per tree path conditional dependence assumption.
*/
void dense_tree_saabas(tfloat *out_contribs, const TreeEnsemble& trees, const ExplanationDataset &data) {
	MedProgress progress("SHAPLEY_SAABAS", data.num_X, 15, 50);

	// build explanation for each sample
	for (int i = 0; i < data.num_X; ++i) {
		TreeEnsemble tree;
		ExplanationDataset instance;
		tfloat *instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
		data.get_x_instance(instance, i);

		// aggregate the effect of explaining each tree
		// (this works because of the linearity property of Shapley values)
		for (unsigned j = 0; j < trees.tree_limit; ++j) {
			trees.get_tree(tree, j);
			tree_saabas(instance_out_contribs, tree, instance);
		}

		// apply the base offset to the bias term
		for (unsigned j = 0; j < trees.num_outputs; ++j) {
			instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset;
		}

		progress.update();
	}
}


// extend our decision path with a fraction of one and zero extensions
inline void extend_path(PathElement *unique_path, unsigned unique_depth,
	tfloat zero_fraction, tfloat one_fraction, int feature_index) {
	unique_path[unique_depth].feature_index = feature_index;
	unique_path[unique_depth].zero_fraction = zero_fraction;
	unique_path[unique_depth].one_fraction = one_fraction;
	unique_path[unique_depth].pweight = (unique_depth == 0 ? 1.0f : 0.0f);
	for (int i = unique_depth - 1; i >= 0; i--) {
		unique_path[i + 1].pweight += one_fraction * unique_path[i].pweight * (i + 1)
			/ static_cast<tfloat>(unique_depth + 1);
		unique_path[i].pweight = zero_fraction * unique_path[i].pweight * (unique_depth - i)
			/ static_cast<tfloat>(unique_depth + 1);
	}
}

// undo a previous extension of the decision path
inline void unwind_path(PathElement *unique_path, unsigned unique_depth, unsigned path_index) {
	const tfloat one_fraction = unique_path[path_index].one_fraction;
	const tfloat zero_fraction = unique_path[path_index].zero_fraction;
	tfloat next_one_portion = unique_path[unique_depth].pweight;

	for (int i = unique_depth - 1; i >= 0; --i) {
		if (one_fraction != 0) {
			const tfloat tmp = unique_path[i].pweight;
			unique_path[i].pweight = next_one_portion * (unique_depth + 1)
				/ static_cast<tfloat>((i + 1) * one_fraction);
			next_one_portion = tmp - unique_path[i].pweight * zero_fraction * (unique_depth - i)
				/ static_cast<tfloat>(unique_depth + 1);
		}
		else {
			unique_path[i].pweight = (unique_path[i].pweight * (unique_depth + 1))
				/ static_cast<tfloat>(zero_fraction * (unique_depth - i));
		}
	}

	for (unsigned i = path_index; i < unique_depth; ++i) {
		unique_path[i].feature_index = unique_path[i + 1].feature_index;
		unique_path[i].zero_fraction = unique_path[i + 1].zero_fraction;
		unique_path[i].one_fraction = unique_path[i + 1].one_fraction;
	}
}

// determine what the total permuation weight would be if
// we unwound a previous extension in the decision path
inline tfloat unwound_path_sum(const PathElement *unique_path, unsigned unique_depth,
	unsigned path_index) {
	const tfloat one_fraction = unique_path[path_index].one_fraction;
	const tfloat zero_fraction = unique_path[path_index].zero_fraction;
	tfloat next_one_portion = unique_path[unique_depth].pweight;
	tfloat total = 0;

	if (one_fraction != 0) {
		for (int i = unique_depth - 1; i >= 0; --i) {
			const tfloat tmp = next_one_portion / static_cast<tfloat>((i + 1) * one_fraction);
			total += tmp;
			next_one_portion = unique_path[i].pweight - tmp * zero_fraction * (unique_depth - i);
		}
	}
	else {
		for (int i = unique_depth - 1; i >= 0; --i) {
			total += unique_path[i].pweight / (zero_fraction * (unique_depth - i));
		}
	}
	return total * (unique_depth + 1);
}


inline int compute_expectations(TreeEnsemble &tree, int i = 0, int depth = 0) {
	unsigned max_depth = 0;

	if (tree.children_right[i] >= 0) {
		const unsigned li = tree.children_left[i];
		const unsigned ri = tree.children_right[i];
		const unsigned depth_left = compute_expectations(tree, li, depth + 1);
		const unsigned depth_right = compute_expectations(tree, ri, depth + 1);
		const tfloat left_weight = tree.node_sample_weights[li];
		const tfloat right_weight = tree.node_sample_weights[ri];
		const unsigned li_offset = li * tree.num_outputs;
		const unsigned ri_offset = ri * tree.num_outputs;
		const unsigned i_offset = i * tree.num_outputs;
		for (unsigned j = 0; j < tree.num_outputs; ++j) {
			const tfloat v = (left_weight * tree.values[li_offset + j] + right_weight * tree.values[ri_offset + j]) / (left_weight + right_weight);
			tree.values[i_offset + j] = v;
		}
		max_depth = std::max(depth_left, depth_right) + 1;
	}

	if (depth == 0) tree.max_depth = max_depth;

	return max_depth;
}


// recursive computation of SHAP values for a decision tree
// feature_sets is a map from features to sets. sets are treated as single features in the process
inline void tree_shap_recursive(const unsigned num_outputs, const int *children_left,
	const int *children_right,
	const int *children_default, const int *features,
	const tfloat *thresholds, const tfloat *values,
	const tfloat *node_sample_weight,
	const tfloat *x, const bool *x_missing, tfloat *phi,
	unsigned node_index, unsigned unique_depth,
	PathElement *parent_unique_path, tfloat parent_zero_fraction,
	tfloat parent_one_fraction, int parent_feature_index,
	int condition, unsigned condition_feature,
	tfloat condition_fraction,
	unsigned *feature_sets) {

	// stop if we have no weight coming down to us
	if (condition_fraction == 0) return;

	// extend the unique path
	PathElement *unique_path = parent_unique_path + unique_depth + 1;
	std::copy(parent_unique_path, parent_unique_path + unique_depth + 1, unique_path);

	if (condition == 0 || condition_feature != static_cast<unsigned>(parent_feature_index)) {
		int parent_set_index = (parent_feature_index >= 0) ? feature_sets[parent_feature_index] : parent_feature_index;
		extend_path(unique_path, unique_depth, parent_zero_fraction, parent_one_fraction, parent_set_index);
	}

	// leaf node
	if (children_right[node_index] < 0) {
		for (unsigned i = 1; i <= unique_depth; ++i) {
			const tfloat w = unwound_path_sum(unique_path, unique_depth, i);
			const PathElement &el = unique_path[i];
			const unsigned phi_offset = el.feature_index * num_outputs;
			const unsigned values_offset = node_index * num_outputs;
			const tfloat scale = w * (el.one_fraction - el.zero_fraction) * condition_fraction;
			for (unsigned j = 0; j < num_outputs; ++j) {
				phi[phi_offset + j] += scale * values[values_offset + j];
			}
		}

		// internal node
	}
	else {
		const unsigned split_index = features[node_index];
		const unsigned split_set = feature_sets[split_index];

		// find which branch is "hot" (meaning x would follow it)
		unsigned hot_index = 0;
		if (x_missing[split_index]) {
			hot_index = children_default[node_index];
		}
		else if (x[split_index] <= thresholds[node_index]) {
			hot_index = children_left[node_index];
		}
		else {
			hot_index = children_right[node_index];
		}
		const unsigned cold_index = (static_cast<int>(hot_index) == children_left[node_index] ?
			children_right[node_index] : children_left[node_index]);
		const tfloat w = node_sample_weight[node_index];
		const tfloat hot_zero_fraction = node_sample_weight[hot_index] / w;
		const tfloat cold_zero_fraction = node_sample_weight[cold_index] / w;
		tfloat incoming_zero_fraction = 1;
		tfloat incoming_one_fraction = 1;

		// see if we have already split on this SET,
		// if so we undo that split so we can redo it for this node
		unsigned path_index = 0;
		for (; path_index <= unique_depth; ++path_index) {
			if (static_cast<unsigned>(unique_path[path_index].feature_index) == split_set) break;
		}
		if (path_index != unique_depth + 1) {
			incoming_zero_fraction = unique_path[path_index].zero_fraction;
			incoming_one_fraction = unique_path[path_index].one_fraction;
			unwind_path(unique_path, unique_depth, path_index);
			unique_depth -= 1;
		}

		// divide up the condition_fraction among the recursive calls
		tfloat hot_condition_fraction = condition_fraction;
		tfloat cold_condition_fraction = condition_fraction;
		if (condition > 0 && split_index == condition_feature) {
			cold_condition_fraction = 0;
			unique_depth -= 1;
		}
		else if (condition < 0 && split_index == condition_feature) {
			hot_condition_fraction *= hot_zero_fraction;
			cold_condition_fraction *= cold_zero_fraction;
			unique_depth -= 1;
		}

		tree_shap_recursive(
			num_outputs, children_left, children_right, children_default, features, thresholds, values,
			node_sample_weight, x, x_missing, phi, hot_index, unique_depth + 1, unique_path,
			hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
			split_index, condition, condition_feature, hot_condition_fraction, feature_sets
		);

		tree_shap_recursive(
			num_outputs, children_left, children_right, children_default, features, thresholds, values,
			node_sample_weight, x, x_missing, phi, cold_index, unique_depth + 1, unique_path,
			cold_zero_fraction * incoming_zero_fraction, 0,
			split_index, condition, condition_feature, cold_condition_fraction, feature_sets
		);
	}
}

inline void tree_shap(const TreeEnsemble& tree, const ExplanationDataset &data, tfloat *out_contribs, int condition, unsigned condition_feature, unsigned *feature_sets) {

	// update the reference value with the expected value of the tree's predictions
	if (condition == 0) {
		for (unsigned j = 0; j < tree.num_outputs; ++j) {
			out_contribs[data.num_Exp * tree.num_outputs + j] += tree.values[j];
		}
	}

	// Pre-allocate space for the unique path data
	const unsigned maxd = tree.max_depth + 2; // need a bit more space than the max depth
	PathElement *unique_path_data = new PathElement[(maxd * (maxd + 1)) / 2];

	tree_shap_recursive(
		tree.num_outputs, tree.children_left, tree.children_right, tree.children_default,
		tree.features, tree.thresholds, tree.values, tree.node_sample_weights, data.X,
		data.X_missing, out_contribs, 0, 0, unique_path_data, 1, 1, -1, condition,
		condition_feature, 1, feature_sets
	);

	delete[] unique_path_data;
}

inline void tree_shap(const TreeEnsemble& tree, const ExplanationDataset &data,
	tfloat *out_contribs, int condition, unsigned condition_feature) {

	vector<unsigned> feature_sets(data.M);
	for (unsigned int i = 0; i < data.M; i++)
		feature_sets[i] = i;

	tree_shap(tree, data, out_contribs, condition, condition_feature, feature_sets.data());
}

unsigned build_merged_tree_recursive(TreeEnsemble &out_tree, const TreeEnsemble &trees,
	const tfloat *data, const bool *data_missing, int *data_inds,
	const unsigned num_background_data_inds, unsigned num_data_inds,
	unsigned M, unsigned row = 0, unsigned i = 0, unsigned pos = 0,
	tfloat *leaf_value = NULL) {
	//tfloat new_leaf_value[trees.num_outputs];
	tfloat *new_leaf_value = (tfloat *)alloca(sizeof(tfloat) * trees.num_outputs); // allocate on the stack
	unsigned row_offset = row * trees.max_nodes;

	// we have hit a terminal leaf!!!
	if (trees.children_left[row_offset + i] < 0 && row + 1 == trees.tree_limit) {

		// create the leaf node
		const tfloat *vals = trees.values + (row * trees.max_nodes + i) * trees.num_outputs;
		if (leaf_value == NULL) {
			for (unsigned j = 0; j < trees.num_outputs; ++j) {
				out_tree.values[pos * trees.num_outputs + j] = vals[j];
			}
		}
		else {
			for (unsigned j = 0; j < trees.num_outputs; ++j) {
				out_tree.values[pos * trees.num_outputs + j] = leaf_value[j] + vals[j];
			}
		}
		out_tree.children_left[pos] = -1;
		out_tree.children_right[pos] = -1;
		out_tree.children_default[pos] = -1;
		out_tree.features[pos] = -1;
		out_tree.thresholds[pos] = 0;
		out_tree.node_sample_weights[pos] = num_background_data_inds;

		return pos;
	}

	// we hit an intermediate leaf (so just add the value to our accumulator and move to the next tree)
	if (trees.children_left[row_offset + i] < 0) {

		// accumulate the value of this original leaf so it will land on all eventual terminal leaves
		const tfloat *vals = trees.values + (row * trees.max_nodes + i) * trees.num_outputs;
		if (leaf_value == NULL) {
			for (unsigned j = 0; j < trees.num_outputs; ++j) {
				new_leaf_value[j] = vals[j];
			}
		}
		else {
			for (unsigned j = 0; j < trees.num_outputs; ++j) {
				new_leaf_value[j] = leaf_value[j] + vals[j];
			}
		}
		leaf_value = new_leaf_value;

		// move forward to the next tree
		row += 1;
		row_offset += trees.max_nodes;
		i = 0;
	}

	// split the data inds by this node's threshold
	const tfloat t = trees.thresholds[row_offset + i];
	const int f = trees.features[row_offset + i];
	const bool right_default = trees.children_default[row_offset + i] == trees.children_right[row_offset + i];
	int low_ptr = 0;
	int high_ptr = num_data_inds - 1;
	unsigned num_left_background_data_inds = 0;
	int low_data_ind;
	while (low_ptr <= high_ptr) {
		low_data_ind = data_inds[low_ptr];
		const int data_ind = std::abs(low_data_ind) * M + f;
		const bool is_missing = data_missing[data_ind];
		if ((!is_missing && data[data_ind] > t) || (right_default && is_missing)) {
			data_inds[low_ptr] = data_inds[high_ptr];
			data_inds[high_ptr] = low_data_ind;
			high_ptr -= 1;
		}
		else {
			if (low_data_ind >= 0) ++num_left_background_data_inds; // negative data_inds are not background samples
			low_ptr += 1;
		}
	}
	int *left_data_inds = data_inds;
	const unsigned num_left_data_inds = low_ptr;
	int *right_data_inds = data_inds + low_ptr;
	const unsigned num_right_data_inds = num_data_inds - num_left_data_inds;
	const unsigned num_right_background_data_inds = num_background_data_inds - num_left_background_data_inds;

	// all the data went right, so we skip creating this node and just recurse right
	if (num_left_data_inds == 0) {
		return build_merged_tree_recursive(
			out_tree, trees, data, data_missing, data_inds,
			num_background_data_inds, num_data_inds, M, row,
			trees.children_right[row_offset + i], pos, leaf_value
		);

		// all the data went left, so we skip creating this node and just recurse left
	}
	else if (num_right_data_inds == 0) {
		return build_merged_tree_recursive(
			out_tree, trees, data, data_missing, data_inds,
			num_background_data_inds, num_data_inds, M, row,
			trees.children_left[row_offset + i], pos, leaf_value
		);

		// data went both ways so we create this node and recurse down both paths
	}
	else {

		// build the left subtree
		const unsigned new_pos = build_merged_tree_recursive(
			out_tree, trees, data, data_missing, left_data_inds,
			num_left_background_data_inds, num_left_data_inds, M, row,
			trees.children_left[row_offset + i], pos + 1, leaf_value
		);

		// fill in the data for this node
		out_tree.children_left[pos] = pos + 1;
		out_tree.children_right[pos] = new_pos + 1;
		if (trees.children_left[row_offset + i] == trees.children_default[row_offset + i]) {
			out_tree.children_default[pos] = pos + 1;
		}
		else {
			out_tree.children_default[pos] = new_pos + 1;
		}

		out_tree.features[pos] = trees.features[row_offset + i];
		out_tree.thresholds[pos] = trees.thresholds[row_offset + i];
		out_tree.node_sample_weights[pos] = num_background_data_inds;

		// build the right subtree
		return build_merged_tree_recursive(
			out_tree, trees, data, data_missing, right_data_inds,
			num_right_background_data_inds, num_right_data_inds, M, row,
			trees.children_right[row_offset + i], new_pos + 1, leaf_value
		);
	}
}


void build_merged_tree(TreeEnsemble &out_tree, const ExplanationDataset &data, const TreeEnsemble &trees) {

	// create a joint data matrix from both X and R matrices
	tfloat *joined_data = new tfloat[(data.num_X + data.num_R) * data.M];
	std::copy(data.X, data.X + data.num_X * data.M, joined_data);
	std::copy(data.R, data.R + data.num_R * data.M, joined_data + data.num_X * data.M);
	bool *joined_data_missing = new bool[(data.num_X + data.num_R) * data.M];
	std::copy(data.X_missing, data.X_missing + data.num_X * data.M, joined_data_missing);
	std::copy(data.R_missing, data.R_missing + data.num_R * data.M, joined_data_missing + data.num_X * data.M);

	// create an starting array of data indexes we will recursively sort
	int *data_inds = new int[data.num_X + data.num_R];
	for (unsigned i = 0; i < data.num_X; ++i) data_inds[i] = i;
	for (unsigned int i = data.num_X; i < data.num_X + data.num_R; ++i) {
		data_inds[i] = -(int)i; // a negative index means it won't be recorded as a background sample
	}

	build_merged_tree_recursive(
		out_tree, trees, joined_data, joined_data_missing, data_inds, data.num_R,
		data.num_X + data.num_R, data.M
	);

	delete[] joined_data;
	delete[] joined_data_missing;
	delete[] data_inds;
}




// https://www.geeksforgeeks.org/space-and-time-efficient-binomial-coefficient/
inline int bin_coeff(int n, int k) {
	int res = 1;
	if (k > n - k)
		k = n - k;
	for (int i = 0; i < k; ++i) {
		res *= (n - i);
		res /= (i + 1);
	}
	return res;
}

// note this only handles single output models, so multi-output models get explained using multiple passes
inline void tree_shap_indep(const unsigned max_depth, const unsigned num_feats,
	const unsigned num_nodes, const tfloat *x,
	const bool *x_missing, const tfloat *r,
	const bool *r_missing, tfloat *out_contribs,
	float *pos_lst, float *neg_lst, signed short *feat_hist,
	float *memoized_weights, int *node_stack, Node *mytree) {

	//     const bool DEBUG = true;
	//     ofstream myfile;
	//     if (DEBUG) {
	//       myfile.open ("/homes/gws/hughchen/shap/out.txt",fstream::app);
	//       myfile << "Entering tree_shap_indep\n";
	//     }
	int ns_ctr = 0;
	std::fill_n(feat_hist, num_feats, 0);
	short node = 0, feat, cl, cr, cd, pnode, pfeat = -1;
	short next_xnode = -1, next_rnode = -1;
	short next_node = -1, from_child = -1;
	float thres, pos_x = 0, neg_x = 0, pos_r = 0, neg_r = 0;
	char from_flag;
	unsigned M = 0, N = 0;

	Node curr_node = mytree[node];
	feat = curr_node.feat;
	thres = curr_node.thres;
	cl = curr_node.cl;
	cr = curr_node.cr;
	cd = curr_node.cd;

	// short circut when this is a stump tree (with no splits)
	if (cl < 0) {
		out_contribs[num_feats] += curr_node.value;
		return;
	}

	//     if (DEBUG) {
	//       myfile << "\nNode: " << node << "\n";
	//       myfile << "x[feat]: " << x[feat] << ", r[feat]: " << r[feat] << "\n";
	//       myfile << "thres: " << thres << "\n";
	//     }

	if (x_missing[feat]) {
		next_xnode = cd;
	}
	else if (x[feat] > thres) {
		next_xnode = cr;
	}
	else if (x[feat] <= thres) {
		next_xnode = cl;
	}

	if (r_missing[feat]) {
		next_rnode = cd;
	}
	else if (r[feat] > thres) {
		next_rnode = cr;
	}
	else if (r[feat] <= thres) {
		next_rnode = cl;
	}

	if (next_xnode != next_rnode) {
		mytree[next_xnode].from_flag = FROM_X_NOT_R;
		mytree[next_rnode].from_flag = FROM_R_NOT_X;
	}
	else {
		mytree[next_xnode].from_flag = FROM_NEITHER;
	}

	// Check if x and r go the same way
	if (next_xnode == next_rnode) {
		next_node = next_xnode;
	}

	// If not, go left
	if (next_node < 0) {
		next_node = cl;
		if (next_rnode == next_node) { // rpath
			N = N + 1;
			feat_hist[feat] -= 1;
		}
		else if (next_xnode == next_node) { // xpath
			M = M + 1;
			N = N + 1;
			feat_hist[feat] += 1;
		}
	}
	node_stack[ns_ctr] = node;
	ns_ctr += 1;
	while (true) {
		node = next_node;
		curr_node = mytree[node];
		feat = curr_node.feat;
		thres = curr_node.thres;
		cl = curr_node.cl;
		cr = curr_node.cr;
		cd = curr_node.cd;
		pnode = curr_node.pnode;
		pfeat = curr_node.pfeat;
		from_flag = curr_node.from_flag;



		//         if (DEBUG) {
		//           myfile << "\nNode: " << node << "\n";
		//           myfile << "N: " << N << ", M: " << M << "\n";
		//           myfile << "from_flag==FROM_X_NOT_R: " << (from_flag==FROM_X_NOT_R) << "\n";
		//           myfile << "from_flag==FROM_R_NOT_X: " << (from_flag==FROM_R_NOT_X) << "\n";
		//           myfile << "from_flag==FROM_NEITHER: " << (from_flag==FROM_NEITHER) << "\n";
		//           myfile << "feat_hist[feat]: " << feat_hist[feat] << "\n";
		//         }

		// At a leaf
		if (cl < 0) {
			//      if (DEBUG) {
			//        myfile << "At a leaf\n";
			//      }

			if (M == 0) {
				out_contribs[num_feats] += mytree[node].value;
			}

			// Currently assuming a single output
			if (N != 0) {
				if (M != 0) {
					pos_lst[node] = mytree[node].value * memoized_weights[N + max_depth * (M - 1)];
				}
				if (M != N) {
					neg_lst[node] = -mytree[node].value * memoized_weights[N + max_depth * M];
				}
			}
			//             if (DEBUG) {
			//               myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
			//               myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
			//             }
			// Pop from node_stack
			ns_ctr -= 1;
			next_node = node_stack[ns_ctr];
			from_child = node;
			// Unwind
			if (feat_hist[pfeat] > 0) {
				feat_hist[pfeat] -= 1;
			}
			else if (feat_hist[pfeat] < 0) {
				feat_hist[pfeat] += 1;
			}
			if (feat_hist[pfeat] == 0) {
				if (from_flag == FROM_X_NOT_R) {
					N = N - 1;
					M = M - 1;
				}
				else if (from_flag == FROM_R_NOT_X) {
					N = N - 1;
				}
			}
			continue;
		}

		const bool x_right = x[feat] > thres;
		const bool r_right = r[feat] > thres;

		if (x_missing[feat]) {
			next_xnode = cd;
		}
		else if (x_right) {
			next_xnode = cr;
		}
		else if (!x_right) {
			next_xnode = cl;
		}

		if (r_missing[feat]) {
			next_rnode = cd;
		}
		else if (r_right) {
			next_rnode = cr;
		}
		else if (!r_right) {
			next_rnode = cl;
		}

		if (next_xnode >= 0) {
			if (next_xnode != next_rnode) {
				mytree[next_xnode].from_flag = FROM_X_NOT_R;
				mytree[next_rnode].from_flag = FROM_R_NOT_X;
			}
			else {
				mytree[next_xnode].from_flag = FROM_NEITHER;
			}
		}

		// Arriving at node from parent
		if (from_child == -1) {
			//      if (DEBUG) {
			//        myfile << "Arriving at node from parent\n";
			//      }
			node_stack[ns_ctr] = node;
			ns_ctr += 1;
			next_node = -1;

			//      if (DEBUG) {
			//        myfile << "feat_hist[feat]" << feat_hist[feat] << "\n";
			//      }
			// Feature is set upstream
			if (feat_hist[feat] > 0) {
				next_node = next_xnode;
				feat_hist[feat] += 1;
			}
			else if (feat_hist[feat] < 0) {
				next_node = next_rnode;
				feat_hist[feat] -= 1;
			}

			// x and r go the same way
			if (next_node < 0) {
				if (next_xnode == next_rnode) {
					next_node = next_xnode;
				}
			}

			// Go down one path
			if (next_node >= 0) {
				continue;
			}

			// Go down both paths, but go left first
			next_node = cl;
			if (next_rnode == next_node) {
				N = N + 1;
				feat_hist[feat] -= 1;
			}
			else if (next_xnode == next_node) {
				M = M + 1;
				N = N + 1;
				feat_hist[feat] += 1;
			}
			from_child = -1;
			continue;
		}

		// Arriving at node from child
		if (from_child != -1) {
			//             if (DEBUG) {
			//               myfile << "Arriving at node from child\n";
			//             }
			next_node = -1;
			// Check if we should unroll immediately
			if ((next_rnode == next_xnode) || (feat_hist[feat] != 0)) {
				next_node = pnode;
			}

			// Came from a single path, so unroll
			if (next_node >= 0) {
				//                 if (DEBUG) {
				//                   myfile << "Came from a single path, so unroll\n";
				//                 }
				// At the root node
				if (node == 0) {
					break;
				}
				// Update and unroll
				pos_lst[node] = pos_lst[from_child];
				neg_lst[node] = neg_lst[from_child];

				//                 if (DEBUG) {
				//                   myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
				//                   myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
				//                 }
				from_child = node;
				ns_ctr -= 1;

				// Unwind
				if (feat_hist[pfeat] > 0) {
					feat_hist[pfeat] -= 1;
				}
				else if (feat_hist[pfeat] < 0) {
					feat_hist[pfeat] += 1;
				}
				if (feat_hist[pfeat] == 0) {
					if (from_flag == FROM_X_NOT_R) {
						N = N - 1;
						M = M - 1;
					}
					else if (from_flag == FROM_R_NOT_X) {
						N = N - 1;
					}
				}
				continue;
				// Go right - Arriving from the left child
			}
			else if (from_child == cl) {
				//                 if (DEBUG) {
				//                   myfile << "Go right - Arriving from the left child\n";
				//                 }
				node_stack[ns_ctr] = node;
				ns_ctr += 1;
				next_node = cr;
				if (next_xnode == next_node) {
					M = M + 1;
					N = N + 1;
					feat_hist[feat] += 1;
				}
				else if (next_rnode == next_node) {
					N = N + 1;
					feat_hist[feat] -= 1;
				}
				from_child = -1;
				continue;
				// Compute stuff and unroll - Arriving from the right child
			}
			else if (from_child == cr) {
				//                 if (DEBUG) {
				//                   myfile << "Compute stuff and unroll - Arriving from the right child\n";
				//                 }
				pos_x = 0;
				neg_x = 0;
				pos_r = 0;
				neg_r = 0;
				if ((next_xnode == cr) && (next_rnode == cl)) {
					pos_x = pos_lst[cr];
					neg_x = neg_lst[cr];
					pos_r = pos_lst[cl];
					neg_r = neg_lst[cl];
				}
				else if ((next_xnode == cl) && (next_rnode == cr)) {
					pos_x = pos_lst[cl];
					neg_x = neg_lst[cl];
					pos_r = pos_lst[cr];
					neg_r = neg_lst[cr];
				}
				// out_contribs needs to have been initialized as all zeros
				// if (pos_x + neg_r != 0) {
				//   std::cout << "val " << pos_x + neg_r << "\n";
				// }
				out_contribs[feat] += pos_x + neg_r;
				pos_lst[node] = pos_x + pos_r;
				neg_lst[node] = neg_x + neg_r;

				//                 if (DEBUG) {
				//                   myfile << "out_contribs[feat]: " << out_contribs[feat] << "\n";
				//                   myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
				//                   myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
				//                 }

				// Check if at root
				if (node == 0) {
					break;
				}

				// Pop
				ns_ctr -= 1;
				next_node = node_stack[ns_ctr];
				from_child = node;

				// Unwind
				if (feat_hist[pfeat] > 0) {
					feat_hist[pfeat] -= 1;
				}
				else if (feat_hist[pfeat] < 0) {
					feat_hist[pfeat] += 1;
				}
				if (feat_hist[pfeat] == 0) {
					if (from_flag == FROM_X_NOT_R) {
						N = N - 1;
						M = M - 1;
					}
					else if (from_flag == FROM_R_NOT_X) {
						N = N - 1;
					}
				}
				continue;
			}
		}
	}
	//  if (DEBUG) {
	//    myfile.close();
	//  }
}


/**
* Runs Tree SHAP with feature independence assumptions on dense data.
*/
void dense_independent(const TreeEnsemble& trees, const ExplanationDataset &data,
	tfloat *out_contribs, tfloat transform(const tfloat, const tfloat)) {

	// reformat the trees for faster access
	Node *node_trees = new Node[trees.tree_limit * trees.max_nodes];
	for (unsigned i = 0; i < trees.tree_limit; ++i) {
		Node *node_tree = node_trees + i * trees.max_nodes;
		for (unsigned j = 0; j < trees.max_nodes; ++j) {
			const unsigned en_ind = i * trees.max_nodes + j;
			node_tree[j].cl = trees.children_left[en_ind];
			node_tree[j].cr = trees.children_right[en_ind];
			node_tree[j].cd = trees.children_default[en_ind];
			if (j == 0) {
				node_tree[j].pnode = 0;
			}
			if (trees.children_left[en_ind] >= 0) { // relies on all unused entries having negative values in them
				node_tree[trees.children_left[en_ind]].pnode = j;
				node_tree[trees.children_left[en_ind]].pfeat = trees.features[en_ind];
			}
			if (trees.children_right[en_ind] >= 0) { // relies on all unused entries having negative values in them
				node_tree[trees.children_right[en_ind]].pnode = j;
				node_tree[trees.children_right[en_ind]].pfeat = trees.features[en_ind];
			}

			node_tree[j].thres = (float)trees.thresholds[en_ind];
			node_tree[j].feat = trees.features[en_ind];
		}
	}

	// preallocate arrays needed by the algorithm
	float *pos_lst = new float[trees.max_nodes];
	float *neg_lst = new float[trees.max_nodes];
	int *node_stack = new int[(unsigned)trees.max_depth];
	signed short *feat_hist = new signed short[data.M];
	tfloat *tmp_out_contribs = new tfloat[(data.M + 1)];

	// precompute all the weight coefficients
	float *memoized_weights = new float[(trees.max_depth + 1) * (trees.max_depth + 1)];
	for (unsigned n = 0; n <= trees.max_depth; ++n) {
		for (unsigned m = 0; m <= trees.max_depth; ++m) {
			memoized_weights[n + trees.max_depth * m] = float(1.0 / (n * bin_coeff(n - 1, m)));
		}
	}

	// compute the explanations for each sample
	tfloat *instance_out_contribs;
	tfloat rescale_factor = 1.0;
	tfloat margin_x = 0;
	tfloat margin_r = 0;
	for (unsigned oind = 0; oind < trees.num_outputs; ++oind) {
		// set the values int he reformated tree to the current output index
		for (unsigned i = 0; i < trees.tree_limit; ++i) {
			Node *node_tree = node_trees + i * trees.max_nodes;
			for (unsigned j = 0; j < trees.max_nodes; ++j) {
				const unsigned en_ind = i * trees.max_nodes + j;
				node_tree[j].value = (float)trees.values[en_ind * trees.num_outputs + oind];
			}
		}

		// loop over all the samples
		for (unsigned i = 0; i < data.num_X; ++i) {
			const tfloat *x = data.X + i * data.M;
			const bool *x_missing = data.X_missing + i * data.M;
			instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
			const tfloat y_i = data.y == NULL ? 0 : data.y[i];

			//print_progress_bar(last_print, start_time, oind * data.num_X + i, data.num_X * trees.num_outputs);

			// compute the model's margin output for x
			if (transform != NULL) {
				margin_x = trees.base_offset;
				for (unsigned k = 0; k < trees.tree_limit; ++k) {
					margin_x += tree_predict(k, trees, x, x_missing)[oind];
				}
			}

			for (unsigned j = 0; j < data.num_R; ++j) {
				const tfloat *r = data.R + j * data.M;
				const bool *r_missing = data.R_missing + j * data.M;
				std::fill_n(tmp_out_contribs, (data.M + 1), 0);

				// compute the model's margin output for r
				if (transform != NULL) {
					margin_r = trees.base_offset;
					for (unsigned k = 0; k < trees.tree_limit; ++k) {
						margin_r += tree_predict(k, trees, r, r_missing)[oind];
					}
				}

				for (unsigned k = 0; k < trees.tree_limit; ++k) {
					tree_shap_indep(
						trees.max_depth, data.M, trees.max_nodes, x, x_missing, r, r_missing,
						tmp_out_contribs, pos_lst, neg_lst, feat_hist, memoized_weights,
						node_stack, node_trees + k * trees.max_nodes
					);
				}

				// compute the rescale factor
				if (transform != NULL) {
					if (margin_x == margin_r) {
						rescale_factor = 1.0;
					}
					else {
						rescale_factor = (*transform)(margin_x, y_i) - (*transform)(margin_r, y_i);
						rescale_factor /= margin_x - margin_r;
					}
				}

				// add the effect of the current reference to our running total
				// this is where we can do per reference scaling for non-linear transformations
				for (unsigned k = 0; k < data.M; ++k) {
					instance_out_contribs[k * trees.num_outputs + oind] += tmp_out_contribs[k] * rescale_factor;
				}

				// Add the base offset
				if (transform != NULL) {
					instance_out_contribs[data.M * trees.num_outputs + oind] += (*transform)(trees.base_offset + tmp_out_contribs[data.M], 0);
				}
				else {
					instance_out_contribs[data.M * trees.num_outputs + oind] += trees.base_offset + tmp_out_contribs[data.M];
				}
			}

			// average the results over all the references.
			for (unsigned j = 0; j < (data.M + 1); ++j) {
				instance_out_contribs[j * trees.num_outputs + oind] /= data.num_R;
			}

			// apply the base offset to the bias term
			// for (unsigned j = 0; j < trees.num_outputs; ++j) {
			//     instance_out_contribs[data.M * trees.num_outputs + j] += (*transform)(trees.base_offset, 0);
			// }
		}
	}

	delete[] tmp_out_contribs;
	delete[] node_trees;
	delete[] pos_lst;
	delete[] neg_lst;
	delete[] node_stack;
	delete[] feat_hist;
	delete[] memoized_weights;
}


/**
* This runs Tree SHAP with a per tree path conditional dependence assumption.
*/
void dense_tree_path_dependent(const TreeEnsemble& trees, const ExplanationDataset &data,
	tfloat *out_contribs, unsigned *feature_sets, tfloat transform(const tfloat, const tfloat)) {

	MedProgress progress("TREE_SHAPLEY", data.num_X, 15, 50);
	// build explanation for each sample
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < data.num_X; ++i) {
		tfloat *instance_out_contribs = out_contribs + i * (data.num_Exp + 1) * trees.num_outputs;
		ExplanationDataset instance;
		TreeEnsemble tree;
		data.get_x_instance(instance, i);

		// aggregate the effect of explaining each tree
		// (this works because of the linearity property of Shapley values)
		for (unsigned j = 0; j < trees.tree_limit; ++j) {
			trees.get_tree(tree, j);
			tree_shap(tree, instance, instance_out_contribs, 0, 0, feature_sets);

		}

		// apply the base offset to the bias term
		for (unsigned j = 0; j < trees.num_outputs; ++j) {
			instance_out_contribs[data.num_Exp * trees.num_outputs + j] += trees.base_offset;
		}

		progress.update();
	}
}

// phi = np.zeros((self._current_X.shape[1] + 1, self._current_X.shape[1] + 1, self.n_outputs))
//         phi_diag = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
//         for t in range(self.tree_limit):
//             self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_diag)
//             for j in self.trees[t].unique_features:
//                 phi_on = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
//                 phi_off = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
//                 self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_on, 1, j)
//                 self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_off, -1, j)
//                 phi[j] += np.true_divide(np.subtract(phi_on,phi_off),2.0)
//                 phi_diag[j] -= np.sum(np.true_divide(np.subtract(phi_on,phi_off),2.0))
//         for j in range(self._current_X.shape[1]+1):
//             phi[j][j] = phi_diag[j]
//         phi /= self.tree_limit
//         return phi

void dense_tree_interactions_path_dependent(const TreeEnsemble& trees, const ExplanationDataset &data,
	tfloat *out_contribs,
	tfloat transform(const tfloat, const tfloat)) {

	// build a list of all the unique features in each tree
	int *unique_features = new int[trees.tree_limit * trees.max_nodes];
	std::fill(unique_features, unique_features + trees.tree_limit * trees.max_nodes, -1);
	for (unsigned j = 0; j < trees.tree_limit; ++j) {
		const int *features_row = trees.features + j * trees.max_nodes;
		int *unique_features_row = unique_features + j * trees.max_nodes;
		for (unsigned k = 0; k < trees.max_nodes; ++k) {
			for (unsigned l = 0; l < trees.max_nodes; ++l) {
				if (features_row[k] == unique_features_row[l]) break;
				if (unique_features_row[l] < 0) {
					unique_features_row[l] = features_row[k];
					break;
				}
			}
		}
	}

	// build an interaction explanation for each sample
	tfloat *instance_out_contribs;
	TreeEnsemble tree;
	ExplanationDataset instance;
	const unsigned contrib_row_size = (data.M + 1) * trees.num_outputs;
	tfloat *diag_contribs = new tfloat[contrib_row_size];
	tfloat *on_contribs = new tfloat[contrib_row_size];
	tfloat *off_contribs = new tfloat[contrib_row_size];
	for (unsigned i = 0; i < data.num_X; ++i) {
		instance_out_contribs = out_contribs + i * (data.M + 1) * contrib_row_size;
		data.get_x_instance(instance, i);

		// aggregate the effect of explaining each tree
		// (this works because of the linearity property of Shapley values)
		std::fill(diag_contribs, diag_contribs + contrib_row_size, 0);
		for (unsigned j = 0; j < trees.tree_limit; ++j) {
			trees.get_tree(tree, j);
			tree_shap(tree, instance, diag_contribs, 0, 0);

			const int *unique_features_row = unique_features + j * trees.max_nodes;
			for (unsigned k = 0; k < trees.max_nodes; ++k) {
				const int ind = unique_features_row[k];
				if (ind < 0) break; // < 0 means we have seen all the features for this tree

									// compute the shap value with this feature held on and off
				std::fill(on_contribs, on_contribs + contrib_row_size, 0);
				std::fill(off_contribs, off_contribs + contrib_row_size, 0);
				tree_shap(tree, instance, on_contribs, 1, ind);
				tree_shap(tree, instance, off_contribs, -1, ind);

				// save the difference between on and off as the interaction value
				for (unsigned l = 0; l < contrib_row_size; ++l) {
					const tfloat val = (on_contribs[l] - off_contribs[l]) / 2;
					instance_out_contribs[ind * contrib_row_size + l] += val;
					diag_contribs[l] -= val;
				}
			}
		}

		// set the diagonal
		for (unsigned j = 0; j < data.M + 1; ++j) {
			const unsigned offset = j * contrib_row_size + j * trees.num_outputs;
			for (unsigned k = 0; k < trees.num_outputs; ++k) {
				instance_out_contribs[offset + k] = diag_contribs[j * trees.num_outputs + k];
			}
		}

		// apply the base offset to the bias term
		const unsigned last_ind = (data.M * (data.M + 1) + data.M) * trees.num_outputs;
		for (unsigned j = 0; j < trees.num_outputs; ++j) {
			instance_out_contribs[last_ind + j] += trees.base_offset;
		}
	}

	delete[] diag_contribs;
	delete[] on_contribs;
	delete[] off_contribs;
	delete[] unique_features;
}

/**
* This runs Tree SHAP with a global path conditional dependence assumption.
*
* By first merging all the trees in a tree ensemble into an equivalent single tree
* this method allows arbitrary marginal transformations and also ensures that all the
* evaluations of the model are consistent with some training data point.
*/
void dense_global_path_dependent(const TreeEnsemble& trees, const ExplanationDataset &data,
	tfloat *out_contribs, tfloat transform(const tfloat, const tfloat)) {

	// allocate space for our new merged tree (we save enough room to totally split all samples if need be)
	TreeEnsemble merged_tree;
	merged_tree.allocate(1, (data.num_X + data.num_R) * 2, trees.num_outputs);

	// collapse the ensemble of trees into a single tree that has the same behavior
	// for all the X and R samples in the dataset
	build_merged_tree(merged_tree, data, trees);

	// compute the expected value and depth of the new merged tree
	compute_expectations(merged_tree);

	// explain each sample using our new merged tree
	ExplanationDataset instance;
	tfloat *instance_out_contribs;
	for (unsigned i = 0; i < data.num_X; ++i) {
		instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
		data.get_x_instance(instance, i);

		// since we now just have a single merged tree we can just use the tree_path_dependent algorithm
		tree_shap(merged_tree, instance, instance_out_contribs, 0, 0);

		// apply the base offset to the bias term
		for (unsigned j = 0; j < trees.num_outputs; ++j) {
			instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset;
		}
	}

	merged_tree.free();
}


/**
* The main method for computing Tree SHAP on model using dense data.
*/
void dense_tree_shap(const TreeEnsemble& trees, const ExplanationDataset &data, tfloat *out_contribs,
	const int feature_dependence, unsigned model_transform, bool interactions) {

	vector<unsigned> feature_sets(data.M);
	for (unsigned int i = 0; i < data.M; i++)
		feature_sets[i] = i;

	dense_tree_shap(trees, data, out_contribs, feature_dependence, model_transform, interactions, feature_sets.data());
}

void dense_tree_shap(const TreeEnsemble& trees, const ExplanationDataset &data, tfloat *out_contribs,
	const int feature_dependence, unsigned model_transform, bool interactions, unsigned *feature_sets) {

	// Temporary patch - create an indicator weather we have sets ...
	bool using_sets = false;
	if (data.M != data.num_Exp)
		using_sets = true;
	else {
		for (unsigned int i = 0; i < data.M; i++) {
			if (feature_sets[i] != i) {
				using_sets = true;
				break;
			}
		}
	}

	// see what transform (if any) we have
	tfloat(*transform)(const tfloat margin, const tfloat y) = NULL;
	switch (model_transform) {
	case MODEL_TRANSFORM::logistic:
		transform = logistic_transform;
		break;

	case MODEL_TRANSFORM::logistic_nlogloss:
		transform = logistic_nlogloss_transform;
		break;

	case MODEL_TRANSFORM::squared_loss:
		transform = squared_loss_transform;
		break;
	}

	// dispatch to the correct algorithm handler
	switch (feature_dependence) {
	case FEATURE_DEPENDENCE::independent:
		if (using_sets) {
			MTHROW_AND_ERR("Currently sets not implemented for FEATURE_DEPENDENCE::independent\n")
		}
		else if (interactions) {
			std::cerr << "FEATURE_DEPENDENCE::independent does not support interactions!\n";
		}
		else dense_independent(trees, data, out_contribs, transform);
		return;

	case FEATURE_DEPENDENCE::tree_path_dependent:
		if (interactions) {
			if (using_sets)
				MTHROW_AND_ERR("Currently sets not implemented for FEATURE_DEPENDENCE::tree_path_dependent with interactions\n")
			else
				dense_tree_interactions_path_dependent(trees, data, out_contribs, transform);
		}
		else
			dense_tree_path_dependent(trees, data, out_contribs, feature_sets, transform);
		return;

	case FEATURE_DEPENDENCE::global_path_dependent:
		if (using_sets) {
			MTHROW_AND_ERR("Currently sets not implemented for FEATURE_DEPENDENCE::global_path_dependent\n")
		}
		else if (interactions) {
			std::cerr << "FEATURE_DEPENDENCE::global_path_dependent does not support interactions!\n";
		}
		else dense_global_path_dependent(trees, data, out_contribs, transform);
		return;
	}
}

void get_weights(const TreeEnsemble& tree, int node_index, unsigned *sets) {

	if (tree.children_right[node_index] >= 0) {
		cerr << node_index << "/" << tree.node_sample_weights[node_index] << "/" << sets[tree.features[node_index]] << "/" << tree.children_left[node_index] << "/" << tree.children_right[node_index] << " ";
		get_weights(tree, tree.children_left[node_index], sets);
		get_weights(tree, tree.children_right[node_index], sets);
	}
	else
		cerr << node_index << "/" << tree.node_sample_weights[node_index] << "/" << tree.values[node_index] << " ";
}

/// <summary>
/// fixes the group contrib using covarinace or mutual information
/// </summary>
/// @param abs_cov_feats - covariance or mi matrix for all features (i,j) symmetric
/// @param feats_instance_contribs - feature contribution
/// @param grp_contribs - the group contrib - to manipulate
/// @param feature_sets - in size of all features - group id for each feature
/// @param mask - vector with groups seleceted already idx
void fix_feature_dependency_in_groups(const MedMat<float>& abs_cov_feats, const MedMat<tfloat> &feats_instance_contribs,
	MedMat<tfloat> &grp_contribs, const unsigned *feature_sets, const vector<int> &mask, bool take_max = false) {

	unordered_map<unsigned, vector<int>> group2Inds;
	for (int i = 0; i < abs_cov_feats.nrows; ++i)
		group2Inds[feature_sets[i]].push_back(i);
	MedMat<tfloat> fixed_contrib(grp_contribs.nrows, 1);
	int nGroups = (int)group2Inds.size();
	//mask of selected groups

	MedMat<tfloat> fixed_cov_abs(nGroups, nGroups); //cov matrix with groups X groups connections, zero inside groups:
	fixed_cov_abs.set_val(1);
	//fix for each group:
	if (take_max) {
		//fix using max
		for (int iGrp1 = 0; iGrp1 < nGroups; iGrp1++) {
			fixed_cov_abs(iGrp1, iGrp1) = 1.0;
			for (int iGrp2 = iGrp1 + 1; iGrp2 < nGroups; iGrp2++) {
				if (mask[iGrp1] || mask[iGrp2]) {
					fixed_cov_abs(iGrp1, iGrp2) = 0;
					fixed_cov_abs(iGrp2, iGrp1) = 0;
					continue;
				}
				float max_coeff = 0;
				for (int idx1 : group2Inds[iGrp1]) {
					for (int idx2 : group2Inds[iGrp2]) {
						if (abs_cov_feats(idx1, idx2) > max_coeff)
							max_coeff = abs_cov_feats(idx1, idx2);
					}
				}
				fixed_cov_abs(iGrp1, iGrp2) = fixed_cov_abs(iGrp2, iGrp1) = max_coeff;
			}
		}
	}
	else {
		for (int iGrp1 = 0; iGrp1 < nGroups; iGrp1++) {
			fixed_cov_abs(iGrp1, iGrp1) = 1;
			for (int iGrp2 = iGrp1 + 1; iGrp2 < nGroups; iGrp2++) {
				if (mask[iGrp1] || mask[iGrp2]) {
					fixed_cov_abs(iGrp1, iGrp2) = 0;
					fixed_cov_abs(iGrp2, iGrp1) = 0;
					continue;
				}
				//not within group
				double w = 0;
				double max_w = 0;
				for (int i = 0; i < group2Inds[iGrp1].size(); ++i)
					for (int j = 0; j < group2Inds[iGrp2].size(); ++j)
					{
						int ind_feat_1 = group2Inds[iGrp1][i];
						int ind_feat_2 = group2Inds[iGrp2][j];
						w += abs_cov_feats(ind_feat_1, ind_feat_2) * abs(feats_instance_contribs(ind_feat_1, 0)) * abs(feats_instance_contribs(ind_feat_2, 0));
						max_w += abs(feats_instance_contribs(ind_feat_1, 0)) * abs(feats_instance_contribs(ind_feat_2, 0)); //as if all cov are 1
					}
				if (max_w > 0)
					w /= max_w;

				//use w to add contribution for group:
				fixed_cov_abs(iGrp1, iGrp2) = w;
				fixed_cov_abs(iGrp2, iGrp1) = w;
			}
		}
	}
	//fast_multiply_medmat(fixed_cov_abs, grp_contribs, fixed_contrib, (float)1.0);
	//grp_contribs - size(nGroups,1) fixed_cov_abs - size(nGroups,nGroups)
	fixed_contrib.resize(fixed_cov_abs.nrows, grp_contribs.ncols);
	Eigen::Map<const Eigen::MatrixXd> x(fixed_cov_abs.data_ptr(), fixed_cov_abs.ncols, fixed_cov_abs.nrows);
	Eigen::Map<const Eigen::MatrixXd> y(grp_contribs.data_ptr(), grp_contribs.ncols, grp_contribs.nrows);
	Eigen::Map<Eigen::MatrixXd> z(fixed_contrib.data_ptr(), fixed_contrib.ncols, fixed_contrib.nrows);
	z = y * x;

	//store output:
	grp_contribs = std::move(fixed_contrib);
}

void iterative_tree_shap(const TreeEnsemble& trees, const ExplanationDataset &data, tfloat *out_contribs,
	const int feature_dependence, unsigned model_transform, bool interactions, unsigned *feature_sets, bool verbose,
	vector<string>& names, const MedMat<float>& abs_cov_mat, int iteration_cnt, bool max_in_groups) {

	// Currently - limited applicabilty to main use-case
	if (feature_dependence != FEATURE_DEPENDENCE::tree_path_dependent || interactions) {
		MTHROW_AND_ERR("Currently iterations implemented only for FEATURE_DEPENDENCE::tree_path_dependent without interatctions \n");
	}

	unsigned int max_iters = data.num_Exp;
	if (iteration_cnt > 0 && iteration_cnt < max_iters)
		max_iters = (unsigned int)iteration_cnt;

	//calc for each feature if needed:
	vector<tfloat> feats_contrib;
	if (abs_cov_mat.nrows && names.size() != data.M) {
		feats_contrib.resize(static_cast<size_t>(data.num_X) * (static_cast<size_t>(data.M) + 1) * static_cast<size_t>(trees.num_outputs), 0);
		ExplanationDataset data_for_features = data;
		data_for_features.num_Exp = data.M;
		dense_tree_shap(trees, data_for_features, feats_contrib.data(), feature_dependence, model_transform, interactions);

		//DEBUG:
		//feats_contrib.resize(data.num_X * (data.M + 1) * trees.num_outputs, 1);
	}

	MedProgress progress("TREE_SHAPLEY_ITERATIVE", data.num_X, 15, 10);
	// build explanation for each sample
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < data.num_X; ++i) {
		tfloat *instance_out_contribs = out_contribs + i * (data.num_Exp + 1) * trees.num_outputs;
		ExplanationDataset instance;
		TreeEnsemble tree;

		data.get_x_instance(instance, i);
		vector<int> mask(data.num_Exp, 0);

		// Do iterations
		MedMat<tfloat> last_instance_contribs(data.num_Exp, 1);
		MedMat<tfloat> first_instance_contribs(data.M, 1); // for cov/mi fix if used when no groups:
		if (abs_cov_mat.nrows && names.size() != data.M) { //need grouping
			tfloat *instance_feats_contrib = &feats_contrib[static_cast<size_t>(i) * (data.M + 1)  * static_cast<size_t>(trees.num_outputs)];
			//calculate when no groups:
			//dense_tree_shap(trees, instance, instance_feats_contrib.data(), feature_dependence, model_transform, interactions); //not thread safe
			for (int j = 0; j < data.M; ++j) //skip bias last
				first_instance_contribs(j, 0) = instance_feats_contrib[j];
		}
		tfloat last_bias = 0;
		for (unsigned k = 0; k < max_iters; k++) {
			// aggregate the effect of explaining each tree
			// (this works because of the linearity property of Shapley values)
			MedMat<tfloat> instance_temp_contrib(data.num_Exp + 1, 1);
			for (unsigned j = 0; j < trees.tree_limit; ++j) {
				trees.get_tree(tree, j);

				// Generate adjusted tree
				TreeEnsemble adjusted_tree;
				tree.create_adjusted_tree(instance, mask.data(), feature_sets, adjusted_tree);

				// Get conditioned SHAP values
				tree_shap(adjusted_tree, instance, instance_temp_contrib.data_ptr(), 0, 0, feature_sets);

				adjusted_tree.free();
			}

			// Bias - last component
			float bias = instance_temp_contrib(data.num_Exp, 0);
			instance_temp_contrib.resize(data.num_Exp, 1); //remove bias

			if (k == 0 && abs_cov_mat.nrows && names.size() == data.M) //has no grouping use this:
				first_instance_contribs = instance_temp_contrib;

			if (k == 0)
				instance_out_contribs[data.num_Exp] = bias;

			// Add covariance correction
			MedMat<float> fixed_contrib(data.num_Exp, 1);
			if (abs_cov_mat.nrows) {
				//instance_temp_contrib is for groups. abs_cov_mat is for features. first_instance_contribs is for features. fixed_contrib - should be output for groups
				fix_feature_dependency_in_groups(abs_cov_mat, first_instance_contribs, instance_temp_contrib, feature_sets, mask, max_in_groups);
				fixed_contrib = instance_temp_contrib;
			}
			else //no cov/mi fix:
				fixed_contrib = instance_temp_contrib;

			// Find main contributer
			int max_idx = 0;
			tfloat max_cont = fabs(fixed_contrib(0, 0));
			for (int j = 1; j < data.num_Exp; j++) {
				if ((!mask[j]) && fabs(fixed_contrib(j, 0)) > max_cont) {
					max_cont = fabs(fixed_contrib(j, 0));
					max_idx = j;
				}
			}

			if (verbose) {
				MLOG("\tIteration %d\n", k);
				MLOG("\tConditioned on");
				for (size_t ii = 0; ii < mask.size(); ii++) {
					if (mask[ii] == 1)
						MLOG(" %s", names[ii].c_str());
				}
				MLOG("\n\tBias=%f\n", bias);
				for (size_t ii = 0; ii < mask.size(); ii++) {
					if (mask[ii] == 0)
						MLOG("\tSHAP-Val(%s)= %f\n", names[ii].c_str(), fixed_contrib(0, ii));
				}
			}

			if (max_cont == 0) {
				//All the remaining variables has 0 contribution - stop iterations:
				if (iteration_cnt > 0)
					last_instance_contribs = fixed_contrib;
				else
					last_bias = bias;
				break;
			}

			instance_out_contribs[max_idx] = fixed_contrib(max_idx, 0);
			mask[max_idx] = 1;

			if (k == max_iters - 1) { //if last iteration 
				if (iteration_cnt > 0)
					last_instance_contribs = fixed_contrib;
				else
					last_bias = bias;
			}
		}

		//copy tail contributions of not masked:
		if (iteration_cnt > 0) {
			for (size_t k = 0; k < last_instance_contribs.size(); ++k)
				if (k >= mask.size() || !mask[k]) //last is bias
					instance_out_contribs[k] = last_instance_contribs(k, 0);
		}
		else //copy only bias
			instance_out_contribs[data.num_Exp * trees.num_outputs] = last_bias;

		// apply the base offset to the bias term
		for (unsigned j = 0; j < trees.num_outputs; ++j) {
			instance_out_contribs[data.num_Exp * trees.num_outputs + j] += trees.base_offset;
		}

		progress.update();
	}

}


long int_calc_nchoose_k(long n, long k) {
	long num = 1;
	for (int i = k + 1; i <= n; ++i)
		num *= i;
	for (int i = 2; i <= n - k; ++i)
		num /= i;
	return num;
}

double medial::shapley::nchoosek(long n, long k) {
	if (k == 0)
		return 1;
	if (k <= 20)
		return int_calc_nchoose_k(n, k);
	double ex = 0;
	for (int i = k + 1; i <= n; ++i)
		ex += log(i);
	for (int i = 2; i <= n - k; ++i)
		ex -= log(i);
	return exp(ex);
}

void medial::shapley::list_all_options_binary(int nfeats, vector<vector<bool>> &all_opts) {
	//int all_opts_count = nchoosek(nfeats, k);
	//all_opts.reserve(all_opts_count);
	vector<vector<int>> all_opts_num;
	for (size_t k = 1; k < nfeats; ++k)
	{
		vector<vector<int>> all_opts_num_batch; //option_num, list of selected idx
		all_opts_num_batch.push_back({}); //builds all options for k by bottom to up
		for (int i = 1; i <= k; ++i)
		{
			//advance to i step when passing on all current options:
			int curr_opts_count = nchoosek(nfeats, i);
			vector<vector<int>> curr_options(curr_opts_count);
			int opt_num = 0;
			for (size_t j = 0; j < all_opts_num_batch.size(); ++j)
			{
				const vector<int> &curr_opt = all_opts_num_batch[j];
				int st = 0;
				if (!curr_opt.empty())
					st = curr_opt.back() + 1;
				for (int k = st; k < nfeats; ++k) {
					//add new option:
					curr_options[opt_num] = curr_opt;
					curr_options[opt_num].push_back(k);
					++opt_num;
				}
			}
			//update all_opts:
			all_opts_num_batch.swap(curr_options);
		}
		all_opts_num.insert(all_opts_num.end(), all_opts_num_batch.begin(), all_opts_num_batch.end());
	}
	//converts all_opts_num to all_opts
	all_opts.resize(all_opts_num.size());
	for (size_t i = 0; i < all_opts.size(); ++i)
	{
		all_opts[i].resize(nfeats, false);
		for (int sel : all_opts_num[i])
			all_opts[i][sel] = true;
	}
}

void medial::shapley::generate_mask(vector<bool> &mask, int nfeat, mt19937 &gen, bool uniform_rand, bool use_shuffle) {
	mask.clear(); //If has values

	if (uniform_rand) {
		mask.reserve(nfeat);
		uniform_int_distribution<> rnd_coin(0, 1);
		for (size_t i = 0; i < nfeat; ++i)
			mask.push_back(rnd_coin(gen) > 0);
		return;
	}

	uniform_int_distribution<> rnd_dist(0, nfeat);
	int zero_count = rnd_dist(gen);
	if (zero_count == nfeat) {
		mask.resize(nfeat);
		return;
	}
	else if (zero_count == 0) {
		mask.resize(nfeat, true);
		return;
	}
	mask.resize(nfeat, true);


	if (use_shuffle) {
		//calc shuffle of indexes and takes top - may be faster for small arrays
		vector<int> sel_idx; sel_idx.reserve(nfeat);
		for (int i = 0; i < nfeat; ++i)
			sel_idx.push_back(i);
		shuffle(sel_idx.begin(), sel_idx.end(), gen);
		for (size_t i = 0; i < zero_count; ++i)
			mask[sel_idx[i]] = false;
	}
	else {
		//calc with no reapets till converges:
		vector<bool> seen_idx(nfeat);
		uniform_int_distribution<> rnd_feat(0, nfeat - 1);
		for (size_t i = 0; i < zero_count; ++i)
		{
			int sel = rnd_feat(gen);
			while (seen_idx[sel])
				sel = rnd_feat(gen);
			seen_idx[sel] = true;
			mask[sel] = false;
		}

	}
}

void medial::shapley::generate_mask_(vector<bool> &mask, int nfeat, mt19937 &gen, bool uniform_rand, 
	float uniform_rand_p, bool use_shuffle,	int limit_zero_cnt) {
	//mask is not empty - sample from non-empty
	if (limit_zero_cnt >= nfeat)
		limit_zero_cnt = nfeat; //problem with arguments

	if (limit_zero_cnt <= 0) //when not set - or default of no limit
		limit_zero_cnt = nfeat;

	int curr_cnt = 0;
	for (size_t i = 0; i < mask.size(); ++i)
		curr_cnt += int(!mask[i]); //how many zeros now
	if (curr_cnt >= limit_zero_cnt)
		return;

	if (uniform_rand) {
		uniform_real_distribution<> rnd_coin(0, 1);
		for (size_t i = 0; i < nfeat; ++i)
			if (mask[i])
				mask[i] = (rnd_coin(gen) > uniform_rand_p);
		return;
	}

	uniform_int_distribution<> rnd_dist(0, limit_zero_cnt);
	int zero_count = rnd_dist(gen);
	if (zero_count == nfeat) {
		mask.clear(); //all zeros
		mask.resize(nfeat);
		return;
	}
	else if (zero_count <= curr_cnt) {
		return; //no change in mask is needed, has more zeros than requested
	}

	if (use_shuffle) {
		//calc shuffle of indexes and takes top - may be faster for small arrays
		vector<int> sel_idx; sel_idx.reserve(nfeat);
		for (int i = 0; i < nfeat; ++i)
			if (mask[i]) //list only potentials
				sel_idx.push_back(i);
		shuffle(sel_idx.begin(), sel_idx.end(), gen);
		for (size_t i = 0; i < zero_count - curr_cnt; ++i)
			mask[sel_idx[i]] = false;
	}
	else {
		//calc with no reapets till converges:
		vector<bool> seen_idx(nfeat);
		for (size_t i = 0; i < mask.size(); ++i)
			if (!mask[i])
				seen_idx[i] = true;

		uniform_int_distribution<> rnd_feat(0, nfeat - 1);
		for (size_t i = 0; i < zero_count - curr_cnt; ++i)
		{
			int sel = rnd_feat(gen);
			while (seen_idx[sel])
				sel = rnd_feat(gen);
			seen_idx[sel] = true;
			mask[sel] = false;
		}

	}
}

/**
* sample uniformally masks - first by choosing number of 0's in the mask(set missings) and than select them randomally
*/
void medial::shapley::sample_options_SHAP(int nfeats, vector<vector<bool>> &all_opts, int opt_count, mt19937 &gen, bool with_repeats
	, bool uniform_rand, bool use_shuffle) {
	all_opts.resize(opt_count);
	unordered_set<vector<bool>> seen_mask;
	for (size_t i = 0; i < all_opts.size(); ++i)
	{
		vector<bool> &cr_opt = all_opts[i];
		//generate mask:
		generate_mask(cr_opt, nfeats, gen);
		while (!with_repeats && seen_mask.find(cr_opt) != seen_mask.end())
			generate_mask(cr_opt, nfeats, gen);
		//finished - already manipulated memory
		if (!with_repeats)
			seen_mask.insert(cr_opt);
	}
}

double medial::shapley::get_c(int p1, int p2, int end_l) {
	//c := (end_l)! / ( p1! * p2! )
	//returns 1/c 
	double c = 0, d = 0;
	if (p1 > p2) {
		for (int i = p1 + 1; i <= end_l; ++i)
			c += log(i);
		for (int i = 2; i <= p2; ++i)
			d += log(i);
		c -= d;
	}
	else {
		for (int i = p2 + 1; i <= end_l; ++i)
			c += log(i);
		for (int i = 2; i <= p1; ++i)
			d += log(i);
		c -= d;
	}

	return 1 / exp(c);
}

void medial::shapley::explain_shapley(const MedFeatures &matrix, int selected_sample, int max_tests,
	MedPredictor *predictor, float missing_value, const vector<vector<int>>& group2index, const vector<string> &groupNames
	, vector<float> &features_coeff, mt19937 &gen,
	bool sample_masks_with_repeats, float select_from_all, bool uniform_rand, bool use_shuffle,
	bool verbose) {

	int ngrps = (int)group2index.size();

	int tot_feat_cnt = (int)matrix.data.size();

	vector<string> full_feat_ls;
	matrix.get_feature_names(full_feat_ls);
	vector<float> fast_access(full_feat_ls.size());
	for (size_t i = 0; i < full_feat_ls.size(); ++i)
		fast_access[i] = matrix.data.at(full_feat_ls[i])[selected_sample];

	features_coeff.resize(ngrps);

	//calc shapley for each variable
	if (verbose)
		MLOG("Start explain_shapely\n");
	MedTimer tm_taker;
	tm_taker.start();
	bool warn_shown = false;
	MedProgress progress_full("shapley", ngrps, 15);

	for (size_t grp_i = 0; grp_i < ngrps; ++grp_i)
	{
		double phi_i = 0;
		bool has_miss = false;
		int param_it = 0;
		while (!has_miss && param_it < group2index[grp_i].size()) {
			has_miss = fast_access[group2index[grp_i][param_it]] == missing_value;
			++param_it;
		}
		if (has_miss)
			continue;

		int grps_opts = ngrps - 1;
		vector<vector<bool>> all_opts;
		bool iter_all = true;
		int max_loop = max_tests;
		double nchoose = grps_opts <= 20 ? pow(2, grps_opts) : -1;
		if (grps_opts <= 20 && nchoose < max_loop)
			max_loop = nchoose;
		else {
			iter_all = false;
			if (!warn_shown && verbose && grps_opts <= 20)
				MLOG("Warning have %d options, and max_test is %d\n", (int)nchoose, max_loop);
		}

		if (grps_opts <= 20 && (iter_all || float(nchoose) / max_tests >= select_from_all)) {
			list_all_options_binary(grps_opts, all_opts);
			vector<bool> empty_vec(grps_opts), full_vec(grps_opts, true);
			all_opts.push_back(empty_vec);
			all_opts.push_back(full_vec);

			//select random masks from all options when not iterating all options:
			if (!iter_all) {
				if (!warn_shown && verbose) {
					MLOG("Warning: not iterating all in feature %zu has %zu candidates, has %d options, max_test=%d\n",
						grp_i, grps_opts, (int)nchoose, max_loop);
#pragma omp critical
					warn_shown = true;
				}
				shuffle(all_opts.begin(), all_opts.end(), gen);
				all_opts.resize(max_loop);
			}
		}
		else
			sample_options_SHAP(grps_opts, all_opts, max_loop, gen, sample_masks_with_repeats, uniform_rand, use_shuffle);

		//complete all_opts to nfeats size:
		bool deafult_not_selected = true; //mark all the rest(missing values that aren't tested) as fixed to missing value
		for (size_t i = 0; i < all_opts.size(); ++i)
		{
			vector<bool> mask(full_feat_ls.size(), deafult_not_selected);
			vector<bool> &truncated_mask = all_opts[i];
			for (int j = 0; j < grps_opts; ++j)
			{
				bool mask_val = truncated_mask[j];
				int cand_idx = j + int(j >= grp_i);
				for (int ind : group2index[cand_idx]) //set all group indexes as in mask
					mask[ind] = mask_val;
				//else keeps default (was missing value) - which means keep as missing value or iterate through
			}
			for (int ind : group2index[grp_i])
				mask[ind] = false; //mark always as false the selected feature to test

			truncated_mask = std::move(mask);
		}

		//build test matrix from all_opts:
		MedTimer tm;
		tm.start();

		vector<bool> mask_grp(tot_feat_cnt);
		for (int ind : group2index[grp_i])
			mask_grp[ind] = true;

		//collect score for each permutition of missing values:
		int end_l = grps_opts;
		vector<float> full_pred_all_masks_without(static_cast<size_t>(max_loop) * static_cast<size_t>(tot_feat_cnt)), full_pred_all_masks_with(static_cast<size_t>(max_loop) * static_cast<size_t>(tot_feat_cnt));
		for (int i = 0; i < max_loop; ++i) {
			float *mat_without = &full_pred_all_masks_without[static_cast<size_t>(i) * static_cast<size_t>(tot_feat_cnt)];
			float *mat_with = &full_pred_all_masks_with[static_cast<size_t>(i) * static_cast<size_t>(tot_feat_cnt)];
			for (size_t j = 0; j < tot_feat_cnt; ++j)
				if (!all_opts[i][j])
					mat_without[j] = missing_value;
				else
					mat_without[j] = fast_access[j];

			for (size_t j = 0; j < tot_feat_cnt; ++j)
				if (!all_opts[i][j] && !mask_grp[j])
					mat_with[j] = missing_value;
				else
					mat_with[j] = fast_access[j];
		}

		vector<float> preds_with, preds_without;
		predictor->predict(full_pred_all_masks_without, preds_without, max_loop, (int)full_feat_ls.size());
		predictor->predict(full_pred_all_masks_with, preds_with, max_loop, (int)full_feat_ls.size());

		for (int i = 0; i < max_loop; ++i) {
			int f_cnt = 0;
			for (size_t j = 0; j < all_opts[i].size(); ++j)
				f_cnt += int(all_opts[i][j]);

			float f_diff = preds_with[i] - preds_without[i];
			int p1 = f_cnt;
			int p2 = end_l - p1;
			double c = get_c(p1, p2, end_l + 1);
			phi_i += c * f_diff;
		}

		if (verbose)
			progress_full.update();
		features_coeff[grp_i] = phi_i;
	}

	tm_taker.take_curr_time();
	if (verbose)
		MLOG("Done explain_shapely. took %2.1f seconds\n", tm_taker.diff_sec());
}

int collect_mask(const vector<float> &x, const vector<bool> &mask, const SamplesGenerator<float> &sampler_gen, mt19937 &rnd_gen
	, int sample_per_row, void *sampling_params, const vector<string> &feat_names, map<string, vector<float>> &gen_matrix) {

	if (sampler_gen.use_vector_api) {
		int size_before = (int)gen_matrix[feat_names.front()].size();
		MedMat<float> mat_inp(x, (int)x.size());
		vector<vector<bool>> masks = { mask };
		MedMat<float> res; //the result matrix
		sampler_gen.get_samples(res, sample_per_row, sampling_params, masks, mat_inp, rnd_gen);


#pragma omp critical
		for (int i = 0; i < feat_names.size(); ++i) {
			gen_matrix[feat_names[i]].resize(size_before + res.nrows);
			for (int k = 0; k < res.nrows; ++k)
				gen_matrix[feat_names[i]][size_before + k] = res(k, i);
		}

		return res.nrows;
	}
	else {
		//no parallel:
		int size_before = (int)gen_matrix[feat_names.front()].size();
		sampler_gen.get_samples(gen_matrix, sampling_params, mask, x, rnd_gen);
		return (int)gen_matrix[feat_names.front()].size() - size_before;
	}
}

template<typename T> double mean_vec(const T *v, int len) {

	double s = 0;
	for (size_t i = 0; i < len; ++i)
		s += v[i];

	if (len == 0)
		MTHROW_AND_ERR("No values given for mean_vec. Cannot return anything valid\n");

	return s / len;
}

template<typename T> void medial::shapley::explain_shapley(const MedFeatures &matrix, int selected_sample, int max_tests,
	MedPredictor *predictor, const vector<vector<int>>& group2index, const vector<string> &groupNames,
	const SamplesGenerator<T> &sampler_gen, mt19937 &rnd_gen, int sample_per_row, void *sampling_params,
	vector<float> &features_coeff, bool use_random_sample, bool verbose) {

	mt19937 gen(globalRNG::rand());

	int tot_feat_cnt = (int)matrix.data.size();
	vector<string> full_feat_ls;
	matrix.get_feature_names(full_feat_ls);
	vector<float> fast_access(tot_feat_cnt);
	for (size_t i = 0; i < full_feat_ls.size(); ++i)
		fast_access[i] = matrix.data.at(full_feat_ls[i])[selected_sample];

	int ngrps = (int)group2index.size();
	features_coeff.resize(ngrps);

	//calc shapley for each variable
	if (verbose)
		MLOG("Start explain_shapely\n");
	MedTimer tm_taker;
	tm_taker.start();
	bool warn_shown = false;
	float select_from_all = (float)0.8;
	MedProgress tm_full("Shapley_Feature", ngrps, 15, 1);

	for (size_t param_i = 0; param_i < ngrps; ++param_i)
	{
		double phi_i = 0;
		int grps_opts = ngrps - 1;
		//iterate on all other features  execpt param_i, and other features that are already missing in the given example
		bool iter_all = true;
		int max_loop = max_tests;
		double nchoose = grps_opts <= 20 ? pow(2, grps_opts) : -1;
		if (grps_opts <= 20 && nchoose < max_loop)
			max_loop = nchoose;
		else {
			iter_all = false;
			if (!warn_shown && verbose && grps_opts <= 20)
				MLOG("Warning have %d options, and max_test is %d\n", (int)nchoose, max_loop);
		}

		vector<vector<bool>> all_opts;
		vector<int> opt_s_sizes(max_loop);
		if (grps_opts <= 20 && (iter_all || float(nchoose) / max_tests >= select_from_all)) {
			list_all_options_binary(grps_opts, all_opts);
			vector<bool> empty_vec(grps_opts), full_vec(grps_opts, true);
			all_opts.push_back(empty_vec);
			all_opts.push_back(full_vec);

			//select random masks from all options when not iterating all options:
			if (!iter_all) {
				if (!warn_shown && verbose) {
					MLOG("Warning: not iterating all in feature %zu has %zu candidates, has %d options, max_test=%d\n",
						param_i, grps_opts, (int)nchoose, max_loop);
#pragma omp critical
					warn_shown = true;
				}
				shuffle(all_opts.begin(), all_opts.end(), gen);
				all_opts.resize(max_loop);
			}
		}
		else
			sample_options_SHAP(grps_opts, all_opts, max_loop, gen, false, use_random_sample, false);
		//populate opt_s_sizes with mask sizes
		for (int i = 0; i < max_loop; ++i) {
			int f_cnt = 0;
			for (size_t j = 0; j < all_opts[i].size(); ++j)
				f_cnt += int(all_opts[i][j]);
			opt_s_sizes[i] = f_cnt;
		}

		//complete all_opts to nfeats size using groups:
		bool deafult_not_selected = true; //mark all the rest(missing values that aren't tested) as fixed to missing value
		for (size_t i = 0; i < all_opts.size(); ++i)
		{
			vector<bool> mask(full_feat_ls.size(), deafult_not_selected);
			vector<bool> &truncated_mask = all_opts[i];

			for (int j = 0; j < grps_opts; ++j)
			{
				bool mask_val = truncated_mask[j];
				int cand_idx = j + int(j >= param_i);
				for (int ind : group2index[cand_idx]) //set all group indexes as in mask
					mask[ind] = mask_val;
				//else keeps default (was missing value) - which means keep as missing value or iterate through
			}
			for (int ind : group2index[param_i])
				mask[ind] = false; //mark always as false the selected feature to test

			truncated_mask = std::move(mask);
		}

		//build test matrix from all_opts:
		MedProgress progress("VV_Shapley(Feat " + to_string(param_i + 1) +
			" out of " + to_string(ngrps) + ")", max_loop, 15, 1);
		//collect score for each permutition of missing values:
		int end_l = grps_opts;

		MedFeatures full_gen_samples;
		full_gen_samples.attributes = matrix.attributes;
		vector<int> splits_without(max_loop), splits_with(max_loop);

		for (int i = 0; i < max_loop; ++i) {

			int add_cnt = collect_mask(fast_access, all_opts[i], sampler_gen, rnd_gen, sample_per_row, sampling_params,
				full_feat_ls, full_gen_samples.data);

			splits_without[i] = add_cnt;

			vector<bool> with_mask = all_opts[i];
			for (int ind : group2index[param_i])
				with_mask[ind] = true; //mark always as false the selected feature to test
			int add_cnt_with = collect_mask(fast_access, with_mask, sampler_gen, rnd_gen, sample_per_row, sampling_params,
				full_feat_ls, full_gen_samples.data);
			splits_with[i] = add_cnt_with;

			if (verbose)
				progress.update();
		}
		full_gen_samples.samples.resize(full_gen_samples.data.begin()->second.size());
		full_gen_samples.init_pid_pos_len();

		predictor->predict(full_gen_samples);
		vector<float> preds_with, preds_without;
		preds_with.reserve(full_gen_samples.samples.size() / 2);
		preds_without.reserve(full_gen_samples.samples.size() / 2);
		vector<int> cumsum_without(max_loop), cumsum_with(max_loop);
		int curr_smp_pos = 0;
		for (int i = 0; i < max_loop; ++i) {
			int cnt_1 = splits_without[i];
			int cnt_2 = splits_with[i];

			int without_st = (int)preds_without.size();
			int with_st = (int)preds_with.size();
			cumsum_without[i] = without_st;
			cumsum_with[i] = with_st;

			for (size_t j = 0; j < cnt_1; ++j)
				preds_without.push_back(full_gen_samples.samples[curr_smp_pos + j].prediction[0]);
			for (size_t j = 0; j < cnt_2; ++j)
				preds_with.push_back(full_gen_samples.samples[curr_smp_pos + cnt_1 + j].prediction[0]);
			curr_smp_pos += cnt_1 + cnt_2;
		}

		vector<int> sizes_hist(ngrps); //can't pass ngrps - feature_i is excluded. uses grps_opts:=ngrps-1
		for (int i = 0; i < max_loop; ++i) {
			int f_cnt = opt_s_sizes[i];
			++sizes_hist[f_cnt];
		}
		int non_zero_grp_sampled = 0;
		for (size_t i = 0; i < ngrps; ++i)
			if (sizes_hist[i] > 0)
				++non_zero_grp_sampled;

		for (int i = 0; i < max_loop; ++i) {
			float score_without, score_with;
			int f_cnt = opt_s_sizes[i];

			score_without = mean_vec(preds_without.data() + cumsum_without[i], splits_without[i]);
			score_with = mean_vec(preds_with.data() + cumsum_with[i], splits_with[i]);

			float f_diff = score_with - score_without;

			int p1 = f_cnt;
			int p2 = end_l - p1;
			double c;
			if (iter_all)
				c = get_c(p1, p2, end_l + 1);
			else
				c = 1.0 / (non_zero_grp_sampled*sizes_hist[f_cnt]);
			phi_i += c * f_diff;
		}

		if (verbose)
			tm_full.update();
		features_coeff[param_i] = phi_i;
	}

	tm_taker.take_curr_time();
	if (verbose)
		MLOG("Done explain_shapely. took %2.1f seconds\n", tm_taker.diff_sec());

	if (verbose) {
		vector<pair<float, string>> feat_rank(ngrps);
		for (size_t i = 0; i < feat_rank.size(); ++i)
		{
			feat_rank[i].first = features_coeff[i];
			feat_rank[i].second = groupNames[i];
		}
		sort(feat_rank.begin(), feat_rank.end(), comp_score_flt_str);

		for (int i = 0; i < ngrps; ++i)
			MLOG("EXPLAIN #%d by %s : feat_score=%f\n",
				i + 1, feat_rank[i].second.c_str(), feat_rank[i].first);

		double sm = 0;
		stringstream str_buff;
		for (int i = 0; i < ngrps - 1; ++i) {
			str_buff << feat_rank[i].first << " +";
			sm += feat_rank[i].first;
		}
		sm += feat_rank[ngrps - 1].first;
		str_buff << feat_rank[ngrps - 1].first << "=" << sm;
		if (!matrix.samples[selected_sample].prediction.empty())
			str_buff << " predictor_score=" << matrix.samples[selected_sample].prediction[0];
		MLOG("%s\n", str_buff.str().c_str());
	}
}

template void medial::shapley::explain_shapley<float>(const MedFeatures &matrix, int selected_sample, int max_tests,
	MedPredictor *predictor, const vector<vector<int>>& group2index, const vector<string> &groupNames,
	const SamplesGenerator<float> &sampler_gen, mt19937 &rnd_gen, int sample_per_row, void *sampling_params,
	vector<float> &features_coeff, bool use_random_sample, bool verbose);

// Generate sampled matrix
void generate_samples(const MedFeatures& data, int isample, const vector<vector<bool>>& masks, SamplesGenerator<float> *generator,
	void *params, MedFeatures *out_data) {

	if (generator->use_vector_api) {
		int ncols = (int)data.data.size();
		MedMat<float> in((int)masks.size(), ncols), out((int)masks.size(), ncols);

		int icol = 0;
		for (auto& rec : data.data) {
#pragma omp parallel for
			for (int irow = 0; irow < masks.size(); irow++)
				in(irow, icol) = rec.second[isample];
			icol++;
		}

		generator->get_samples(out, 1, params, masks, in);

		icol = 0;
		for (auto& rec : data.data) {
			out_data->attributes[rec.first] = data.attributes.at(rec.first);
			out_data->data[rec.first].resize(masks.size());

#pragma omp parallel for
			for (int irow = 0; irow < masks.size(); irow++)
				(out_data->data)[rec.first][irow] = out(irow, icol);
			icol++;
		}
	}
	else {
		vector<float> init_data(data.data.size());
		int icol = 0;
		for (auto it = data.data.begin(); it != data.data.end(); ++it)
		{
			init_data[icol] = it->second[isample];
			++icol;
		}

		for (int i = 0; i < masks.size(); ++i)
			generator->get_samples(out_data->data, params, masks[i], init_data);
	}
}

// Learn a Shapely-Lime model
// Helper - 
// Calculate:
//			(n - 1.0) / (n choose k)*k*(n - k));
//			------------------------------------
//			(n - 1.0) / (n choose k0)*k0*(n - k0));
// where k0 = n*p

double get_normalized_weight(int n, int k, int k0) {

	double logFactor = 0;
	for (int i = k0 + 1; i <= k; i++)
		logFactor += log((float)i);
	for (int i = (n - k) + 1; i <= (n - k0); i++)
		logFactor -= log((float)i);

	return exp(logFactor);
}

double get_normalized_weight(int n, int k, float p) {

	int k0 = n * p + 0.5;
	if (k > k0)
		return get_normalized_weight(n, k, k0);
	else
		return 1.0 / get_normalized_weight(n, k0, k);
}

// Get weights according to the original LIME paper = exp(-d(X)/n)
void get_lime_weights(const MedFeatures& data, int isample, MedFeatures& p_features, vector<float>& wgts) {

	int n = (int)p_features.samples.size();
	wgts.resize(n, 0.0);

	for (auto& rec : data.data) {
		vector<float>& vec = p_features.data[rec.first];
		for (int i = 0; i < n; i++)
			wgts[i] += (rec.second[isample] - vec[i]) * (rec.second[isample] - vec[i]);
	}

	for (int i = 0; i < n; i++)
		wgts[i] = exp(-wgts[i] / data.data.size());
}

// Get shapley weights -
// Sum of weights per k = (n-1)/(k*(n-k))
// # of samples with k = Nk
// => Weight per samples with k = (n-1)/((Nk*k*(n-k))
void get_shapley_weights(int n, int ngrps, vector<int>&ks, vector<float>& wgts) {

	wgts.resize(n, 0.0);

	vector<int> nk(ngrps, 0);
	for (int k : ks)
		nk[k]++;

	//	for (int i = 1; i < ngrps; i++)
	//		cerr << i << " " << nk[i] << " " << (ngrps - 1.0) / ((nk[i] * i * (ngrps - i))) << "\n";

	for (int i = 0; i < n; i++)
		wgts[i] = (ngrps - 1.0) / ((nk[ks[i]] * ks[i] * (ngrps - ks[i])));

}

// Main functions for Shapley_lime with forced groups
void medial::shapley::get_shapley_lime_params(const MedFeatures& data, const MedPredictor *model,
	SamplesGenerator<float> *generator, float p, int n, LimeWeightMethod weighting, float missing,
	void *params, const vector<vector<int>>& group2index, const vector<string>& group_names, vector<vector<float>>& alphas) {

	// forced = groups that are forced to be 'ON'
	vector<vector<int>> forced(data.samples.size(), vector<int>(group2index.size(), 0));
	get_shapley_lime_params(data, model, generator, p, n, weighting, missing, params, group2index, group_names, forced, alphas);
}

void medial::shapley::get_shapley_lime_params(const MedFeatures& data, const MedPredictor *model,
	SamplesGenerator<float> *generator, float p, int n, LimeWeightMethod weighting, float missing,
	void *params, const vector<vector<int>>& group2index, const vector<string>& group_names, vector<vector<int>>& forced, vector<vector<float>>& alphas) {

	int N_TH = omp_get_max_threads();
	vector<mt19937> gen(N_TH);

	for (size_t i = 0; i < N_TH; ++i)
		gen[i] = mt19937(globalRNG::rand());

	uniform_real_distribution<> coin_dist(0, 1);

	int nsamples = (int)data.samples.size();
	if (nsamples == 0) return;

	vector<string> features;
	data.get_feature_names(features);
	int nftrs = (int)features.size();
	int ngrps = (int)group_names.size();

	alphas.resize(nsamples);

	MedFeatures p_features;
	p_features.attributes = data.attributes;
	p_features.samples.resize(n);

	// Generate samples and predict
	MedProgress tm("Lime", nsamples, 30, 1);
	double sample_size_sum = 0.0;
	for (int isample = 0; isample < nsamples; isample++) {
		// Forced groups
		int nForced = 0;
		for (int _forced : forced[isample])
			nForced += _forced;
		vector<int> grp2trainIdx(ngrps, -1), trainIdx2grp(ngrps - nForced);
		int trainIdx = 0;
		for (int i = 0; i < ngrps; i++) {
			if (!forced[isample][i]) {
				grp2trainIdx[i] = trainIdx;
				trainIdx2grp[trainIdx] = i;
				trainIdx++;
			}
		}

		// Generate random masks
		MedMat<float> train((int)trainIdx2grp.size(), n);
		vector<float> wgts(n);
		vector<vector<bool>> masks(n, vector<bool>(nftrs));

		vector<bool> missing_v(nftrs, false), missing_g(ngrps, true);
		int nMissing = 0;

		for (int igrp = 0; igrp < ngrps; igrp++) {
			for (int iftr : group2index[igrp]) {
				if (data.data.at(features[iftr])[isample] == missing) {
					missing_v[iftr] = true;
					nMissing++;
				}
				else
					missing_g[igrp] = false;
			}
		}

		if (nMissing == nftrs)
			MTHROW_AND_ERR("All values are missing for entry %d , Cannot explain\n", isample);

		// Generate maks and train-data
		vector<int> ks(n);
		vector<int> sample_size(n);
#pragma omp parallel for schedule(dynamic)
		for (int irow = 0; irow < n; irow++) {

			int n_th = omp_get_thread_num();
			// Mask
			int S = 0;
			if (p == 0) {
				// Gropus that are not all missing nor forced
				vector<int> valid_grps;
				for (int igrp = 0; igrp < ngrps; igrp++) {
					if (!forced[isample][igrp]) {
						for (int iftr : group2index[igrp]) {
							if (!missing_v[iftr]) {
								valid_grps.push_back(igrp);
								break;
							}
						}
					}
				}

				// randomly select S from 1 to nValid and choose S groups to keep
				S = 1 + (int)(coin_dist(gen[n_th])*(valid_grps.size() - 1));
				for (int idx = 0; idx < S; idx++) {
					int grp_idx = (int)(coin_dist(gen[n_th])*(valid_grps.size()));
					int igrp = valid_grps[grp_idx];
					for (int iftr : group2index[igrp]) {
						if (missing_v[iftr])
							masks[irow][iftr] = false;
						else
							masks[irow][iftr] = true;
					}
					train(grp2trainIdx[igrp], irow) = 1.0;
					valid_grps[grp_idx] = valid_grps.back();
					valid_grps.pop_back();
				}
				// Add masks for forced groups
				for (size_t igrp = 0; igrp < forced[isample].size(); igrp++) {
					if (forced[isample][igrp]) {
						for (int iftr : group2index[igrp]) {
							if (missing_v[iftr])
								masks[irow][iftr] = false;
							else
								masks[irow][iftr] = true;
						}
					}
				}
			}
			else {
				while (S == 0 || S == ngrps) {
					S = 0;
					for (int igrp = 0; igrp < ngrps; igrp++) {
						if (!forced[isample][igrp]) {
							bool grp_mask = forced[isample][igrp] || (coin_dist(gen[n_th]) < p);
							if (grp_mask) { // Keep, unless all are missing
								size_t nMissing = 0;
								for (int iftr : group2index[igrp]) {
									if (missing_v[iftr]) {
										nMissing++;
										masks[irow][iftr] = false;
									}
									else
										masks[irow][iftr] = true;
								}

								if (nMissing == group2index[igrp].size()) // All are missing ...
									train(grp2trainIdx[igrp], irow) = 0.0;
								else {
									train(grp2trainIdx[igrp], irow) = 1.0;
									S++;
								}
							}
							else { // Mask
								for (int iftr : group2index[igrp])
									masks[irow][iftr] = false;
								train(grp2trainIdx[igrp], irow) = 0.0;
							}
						}
						else { // Forced group
							for (int iftr : group2index[igrp]) {
								if (missing_v[iftr])
									masks[irow][iftr] = false;
								else
									masks[irow][iftr] = true;
							}
						}
					}
				}
			}

			ks[irow] = S;

			// Weights
			if (weighting == LimeWeightShap)
				wgts[irow] = get_normalized_weight(ngrps, S, p);
			else if (weighting == LimeWeightUniform)
				wgts[irow] = 1.0;
		}

		// Generate sampled data
		generate_samples(data, isample, masks, generator, params, &p_features);

		// effective ngrps
		int eff_ngrps = 0;
		for (int igrp = 0; igrp < ngrps; igrp++) {
			if ((!forced[isample][igrp]) && (!missing_g[igrp]))
				eff_ngrps++;
		}

		// Get weights
		if (weighting == LimeWeightLime)
			get_lime_weights(data, isample, p_features, wgts);
		else if (weighting == LimeWeightSum)
			get_shapley_weights(n, eff_ngrps, ks, wgts);

		double s1 = 0.0, s2 = 0.0;
		for (int i = 0; i < n; i++) {
			s1 += fabs(wgts[i]);
			s2 += static_cast<double>(wgts[i]) * static_cast<double>(wgts[i]);
		}
		sample_size_sum += s1 * s1 / s2;

		// Get predictions for samples
		model->predict(p_features);

		vector<float> preds(n);
		double sum = 0;
		for (int irow = 0; irow < n; ++irow) {
			preds[irow] = p_features.samples[irow].prediction[0];
			sum += preds[irow];
		}

		// Learn linear model
		MedLM lm;
		lm.params.rfactor = (float)0.98;
		float mean, std;

		// No all-zero columns
		if (eff_ngrps == ngrps - nForced) {
			train.transposed_flag = 1;

			train.normalize(train.Normalize_Rows);
			medial::stats::get_mean_and_std_without_cleaning(preds, mean, std);
			for (unsigned int i = 0; i < preds.size(); i++)
				preds[i] -= mean;
			train.normalized_flag = 1;

			lm.learn(train, preds, wgts);
		}
		// With all-zero columns - need to remove
		else {
			MedMat<float> eff_train(eff_ngrps, n);

			int eff_igrp = 0;
			for (int igrp = 0; igrp < ngrps; igrp++) {
				if ((!forced[isample][igrp]) & (!missing_g[igrp])) {
					for (int irow = 0; irow < n; irow++)
						eff_train(eff_igrp, irow) = train(grp2trainIdx[igrp], irow);
					eff_igrp++;
				}
			}

			eff_train.transposed_flag = 1;

			eff_train.normalize(eff_train.Normalize_Rows);
			medial::stats::get_mean_and_std_without_cleaning(preds, mean, std);
			for (unsigned int i = 0; i < preds.size(); i++)
				preds[i] -= mean;
			eff_train.normalized_flag = 1;

			lm.learn(eff_train, preds, wgts);
		}

		// Extract alphas
		alphas[isample].resize(ngrps);
		int eff_igrp = 0;
		for (int igrp = 0; igrp < ngrps; igrp++) {
			if (!forced[isample][igrp]) {
				if (!missing_g[igrp]) {
					alphas[isample][igrp] = lm.b[eff_igrp];
					eff_igrp++;
				}
				else
					alphas[isample][igrp] = 0;
			}
		}

		tm.update();
	}

	MLOG("LimeExplainer: mean effective sample size for %d and % f = %f\n", n, p, sample_size_sum / nsamples);
}

// Iterative Shapley_lime
void medial::shapley::get_iterative_shapley_lime_params(const MedFeatures& data, const MedPredictor *model,
	SamplesGenerator<float> *generator, float p, int n, LimeWeightMethod weighting, float missing,
	void *params, const vector<vector<int>>& group2index, const vector<string>& group_names, const MedMat<float>& abs_cov_mat, int iteration_cnt, vector<vector<float>>& alphas, bool max_in_groups) {

	// forced = groups that are forced to be 'ON'
	vector<vector<int>> forced(data.samples.size(), vector<int>(group2index.size(), 0));
	vector<unsigned> feature_set(data.data.size());
	for (unsigned i = 0; i < group2index.size(); ++i)
		for (int ii : group2index[i])
			feature_set[ii] = i;


	// Add iteratively
	int stop_iter = (int)group2index.size() - 1;
	if (iteration_cnt > 0 && iteration_cnt < stop_iter)
		stop_iter = iteration_cnt;

	alphas.resize(data.samples.size());
	for (size_t i = 0; i < alphas.size(); i++)
		alphas[i].resize(group2index.size());

	for (size_t i = 0; i < stop_iter; i++) {
		MLOG("Working with %d/%d forced\n", i + 1, stop_iter);

		// Get Shapley Values
		vector<vector<float>> temp_alphas;
		get_shapley_lime_params(data, model, generator, p, n, weighting, missing, params, group2index, group_names, forced, temp_alphas);

		// Find optimal NEW alpha per sample
		for (size_t isample = 0; isample < temp_alphas.size(); isample++) {
			int opt_grp = -1;
			float max_contrib = 0.0;
			for (size_t igrp = 0; igrp < temp_alphas[isample].size(); igrp++) {
				if (!forced[isample][igrp]) {
					// Correct for covariance
					float contrib = 0;
					if (abs_cov_mat.nrows) { //do fix:
						MedMat<double> feat_single((int)data.data.size(), 1);
						MedMat<double> tmp_sing((int)group_names.size(), 1);
						feat_single.set_val(1); //TODO: use real feature contribution on single features
						for (size_t kk = 0; kk < group_names.size(); ++kk)
							tmp_sing(kk, 0) = (double)temp_alphas[isample][kk];

						fix_feature_dependency_in_groups(abs_cov_mat, feat_single, tmp_sing, feature_set.data(), forced[isample], max_in_groups);
						/*for (size_t jgrp = 0; jgrp < temp_alphas[isample].size(); jgrp++) {
							if (!forced[isample][jgrp])
								contrib += temp_alphas[isample][jgrp] * abs_cov_mat(igrp, jgrp);
						}*/
					}
					else
						contrib = temp_alphas[isample][igrp];

					if (fabs(contrib) >= fabs(max_contrib)) {
						max_contrib = contrib;
						opt_grp = (int)igrp;
					}
				}
			}

			alphas[isample][opt_grp] = max_contrib;
			forced[isample][opt_grp] = 1;
		}
	}
}
enum SelectionMode
{
	BY_SCORE, //selects the one minimize score diff to original
	BY_MAX_CHANGE, //selects the one maximize score diff from prev
	BY_ALL //use all 3 metrics to select and calc new measure
};

void update_best_selection(double avg_diff, double std_in_score, double diff_prev, double avg_score, int grp_i,
	SelectionMode mode, int &selected_idx, double &selected_value, float &selected_score,
	double param_all_alpha = 1, double param_all_beta = 1,
	double param_all_k1 = 2, double param_all_k2 = 2) {

	double calc_score = 0;

	switch (mode)
	{
	case BY_SCORE:
		if (selected_idx < 0 || avg_diff < selected_value) {
			selected_idx = grp_i;
			selected_score = avg_score;
			selected_value = avg_diff;
		}
		break;
	case BY_MAX_CHANGE:
		if (selected_idx < 0 || diff_prev > selected_value) {
			selected_idx = grp_i;
			selected_score = avg_score;
			selected_value = diff_prev;
		}
		break;
	case BY_ALL:
		//TODO: fix the weight function shape!! it's wrong!!
		if (avg_diff > 0)
			calc_score = pow(avg_diff, param_all_k1);
		if (std_in_score > 0)
			calc_score += param_all_alpha * pow(std_in_score, param_all_k1);
		if (diff_prev > 0)
			calc_score -= param_all_beta * pow(diff_prev, 1 / param_all_k2);

		if (selected_idx < 0 || calc_score < selected_value) {
			selected_idx = grp_i;
			selected_score = avg_score;
			selected_value = calc_score;
		}

		break;
	default:
		MTHROW_AND_ERR("Unsupported Mode\n");
	}

}

//return feature/groups order from 1 to size. sign is contribution influelunce
void medial::shapley::explain_minimal_set(const MedFeatures &matrix, int selected_sample, int max_tests,
	MedPredictor *predictor, float missing_value, const vector<vector<int>>& group2index
	, vector<float> &features_coeff, vector<float> &scores_history, int max_set_size
	, float baseline_score, float param_all_alpha, float param_all_beta
	, float param_all_k1, float param_all_k2, bool verbose) {

	int ngrps = (int)group2index.size();

	int tot_feat_cnt = (int)matrix.data.size();

	SelectionMode mode = SelectionMode::BY_ALL;

	vector<string> full_feat_ls;
	matrix.get_feature_names(full_feat_ls);
	vector<float> fast_access(full_feat_ls.size());
	for (size_t i = 0; i < full_feat_ls.size(); ++i)
		fast_access[i] = matrix.data.at(full_feat_ls[i])[selected_sample];

	features_coeff.resize(ngrps);
	if (matrix.samples[selected_sample].prediction.empty())
		MTHROW_AND_ERR("Error please use only after predict\n");
	float original_pred = matrix.samples[selected_sample].prediction[0];
	//calc shapley for each variable
	if (verbose)
		MLOG("Start explain_shapely minimal set\n");
	MedTimer tm_taker;
	tm_taker.start();
	MedProgress progress_full("shapley_minimal_set", ngrps, 15);

	//additional_params:
	float score_error_th = -1;
	float score_error_percentage_th = -1;
	int actual_size = 0;
	for (int grp_i = 0; grp_i < ngrps; ++grp_i)
	{
		bool has_miss = false;
		int param_it = 0;
		while (!has_miss && param_it < group2index[grp_i].size()) {
			has_miss = fast_access[group2index[grp_i][param_it]] == missing_value;
			++param_it;
		}
		if (has_miss)
			continue;
		++actual_size;
	}
	if (max_set_size > actual_size)
		max_set_size = actual_size;

	//greedy search to minimize till set size, score error, score_percent error
	int set_size = 0;
	vector<bool> current_mask(ngrps);
	float prev_score = baseline_score; // need to init to baseline score
	scores_history.reserve(max_set_size);

	while (set_size < max_set_size) {
		//find 
		double selected_value = -1;
		int selected_index = -1;
		float selected_score = -1;

		for (int grp_i = 0; grp_i < ngrps; ++grp_i)
		{
			if (current_mask[grp_i])
				continue;
			bool has_miss = false;
			int param_it = 0;
			while (!has_miss && param_it < group2index[grp_i].size()) {
				has_miss = fast_access[group2index[grp_i][param_it]] == missing_value;
				++param_it;
			}
			if (has_miss)
				continue;

			vector<bool> mask(full_feat_ls.size(), false);
			for (int i = 0; i < current_mask.size(); ++i)
				if (i == grp_i || current_mask[i])  //turn on bits:
					for (int ind : group2index[i]) //set all group indexes as in mask
						mask[ind] = true;

			//collect score for each several samples - no need for more than 1:
			vector<float> full_pred_all_masks_with(static_cast<size_t>(max_tests) * static_cast<size_t>(tot_feat_cnt));
			for (int i = 0; i < max_tests; ++i) {
				float *mat_with = &full_pred_all_masks_with[static_cast<size_t>(i) * static_cast<size_t>(tot_feat_cnt)];
				for (size_t j = 0; j < tot_feat_cnt; ++j)
					if (!mask[j])
						mat_with[j] = missing_value;
					else
						mat_with[j] = fast_access[j];
			}

			vector<float> preds_with;
			if (max_tests == 1)
				predictor->predict_single(full_pred_all_masks_with, preds_with); //faster API
			else
				predictor->predict(full_pred_all_masks_with, preds_with, max_tests, (int)full_feat_ls.size());

			//calcluate diff from original:
			double avg_diff = 0, avg_score = 0, diff_prev = 0, avg_diff_2 = 0;
			for (size_t i = 0; i < preds_with.size(); ++i) {
				avg_diff += abs(original_pred - preds_with[i]);
				avg_score += preds_with[i];
				diff_prev += abs(prev_score - preds_with[i]);
				avg_diff_2 += pow(original_pred - preds_with[i], 2); //second moment for variance calc
			}
			if (preds_with.empty())
				MTHROW_AND_ERR("Error medial::shapley::explain_minimal_set - empty samples\n");
			avg_diff /= preds_with.size(); //diff from original pred
			avg_score /= preds_with.size();
			diff_prev /= preds_with.size(); //diff from prev
			double std_in_score = sqrt(avg_diff_2 - pow(avg_diff, 2)); //variance in score
			// use all measure avg_diff, std_in_score, diff_prev to choose best next explain parameter

			update_best_selection(avg_diff, std_in_score, diff_prev, avg_score, grp_i, mode,
				selected_index, selected_value, selected_score, param_all_alpha, param_all_beta,
				param_all_k1, param_all_k2);

			if (verbose)
				progress_full.update();
		}

		++set_size;

		if (selected_score > prev_score)
			features_coeff[selected_index] = set_size;
		else
			features_coeff[selected_index] = -set_size; //negative contribution

		scores_history.push_back(selected_score);
		current_mask[selected_index] = true;
		prev_score = selected_score;

		double diff_val = abs(selected_score - original_pred);

		if (score_error_th > 0 && diff_val < score_error_th)
			break;
		if (score_error_percentage_th > 0 && original_pred > 0 &&
			100 * (diff_val / original_pred) < score_error_percentage_th)
			break;
	}

	tm_taker.take_curr_time();
	if (verbose)
		MLOG("Done explain_shapley_minimal_set. took %2.1f seconds\n", tm_taker.diff_sec());
}

//return feature/groups order from 1 to size. sign is contribution influelunce - using SampleGenerator
void medial::shapley::explain_minimal_set(const MedFeatures &matrix, int selected_sample, int max_tests,
	MedPredictor *predictor, float missing_value, const vector<vector<int>>& group2index,
	const SamplesGenerator<float> &sampler_gen, mt19937 &rnd_gen, void *sampling_params
	, vector<float> &features_coeff, vector<float> &scores_history, int max_set_size
	, float baseline_score, float param_all_alpha, float param_all_beta
	, float param_all_k1, float param_all_k2, bool verbose) {

	int ngrps = (int)group2index.size();

	int tot_feat_cnt = (int)matrix.data.size();

	SelectionMode mode = SelectionMode::BY_ALL;

	vector<string> full_feat_ls;
	matrix.get_feature_names(full_feat_ls);
	vector<float> fast_access(full_feat_ls.size());
	for (size_t i = 0; i < full_feat_ls.size(); ++i)
		fast_access[i] = matrix.data.at(full_feat_ls[i])[selected_sample];

	features_coeff.resize(ngrps);
	if (matrix.samples[selected_sample].prediction.empty())
		MTHROW_AND_ERR("Error please use only after predict\n");
	float original_pred = matrix.samples[selected_sample].prediction[0];
	//calc shapley for each variable
	if (verbose)
		MLOG("Start explain_shapely minimal set\n");
	MedTimer tm_taker;
	tm_taker.start();
	MedProgress progress_full("shapley_minimal_set", ngrps, 15);

	//additional_params:
	float score_error_th = -1;
	float score_error_percentage_th = -1;
	int actual_size = 0;
	for (int grp_i = 0; grp_i < ngrps; ++grp_i)
	{
		bool has_miss = false;
		int param_it = 0;
		while (!has_miss && param_it < group2index[grp_i].size()) {
			has_miss = fast_access[group2index[grp_i][param_it]] == missing_value;
			++param_it;
		}
		if (has_miss)
			continue;
		++actual_size;
	}
	if (max_set_size > actual_size)
		max_set_size = actual_size;

	//greedy search to minimize till set size, score error, score_percent error
	int set_size = 0;
	vector<bool> current_mask(ngrps);
	float prev_score = baseline_score; // need to init to baseline score
	scores_history.reserve(max_set_size);

	while (set_size < max_set_size) {
		//find 
		double selected_value = -1;
		int selected_index = -1;
		float selected_score = -1;


		for (int grp_i = 0; grp_i < ngrps; ++grp_i)
		{
			if (current_mask[grp_i])
				continue;
			bool has_miss = false;
			int param_it = 0;
			while (!has_miss && param_it < group2index[grp_i].size()) {
				has_miss = fast_access[group2index[grp_i][param_it]] == missing_value;
				++param_it;
			}
			if (has_miss)
				continue;

			vector<bool> mask(full_feat_ls.size(), false);
			for (int i = 0; i < current_mask.size(); ++i)
				if (i == grp_i || current_mask[i])  //turn on bits:
					for (int ind : group2index[i]) //set all group indexes as in mask
						mask[ind] = true;

			//collect score for each several samples - no need for more than 1:

			map<string, vector<float>> matrix_with_imputation;
			collect_mask(fast_access, mask, sampler_gen, rnd_gen, max_tests, sampling_params,
				full_feat_ls, matrix_with_imputation);
			int final_size = (int)matrix_with_imputation.begin()->second.size();
			vector<float> full_pred_all_masks_with(static_cast<size_t>(final_size) * static_cast<size_t>(tot_feat_cnt));
			vector<const vector<float> *> gen_data_pointers(matrix_with_imputation.size());
			int ind_ii = 0;
			for (auto it = matrix_with_imputation.begin(); it != matrix_with_imputation.end(); ++it)
			{
				gen_data_pointers[ind_ii] = &it->second;
				++ind_ii;
			}
			for (int i = 0; i < final_size; ++i) {
				float *mat_with = &full_pred_all_masks_with[static_cast<size_t>(i) * static_cast<size_t>(tot_feat_cnt)];
				for (size_t j = 0; j < tot_feat_cnt; ++j)
					mat_with[j] = gen_data_pointers[j]->at(i);
			}

			vector<float> preds_with;
			if (final_size == 1)
				predictor->predict_single(full_pred_all_masks_with, preds_with); //faster API
			else
				predictor->predict(full_pred_all_masks_with, preds_with, final_size, (int)full_feat_ls.size());

			//calcluate diff from original:
			double avg_diff = 0, avg_score = 0, diff_prev = 0, avg_diff_2 = 0;
			for (size_t i = 0; i < preds_with.size(); ++i) {
				avg_diff += abs(original_pred - preds_with[i]);
				avg_score += preds_with[i];
				diff_prev += abs(prev_score - preds_with[i]);
				avg_diff_2 += pow(original_pred - preds_with[i], 2); //second moment for variance calc
			}
			if (preds_with.empty())
				MTHROW_AND_ERR("Error medial::shapley::explain_minimal_set - empty samples\n");
			avg_diff /= preds_with.size(); //diff from original pred
			avg_score /= preds_with.size();
			diff_prev /= preds_with.size(); //diff from prev
			double std_in_score = sqrt(avg_diff_2 - pow(avg_diff, 2)); //variance in score
			// use all measure avg_diff, std_in_score, diff_prev to choose best next explain parameter

			update_best_selection(avg_diff, std_in_score, diff_prev, avg_score, grp_i, mode,
				selected_index, selected_value, selected_score, param_all_alpha, param_all_beta,
				param_all_k1, param_all_k2);

			if (verbose)
				progress_full.update();
		}

		++set_size;
		if (selected_score > prev_score)
			features_coeff[selected_index] = set_size;
		else
			features_coeff[selected_index] = -set_size; //negative contribution

		scores_history.push_back(selected_score);
		current_mask[selected_index] = true;
		prev_score = selected_score;

		double diff_val = abs(selected_score - original_pred);

		if (score_error_th > 0 && diff_val < score_error_th)
			break;
		if (score_error_percentage_th > 0 && original_pred > 0 &&
			100 * (diff_val / original_pred) < score_error_percentage_th)
			break;
	}

	tm_taker.take_curr_time();
	if (verbose)
		MLOG("Done explain_shapley_minimal_set. took %2.1f seconds\n", tm_taker.diff_sec());
}