//NTREE
// QuantizedRF - a version of RF that uses the idea of quantizing the data in order to gain speed.
//

#include "QRF.h"
#include "Utils.h"
#include <cstring>
#include <algorithm>
#include <map>
#include <random>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <thread>
#include <omp.h>

//#include <process.h>
//#include <Windows.h>

//#define DEBUG
//#define DEBUG_INIT
//#define DEBUG_ALGO
//#define DEBUG_ALGO_2

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL


//======================================================================================================================================
// QRF_Tree
//======================================================================================================================================
void QRF_Tree::print(FILE *f)
{
	fprintf(f, "N_nodes: %d\n", n_nodes); fflush(f);
	for (int i = 0; i < n_nodes; i++) {
		fprintf(f, "node %d :: size %d (%d/%d) leaf %d l %d r %d from/to %d %d split %d %f\n",
			i, nodes[i].size(), nodes[i].counts[0], nodes[i].counts[1], nodes[i].is_leaf, nodes[i].l_ind, nodes[i].r_ind, nodes[i].from_sample, nodes[i].to_sample, nodes[i].split_feat, nodes[i].split_val);
		fflush(f);
	}
}


void QRF_Tree::init_rand_state()
{
	unsigned int rand_seed = 0;
	for (int i = 0; i < 32; i++) {
		rand_seed = (rand_seed << 1) + (QRFglobalRNG::rand() & 0x1);
	}
	for (int i = 0; i < 32; i++) {
		rand_seed = (rand_seed << 3) ^ QRFglobalRNG::rand();
	}
	rand_gen.seed(rand_seed);
}


//======================================================================================================================================
// QuantizedRF
//======================================================================================================================================
/*
//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::quantize_greedy(vector<ValInd> &vals, int nsamples, int maxq, vector<float> &v)
{
	sort(vals.begin(), vals.end(), [](const ValInd &v1, const ValInd &v2) {return v1.val < v2.val;});
	int n_diff = 1;
	for (int i=1; i<nsamples; i++) {
		if (vals[i].val != vals[i-1].val)
			n_diff++;
	}

	// if the number of possible different values is less than n_diff
	// then all is well - we return that as our quantization
	if (n_diff <= maxq) {
		for (int i=0; i<nsamples; i++)
			v[i] = vals[i].val;
		return n_diff;
	}

	// our values if we have lots of options will typically be of the following form:
	// - potential min or max group, which are the missing values
	// - normally or log notmally distributed data
	// when quantizing this we want to slightly over sample the edges of the distribution
	// hence besides the large min/max column, we



	// if the number of possible different values is not too large
	// we can apply a greedy me


	// if needed transfer data to a quantized space of 0...maxq
	// ToDo:: consider next part after similar cells are joined, makes much more sense.
	float trans=0.0, fact=1.0;
	vector<float> v(nsamples);
	if (n_diff > maxq) {
		trans = vals[0].val;
		fact = (float)(maxq-1)/(float)(vals[nsamples-1].val - trans);
		for (int i=0; i<nsamples; i++) {
			double x = (double)fact*(double)(vals[i].val-trans);
			v[i] = (float)((int)(x));
		}
#ifdef DEBUG_INIT
		fprintf(stderr, "qrf: quantize_no_loss: trans %f fact %f min/max %f %f v %f %f maxq %d\n", trans, fact, vals[0].val, vals[nsamples-1].val, v[0], v[nsamples-1], maxq); fflush(stderr);
#endif
	}
	else {
#ifdef DEBUG_INIT
		fprintf(stderr, "qrf: quantize_no_loss: n_diff is %d\n", n_diff); fflush(stderr);
#endif
		for (int i=0; i<nsamples; i++)
			v[i] = vals[i].val;
	}

}
*/
//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::quantize_no_loss(vector<ValInd> &vals, int nsamples, int maxq, vector<float> &quant_val, vector<short> &qd)
{
	if (nsamples <= 0)
		return 0;

	if ((int)vals.size() != nsamples)
		fprintf(stderr, "ERROR:!!!!!! bug ... got %d elements in val, while nsamples is %d\n", (int)vals.size(), nsamples);
	// sort and check how many different values there are
	sort(vals.begin(), vals.end(), [](const ValInd &v1, const ValInd &v2) {return v1.val < v2.val; });
	int n_diff = 1;
	for (int i = 1; i < nsamples; i++) {
		if (vals[i].val != vals[i - 1].val)
			n_diff++;
	}

	// if needed transfer data to a quantized space of 0...maxq
	// ToDo:: consider next part after similar cells are joined, makes much more sense.
	float trans = 0.0, fact = 1.0;
	vector<float> v(nsamples);
	if (n_diff > maxq) {
		trans = vals[0].val;
		fact = (float)(maxq - 1) / (float)(vals[nsamples - 1].val - trans);
		for (int i = 0; i < nsamples; i++) {
			double x = (double)fact*(double)(vals[i].val - trans);
			v[i] = (float)((int)(x));
		}
#ifdef DEBUG_INIT
		fprintf(stderr, "qrf: quantize_no_loss: trans %f fact %f min/max %f %f v %f %f maxq %d\n", trans, fact, vals[0].val, vals[nsamples - 1].val, v[0], v[nsamples - 1], maxq); fflush(stderr);
#endif
	}
	else {
#ifdef DEBUG_INIT
		fprintf(stderr, "qrf: quantize_no_loss: n_diff is %d\n", n_diff); fflush(stderr);
#endif
		for (int i = 0; i < nsamples; i++)
			v[i] = vals[i].val;
	}


	// following part further compresses the quantized data, using the idea that cells that are all 
	// in one direction can be united as there will never be a reason to split them.
	// This is done while preparing the quant_val vector - which holds the actual quantization boundaries

	int i, j;

	int cnt[2], cnt_curr[2];
	cnt[0] = 0;	cnt[1] = 0;
	int k = 0;
	int n_diff2 = 0;
	float prev;
	if (tree_mode == QRF_BINARY_TREE) { // ToDo:: same works also for multicategory of size 2, also - can generalize it to any multicategory
		quant_val.clear();
		quant_val.push_back(vals[0].val);
		for (i = 0; i < nsamples; i++) {
			prev = v[i];
			n_diff2++;
			// counting all those with value similar to i
			cnt_curr[0] = 0; cnt_curr[1] = 0;
			for (j = i; j < nsamples; j++) {
				if (v[j] != prev)
					break;
				cnt_curr[(int)y[vals[j].idx]]++;
			}


			// deciding if to join it to current, or if to open a new one
			if (i == 0 || (cnt[0] == 0 && cnt_curr[0] == 0) || (cnt[1] == 0 && cnt_curr[1] == 0)) {
				cnt[0] += cnt_curr[0];
				cnt[1] += cnt_curr[1];
				quant_val[k] = vals[j - 1].val;
			}
			else {

				k++;
				quant_val.push_back(vals[j - 1].val);
				cnt[0] = cnt_curr[0];
				cnt[1] = cnt_curr[1];
			}
			i = j - 1;
		}
	}
	else {
		// regresssion tree mode
		// in this case we use the simple quantification we got
		quant_val.clear();
		quant_val.push_back(vals[0].val);
		for (i = 1; i < nsamples; i++)
			if (v[i] > v[i - 1])
				quant_val.push_back(vals[i].val);
	}
	qd.resize(nsamples);
	k = 0;
	for (i = 0; i < nsamples; i++) {
		while (vals[i].val > quant_val[k] && k < (quant_val.size() - 1))
			k++;
		qd[vals[i].idx] = k;
	}

#ifdef DEBUG_INIT
	fprintf(stderr, "qrf: quantize_no_loss: sizes %d %d n_diff %d %d no_loss %d quant_val.size %d\n", (int)quant_val.size(), (int)qd.size(), n_diff, n_diff2, k + 1, (int)quant_val.size()); fflush(stderr);
#endif

	return (k + 1);
}


//-----------------------------------------------------------------------------------------------------------------------------------
void QuantizedRF::clear()
{
	y.clear();
	yr.clear();
	ids[0].clear();
	ids[1].clear();
	quant_values.clear();
	q_data.clear();
	max_q.clear();
	cv.clear();
	test_s.clear();
	w.clear();

	NSamples = 0;
	NFeat = 0;
	MaxQ = 0;
	n_called = 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::init(float *X, int *Y, int nfeat, int nsamples, int maxq)
{
	return(init_all(X, Y, NULL, NULL, nfeat, nsamples, maxq));
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::init_regression(float *X, float *Y, const float *W, int nfeat, int nsamples, int maxq)
{
	return(init_all(X, NULL, Y, W, nfeat, nsamples, maxq));
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::init_all(float *X, int *Y, float *Yr, const float *W, int nfeat, int nsamples, int maxq)
{
	int i;

	clear();

	NSamples = nsamples;
	NFeat = nfeat;
	MaxQ = maxq + 1;
	n_called = 0;

	// sanity
	if (Y == NULL && Yr == NULL) {
		fprintf(stderr, "qrf: init: error: at least one of Y or Yr need to be non-null\n"); fflush(stderr);
		return -1;
	}

	// precomputed n*log(n) values (for entropy case)
	if (log_table.size() == 0) {
		log_table.resize(NSamples + 1);
		log_table[0] = 0; // we will always use it in an nlogn manner hence the 0 rather than -inf
		for (i = 1; i < NSamples + 1; i++)
			log_table[i] = (double)i * log((double)i);
	}

	// initializing y or yr (depends on mode)
	if (Y != NULL) {
		// binary mode
		for (i = 0; i < nsamples; i++) {
			if (Y[i] != 0 && Y[i] != 1) {
				fprintf(stderr, "qrf: init: error: >>>> weird Y (currently accepting only 0/1 values) !!! : %d\n", Y[i]); fflush(stderr);
				return -1;
			}
			y.push_back(Y[i]);
			if (Y[i])
				ids[1].push_back(i);
			else
				ids[0].push_back(i);
		}
	}
	else {
		// regression mode
		yr.assign(Yr, Yr + nsamples);

		// This is used to make histogram quicker - we add only active labels (assumed that it sparse);
		if (tree_mode == QRF_MULTILABEL_ENTROPY_TREE)
		{ 
			int log_categ = (int)log2(n_categ);
			yr_multilabel.resize(nsamples);
			for (int i = 0; i < nsamples; i++)
			{
				for (int j = 0; j < log_categ; j++)
				{
					if ((1 << j) & int(yr[i]))
						yr_multilabel[i].push_back(j);
				}
			}
		}


		if (W != NULL)
			w.assign(W, W + nsamples);
#ifdef DEBUG
		fprintf(stderr, "QRF_init(): regression mode, yr size is %d\n", (int)yr.size()); fflush(stderr);
#endif
	}

#ifdef DEBUG
	fprintf(stderr, "QRF init():: nfeat %d nsamples %d maxq %d ids0 %d ids1 %d y %d yr %d\n",
		nfeat, nsamples, maxq, (int)ids[0].size(), (int)ids[1].size(), (int)y.size(), (int)yr.size());
	fflush(stderr);
#endif

	// quantizing data values to a (relatively) small range, allowing for much faster tree building later.
	max_q.resize(nfeat);
	quant_values.clear();
	quant_values.resize(nfeat);
	q_data.clear();
	q_data.resize(nfeat);
	//vector<ValInd> vals(nsamples);
#pragma omp parallel for
	for (i = 0; i < nfeat; i++) {
#ifdef DEBUG_INIT
		fprintf(stderr, "QRF init:: working on feature i=%d\n", i); fflush(stderr);
#endif
		vector<ValInd> vals(nsamples);
		vector<float> quant_val;
		vector<short> qd;
		for (size_t j = 0; j < nsamples; j++) {
			vals[j].val = X[(size_t)nfeat*j + i];
			vals[j].idx = (int)j;
		}

		max_q[i] = quantize_no_loss(vals, nsamples, maxq, quant_values[i], q_data[i]);
		//max_q[i] = quantize_no_loss(vals, nsamples, maxq, quant_val, qd);
		//quant_values.push_back(quant_val);
		//q_data.push_back(qd);
	}

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::get_regression_Tree(int *sampsize, int ntry, QRF_Tree &tree)
{
	int i, j;

	n_called++;
	vector<int> take_group;
	if (n_groups > 0) {
		take_group.resize(n_groups, 0);
		uniform_int_distribution<> dist(0, n_groups - 1);
		for (i = 0; i < n_groups; i++)
			take_group[dist(tree.rand_gen)] = 1;
	}

	//	fprintf(stderr, "get_regression_Tree: tree_mode %d n_categ %d sampsize %d %d\n", tree_mode, n_categ, sampsize[0], sampsize[1]);

#ifdef DEBUG_ALGO
	fprintf(stderr, "get_regression_Tree() n_called %d\n", n_called);
#endif
	if (sampsize == NULL || (tree_mode == QRF_REGRESSION_TREE)) {
		int n;
		// in this case we simply randomize a bootstrap on all samples
		if ((sampsize == NULL) || sampsize[0] == 0)
			n = (int)yr.size();
		else
			n = sampsize[0];
		tree.sample_ids.resize(n);
		tree.max_nodes = 2 * n;
		tree.nodes.resize(tree.max_nodes);
		tree.n_nodes = 0;
		uniform_int_distribution<> dist(0, (int)yr.size() - 1);
		if (take_all_samples) { //when creating 1 tree - look at best seperation with min_node (similar to knn where k=min_node)
			j = 0;
			while (j < n) {
				tree.sample_ids[j] = j;
				++j;
			}
		}
		else if (n_groups == 0) {
			for (j = 0; j < n; j++) {
				int r = dist(tree.rand_gen);
				tree.sample_ids[j] = r;
			}
		}
		else {
			j = 0;
			while (j < n) {
				int r = dist(tree.rand_gen);
				if (take_group[groups[r]])
					tree.sample_ids[j++] = r;
			}
		}
	}
	else {
		// in this case we need to choose sampsize[i] samples from category i
		vector<vector<int>> categ_ids(n_categ);
		tree.sample_ids.clear();
		for (j = 0; j < NSamples; j++)
			categ_ids[(int)yr[j]].push_back(j);
		for (i = 0; i < n_categ; i++) {
			//fprintf(stderr,"Categ %d: %d samples needed %d\n",i,categ_ids[i].size(),sampsize[i]); fflush(stderr);
			uniform_int_distribution<> dist(0, (int)categ_ids[i].size() - 1);
			if (n_groups == 0) {
				for (j = 0; j < sampsize[i]; j++) {
					int r = dist(tree.rand_gen);
					tree.sample_ids.push_back(categ_ids[i][r]);
				}
			}
			else {
				j = 0;
				while (j < sampsize[i]) {
					int r = categ_ids[i][dist(tree.rand_gen)];
					if (take_group[groups[r]]) {
						tree.sample_ids.push_back(r);
						j++;
					}
				}
			}
		}
	}

	int n = (int)tree.sample_ids.size();
	tree.max_nodes = 2 * (1 +  (n / min_split_node_size));
	tree.nodes.resize(tree.max_nodes);
	tree.n_nodes = 0;
	// ToDo :: when a sample is given more than once, simply keep it as a weight.

#ifdef DEBUG_ALGO
	fprintf(stderr, "get_regression_Tree() initializing root node\n", n_called);
#endif
	// initialize first node
	QRF_Node root;

	root.split_feat = -1;
	root.is_leaf = 0;
	root.from_sample = 0;
	root.to_sample = (int)tree.sample_ids.size() - 1;
	// in regression trees we use the pred field to keep the average
	float avg = 0.0;
	for (j = root.from_sample; j <= root.to_sample; j++) {
		avg += yr[tree.sample_ids[j]];
	}
	avg = avg / (float)root.size();
	root.pred = avg;

	tree.nodes[0] = root;
	tree.n_nodes++;

	// a few additional preparations for easier/faster computation and no allocation while growing tree.
	for (i = 0; i < NFeat; i++)
		tree.feat_chosen[i] = -1;
	tree.histr_sum.resize(MaxQ);
	tree.histr_num.resize(MaxQ);
	tree.inds.resize(tree.sample_ids.size());
	tree.qy.resize(tree.sample_ids.size());

	// building the tree - going over the nodes, until the end
	int curr_node = 0;
	while (curr_node < tree.n_nodes) {

#ifdef DEBUG_ALGO
		fprintf(stderr, "regression tree build: curr_node = %d\n", curr_node); fflush(stderr);
#endif
		if (tree.nodes[curr_node].split_feat < 0 && !tree.nodes[curr_node].is_leaf) {

			// find the split
			if (tree_mode == QRF_REGRESSION_TREE)
				find_best_regression_split(tree, curr_node, ntry);
			else if (tree_mode == QRF_CATEGORICAL_CHI2_TREE)
				find_best_categories_chi2_split(tree, curr_node, ntry);
			else if (tree_mode == QRF_CATEGORICAL_ENTROPY_TREE)
				find_best_categories_entropy_split(tree, curr_node, ntry);
			else if (tree_mode == QRF_MULTILABEL_ENTROPY_TREE)
				find_best_categories_entropy_split_multilabel(tree, curr_node, ntry);

#ifdef DEBUG_ALGO
			fprintf(stderr, "regression tree build: before split curr_node = %d\n", curr_node); fflush(stderr);
#endif

			// make the split
			split_regression_node(tree, curr_node);
		}

		curr_node++;
	}
#ifdef DEBUG
	fprintf(stderr, "Built tree on %d samples and ntry %d with %d nodes \n", (int)tree.sample_ids.size(), ntry, tree.n_nodes); fflush(stderr);
#endif
	// free some unused memory
	tree.inds.clear();
	tree.qy.clear();
	tree.hist[0].clear();
	tree.hist[1].clear();
	tree.histr_sum.clear();
	tree.histr_num.clear();
	tree.feat_chosen.clear();

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::get_Tree(int *sampsize, int ntry, QRF_Tree &tree)
{
	int i, j;
	int local_sampsize[2] = { 0,0 };

	n_called++;

	vector<int> take_group;
	if (n_groups > 0) {
		take_group.resize(n_groups, 0);
		uniform_int_distribution<> dist(0, n_groups - 1);
		for (i = 0; i < n_groups; i++)
			take_group[dist(tree.rand_gen)] = 1;
	}

	if (sampsize != NULL) {
		//fprintf(stderr, "QRF: bagging with sampsize %d %d\n", sampsize[0], sampsize[1]);
		// first we bootstrap with sizes given to us in sampsize
		tree.sample_ids.resize(sampsize[0] + sampsize[1]);
		tree.max_nodes = 2 * (sampsize[0] + sampsize[1]);
		tree.nodes.resize(tree.max_nodes);
		tree.n_nodes = 0;
		int k = 0;
		if (n_groups == 0) {
			for (i = 1; i >= 0; i--) {
				uniform_int_distribution<> dist(0, (int)ids[i].size() - 1);
				for (j = 0; j < sampsize[i]; j++) {
					tree.sample_ids[k++] = ids[i][dist(tree.rand_gen)];
				}
			}
		}
		else {
			for (i = 1; i >= 0; i--) {
				uniform_int_distribution<> dist(0, (int)ids[i].size() - 1);
				int n = 0;
				while (n < sampsize[i]) {
					int id = ids[i][dist(tree.rand_gen)];
					if (take_group[groups[id]]) {
						tree.sample_ids[k++] = id;
						n++;
					}
				}
			}
		}
	}
	else {
		// in this case we simply randomize a bootstrap on all samples
		int n0 = (int)ids[0].size();
		int n1 = (int)ids[1].size();
		int n = n0 + n1;
		tree.sample_ids.resize(n);
		tree.max_nodes = 2 * n;
		tree.nodes.resize(tree.max_nodes);
		tree.n_nodes = 0;
		uniform_int_distribution<> dist(0, (int)n - 1);
		if (n_groups == 0) {
			for (j = 0; j < n; j++) {
				int r = dist(tree.rand_gen);
				if (r < n0) {
					tree.sample_ids[j] = ids[0][r];
					local_sampsize[0]++;
				}
				else {
					tree.sample_ids[j] = ids[1][r - n0];
					local_sampsize[1]++;
				}

			}
		}
		else {
			int j = 0, id = 0;
			while (j < n) {
				int r = dist(tree.rand_gen);
				if (r < n0)
					id = ids[0][r];
				else
					id = ids[1][r - n0];
				if (take_group[groups[id]]) {
					tree.sample_ids[j++] = id;
					if (r < n0)
						local_sampsize[0]++;
					else
						local_sampsize[1]++;
				}
			}

		}
		sampsize = local_sampsize;
	}

	// ToDo :: when a sample is given more than once, simply keep it as a weight.

	// initialize first node
	QRF_Node root;

	root.split_feat = -1;
	root.is_leaf = 0;
	for (i = 0; i < 2; i++)
		root.counts[i] = sampsize[i];
	root.from_sample = 0;
	root.to_sample = (int)tree.sample_ids.size() - 1;
	root.pred = (float)root.counts[1] / (float)(root.counts[1] + root.counts[0]);
	tree.nodes[0] = root;
	tree.n_nodes++;

	// a few additional preparations for easier/faster computation and no allocation while growing tree.
	for (i = 0; i < NFeat; i++)
		tree.feat_chosen[i] = -1;
	tree.hist[0].resize(MaxQ);
	tree.hist[1].resize(MaxQ);
	tree.inds.resize(tree.sample_ids.size());
	tree.qy.resize(tree.sample_ids.size());

#ifdef DEBUG
	fprintf(stderr, "starting to build tree sampsize: %d %d , nfeat %d\n", sampsize[0], sampsize[1], NFeat); fflush(stderr);
#endif
	// building the tree - going over the nodes, until the end
	int curr_node = 0;
	while (curr_node < tree.n_nodes) {
#ifdef DEBUG_ALGO
		fprintf(stderr, "regression tree build: curr_node = %d\n", curr_node); fflush(stderr);
#endif
		if (tree.nodes[curr_node].split_feat < 0 && !tree.nodes[curr_node].is_leaf) {

			// find the split
			find_best_split(tree, curr_node, ntry);
#ifdef DEBUG_ALGO
			fprintf(stderr, "regression tree build: before split curr_node = %d\n", curr_node); fflush(stderr);
#endif
			// make the split
			split_node(tree, curr_node);
#ifdef DEBUG_ALGO
			fprintf(stderr, "regression tree build: after split curr_node = %d\n", curr_node); fflush(stderr);
#endif
		}

		curr_node++;
	}
#ifdef DEBUG
	fprintf(stderr, "Built tree on %d samples and ntry %d with %d nodes \n", (int)tree.sample_ids.size(), ntry, tree.n_nodes); fflush(stderr);
#endif
	// free some unused memory
	tree.inds.clear();
	tree.qy.clear();
	tree.hist[0].clear();
	tree.hist[1].clear();
	tree.feat_chosen.clear();
	tree.histr_sum.clear();
	tree.histr_num.clear();

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::find_best_categories_chi2_split(QRF_Tree &tree, int node, int ntry)
{
	int try_num;
	int i, j;
	double left_sum, right_sum, tot_sum;
	double chi2s, expL, expR;

	int node_size = tree.nodes[node].size();
	QRF_Node *nd = &tree.nodes[node];

	double max_score;
	int best_feat = -1;
	int best_feat_i;
	int max_i;

	max_score = 0.0;

	uniform_int_distribution<> dist(0, NFeat - 1);

	tree.histr_num.resize(static_cast<size_t>(n_categ) * static_cast<size_t>(MaxQ));
	tree.histr_sum.resize(MaxQ);

	vector<int> histL(n_categ);
	vector<int> histR(n_categ);

	double epsilon = 1e-6;
	for (try_num = 0; try_num < ntry; try_num++) {

		int ifeat = dist(tree.rand_gen);

		if (tree.feat_chosen[ifeat] != node && max_q[ifeat] > 1) {

			tree.feat_chosen[ifeat] = node;

			if (1) { //node_size > (max_q[ifeat]>>4) && node_size > 16) { 

				// zero hist and then fill it back with values
				fill(tree.histr_num.begin(), tree.histr_num.begin() + (n_categ*max_q[ifeat]), 0);
				fill(tree.histr_sum.begin(), tree.histr_sum.begin() + max_q[ifeat], (float)0);

				short *pq_data = &q_data[ifeat][0];
				for (i = nd->from_sample; i <= nd->to_sample; i++) {
					//tree.histr_num[n_categ*q_data[ifeat][tree.sample_ids[i]] + (int)yr[tree.sample_ids[i]]]++;
					tree.histr_num[n_categ*pq_data[tree.sample_ids[i]] + (int)yr[tree.sample_ids[i]]]++;
					tree.histr_sum[q_data[ifeat][tree.sample_ids[i]]]++;
				}

				fill(histL.begin(), histL.end(), 0);
				fill(histR.begin(), histR.end(), 0);
				for (i = 0; i < max_q[ifeat]; i++)
					if (tree.histr_sum[i] > 0) {
						for (j = 0; j < n_categ; j++)
							histR[j] += tree.histr_num[i*n_categ + j];
					}

				// now we need to go over the hist and find the best split
				// we are searching to maximize the chi^2 score
				left_sum = 0.0;
				right_sum = (double)node_size;
				tot_sum = (double)node_size;
				max_i = -1;
				for (i = 0; i < max_q[ifeat] - 1; i++) {

					if (tree.histr_sum[i] > 0) {
						for (j = 0; j < n_categ; j++) {
							histL[j] += tree.histr_num[i*n_categ + j];
							histR[j] -= tree.histr_num[i*n_categ + j];
							left_sum += tree.histr_num[i*n_categ + j];
							right_sum -= tree.histr_num[i*n_categ + j];
						}

						if (right_sum == 0)
							break;

						chi2s = 0.0;
						for (j = 0; j < n_categ; j++) {
							expL = (double)(histL[j] + histR[j])*left_sum / (double)tot_sum;
							expR = (double)(histL[j] + histR[j])*right_sum / (double)tot_sum;
							if (expL > epsilon)
								chi2s += SQUARE(histL[j] - expL) / expL;
							if (expR > epsilon)
								chi2s += SQUARE(histR[j] - expR) / expR;
						}

						if (!(left_sum >= min_split_node_size && right_sum >= min_split_node_size))
							chi2s = max_score - 1;
#ifdef DEBUG_ALGO_2
						fprintf(stderr, "node %d ifeat %d i %d max_score %f node_size %d left_sum %f right_sum %f score %f\n",
							node, ifeat, i, max_score, node_size, left_sum, right_sum, chi2s); fflush(stderr);
#endif

						if (chi2s > max_score) {
							max_score = chi2s;
							max_i = i;
						}
					}

				}

#ifdef DEBUG_ALGO
				fprintf(stderr, "node %d ifeat %d max_q %d max_score %f score %f node_size %d\n", node, ifeat, max_q[ifeat], max_score, chi2s, node_size);
#endif
			}
			else {
				// use sort, as hist it too slow for low number of elements

				// ToDo:: write fast version for small nodes
			}

			if (max_i >= 0) {
				best_feat = ifeat;
				best_feat_i = max_i;
			}
		}

	}

	if (best_feat >= 0) {
		nd->split_feat = best_feat;
		nd->split_val = quant_values[best_feat][best_feat_i];
		nd->split_q_idx = best_feat_i;
	}
	else {
		nd->split_feat = -1;
	}

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::find_best_categories_entropy_split(QRF_Tree &tree, int node, int ntry)
{
	int try_num;
	int i, j;
	double left_sum, right_sum;
	double H, HL, HR;

	int node_size = tree.nodes[node].size();
	QRF_Node *nd = &tree.nodes[node];

	double min_score;
	int best_feat = -1;
	int best_feat_i;
	int max_i;

	min_score = 2.0*(double)NSamples*(double)NSamples*(double)n_categ;

	uniform_int_distribution<> dist(0, NFeat - 1);

	tree.histr_num.resize(static_cast<size_t>(n_categ) * static_cast<size_t>(MaxQ));
	tree.histr_sum.resize(MaxQ);

	vector<int> histL(n_categ);
	vector<int> histR(n_categ);
	vector<double> histL_w, histR_w;

	if (!w.empty()) {
		histL_w.resize(n_categ);
		histR_w.resize(n_categ);
	}

#ifdef DEBUG_ALGO
	fprintf(stderr, "Searching a new spilt: node = %d , min_score %f , Entropy minimizing\n", node, min_score); fflush(stderr);
#endif

	for (try_num = 0; try_num < ntry; try_num++) {

		int ifeat = dist(tree.rand_gen);

		if (tree.feat_chosen[ifeat] != node && max_q[ifeat] > 1) {

			tree.feat_chosen[ifeat] = node;

			if (1) { //node_size > (max_q[ifeat]>>4) && node_size > 16) { 
				vector<double> histr_num_w;
				double max_start = (double)node_size;
				if (!w.empty()) {
					max_start = 0;
					histr_num_w.resize(static_cast<size_t>(n_categ) * static_cast<size_t>(max_q[ifeat]));
					fill(tree.histr_sum.begin(), tree.histr_sum.begin() + max_q[ifeat], (float)0);
					fill(tree.histr_num.begin(), tree.histr_num.begin() + (n_categ*max_q[ifeat]), 0);
					for (i = nd->from_sample; i <= nd->to_sample; i++) {
						histr_num_w[n_categ*q_data[ifeat][tree.sample_ids[i]] + (int)yr[tree.sample_ids[i]]] += w[tree.sample_ids[i]];
						tree.histr_sum[q_data[ifeat][tree.sample_ids[i]]] += w[tree.sample_ids[i]];
						++tree.histr_num[n_categ*q_data[ifeat][tree.sample_ids[i]] + (int)yr[tree.sample_ids[i]]];
					}

					fill(histL_w.begin(), histL_w.end(), double(0));
					fill(histR_w.begin(), histR_w.end(), double(0));
					for (i = 0; i < max_q[ifeat]; i++)
						if (tree.histr_sum[i] > 0) {
							for (j = 0; j < n_categ; j++)
								histR_w[j] += histr_num_w[i*n_categ + j];

						}
					for (j = 0; j < n_categ; j++) //all sum
						max_start += histR_w[j];
				}
				else {
					// zero hist and then fill it back with values
					fill(tree.histr_num.begin(), tree.histr_num.begin() + (n_categ*max_q[ifeat]), 0);
					fill(tree.histr_sum.begin(), tree.histr_sum.begin() + max_q[ifeat], (float)0);
					for (i = nd->from_sample; i <= nd->to_sample; i++) {
						++tree.histr_num[n_categ*q_data[ifeat][tree.sample_ids[i]] + (int)yr[tree.sample_ids[i]]];
						++tree.histr_sum[q_data[ifeat][tree.sample_ids[i]]];
					}

					fill(histL.begin(), histL.end(), 0);
					fill(histR.begin(), histR.end(), 0);
					for (i = 0; i < max_q[ifeat]; i++)
						if (tree.histr_sum[i] > 0) {
							for (j = 0; j < n_categ; j++)
								histR[j] += tree.histr_num[i*n_categ + j];
						}
				}

				// now we need to go over the hist and find the best split
				// we are searching to maximize the H entropy score
				left_sum = 0.0;
				right_sum = max_start;
				int left_cnt = 0, right_cnt = node_size;
				max_i = -1;
				for (i = 0; i < max_q[ifeat] - 1; i++) {

					if (tree.histr_sum[i] > 0) {
						for (j = 0; j < n_categ; j++) {
							if (w.empty()) {
								histL[j] += tree.histr_num[i*n_categ + j];
								histR[j] -= tree.histr_num[i*n_categ + j];
								left_sum += tree.histr_num[i*n_categ + j];
								right_sum -= tree.histr_num[i*n_categ + j];

								if (right_sum == 0)
									break;
							}
							else {
								histL_w[j] += histr_num_w[i*n_categ + j];
								histR_w[j] -= histr_num_w[i*n_categ + j];
								left_sum += histr_num_w[i*n_categ + j];
								right_sum -= histr_num_w[i*n_categ + j];
								left_cnt += tree.histr_num[i*n_categ + j];
								right_cnt -= tree.histr_num[i*n_categ + j];

								if (right_cnt == 0)
									break;
							}
						}
						HL = 0.0;
						HR = 0.0;
						for (j = 0; j < n_categ; j++) {
							if (w.empty()) {
								HL -= log_table[histL[j]];
								HR -= log_table[histR[j]];
							}
							else {
								int ind_L = (int)round(histL_w[j]);
								int ind_R = (int)round(histR_w[j]);
								if (ind_L < log_table.size())
									HL -= log_table[(int)round(histL_w[j])];
								else
									HL -= log(histL_w[j]);
								if (ind_R < log_table.size())
									HR -= log_table[(int)round(histR_w[j])];
								else
									HR -= log(histR_w[j]);
							}
						}

						if ((int)left_sum < log_table.size())
							HL += log_table[(int)left_sum];
						else
							HL += left_sum * log(left_sum);
						if ((int)right_sum < log_table.size())
							HR += log_table[(int)right_sum];
						else
							HR += right_sum * log(right_sum);


						H = HL + HR;

						if (!(left_sum >= min_split_node_size && right_sum >= min_split_node_size))
							H = min_score + 1; // makes sure we do not split cases that split to nodes too small
#ifdef DEBUG_ALGO_2
						fprintf(stderr, "node %d ifeat %d i %d min_score %f node_size %d left_sum %f right_sum %f score %f\n",
							node, ifeat, i, min_score, node_size, left_sum, right_sum, H); fflush(stderr);
#endif

						if (H < min_score) {
							min_score = H;
							max_i = i;
						}
					}

				}

#ifdef DEBUG_ALGO
				fprintf(stderr, "node %d ifeat %d max_q %d min_score %f score %f node_size %d max_i %d best_feat %d best_feat_i %d\n", node, ifeat, max_q[ifeat], min_score, H, node_size, max_i, best_feat, best_feat_i);
#endif
			}
			else {
				// use sort, as hist it too slow for low number of elements

				// ToDo: write fast version for small node sizes
			}

			if (max_i >= 0) {
				best_feat = ifeat;
				best_feat_i = max_i;
			}
		}

	}

	if (best_feat >= 0) {
		nd->split_feat = best_feat;
		nd->split_val = quant_values[best_feat][best_feat_i];
		nd->split_q_idx = best_feat_i;
	}
	else {
		nd->split_feat = -1;
	}

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::find_best_categories_entropy_split_multilabel(QRF_Tree &tree, int node, int ntry)
{
	int try_num;
	int i, j;
	double left_sum, right_sum;
	double H, HL, HR;

	int node_size = tree.nodes[node].size();
	QRF_Node *nd = &tree.nodes[node];

	double min_score;
	int best_feat = -1;
	int best_feat_i;
	int max_i;

	int log_n_categ = (int)log2(n_categ);

	min_score = 2.0*(double)NSamples*(double)NSamples*(double)log_n_categ;

	uniform_int_distribution<> dist(0, NFeat - 1);
	
	tree.histr_num.resize(static_cast<size_t>(log_n_categ) * static_cast<size_t>(MaxQ));
	tree.histr_sum.resize(MaxQ);

	vector<int> histL(log_n_categ);
	vector<int> histR(log_n_categ);
	vector<double> histL_w, histR_w;

	if (!w.empty()) {
		histL_w.resize(n_categ);
		histR_w.resize(n_categ);
	}

#ifdef DEBUG_ALGO
	fprintf(stderr, "Searching a new spilt: node = %d ,node_size: %d min_score %f , Entropy minimizing\n", node, node_size, min_score); fflush(stderr);
#endif

	for (try_num = 0; try_num < ntry; try_num++) {

		int ifeat = dist(tree.rand_gen);

		if (tree.feat_chosen[ifeat] != node && max_q[ifeat] > 1) {

			tree.feat_chosen[ifeat] = node;

			if (1) { //node_size > (max_q[ifeat]>>4) && node_size > 16) { 
				vector<double> histr_num_w;
				double max_start = (double)node_size;
				if (!w.empty()) {
					max_start = 0;
					histr_num_w.resize(static_cast<size_t>(n_categ) * static_cast<size_t>(max_q[ifeat]));
					fill(tree.histr_sum.begin(), tree.histr_sum.begin() + max_q[ifeat], (float)0);
					fill(tree.histr_num.begin(), tree.histr_num.begin() + (n_categ*max_q[ifeat]), 0);
					for (i = nd->from_sample; i <= nd->to_sample; i++) {
						histr_num_w[n_categ*q_data[ifeat][tree.sample_ids[i]] + (int)yr[tree.sample_ids[i]]] += w[tree.sample_ids[i]];
						tree.histr_sum[q_data[ifeat][tree.sample_ids[i]]] += w[tree.sample_ids[i]];
						++tree.histr_num[n_categ*q_data[ifeat][tree.sample_ids[i]] + (int)yr[tree.sample_ids[i]]];
					}

					fill(histL_w.begin(), histL_w.end(), double(0));
					fill(histR_w.begin(), histR_w.end(), double(0));
					for (i = 0; i < max_q[ifeat]; i++)
						if (tree.histr_sum[i] > 0) {
							for (j = 0; j < n_categ; j++)
								histR_w[j] += histr_num_w[i*n_categ + j];

						}
					for (j = 0; j < n_categ; j++) //all sum
						max_start += histR_w[j];
				}
				else {
					// zero hist and then fill it back with values
					fill(tree.histr_num.begin(), tree.histr_num.begin() + (log_n_categ*max_q[ifeat]), 0);
					fill(tree.histr_sum.begin(), tree.histr_sum.begin() + max_q[ifeat], (double)0);


					for (i = nd->from_sample; i <= nd->to_sample; i++) {
						for (int j = 0; j < yr_multilabel[tree.sample_ids[i]].size(); j++)
						{
							++tree.histr_num[log_n_categ*q_data[ifeat][tree.sample_ids[i]] + yr_multilabel[tree.sample_ids[i]][j]];
						}
					

						++tree.histr_sum[q_data[ifeat][tree.sample_ids[i]]];

					}
 

					fill(histL.begin(), histL.end(), 0);
					fill(histR.begin(), histR.end(), 0);
					for (i = 0; i < max_q[ifeat]; i++)
						if (tree.histr_sum[i] > 0) {
							for (j = 0; j < log_n_categ; j++)
								histR[j] += tree.histr_num[i*log_n_categ + j];
						}
				}

				// now we need to go over the hist and find the best split
				// we are searching to maximize the H entropy score
				left_sum = 0.0;
				right_sum = max_start;
				int left_cnt = 0, right_cnt = node_size;
				max_i = -1;
				for (i = 0; i < max_q[ifeat] - 1; i++) {

					if (tree.histr_sum[i] > 0) {
						left_sum += tree.histr_sum[i];
						right_sum -= tree.histr_sum[i];
						for (j = 0; j < log_n_categ; j++) {
							if (w.empty()) {
								histL[j] += tree.histr_num[i*log_n_categ + j];
								histR[j] -= tree.histr_num[i*log_n_categ + j];
							}
							else {
								histL_w[j] += histr_num_w[i*n_categ + j];
								histR_w[j] -= histr_num_w[i*n_categ + j];
								left_sum += histr_num_w[i*n_categ + j];
								right_sum -= histr_num_w[i*n_categ + j];
								left_cnt += tree.histr_num[i*n_categ + j];
								right_cnt -= tree.histr_num[i*n_categ + j];

								if (right_cnt == 0)
									break;
							}
						}
						HL = 0.0;
						HR = 0.0;
						for (j = 0; j < log_n_categ; j++) {
							if (w.empty()) {
								
								HL -= (log_table[histL[j]] + log_table[(int)(left_sum - histL[j])]);
								HR -= (log_table[histR[j]] + log_table[(int)(right_sum - histR[j])]);
								//MLOG("Right: j: HR: %f, %d, log_table[histR[j]]: %f, log_table[(int)righ_sum - histR[j]] %f sum: %f \n", j, HR, log_table[histR[j]], log_table[(int)right_sum - histR[j]], (log_table[histR[j]] + log_table[(int)right_sum - histR[j]]));
							}
							else {
								int ind_L = (int)round(histL_w[j]);
								int ind_R = (int)round(histR_w[j]);
								if (ind_L < log_table.size())
									HL -= log_table[(int)round(histL_w[j])];
								else
									HL -= histL_w[j] * log(histL_w[j]);
								if (ind_R < log_table.size())
									HR -= log_table[(int)round(histR_w[j])];
								else
									HR -= log(histR_w[j]);
							}
						}
						
						if ((int)left_sum < log_table.size())
							HL += log_n_categ*log_table[(int)left_sum];
						else
							HL += log_n_categ*left_sum * log(left_sum);
						if ((int)right_sum < log_table.size())
							HR += log_n_categ*log_table[(int)right_sum];
						else
							HR += log_n_categ*right_sum * log(right_sum);

						H = HL + HR;

						if (!(left_sum >= min_split_node_size && right_sum >= min_split_node_size))
							H = min_score + 1; // makes sure we do not split cases that split to nodes too small
#ifdef DEBUG_ALGO_2
						fprintf(stderr, "node %d ifeat %d i %d min_score %f node_size %d left_sum %f right_sum %f score %f max_i %d\n",
							node, ifeat, i, min_score, node_size, left_sum, right_sum, H, max_i); fflush(stderr);
#endif

						if (H < min_score) {
							min_score = H;
							max_i = i;
						}
					}

				}

#ifdef DEBUG_ALGO
				fprintf(stderr, "node %d ifeat %d max_q %d min_score %f score %f node_size %d max_i %d best_feat %d best_feat_i %d  \n", node, ifeat, max_q[ifeat], min_score, H, node_size, max_i, best_feat, best_feat_i);
#endif
			}
			else {
				// use sort, as hist it too slow for low number of elements

				// ToDo: write fast version for small node sizes
			}

			if (max_i >= 0) {
				best_feat = ifeat;
				best_feat_i = max_i;
			}
		}

	}

	if (best_feat >= 0) {
		nd->split_feat = best_feat;
		nd->split_val = quant_values[best_feat][best_feat_i];
		nd->split_q_idx = best_feat_i;
	}
	else {
		nd->split_feat = -1;
	}

	return 0;
}
//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::find_best_regression_split(QRF_Tree &tree, int node, int ntry)
{
	int try_num;
	int i, j;
	double left_sum, right_sum;
	double left_num, right_num;
	double left_w, right_w;
	double left_avg, right_avg;

	int node_size = tree.nodes[node].size();
	QRF_Node *nd = &tree.nodes[node];

	double score, max_score;
	int best_feat = -1;
	int best_feat_i;
	int max_i;

	max_score = (float)node_size*(nd->pred)*(nd->pred)*REGRESSION_SPLIT_IMPROVE;

	uniform_int_distribution<> dist(0, NFeat - 1);

	for (try_num = 0; try_num < ntry; try_num++) {

		int ifeat = dist(tree.rand_gen);

		if (tree.feat_chosen[ifeat] != node && max_q[ifeat] > 1) {

			tree.feat_chosen[ifeat] = node;
			vector<float> hist_w;
			if (!w.empty())
				hist_w.resize(max_q[ifeat]);

			if (node_size > (max_q[ifeat] >> 4) && node_size > 16) {

				// zero hist and then fill it back with values
				for (i = 0; i < max_q[ifeat]; i++) {
					tree.histr_sum[i] = 0.0;
					tree.histr_num[i] = 0;
				}

				for (i = nd->from_sample; i <= nd->to_sample; i++) {
					tree.histr_sum[q_data[ifeat][tree.sample_ids[i]]] += yr[tree.sample_ids[i]];
					tree.histr_num[q_data[ifeat][tree.sample_ids[i]]]++;
					if (!w.empty())
						hist_w[q_data[ifeat][tree.sample_ids[i]]] += w[tree.sample_ids[i]];
				}

				// now we need to go over the hist and find the best split
				// we are searching to maximize : L*(avg_L)^2 + R*(avg_R)^2
				left_sum = 0.0;
				left_num = 0.0;
				left_w = 0.0;
				right_w = 0.0;
				right_sum = static_cast<double>(node_size) * static_cast<double>(nd->pred);
				right_num = (float)node_size;

				max_i = -1;
				for (i = 0; i < max_q[ifeat] - 1; i++) {
					if (tree.histr_num[i] > 0) {
						left_sum += tree.histr_sum[i];
						left_num += tree.histr_num[i];
						right_sum -= tree.histr_sum[i];
						right_num -= tree.histr_num[i];
						if (!w.empty()) {
							left_w += hist_w[i];
							right_w -= hist_w[i];
						}

						if (right_num == 0)
							break;

						if (!w.empty()) {
							left_avg = left_sum / left_w;
							right_avg = right_sum / right_w;
							score = left_w * left_avg*left_avg + right_w * right_avg*right_avg;
						}
						else {
							left_avg = left_sum / left_num;
							right_avg = right_sum / right_num;
							score = left_num * left_avg*left_avg + right_num * right_avg*right_avg;
						}

						if (!(left_num >= min_split_node_size && right_num >= min_split_node_size))
							score = max_score - 1; // don't split when getting to too small nodes
#ifdef DEBUG_ALGO_2
						fprintf(stderr, "node %d ifeat %d i %d max_score %f node_size %d left_sum %f left_num %f right_sum %f right_num %f right_avg %f left_avg %f score %f\n",
							node, ifeat, i, max_score, node_size, left_sum, left_num, right_sum, right_num, right_avg, left_avg, score); fflush(stderr);
#endif

						if (score > max_score) {
							max_score = score;
							max_i = i;
						}
					}

				}

#ifdef DEBUG_ALGO
				fprintf(stderr, "node %d ifeat %d max_q %d max_score %f score %f node_size %d\n", node, ifeat, max_q[ifeat], max_score, score, node_size);
#endif
			}
			else {
				// use sort, as hist it too slow for low number of elements
				j = 0;
				left_w = 0.0;
				right_w = 0.0;
				for (i = nd->from_sample; i <= nd->to_sample; i++) {
					// note we swapped the idx and val roles here compared with the binary tree version
					tree.qy[j].idx = q_data[ifeat][tree.sample_ids[i]];
					tree.qy[j].val = yr[tree.sample_ids[i]];
					if (!w.empty())
						right_w += w[tree.sample_ids[i]];
					j++;
				}
				sort(tree.qy.begin(), tree.qy.begin() + node_size, [](const ValInd &v1, const ValInd &v2) {return v1.idx < v2.idx; });

				left_sum = 0.0;
				left_num = 0.0;

				right_sum = static_cast<double>(node_size) * static_cast<double>(nd->pred);
				right_num = (float)node_size;
				max_i = -1;

				for (i = 0; i < node_size - 1; i++) {
					left_sum += tree.qy[i].val;
					left_num++;
					if (!w.empty())
						left_w += w[tree.sample_ids[nd->from_sample + i]];
					right_sum -= tree.qy[i].val;
					right_num--;
					if (!w.empty())
						right_w -= w[tree.sample_ids[nd->from_sample + i]];

					if (tree.qy[i].idx < tree.qy[i + 1].idx) {
						if (w.empty()) {
							left_avg = left_sum / left_num;
							right_avg = right_sum / right_num;
							score = left_num * left_avg*left_avg + right_num * right_avg*right_avg;
						}
						else {
							left_avg = left_sum / left_w;
							right_avg = right_sum / right_w;
							score = left_w * left_avg*left_avg + right_w * right_avg*right_avg;
						}

						if (!(left_num >= min_split_node_size && right_num >= min_split_node_size))
							score = max_score - 1; // don't split when getting to too small nodes

						if (score > max_score) {
							max_score = score;
							max_i = tree.qy[i].idx;
						}
					}

				}
			}

			if (max_i >= 0) {
				best_feat = ifeat;
				best_feat_i = max_i;
			}
		}

	}

	if (best_feat >= 0) {
		nd->split_feat = best_feat;
		nd->split_val = quant_values[best_feat][best_feat_i];
		nd->split_q_idx = best_feat_i;
	}
	else {
		nd->split_feat = -1;
	}

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::find_best_split(QRF_Tree &tree, int node, int ntry)
{
	int try_num;
	int i, j;
	double left_d, right_d;
	double left_n0, left_n1;
	double right_n0, right_n1;
	double tot;

	int node_size = tree.nodes[node].size();
	QRF_Node *nd = &tree.nodes[node];

	double score, min_score = 2.0;
	int min_feat = -1;
	int min_feat_i;
	int min_i;
	//int s;

	uniform_int_distribution<> dist(0, NFeat - 1);

	for (try_num = 0; try_num < ntry; try_num++) {

		int ifeat = dist(tree.rand_gen);

		if (tree.feat_chosen[ifeat] != node && max_q[ifeat] > 1) {

			tree.feat_chosen[ifeat] = node;

			if (node_size > (max_q[ifeat] >> 4) && node_size > 16) {

				// zero hist and then fill it back with values
				for (i = 0; i < max_q[ifeat]; i++) {
					tree.hist[0][i] = 0;
					tree.hist[1][i] = 0;
				}

				for (i = nd->from_sample; i <= nd->to_sample; i++)
					tree.hist[(int)y[tree.sample_ids[i]]][q_data[ifeat][tree.sample_ids[i]]]++;

				// now we need to go over the hist and find the best split
				left_d = 0; right_d = node_size;
				left_n0 = 0;  left_n1 = 0;
				right_n0 = nd->counts[0];
				right_n1 = nd->counts[1];
				tot = node_size;

				min_i = -1;
				for (i = 0; i < max_q[ifeat] - 1; i++) {
					if (tree.hist[0][i] > 0 || tree.hist[1][i] > 0) {
						left_n0 += tree.hist[0][i];
						left_n1 += tree.hist[1][i];
						left_d += tree.hist[0][i] + tree.hist[1][i];
						right_n0 -= tree.hist[0][i];
						right_n1 -= tree.hist[1][i];
						right_d -= (tree.hist[0][i] + tree.hist[1][i]);

						if (right_d > 0 && left_d > 0 && right_d >= min_split_node_size && left_d >= min_split_node_size) {
							score = (left_n0 / tot) * (left_n1 / left_d) + (right_n0 / tot)*(right_n1 / right_d);
							if (score < min_score) {
								min_score = score;
								min_i = i;
							}
							else if (score == min_score && (tree.rand_gen() & 0x1))
								min_i = i;
						}
					}

				}
			}
			else {
				// use sort, as hist it too slow for low number of elements
				j = 0;
				for (i = nd->from_sample; i <= nd->to_sample; i++) {
					tree.qy[j].val = (float)q_data[ifeat][tree.sample_ids[i]];
					tree.qy[j].idx = y[tree.sample_ids[i]];
					j++;
				}
				sort(tree.qy.begin(), tree.qy.begin() + node_size, [](const ValInd &v1, const ValInd &v2) {return v1.val < v2.val; });

				left_d = 0; right_d = node_size;
				left_n0 = 0;  left_n1 = 0;
				right_n0 = nd->counts[0];
				right_n1 = nd->counts[1];
				tot = node_size;
				min_i = -1;
				for (i = 0; i < node_size - 1; i++) {
					left_n1 += tree.qy[i].idx;
					left_n0 += (1 - tree.qy[i].idx);
					left_d++;
					right_n1 -= tree.qy[i].idx;
					right_n0 -= (1 - tree.qy[i].idx);
					right_d--;
					if (tree.qy[i].val < tree.qy[i + 1].val && right_d>0 && left_d > 0 && right_d >= min_split_node_size && left_d >= min_split_node_size) {
						score = (left_n0 / tot) * (left_n1 / left_d) + (right_n0 / tot)*(right_n1 / right_d);
						if (score < min_score) {
							min_score = score;
							min_i = (int)tree.qy[i].val;
						}
						else if (score == min_score && (tree.rand_gen() & 0x1))
							min_i = (int)tree.qy[i].val;
					}
				}

			}

			if (min_i >= 0) {
				min_feat = ifeat;
				min_feat_i = min_i;
			}
		}

	}

	if (min_feat >= 0) {
		nd->split_feat = min_feat;
		nd->split_val = quant_values[min_feat][min_feat_i];
		nd->split_q_idx = min_feat_i;
	}
	else {
		nd->split_feat = -1;
	}

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::split_regression_node(QRF_Tree &tree, int node)
{
	QRF_Node *nd = &tree.nodes[node];

	if (nd->split_feat < 0 || (max_depth > 0 && nd->depth >= max_depth)) {
		// this case means that although we searched lots of options none splut the node...
		// this happens when the SAME value elements are in the same node.
		// we declare such nodes as leaves.
		nd->is_leaf = 1;
#ifdef DEBUG_ALGO
		fprintf(stderr, "split: node %d split feat is -1, hence a leaf\n", node); fflush(stderr);
#endif
		return 0;
	}


	// we split by the chosen criteria
	// we need to change the order of indexes in sample_ids in the node range
	// and then we can split to the left and right son nodes.

	int i;
	int j = nd->from_sample;
	int k = nd->to_sample;
	int ifeat = nd->split_feat;
	int qval = nd->split_q_idx;
	//MLOG("node: %d, j: %d, k: %d, ifeat: %d, qval %d \n", node, j, k, ifeat, qval);
	float  sumL = 0.0, minL = 1e10, maxL = -1e10;
	float  sumR = 0.0, minR = 1e10, maxR = -1e10;
	for (i = nd->from_sample; i <= nd->to_sample; i++) {
		if (q_data[ifeat][tree.sample_ids[i]] <= qval) {
			tree.sample_ids[j++] = tree.sample_ids[i];
			sumL += (w.empty() ? 1 : w[tree.sample_ids[i]]) * yr[tree.sample_ids[i]];
			if (yr[tree.sample_ids[i]] > maxL) maxL = yr[tree.sample_ids[i]];
			if (yr[tree.sample_ids[i]] < minL) minL = yr[tree.sample_ids[i]];
		}
		else {
			tree.inds[k--] = tree.sample_ids[i];
		}
	}

	for (i = k + 1; i <= nd->to_sample; i++) {
		tree.sample_ids[i] = tree.inds[i];
		sumR += (w.empty() ? 1 : w[tree.sample_ids[i]]) * yr[tree.inds[i]];
		if (yr[tree.inds[i]] > maxR) maxR = yr[tree.inds[i]];
		if (yr[tree.inds[i]] < minR) minR = yr[tree.inds[i]];
	}


	QRF_Node *Left = &tree.nodes[tree.n_nodes];
	QRF_Node *Right = &tree.nodes[tree.n_nodes + 1];

	double spread_test;

	Left->depth = nd->depth + 1;
	Left->split_feat = -1;
	Left->from_sample = nd->from_sample;
	Left->to_sample = k;
	Left->pred = sumL / (float)(Left->size());
	spread_test = abs(maxL - minL);
	if (Left->size() < min_split_node_size || spread_test < min_split_spread)
		Left->is_leaf = 1;
	else
		Left->is_leaf = 0;

	Right->depth = nd->depth + 1;
	Right->split_feat = -1;
	Right->from_sample = k + 1;
	Right->to_sample = nd->to_sample;
	//MLOG("Right->depth %d, Right->split_feat %d, Right->from_sample %d, Right->to_sample  %d \n ", Right->depth, Right->split_feat, Right->from_sample, Right->to_sample);
	//MLOG("Left->depth %d, Left->split_feat %d, Left->from_sample %d, Left->to_sample  %d \n ", Left->depth, Left->split_feat, Left->from_sample, Left->to_sample);
	Right->pred = sumR / (float)(Right->size());
	spread_test = abs(maxR - minR);
	if (Right->size() < min_split_node_size || spread_test < min_split_spread)
		Right->is_leaf = 1;
	else
		Right->is_leaf = 0;

#ifdef DEBUG_ALGO
	fprintf(stderr, "split: node %d left size %d pred %f , right size %d pred %f\n", node, Left->size(), Left->pred, Right->size(), Right->pred); fflush(stderr);
#endif

	nd->l_ind = (int)tree.n_nodes;
	nd->r_ind = (int)tree.n_nodes + 1;

	tree.n_nodes += 2;

	return(0);
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::split_node(QRF_Tree &tree, int node)
{
	QRF_Node *nd = &tree.nodes[node];

	if (nd->split_feat < 0) {
		// this case means that although we searched lots of options none splut the node...
		// this happens when the SAME value elements are in the same node.
		// we declare such nodes as leaves.
		nd->is_leaf = 1;
		return 0;
	}

	// we split by the chosen criteria
	// we need to change the order of indexes in sample_ids in the node range
	// and then we can split to the left and right son nodes.

	int i;
	int j = nd->from_sample;
	int k = nd->to_sample;
	int ifeat = nd->split_feat;
	int qval = nd->split_q_idx;
	int nL = 0;
	for (i = nd->from_sample; i <= nd->to_sample; i++) {
		if (q_data[ifeat][tree.sample_ids[i]] <= qval) {
			tree.sample_ids[j++] = tree.sample_ids[i];
			nL += y[tree.sample_ids[i]];
		}
		else {
			tree.inds[k--] = tree.sample_ids[i];
		}
	}

	for (i = k + 1; i <= nd->to_sample; i++) {
		tree.sample_ids[i] = tree.inds[i];
	}

	QRF_Node *Left = &tree.nodes[tree.n_nodes];
	QRF_Node *Right = &tree.nodes[tree.n_nodes + 1];

	Left->split_feat = -1;
	Left->from_sample = nd->from_sample;
	Left->to_sample = k;
	Left->counts[1] = nL;
	Left->counts[0] = Left->size() - Left->counts[1];
	Left->pred = (float)(Left->counts[1]) / (float)(Left->size());
	if (Left->counts[0] == 0 || Left->counts[1] == 0)
		Left->is_leaf = 1;
	else
		Left->is_leaf = 0;

	Right->split_feat = -1;
	Right->from_sample = k + 1;
	Right->to_sample = nd->to_sample;
	Right->counts[1] = nd->counts[1] - nL;
	Right->counts[0] = Right->size() - Right->counts[1];
	Right->pred = (float)(Right->counts[1]) / (float)(Right->size());
	if (Right->counts[0] == 0 || Right->counts[1] == 0)
		Right->is_leaf = 1;
	else
		Right->is_leaf = 0;

	nd->l_ind = (int)tree.n_nodes;
	nd->r_ind = (int)tree.n_nodes + 1;

	tree.n_nodes += 2;

	return(0);
}

//-----------------------------------------------------------------------------------------------------------------------------------
void QuantizedRF::score_tree_by_index(float *x, int nfeat, QRF_ResTree &tree, int id, float& score, int& majority, vector<int> &counts)
{
	int node = 0;

	while (!tree.qnodes[node].is_leaf) {
		if (x[(size_t)nfeat*(size_t)id + (size_t)tree.qnodes[node].ifeat] <= tree.qnodes[node].split_val)
			//		if (q_data[tree.qnodes[node].ifeat][id] <= tree.qnodes[node].split_val)
			node = tree.qnodes[node].left;
		else
			node = tree.qnodes[node].right;
	}


	score = tree.qnodes[node].pred;
	majority = tree.qnodes[node].majority;
	counts = tree.qnodes[node].counts;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QuantizedRF::collect_Tree_oob_scores(float *x, int nfeat, QRF_ResTree &tree, vector<int> &sample_ids)
{

	if (cv.size() == 0) {
		cv.resize(NSamples);
		for (int i = 0; i < NSamples; i++) {
			cv[i].mean = 0;
			cv[i].std = 0;
			cv[i].n_tests = 0;
			cv[i].sum = 0;
			cv[i].cnts.resize(n_categ, 0);
		}
	}


	int i, j;

	test_s.clear();
	test_s.resize(NSamples, 0);
	if (n_groups == 0) {
		for (i = 0; i < sample_ids.size(); i++)
			test_s[sample_ids[i]] = 1;
	}
	else {
		//fprintf(stderr, "QRF DEBUG:: Nsamples %d , n_groups %d , groups size() %d , sample_ids size() %d\n", NSamples, n_groups, groups.size(), sample_ids.size());
		vector<int> group_used(n_groups, 0);
		for (i = 0; i < sample_ids.size(); i++)
			group_used[groups[sample_ids[i]]] = 1;
		//fprintf(stderr, "QRF DEBUG:: Nsamples %d , n_groups %d\n", NSamples, n_groups);
		int n = 0;
		for (i = 0; i < sample_ids.size(); i++)
			if (group_used[groups[sample_ids[i]]]) {
				test_s[sample_ids[i]] = 1;
				n++;
			}
		//fprintf(stderr, "QRF DEBUG:: Nsamples %d , n chosen %d\n", NSamples, n);
	}

#pragma omp parallel for
	for (i = 0; i < NSamples; i++) {
		if (test_s[i] == 0) {

			float score;
			int majority;

			vector<int> cnts;
			score_tree_by_index(x, nfeat, tree, i, score, majority, cnts);
			cv[i].n_tests++;

			// Collect scores according to mode
			if (tree_mode == QRF_REGRESSION_TREE) {
				// Use default behavoiur (PREDS_REGRESSION_AVG)
				cv[i].sum += score;
				cv[i].std += score * score;
			}
			else {
				// Use defauly behaviour (PROBS_CATEG_MAJORITY_AVG)
//				cv[i].cnts[majority]++ ;
				for (j = 0; j < cnts.size(); j++)
					cv[i].cnts[j] += cnts[j];
			}
		}
	}

	return 0;
}


//-----------------------------------------------------------------------------------------------------------------------------------
void QuantizedRF::complete_oob_cv()
{
	int n = 1;
	if (n_categ > 0) n = n_categ;
	float def_val = (float)1 / (float)n;
	vector<float> def_prob(n, def_val);

#pragma omp parallel for
	for (int i = 0; i < cv.size(); i++) {
		if (cv[i].n_tests > 0) {
			if (tree_mode == QRF_REGRESSION_TREE) {
				float m = cv[i].sum / (float)cv[i].n_tests;
				float s = cv[i].std / (float)cv[i].n_tests;
				s = sqrt(max((float)0, s - m * m));
				cv[i].mean = m;
				cv[i].std = s;
			}
			else {
				for (int k = 0; k < n_categ; k++)
					cv[i].probs.push_back(((float)cv[i].cnts[k]) / cv[i].n_tests);
			}
		}
		else {
			cv[i].std = -1.0;
			cv[i].probs = def_prob;
		}
	}
}

//-----------------------------------------------------------------------------------------------------------------------------------
double QuantizedRF::get_cross_validation_auc()
{
	int i;

	vector<float> scores(NSamples);
	for (i = 0; i < NSamples; i++)
		scores[i] = cv[i].probs[1];

	double auc = Get_AUC(scores, y);

	return (auc);

}

//-----------------------------------------------------------------------------------------------------------------------------------
void QuantizedRF::init_groups(vector<int> &groups_in)
{
	//printf(stderr, "In init_groups groups_in.size() %d , n_groups %d\n", groups_in.size(), n_groups);
	groups.clear();
	n_groups = 0;
	//fprintf(stderr, "In init_groups groups_in.size() %d , n_groups %d\n", groups_in.size(), n_groups);

	if (groups_in.size() == 0)
		return;

	// data is split into groups.
	// We move the groups range to 0:ngroups-1 to get some efficiency

	map<int, int> group_id; // from id of a sample to 0:n_groups-1

	groups.resize(groups_in.size());
	for (int i = 0; i < groups_in.size(); i++) {
		if (group_id.find(groups_in[i]) == group_id.end()) {
			group_id[groups_in[i]] = n_groups;
			n_groups++;
		}
		groups[i] = group_id[groups_in[i]];
	}
	//fprintf(stderr, "In init_groups groups_in.size() %d (NSamples %d), n_groups %d\n", groups_in.size(), NSamples, n_groups);

}

//======================================================================================================================================
// QRF_Forest
//======================================================================================================================================

//-----------------------------------------------------------------------------------------------------------------------------------
int QRF_Forest::transfer_tree_to_res_tree(QuantizedRF &qrf, QRF_Tree &tree, QRF_ResTree &qt, int mode, map<float, int> &all_values)
{
	qt.qnodes.clear();
	qt.qnodes.resize(tree.n_nodes);
	for (int j = 0; j < tree.n_nodes; j++) {
		QRF_ResNode &qn = qt.qnodes[j];
		qn.mode = mode;
		qn.ifeat = tree.nodes[j].split_feat;
		qn.split_val = tree.nodes[j].split_val;
		qn.is_leaf = tree.nodes[j].is_leaf;
		qn.right = tree.nodes[j].r_ind;
		qn.left = tree.nodes[j].l_ind;
		qn.pred = tree.nodes[j].pred;
		qn.n_size = tree.nodes[j].size();
		if (mode != QRF_REGRESSION_TREE) {
			if (tree.nodes[j].is_leaf) {
			//if (1) {
				qn.counts.resize(n_categ);
				fill(qn.counts.begin(), qn.counts.end(), 0);
				for (int k = tree.nodes[j].from_sample; k <= tree.nodes[j].to_sample; k++) {
					if (mode == QRF_BINARY_TREE)
						qn.counts[(int)(qrf.y[tree.sample_ids[k]])]++;
					else
						qn.counts[(int)(qrf.yr[tree.sample_ids[k]])]++;
				}
				int maxv = 0, maxi = -1;
				for (int j = 0; j < n_categ; j++) {
					if (qn.counts[j] > maxv) {
						maxv = qn.counts[j];
						maxi = j;
					}
				}
				qn.majority = maxi;
				if (maxi < 0) {
					fprintf(stderr, "BUG in getting majority !\n"); fflush(stderr);
				}
			}
		}
		else if (keep_all_values && qn.is_leaf) {
			qn.tot_n_values = tree.nodes[j].to_sample + 1 - tree.nodes[j].from_sample;
			qn.values.assign(sorted_values.size(), 0);
			for (int k = 0; k < qn.tot_n_values; k++)
				qn.values[all_values[qrf.yr[tree.sample_ids[k + tree.nodes[j].from_sample]]]]++;

			// Rearrange for sparse-values mode
			if (sparse_values) {
				qn.value_counts.clear();
				for (unsigned int iVal = 0; iVal < qn.values.size(); iVal++) {
					if (qn.values[iVal] > 0) {
						//							if (qn.values[iVal] > 0xffff)
						//								MTHROW_AND_ERR("Cannot work in sparse-mode. Reached count > %d\n", 0xffff);
						//							qn.value_counts.push_back({ iVal,  (unsigned short int) qn.values[iVal] });
						qn.value_counts.push_back({ iVal,   qn.values[iVal] });
					}
				}
				qn.values.clear();
			}
			else
				qn.value_counts.clear();
		}
		else
			qn.values.clear();
	}
	return 0;
}


//-----------------------------------------------------------------------------------------------------------------------------------
int QRF_Forest::init_keep_all_values(QuantizedRF &qrf, int mode, map<float, int> &all_values)
{
	// Get ordered values
	if (keep_all_values) {
		// Collect
		for (float y : qrf.yr)
			all_values[y] = 1;

		// Sort
		sorted_values.clear();
		for (auto& value : all_values)
			sorted_values.push_back(value.first);
		sort(sorted_values.begin(), sorted_values.end());

		// Get Positions
		for (unsigned int i = 0; i < sorted_values.size(); i++)
			all_values[sorted_values[i]] = i;

	}
	else
		sorted_values.clear();

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QRF_Forest::transfer_to_forest(vector<QRF_Tree> &trees, QuantizedRF &qrf, int mode)
{
	QRF_ResTree qt;
	QRF_ResNode qn;

	// Get ordered values
	map<float, int> all_values;
	init_keep_all_values(qrf, mode, all_values);

	qtrees.clear();
	qtrees.resize(trees.size());
	for (int i = 0; i < trees.size(); i++) {
		transfer_tree_to_res_tree(qrf, trees[i], qtrees[i], mode, all_values);
	}

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
void QRF_Forest::write(FILE *fp)
{

	fprintf(stderr, "Mode: %d\n", mode);

	for (unsigned int i = 0; i < qtrees.size(); i++) {
		for (unsigned int j = 0; j < qtrees[i].qnodes.size(); j++) {
			QRF_ResNode& node = qtrees[i].qnodes[j];

			fprintf(fp, "Tree %d Node %d ", i, j);
			if (node.is_leaf)
				fprintf(fp, "Prediction %f\n", node.pred);
			else
				fprintf(fp, "Split by %d at %f to %d and %d\n", node.ifeat, node.split_val, node.left, node.right);
		}
	}
}


//-----------------------------------------------------------------------------------------------------------------------------------
int QRF_Forest::get_forest(double *x, double *y, int nfeat, int nsamples, int *sampsize, int ntry, int ntrees, int maxq)
{
	float *xf;
	int *yi;

	// simply converting data to float *x, int *y and applying the regular call.

	xf = new float[(size_t)nsamples*(size_t)nfeat];
	yi = new int[nsamples];

	if (xf == NULL || yi == NULL) {
		delete xf;
		delete yi;
		return -1;
	}

	for (size_t i = 0; i < (size_t)nfeat*(size_t)nsamples; i++)
		xf[i] = (float)x[i];
	for (int i = 0; i < nsamples; i++)
		yi[i] = (int)y[i];

	int rc = get_forest(xf, yi, nfeat, nsamples, sampsize, ntry, ntrees, maxq);

	delete []xf;
	delete []yi;

	return (rc);
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QRF_Forest::get_forest(float *x, int *y, int nfeat, int nsamples, int *sampsize, int ntry, int ntrees, int maxq)
{
	n_categ = 2;
	if (mode == QRF_BINARY_TREE)
		return(get_forest_trees_all_modes(x, (void *)y, NULL, nfeat, nsamples, sampsize, ntry, ntrees, maxq, QRF_BINARY_TREE));
	else {
		fprintf(stderr, "qrf: get_forest: error: unsupported mode in this API %d\n", mode); fflush(stderr);
		return -1;
	}

	return(0);
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QRF_Forest::get_forest_regression_trees(float *x, float *y, int nfeat, int nsamples, int sampsize, int ntry, int ntrees, int maxq, int min_node, float spread)
{
	min_node_size = min_node;
	min_spread = spread;
	if (min_node < 0) min_node_size = MIN_SPLIT_NODE_SIZE;
	if (spread < 0) min_spread = (float)MIN_SPLIT_SPREAD;
	n_categ = 0;
	return(get_forest_trees_all_modes(x, (void *)y, NULL, nfeat, nsamples, &sampsize, ntry, ntrees, maxq, QRF_REGRESSION_TREE));
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QRF_Forest::get_forest_categorical(float *x, float *y, const float *w, int nfeat, int nsamples, int *sampsize, int ntry, int ntrees, int maxq, int min_node, int ncateg, int splitting_method)
{

	if (splitting_method != QRF_CATEGORICAL_CHI2_TREE && splitting_method != QRF_CATEGORICAL_ENTROPY_TREE && splitting_method != QRF_MULTILABEL_ENTROPY_TREE) {
		fprintf(stderr, "Unknown splitting method %d\n", splitting_method);
		return -1;
	}

	if (min_node < 0)
		min_node_size = MIN_SPLIT_NODE_SIZE;
	else
		min_node_size = min_node;

	min_spread = (float)MIN_SPLIT_SPREAD;
	if (ncateg < 0) return -1;
	n_categ = ncateg;

	//fprintf(stderr, "QRF1: sampsize %d %d type is %d\n", sampsize[0], sampsize[1], splitting_method);
	return(get_forest_trees_all_modes(x, (void *)y, w, nfeat, nsamples, sampsize, ntry, ntrees, maxq, splitting_method));
}

//-----------------------------------------------------------------------------------------------------------------------------------
// in regression and categorical modes the sampsize is sampsize[0]
int QRF_Forest::get_forest_trees_all_modes(float *x, void *y, const float *w, int nfeat, int nsamples, int *sampsize, int ntry, int ntrees, int maxq, int tree_mode)
{
	QuantizedRF qrf;

	omp_set_num_threads((int)std::thread::hardware_concurrency());
	mode = tree_mode;

	if (mode != QRF_BINARY_TREE && mode != QRF_REGRESSION_TREE && mode != QRF_CATEGORICAL_CHI2_TREE && mode != QRF_CATEGORICAL_ENTROPY_TREE && mode != QRF_MULTILABEL_ENTROPY_TREE)
		return -1;

	qrf.tree_mode = mode;
	if (mode == QRF_CATEGORICAL_CHI2_TREE || mode == QRF_CATEGORICAL_ENTROPY_TREE || mode == QRF_MULTILABEL_ENTROPY_TREE)
		qrf.n_categ = n_categ;
	else if (mode == QRF_BINARY_TREE) {
		n_categ = 2;
		qrf.n_categ = 2;
	}
	else {
		// REGRESSION
		n_categ = 0;
		qrf.n_categ = 0;
	}
	qrf.min_split_node_size = min_node_size;
	qrf.min_split_spread = min_spread;
	qrf.take_all_samples = take_all_samples;
	qrf.max_depth = max_depth;

	if (mode == QRF_BINARY_TREE)
		qrf.init(x, (int *)y, nfeat, nsamples, maxq);
	if ((mode == QRF_REGRESSION_TREE) || (mode == QRF_CATEGORICAL_CHI2_TREE) || (mode == QRF_CATEGORICAL_ENTROPY_TREE) || (mode == QRF_MULTILABEL_ENTROPY_TREE))
		qrf.init_regression(x, (float *)y, w, nfeat, nsamples, maxq);


	if (ntry <= 0) {
		ntry = (int)(sqrt((double)nfeat) + 1.0);
	}

#ifdef DEBUG
	fprintf(stderr, "get_forest: mode %d ntry %d nfeat %d nsamples %d ntrees %d n_categ %d min_node %d min_spread %f\n", mode, ntry, nfeat, nsamples, ntrees, n_categ, min_node_size, min_spread);
	if (sampsize != NULL) { fprintf(stderr, "get_forest: sampsize %d %d ....\n", sampsize[0], sampsize[1]); };
	if (sampsize != NULL && ((mode == QRF_REGRESSION_TREE) || (mode == QRF_CATEGORICAL_CHI2_TREE) || (mode == QRF_CATEGORICAL_ENTROPY_TREE))) { fprintf(stderr, "get_forest: sampsize %d\n", sampsize[0]); };
	fflush(stderr);
#endif
	qrf.init_groups(groups);

	vector<QRF_Tree> trees;
	if (collect_oob) trees.resize(ntrees);
	map<float, int> all_values;
	init_keep_all_values(qrf, mode, all_values);
	qtrees.clear();
	qtrees.resize(ntrees);
#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < ntrees; i++) {
		QRF_Tree tree;
		tree.init_rand_state();
		if (mode == QRF_BINARY_TREE)
			qrf.get_Tree(sampsize, ntry, tree);
		if ((mode == QRF_REGRESSION_TREE) || (mode == QRF_CATEGORICAL_CHI2_TREE) || (mode == QRF_CATEGORICAL_ENTROPY_TREE) || (mode == QRF_MULTILABEL_ENTROPY_TREE)) {
			qrf.get_regression_Tree(sampsize, ntry, tree);
		}
		tree.nodes.resize(tree.n_nodes);
		transfer_tree_to_res_tree(qrf, tree, qtrees[i], mode, all_values);

		if (collect_oob) trees[i] = tree;
	}

#if DEBUG
	fprintf(stderr, "%d trees transffered\n", ntrees); fflush(stderr);
	for (int i = 0; i < qtrees.size(); i++)
		fprintf(stderr, " qtree %d : esize %ld nodes %d\n", i, qtrees[i].estimated_size(), (int)qtrees[i].qnodes.size()); fflush(stderr);
#endif

	if (collect_oob > 0)
		fprintf(stderr, "collect oob = %d\n", collect_oob);
	fflush(stderr);
	if (collect_oob) {
		for (int i = 0; i < ntrees; i++)
			qrf.collect_Tree_oob_scores(x, nfeat, qtrees[i], trees[i].sample_ids);
		qrf.complete_oob_cv();

		//		if (mode == QRF_BINARY_TREE) {
		//			float auc = (float)qrf.get_cross_validation_auc();
		//			fprintf(stderr,"qrf oob cv: auc = %f\n",auc); fflush(stderr);
		//		}

		oob_scores.resize(nsamples);
#pragma omp parallel for
		for (int i = 0; i < nsamples; i++) {
			if (tree_mode == QRF_REGRESSION_TREE) {
				oob_scores[i].resize(2);
				oob_scores[i][0] = qrf.cv[i].mean;
				oob_scores[i][1] = qrf.cv[i].std;
			}
			else {
				oob_scores[i].resize(n_categ);
				for (int j = 0; j < n_categ; j++)
					oob_scores[i][j] = qrf.cv[i].probs[j];
			}
		}
	}

#ifdef DEBUG
	fprintf(stderr, "get_forest: got %d trees\n", ntrees); fflush(stderr);
#endif

	return 0;
}

//---------------------------------------------------------------------------------------------
void get_scoring_thread_params(vector<qrf_scoring_thread_params> &tp, const vector<QRF_ResTree> *qtrees, float *res, int nsamples, int nfeat, float *x, int nsplit, int mode, int n_categ, int get_counts,
	vector<float> *quantiles, const vector<float> *sorted_values, bool sparse_values)
{
	tp.clear();
	tp.resize(nsplit);
	int jump = nsamples / nsplit;
	int leftover = nsamples - jump * nsplit;
	int prev = -1;
	for (int i = 0; i < nsplit; i++) {
		tp[i].x = x;
		tp[i].trees = qtrees;
		tp[i].nfeat = nfeat;
		tp[i].nsamples = nsamples;
		tp[i].from = prev + 1; //i*jump;
		tp[i].to = prev + jump; //(i+1)*jump-1;
		if (leftover > 0) { leftover--; tp[i].to++; }
		prev = tp[i].to;
		tp[i].serial = i;
		tp[i].state = 0;
		tp[i].res = res;
		tp[i].mode = mode;
		tp[i].n_categ = n_categ;
		tp[i].get_counts = get_counts;
		tp[i].quantiles = quantiles;
		tp[i].sorted_values = sorted_values;
		tp[i].sparse_values = sparse_values;
	}
	tp[nsplit - 1].to = nsamples - 1;
}

void QRF_ResNode::get_scores(int mode, int get_counts_flag, int n_categ, vector<float> &scores) const {
	//scores is allocated
	//int pred_size = (int)scores.size();

	vector<float> cnts(n_categ);
	float sum = 0;
	float norm = 0;

	fill(cnts.begin(), cnts.end(), (float)0);

	if (mode == QRF_REGRESSION_TREE) {
		if (get_counts_flag == PREDS_REGRESSION_AVG) { // Average on predictions
			sum += pred;
			norm++;
		}
		else if (get_counts_flag == PREDS_REGRESSION_WEIGHTED_AVG) { // Weighted average on predictions
			sum += pred * n_size;
			norm += n_size;
		}
		else { // Quantile Regression or sampling
			MTHROW_AND_ERR("Unsupported for single node\n");
		}
	}
	else {
		if (get_counts_flag == PROBS_CATEG_MAJORITY_AVG || get_counts_flag == PREDS_CATEG_MAJORITY_AVG) { // Majority
			cnts[majority]++;
			++norm;
		}
		else if (get_counts_flag == PROBS_CATEG_AVG_PROBS || get_counts_flag == PREDS_CATEG_AVG_PROBS) { // Average on probabilities
			assert(n_size > 0);
			for (int k = 0; k < n_categ; k++)
				cnts[k] += ((float)counts[k]) / ((float)n_size);
			norm++;
		}
		else { // Average on counts
			for (int k = 0; k < n_categ; k++)
				cnts[k] += (float)counts[k];
			norm += n_size;
		}
	}


	if (mode == QRF_REGRESSION_TREE) {
		if (get_counts_flag == PREDS_REGRESSION_WEIGHTED_AVG || get_counts_flag == PREDS_REGRESSION_AVG)
			scores[0] = sum / norm;
		else if (get_counts_flag == PREDS_REGRESSION_QUANTILE || get_counts_flag == PREDS_REGRESSION_WEIGHTED_QUANTILE ||
			get_counts_flag == PREDS_REGRESSION_SAMPLE) {
			MTHROW_AND_ERR("Unsupported for single node\n");
		}
	}
	else if (get_counts_flag == PREDS_CATEG_MAJORITY_AVG || get_counts_flag == PREDS_CATEG_AVG_COUNTS || get_counts_flag == PREDS_CATEG_AVG_PROBS) {

		// collapse cnts/norm to a single prediction by expectation
		scores[0] = 0;
		for (int k = 0; k < n_categ; k++)
			scores[0] += (cnts[k] / norm)*(float)k;

	}
	else {
		for (int k = 0; k < n_categ; k++)
			scores[k] = cnts[k] / norm;
	}

}

//-----------------------------------------------------------------------------------------------------------------------------------
void get_score_thread(void *p)
{
	qrf_scoring_thread_params *tp = (qrf_scoring_thread_params *)p;

	//	fprintf(stderr,"Starting scoring thread %d :: from %d to %d mode %d n_categ %d\n",tp->serial, tp->from, tp->to, tp->mode, tp->n_categ); fflush(stderr);

	const float *xf = tp->x;

	int node;
	int nfeat = tp->nfeat;
	const vector<QRF_ResTree> *trees = tp->trees;
	vector<float> *quantiles = tp->quantiles;
	int n_quantiles = (int)(*quantiles).size();
	int n_categ = tp->n_categ;
	vector<float> cnts(n_categ);

	// Mat must be FLOAT & REGULAR transposed
	for (size_t i = tp->from; i <= tp->to; i++) {
		float sum = 0;
		float norm = 0;


		vector<float> values((*(tp->sorted_values)).size());
		vector<int> sizes((*(tp->trees)).size());
		float totWeight = 0, totUnweighted = 0;

		fill(cnts.begin(), cnts.end(), (float)0);

		for (size_t j = 0; j < (*(tp->trees)).size(); j++) {
			node = 0;

			// Find relevant leaf
			while (!(*trees)[j].qnodes[node].is_leaf) {
				if (xf[i*(size_t)nfeat + (size_t)((*trees)[j].qnodes[node].ifeat)] <= (*trees)[j].qnodes[node].split_val)
					node = (*trees)[j].qnodes[node].left;
				else
					node = (*trees)[j].qnodes[node].right;
			}

			// Add to counts
			if (tp->mode == QRF_REGRESSION_TREE) {
				if (tp->get_counts == PREDS_REGRESSION_AVG) { // Average on predictions
					sum += (*trees)[j].qnodes[node].pred;
					norm++;
				}
				else if (tp->get_counts == PREDS_REGRESSION_WEIGHTED_AVG) { // Weighted average on predictions
					sum += (*trees)[j].qnodes[node].pred * (*trees)[j].qnodes[node].n_size;
					norm += (*trees)[j].qnodes[node].n_size;
				}
				else { // Quantile Regression or sampling
					float w = (tp->get_counts == PREDS_REGRESSION_QUANTILE || tp->get_counts == PREDS_REGRESSION_SAMPLE) ? (1.0F) : (1.0F / (*trees)[j].qnodes[node].tot_n_values);
					if (tp->sparse_values) {
						for (auto& rec : (*trees)[j].qnodes[node].value_counts)
							values[rec.first] += w * rec.second;
					}
					else {
						for (unsigned int iVal = 0; iVal < (*trees)[j].qnodes[node].values.size(); iVal++)
							values[iVal] += w * (*trees)[j].qnodes[node].values[iVal];
					}
					sizes[j] = (*trees)[j].qnodes[node].tot_n_values;
					totWeight += w * sizes[j];
					totUnweighted += sizes[j];
				}
			}
			else {
				if (tp->get_counts == PROBS_CATEG_MAJORITY_AVG || tp->get_counts == PREDS_CATEG_MAJORITY_AVG) { // Majority
					cnts[(*trees)[j].qnodes[node].majority]++;
					norm++;
				}
				else if (tp->get_counts == PROBS_CATEG_AVG_PROBS || tp->get_counts == PREDS_CATEG_AVG_PROBS) { // Average on probabilities
					assert(((*trees)[j].qnodes[node].n_size) > 0);
					for (int k = 0; k < n_categ; k++)
						cnts[k] += ((float)(*trees)[j].qnodes[node].counts[k]) / ((float)(*trees)[j].qnodes[node].n_size);
					norm++;
				}
				else { // Average on counts
					for (int k = 0; k < n_categ; k++)
						cnts[k] += (float)(*trees)[j].qnodes[node].counts[k];
					norm += (*trees)[j].qnodes[node].n_size;
				}
			}
		}

		if (tp->mode == QRF_REGRESSION_TREE) {
			if (tp->get_counts == PREDS_REGRESSION_WEIGHTED_AVG || tp->get_counts == PREDS_REGRESSION_AVG)
				tp->res[i] = sum / norm;
			else if (tp->get_counts == PREDS_REGRESSION_QUANTILE || tp->get_counts == PREDS_REGRESSION_WEIGHTED_QUANTILE) {

				int ptr = 0;
				float sumWeight = 0.0F;
				for (int k = 0; k < n_quantiles; k++) {
					float q = (*quantiles)[k];

					// -2 >=  q  > -(nTrees + 2) : size of relevant node in tree -(q+2)
					if ((-q - 2) >= 0 && (-q - 2) < (*(tp->trees)).size())
						tp->res[i*n_quantiles + k] = (float)sizes[(int)(-q - 2)];
					else if (q == -1)
						// Total number of  values
						tp->res[i*n_quantiles + k] = totUnweighted;
					else {
						// Quantile
						while (sumWeight / totWeight < q && ptr < values.size())
							sumWeight += values[ptr++];
						if (ptr > 0) {
							ptr--;
							sumWeight -= values[ptr];
						}
						tp->res[i*n_quantiles + k] = (*(tp->sorted_values))[ptr];
					}
				}
			}
			else if (tp->get_counts == PREDS_REGRESSION_SAMPLE) {
				float p = (0.0 + QRFglobalRNG::rand()) / QRFglobalRNG::max();
				float sumWeight = 0;
				for (size_t j = 0; j < values.size(); j++) {
					sumWeight += values[j];
					if (sumWeight / totWeight >= p) {
						tp->res[i] = (*(tp->sorted_values))[j];
						break;
					}
				}
			}
		}
		else if (tp->get_counts == PREDS_CATEG_MAJORITY_AVG || tp->get_counts == PREDS_CATEG_AVG_COUNTS || tp->get_counts == PREDS_CATEG_AVG_PROBS) {

			// collapse cnts/norm to a single prediction by expectation
			tp->res[i] = 0;
			for (int k = 0; k < n_categ; k++)
				tp->res[i] += (cnts[k] / norm)*(float)k;

		}
		else {
			for (int k = 0; k < n_categ; k++)
				tp->res[i*n_categ + k] = cnts[k] / norm;
		}
	}

	tp->state = 1; // signing we end the thread
}

//-----------------------------------------------------------------------------------------------------------------------------------
void QRF_Forest::score_with_threads(float *x, int nfeat, int nsamples, float *res) const
{
	vector<qrf_scoring_thread_params> stp;

	// order quantilers (keeping original order)
	vector<pair<float, int>> indexd_quantiles(quantiles.size());
	vector<float> sorted_quantiles(quantiles.size());

	if (get_counts_flag == PREDS_REGRESSION_WEIGHTED_QUANTILE || get_counts_flag == PREDS_REGRESSION_QUANTILE) {
		for (unsigned int i = 0; i < quantiles.size(); i++) indexd_quantiles[i] = { quantiles[i],i };
		sort(indexd_quantiles.begin(), indexd_quantiles.end(), [](const pair<float, int> &v1, const pair<float, int> &v2) {return v1.first < v2.first; });
		for (unsigned int i = 0; i < quantiles.size(); i++) sorted_quantiles[i] = indexd_quantiles[i].first;
	}

	// handle case where nthreads is larger than nsamples
	int eff_nthreads = MIN(nthreads, nsamples);

	//	fprintf(stderr, "QRF: mode %d get_counts_flag %d n_categ %d\n", mode, get_counts_flag, n_categ);
	get_scoring_thread_params(stp, &qtrees, res, nsamples, nfeat, x, eff_nthreads, mode, n_categ, get_counts_flag, &sorted_quantiles, &sorted_values, sparse_values);
	vector<thread> th_handle(eff_nthreads);
	for (int i = 0; i < eff_nthreads; i++) {
		th_handle[i] = thread(get_score_thread, (void *)&stp[i]);
	}

	int n_state = 0;
	while (n_state < eff_nthreads) {
		this_thread::sleep_for(chrono::milliseconds(10));
		n_state = 0;
		for (int i = 0; i < eff_nthreads; i++)
			n_state += stp[i].state;
	}
	for (int i = 0; i < eff_nthreads; i++)
		th_handle[i].join();

	// Reorderd quantiles to original orderd.diff 
	if (get_counts_flag == PREDS_REGRESSION_WEIGHTED_QUANTILE) {
		int nquantiles = (int)quantiles.size();
		vector<float> tempRes(static_cast<size_t>(nsamples) * static_cast<size_t>(nquantiles));
		for (int i = 0; i < nsamples; i++) {
			for (int j = 0; j < nquantiles; j++)
				tempRes[i*nquantiles + indexd_quantiles[j].second] = res[i*nquantiles + j];
		}
		memcpy(res, &(tempRes[0]), static_cast<size_t>(nquantiles) * static_cast<size_t>(nsamples) * sizeof(float));
	}

}

void QRF_Forest::get_single_score_fast(qrf_scoring_thread_params &params, const vector<float> &x, vector<float> &preds) const {
	int pred_size = n_categ;
	if (get_only_this_categ >= 0)
		pred_size = 1;
	preds.resize(pred_size);

	params.x = x.data();
	params.res = preds.data();

	get_score_thread(&params);
}

//-----------------------------------------------------------------------------------------------------------------------------------
int QRF_Forest::score_samples(float *x_in, int nfeat, int nsamples, float *&res) const
{
	return(score_samples(x_in, nfeat, nsamples, res, 0));
}


//-----------------------------------------------------------------------------------------------------------------------------------
int QRF_Forest::score_samples(float *x_in, int nfeat, int nsamples, float *&res, int get_counts) const
{

	//get_counts_flag = get_counts; //already set in learn
	if (mode != QRF_BINARY_TREE && mode != QRF_REGRESSION_TREE && mode != QRF_CATEGORICAL_CHI2_TREE && mode != QRF_CATEGORICAL_ENTROPY_TREE && (mode != QRF_MULTILABEL_ENTROPY_TREE)) {
		fprintf(stderr, "qrf: score_samples - mode %d not supported\n", mode); fflush(stderr);
		return -1;
	}

#ifdef DEBUG
	fprintf(stderr, "qrf: score_samples: scoring %d samples with %d features, get_only_this_categ=%d\n", nsamples, nfeat, get_only_this_categ); fflush(stderr);
#endif

	if (mode == QRF_REGRESSION_TREE || get_only_this_categ < 0)
		score_with_threads(x_in, nfeat, nsamples, res);
	else {
		vector<float> resall(static_cast<size_t>(nsamples) * static_cast<size_t>(n_categ));
		score_with_threads(x_in, nfeat, nsamples, &resall[0]);
		if (get_only_this_categ < n_categ)
			for (int i = 0; i < nsamples; i++)
				res[i] = resall[i*n_categ + get_only_this_categ];
		else {
			// of get_only_this_categ >= n_categ it is a sign for us to give out a full prediction vector for each sample
			for (size_t i = 0; i < nsamples; i++) {
				for (size_t j = 0; j < n_categ; j++)
					res[i*(size_t)n_categ + j] += resall[i*(size_t)n_categ + j];
			}

		}

	}

	return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
// ToDo:: This one needs to be abandoned... all matrix preps should be done outside the scope of this package.
int QRF_Forest::score_samples_t(double *x, int nfeat, int nsamples, double *&res)
{
	float *rf = (float *)new float[2 * nsamples];
	float *xf = (float *) new float[(size_t)nsamples*(size_t)nfeat];

	if (xf == NULL || rf == NULL) {
		fprintf(stderr, "qrf: score_samples_t: error: Can't allocate transposed mat\n"); fflush(stderr);
		return -1;
	}

	for (size_t i = 0; i < nsamples; i++)
		for (size_t j = 0; j < nfeat; j++)
			xf[i*(size_t)nfeat + j] = (float)x[j*(size_t)nsamples + i];

	int rc = score_samples(xf, nfeat, nsamples, rf);

	// Prediction is second column
	for (int i = 0; i < nsamples; i++)
		res[i] = (double)rf[2 * i + 1];
	delete[] rf;
	delete[] xf;

	return rc;
}

// fills the vector rankedFeatures with pairs of feature index and importance score, ranked by decreasing score.
// requires the number of features as input (features are indexed 0 to nFeatures - 1)
// for each feature, all nodes split by that feature are considered, and their splitting scores are summed. This summed score 
// (normalized by the number of trees) measures the importance of the feature.
void QRF_Forest::variableImportance(vector<pair<short, double> >& rankedFeatures, unsigned int nFeatures)
{
	rankedFeatures.resize(nFeatures);

	vector<double> summedScores(nFeatures);
	double curScore;
	const double epsilon = 1e-6; //for QRF_CATEGORICAL_CHI2_TREE

	for (unsigned int i = 0; i < qtrees.size(); i++) { //go over all trees
		for (unsigned int j = 0; j < qtrees[i].qnodes.size(); j++) { //traverse a single tree
			QRF_ResNode& node = qtrees[i].qnodes[j];
			if (node.ifeat >= 0 && !node.is_leaf) {
				QRF_ResNode& leftChild = qtrees[i].qnodes[node.left];
				QRF_ResNode& rightChild = qtrees[i].qnodes[node.right];
				//calculate score for the node
				if (mode == QRF_BINARY_TREE) {
					curScore = ((double)(node.counts[0])*node.counts[1] / node.n_size - (double)(leftChild.counts[0])*leftChild.counts[1] / leftChild.n_size -
						(double)(rightChild.counts[0])*rightChild.counts[1] / rightChild.n_size) / (double)node.n_size;
				}
				else if (mode == QRF_REGRESSION_TREE) {
					curScore = (double)leftChild.n_size*(leftChild.pred)*(leftChild.pred) +
						(double)rightChild.n_size*(rightChild.pred)*(rightChild.pred) -
						(double)node.n_size*(node.pred)*(node.pred);
				}
				else if (mode == QRF_CATEGORICAL_CHI2_TREE) {
					curScore = 0.0;

					for (int k = 0; k < n_categ; ++k) {
						double expL = (double)(node.counts[k])*leftChild.n_size / node.n_size;
						double expR = (double)(node.counts[k])*rightChild.n_size / node.n_size;
						if (expL > epsilon)
							curScore += SQUARE(leftChild.counts[k] - expL) / expL;
						if (expR > epsilon)
							curScore += SQUARE(rightChild.counts[k] - expR) / expR;
					}

					curScore /= node.n_size;
				}
				else if ((mode == QRF_CATEGORICAL_ENTROPY_TREE) || (mode == QRF_MULTILABEL_ENTROPY_TREE)) {
					double H = (double)node.n_size * log((double)node.n_size);
					double HR = (double)rightChild.n_size * log((double)rightChild.n_size);
					double HL = (double)leftChild.n_size * log((double)leftChild.n_size);

					int c;

					for (int k = 0; k < n_categ; ++k) {
						c = node.counts[k];
						if (c)
							H -= ((double)c * log((double)c));

						c = leftChild.counts[k];
						if (c)
							HL -= ((double)c * log((double)c));

						c = rightChild.counts[k];
						if (c)
							HR -= ((double)c * log((double)c));
					}

					curScore = (double)(H - HR - HL) / node.n_size;
				}
				else {//unexpected 
					fprintf(stderr, "invalid mode: %d!\n", mode);
					return;
				}

				summedScores[node.ifeat] += curScore;
				/*
				if (curScore < -epsilon)
					printf("found negative score: %f\n", curScore);
				*/
			}
		}
	}

	for (unsigned int i = 0; i < nFeatures; ++i)
		rankedFeatures[i] = pair<short, double>((short)i, summedScores[i] / qtrees.size());

	//higher score means greater importance
	struct VarImpComparator {
		bool operator()(const pair<short, double>& left, const pair<short, double>& right) {
			return (left.second > right.second);
		}
	};

	sort(rankedFeatures.begin(), rankedFeatures.end(), VarImpComparator());
}
