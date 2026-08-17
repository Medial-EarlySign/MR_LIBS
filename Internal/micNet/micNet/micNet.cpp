//=================================================================================
// micNet.cpp - implementation file for micNet: simple deep learning class
//=================================================================================
#ifndef _SCL_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#endif


#include <Logger/Logger/Logger.h>
#include <MedUtils/MedUtils/MedGenUtils.h>

#include <random>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_APP
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

#include "micNet.h"


//=================================================================================
// micNode
//=================================================================================
//.................................................................................
int micNode::init_wgts_rand(float min_range, float max_range)
{
	wgt.resize(n_in + 1, k_out + 1);

	int i, j;
	for (i = 0; i < n_in + 1; i++) {
		for (j = 0; j < k_out; j++)
			wgt(i, j) = rand_range(min_range, max_range);
		wgt(i, k_out) = 0;
	}

	wgt(n_in, k_out) = 1;

	return 0;
}

//.................................................................................
int micNode::init_wgts_rand_normal(float mean, float std)
{
	wgt.resize(n_in + 1, k_out + 1);

	std::default_random_engine gen;
	std::normal_distribution<float> dist(mean, std);

	int i, j;
	wgt.zero();
	for (i = 0; i < n_in; i++) {
		for (j = 0; j < k_out; j++) {
			//if (rand_1()<0.25)
			//	wgt(i, j) = 2;
			wgt(i, j) = dist(gen);
			//MLOG("wgt init (%d,%d) = %f %f\n", i, j, std, wgt(i, j));
		}
		//		wgt(i, k_out) = 0;
	}

	wgt(n_in, k_out) = 1; // last column is 0,0,...1 so that input 1's for bias are kept over to next layer

	return 0;
}

//.................................................................................................
// copies x data (shuffled batch) into an input node
int micNode::fill_input_node(int *perm, int len, MedMat<float> &x_mat, int last_is_bias_flag)
{
	int nfeat = x_mat.ncols;
	if (last_is_bias_flag) nfeat--;

	batch_out.resize(len, nfeat + 1);
	float *b_out = batch_out.data_ptr();

	// now going over our 'len' permutated lines and copying each line from x to the batch_out buffer.
	for (int i = 0; i < len; i++) {
		memcpy(b_out, x_mat.data_ptr(perm[i],0), nfeat * sizeof(float));
		b_out[nfeat] = 1; // bias term
		b_out += nfeat + 1;
	}

	return 0;
}

//.................................................................................................
// copies y data (shuffled batch) into an output node
int micNode::fill_output_node(int *perm, int len, MedMat<float> &y_mat, vector<float> &sample_weights)
{
	y.resize(len, y_mat.ncols);
	for (int i = 0; i < len; i++) {
		int ii = perm[i];
		for (int j = 0; j < y_mat.ncols; j++)
			y(i, j) = y_mat(ii, j);
	}

	if (sample_weights.size() > 0) {
		sweights.resize(len);
		for (int i = 0; i < len; i++)
			sweights[i] = sample_weights[perm[i]];
	}
	else
		sweights.clear();

	return 0;
}

void micNode::get_input_batch(const vector<MedMat<float>> &nodes_out, MedMat<float> &in) const {
	// simpler code for just a single input node
	if (ir.n_input_nodes == 1) {

		// get the input node
		int j = ir.in_node_id[0];

		// copy batch from input and check it
		//MLOG("Copying input to batch_in of node %d from batch_out of node %d\n", id, in_node.id);
		if (ir.mode[0] == "all") {
			//MLOG("Copying batch_out from node %d to batch_in in node %d : batch_out is: %d x %d\n", in_node.id, id, in_node.batch_out.nrows, in_node.batch_out.ncols);
			in = nodes_out[j];
		}
	}
}

//.................................................................................
// Sets the input for the node, using the InputRules 
// Current Options:
// - get input from several nodes
// - model "all" : simply uses all outputs of last node
int micNode::get_input_batch(int do_grad_flag)
{
	int n_in_dim = 0;
	for (int i = 0; i < ir.n_input_nodes; i++) {
		n_in_dim += my_net->nodes[ir.in_node_id[i]].k_out;
	}

	if (n_in_dim != n_in) {
		MERR("%s :: get_input_batch() :: id %d type %s :: ERROR :: non matching input dimensions %d vs. %d\n", name.c_str(), id, type.c_str(), n_in_dim, n_in);
		return -1;
	}

	// getting y for the case of autoencoders
	if (data_node >= 0) {
		y = data_node_p->batch_out;
	}

	//y = ir.in_node_ptr[0]->y; // if there are several input nodes they MUST have the same y for the batch

	// simpler code for just a single input node
	if (ir.n_input_nodes == 1) {

		// get the input node
		int j = ir.in_node_id[0];
		micNode &in_node = my_net->nodes[j];

		// copy batch from input and check it
		//MLOG("Copying input to batch_in of node %d from batch_out of node %d\n", id, in_node.id);
		if (ir.mode[0] == "all") {
			//MLOG("Copying batch_out from node %d to batch_in in node %d : batch_out is: %d x %d\n", in_node.id, id, in_node.batch_out.nrows, in_node.batch_out.ncols);
			batch_in = in_node.batch_out;
			dropout_in = in_node.dropout_out;
		}

		if (batch_in.ncols - 1 != n_in_dim) {
			MERR("%s :: get_input_batch() :: ERROR :: node %d :: non matching dimensions %d <-> %d\n", name.c_str(), id, batch_in.ncols, n_in_dim);
			return -1;
		}

		// handle dropout: actually randomizing the dropout matrix to decide which weights will be used in this batch
		if (do_grad_flag && dropout_prob_in < 1) {
			if (dropout_in.size() == 0) {
				for (int i = 0; i < n_in; i++)
					if (rand_1() <= dropout_prob_in)
						dropout_in.push_back((float)1);
					else
						dropout_in.push_back((float)0);
				dropout_in.push_back(1); // bias always chosen
			}

			dropout_out.resize(k_out + 1);
			for (int i = 0; i < k_out; i++)
				if (rand_1() <= dropout_prob_out)
					dropout_out[i] = (float)1;
				else
					dropout_out[i] = (float)0;
			dropout_out[k_out] = 1; //bias always chosen

			for (int i = 0; i < batch_in.nrows; i++)
				for (int j = 0; j < n_in; j++)
					batch_in(i, j) *= dropout_in[j]; // this zeros the columns we don't want to sum on. Probably an over kill and could have been 
													 // computed faster when multiplying weights on batch
		}
	}
	else {
		// TBD:: code for input from several previous nodes
	}

	return 0;
}

void micNode::forward_batch_leaky_relu(const MedMat<float> &in, MedMat<float> &out) const
{
	// sanity checks
	if (in.ncols != n_in + 1 || wgt.nrows != n_in + 1 || wgt.ncols != k_out + 1 || lr_params.nrows != k_out || lr_params.ncols != 2 || lambda.nrows != k_out) {
		MTHROW_AND_ERR("ERROR:: micNode::forward_batch_leaky_relu : input: %lld x %lld , wgt %lld x %lld , n_in %d , k_out %d , lr_params: %lld x %lld lambda: %lld x %lld\n",
			in.nrows, in.ncols, wgt.nrows, wgt.ncols, n_in, k_out, lr_params.nrows, lr_params.ncols, lambda.nrows, lambda.ncols);
	}

	// multiplying to get W * x
	//	print("debug before mult mat", 1, 6);
	if (dropout_prob_out >= 1) {
		if (fast_multiply_medmat_(in, wgt, out) < 0)
			MTHROW_AND_ERR("Failed\n");
	}
	else {
		if (fast_multiply_medmat(in, wgt, out, dropout_prob_out) < 0)
			MTHROW_AND_ERR("Failed\n");
	}

	// Reminder: the last col in weights is 0,0,....0,1 
	// That with the last in each input row being 1 makes sure the output will also have 1's at the end.
	int i = 0;
	int n_b = out.nrows;

	// applying leaky max func on out

		// faster, without grad_s calculation
//#pragma omp parallel for private(i) if (n_b>100 || k_out>100)
	for (i = 0; i < n_b; i++)
		for (int j = 0; j < k_out; j++)
			if (out(i, j) >= 0)
				out(i, j) *= lr_params(j, 0); // a
			else
				out(i, j) *= lr_params(j, 1); // b

}

void micNode::forward_batch_normalization(const MedMat<float> &in, MedMat<float> &out) const {
	// sanity checks
	if (in.ncols != n_in + 1 || wgt.nrows != n_in + 1 || wgt.ncols != k_out + 1 || n_in != k_out) {
		MTHROW_AND_ERR("ERROR:: micNode::forward_batch_normalization : input: %lld x %lld , wgt %lld x %lld , n_in %d , k_out %d , lr_params: %lld x %lld lambda: %lld x %lld\n",
			in.nrows, in.ncols, wgt.nrows, wgt.ncols, n_in, k_out, lr_params.nrows, lr_params.ncols, lambda.nrows, lambda.ncols);
	}

	// multiplying to get W * x - leaving this code, although it should be made faster by multiplying a diagonal matrix.
	int n_b = in.nrows;
	out.resize(n_b, n_in + 1);

	for (int i = 0; i < n_b; i++) {
		for (int j = 0; j < n_in; j++) {
			int alpha_j = 1, beta_j = 0;
			if (j < alpha.size())
				alpha_j = alpha[j];
			if (j < beta.size())
				beta_j = beta[j];
			out(i, j) = in(i, j) * alpha_j + beta_j;
		}
		out(i, n_in) = 1;
	}

}

void micNode::forward_batch_softmax(const MedMat<float> &in, MedMat<float> &out) const {
	int i, j;
	int n_b = in.nrows;

	// sanity
	if (in.ncols != n_in + 1 || k_out != n_categ || n_in != n_categ * n_per_categ) {
		MTHROW_AND_ERR("ERROR:: forward_batch_softmax() :: non matching sizes :: input %lld x %lld , n_in %d k_out %d\n", in.nrows, in.ncols, n_in, k_out);
	}


	out.resize(in.nrows, k_out + 1);
	MedMat<float> &full_probs = out;
	//full_probs.resize(in.nrows, n_in + 1);
	//	print("softmax_forward before calcs:", 1, 6);
	float max_exp = (float)34;
	float min_exp = (float)-34;
	// softmax calc
	//#pragma omp parallel for if (n_b>100 || n_in>100)
	for (i = 0; i < n_b; i++) {
		double sum = 0;
		float max_val = in(i, 0);
		for (j = 1; j < n_in; j++)
			if (max_val < in(i, j))
				max_val = in(i, j);
		for (j = 0; j < n_in; j++) {
			full_probs(i, j) = min(max(in(i, j) - max_val, min_exp), max_exp);
			full_probs(i, j) = (float)exp((double)full_probs(i, j));
			sum += (double)full_probs(i, j);
		}

		for (j = 0; j < n_in; j++)
			full_probs(i, j) = (float)(full_probs(i, j) / sum);
		full_probs(i, n_in) = 1; // bias column
	}

	if (n_per_categ != 1) {
		MedMat<float> copy_preds = out;
		float epsilon = (float)1e-10;
		//#pragma omp parallel for private(i) if (n_b>100 || n_in>100)
		for (i = 0; i < n_b; i++) {
			int m = 0;
			for (j = 0; j < n_in; j += n_per_categ) {
				float sum = 0;
				for (int k = j; k < j + n_per_categ; k++)
					sum += copy_preds(i, k);
				out(i, m++) = min(sum, (float)1 - epsilon);
			}
			if (m != n_categ) {
				MTHROW_AND_ERR("ERROR: m %d n_categ %d\n", m, n_categ);
			}
			out(i, k_out) = 1;
		}

	}
}

void micNode::forward_batch_regression(const MedMat<float> &in, MedMat<float> &out) const {
	// sanity
	if (in.ncols != n_in + 1 || n_in != k_out) {
		MTHROW_AND_ERR("ERROR:: forward_batch_regression() :: non matching sizes :: input %lld x %lld , n_in %d k_out %d :: wgt size %lld x %lld\n",
			in.nrows, in.ncols, n_in, k_out, wgt.ncols, wgt.nrows);
	}

	out = in;
}

//.................................................................................
// we have input of size batch_size x (n_in + 1) :: +1 for the bias
// weights of size (n_in + 1) x (k_out + 1) :: last col is 0,0,0...,0,1 to transfer bias "1" forward
// the Leaky Relu function is ::
// y = max(a * W * x, b * W * x) , a,b non negative parameters (not a weight !) , a >= b
// while at it we also estimate the average gradient for W on the batch. 
int micNode::forward_batch_leaky_relu(int do_grad_flag)
{
	// sanity checks
	if (batch_in.ncols != n_in + 1 || wgt.nrows != n_in + 1 || wgt.ncols != k_out + 1 || lr_params.nrows != k_out || lr_params.ncols != 2 || lambda.nrows != k_out) {
		MERR("ERROR:: micNode::forward_batch_leaky_relu : batch_in: %d x %d , wgt %d x %d , n_in %d , k_out %d , lr_params: %d x %d lambda: %d x %d\n",
			batch_in.nrows, batch_in.ncols, wgt.nrows, wgt.ncols, n_in, k_out, lr_params.nrows, lr_params.ncols, lambda.nrows, lambda.ncols);
		return -1;
	}


	//// DEBUG !!!!
	//if (do_grad_flag == 0) {
	//	for (auto w : wgt.m)
	//		if (isnan(w)) {
	//			MLOG("!!!!!!!!!!!! found a Nan Weight !!! %g :: id %d\n", w, id);
	//			exit(-1);
	//		}
	//	//micNode::init_wgts_rand_normal(0, 1);
	//}

//	MLOG("micNode::forward_batch_leaky_relu : batch_in: %d x %d , wgt %d x %d , n_in %d , k_out %d , lr_params: %d x %d lambda: %d x %d\n",
//		batch_in.nrows, batch_in.ncols, wgt.nrows, wgt.ncols, n_in, k_out, lr_params.nrows, lr_params.ncols, lambda.nrows, lambda.ncols);


	// multiplying to get W * x
//	print("debug before mult mat", 1, 6);
	if (do_grad_flag || dropout_prob_out >= 1) {
		if (fast_multiply_medmat_(batch_in, wgt, batch_out) < 0)
			return -1;
	}
	else {
		//wgt_test = wgt;
		//for (int i=0; i<n_in; i++)
		//	for (int j=0; j<k_out; j++)
		//		wgt_test(i, j) *= dropout_prob_out;
		//if (fast_multiply_medmat(batch_in, wgt_test, batch_out) < 0)
		if (fast_multiply_medmat(batch_in, wgt, batch_out, dropout_prob_out) < 0)
			return -1;
	}
	//	print("debug after mult mat", 1, 6);

		// Reminder: the last col in weights is 0,0,....0,1 
		// That with the last in each input row being 1 makes sure the output will also have 1's at the end.
	int i = 0;
	int n_b = batch_out.nrows;

	// applying leaky max func on batch_out

	if (!do_grad_flag) {
		// faster, without grad_s calculation

#pragma omp parallel for private(i) if (n_b>100 || k_out>100)
		for (i = 0; i < n_b; i++)
			for (int j = 0; j < k_out; j++)
				if (batch_out(i, j) >= 0)
					batch_out(i, j) *= lr_params(j, 0); // a
				else
					batch_out(i, j) *= lr_params(j, 1); // b
	}
	else {

		grad_s.resize(n_b, k_out + 1);

#pragma omp parallel for private(i) if (n_b>100 || k_out>100)
		for (i = 0; i < n_b; i++) {
			for (int j = 0; j < k_out; j++) {
				if (batch_out(i, j) >= 0) {
					grad_s(i, j) = lr_params(j, 0);
					batch_out(i, j) *= lr_params(j, 0); // a
				}
				else {
					grad_s(i, j) = lr_params(j, 1);
					batch_out(i, j) *= lr_params(j, 1); // b
				}
			}
			grad_s(i, k_out) = 0;
		}

		//		print("debug forward after grad step:",1,6);
	}

	return 0;
}

//.................................................................................
// The softmax func does not use the bias
// it transforms each x in the the input to
// < e^Xi / Sum(i,e^Xi) >
// We still maintain the bias columns, in order this one will be internal (oddly).
//
// if the node is marked as an output node, we can also calculate the average gradient
// of it on the batch.
// The 
int micNode::forward_batch_softmax(int do_grad_flag)
{
	int i, j;
	int n_b = batch_in.nrows;

	// sanity
	if (batch_in.ncols != n_in + 1 || k_out != n_categ || n_in != n_categ * n_per_categ) {
		MERR("ERROR:: forward_batch_softmax() :: non matching sizes :: batch_in %d x %d , n_in %d k_out %d\n", batch_in.nrows, batch_in.ncols, n_in, k_out);
		return -1;
	}

	//	MLOG("forward_batch_softmax() ::  batch_in %d x %d , n_b %d n_in %d k_out %d\n", batch_in.nrows, batch_in.ncols, n_b, n_in, k_out);

	batch_out.resize(batch_in.nrows, k_out + 1);
	full_probs.resize(batch_in.nrows, n_in + 1);
	//	print("softmax_forward before calcs:", 1, 6);
	float max_exp = (float)34;
	float min_exp = (float)-34;
	// softmax calc
//#pragma omp parallel for if (n_b>100 || n_in>100)
	for (i = 0; i < n_b; i++) {
		double sum = 0;
		float max_val = batch_in(i, 0);
		for (j = 1; j < n_in; j++)
			if (max_val < batch_in(i, j))
				max_val = batch_in(i, j);
		for (j = 0; j < n_in; j++) {
			full_probs(i, j) = min(max(batch_in(i, j) - max_val, min_exp), max_exp);
			full_probs(i, j) = (float)exp((double)full_probs(i, j));
			sum += (double)full_probs(i, j);
		}

		//for (j=0; j<n_in; j++) {
		//	batch_out(i, j) = max(epsilon,exp(min(batch_in(i, j),max_exp)));
		//	sum += batch_out(i, j);
		//}
		//MLOG("i %d %f n_b %d :: %f %f :: %f %f\n", i, sum, n_b, batch_in(i,0),batch_in(i,1),batch_out(i,0),batch_out(i,1));
		for (j = 0; j < n_in; j++)
			full_probs(i, j) = (float)(full_probs(i, j) / sum);
		full_probs(i, n_in) = 1; // bias column
	}

	if (n_per_categ == 1)
		batch_out = full_probs;
	else {

		float epsilon = (float)1e-10;
		//#pragma omp parallel for private(i) if (n_b>100 || n_in>100)
		for (i = 0; i < n_b; i++) {
			int m = 0;
			for (j = 0; j < n_in; j += n_per_categ) {
				float sum = 0;
				for (int k = j; k < j + n_per_categ; k++)
					sum += full_probs(i, k);
				batch_out(i, m++) = min(sum, (float)1 - epsilon);
			}
			if (m != n_categ) {
				MERR("ERROR: m %d n_categ %d\n", m, n_categ);
				exit(-1);
			}
			batch_out(i, k_out) = 1;
		}

	}

	//	MLOG("forward_batch_softmax() ::  before grad()\n");
	if (do_grad_flag && (!is_terminal)) { // next section is not very useful, and probably bugged...

		if (loss == "") {
			grad_s.resize(n_b, n_in);
			grad_s.zero();
			for (int k = 0; k < n_b; k++)
				for (i = 0; i < n_in; i++)
					grad_s(k, i) = full_probs(k, i) - full_probs(k, i)*full_probs(k, i);
		}


	}


	return 0;
}


//........................................................................................
// The regression func is simply here to get the gradients for least-squares regression
// There are several options:
// If n_categ is 1 - n_in must be 1 as well, and the relevant loss term is (x_in - y)^2
// If n_categ > 1 - this is a classification problem using least squares as loss
// In this case the term should sum over n_categ y's which are 1 only for the given y, and 0 otherwise.
int micNode::forward_batch_regression(int do_grad_flag)
{
	int i, j;
	int n_b = batch_in.nrows;

	// sanity
	if (batch_in.ncols != n_in + 1 || n_in != k_out) {
		MERR("ERROR:: forward_batch_regression() :: non matching sizes :: batch_in %d x %d , n_in %d k_out %d :: wgt size %d x %d\n",
			batch_in.nrows, batch_in.ncols, n_in, k_out, wgt.ncols, wgt.nrows);
		return -1;
	}


	batch_out = batch_in;

	return 0; // grad is done at the back_prop stage


	if (do_grad_flag) {
		float fact = 1 / (float)n_b; // normalizing for cases of different batch sizes

		if (n_in == 1) {

			// classical regression problem
			delta.resize(n_in, 1);
			for (i = 0; i < n_b; i++)
				delta(i, 0) = fact * (batch_in(i, 0) - y(i, 0));

		}
		else if (n_in > 1) {

			// multi-categ embedded regression problem
			if (y.ncols == 1) {
				// y is given as the value
				for (i = 0; i < n_b; i++) {
					for (j = 0; j < n_in; j++)
						delta(i, j) = fact * batch_in(i, j);
					delta(i, (int)y(i, 0)) -= fact;
				}

			}
			else {
				// y is given as a vector
				for (i = 0; i < n_b; i++) {
					for (j = 0; j < n_in; j++)
						delta(i, j) = fact * (batch_in(i, j) - y(i, j));
				}

			}
		}
	}

	batch_out = batch_in;


	return 0;
}


//.................................................................................
// A normalization layer simply runs an affine transform on the data in each coordinate
// The number of pramaters (non 0 elements in wgt) is 2*n_in
// The alphas are sitting at the diagonal, and the betas are at the bias term
// Thus an input x_ij is transformed to output: alpha_j*x_ij + beta_j
// The general propagation is just like leaky relu with the following changes:
// (1) Most of the wgt matrix is 0's.
// (2) The grad_s is always 1
// (3) We need to add more terms to push the average per column to 0, and the variance to 1... 
//     These will be added like regularization terms in the back_prop stage. We can also learn without it, 
//     But it is preffered to learn ONLY with it... That is to force the network towards the N(0,1) distribution
int micNode::forward_batch_normalization(int do_grad_flag)
{
	// sanity checks
	if (batch_in.ncols != n_in + 1 || wgt.nrows != n_in + 1 || wgt.ncols != k_out + 1 || n_in != k_out) {
		MERR("ERROR:: micNode::forward_batch_normalization : batch_in: %d x %d , wgt %d x %d , n_in %d , k_out %d , lr_params: %d x %d lambda: %d x %d\n",
			batch_in.nrows, batch_in.ncols, wgt.nrows, wgt.ncols, n_in, k_out, lr_params.nrows, lr_params.ncols, lambda.nrows, lambda.ncols);
		return -1;
	}

	//	MLOG("micNode::forward_batch_leaky_relu : batch_in: %d x %d , wgt %d x %d , n_in %d , k_out %d , lr_params: %d x %d lambda: %d x %d\n",
	//		batch_in.nrows, batch_in.ncols, wgt.nrows, wgt.ncols, n_in, k_out, lr_params.nrows, lr_params.ncols, lambda.nrows, lambda.ncols);


	// multiplying to get W * x - leaving this code, although it should be made faster by multiplying a diagonal matrix.
	if (1) { //do_grad_flag || dropout_prob_out >= 1) {
		int n_b = batch_in.nrows;
		batch_out.resize(n_b, n_in + 1);
		if (alpha.size() < n_in) { // this init should move to initialization code.
			alpha.resize(n_in, (float)1);
			alpha.push_back(0);
			beta.resize(n_in + 1, 0);
		}

		//#pragma omp parallel for
				//MLOG("## id %d [0] alpha %f beta %f\n", id, alpha[0], beta[0]);
		for (int i = 0; i < n_b; i++) {
			for (int j = 0; j < n_in; j++) {
				batch_out(i, j) = batch_in(i, j) * alpha[j] + beta[j];
			}
			batch_out(i, n_in) = 1;
		}

		//if (fast_multiply_medmat(batch_in, wgt, batch_out) < 0)
		//	return -1;
	}
	else {
		//wgt_test = wgt;
		//for (int i=0; i<n_in; i++)
		//	for (int j=0; j<k_out; j++)
		//		wgt_test(i, j) *= dropout_prob_out;
		if (fast_multiply_medmat(batch_in, wgt, batch_out, dropout_prob_out) < 0)
			return -1;
	}

	// applying leaky max func on batch_out

	if (do_grad_flag) {

		grad_s.clear();
		if (grad_s.size() == 0) {
			//grad_s.resize(n_b, k_out+1);
			//fill(grad_s.m.begin(), grad_s.m.end(), (float)1);
		}

	}

	return 0;
}


void micNode::forward_batch(const vector<MedMat<float>> &nodes_outputs, MedMat<float> &out) const
{

	int j = 0;
	if (ir.n_input_nodes == 1 && ir.mode[0] == "all") {
		j = ir.in_node_id[0];
	}
	else
		MWARN("warning n_input_nodes != 1");
	const MedMat<float> &in = nodes_outputs[j];

	//calculate current layer on in as input
	if (type == "Input") {
		out = std::move(in);
		return;
	}

	if (type == "LeakyReLU") {
		forward_batch_leaky_relu(in, out);
		return;
	}

	if (type == "SoftMax") {
		forward_batch_softmax(in, out);
		return;
	}

	if (type == "Regression") {
		forward_batch_regression(in, out);
		return;
	}

	if (type == "Normalization") {
		forward_batch_normalization(in, out);
		return;
	}

	MTHROW_AND_ERR("ERROR:: micNode::forward_batch() :: no such node type : %s \n", type.c_str());
}

//.................................................................................
// runs the node forward on the current batch_in
// options for the node are:
// "Input"     :: simply copies batch to output
// "LeakyReLU" :: running a leaky rectified linear unit on input based on internal params for it
// "Linear"    :: Linear model 
// "Logistic"  :: Logistic func
// "SoftMax"   :: SoftMax functional
// "Output"    :: simply copy batch to output
//
// Note that the last column in batch should always be the bias ("1"), so depending on case we will use it or not
//
// while doing the forward sweep we are also able to calculate the gradient (this is needed in train time)
// to do that use do_grad_flag != 0
//
int micNode::forward_batch(int do_grad_flag)
{
	if (get_input_batch(do_grad_flag) < 0)
		return -1;

	if (type == "Input") {
		batch_out = batch_in;
		return 0;
	}

	if (type == "LeakyReLU")
		return(forward_batch_leaky_relu(do_grad_flag));

	if (type == "SoftMax")
		return(forward_batch_softmax(do_grad_flag));

	if (type == "Regression")
		return(forward_batch_regression(do_grad_flag));

	if (type == "Normalization")
		return(forward_batch_normalization(do_grad_flag));

	MERR("ERROR:: micNode::forward_batch() :: no such node type : %s \n", type.c_str());
	return -1;
}

//.................................................................................................
// gets the back_prop delta for the last Loss layer
int micNode::get_backprop_delta()
{
	int i, j, k;
	int n_b = batch_in.nrows;
	float fact = (float)1 / (float)n_b; // normalizing for cases of different batch sizes

	if (loss == "log") {

		// this is a log loss softmax output node , batch_out has the probabilities, we get the initial delta to propagate
		delta.resize(n_b, n_in + 1);
		delta.zero();

		//#pragma omp parallel for private(k) if (n_b>100 || n_in>100)
		if (n_per_categ == 1) {
			for (k = 0; k < n_b; k++) {
				for (int j = 0; j < n_in; j++)
					delta(k, j) = fact * batch_out(k, j);
				delta(k, (int)y(k, 0)) -= fact;
			}
			if (sweights.size() != 0)
				for (k = 0; k < n_b; k++)
					for (int j = 0; j < n_in; j++)
						delta(k, j) *= sweights[k];
		}
		else {

			for (i = 0; i < n_b; i++) {
				for (j = 0; j < n_in; j++)
					delta(i, j) = fact * full_probs(i, j);
				int k = (int)y(i, 0);
				float epsilon = (float)1e-5;
				float p_J = fact / (batch_out(i, k) + epsilon);
				for (j = k * n_per_categ; j < (k + 1)*n_per_categ; j++) {
					delta(i, j) -= p_J * (full_probs(i, j) + epsilon);
				}
			}

		}
		//delta(k,(int)y(k, 0)) = batch_out(k, (int)y(k, 0)) - (float)1;

		//			print("debug backprop after delta in logloss mode",1,6);

	}

	if (loss == "lsq") {
		delta.resize(n_b, n_in + 1);
		delta.zero();

		if (n_in == 1) {

			// classical regression problem
			for (i = 0; i < n_b; i++)
				delta(i, 0) = fact * (batch_in(i, 0) - y(i, 0));
			if (sweights.size() != 0)
				for (i = 0; i < n_b; i++)
					delta(i, 0) *= sweights[i];


		}
		else if (n_in > 1) {


			// multi-categ embedded regression problem
			if (y.ncols == 1) {
				// y is given as the value --> this turns out to be exactly like the log loss...
				for (i = 0; i < n_b; i++) {
					for (j = 0; j < n_in; j++)
						delta(i, j) = fact * batch_in(i, j);
					delta(i, (int)y(i, 0)) -= fact;
				}

			}
			else {
				// y is given as a vector
				for (i = 0; i < n_b; i++) {
					for (j = 0; j < n_in; j++)
						delta(i, j) = fact * (batch_in(i, j) - y(i, j));
				}

			}
		}
	}
	return 0;
}


//.................................................................................................
// runs back propagation on the node given the next node (assuming it was already back propagated)
int micNode::back_propagete_from(micNode *next)
{
	int i;

	// Propagates the gradients backwards from "next" node to this one.
	// If next is NULL, we are at a terminal node (with Loss applied).
	// Propagation sums over all 

	if (next == NULL || next->ir.n_input_nodes == 1) {

		//		print("debug back_propagate_from start", 1, 6);
		int n_b = batch_out.nrows;

		// single input case
		if (loss != "") get_backprop_delta();

		if (loss == "") {

			// using the recursion for delta
			// delta = grad_s (dot) next->delta * next->W (if there is W)
			// if next is the loss, there's only delta = grad_s (dot) next->delta

			if (next->loss != "") {

				delta.resize(n_b, k_out + 1);
				if (grad_s.size() > 0) { // this condition saves computation when all grad_s are 1 (as in normalization layers).

					fast_element_dot_vector_vector(next->delta.get_vec(), grad_s.get_vec(), delta.get_vec());
					//#pragma omp parallel for private(i) if (n_b>100 || k_out>100)
//					for (i=0; i<n_b; i++)
//						for (int j=0; j<k_out; j++)
//							delta(i, j) = grad_s(i, j) * next->delta(i, j);
				}
				else
					delta = next->delta;
				//#pragma omp parallel for private(i)
				for (i = 0; i < n_b; i++)
					delta(i, k_out) = 0;

			}
			else {

				if (fast_multiply_medmat_transpose(next->delta, next->wgt, delta, 0x2) < 0)
					return -1;

				if (grad_s.size() > 0)
					fast_element_dot_vector_vector(delta.get_vec(), grad_s.get_vec());
				//#pragma omp parallel for private(i) if (n_b>100 || k_out>100)
				//					for (i=0; i<n_b; i++)
				//						for (int j=0; j<k_out; j++)
				//							delta(i, j) *= grad_s(i, j);
				//#pragma omp parallel for private(i)
				for (i = 0; i < n_b; i++)
					delta(i, k_out) = 0;

			}

			if (type != "Normalization") {
				// actual grad_w calculation
				if (fast_multiply_medmat_transpose(batch_in, delta, grad_w, 0x1) < 0)
					return -1;

				// grad_w = grad_w + lambda * wgt
				fast_element_affine_scalar(grad_w.get_vec(), lambda(0, 0), wgt.get_vec());
				//#pragma omp parallel for private(i) if (n_in>100)
				//				for (i=0; i<n_in; i++)
				//					for (int j=0; j<k_out; j++)
				//						grad_w(i, j) = grad_w(i, j) + lambda(j, 0)*wgt(i, j);

				for (i = 0; i < n_in + 1; i++)
					grad_w(i, k_out) = 0;  // last column in wgts is 0,0...0,1 to make it easy to forward bias. Hence we have grad 0 for it.
			}

		}
		//		print("debug back_propagate_from after propagation", 1, 6);

	}
	else {
		MERR("micNode::propagate_from multi nodes input not supported yet...(%d)\n", next->ir.n_input_nodes);
		return -1;
	}

	return 0;
}

//........................................................................................................
// here we calculate an approximation to the mean and variance, taking past mean and variance into account
// This is done simply by approximations of the type m_new = r * m_old + (1-r) * m_estimation.
// We need to take care of dropouts correctly.
int micNode::weights_normalization_step()
{
	int i, j;

	int n_b = batch_out.nrows;

	curr_mean.resize(n_in);
	curr_var.resize(n_in);
	fill(curr_mean.begin(), curr_mean.end(), (double)0);
	fill(curr_var.begin(), curr_var.end(), (double)0);

	if (b_mean.size() < n_in) {
		b_mean.resize(n_in, 0);
		b_var.resize(n_in, (float)1);
		alpha.resize(n_in, 1);
		beta.resize(n_in, 0);
	}

	// calculating current estimators
	for (i = 0; i < n_b; i++)
		for (j = 0; j < n_in; j++)
			if (dropout_prob_in >= 1 || dropout_in[j]) {
				curr_mean[j] += batch_out(i, j);
				curr_var[j] += static_cast<double>(batch_out(i, j)) * static_cast<double>(batch_out(i, j));
			}

	double fact = (double)1 / (double)n_b;
	double epsilon = 1e-2;
	for (j = 0; j < n_in; j++) {
		curr_mean[j] *= fact;
		curr_var[j] *= fact;
		curr_var[j] -= curr_mean[j] * curr_mean[j];
		curr_var[j] += epsilon;
	}

	// updating prev estimates and actual alpha/beta
	float one_minus_factor = (float)1 - normalization_update_factor;
	float eps = (float)1e-2;
	for (j = 0; j < n_in; j++)
		if (dropout_prob_in >= 1 || dropout_in[j]) {
			b_mean[j] = normalization_update_factor * b_mean[j] + one_minus_factor * (float)curr_mean[j];
			b_var[j] = normalization_update_factor * b_var[j] + one_minus_factor * (float)curr_var[j];
			float std = sqrt(b_var[j] + eps);
			alpha[j] = wgt(j, j) = (float)1 / std;
			beta[j] = wgt(n_in, j) = -(float)b_mean[j] / std;
		}

	//MLOG("Node id: %d [0] mean %f var %f alpha %f beta %f\n", id, b_mean[0], b_var[0], alpha[0], beta[0]);
	return 0;
}

//.................................................................................................
// weights_gd_step - 
// using the learning rates and the ready gradients to make a gradient descent step
// this step also updates the prev grad in order to be able to use momentum
int micNode::weights_gd_step()
{
	if (type == "Normalization")
		return (weights_normalization_step());
	//	print("weights gd_step before", 1, 6);
		// first update prev_grad_w

	if (dropout_prob_in < 1 || dropout_prob_out < 1) {
		// if that is the case we make sure the gradients are 0 for cases in which there was no input or no output
#pragma omp parallel for if(n_in>100)
		for (int i = 0; i < n_in + 1; i++)
			for (int j = 0; j < k_out; j++)
				grad_w(i, j) *= dropout_in[i] * dropout_out[j];
	}

	if (prev_grad_w.size() != grad_w.size()) {
		prev_grad_w = grad_w;
	}
	else {
		float one_minus_momentum = (float)1 - momentum;
		fast_element_affine_scalar(momentum, prev_grad_w.get_vec(), one_minus_momentum, grad_w.get_vec());
		//#pragma omp parallel for private(i) if(n_in>100)
		//		for (i=0; i<n_in+1; i++) {
		//			for (int j=0; j<k_out; j++)
		//				prev_grad_w(i, j) = momentum*prev_grad_w(i, j) + one_minus_momentum*grad_w(i, j);
		//		}

	}
	//	print("weights gd_step middle", 1, 6);

		// now do the gradient descent step
	//#pragma omp parallel for private(i) if(n_in>100)
		// wgt = wgt - lr*prev_grad_w
	fast_element_affine_scalar(wgt.get_vec(), -rate_factor * learn_rates(0, 0), prev_grad_w.get_vec());
	//for (i=0; i<n_in+1; i++) {
	//	for (int j=0; j<k_out; j++)
	//		wgt(i, j) = wgt(i, j) - rate_factor*learn_rates(j, 0)*prev_grad_w(i, j);
	//}

	if (sparse_zero_prob > 0) {
#pragma omp parallel for if(n_in>100 || k_out>100)
		for (int i = 0; i < n_in; i++)
			for (int j = 0; j < k_out; j++)
				if (sparse_bit(i, j) == 0)
					wgt(i, j) = 0;
	}

	// if max_norm needed - rescale weights
	if (max_wgt_norm > 0 || min_wgt_norm > 0) {
		float epsilon = (float)1e-3;
		vector<float> sum_sq(k_out, 0);
#pragma omp parallel for if(n_in>100 || k_out>100)
		for (int i = 0; i <= n_in; i++)
			for (int j = 0; j < k_out; j++)
				sum_sq[j] += wgt(i, j)*wgt(i, j);

#pragma omp parallel for if(n_in>100 || k_out>100)
		for (int j = 0; j < k_out; j++) {
			if (max_wgt_norm > 0 && sum_sq[j] > max_wgt_norm*max_wgt_norm + epsilon) {
				float fact_max = sqrt((float)max_wgt_norm / sum_sq[j]);
				for (int i = 0; i <= n_in; i++)
					wgt(i, j) *= fact_max;
			}
			//if (min_wgt_norm > 0 && sum_sq[j] < min_wgt_norm*min_wgt_norm - epsilon) {
			//	float fact_min = (float)sqrt(min_wgt_norm/(epsilon+sum_sq[j]));
			//	for (int i=0; i<=n_in; i++)
			//		wgt(i, j) *= fact_min;
			//}
		}
	}

	//	print("weights gd_step after", 1, 6);
	return 0;
}

//.................................................................................................
void micNode::print(const string &prefix, int i_state, int i_in)
{
	//	if (!is_terminal)
	return;

	int print_y = 1;

	MLOG("%s node %d : %s : batch_in %d x %d : wgt %d x %d : batch_out %d x %d : y %d x %d : lr_params %d x %d : lambda %d x %d : learn_rates %d x %d : grad_w %d x %d : prev_grad_w %d x %d : delta %d x %d : momentum %f\n",
		prefix.c_str(), id, type.c_str(),
		batch_in.nrows, batch_in.ncols, wgt.nrows, wgt.ncols, batch_out.nrows, batch_out.ncols, y.nrows, y.ncols,
		lr_params.nrows, lr_params.ncols, lambda.nrows, lambda.ncols, learn_rates.nrows, learn_rates.ncols,
		grad_w.nrows, grad_w.ncols, prev_grad_w.nrows, prev_grad_w.ncols, delta.nrows, delta.ncols, momentum);

	if (0) { //i_state >= 0 && i_state < wgt.ncols) {
		MLOG("%s node %d : state %d : lr_params %f %f : lambda %f : learn_rate %f\n",
			prefix.c_str(), id, i_state, lr_params(i_state, 0), lr_params(i_state, 1), lambda(i_state, 0), learn_rates(i_state, 0));
		MLOG("%s node %d : wgts(:,%d) :", prefix.c_str(), id, i_state);
		for (int i = 0; i < wgt.nrows; i++)
			for (int j = 0; j < wgt.ncols; j++)
				MLOG(" (%d,%d) %5.3f", i, j, wgt(i, j));
		//		MLOG(" (%d) %5.3f", i, wgt(i, i_state));
		MLOG("\n");
	}

	if (i_state >= 0 && i_state < grad_s.ncols) {
		MLOG("%s node %d : grad_s(:,%d) :", prefix.c_str(), id, i_state);
		for (int i = 0; i < grad_s.nrows; i++)
			MLOG(" (%d) %5.3f", i, grad_s(i, i_state));
		MLOG("\n");
	}

	if (i_state >= 0 && i_state < delta.ncols) {
		MLOG("%s node %d : delta(:,%d) :", prefix.c_str(), id, i_state);
		for (int i = 0; i < delta.nrows; i++)
			for (int j = 0; j < delta.ncols; j++)
				MLOG(" (%d,%d) %5.3f", i, j, delta(i, j));
		//		MLOG(" (%d) %5.3f", i, delta(i, i_state));
		MLOG("\n");
	}

	if (0) { //i_state >= 0 && i_state < grad_w.ncols) {
		MLOG("%s node %d : grad_w(:,%d) :", prefix.c_str(), id, i_state);
		for (int i = 0; i < grad_w.nrows; i++)
			for (int j = 0; j < grad_w.ncols; j++)
				MLOG(" (%d,%d) %5.3f", i, j, grad_w(i, j));
		//		MLOG(" (%d) %5.3f", i, grad_w(i, i_state));
		MLOG("\n");
	}

	if (0) { //i_state >= 0 && i_state < prev_grad_w.ncols) {
		MLOG("%s node %d : prev_grad_w(:,%d) :", prefix.c_str(), id, i_state);
		for (int i = 0; i < prev_grad_w.nrows; i++)
			for (int j = 0; j < prev_grad_w.ncols; j++)
				MLOG(" (%d,%d) %5.3f", i, j, prev_grad_w(i, j));
		//MLOG(" (%d) %5.3f", i, prev_grad_w(i, i_state));
		MLOG("\n");
	}

	if (0) { //i_in >= 0 && i_in < batch_in.nrows) {
		MLOG("%s node %d : batch_in(%d,:) :", prefix.c_str(), id, i_in);
		for (int i = 0; i < batch_in.ncols; i++)
			MLOG(" (%d) %5.3f", i, batch_in(i_in, i));
		MLOG("\n");
	}

	if (i_in >= 0 && i_in < batch_out.nrows) {
		for (int j = 0; j < 10; j++) {//batch_out.nrows; j++)
			MLOG("%s node %d : batch_out(%d,:) :", prefix.c_str(), id, i_in);
			for (int i = 0; i < batch_out.ncols; i++)
				MLOG(" (%d,%d) %5.3f", j, i, batch_out(j, i));
			//		MLOG(" (%d) %5.3f", i, batch_out(i_in, i));
			MLOG("\n");
		}
	}
	if (i_in >= 0 && i_in < y.nrows) {
		MLOG("%s node %d : y(%d) : %f\n", prefix.c_str(), id, i_state, y(i_in, 0));
	}


	if (i_in >= 0 && i_in < full_probs.nrows) {
		for (int j = 0; j < 10; j++) { //full_probs.nrows; j++)
			MLOG("%s node %d : full_probs(%d,:) :", prefix.c_str(), id, i_in);
			for (int i = 0; i < full_probs.ncols; i++)
				MLOG(" (%d,%d) %5.3f", j, i, full_probs(j, i));
			//		MLOG(" (%d) %5.3f", i, batch_out(i_in, i));
			MLOG("\n");

		}
	}

	if (print_y && y.size() > 0) {
		MLOG("%s node %d : y(:) :", prefix.c_str(), id);
		for (int i = 0; i < y.nrows; i++)
			MLOG(" (%d) %3.1f", i, y(i, 0));
		MLOG("\n");
	}
}

//.................................................................................................
void InputRules::push(int node_id, const string &_mode)
{
	in_node_id.push_back(node_id);
	mode.push_back(_mode);
	n_input_nodes++;
}

//.................................................................................................
int micNetParams::init_from_string(const string &init_str)
{
	params_init_string = string(init_str);

	vector<string> fields;
	boost::split(fields, init_str, boost::is_any_of("=,;:"));

	n_hidden.clear();
	dropout_in_probs.clear();
	samp_ratio.clear();

	for (int i = 0; i < fields.size(); i++) {

		//cerr << "parsing i " << i << " f[i] " << fields[i] << " f[i+1] " << fields[i+1] << "\n";
		//! [micNetParams::init_from_string]
		if (fields[i] == "A") def_A = stof(fields[++i]);
		if (fields[i] == "B") def_B = stof(fields[++i]);
		if (fields[i] == "lambda") def_lambda = stof(fields[++i]);
		if (fields[i] == "momentum") def_momentum = stof(fields[++i]);
		if (fields[i] == "batch_size") batch_size = stoi(fields[++i]);
		if (fields[i] == "max_norm") max_wgt_norm = stof(fields[++i]);
		if (fields[i] == "min_norm") min_wgt_norm = stof(fields[++i]);
		if (fields[i] == "wgt_std") weights_init_std = stof(fields[++i]);
		if (fields[i] == "rate_decay") rate_decay = stof(fields[++i]);
		if (fields[i] == "n_categ") n_categ = stoi(fields[++i]);
		if (fields[i] == "n_per_categ") n_per_categ = stoi(fields[++i]);
		if (fields[i] == "nfeat") nfeat = stoi(fields[++i]);
		if (fields[i] == "n_norm") n_norm_layers = stoi(fields[++i]);
		if (fields[i] == "norm_facor") normalization_factor = stof(fields[++i]);
		if (fields[i] == "sparse") sparse_zero_prob = stof(fields[++i]);
		if (fields[i] == "loss_type") loss_type = fields[++i];
		if (fields[i] == "net_type") net_type = fields[++i];
		if (fields[i] == "min_epochs") min_epochs = stoi(fields[++i]);
		if (fields[i] == "max_epochs") max_epochs = stoi(fields[++i]);
		if (fields[i] == "n_back") n_back = stoi(fields[++i]);
		if (fields[i] == "min_improve") min_improve_n_back = stof(fields[++i]);
		if (fields[i] == "n_preds_per_sample") n_preds_per_sample = stoi(fields[++i]);
		if (fields[i] == "pred_class") pred_class = stoi(fields[++i]);
		if (fields[i] == "last_layer") last_layer_to_keep = stoi(fields[++i]);
		//if (fields[i] == "learning_rate") def_learning_rate = stof(fields[++i]);

		learning_rates.resize(500, def_learning_rate);
		if (fields[i] == "learning_rate") {
			string s = fields[++i];
			vector<string> f;
			boost::split(f, s, boost::is_any_of("-/#"));
			int k = 0;
			for (int j = 0; j < f.size(); j++)
				learning_rates[k++] = stof(f[j]);
			if (k > 0) {
				for (int j = k; j < learning_rates.size(); j++)
					learning_rates[j] = learning_rates[k - 1];
			}
		}

		if (fields[i] == "hidden") {
			string s = fields[++i];
			vector<string> f;
			boost::split(f, s, boost::is_any_of("-/#"));
			for (int j = 0; j < f.size(); j++)
				n_hidden.push_back(stoi(f[j]));
		}
		if (fields[i] == "dropout") {
			string s = fields[++i];
			vector<string> f;
			boost::split(f, s, boost::is_any_of("-/#"));
			for (int j = 0; j < f.size(); j++)
				dropout_in_probs.push_back(stof(f[j]));
		}

		if (fields[i] == "samp_ratio") {
			string s = fields[++i];
			vector<string> f;
			boost::split(f, s, boost::is_any_of("-/#"));
			for (int j = 0; j < f.size(); j++)
				samp_ratio.push_back(stof(f[j]));
		}
		//! [micNetParams::init_from_string]
		//if (fields[i] == "nodes") {

		//	string s = fields[++i];
		//	vector<string> f;
		//	boost::split(f, s, boost::is_any_of(">"));
		//	for (auto snode : f) {
		//		if (snode.size() > 1) {
		//			NodeInfo ni;
		//			ni.init_from_string(snode);
		//			node_infos.push_back(ni);
		//		}
		//		
		//	}
		//	

		//}
	}

	if (1) { //n_hidden.size() == 0) {
		n_hidden.push_back(n_categ*n_per_categ); // default is a very simple single layer
	}

	if (dropout_in_probs.size() != n_hidden.size()) {

		if (dropout_in_probs.size() > n_hidden.size())
			dropout_in_probs.resize(n_hidden.size());
		else {
			for (int j = (int)dropout_in_probs.size(); j < (int)n_hidden.size(); j++)
				dropout_in_probs.push_back((float)1);
		}

	}
	if (n_preds_per_sample < 0)
		n_preds_per_sample = n_categ * n_per_categ;

	return 0;
}
//
////............................................................................................................
//int micNetParams::node_infos_init_finish()
//{
//	for (int i=0; i<node_infos.size(); i++) {
//
//		NodeInfo *ni = &node_infos[i];
//		if (ni->id != i) {
//			MERR("micNet ERROR: Wrong id number for node %d :: %d\n", i, ni->id);
//			return -1;
//		}
//
//		if (ni->n_hidden == 0) {
//			MERR("micNet ERROR: No hidden states for node %d\n", i);
//			return -1;
//		}
//
//		if (ni->sources.size() == 0 && ni->type != "Input") {
//			MERR("micNet ERROR: No sources for node %d\n", i);
//			return -1;
//		}
//
//		if (ni->learn_rate <= 0) ni->learn_rate = def_learning_rate;
//		if (ni->momentum < 0) ni->momentum = def_momentum;
//		if (ni->rate_decay < 0) ni->rate_decay = rate_decay;
//		if (ni->drop_in < 0) ni->drop_in = 0;
//		if (ni->A < 0) ni->A = def_A;
//		if (ni->B < 0) ni->B = def_B;
//		if (ni->lambda < 0) ni->lambda = def_lambda;
//		if (ni->noise_std < 0) ni->noise_std = 0;
//		if (ni->wgt_std < 0) ni->wgt_std = weights_init_std;
//
//		// setting sinks from sources
//		for (auto in : ni->sources)
//			node_infos[in].sinks.push_back(i);
//
//		// getting input and output sizes (without bias)
//		ni->in_dim = 0;
//		ni->out_dim = ni->n_hidden;
//		for (auto in : ni->sources)
//			ni->in_dim += node_infos[in].n_hidden;
//
//	}
//
//	// TBD: sanity tests (dimensions match to type , etc).
//
//	return 0;
//}

//=================================================================================
// micNet
//=================================================================================
//.................................................................................................
int micNet::init_fully_connected(const string &init_str)
{
	params.init_defaults();
	params.init_from_string(init_str);
	return (init_fully_connected(params));
}

//.................................................................................................
// auto choose initialization (using the net_type param)
int micNet::init_net(const string &init_str)
{
	params.init_defaults();
	params.init_from_string(init_str);
	return init_net(params);

}

//.................................................................................................
int micNet::init_net(micNetParams &in_params)
{
	if (params.net_type == "fc")
		return init_fully_connected(params);
	if (params.net_type == "autoencoder")
		return init_fully_connected(params);
	//	return init_autoencoder(params);


	return -1;
}

//.................................................................................................
int micNet::add_input_layer()
{
	micNode node;

	// set Input node
	node.id = (int)nodes.size();
	node.name = "Input Layer";
	node.type = "Input";
	node.n_in = params.nfeat;
	node.k_out = params.nfeat;
	node.forward_nodes.push_back(1);
	node.is_terminal = 0;
	node.dropout_prob_in = 1;
	node.dropout_prob_out = 1;

	nodes.push_back(node);

	return 0;
}

//.................................................................................................
int micNet::add_fc_leaky_relu_layer(int in_node, int n_hidden, float dropout_in_p, float sparse_prob, float learn_rate)
{
	if (nodes.size() == 0) {
		MERR("ERROR:: micNet::add_fc_leaky_relu_layer :: can't add this layer as first\n");
		return -1;
	}

	micNode node;
	micNode *prev_node = &nodes[in_node];

	node.id = (int)nodes.size();
	node.name = "Hidden Layer : " + to_string(n_hidden) + " LeakyReLU states";
	node.type = "LeakyReLU";
	node.n_in = prev_node->k_out;
	node.k_out = n_hidden;
	node.ir.push(prev_node->id, "all");
	float std = params.weights_init_std;
	if (params.weights_init_std == 0)
		std = sqrt((float)2 / (float)node.n_in);
	node.init_wgts_rand_normal(0, std);
	node.forward_nodes.push_back(node.id + 1);
	node.is_terminal = 0;
	node.lr_params.resize(n_hidden, 2);
	node.lambda.resize(n_hidden, 1);
	node.learn_rates.resize(n_hidden, 1);

	node.lambda.set_val(params.def_lambda);
	node.learn_rates.set_val(learn_rate);

	for (int j = 0; j < n_hidden; j++) {
		node.lr_params(j, 0) = params.def_A;
		node.lr_params(j, 1) = params.def_B;
		//node.lambda(j, 0) = params.def_lambda;
		////node.learn_rates(j, 0) = params.def_learning_rate;
		//node.learn_rates(j, 0) = learn_rate;
	}

	node.momentum = params.def_momentum;
	node.rate_factor = (float)1.0;
	node.max_wgt_norm = params.max_wgt_norm;
	node.min_wgt_norm = params.min_wgt_norm;

	prev_node->dropout_prob_out = dropout_in_p;
	node.dropout_prob_in = dropout_in_p;
	node.dropout_prob_out = 1;

	node.dropout_out.resize(node.k_out + 1, 1);
	node.dropout_in.resize(node.n_in + 1, 1);

	node.sparse_zero_prob = sparse_prob;

	if (node.sparse_zero_prob > 0) {
		node.sparse_bit.resize(node.n_in + 1, node.k_out + 1);

		for (int i = 0; i < node.n_in; i++) {
			for (int j = 0; j < node.k_out; j++) {
				if (rand_1() >= node.sparse_zero_prob)
					node.sparse_bit(i, j) = 1;
				else {
					node.sparse_bit(i, j) = 0;
					node.wgt(i, j) = 0;
				}
				node.sparse_bit(i, node.k_out) = 1;
			}
			for (int j = 0; j <= node.k_out; j++) {
				node.sparse_bit(node.n_in, j) = 1;
			}
		}

	}

	nodes.push_back(node);
	return 0;
}

//.................................................................................................
int micNet::add_normalization_layer(int in_node)
{
	if (nodes.size() == 0) {
		MERR("ERROR:: micNet::add_normalization_layer :: can't add this layer as first\n");
		return -1;
	}

	micNode node;
	micNode *prev_node = &nodes[in_node];

	int n_hidden = prev_node->k_out;

	node.id = (int)nodes.size();
	node.name = "Hidden Layer : " + to_string(n_hidden) + " Normalization states";
	node.type = "Normalization";
	node.n_in = prev_node->k_out;
	node.k_out = n_hidden;
	node.ir.push(node.id - 1, "all");

	// initializing wgt matrix to be a unit matrix
	node.wgt.resize(n_hidden + 1, n_hidden + 1);
	node.wgt.zero();
	for (int i = 0; i < n_hidden + 1; i++)
		node.wgt(i, i) = 1;

	node.forward_nodes.push_back(node.id + 1);
	node.is_terminal = 0;

	node.normalization_update_factor = params.normalization_factor;

	node.dropout_prob_in = prev_node->dropout_prob_out;
	node.dropout_prob_out = prev_node->dropout_prob_out;

	node.dropout_out.resize(node.k_out + 1, 1);
	node.dropout_in.resize(node.n_in + 1, 1);

	nodes.push_back(node);
	return 0;
}

//.................................................................................................
int micNet::add_softmax_output_layer(int in_node)
{
	if (nodes.size() == 0) {
		MERR("ERROR:: micNet::add_softmax_output_layer :: can't add this layer as first\n");
		return -1;
	}

	micNode node;
	micNode *prev_node = &nodes[in_node];

	int n_hidden = prev_node->k_out;

	if (n_hidden != params.n_categ*params.n_per_categ) {
		MERR("ERROR:: micNet::add_softmax_output_layer :: prev layer states %d do not match n_categ %d x %d\n", n_hidden, params.n_categ, params.n_per_categ);
		return -1;
	}

	node.n_categ = params.n_categ;
	node.n_per_categ = params.n_per_categ;
	node.id = (int)nodes.size();
	node.name = "Output SoftMax/Loss state";
	node.type = "SoftMax";
	node.loss = params.loss_type;
	node.n_in = params.n_categ * params.n_per_categ;
	node.k_out = params.n_categ;
	node.ir.push(prev_node->id, "all");
	node.is_terminal = 1;
	node.dropout_prob_in = 1;
	node.dropout_prob_out = 1;

	node.dropout_out.resize(node.k_out + 1, 1);
	node.dropout_in.resize(node.n_in + 1, 1);

	nodes.push_back(node);
	return 0;
}

//.................................................................................................
int micNet::add_regression_output_layer(int in_node)
{
	if (nodes.size() == 0) {
		MERR("ERROR:: micNet::add_regression_output_layer :: can't add this layer as first\n");
		return -1;
	}

	micNode node;
	micNode *prev_node = &nodes[in_node];

	int n_hidden = prev_node->k_out;

	if (params.net_type == "fc") {
		if (n_hidden != params.n_categ*params.n_per_categ || params.n_per_categ != 1) {
			MERR("ERROR:: micNet::add_regression_output_layer :: prev layer states %d do not match n_categ %d x %d\n", n_hidden, params.n_categ, params.n_per_categ);
			return -1;
		}
	}
	if (params.net_type == "autoencoder") {
		params.n_categ = n_hidden;
		params.n_per_categ = 1;
	}


	node.n_categ = params.n_categ;
	node.n_per_categ = params.n_per_categ;
	node.id = (int)nodes.size();
	node.name = "Regression lsq loss layer";
	node.type = "Regression";
	node.loss = params.loss_type;
	node.n_in = params.n_categ * params.n_per_categ;
	node.k_out = params.n_categ;
	node.ir.push(prev_node->id, "all");
	node.is_terminal = 1;
	node.dropout_prob_in = 1;
	node.dropout_prob_out = 1;

	node.dropout_out.resize(node.k_out + 1, 1);
	node.dropout_in.resize(node.n_in + 1, 1);

	nodes.push_back(node);
	return 0;
}

//.................................................................................................
int micNet::add_autoencoder_loss(int in_node, int data_node)
{
	if (nodes.size() <= 1) {
		MERR("ERROR:: micNet::add_autoencoder_loss :: can't add this layer as first or second\n");
		return -1;
	}

	micNode node;
	micNode *prev_node = &nodes[in_node];

	node.data_node = data_node;
	node.data_node_p = &nodes[data_node];

	int n_hidden = nodes[data_node].k_out;

	if (n_hidden != params.n_categ) {
		MERR("ERROR:: micNet::add_softmax_output_layer :: prev layer states %d do not match n_categ %d\n", n_hidden, params.n_categ);
		return -1;
	}

	node.id = (int)nodes.size();
	node.name = "Autoencoder regression loss";
	node.type = "Regression";
	node.loss = "lsq";
	node.n_in = prev_node->k_out;
	node.k_out = n_hidden;
	node.ir.push(prev_node->id, "all");
	node.is_terminal = 1;
	node.dropout_prob_in = 1;
	node.dropout_prob_out = 1;

	node.dropout_out.resize(node.k_out + 1, 1);
	node.dropout_in.resize(node.n_in + 1, 1);

	nodes.push_back(node);
	return 0;
}


//.................................................................................................
// one option to initialize a micNet :: fully-connected , softmax, log-loss
//int micNet::init_fully_connected(const string &loss_type, int nfeat, int n_categ, vector<int> n_hidden, float def_A, float def_B, float def_lambda, float def_learning_rate, float def_momentum)
int micNet::init_fully_connected(micNetParams &in_params)
{
	string prefix = "micNet:: init_fully_connected ::";

	params = in_params;

	// sanity
	if (params.n_hidden.size() == 0) {
		MERR("%s ERROR: got 0 hidden layers...\n", prefix.c_str());
		return -1;
	}

	if (params.net_type == "fc") {
		if ((params.n_categ > 0 && params.n_hidden.back() != params.n_categ*params.n_per_categ) || (params.n_categ == 0 && params.n_hidden.back() != 1)) {
			MERR("%s (type %s) ERROR: mismatch in n_categ and last layer outputs: n_categ %d , n_per_categ %d,  last layer %d\n",
				prefix.c_str(), params.net_type.c_str(), params.n_categ, params.n_per_categ, params.n_hidden.back());
			return -1;
		}
	}
	else if (params.net_type == "autoencoder") {
		if (params.loss_type != "lsq") {
			MERR("%s (type %s) ERROR: in autoencoder mode only lsq loss_type is supported, while %s was chosen\n",
				prefix.c_str(), params.net_type.c_str(), params.loss_type.c_str());
			return -1;
		}
	}

	MLOG("init_fc: last_keep %d n_hidden %d ", params.last_layer_to_keep, params.n_hidden.size());
	for (auto i : params.n_hidden) { MLOG(" %d ", i); }
	MLOG("\n");
	MLOG("init_fc: n_norm %d sparse %f loss_type %s\n", params.n_norm_layers, params.sparse_zero_prob, params.loss_type.c_str());

	if (params.last_layer_to_keep < 0)
		nodes.clear();
	else
		nodes.resize(params.last_layer_to_keep + 1);
	nodes.reserve(500);

	vector<float> d_in = params.dropout_in_probs;
	d_in.insert(d_in.end(), 100, (float)1);

	if (params.last_layer_to_keep < 0) {
		if (add_input_layer() < 0) return -1;
	}
	else {
		// need to reinit d_in, and learning rates
		for (int i = 1; i <= params.last_layer_to_keep; i++) {
			nodes[i].learn_rates.set_val(params.learning_rates[i - 1]);
			nodes[i].lambda.set_val(params.def_lambda);
			nodes[i].rate_factor = 1.0;
		}
	}

	int pnode = max(0, params.last_layer_to_keep);
	int n_bef = params.last_layer_to_keep + 1;
	for (int i = 0; i < params.n_hidden.size(); i++) {
		float sparsness = params.sparse_zero_prob;
		if (i == params.n_hidden.size() - 1) sparsness = 0;
		MLOG("d_in %f i %d n_bef %d\n", d_in[i + n_bef], i, n_bef);
		if (add_fc_leaky_relu_layer(pnode++, params.n_hidden[i], d_in[i + n_bef], sparsness, params.learning_rates[i + n_bef]) < 0)
			return -1;
		if (i < params.n_norm_layers)
			if (add_normalization_layer(pnode++) < 0)
				return -1;
	}
	if (params.loss_type == "log")
		if (add_softmax_output_layer(pnode++) < 0) return -1;
	if (params.loss_type == "lsq")
		if (add_regression_output_layer(pnode++) < 0) return -1;

	MLOG("%s initialized micNet of %d nodes:\n", prefix.c_str(), nodes.size());
	for (int i = 0; i < nodes.size(); i++) {
		if (nodes[i].learn_rates.size() > 0)
			MLOG("%s node %d %d : %d --> %d , type %s , learn_rate %f , dropout %f ",
				prefix.c_str(), i, nodes[i].id, nodes[i].n_in, nodes[i].k_out, nodes[i].type.c_str(), nodes[i].learn_rates(0, 0), nodes[i].dropout_prob_in);
		else
			MLOG("%s node %d %d : %d --> %d , type %s , dropout %f ",
				prefix.c_str(), i, nodes[i].id, nodes[i].n_in, nodes[i].k_out, nodes[i].type.c_str(), nodes[i].dropout_prob_in);


		if (i == 0)
			MLOG("\n");
		else
			MLOG("prev_id %d\n", nodes[i].ir.in_node_id[0]);
	}

	for (auto &node : nodes) node.my_net = this;

	return 0;
}

//.................................................................................................
int micNet::init_autoencoder(micNetParams &in_params)
{
	string prefix = "micNet:: init_autoencoder ::";

	params = in_params;

	// sanity
	if (params.n_hidden.size() != 1) {
		MERR("%s ERROR: currently supporting only single hidden layer for autoencoders. (got %d)...\n", prefix.c_str(), params.n_hidden.size());
		return -1;
	}

	nodes.reserve(500);

	vector<float> d_in = params.dropout_in_probs;
	d_in.insert(d_in.end(), 100, (float)1);

	if (add_input_layer() < 0) return -1;
	int pnode = 0;

	// encoding layer
	float sparsness = params.sparse_zero_prob;
	if (add_fc_leaky_relu_layer(pnode++, params.n_hidden[0], d_in[0], sparsness, params.def_learning_rate) < 0)
		return -1;
	nodes[pnode - 1].subtype = "encoder";

	// decoding layer
	if (add_fc_leaky_relu_layer(pnode++, params.nfeat, d_in[0], sparsness, params.def_learning_rate) < 0)
		return -1;
	nodes[pnode - 1].subtype = "decoder";

	// loss layer
	add_autoencoder_loss(pnode++, 0);

	MLOG("%s initialized micNet of %d nodes:\n", prefix.c_str(), nodes.size());
	for (int i = 0; i < nodes.size(); i++) {
		MLOG("%s node %d : %d --> %d , type %s , dropout %f , subtype %s",
			prefix.c_str(), nodes[i].id, nodes[i].n_in, nodes[i].k_out, nodes[i].type.c_str(), nodes[i].dropout_prob_in, nodes[i].subtype.c_str());

		if (i == 0)
			MLOG("\n");
		else
			MLOG("prev_id %d\n", nodes[i].ir.in_node_id[0]);
	}

	return 0;
}


//.................................................................................................
// assumes Input nodes contain the batch in batch_out
// assumed nodes are ordered in a DAG way...that is each node contains ALL the nodes needed BEFORE
int micNet::forward_batch(int do_grad_flag)
{
	for (int i = 0; i < nodes.size(); i++) {
		if (nodes[i].type != "Input") {
			if (nodes[i].forward_batch(do_grad_flag) < 0) {
				MERR("micNet::forward_batch() error in node %d\n", i);
				return -1;
			}
		}
	}

	return 0;
}

//.................................................................................................
// assumes forward batch was run before
// assumes y for batches are in the output nodes
// assumed backward order is also correct in the sense each node has all its nodes for bacp prop after.
int micNet::back_prop_batch()
{
	nodes.back().back_propagete_from(NULL);
	for (int i = (int)nodes.size() - 1; i >= 0; i--) {
		if (nodes[i].type != "Input") {

			for (int j = 0; j < nodes[i].ir.in_node_id.size(); j++) {
				micNode *prev = &nodes[nodes[i].ir.in_node_id[j]];

				//MLOG("back_prop i=%d j=%d : node_id %d prev_id %d\n", i, j, nodes[i].id, prev->id);
				if (prev->type != "Input") {
					prev->print("debug backward before", 1, 6);

					prev->back_propagete_from(&nodes[i]);

					prev->print("debug backward after", 1, 6);
					//if (prev->wgt.m.size() > 0)
					//	prev->weights_gd_step();

					prev->print("debug backward after gd step", 1, 6);
				}
			}

		}
	}

	//	MLOG("Before Weights Step");
	//	test_grad_numerical(1, 50, 50, (float)1e-1);

	for (int i = (int)nodes.size() - 1; i >= 0; i--) {
		if (nodes[i].type != "Input") {

			for (int j = 0; j < nodes[i].ir.in_node_id.size(); j++) {
				micNode *prev = &nodes[nodes[i].ir.in_node_id[j]];

				if (prev->type != "Input") {

					if (prev->wgt.size() > 0)
						prev->weights_gd_step();

					prev->print("debug backward after gd step", 1, 6);
				}
			}

		}
	}
	return 0;
}


//.................................................................................................
int micNet::get_batch_with_samp_ratio(MedMat<float> &y_train, int batch_len, vector<int> &chosen)
{
	if (index_by_categ.size() == 0) {

		// initializing index_by_categ
		index_by_categ.resize(params.n_categ);
		for (int i = 0; i < y_train.size(); i++) {
			index_by_categ[(int)y_train(i, 0)].push_back(i);
		}
		for (int i = 0; i < params.n_categ; i++)
			shuffle(index_by_categ[i].begin(), index_by_categ[i].end(), globalRNG::get_engine());


		// making sure samp_ratio is given in probabilities
		float sum = 0;
		for (int i = 0; i < params.samp_ratio.size(); i++)
			sum += params.samp_ratio[i];
		if (sum > 0)
			for (int i = 0; i < params.samp_ratio.size(); i++) {
				params.samp_ratio[i] = params.samp_ratio[i] / sum;
			}
	}

	chosen.clear();
	for (int i = 0; i < params.samp_ratio.size(); i++) {
		int n_take = (int)(params.samp_ratio[i] * (float)batch_len);
		for (int j = 0; j < n_take; j++)
			chosen.push_back(index_by_categ[i][rand_N((int)index_by_categ[i].size())]);
	}
	while (chosen.size() < batch_len) {
		int i = rand_N((int)params.samp_ratio.size());
		int j = rand_N((int)index_by_categ[i].size());
		chosen.push_back(index_by_categ[i][j]);
	}
	chosen.resize(batch_len);
	return 0;
}

//.................................................................................................
int micNet::learn_single_epoch(MedMat<float> &x_train, MedMat<float> &y_train, vector<float> &weights, int last_is_bias_flag)
{
	string prefix = "micNet::learn_single_epoch() ::";

	int nsamples = x_train.nrows;
	int nfeat = x_train.ncols;
	if (last_is_bias_flag) nfeat--;

	if (nodes[0].n_in != nfeat) {
		MERR("%s non matching Input node and mat size : n_in %d x_train %d x %d , last_is_bias %d\n", prefix.c_str(), nodes[0].n_in, x_train.nrows, x_train.ncols, last_is_bias_flag);
		return -1;
	}
	// first getting a shuffle
	vector<int> perm;
	vector<int> chosen;
	int *taken_to_batch = NULL;
	get_rand_vector_no_repetitions(perm, nsamples, nsamples);

	int n_batches = nsamples / params.batch_size;
	if (nsamples % params.batch_size != 0) n_batches++;

	MLOG("...................................................................................................................\n");
	MLOG("%s start going over batches batch_size %d n_batches %d\n", prefix.c_str(), params.batch_size, n_batches);
	// going over batches
	for (int b = 0; b < n_batches; b++) {
		int from = b * params.batch_size;
		int to = min(from + params.batch_size, nsamples);
		int len = to - from;

		if (params.samp_ratio.size() == params.n_categ) {
			get_batch_with_samp_ratio(y_train, len, chosen);
			taken_to_batch = &chosen[0];
		}
		else
			taken_to_batch = &perm[from];


		//		MLOG("%s b %d before fill input, from %d to %d len %d\n", prefix.c_str(), b,from,to,len);
		nodes[0].fill_input_node(taken_to_batch, len, x_train, last_is_bias_flag);

		//nodes[0].print("debug input", 1, 6);
		// copy y to output nodes
//		MLOG("%s b %d before fill output\n", prefix.c_str(), b);
		for (int i = 0; i < nodes.size(); i++)
			if (nodes[i].is_terminal) {
				nodes[i].fill_output_node(taken_to_batch, len, y_train, weights);
				//nodes[i].print("debug output", 1, 6);
			}

		// forward
		//MLOG("%s b %d before forward batch\n", prefix.c_str(), b);
		if (forward_batch(1) < 0) return -1;

		// backward
		//MLOG("%s b %d before backward batch\n", prefix.c_str(), b);
		if (back_prop_batch() < 0) return -1;

	}

	return 0;
}

//..................................................................................................................................................
int micNet::learn(MedMat<float> &x_train, MedMat<float> &y_train, vector<float> &weights,
	MedMat<float> &x_test, MedMat<float> &y_test, int n_epochs, int eval_freq, int last_is_bias_flag)
{

	string prefix = "micNet::learn() ::";

	vector<NetEval> on_train_evals;
	vector<NetEval> on_test_evals;

	MLOG("%s :: initializing net\n", prefix.c_str());
	params.nfeat = x_train.ncols - last_is_bias_flag;
	if (nodes.size() == 0) init_fully_connected(params);

	MLOG("%s starting on x_train %d x %d, y_train %d x %d : x_test %d x %d , y_test %d x %d : nepochs %d : eval_freq %d\n",
		prefix.c_str(), x_train.nrows, x_train.ncols, y_train.nrows, y_train.ncols, x_test.nrows, x_test.ncols, y_test.nrows, y_test.ncols, n_epochs, eval_freq);

	for (int i_epoch = 0; i_epoch < n_epochs; i_epoch++) {

		MedTimer et;
		et.start();

		if (learn_single_epoch(x_train, y_train, weights, last_is_bias_flag) < 0) {
			MERR("%s ERROR: failed learn_single_epoch in epoch %d\n", prefix.c_str(), i_epoch);
			return -1;
		}

		MLOG("....>>");

		if (((i_epoch + 1) % eval_freq) == 0) {

			NetEval ne;
			et.take_curr_time();
			ne.epoch = i_epoch;
			ne.dt = et.diff_sec();
			eval("On-Train", x_train, y_train, ne, last_is_bias_flag);
			on_train_evals.push_back(ne);
			eval("Test", x_test, y_test, ne, last_is_bias_flag);
			on_test_evals.push_back(ne);

			et.take_curr_time();
			MLOG("%s epoch %d :: On-Train: err %f log-loss %f lsq %f :: Test err: err %f log-loss %f lsq %f :: dt %g :: dt+eval %g\n",
				prefix.c_str(), i_epoch,
				on_train_evals.back().acc_err, on_train_evals.back().log_loss, on_train_evals.back().lsq_loss,
				on_test_evals.back().acc_err, on_test_evals.back().log_loss, on_test_evals.back().lsq_loss,
				ne.dt, et.diff_sec());

			if (on_train_evals.size() > 1) {

				int s = (int)on_train_evals.size() - 1;

				if (s > 5)
					if (on_train_evals[s].log_loss >= (float)on_train_evals[s - 1].log_loss)
						for (int j = 0; j < nodes.size(); j++)
							nodes[j].prev_grad_w.clear();

				for (int j = 0; j < nodes.size(); j++)
					if (nodes[j].rate_factor > (float)1e-3)
						nodes[j].rate_factor *= (float)params.rate_decay;

				if (s > 5000) {
					if (on_train_evals[s].log_loss < (float)on_train_evals[s - 1].log_loss) {
						nodes_last_best = nodes;
						if (on_train_evals[s].log_loss > (float)0.995*on_train_evals[s - 1].log_loss) {
							MLOG("%f < %f ---> increasing rate rate = %f\n", on_train_evals[s].log_loss, on_train_evals[s - 1].log_loss, nodes[1].rate_factor);
							for (int j = 0; j < nodes.size(); j++) nodes[j].rate_factor *= (float)1.05;
						}
					}
					else {

						// we got worse...
						// hence we go back to the best option and decrease the rate
						on_train_evals.pop_back();
						on_test_evals.pop_back();
						float orig_factor = nodes[1].rate_factor;
						nodes = nodes_last_best;
						for (int j = 0; j < nodes.size(); j++)
							nodes[j].rate_factor = orig_factor;
						MLOG("%f > %f ---> decreasing rate rate = %f\n", on_train_evals[s].log_loss, on_train_evals[s - 1].log_loss, nodes[1].rate_factor);
						for (int j = 0; j < nodes.size(); j++)
							if (nodes[j].rate_factor > (float)1e-2)
								nodes[j].rate_factor *= (float)0.9;


					}
				}


			}

		}

	}

	return 0;
}

//.................................................................................................
// predictions taken in the Last Node (pred_node)
int micNet::predict(MedMat<float> &x, MedMat<float> &preds, int last_is_bias_flag)
{
	//MLOG("predict(Mat,Mat) API\n");
	string prefix = "micNet::predict() ::";

	int nsamples = x.nrows;
	int nfeat = x.ncols;
	if (last_is_bias_flag) nfeat--;

	if (nodes[0].n_in != nfeat) {
		MERR("%s non matching Input node and mat size : n_in %d x: %d x %d , last_is_bias %d\n", prefix.c_str(), nodes[0].n_in, x.nrows, x.ncols, last_is_bias_flag);
		return -1;
	}

	//MERR("micNet predict() : n_in %d x: %d x %d , last_is_bias %d\n", nodes[0].n_in, x.nrows, x.ncols, last_is_bias_flag);
	// first getting a shuffle
	vector<int> unit(nsamples);
	for (int i = 0; i < nsamples; i++) { unit[i] = i; }

	int n_batches = nsamples / params.predict_batch_size;
	if (nsamples % params.predict_batch_size != 0) n_batches++;

	micNode *pred_node = &nodes.back();
	int n_categ = pred_node->k_out;
	preds.resize(nsamples, n_categ);

	for (auto &node : nodes) node.my_net = this;

	// going over batches
	for (int b = 0; b < n_batches; b++) {
		int from = b * params.predict_batch_size;
		int to = min(from + params.predict_batch_size, nsamples);
		int len = to - from;

		//MLOG("micNet predict: predict batch size %d , batch %d, from %d , to %d , len %d\n", params.predict_batch_size, b, from, to, len);
		nodes[0].fill_input_node(&unit[from], len, x, last_is_bias_flag);

		//for (auto &node : nodes) node.my_net = this;
		//MLOG("Before forward batch %d : batch_out %d x %d : %d %d : %x %x\n", b, nodes[0].batch_out.nrows, nodes[0].batch_out.ncols, nodes[1].my_net->nodes.size(), nodes[1].my_net->nodes[0].batch_out.nrows, this, &(nodes[1].my_net));
		// forward without gradients
		if (forward_batch(0) < 0) return -1;
		//MLOG("After forward batch %d\n", b);

		// copy results to preds mat
		for (int i = from; i < to; i++)
			for (int j = 0; j < n_categ; j++)
				preds(i, j) = pred_node->batch_out(i - from, j);

	}

	return 0;
}

void micNet::predict_single(const vector<float> &x, vector<float> &preds) const
{
	string prefix = "micNet::predict() ::";

	int nfeat = (int)x.size();

	if (nodes[0].n_in != nfeat)
		MTHROW_AND_ERR("%s non matching Input node and mat size : n_in %d x: %d\n", prefix.c_str(), nodes[0].n_in, nfeat);

	// going over batches
	vector<MedMat<float>> nodes_out(nodes.size());
	MedMat<float> &first_batch_out = nodes_out[0]; // = nodes[0].batch_out;
	MedMat<float> &last_pred = nodes_out.back();
	first_batch_out.resize(1, nfeat + 1);
	float *b_out = first_batch_out.data_ptr();
	memcpy(b_out, &x[0], nfeat * sizeof(float));
	b_out[nfeat] = 1; // bias term

	for (int i = 0; i < nodes.size(); i++) {
		if (nodes[i].type != "Input") {
			nodes[i].forward_batch(nodes_out, nodes_out[i]);
		}
	}

	// copy results to preds mat
	preds = std::move(last_pred.get_vec());
	if (preds.size() > 1) //has additional channel of 1 for bias - remove it
		preds.resize(preds.size() - 1);
	/*for (int j = 0; j < n_categ; j++)
		preds[j] = last_pred(0, j);*/
}


//.................................................................................................
void micNet::copy_nodes(vector<micNode> &in_nodes)
{
	nodes = in_nodes;
}

//.................................................................................................
// evaluating performance (typically once an epoch, or a few epochs)
// runs prediction on current micNet, then calculates several perf measures
// n_categ == 1 :: means we are in regression mode
// n_categ > 1  :: means we are in classification mode
int micNet::eval(const string &name, MedMat<float> &x, MedMat<float> &y, NetEval &eval, int last_is_bias_flag)
{
	if (params.n_categ <= 0) return -1;

	eval.name = name;

	MedMat<float> preds;

	if (predict(x, preds, last_is_bias_flag) < 0) return -1;

	MLOG("eval() after predict for %s : x %d x %d , preds %d x %d loss type: %s\n", name.c_str(), x.nrows, x.ncols, preds.nrows, preds.ncols, params.loss_type.c_str());
	int nsamples = x.nrows;

	if (0) { //params.n_categ == 1) {
		// regression case


	}
	else {
		if (params.net_type == "fc") {
			// classification case
			vector<float> max_pred(nsamples);

			// get the prediction with maximal probability
			for (int i = 0; i < nsamples; i++) {
				int max_j = 0;
				float max_p = preds(i, 0);
				for (int j = 0; j < params.n_categ; j++) {
					if (params.loss_type == "log" && ((preds(i, j) < 0) || preds(i, j) > 1)) {
						MERR("ERROR: preds(%d,%d) = %f\n", i, j, preds(i, j));
						exit(-1);
					}
					if (preds(i, j) > max_p) {
						max_j = j;
						max_p = preds(i, j);
					}
				}
				max_pred[i] = (float)max_j;
				if (preds.ncols == 1) {
					max_pred[i] = (float)((int)(preds(i, 0) + (float)0.5));
				}
			}

			// get acc_err
			if (1) { //params.loss_type == "log") {
				int nneg = 0;
				for (int i = 0; i < nsamples; i++)
					if (max_pred[i] != y(i, 0))
						nneg++;
				eval.acc_err = (float)nneg / (float)nsamples;
			}

			if (1) {

				eval.lsq_loss = 0;
				if (preds.ncols == 1) {
					for (int i = 0; i < nsamples; i++)
						eval.lsq_loss += (float)0.5*(preds(i, 0) - y(i, 0))*(preds(i, 0) - y(i, 0));
				}
				else {
					for (int i = 0; i < nsamples; i++)
						for (int j = 0; j < preds.ncols; j++)
							if ((int)y(i, 0) == j)
								eval.lsq_loss += (float)0.5*(preds(i, j) - 1)*(preds(i, j) - 1);
							else
								eval.lsq_loss += (float)0.5*(preds(i, j))*(preds(i, j));


				}

				eval.lsq_loss /= (float)nsamples;
			}

			if (1) {
				// get log loss
				float loss = 0;
				float epsilon = (float)1e-5;
				for (int i = 0; i < nsamples; i++) {
					//					float p = max(epsilon, preds(i, (int)max_pred[i]));
					float p = max(epsilon, preds(i, (int)y(i, 0)));
					loss += -log(p);
				}
				eval.log_loss = loss;
				eval.log_loss /= (float)nsamples;
			}
#if 0
			// debug
			for (int i = 0; i < nsamples; i += nsamples / 20) {
				MLOG("x: ");
				for (int j = 0; j < x.ncols; j++)
					MLOG(" %f", x(i, j));
				MLOG("\n");
				MLOG("preds(%d,*) (%d) ::", i, (int)y(i, 0));
				for (int j = 0; j < preds.ncols; j++)
					MLOG(" (%d) %f", j, preds(i, j));
				MLOG("\n");
			}

#endif
			// TBD: more measures....auc,corr, etc....
			}

		if (params.net_type == "autoencoder") {
			MLOG("Eval autoencoder: preds %d x %d , y %d x %d nsamples %d\n", preds.nrows, preds.ncols, y.nrows, y.ncols, nsamples);
			eval.lsq_loss = 0;
			float fact = (float)1 / (float)nsamples;
			for (int i = 0; i < nsamples; i++)
				for (int j = 0; j < preds.ncols; j++) {
					if (isnan(y(i, j)) || isnan(preds(i, j))) {
						MLOG("i %d j %d preds %g y %g\n", i, j, preds(i, j), y(i, j));
						exit(-1);
					}
					eval.lsq_loss += fact * (float)0.5*(preds(i, j) - y(i, j))*(preds(i, j) - y(i, j));
				}
		}
		}

	return 0;
	}

int micNet::test_grad_numerical(int i_node, int i_in, int i_out, float epsilon)
{

	// need to calculate L for current batch with wgt(i,j)+- epsilon

	// creating a copy of our current state, in order not to destroy current net
	micNet testNet;
	testNet.params = params;
	testNet.copy_nodes(nodes);

	int n_nodes = (int)nodes.size();
	float orig_wgt = nodes[i_node].wgt(i_in, i_out);

	// Now calculating sum of Loss function over all elements in the batch with +- epsilon , and averaging.

	//MLOG("Testing, input_batch: %d x %d orig_wgt %f epsilon %f\n", 
	//	testNet.nodes[0].batch_out.nrows, testNet.nodes[0].batch_out.ncols, orig_wgt, epsilon);
	// results with +epsilon
	testNet.nodes[i_node].wgt(i_in, i_out) = orig_wgt + epsilon;
	testNet.forward_batch(0);
	MedMat<float> out_plus = testNet.nodes[n_nodes - 1].batch_out;

	// results with -epsilon
	//testNet.nodes[i_node].init_wgts_rand_normal(0, 1);
	testNet.nodes[i_node].wgt(i_in, i_out) = orig_wgt - epsilon;
	testNet.forward_batch(0);
	MedMat<float> out_minus = testNet.nodes[n_nodes - 1].batch_out;


	//for (int i=0; i<testNet.nodes[2].batch_out.nrows; i++)
	//	for (int j=0; j<testNet.nodes[2].batch_out.ncols; j++)
	//		if (testNet.nodes[2].batch_out(i, j) != testNet.nodes[3].batch_in(i, j))
	//		//if (nodes[2].batch_out(i, j) != nodes[3].batch_in(i, j))
	//			MLOG("#!!!!! ERROR??? : %f %f\n", nodes[2].batch_out(i, j), nodes[3].batch_in(i, j));
	//		//	MLOG("#!!!!! ERROR??? : %f %f\n", testNet.nodes[2].batch_out(i, j), testNet.nodes[3].batch_in(i, j));


	// store y
	MedMat<float> y = testNet.nodes[n_nodes - 1].y;

	// get losses - currently without regularization terms
	vector<float> loss_plus, loss_minus;
	int n_b = out_plus.nrows;
	float eps = (float)1e-50;
	float lambda = testNet.nodes[i_node].lambda(0, 0);
	for (int i = 0; i < n_b; i++) {

		float val_plus = max(out_plus(i, (int)y(i, 0)), eps);
		loss_plus.push_back(-log(val_plus) + (float)0.5*lambda*(orig_wgt + epsilon)*(orig_wgt + epsilon));

		float val_minus = max(out_minus(i, (int)y(i, 0)), eps);
		loss_minus.push_back(-log(val_minus) + (float)0.5*lambda*(orig_wgt - epsilon)*(orig_wgt - epsilon));

		//MLOG("i=%d y=%d val_plus %f val_minus %f loss_plus %f loss_minus %f\n",
		//	i, (int)y(i, 0), val_plus, val_minus, loss_plus.back(), loss_minus.back());
	}

	// sum losses average and compare
	float s_plus = 0, s_minus = 0;

	for (int i = 0; i < n_b; i++) {
		s_plus += loss_plus[i];
		s_minus += loss_minus[i];
	}

	float dloss = (s_plus - s_minus) / ((float)2 * epsilon);
	float avg_dloss = dloss / (float)n_b;

	float grad = nodes[i_node].grad_w(i_in, i_out);
	float d_grad = grad - avg_dloss;
	float r_grad = d_grad / grad;

	MLOG("#### Gradient Test node %d , wgt(%d,%d) %f %f : grad_w %f : num grad %f :  %f %f\n",
		i_node, i_in, i_out,
		nodes[i_node].wgt(i_in, i_out),
		testNet.nodes[i_node].wgt(i_in, i_out),
		grad, avg_dloss, d_grad, r_grad);

	return 0;
}

//=====================================================================================
// micNet APIs for MedAlgo
//=====================================================================================
int micNet::init_from_string(string init_str)
{
	if (params.init_from_string(init_str) < 0) return -1;

	/*
		if (params.net_type == "fc")
			return init_fully_connected(params);

		if (params.net_type == "autoencoder")
			return init_fully_connected(params);
	*/
	return 0;
}

//--------------------------------------------------------------------------------------
int micNet::learn(MedMat<float> &x_train, MedMat<float> &y_train, vector<float> &weights)
{
	string prefix = "micNet::learn(*)";
	vector<NetEval> on_train_evals;
	MedTimer et;

	params.nfeat = x_train.ncols;
	params.params_init_string += ";nfeat=" + to_string(params.nfeat);
	if (nodes.size() == 0) init_fully_connected(params);

	int i_epoch = 0;
	int go_on = 1;

	while (i_epoch < params.max_epochs && go_on) {
		et.start();

		// an epoch learn step
		if (learn_single_epoch(x_train, y_train, weights) < 0) return -1;

		// evaluation on train set
		NetEval ne;
		ne.epoch = i_epoch;
		et.take_curr_time();
		ne.dt = et.diff_sec();
		eval("On-Train", x_train, y_train, ne, 0);
		on_train_evals.push_back(ne);

		float curr = 0, back = 0, err = -1;
		if (i_epoch > params.min_epochs && i_epoch > params.n_back) {
			if (params.loss_type == "log") {
				curr = on_train_evals[i_epoch].log_loss;
				back = on_train_evals[i_epoch - params.n_back].log_loss;
				err = (back - curr) / back;
			}
			if (params.loss_type == "lsq") {
				curr = on_train_evals[i_epoch].lsq_loss;
				back = on_train_evals[i_epoch - params.n_back].lsq_loss;
				err = (back - curr) / back;
			}
		}

		et.take_curr_time();
		MLOG("%s epoch %d :: On-Train: err %f log-loss %f lsq %f :: back_err %f curr %f back %f :: dt %g :: dt+eval %g\n",
			prefix.c_str(), i_epoch,
			on_train_evals.back().acc_err, on_train_evals.back().log_loss, on_train_evals.back().lsq_loss, err, curr, back,
			ne.dt, et.diff_sec());

		// test for stop criteria
		if (i_epoch > params.min_epochs && i_epoch > params.n_back) {
			if (err < params.min_improve_n_back) {
				MLOG("%s epoch %d :: Met Stopping Criteria: curr %f back %f err %f (bound %f)\n",
					prefix.c_str(), i_epoch, curr, back, err, params.min_improve_n_back);
				go_on = 0;
			}
		}

		i_epoch++;
	}

	return 0;
}

//--------------------------------------------------------------------------------------
int micNet::predict(MedMat<float> &x, vector<float> &preds)
{

	MLOG_D("predict(Mat,vector) API\n");
	MedMat<float> mpreds;

	if (predict(x, mpreds) < 0) return -1;

	if (params.n_categ == params.n_preds_per_sample && mpreds.ncols == params.n_categ)
		preds = std::move(mpreds.get_vec());
	else if (params.n_preds_per_sample == 1 && params.pred_class < params.n_categ && mpreds.ncols == params.n_categ)
		mpreds.get_col(params.pred_class, preds);
	else {
		MERR("micNet:: predict: wrong params to get prediction: ncateg %d ncols %d n_preds_per_sample %d pred_class %d\n", params.n_categ, mpreds.ncols, params.n_preds_per_sample, params.pred_class);
		return -1;
	}
	return 0;
}

//--------------------------------------------------------------------------------------
// serializations
//--------------------------------------------------------------------------------------
#define MICNET_MAGIC_NUMBER 0x02468acefdb97531
/*
//--------------------------------------------------------------------------------------
size_t micNode::get_size()
{

	size_t size = 0;

	// magic num
	size += sizeof(unsigned long long);

	// version
	size += sizeof(int);

	// sanity id
	size += sizeof(int);

	// wgt_mat
	size += wgt.get_size();

	// alpha , beta
	size += sizeof(int);
	size += alpha.size()*sizeof(float);

	size += sizeof(int);
	size += beta.size()*sizeof(float);

	return size;
}

//--------------------------------------------------------------------------------------
size_t micNode::serialize(unsigned char *buf)
{
	size_t size = 0;
	int curr_version = 0;

	//cerr << "micNode serialize ! id " << id << "\n";
	// magic number followed by serialize version (currently 0)
	*((unsigned long long *)&buf[size]) = (unsigned long long)MICNET_MAGIC_NUMBER; size += sizeof(unsigned long long);
	*((int *)&buf[size]) = curr_version; size += sizeof(int);

	// serializing id for sanity (most other params are assumed to be loaded from init)
	*((int *)&buf[size]) = id; size += sizeof(int);

	// wgt mat
	size_t wgt_size = wgt.get_size();
	size_t rc = wgt.serialize(&buf[size]);
	if (rc != wgt_size) {
		MERR("micNode: serialize: wgt serialization error: got %lld bytes instead of %lld\n", rc, wgt_size);
		return (size_t)-1;
	}
	size += wgt_size;

	// normalization params
	// alpha
	int a_size = (int)alpha.size();
	*((int *)&buf[size]) = a_size; size += sizeof(int);
	if (a_size > 0) {
		memcpy(&buf[size], &alpha[0], (size_t)a_size*sizeof(float));
		size += (size_t)a_size*sizeof(float);
	}

	// beta
	int b_size = (int)beta.size();
	*((int *)&buf[size]) = b_size; size += sizeof(int);
	MLOG("micnode serialize node id %d size %lld b_size %d\n", id, size, b_size);
	if (b_size > 0) {
		memcpy(&buf[size], &beta[0], (size_t)b_size*sizeof(float));
		size += (size_t)b_size*sizeof(float);
	}

	// Done !

	return size;

}

//--------------------------------------------------------------------------------------
size_t micNode::deserialize(unsigned char *buf)
{
	size_t size = 0;

	// magic number followed by serialize version (currently 0)
	unsigned long long magic = *((unsigned long long *)&buf[size]); size += sizeof(unsigned long long);
	if (magic != (unsigned long long)MICNET_MAGIC_NUMBER) {
		MERR("micNode: deserialize: ERROR : Wrong Magic number.\n");
		return -1;
	}

	int curr_version = *((int *)&buf[size]); size += sizeof(int);

	if (curr_version != 0) {
		MERR("micNode: deserialize: ERROR : unsupported version %d.\n",curr_version);
		return -1;
	}

	// serializing id for sanity (most other params are assumed to be loaded from init)
	int id = *((int *)&buf[size]); size += sizeof(int);

	// wgt mat
	size_t wsize = wgt.deserialize(&buf[size]);
	if (wsize == (size_t)-1) return -1;
	size += wsize;

	// normalization params
	// alpha
	int a_size = *((int *)&buf[size]); size += sizeof(int);
	alpha.clear();
	if (a_size > 0) {
		alpha.resize(a_size);
		memcpy(&alpha[0], &buf[size], (size_t)a_size*sizeof(float));
		size += (size_t)a_size*sizeof(float);
	}

	// beta
	int b_size = *((int *)&buf[size]); size += sizeof(int);
	MLOG("micnode deserialize node id %d size %lld b_size %d\n", id, size, b_size);
	if (b_size > 1000000) {
		MERR("micnode deserialize b_size is %d.... this is a BUG!!! ... changing it to 0\n", b_size);
		b_size = 0;
		//return -1;
	}
	beta.clear();
	if (b_size > 0) {
		beta.resize(b_size);
		memcpy(&beta[0], &buf[size], (size_t)b_size*sizeof(float));
		size += (size_t)b_size*sizeof(float);
	}

	// Done !

	return size;

}

*/


/*
//--------------------------------------------------------------------------------------
size_t micNet::get_size()
{
	size_t size = 0;

	// magic num & version
	size += sizeof(unsigned long long);
	size += sizeof(int);

	// params_init_string
	size += sizeof(int);
	size += params.params_init_string.length();

	// n_nodes and nodes
	size += sizeof(int);
	for (int i=0; i<(int)nodes.size(); i++)
		size += nodes[i].get_size();

	return size;
}

//--------------------------------------------------------------------------------------
size_t micNet::serialize(unsigned char *buf)
{
	size_t size = 0;
	int curr_version = 0;

	// magic num & version
	*((unsigned long long *)&buf[size]) = (unsigned long long)MICNET_MAGIC_NUMBER; size += sizeof(unsigned long long);
	*((int *)&buf[size]) = curr_version; size += sizeof(int);

	// params.params_init_string
	int slen = (int)params.params_init_string.size();
	*((int *)&buf[size]) = slen + 1; size += sizeof(int);
	memcpy((char *)&buf[size], params.params_init_string.c_str(), slen);
	size += slen;
	buf[size] = 0; size++;

	// n_nodes, and all the nodes after
	int n_nodes = (int)nodes.size();
	*((int *)&buf[size]) = n_nodes; size += sizeof(int);

	for (int i=0; i<n_nodes; i++) {
		size_t node_size = nodes[i].serialize(&buf[size]);
		if (node_size == (size_t)-1) return ((size_t)-1);
		size += node_size;
	}

	// Done !
	return size;

}

//--------------------------------------------------------------------------------------
size_t micNet::deserialize(unsigned char *buf)
{
	size_t size = 0;

	// magic num & version
	unsigned long long magic = *((unsigned long long *)&buf[size]); size += sizeof(unsigned long long);
	if (magic != (unsigned long long)MICNET_MAGIC_NUMBER) {
		MERR("micNet: deserialize: ERROR : Wrong Magic number.\n");
		return (size_t)-1;
	}

	int curr_version = *((int *)&buf[size]); size += sizeof(int);

	if (curr_version != 0) {
		MERR("micNet: deserialize: ERROR : unsupported version %d.\n", curr_version);
		return (size_t)-1;
	}

	// params.params_init_string
	string init_string;
	int slen = *((int *)&buf[size]); size += sizeof(int);
	init_string.assign((char *)&buf[size], slen);
	size += slen;

	// actually initializing
	init_net(init_string);

	// n_nodes, and all the nodes after
	int n_nodes = *((int *)&buf[size]); size += sizeof(int);

	for (int i=0; i<n_nodes; i++) {
		size_t node_size = nodes[i].deserialize(&buf[size]);
		if (node_size == (size_t)-1) return ((size_t)-1);
		size += node_size;
	}

	// Done !
	return size;
}
*/