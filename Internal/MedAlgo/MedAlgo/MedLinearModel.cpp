#include <stdexcept>
#include <map>
#include <cmath>
#include <iostream>
#include <random>
#include "MedLinearModel.h"
#include "SGD.h"
#include "MedMat/MedMat/MedMatConstants.h"

void generate_data_learn_rec(int poly_degree, const vector<const float *> &data, int samples_size,
	vector<vector<float>> &gen_data, int &pos, const vector<int> &path, int max_l = -1) {
	//skip empty path (all -1) for bias, no need to copy to gen_data
	if (path.size() == poly_degree) {
		if (pos == 0) {
			++pos;
			return;
		}
		//actual do work for all samples in curretn path
		gen_data[pos - 1].resize(samples_size);
		for (int idx = 0; idx < samples_size; ++idx)
		{
			double c = 1;
			for (int p : path)
				if (p >= 0) //If -1 than not selected, sub degree, skip
					c *= data[p][idx]; //featre than dample
			gen_data[pos - 1][idx] = c;
		}
		++pos; //counter for coeff order
		return;
	}
	int feat_num = (int)data.size();
	for (int i = max_l; i < feat_num; ++i)
	{
		vector<int> new_p(path);
		new_p.push_back(i);
		generate_data_learn_rec(poly_degree, data, samples_size, gen_data, pos, new_p, i);
	}
}
void generate_data_learn(int poly_degree, const vector<const float *> &data,
	int samples_size, vector<vector<float>> &gen_data) {
	//assume gen_data has the right size fo coeff of model
	int pos = 0;
	vector<int> path;
	generate_data_learn_rec(poly_degree, data, samples_size, gen_data, pos, path);
}

void poly_get_score_rec(int poly_degree, const vector<double> &coeff, const vector<const float *> &data, vector<double> &scores,
	int &pos, vector<int> &path, int max_l = -1) {

	if (path.size() == poly_degree) {
		for (int idx = 0; idx < scores.size(); ++idx)
		{
			//actual do work and exit - has all selected variables idx's in path
			double c = coeff[pos];
			for (int p : path)
				if (p >= 0) //If -1 than not selected, sub degree
					c *= data[p][idx]; //featre than dample

#pragma omp critical
			scores[idx] += c; //adds coeff, coeff to results
		}
		++pos;
		return;
	}
	int feat_num = (int)data.size();
	for (int i = max_l; i < feat_num; ++i)
	{
		vector<int> new_p(path);
		new_p.push_back(i);
		poly_get_score_rec(poly_degree, coeff, data, scores, pos, new_p, i);
	}
}
void poly_get_score(int poly_degree, const vector<double> &coeff, const vector<const float *> &data, vector<double> &scores) {
	int pos = 0;
	vector<int> path;
	poly_get_score_rec(poly_degree, coeff, data, scores, pos, path);
}
double poly_get_score_rec(int poly_degree, const vector<double> &coeff, const vector<const float *> &data, int idx,
	int &pos, vector<int> &path, int max_l = -1) {

	if (path.size() == poly_degree) {

		//actual do work and exit - has all selected variables idx's in path
		double c = coeff[pos];
		for (int p : path)
			if (p >= 0) //If -1 than not selected, sub degree
				c *= data[p][idx]; //featre than dample

		++pos;
		return c;
	}
	int feat_num = (int)data.size();
	double res = 0;
	for (int i = max_l; i < feat_num; ++i)
	{
		vector<int> new_p(path);
		new_p.push_back(i);
		res += poly_get_score_rec(poly_degree, coeff, data, idx, pos, new_p, i);
	}
	return res;
}
double poly_get_score(int poly_degree, const vector<double> &coeff, const vector<const float *> &data, int idx) {
	int pos = 0;
	vector<int> path;
	return poly_get_score_rec(poly_degree, coeff, data, idx, pos, path);
}

//not in use
double MedLinearModel::predict(const vector<float> &input) const {
	if (input.size() != model_params.size() - 1)
		throw invalid_argument("input has wrong number of signals. expeced" + to_string(model_params.size() - 1) + " got "
			+ to_string(input.size()));
	if (poly_degree == 1) {
		double res = model_params[0];
		for (size_t i = 0; i < input.size(); ++i)
			res += input[i] * model_params[i + 1];

		//res += model_params[model_params.size() - 1];
		return res;
	}
	else {
		vector<const float *> data(input.size());
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = &input[i];
		vector<double> prds(1);
		poly_get_score(poly_degree, model_params, data, prds);
		return prds[0];
	}
}

void MedLinearModel::predict(const vector<vector<float>> &inputs, vector<double> &preds) const {
	vector<vector<float>> copy_inp;
	const vector<float> *access_arr = inputs.data();
	if (inputs.size() == 0)
		throw invalid_argument("must have at least one signal");

	if (mark_learn_finish && _meanShift.size() > 0) {
		copy_inp = vector<vector<float>>(inputs);
		apply_normalization(copy_inp);
		access_arr = copy_inp.data();
	}
	if (preds.size() < inputs[0].size())
		preds.resize(inputs[0].size());
	if (poly_degree == 1 || !mark_learn_finish) { //in learn it's just simple calc
		for (size_t i = 0; i < preds.size(); ++i)
		{
			preds[i] = model_params[0];
			for (size_t k = 0; k < inputs.size(); ++k)
				preds[i] += access_arr[k][i] * model_params[k + 1];
		}
	}
	else {
		vector<const float *> data(inputs.size());
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = access_arr[i].data();
		poly_get_score(poly_degree, model_params, data, preds);
	}
}

#define REG_LAMBDA 0.0
#define SMOOTH false
#define REGULARIZATION_GEOM 0.1
subGradientFunction  MedLinearModel::getSubGradients() {
	//This is subGradient in L2, for other loss function you need to change this function
	subGradientFunction func = [](int ind, const vector<double> &params, const vector<vector<float>> &x, const vector<float> &y, const vector<float> *weights) {
		double res = 0;
		if (weights == NULL || weights->empty()) {
			for (size_t i = 0; i < y.size(); ++i)
			{
				double productRes = params[0];
				for (size_t k = 0; k < x.size(); ++k)
					productRes += params[k + 1] * x[k][i];
				float x_val = 1;
				if (ind > 0)
					x_val = x[ind - 1][i];
				res += 2 * (params[ind] * static_cast<double>(x_val) * static_cast<double>(x_val) + (productRes - params[ind] * static_cast<double>(x_val)) * static_cast<double>(x_val) - static_cast<double>(y[i]) * static_cast<double>(x_val));
			}
			//res /= y.size(); - constant, not needed

			return res;
		}
		else {
			for (size_t i = 0; i < y.size(); ++i)
			{
				double productRes = params[0];
				for (size_t k = 0; k < x.size(); ++k)
					productRes += params[k + 1] * x[k][i];
				float x_val = 1;
				if (ind > 0)
					x_val = x[ind - 1][i];
				res += 2 * (params[ind] * static_cast<double>(x_val) * static_cast<double>(x_val) + (productRes - params[ind] * static_cast<double>(x_val)) * static_cast<double>(x_val) - static_cast<double>(y[i]) * static_cast<double>(x_val)) * static_cast<double>(weights->at(i));
			}
			return res;
		}
	};

	return func;
}
subGradientFunction  MedLinearModel::getSubGradientsAUC() {
	//This is subGradient in L2, for other loss function you need to change this function. (need fix for bias param, W{n+1})
	subGradientFunction func = [](int ind, const vector<double> &params, const vector<vector<float>> &x, const vector<float> &y, const vector<float> *weights) {
		if (ind == 0)
			return (double)0;

		double res = 0;
		vector<vector<int>> targetToInd(2);
		for (int i = 0; i < y.size(); ++i)
			targetToInd[int(y[i] > 0)].push_back(i);

		vector<int> &posInds = targetToInd[1];
		vector<int> &negInds = targetToInd[0]; //change to -1 if y is given that way
		if (weights == NULL || weights->empty()) {
			for (size_t i = 0; i < posInds.size(); ++i)
			{
				int posIndex = posInds[i];
				for (size_t j = 0; j < negInds.size(); ++j)
				{
					int negIndex = negInds[j];
					double sumDiff = 0;
					for (size_t k = 1; k < params.size(); ++k) //param 0 should be zero
					{
						sumDiff += params[k] * (x[k - 1][posIndex] - x[k - 1][negIndex]);
					}
					sumDiff *= 1;
					if (sumDiff > 100) {
						res += 0;
						continue;
					}
					if (sumDiff < -100) {
						res += (x[ind - 1][posIndex] - x[ind - 1][negIndex]);
						continue;
					}
					double divider = 1 + exp(-sumDiff);
					//avoid overflow:
					if (divider < 1e10) {
						divider = divider * divider;
						res += (x[ind - 1][posIndex] - x[ind - 1][negIndex]) * exp(-sumDiff) / divider;
					}

				}

			}
			res /= (posInds.size() * negInds.size());
			res = -res; //because we need to minimize and auc we need to maximize
			res += REG_LAMBDA * 2 * params[ind];  //regularization
			return res;
		}
		else {
			//Not Supported Weights!!!
			for (size_t i = 0; i < posInds.size(); ++i)
			{
				int posIndex = posInds[i];
				for (size_t j = 0; j < negInds.size(); ++j)
				{
					int negIndex = negInds[j];
					double sumDiff = 0;
					for (size_t k = 1; k < params.size(); ++k) //param 0 should be zero
						sumDiff += params[k] * (x[k - 1][posIndex] - x[k - 1][negIndex]);
					sumDiff *= 1;
					if (sumDiff > 100) {
						res += 0;
						continue;
					}
					if (sumDiff < -100) {
						res += (x[ind - 1][posIndex] - x[ind - 1][negIndex]);
						continue;
					}
					double divider = 1 + exp(-sumDiff);
					//avoid overflow:
					if (divider < 1e10) {
						divider = divider * divider;
						res += (x[ind - 1][posIndex] - x[ind - 1][negIndex]) * exp(-sumDiff) / divider;
					}

				}

			}
			res /= (posInds.size() * negInds.size());
			res = -res; //because we need to minimize and auc we need to maximize
			res += REG_LAMBDA * 2 * params[ind];  //regularization
			return res;
		}
	};

	return func;
}
subGradientFunction  MedLinearModel::getSubGradientsSvm() {
	//This is subGradient in L2, for other loss function you need to change this function. (need fix for bias param, W{n+1})
	subGradientFunction func = [](int ind, const vector<double> &params, const vector<vector<float>> &x, const vector<float> &y, const vector<float> *weights) {
		double res = 0;
		if (weights == NULL || weights->empty()) {
			for (size_t i = 0; i < y.size(); ++i)
			{
				double fx = 1; //first param (ind ==0) is bais like x vector has 1 vector;
				if (ind > 0)
					fx = x[ind - 1][i];
				double diff = 1 - ((y[i] > 0) * 2 - 1) * params[ind] * fx;
				if (diff > 0)
					res += 1 - ((y[i] > 0) * 2 - 1) * fx;
			}

			res /= y.size();
			//add regularization:
			double reg = 0;
			if (REG_LAMBDA > 0) {
				double n_params = 0;
				for (size_t i = 0; i < params.size(); ++i)
					n_params += params[i] * params[i];
				n_params = sqrt(n_params);
				reg = -params[ind] / n_params;
			}

			return res + REG_LAMBDA * reg;
		}
		else {
			for (size_t i = 0; i < y.size(); ++i)
			{
				double fx = 1; //first param (ind ==0) is bais like x vector has 1 vector;
				if (ind > 0)
					fx = x[ind - 1][i];
				double diff = 1 - ((y[i] > 0) * 2 - 1) * params[ind] * fx;
				if (diff > 0)
					res += (1 - ((y[i] > 0) * 2 - 1) * fx) * weights->at(i);
			}

			res /= y.size();
			//add regularization:
			double reg = 0;
			if (REG_LAMBDA > 0) {
				double n_params = 0;
				for (size_t i = 0; i < params.size(); ++i)
					n_params += params[i] * params[i];
				n_params = sqrt(n_params);
				reg = -params[ind] / n_params;
			}

			return res + REG_LAMBDA * reg;
		}
	};

	return func;
}
double _linear_loss_target_auc(const vector<double> &preds, const vector<float> &y, const vector<float> *weights) {
	//WEIGHTS not Supported
	vector<vector<int>> targetToInd(2);
	for (size_t i = 0; i < y.size(); ++i)
		targetToInd[int(y[i] > 0)].push_back((int)i);

	double res = 0;
	for (size_t i = 0; i < targetToInd[1].size(); ++i)
	{
		for (size_t j = 0; j < targetToInd[0].size(); ++j)
		{
			double diffScores = preds[targetToInd[1][i]] - preds[targetToInd[0][j]];
			//res += 1.0 / (1.0 + exp(-10 * diffScores));
			if (SMOOTH) {
				diffScores *= 5;
				if (diffScores > -100 && diffScores < 100)  res += 1.0 / (1.0 + exp(-diffScores)); else if (diffScores >= 100) res += 1;
			}
			else {
				res += diffScores > REGULARIZATION_GEOM;
				res += 0.5*(diffScores == REGULARIZATION_GEOM);
				//res += diffScores;
			}
		}
	}
	res /= (targetToInd[1].size() * targetToInd[0].size());
	res = -res; //auc needs to be maximize
	return res;
}
double _linear_loss_step_auc(const vector<double> &preds, const vector<float> &y, const vector<double> &params, const vector<float> *weights) {
	//WEIGHTS not Supported
	double res = _linear_loss_target_auc(preds, y, weights);
	double nrm = 0;
	for (size_t i = 0; i < params.size(); ++i)
		nrm += params[i] * params[i];
	nrm /= 2;
	res += REG_LAMBDA * nrm; //not needed projecting to 1 after each iteration
	return res;
}
double _linear_loss_step_auc_fast(const vector<double> &preds, const vector<float> &y, const vector<float> *weights) {
	//WEIGHTS not supported
	vector<vector<int>> targetToInd(2);
	for (size_t i = 0; i < y.size(); ++i)
		targetToInd[(int)y[i]].push_back((int)i);

	double res = 0;
	int smp_cnt = 500;
	//random_device rd;
	//auto gen = mt19937(rd());
	std::default_random_engine gen;
	gen.seed( /* your seed for the RNG goes here */);
	uniform_int_distribution<> pos_rand(0, (int)targetToInd[1].size() - 1);
	uniform_int_distribution<> neg_rand(0, (int)targetToInd[0].size() - 1);
	for (size_t i = 0; i < smp_cnt; ++i)
	{
		int posInd = targetToInd[1][pos_rand(gen)];
		int negInd = targetToInd[0][neg_rand(gen)];
		double diffScores = preds[posInd] - preds[negInd];
		//res += 1.0 / (1.0 + exp(-diffScores));
		res += diffScores > 0;
		res += (diffScores == 0)* 0.5;
	}
	//res /= smp_cnt;
	res = -res; //auc needs to be maximize

	return res;
}
double _linear_loss_target_work_point(const vector<double> &preds, const vector<float> &y, const vector<float> *weights) {
	//WEIGHTS notr supported
	double res = 0;
	int totPos = 0;
	float deired_sen = (float)0.75; //Take AUC @ deired_sen (smooth local AUC)
	int local_points_diff = 1;
	for (size_t i = 0; i < y.size(); ++i)
	{
		totPos += int(y[i] > 0);
	}
	if (totPos == 0) {
		//can't be!! 
		res = INFINITY; //exception
		return res; //change to mssing value - take other sample - can't be
	}

	int stopCnt = int(totPos * deired_sen);
	vector<tuple<double, bool>> predSorted((int)preds.size());
	for (size_t i = 0; i < predSorted.size(); ++i)
	{
		predSorted[i] = tuple<double, bool>(preds[i], y[i] > 0);
	}
	sort(predSorted.begin(), predSorted.end());

	//calc AUC on reversed matchY:
	int posCnt = 0;
	int negCnt = 0;
	int bin_size = 0;
	for (int i = (int)predSorted.size() - 1; i >= 0; --i) {
		posCnt += get<1>(predSorted[i]);
		negCnt += !get<1>(predSorted[i]);
		if (posCnt >= stopCnt - local_points_diff) {
			res += float(negCnt) / (y.size() - totPos);
			++bin_size;
		}
		if (posCnt >= stopCnt + local_points_diff) {
			break;
		}
	}
	if (bin_size > 0)
		res /= bin_size;
	else
		res = 0.5; //unknown

	return res;
}
double _linear_loss_step_work_point(const vector<double> &preds, const vector<float> &y, const vector<double> &model_params, const vector<float> *weights) {
	float loss_val = (float)_linear_loss_target_work_point(preds, y, weights);
	double nrm = 0;
	if (REG_LAMBDA > 0) {
		for (size_t i = 0; i < model_params.size(); ++i)
		{
			nrm += model_params[i] * model_params[i];
		}
		nrm /= model_params.size();
	}

	return loss_val + REG_LAMBDA * nrm;
}
double _linear_loss_target_rmse(const vector<double> &preds, const vector<float> &y, const vector<float> *weights) {
	double res = 0;
	if (weights == NULL || weights->empty()) {
		for (size_t i = 0; i < y.size(); ++i)
			res += (y[i] - preds[i]) * (y[i] - preds[i]);
		res /= y.size();
		res = sqrt(res);
	}
	else {
		for (size_t i = 0; i < y.size(); ++i)
			res += (y[i] - preds[i]) * (y[i] - preds[i]) * weights->at(i);
		res /= y.size();
		res = sqrt(res);
	}
	return res;
}
double _linear_loss_step_rmse(const vector<double> &preds, const vector<float> &y, const vector<double> &model_params, const vector<float> *weights) {
	double res = _linear_loss_target_rmse(preds, y, weights);

	double reg = 0;
	if (REG_LAMBDA > 0) {
		for (size_t i = 0; i < model_params.size(); ++i)
			reg += model_params[i] * model_params[i];
		reg = sqrt(reg);
	}

	return res + REG_LAMBDA * reg;
}
double _linear_loss_target_svm(const vector<double> &preds, const vector<float> &y, const vector<float> *weights) {
	double res = 0;
	if (weights == NULL || weights->empty()) {
		for (size_t i = 0; i < y.size(); ++i)
		{
			double diff = 1 - (2 * (y[i] > 0) - 1) * preds[i];
			if (diff > 0)
				res += diff;
		}
		res /= y.size();
	}
	else {
		for (size_t i = 0; i < y.size(); ++i)
		{
			double diff = 1 - (2 * (y[i] > 0) - 1) * preds[i];
			if (diff > 0)
				res += diff * weights->at(i);
		}
		res /= y.size();
	}

	return res; //no reg - maybe count only accourcy beyond 1 and beyond 0
}
double _linear_loss_step_svm(const vector<double> &preds, const vector<float> &y, const vector<double> &model_params, const vector<float> *weights) {
	double res = _linear_loss_target_svm(preds, y, weights);

	double reg = 0;
	if (REG_LAMBDA > 0) {
		for (size_t i = 0; i < model_params.size(); ++i)
			reg += model_params[i] * model_params[i];
		reg = sqrt(reg);
	}

	return res + REG_LAMBDA * reg;
}

MedLinearModel::MedLinearModel() :
	PredictiveModel("LinearModel"), MedPredictor() {
	//model_params = vector<double>(numOdSignals + 1); //for bias
	mark_learn_finish = false;
	loss_function = _linear_loss_target_auc; //default target function, can be changed programitaclly
	loss_function_step = _linear_loss_step_auc; //default gradient function (with regularzation), can be changed programitaclly

	transpose_for_learn = false;
	normalize_y_for_learn = false;
	transpose_for_predict = false;
	features_count = 0;

	normalize_for_learn = false; //doing internal and not with MedMat to save normalization params for predict
	normalize_for_predict = false;
	classifier_type = MODEL_LINEAR_SGD;
	//sample_count = numOdSignals * 15;
	sample_count = -1; //default
	tot_steps = 10000;
	learning_rate = 3 * 1e-7;
	block_num = 1.0;
	norm_l1 = false;
	print_steps = 10;
	print_model = false;
	poly_degree = 1;
	min_cat = 1;
}

void MedLinearModel::print(const vector<string> &signalNames) const {
	cout << "Param0=" << model_params[0] << endl;

	vector<pair<double, int>> tps((int)signalNames.size());
	for (size_t i = 0; i < signalNames.size(); ++i) {
		tps[i].first = model_params[i + 1];
		tps[i].second = (int)i;
	}
	sort(tps.begin(), tps.end(), [](const pair<double, int>&a, const pair<double, int> &b) {
		if (abs(a.first) == abs(b.first))
			return a.second < b.second;
		return abs(a.first) > abs(b.first);
	});

	for (size_t i = 0; i < tps.size(); ++i)
		cout << "Param" << tps[i].second + 1 << " " << signalNames[tps[i].second] << "=" << tps[i].first << endl;

}

void MedLinearModel::set_normalization(const vector<float> &meanShift, const vector<float> &factors) {
	_meanShift = meanShift;
	_factor = factors;
}

void MedLinearModel::get_normalization(vector<float> &meanShift, vector<float> &factors) const {
	meanShift = _meanShift;
	factors = _factor;
}

void MedLinearModel::apply_normalization(vector<vector<float>> &input) const {
	for (size_t i = 0; i < input.size(); ++i)
	{
		for (size_t j = 0; j < input[i].size(); ++j)
		{
			if (input[i][j] != MED_MAT_MISSING_VALUE)
				input[i][j] = (input[i][j] - _meanShift[i]) / _factor[i];
			else
				input[i][j] = 0;
		}
	}
}

PredictiveModel *MedLinearModel::clone() const {
	MedLinearModel *copy = new MedLinearModel;
	copy->model_params.resize((int)model_params.size() - 1);
	copy->model_name = "LinearModel(" + to_string((int)model_params.size() - 1) + ")";

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> rand_gen;
	for (size_t i = 0; i < copy->model_params.size(); ++i)
		copy->model_params[i] = rand_gen(gen);
	copy->loss_function = loss_function;
	copy->loss_function_step = loss_function_step;
	copy->sample_count = sample_count;
	copy->tot_steps = tot_steps;
	copy->learning_rate = learning_rate;
	copy->block_num = block_num;
	copy->norm_l1 = norm_l1;

	//dont copy values of params and normalization - not need for now
	return copy;
}

int MedLinearModel::Predict(float *x, float *&preds, int nsamples, int nftrs) const {
	if (poly_degree == 1) {
#pragma omp parallel for
		for (int i = 0; i < nsamples; ++i)
		{
			double p = model_params[0];
			for (size_t k = 0; k < nftrs; ++k) {
				float val = x[static_cast<size_t>(i)*static_cast<size_t>(nftrs) + k];
				// has normalization in MedMat - but want to use same from train. when calling this function, it's always need normalizations
				if (normalize) {
					if (val == MED_MAT_MISSING_VALUE)
						val = 0;
					else
						val = (val - _meanShift[k]) / _factor[k];
				}
				p += val * model_params[k + 1];
			}
#pragma omp critical
			preds[i] = (float)p;
		}
	}
	else {
		vector<vector<float>> xData_degree(nftrs); //transposed
		for (size_t i = 0; i < xData_degree.size(); ++i)
		{
			xData_degree[i].resize(nsamples);
			for (size_t j = 0; j < nsamples; ++j)
				xData_degree[i][j] = x[j* nftrs + i];
		}
		vector<const float *> data(nftrs);
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = xData_degree[i].data();

		//vector<double> prds(nsamples);
		//poly_get_score(poly_degree, model_params, data, prds);
#pragma omp parallel for
		for (int i = 0; i < nsamples; ++i)
		{
			double p = poly_get_score(poly_degree, model_params, data, i);
#pragma omp critical
			preds[i] = (float)p;
		}
	}
	return 0;
}

void MedLinearModel::calc_feature_importance(vector<float> &features_importance_scores,
	const string &general_params, const MedFeatures *features) {
	features_importance_scores.resize(model_features.size());
	for (size_t i = 1; i < model_params.size(); ++i)
		features_importance_scores[i - 1] = abs(model_params[i]);
}

float _maxDiffVec(const vector<float> &vec) {
	if (vec.size() == 0)
		throw invalid_argument("vector can't  be empty");

	float mmax = vec[0];
	float mmin = vec[0];

	for (size_t i = 0; i < vec.size(); ++i)
	{
		if (mmax < vec[i]) {
			mmax = vec[i];
		}
		if (mmin > vec[i]) {
			mmin = vec[i];
		}
	}

	return mmax - mmin;
}

void _learnModel(SGD &learner, const vector<vector<float>> &xData, const vector<float> &yData, int categoryCnt,
	int T_Steps, int print_steps, double learnRate, int sampleSize, bool print_auc) {
	float h_size = (float)0.01;
	int numSteps = T_Steps;
	//float learn_rate = (float)0.0000001;
	float blocking_num = (float)sqrt((int)xData.size()) * 1;
	if (learner.get_blocking() == 0) {
		learner.set_blocking(blocking_num);
	}
	else {
		blocking_num = abs(learner.get_blocking());
	}

	learner.set_gradient_params(sampleSize, h_size, categoryCnt);

	float maxP = _maxDiffVec(xData[0]); //p = the maximal number in data x
	for (size_t i = 1; i < xData.size(); ++i)
	{
		float maxSignal = _maxDiffVec(xData[i]);
		if (maxSignal > maxP) {
			maxP = maxSignal;
		}
	}
	if (print_steps > 0)
		cout << "maxDiffSignal=" << maxP << " sample_count=" << sampleSize <<
		" cat_cnt=" << categoryCnt << " h=" << h_size << " blocking=" << blocking_num << endl;
	float blockDer = maxP / h_size;
	if (learnRate > 0) {
		learner.set_learing_rate((float)learnRate);
	}
	else {
		learner.set_learing(blocking_num, blockDer, T_Steps);
	}
	//learner.subGradientI = model->getSubGradients();
	//learner.subGradientI = ((LinearModel *)model)->getSubGradientsAUC(); 

	//learner.set_learing_rate(learner.get_learing_rate() / 5);
	if (print_steps > 0) {
		learner.output_num = T_Steps / print_steps;
		cout << "learning_rate = " << learner.get_learing_rate() << ", eppsilon=" << learner.get_learing_eppsilon(blocking_num, blockDer, T_Steps) << endl;
	}

	learner.Learn(xData, yData, numSteps, NULL, print_auc);
}

template<class T> T _avgVec(const vector<T> &vec) {
	T res = 0;
	int sz = 0;
	for (size_t i = 0; i < vec.size(); ++i)
	{
		if (vec[i] == MED_MAT_MISSING_VALUE) {
			continue;
		}
		res += vec[i];
		++sz;
	}
	if (sz == 0) {
		return MED_MAT_MISSING_VALUE;
	}
	return res / sz;
}

float _stdVec(const vector<float> &vec, float avg) {
	float res = 0;
	int sz = 0;
	for (size_t i = 0; i < vec.size(); ++i)
	{
		if (vec[i] == MED_MAT_MISSING_VALUE) {
			continue;
		}
		res += (vec[i] - avg)*(vec[i] - avg);
		++sz;
	}
	if (sz == 0) {
		return MED_MAT_MISSING_VALUE;
	}
	res /= sz;
	res = sqrt(res);

	return res;
}

void _normalizeSignalToAvg(vector<vector<float>> &xData, vector<float> &meanShift, vector<float> &factor) {
	meanShift.resize((int)xData.size());
	factor.resize((int)xData.size());
	for (size_t i = 0; i < xData.size(); ++i)
	{
		//fixOutlyers(xData[i]);
		float avg = _avgVec(xData[i]);
		meanShift[i] = avg;
		float std = _stdVec(xData[i], avg);
		factor[i] = std;
		for (size_t k = 0; k < xData[i].size(); ++k)
		{
			if (xData[i][k] != MED_MAT_MISSING_VALUE && std != 0) {
				xData[i][k] = (xData[i][k] - avg) / std; //z-Score
			}
			else {
				xData[i][k] = 0; //maybe Avg?
			}
		}
	}
}

int MedLinearModel::Learn(float *x, float *y, const float *w, int nsamples, int nftrs) {
	int sz_vec = 1;
	for (int i = 0; i < poly_degree; ++i)
		sz_vec *= (nftrs + i + 1);
	for (int i = 0; i < poly_degree; ++i)
		sz_vec /= (i + 1);

	model_params.resize(sz_vec); //first is for bias, the rest are polynom coef
	if (sample_count == -1)
		sample_count = 50 * nftrs;
	this->model_name = "LinearModel(" + to_string(nftrs) + ")" + "Poly_Deg=" + to_string(poly_degree);

	vector<float> avg_diff, factors;
	vector<float> yData(y, y + nsamples);
	vector<vector<float>> xData(sz_vec - 1);
	vector<vector<float>> xData_degree(nftrs);
	if (poly_degree == 1) {
		for (size_t i = 0; i < xData.size(); ++i)
		{
			xData[i].resize(nsamples);
			for (size_t j = 0; j < nsamples; ++j)
				xData[i][j] = x[j* nftrs + i];
		}
	}
	else {
		for (size_t i = 0; i < xData_degree.size(); ++i)
		{
			xData_degree[i].resize(nsamples);
			for (size_t j = 0; j < nsamples; ++j)
				xData_degree[i][j] = x[j* nftrs + i];
		}
		vector<const float *> ddata(nftrs);
		for (size_t i = 0; i < nftrs; ++i)
			ddata[i] = xData_degree[i].data();
		generate_data_learn(poly_degree, ddata, nsamples, xData);
	}
	if (normalize) {
		_normalizeSignalToAvg(xData, avg_diff, factors);
		set_normalization(avg_diff, factors);
	}
	SGD learner(this, loss_function);
	learner.subGradientI = NULL; //((LinearModel *)mdl)->getSubGradientsSvm();
	learner.norm_l1 = norm_l1;
	learner.set_blocking(block_num);
	//learner.set_model_precision(1e-5);
	learner.set_special_step_func(loss_function_step); //not in use if learner.subGradientI is not NULL

	mark_learn_finish = false;
	_learnModel(learner, xData, yData, min_cat, tot_steps, print_steps, learning_rate, sample_count, print_auc);
	mark_learn_finish = true;

	if (print_model) {
		vector<string> names = model_features;
		if (names.empty())
			names.resize(nftrs);
		print(names);
	}

	if (print_auc) {
		vector<double> preds;
		predict(xData, preds);
		cout << "########################" << endl;
		cout << "AUC AFTER Learn is : " << medial::performance::auc_q(preds, yData) << endl;
		cout << "########################" << endl;
	}

	return 0;
}

int MedLinearModel::set_params(map<string, string>& mapper) {
	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		//! [MedLinearModel::init]
		if (it->first == "sample_count")
			sample_count = stoi(it->second);
		else if (it->first == "tot_steps")
			tot_steps = stoi(it->second);
		else if (it->first == "learning_rate")
			learning_rate = stod(it->second);
		else if (it->first == "block_num")
			block_num = stof(it->second);
		else if (it->first == "norm_l1")
			norm_l1 = stoi(it->second) > 0;
		else if (it->first == "normalize")
			normalize = stoi(it->second) > 0;
		else if (it->first == "print_steps")
			print_steps = stoi(it->second);
		else if (it->first == "poly_degree")
			poly_degree = stoi(it->second);
		else if (it->first == "print_model")
			print_model = stoi(it->second) > 0;
		else if (it->first == "min_cat")
			min_cat = stoi(it->second);
		else if (it->first == "loss_function") {
			if (it->second == "rmse") {
				loss_function = _linear_loss_target_rmse;
				loss_function_step = _linear_loss_step_rmse;
				min_cat = 0; //regression
			}
			else if (it->second == "auc") {
				loss_function = _linear_loss_target_auc;
				loss_function_step = _linear_loss_step_auc;
			}
			else if (it->second == "svm") {
				loss_function = _linear_loss_target_svm;
				loss_function_step = _linear_loss_step_svm;
				print_auc = true;
			}
			else if (it->second == "work_point") {
				loss_function = _linear_loss_target_work_point;
				loss_function_step = _linear_loss_step_work_point;
				print_auc = true;
			}
			else
				invalid_argument("MedLinearModel::set_params - Unknown loss_function \"" + it->second + "\"");
		}
		else
			throw invalid_argument("MedLinearModel::set_params - Unknown parameter \"" + it->first + "\"");
		//! [MedLinearModel::init]
	}

	return 0;
}