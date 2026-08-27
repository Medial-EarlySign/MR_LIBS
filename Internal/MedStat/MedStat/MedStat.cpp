//
// MedStat
//

#include "MedStat.h"

#include <MedUtils/MedUtils/MedGlobalRNG.h>
#include <MedUtils/MedUtils/MedGenUtils.h>
#include <MedUtils/MedUtils/MedUtils.h>
#include <Logger/Logger/Logger.h>
#include <boost/math/distributions/students_t.hpp>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDSTAT
#define LOCAL_LEVEL	LOG_DEF_LEVEL

extern MedLogger global_logger;

// Performance functions: analysis of two vectors.
//...............................................................................................................................
// Pearson Correlation
template <typename T> float _pearson_corr(const vector<T> &v1, const vector<T> &v2, const vector<float>& w) {

	if (v1.size() != v2.size() || w.size() != v1.size())
		MTHROW_AND_ERR("Length inconsistency in calculating Pearson correlation\n");

	int len = (int)v1.size();

	if (len == 0)
		return -2.0;

	double sx, sy, sxy, sxx, syy, n;

	sx = sy = sxy = sxx = syy = n = 0;

	double fact = 1e-5;
	for (int i = 0; i < len; i++) {
		sx += fact * w[i] * v1[i];
		sy += fact * w[i] * v2[i];
		sxx += fact * w[i] * v1[i] * v1[i];
		syy += fact * w[i] * v2[i] * v2[i];
		sxy += fact * w[i] * v1[i] * v2[i];
		n += w[i];
	}

	sx /= fact * n;
	sy /= fact * n;
	sxx /= fact * n;
	syy /= fact * n;
	sxy /= fact * n;

	double c1 = sxy - sx * sy;
	double c2 = sqrt(sxx - sx * sx);
	double c3 = sqrt(syy - sy * sy);

	double epsilon = 1e-12;
	if (c2 < epsilon || c3 < epsilon)
		return 0;
	return (float)(c1 / (c2*c3));
}
template float _pearson_corr<float>(const vector<float> &v1, const vector<float> &v2, const vector<float>& w);
template float _pearson_corr<double>(const vector<double> &v1, const vector<double> &v2, const vector<float>& w);

template <typename T> float medial::performance::pearson_corr_without_cleaning(const vector<T> &v1, const vector<T> &v2, const vector<float> *weights) {

	if (weights == NULL) {
		vector<float> w(v1.size(), 1.0);
		return _pearson_corr(v1, v2, w);
	}
	else
		return _pearson_corr(v1, v2, *weights);

}

template float medial::performance::pearson_corr_without_cleaning<float>(const vector<float> &v1, const vector<float> &v2, const vector<float> *weights);
template float medial::performance::pearson_corr_without_cleaning<double>(const vector<double> &v1, const vector<double> &v2, const vector<float> *weights);

template <typename T> float medial::performance::pearson_corr(const vector<T> &v1, const vector<T> &v2, T missing_value, int& n, const vector<float> *weights) {

	vector<T> clean_v1, clean_v2;
	vector<float> clean_w;
	for (unsigned int i = 0; i < v1.size(); i++) {
		if (v1[i] != missing_value && v2[i] != missing_value) {
			clean_v1.push_back(v1[i]);
			clean_v2.push_back(v2[i]);
			if (weights != NULL)
				clean_w.push_back((*weights)[i]);
		}
	}

	n = (int)clean_v1.size();

	if (weights == NULL)
		clean_w.resize(clean_v1.size(), 1.0);
	return _pearson_corr(clean_v1, clean_v2, clean_w);

}
template float medial::performance::pearson_corr<float>(const vector<float> &v1, const vector<float> &v2, float missing_value, int& n, const vector<float> *wrights);
template float medial::performance::pearson_corr<double>(const vector<double> &v1, const vector<double> &v2, double missing_value, int& n, const vector<float> *weights);

// Spearman Correlation
//...............................................................................................................................
template <typename T, typename S> float medial::performance::spearman_corr_without_cleaning(const vector<T> &v1, const vector<S> &v2, const vector<float> *weights) {

	if (v1.size() != v2.size())
		MTHROW_AND_ERR("Length inconsistency in calculating Spearman correlation\n");

	int len = (int)v1.size();

	// Sort v1 and v2
	vector<pair<float, int> > v1_i(len), v2_i(len);

	for (int i = 0; i < len; i++) {
		v1_i[i] = { (float)v1[i],i };
		v2_i[i] = { (float)v2[i],i };
	}

	sort(v1_i.begin(), v1_i.end(), [](const pair<float, int>& left, const pair<float, int>& right) { return left.first < right.first; });
	sort(v2_i.begin(), v2_i.end(), [](const pair<float, int>& left, const pair<float, int>& right) { return left.first < right.first; });

	// Pearson correlation of indices
	vector<float> v1_v(len), v2_v(len);

	for (int i = 0; i < len; i++) {
		v1_v[v1_i[i].second] = (float)i;
		v2_v[v2_i[i].second] = (float)i;
	}

	return medial::performance::pearson_corr_without_cleaning(v1_v, v2_v, weights);

}
template float medial::performance::spearman_corr_without_cleaning<float, float>(const vector<float> &v1, const vector<float> &v2, const vector<float> *weights);
template float medial::performance::spearman_corr_without_cleaning<float, double>(const vector<float> &v1, const vector<double> &v2, const vector<float> *weights);
template float medial::performance::spearman_corr_without_cleaning<double, float>(const vector<double> &v1, const vector<float> &v2, const vector<float> *weights);
template float medial::performance::spearman_corr_without_cleaning<double, double>(const vector<double> &v1, const vector<double> &v2, const vector<float> *weights);

template <typename T> float medial::performance::spearman_corr(const vector<T> &v1, const vector<T> &v2, T missing_val, int &n, const vector<float> *weights) {

	vector<T> clean_v1, clean_v2;
	vector<float> clean_w;
	for (unsigned int i = 0; i < v1.size(); i++) {
		if (v1[i] != missing_val && v2[i] != missing_val) {
			clean_v1.push_back(v1[i]);
			clean_v2.push_back(v2[i]);
			if (weights != NULL)
				clean_w.push_back((*weights)[i]);
		}
	}

	n = (int)clean_v1.size();
	if (weights != NULL)
		return medial::performance::spearman_corr_without_cleaning(clean_v1, clean_v2, &clean_w);
	else
		return medial::performance::spearman_corr_without_cleaning(clean_v1, clean_v2);

}
template float medial::performance::spearman_corr<float>(const vector<float> &v1, const vector<float> &v2, float missing_val, int &n, const vector<float> *weights);
template float medial::performance::spearman_corr<double>(const vector<double> &v1, const vector<double> &v2, double missing_val, int &n, const vector<float> *weights);

// Jaccard
void medial::performance::get_jaccard_matrix(int n, vector<vector<float>>& jaccard_dist) {

	jaccard_dist.resize(n, vector<float>(n));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			jaccard_dist[i][j] = jaccard_distance(i, j);

	}
}

float medial::performance::jaccard_similarity(int val1, int val2)
{
	float num = std::bitset<16>(val1 & val2).count();
	float den = std::bitset<16>(val1 | val2).count();
	if (den != 0)
		return num / den;
	else
		if (num == 0)
			return 1.;
		else
			return 0.;
}

float medial::performance::jaccard_distance(int val1, int val2)
{
	return 1. - jaccard_similarity(val1, val2);
}

// RMSE
//...............................................................................................................................
template <typename T> float medial::performance::rmse_without_cleaning(const vector<T> &preds, const vector<T> &y, const vector<float> *weights) {

	if (preds.size() != y.size() || (weights != NULL && preds.size() != weights->size()))
		MTHROW_AND_ERR("Length inconsistency in calculating RMSE\n");

	double res = 0;
	if (weights == NULL) {
		for (size_t i = 0; i < y.size(); ++i)
			res += static_cast<double>((y[i] - preds[i])) * static_cast<double>((y[i] - preds[i]));
		res /= y.size();
		res = sqrt(res);
		return (float)res;
	}
	else {
		double cnt = 0;
		for (size_t i = 0; i < y.size(); ++i) {
			res += static_cast<double>((*weights)[i]) * static_cast<double>((y[i] - preds[i])) * static_cast<double>((y[i] - preds[i]));
			cnt += (*weights)[i];
		}
		res /= cnt;
		res = sqrt(res);
		return (float)res;
	}
}
template float medial::performance::rmse_without_cleaning<float>(const vector<float> &preds, const vector<float> &y, const vector<float> *weights);
template float medial::performance::rmse_without_cleaning<double>(const vector<double> &preds, const vector<double> &y, const vector<float> *weights);

template <typename T> float medial::performance::rmse(const vector<T> &preds, const vector<T> &y, T missing_val, int &n, const vector<float> *weights) {

	vector<T> clean_preds, clean_y;
	vector<float> clean_w;
	for (unsigned int i = 0; i < preds.size(); i++) {
		if (preds[i] != missing_val && y[i] != missing_val) {
			clean_preds.push_back(preds[i]);
			clean_y.push_back(y[i]);
			if (weights != NULL) clean_w.push_back((*weights)[i]);
		}
	}

	n = (int)clean_preds.size();
	if (weights != NULL)
		return medial::performance::rmse_without_cleaning(clean_preds, clean_y, &clean_w);
	else
		return medial::performance::rmse_without_cleaning(clean_preds, clean_y);
}
template float medial::performance::rmse<float>(const vector<float> &preds, const vector<float> &y, float missing_val, int &n, const vector<float> *weights);
template float medial::performance::rmse<double>(const vector<double> &preds, const vector<double> &y, double missing_val, int &n, const vector<float> *weights);

// L1 distance
//...............................................................................................................................
template <typename T> float medial::performance::L1_dist_without_cleaning(const vector<T> &preds, const vector<T> &y, const vector<float> *weights) {

	if (preds.size() != y.size() || (weights != NULL && preds.size() != weights->size()))
		MTHROW_AND_ERR("Length inconsistency in calculating L1 distance\n");

	double res = 0;
	if (weights == NULL) {
		for (size_t i = 0; i < y.size(); ++i)
			res += abs(y[i] - preds[i]);
		return (float)(res / y.size());
	}
	else {
		double cnt = 0;
		for (size_t i = 0; i < y.size(); ++i) {
			res += static_cast<double>((*weights)[i]) * static_cast<double>(abs(y[i] - preds[i]));
			cnt += (*weights)[i];
		}
		return (float)(res / cnt);
	}
}
template float medial::performance::L1_dist_without_cleaning<float>(const vector<float> &preds, const vector<float> &y, const vector<float> *weights);
template float medial::performance::L1_dist_without_cleaning<double>(const vector<double> &preds, const vector<double> &y, const vector<float> *weights);

template <typename T> float medial::performance::L1_dist(const vector<T> &preds, const vector<T> &y, T missing_val, int &n, const vector<float> *weights) {

	vector<T> clean_preds, clean_y;
	vector<float> clean_w;
	for (unsigned int i = 0; i < preds.size(); i++) {
		if (preds[i] != missing_val && y[i] != missing_val) {
			clean_preds.push_back(preds[i]);
			clean_y.push_back(y[i]);
			if (weights != NULL) clean_w.push_back((*weights)[i]);
		}
	}

	n = (int)clean_preds.size();
	if (weights != NULL)
		return medial::performance::L1_dist_without_cleaning(clean_preds, clean_y, &clean_w);
	else
		return medial::performance::L1_dist_without_cleaning(clean_preds, clean_y);
}
template float medial::performance::L1_dist<float>(const vector<float> &preds, const vector<float> &y, float missing_val, int &n, const vector<float> *weights);
template float medial::performance::L1_dist<double>(const vector<double> &preds, const vector<double> &y, double missing_val, int &n, const vector<float> *weights);

// Relative L1 distance
//...............................................................................................................................
template <typename T> float medial::performance::relative_L1_dist_without_cleaning(const vector<T> &preds, const vector<T> &y, const vector<float> *weights) {

	if (preds.size() != y.size() || (weights != NULL && preds.size() != weights->size()))
		MTHROW_AND_ERR("Length inconsistency in calculating relative-L1 distance\n");

	if (preds.size() != y.size() || (weights != NULL && preds.size() != weights->size()))
		MTHROW_AND_ERR("Length inconsistency in calculating RMSE\n");

	double res = 0;
	double cnt = 0;
	if (weights == NULL) {
		for (size_t i = 0; i < y.size(); ++i) {
			if (preds[i] != 0) {
				res += abs(y[i] - preds[i]);
				cnt += 1.0;
			}
		}
	}
	else {
		for (size_t i = 0; i < y.size(); ++i) {
			if (preds[i] != 0) {
				res += static_cast<double>((*weights)[i]) * static_cast<double>(abs(y[i] - preds[i]));
				cnt += (*weights)[i];
			}
		}
	}
	return (float)(res / cnt);
}
template float medial::performance::relative_L1_dist_without_cleaning<float>(const vector<float> &preds, const vector<float> &y, const vector<float> *weights);
template float medial::performance::relative_L1_dist_without_cleaning<double>(const vector<double> &preds, const vector<double> &y, const vector<float> *weights);

template <typename T> float medial::performance::relative_L1_dist(const vector<T> &preds, const vector<T> &y, T missing_val, int &n, const vector<float> *weights) {

	vector<T> clean_preds, clean_y;
	vector<float> clean_w;
	for (unsigned int i = 0; i < preds.size(); i++) {
		if (preds[i] != missing_val && y[i] != missing_val) {
			clean_preds.push_back(preds[i]);
			clean_y.push_back(y[i]);
			if (weights != NULL) clean_w.push_back((*weights)[i]);
		}
	}

	n = (int)clean_preds.size();
	if (weights != NULL)
		return medial::performance::relative_L1_dist_without_cleaning(clean_preds, clean_y, &clean_w);
	else
		return medial::performance::relative_L1_dist_without_cleaning(clean_preds, clean_y);
}
template float medial::performance::relative_L1_dist<float>(const vector<float> &preds, const vector<float> &y, float missing_val, int &n, const vector<float> *weights);
template float medial::performance::relative_L1_dist<double>(const vector<double> &preds, const vector<double> &y, double missing_val, int &n, const vector<float> *weights);

// Accuracy
//...............................................................................................................................
template <typename T> float medial::performance::accuracy(const vector<T> &preds, const vector<float> &y, const vector<float> *weights) {
	double res = 0;
	if (weights == NULL || weights->empty()) {
		for (size_t i = 0; i < y.size(); ++i)
			res += ((T)y[i] == preds[i]);
		return float(res / y.size());
	}
	else {
		double cnt = 0;
		for (size_t i = 0; i < y.size(); ++i) {
			res += (*weights)[i] * ((T)y[i] == preds[i]);
			cnt += (*weights)[i];
		}
		return float(res / cnt);
	}
}
template float medial::performance::accuracy<float>(const vector<float> &preds, const vector<float> &y, const vector<float> *weights);
template float medial::performance::accuracy<double>(const vector<double> &preds, const vector<float> &y, const vector<float> *weights);

// Accuracy up to epsilon
//...............................................................................................................................
template <typename T> float medial::performance::approx_accuracy(const vector<T> &preds, const vector<float> &y, T epsilon, const vector<float> *weights) {
	double res = 0;
	if (weights == NULL || weights->empty()) {
		for (size_t i = 0; i < y.size(); ++i) {
			if (abs((double)(preds[i] - (T)y[i])) <= epsilon)
				res++;
		}
		return float(res / y.size());
	}
	else {
		double cnt = 0;
		for (size_t i = 0; i < y.size(); ++i) {
			if (abs((double)(preds[i] - (T)y[i])) <= epsilon)
				res += (*weights)[i];
			cnt += (*weights)[i];
		}
		return float(res / cnt);
	}

}
template float medial::performance::approx_accuracy<float>(const vector<float> &preds, const vector<float> &y, float epsilon, const vector<float> *weights);
template float medial::performance::approx_accuracy<double>(const vector<double> &preds, const vector<float> &y, double epsilon, const vector<float> *weights);

// Kendall's Rank Correlation
//...............................................................................................................................
//iterative algorithm for calculating Kendall Tau
//v1,v2 should be two equal length vectors, which may be of any type that can be cast to double (float, int, etc.)
//is01Vec1, is01Vec2 - are v1, v2 0/1 vectors, respectively. It is the user's responsibility that this is correct
//returns the KendallTau vector, or < -1.0 on bad input
//in general, ties are considered as not contributing to the score, while the tied pairs are counted.
//linear-time algorithm used if both vectors are 0/1. Otherwise, the algorithm is nlog(n), where n is the v1/2.size().
//If not both are 0/1 vectors, random noise is used to break ties.
template <typename T, typename S> double medial::performance::kendall_tau_without_cleaning(const vector<T>& _v1, const vector<S>& _v2, bool is01Vec1, bool is01Vec2) {

	if (_v1.size() != _v2.size())
		MTHROW_AND_ERR("Length inconsistency in calculating Kendall rank correlation\n");

	if (_v1.size() <= 1)
		return -2.0;

	vector<double> v1(_v1.begin(), _v1.end());
	vector<double> v2(_v2.begin(), _v2.end());

	int len = (int)v1.size();

	long long nPairs = (long long)len * (long long)(len - 1) / 2;

	//case 1 - both vectors are 0/1. in this case, (n11*n00 - n10*n01) / nPairs

	if (is01Vec1 && is01Vec2) {
		long long n10 = 0, n01 = 0, n11 = 0;

		for (int i = 0; i < len; ++i) {
			if (v1[i] == 1) {
				if (v2[i] == 1)
					n11++;
				else
					n10++;
			}
			else {//v1[i] == 0
				if (v2[i] == 1)
					n01++;
			}
		}

		long long n00 = (long long)len - n11 - n10 - n01;

		return (double)(n11*n00 - n01 * n10) / (double)nPairs;
	}

	//random mechanism used in resolving ties
	default_random_engine gen;
	unsigned int seed = 13;
	gen.seed(seed);
	uniform_real_distribution<double> dist(-0.25, 0.25); //can only change order for equal integers, not different ones

	long long nBadPairs = 0; //pairs that violate order. 

	//case 2 - one vector is 0/1. WLOG it is v2 (otherwise we swap)
	if (is01Vec1 && !is01Vec2) {
		swap(v1, v2);
		swap(is01Vec1, is01Vec2);
	}

	if (!is01Vec1 && is01Vec2) {
		//ties are resolved fairly. we add [-0.25,0.25] random noise, after replacing the values with (tied) ranks		
		map<double, int> m;
		for (auto x : v1)
			m[x] = 1;//dummy value

		int i = 0;
		for (auto& x : m)
			x.second = i++;//translate value to rank. Thus distinct values are separated by at least 1

		for (auto& x : v1)
			x = (double)(m[x]) + dist(gen);

		vector<pair<double, double> > p; p.reserve(len);

		for (int i = 0; i < len; ++i)
			p.push_back(pair<double, double>(v1[i], v2[i]));

		//sort by the first vector
		sort(p.begin(), p.end());

		//the number of reversals is the rank sum of the n0 0-elements in v[2] minus n0*(n0-1)/2
		int n0 = 0;
		for (int i = 0; i < len; ++i) {
			if (p[i].second == 0) {
				nBadPairs += i;
				n0++;
			}
		}

		nBadPairs -= ((long long)n0*(long long)(n0 - 1) / 2);

		return (double)((long long)n0*(long long)(len - n0) - 2 * nBadPairs) / (double)nPairs;
	}

	//at this point, neither v1 nor v2 is 0/1

	//ties in both v1 and v2 are resolved fairly. we add [-0.25,0.25] random noise, after replacing the values with (tied) ranks

	map<double, int> m;
	for (auto x : v1)
		m[x] = 1;//dummy value

	for (auto x : v2)
		m[x] = 1;

	int i = 0;
	for (auto& x : m)
		x.second = i++;//translate value to rank

	for (auto& x : v1)
		x = (double)(m[x]) + dist(gen);

	for (auto& x : v2)
		x = (double)(m[x]) + dist(gen);

	//fast algorithm: 
	//basic idea: if we pair the values of the two vectors, sort the pairs by one coordinate, then replace its values with 0,2,..len-1, 
	//and then sort by the second, then we are reduced to counting reversals of the indices (integers) in the first coordinate. 
	//this is done recursively as follows:
	//we start with a range of size len of indices from 0 to len-1. Locate the subset with the values above len/2 (S1). Denote the other subset by S0. 
	//Reversals are either "inter" (one element in each set) or "intra" (two elements inside the same set). We count the "inter" reversals, and then
	//move S0 to the first half of the range and S1 to the second half without changing their relative internal order.
	//We subtract a constant from the values in each subrange (half) so that the values are again 0 - subrange length, and we can call the function recursively.
	//the recursive calls will get the "intra" reversals, and together we get the total reversals. 
	//the heart if the calculation is realizing the the "inter" count is a simple function of the rank sum of S0 elements inside the range; if their positions are 
	//i_0, ..., i_k-1, then the inter count is sum_{j=0}^{k-1} (i_j - j).
	//The recursion is simulated by adding each new subrange to a "to do" stack, instead of a recursive call stack. We continue until all subranges are processed.
	//Note that counting "intra" reversals in a range is independent of all other ranges, so the order of processing the subranges is immaterial.

	vector<pair<double, double> > p; p.reserve(len);

	for (int i = 0; i < len; ++i)
		p.push_back(pair<double, double>(v1[i], v2[i]));

	//sort by the first vector
	sort(p.begin(), p.end(), ComparePairByFirst<double, double>());

	//translate it to rank
	for (int i = 0; i < len; ++i)
		p[i].first = i;

	//sort by the second vector. We will then need only count reversals in the first vector
	//which has values [0,len)
	sort(p.begin(), p.end(), ComparePairBySecond<double, double>());

	//we now want to count the number of reversals in the first coordinate
	vector<int> res; res.reserve(len);

	for (int i = 0; i < len; ++i)
		res.push_back((int)(p[i].first));

	//the algorithm requires two vectors as workspaces
	vector<int> ws[2];
	ws[0].reserve(len);
	ws[1].reserve(len);

	//use a stack to emulate recursion calls iteratively
	//each pair is <start point, length> for a range inside res
	//which requires processing
	vector<pair<int, int> > stack; stack.reserve(len);
	stack.push_back(pair<int, int>(0, len)); //initially, the whole range requires processing

											 //while there are sub-ranges of res to process
	while (!stack.empty()) {
		//	for (int i = 0; i < stack.size(); ++i)
		//		cout << "(" << stack[i].first << ", " << stack[i].second << ") ";
		//	cout << endl;

		//pop the last element
		pair<int, int> inds = stack.back();
		int n = inds.second;
		stack.pop_back();
		ws[0].clear(); ws[1].clear();

		//we now handle the range indicated by inds
		if (n <= 1)
			continue;

		//the lowest n/2 elements and the highest n - n/2 elements are separated
		//the violations *between* the two sets are a function of the rank sum of the former set

		//go over elements in the range (in the res vector, from indices inds.first to inds.first + inds.second-1)
		for (int i = 0; i < n; ++i) {
			int ind = (int)inds.first + i;
			if (res[ind] >= n / 2)
				ws[1].push_back(res[ind]);
			else {
				nBadPairs += i;
				ws[0].push_back(res[ind]);
			}
		}
		nBadPairs -= ((long long)(ws[0].size()) * (long long)(ws[0].size() - 1) / 2);

		//copy back to res
		for (int i = 0; i < n / 2; ++i)
			res[inds.first + i] = ws[0][i];

		stack.push_back(pair<int, int>(inds.first, n / 2));

		for (int i = n / 2; i < n; ++i)
			res[inds.first + i] = ws[1][i - n / 2] - n / 2;

		stack.push_back(pair<int, int>(inds.first + n / 2, n - n / 2));
	}

	return 1.0 - 2.0 * (double)nBadPairs / (double)nPairs;
}
template double medial::performance::kendall_tau_without_cleaning<float, float>(const vector<float>& _v1, const vector<float>& _v2, bool is01Vec1, bool is01Vec2);
template double medial::performance::kendall_tau_without_cleaning<float, double>(const vector<float>& _v1, const vector<double>& _v2, bool is01Vec1, bool is01Vec2);
template double medial::performance::kendall_tau_without_cleaning<double, float>(const vector<double>& _v1, const vector<float>& _v2, bool is01Vec1, bool is01Vec2);
template double medial::performance::kendall_tau_without_cleaning<double, double>(const vector<double>& _v1, const vector<double>& _v2, bool is01Vec1, bool is01Vec2);

template <typename T> double medial::performance::kendall_tau(const vector<T> &v1, const vector<T> &v2, T missing_value, int& n, bool is01Vec1, bool is01Vec2) {

	vector<T> clean_v1, clean_v2;
	vector<float> clean_w;
	for (unsigned int i = 0; i < v1.size(); i++) {
		if (v1[i] != missing_value && v2[i] != missing_value) {
			clean_v1.push_back(v1[i]);
			clean_v2.push_back(v2[i]);
		}
	}

	n = (int)clean_v1.size();
	return medial::performance::kendall_tau_without_cleaning(v1, v2, is01Vec1, is01Vec2);

}
template double medial::performance::kendall_tau<float>(const vector<float> &v1, const vector<float> &v2, float missing_value, int& n, bool is01Vec1, bool is01Vec2);
template double medial::performance::kendall_tau<double>(const vector<double> &v1, const vector<double> &v2, double missing_value, int& n, bool is01Vec1, bool is01Vec2);

// Efficient version for quantized vectors
template <typename T, typename S> double medial::performance::kendall_tau_without_cleaning_q(const vector<T>& preds, const vector<S>& y, const vector<float> *weights) {

	double tau = 0, cnt = 0;
	if (weights == NULL || weights->empty()) {
		unordered_map<S, vector<T>> label_to_scores;
		for (size_t i = 0; i < y.size(); ++i)
			label_to_scores[y[i]].push_back(preds[i]);
		for (auto it = label_to_scores.begin(); it != label_to_scores.end(); ++it)
			sort(it->second.begin(), it->second.end());
		for (auto it = label_to_scores.begin(); it != label_to_scores.end(); ++it)
		{
			auto bg = it;
			++bg;
			vector<T> *preds = &it->second;
			int pred_i_bigger;
			double pred_i_smaller;
			for (auto jt = bg; jt != label_to_scores.end(); ++jt)
			{
				vector<T> *preds_comp = &jt->second;
				double p_size = (double)preds_comp->size();
				for (T pred : *preds)
				{
					pred_i_bigger = medial::process::binary_search_position(preds_comp->data(), preds_comp->data() + preds_comp->size() - 1, pred);
					pred_i_smaller = p_size - medial::process::binary_search_position_last(preds_comp->data(), preds_comp->data() + preds_comp->size() - 1, pred);
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

		if (cnt > 1)
			tau /= cnt;

		return tau;
	}
	unordered_map<S, vector<T>> label_to_scores;
	unordered_map<S, vector<pair<int, T>>> label_to_scores_w;
	for (size_t i = 0; i < y.size(); ++i)
		label_to_scores[y[i]].push_back(preds[i]);
	for (size_t i = 0; i < y.size(); ++i)
		label_to_scores_w[y[i]].push_back(pair<int, T>((int)i, preds[i]));
	for (auto it = label_to_scores_w.begin(); it != label_to_scores_w.end(); ++it)
		sort(it->second.begin(), it->second.end(), ComparePairBySecond<int, T>());
	for (auto it = label_to_scores.begin(); it != label_to_scores.end(); ++it)
		sort(it->second.begin(), it->second.end());

	vector<double> group_weights(label_to_scores.size());
	vector<vector<double>> group_weights_cumsum(label_to_scores.size());
	int iter = 0;
	for (auto it = label_to_scores_w.begin(); it != label_to_scores_w.end(); ++it) {
		for (size_t i = 0; i < it->second.size(); ++i) {
			group_weights[iter] += (*weights)[it->second[i].first];
			group_weights_cumsum[iter].push_back((*weights)[it->second[i].first]);
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
		vector<T> *preds = &it->second;
		//double i_size = preds->size();
		double i_size = group_weights[iter];
		double pred_i_bigger;
		double pred_i_smaller;
		int pred_i_bigger_i;
		int pred_i_smaller_i;
		int inside_group_idx = iter + 1;
		for (auto jt = bg; jt != label_to_scores.end(); ++jt)
		{
			vector<T> *preds_comp = &jt->second;
			//double p_size = (double)preds_comp->size();
			double p_size = group_weights[inside_group_idx];
			for (T pred : *preds)
			{
				pred_i_bigger_i = medial::process::binary_search_position(preds_comp->data(), preds_comp->data() + preds_comp->size() - 1, pred);
				pred_i_smaller_i = medial::process::binary_search_position_last(preds_comp->data(), preds_comp->data() + preds_comp->size() - 1, pred);
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
			cnt += p_size * i_size;
			++inside_group_idx;
		}
		++iter;
	}

	if (cnt > 0)
		tau /= cnt;

	return tau;
}
template double medial::performance::kendall_tau_without_cleaning_q<double, double>(const vector<double>& preds, const vector<double>& y, const vector<float> *weights);
template double medial::performance::kendall_tau_without_cleaning_q<float, float>(const vector<float>& preds, const vector<float>& y, const vector<float> *weights);
template double medial::performance::kendall_tau_without_cleaning_q<double, float>(const vector<double>& preds, const vector<float>& y, const vector<float> *weights);
template double medial::performance::kendall_tau_without_cleaning_q<float, double>(const vector<float>& preds, const vector<double>& y, const vector<float> *weights);

template <typename T, typename S> double medial::performance::kendall_tau_q(const vector<T> &v1, const vector<S> &v2, T missing_val1, S missing_val2, int& n,
	const vector<float> *weights) {

	vector<T> clean_v1, clean_v2;
	vector<float> clean_w;
	for (unsigned int i = 0; i < v1.size(); i++) {
		if (v1[i] != missing_val1 && v2[i] != missing_val2) {
			clean_v1.push_back(v1[i]);
			clean_v2.push_back(v2[i]);
			if (weights != NULL)
				clean_w.push_back((*weights)[i]);
		}
	}

	n = (int)clean_v1.size();
	if (weights != NULL)
		return medial::performance::kendall_tau_without_cleaning_q(clean_v1, clean_v2, &clean_w);
	else
		return medial::performance::kendall_tau_without_cleaning_q(clean_v1, clean_v2);
}
template double medial::performance::kendall_tau_q<float, float>(const vector<float> &v1, const vector<float> &v2, float missing_val1, float missing_val2, int& n,
	const vector<float> *weights);
template double medial::performance::kendall_tau_q<float, double>(const vector<float> &v1, const vector<double> &v2, float missing_val1, double missing_val2, int& n,
	const vector<float> *weights);
template double medial::performance::kendall_tau_q<double, float>(const vector<double> &v1, const vector<float> &v2, double missing_val1, float missing_val2, int& n,
	const vector<float> *weights);
template double medial::performance::kendall_tau_q<double, double>(const vector<double> &v1, const vector<double> &v2, double missing_val1, double missing_val2, int& n,
	const vector<float> *weights);

// Mutual information of binned-vectors
//.........................................................................................................................................
float medial::performance::mutual_information(const vector<float>& x, const vector<float>& y, int &n) {

	// Sanity
	if (y.size() != x.size())
		MTHROW_AND_ERR("Size mismatch in calculating mutual information\n");
	//transofrm into bin idx:
	unordered_map<float, int> val_to_ind_x, val_to_ind_y;
	for (float v : x)
	{
		if (val_to_ind_x.find(v) == val_to_ind_x.end()) {
			int curr_sz = (int)val_to_ind_x.size();
			val_to_ind_x[v] = curr_sz;
		}
	}
	for (float v : y)
	{
		if (val_to_ind_y.find(v) == val_to_ind_y.end()) {
			int curr_sz = (int)val_to_ind_y.size();
			val_to_ind_y[v] = curr_sz;
		}
	}

	vector<int> bins_x(x.size()), bins_y(y.size());
	for (size_t i = 0; i < bins_x.size(); ++i)
	{
		bins_x[i] = val_to_ind_x.at(x[i]);
		bins_y[i] = val_to_ind_y.at(y[i]);
	}

	return mutual_information(bins_x, bins_y, n);
}

float medial::performance::mutual_information(vector<int>& x, vector<int>& y, int &n) {

	// Sanity
	if (y.size() != x.size())
		MTHROW_AND_ERR("Size mismatch in calculating mutual information\n");

	// Count bins
	int nXbins = 0, nYbins = 0;
	for (unsigned int i = 0; i < x.size(); i++) {
		if (x[i] + 1 > nXbins)
			nXbins = x[i] + 1;
		if (y[i] + 1 > nYbins)
			nYbins = y[i] + 1;
	}

	// Collect
	vector<int> xCounts(nXbins, 0), yCounts(nYbins, 0), coCounts(static_cast<size_t>(nXbins) * static_cast<size_t>(nYbins), 0);
	n = 0;
	for (unsigned int i = 0; i < x.size(); i++) {
		if (x[i] >= 0 && y[i] >= 0) {
			xCounts[x[i]]++;
			yCounts[y[i]]++;
			coCounts[y[i] * nXbins + x[i]]++;
			n++;
		}
	}

	if (n < 2) {
		MLOG_V("Not enough common non-missing entries for mutual information.\n");
		return -1.0;
	}

	return mutual_information(xCounts, yCounts, coCounts, n);
}

float medial::performance::mutual_information(vector<int>& xCounts, vector<int>& yCounts, vector<int> coCounts, int n) {

	double mi = 0;
	int nXbins = (int)xCounts.size();
	int nYbins = (int)yCounts.size();

	for (int iX = 0; iX < nXbins; iX++) {
		for (int iY = 0; iY < nYbins; iY++) {
			if (coCounts[iY*nXbins + iX] != 0) {
				double p = (coCounts[iY*nXbins + iX] + 0.0) / n;
				double px = (xCounts[iX] + 0.0) / n;
				double py = (yCounts[iY] + 0.0) / n;

				mi += p * log(p / px / py) / log(2.0);
			}
		}
	}

	return (float)mi;
}

// AUC
//.........................................................................................................................................
template <typename T> float medial::performance::auc(vector<T> &preds, vector<float> &y) {

	vector<pair<double, double>> preds_y;

	if (preds.size() != y.size())
		MTHROW_AND_ERR("Size mismatch in calculating AUC\n");

	preds_y.resize(preds.size());
	for (int i = 0; i < preds.size(); i++) {
		preds_y[i].first = (double)preds[i];
		preds_y[i].second = y[i];
	}

	// Sort from high score to low
	sort(preds_y.begin(), preds_y.end(), [](const pair<double, double>& left, const pair<double, double>& right) { return left.first > right.first; });

	// Loop - for each Negative, count all positives above it ...
	unsigned long long auc = 0;
	unsigned long long nneg = 0, npos = 0;
	for (unsigned int i = 0; i < preds.size(); i++) {
		if (preds_y[i].second > 0)
			npos++;
		else {
			auc += npos;
			nneg++;
		}
	}

	double epsilon = 1e-15;
	return (float)(((double)auc) / (epsilon + (double)npos*(double)nneg));
}
template float medial::performance::auc<float>(vector<float> &preds, vector<float> &y);
template float medial::performance::auc<double>(vector<double> &preds, vector<float> &y);

template<class T>float _auc_q_weighted(const vector<T> &preds, const vector<float> &y, const vector<float> &weights) {
	vector<T> pred_threshold;
	unordered_map<T, vector<int>> pred_indexes;
	double tot_true_labels = 0, tot_false_labels = 0;
	for (size_t i = 0; i < preds.size(); ++i)
	{
		pred_indexes[preds[i]].push_back((int)i);
		tot_true_labels += int(y[i] > 0) * weights[i];
		tot_false_labels += int(y[i] <= 0) * weights[i];
	}
	if (tot_true_labels <= 0)
		MTHROW_AND_ERR("Error _auc_q_weighted - tot_true_labels(%2.1f) <= 0.\n", tot_true_labels);
	if (tot_false_labels <= 0)
		MTHROW_AND_ERR("Error _auc_q_weighted - tot_false_labels(%2.1f) <= 0.\n", tot_false_labels);
	pred_threshold.resize((int)pred_indexes.size());
	auto it = pred_indexes.begin();
	for (size_t i = 0; i < pred_threshold.size(); ++i)
	{
		pred_threshold[i] = it->first;
		++it;
	}
	sort(pred_threshold.begin(), pred_threshold.end());

	//From up to down sort:
	double t_cnt = 0;
	double f_cnt = 0;
	vector<float> true_rate = vector<float>((int)pred_indexes.size());
	vector<float> false_rate = vector<float>((int)pred_indexes.size());
	int st_size = (int)pred_threshold.size() - 1;
	for (int i = st_size; i >= 0; --i)
	{
		vector<int> &indexes = pred_indexes[pred_threshold[i]];
		//calc AUC status for this step:
		for (int ind : indexes)
		{
			bool true_label = y[ind] > 0;
			t_cnt += int(true_label) * weights[ind];
			f_cnt += int(!true_label) * weights[ind];
		}
		true_rate[st_size - i] = float(t_cnt / tot_true_labels);
		false_rate[st_size - i] = float(f_cnt / tot_false_labels);
	}

	float auc = false_rate[0] * true_rate[0] / 2;
	for (size_t i = 1; i < true_rate.size(); ++i)
		auc += (false_rate[i] - false_rate[i - 1]) * (true_rate[i - 1] + true_rate[i]) / 2;
	return auc;
}
template float _auc_q_weighted<float>(const vector<float> &preds, const vector<float> &y, const vector<float> &weights);
template float _auc_q_weighted<double>(const vector<double> &preds, const vector<float> &y, const vector<float> &weights);

template<typename T> float medial::performance::auc_q(const vector<T> &preds, const vector<float> &y, const vector<float>* weights) {

	if (weights == NULL || weights->empty()) {
		vector<float> w(preds.size(), 1.0);
		return _auc_q_weighted(preds, y, w);
	}
	else
		return _auc_q_weighted(preds, y, *weights);


}
template float medial::performance::auc_q<float>(const vector<float> &preds, const vector<float> &y, const vector<float>* weights);
template float medial::performance::auc_q<double>(const vector<double> &preds, const vector<float> &y, const vector<float>* weights);

// Collect cnts : TP,FP,FN,TN per positive rate (as given by size)
//.........................................................................................................................................
template<typename T> void medial::performance::get_preds_perf_cnts(vector<T> &preds, vector<float> &y, vector<float> &size, int direction, vector<vector<int>> &cnts)
{
	vector<pair<float, float>> preds_y;

	if (preds.size() != y.size())
		MTHROW_AND_ERR("Size mismatch in collecting counts\n");

	preds_y.resize(preds.size());
	for (int i = 0; i < preds.size(); i++) {
		if (direction > 0)
			preds_y[i].first = preds[i];
		else
			preds_y[i].first = -preds[i];

		preds_y[i].second = y[i];
		//preds_y[i].second = (y[i] > 3);
	}

	// Sort from high score to low
	sort(preds_y.begin(), preds_y.end(), [](const pair<float, float>& left, const pair<float, float>& right) { return left.first > right.first; });

	cnts.clear();
	for (auto sz : size) {
		int pos = (int)(sz * (float)preds_y.size());
		int a = 0, b = 0, c = 0, d = 0;
		for (int i = 0; i <= pos; i++) {
			if (preds_y[i].second > 0)
				a++;
			else
				b++;
		}
		for (int i = pos + 1; i < preds_y.size(); i++) {
			if (preds_y[i].second > 0)
				c++;
			else
				d++;
		}
		cnts.push_back({ a,b,c,d });
		//MLOG("sz %f pos %d/%d cnts %d %d %d %d\n", sz, pos, preds_y.size(), a, b, c, d);
	}

	return;
}
template void medial::performance::get_preds_perf_cnts<float>(vector<float> &preds, vector<float> &y, vector<float> &size, int direction, vector<vector<int>> &cnts);
template void medial::performance::get_preds_perf_cnts<double>(vector<double> &preds, vector<float> &y, vector<float> &size, int direction, vector<vector<int>> &cnts);

// Translate counts (TP,FP,FN,TN) to performance measurements (snes,spec,ppv,rr)
//.........................................................................................................................................
void medial::performance::cnts_to_perf(vector<int> &cnt, float &sens, float &spec, float &ppv, float &rr)
{
	float a = (float)cnt[0];
	float b = (float)cnt[1];
	float c = (float)cnt[2];
	float d = (float)cnt[3];

	float epsilon = (float)1e-5;
	sens = a / (a + c + epsilon);
	spec = d / (b + d + epsilon);
	ppv = a / (a + b + epsilon);
	rr = (a / (a + b + epsilon)) / ((c / (c + d + epsilon)) + epsilon);

	return;
}

// Distance Correlations
//.........................................................................................................................................
// Get Distances matrix
template <typename T> void medial::performance::get_dMatrix(vector<T>& values, MedMat<T>& dMatrix, T missing_value) {

	int n = (int)values.size();
	dMatrix.resize(n, n);

	// Matrix + norms
	vector<double> norm(n, 0);
	vector<int> counts(n, 0);
	double totNorm = 0;
	int totCount = 0;

	for (int i = 1; i < n; i++) {
		if (values[i] == missing_value) {
			for (int j = 0; j < i; j++)
				dMatrix(i, j) = -1;
		}
		else {
			for (int j = 0; j < i; j++) {
				if (values[j] == missing_value)
					dMatrix(i, j) = -1;
				else {
					dMatrix(i, j) = fabs(values[i] - values[j]);
					norm[i] += 2 * dMatrix(i, j); counts[i] += 2;
					norm[j] += 2 * dMatrix(i, j); counts[j] += 2;
					totNorm += 2 * dMatrix(i, j); totCount += 2;
				}
			}
		}
	}

	// Normalize
	for (int i = 0; i < n; i++)
		norm[i] /= counts[i];
	totNorm /= totCount;

	for (int i = 1; i < n; i++) {
		for (int j = 0; j < i; j++) {
			if (dMatrix(i, j) != -1)
				dMatrix(i, j) = dMatrix(i, j) - (float)(norm[i] + norm[j] - totNorm);
		}
	}
}
template void medial::performance::get_dMatrix<float>(vector<float>& values, MedMat<float>& dMatrix, float missing_value);
template void medial::performance::get_dMatrix<double>(vector<double>& values, MedMat<double>& dMatrix, double missing_value);

// Get Distance variance
template <typename T> float medial::performance::get_dVar(MedMat<T>& dMatrix) {

	int n = dMatrix.nrows;

	double sum = 0;
	int num = 0;
	for (int i = 1; i < n; i++) {
		for (int j = 0; j < i; j++) {
			if (dMatrix(i, j) != -1) {
				sum += 2 * static_cast<double>(dMatrix(i, j)) * static_cast<double>(dMatrix(i, j));
				num += 2;
			}
		}
	}

	if (num)
		return (float)(sum / num);
	else
		return -1;
}
template float medial::performance::get_dVar<float>(MedMat<float>& dMatrix);
template float medial::performance::get_dVar<double>(MedMat<double>& dMatrix);

// Get Distance covariance
template <typename T> float medial::performance::get_dCov(MedMat<T>& xDistMat, MedMat<T>& yDistMat) {

	int n = xDistMat.nrows;

	double sum = 0;
	int num = 0;
	for (int i = 1; i < n; i++) {
		for (int j = 0; j < i; j++) {
			if (xDistMat(i, j) != -1 && yDistMat(i, j) != -1) {
				sum += 2 * static_cast<double>(xDistMat(i, j)) * static_cast<double>(yDistMat(i, j));
				num += 2;
			}
		}
	}

	if (num)
		return (float)(sum / num);
	else
		return -1;
}
template float medial::performance::get_dCov<float>(MedMat<float>& xDistMat, MedMat<float>& yDistMat);
template float medial::performance::get_dCov<double>(MedMat<double>& xDistMat, MedMat<double>& yDistMat);

// Helpers for multi-category predictions
//.........................................................................................................................................
// given multicategory probs (or probs-like) predictions generates a single prediction of the categ with max prob for each sample
template <typename T> void medial::performance::multicateg_get_max_pred(vector<T> &probs, int nsamples, int ncateg, vector<float> &max_pred)
{
	int i, j;

	max_pred.resize(nsamples);
	for (i = 0; i < nsamples; i++) {
		float max = probs[static_cast<size_t>(i) * static_cast<size_t>(ncateg)];
		int max_j = 0;
		for (j = 1; j < ncateg; j++) {
			if (probs[i*ncateg + j] > max) {
				max = probs[i*ncateg + j];
				max_j = j;
			}
		}
		max_pred[i] = (float)max_j;
	}

	return;
}
template void medial::performance::multicateg_get_max_pred<float>(vector<float> &probs, int nsamples, int ncateg, vector<float> &max_pred);
template void medial::performance::multicateg_get_max_pred<double>(vector<double> &probs, int nsamples, int ncateg, vector<float> &max_pred);

// given multicategory probs (or probs-like) predictions generates a single prediction of a weighted average of categories
template <typename T> void medial::performance::multicateg_get_avg_pred(vector<T> &probs, int nsamples, int ncateg, vector<T> &avg_pred)
{
	int i, j;

	avg_pred.resize(nsamples);
	for (i = 0; i < nsamples; i++) {
		float sum = 0, sum_p = 0;
		for (j = 0; j < ncateg; j++) {
			sum += (float)j*probs[i*ncateg + j];
			sum_p += probs[i*ncateg + j];
		}
		if (sum_p <= 0) sum_p = (float)1e-15;
		avg_pred[i] = sum / sum_p;
	}

	return;
}
template void medial::performance::multicateg_get_avg_pred<float>(vector<float> &probs, int nsamples, int ncateg, vector<float> &avg_pred);
template void medial::performance::multicateg_get_avg_pred<double>(vector<double> &probs, int nsamples, int ncateg, vector<double> &avg_pred);

// given multicategory probs (or probs-like) predictions gets the classification error rate and the rms (also for the avg preds)
template <typename T> void medial::performance::multicateg_get_error_rate(vector<T> &probs, vector<float> &y, int nsamples, int ncateg, float &err_rate, T &rms, T &avg_rms)
{
	vector<float> max_preds;
	vector<T> avg_preds;

	medial::performance::multicateg_get_max_pred(probs, nsamples, ncateg, max_preds);
	medial::performance::multicateg_get_avg_pred(probs, nsamples, ncateg, avg_preds);

	err_rate = 0;
	rms = 0;
	avg_rms = 0;

	int i;

	T fact = (T)1 / (T)nsamples;
	for (i = 0; i < nsamples; i++) {

		if (max_preds[i] != (T)y[i]) err_rate++;
		rms += fact * (max_preds[i] - (T)y[i])*(max_preds[i] - (T)y[i]);
		avg_rms += fact * (avg_preds[i] - (T)y[i])*(avg_preds[i] - (T)y[i]);

	}

	err_rate /= (float)nsamples;

	return;
}
template void medial::performance::multicateg_get_error_rate<float>(vector<float> &probs, vector<float> &y, int nsamples, int ncateg, float &err_rate, float &rms, float &avg_rms);
template void medial::performance::multicateg_get_error_rate<double>(vector<double> &probs, vector<float> &y, int nsamples, int ncateg, float &err_rate, double &rms, double &avg_rms);

//.........................................................................................................................................
// Given two vectors and a vector of quantization bounds, create the 'confusion' matrix counts
template <typename T> void medial::performance::get_quantized_breakdown(vector<T> &preds, vector<T> &y, vector<T> &bounds, MedMat<int> &counts)
{
	int i, ip, iy;
	int nb = (int)bounds.size() - 1;

	counts.resize(nb, nb);
	counts.zero();

	for (i = 0; i < preds.size(); i++) {

		ip = 0;
		while (ip<nb && preds[i]>bounds[ip + 1]) ip++;
		iy = 0;
		while (iy<nb && y[i]>bounds[iy + 1]) iy++;

		if (preds[i] <= bounds[ip + 1] && y[i] <= bounds[iy + 1]) {
			counts(ip, iy)++;
		}

	}

	return;
}
template void medial::performance::get_quantized_breakdown<float>(vector<float> &preds, vector<float> &y, vector<float> &bounds, MedMat<int> &counts);
template void medial::performance::get_quantized_breakdown<double>(vector<double> &preds, vector<double> &y, vector<double> &bounds, MedMat<int> &counts);

/// <sumary>print the 'confusion' matrix counts </summary>
template <typename T> void medial::performance::print_quantized_breakdown(MedMat<int> &cnt, vector<T> &bounds)
{
	MOUT("Quantized results distribution:\n");
	for (int i = 0; i < cnt.nrows; i++) {
		MOUT("preds %7.2lf - %7.2lf ::", (double)bounds[i], (double)bounds[i + 1]);
		for (int j = 0; j < cnt.ncols; j++) {
			MOUT(" %6d", cnt(i, j));
		}
		MOUT("\n");
	}
}
template void medial::performance::print_quantized_breakdown<float>(MedMat<int> &cnt, vector<float> &bounds);
template void medial::performance::print_quantized_breakdown<double>(MedMat<int> &cnt, vector <double> &bounds);

template <typename T> double medial::performance::integrated_calibration_index(const vector<T> &predicted_prob, const vector<float> &y_label, const vector<float>* weights) {
	double res = 0;
	if (predicted_prob.size() != y_label.size())
		MTHROW_AND_ERR("Error medial::performance::integrated_calibration_index - predicted_prob(%zu),y_label(%zu) should have same size\n",
			predicted_prob.size(), y_label.size());
	if (weights == NULL || weights->empty()) {
		//seperate into bins of predicted values prob into observed prob:
		unordered_map<float, int> bin_size;
		unordered_map<float, double> bin_observed_prob;
		for (size_t i = 0; i < predicted_prob.size(); ++i)
		{
			++bin_size[predicted_prob[i]];
			bin_observed_prob[predicted_prob[i]] += int(y_label[i] > 0);
		}
		for (auto &it : bin_observed_prob)
			it.second /= bin_size.at(it.first);

		double total_cnt = (double)predicted_prob.size();
		//calulate ICI based on bin_observed_prob, bin_size:
		for (const auto &it : bin_size)
		{
			double prb_bin = it.second / total_cnt;
			res += prb_bin * abs(bin_observed_prob.at(it.first) - it.first);
		}

		return res;
	}
	else {
		if (predicted_prob.size() != weights->size())
			MTHROW_AND_ERR("Error medial::performance::integrated_calibration_index - predicted_prob(%zu), weights(%zu) should have same size when weights are given\n",
				predicted_prob.size(), weights->size());

		unordered_map<float, double> bin_size;
		unordered_map<float, double> bin_observed_prob;
		double total_cnt = 0;
		for (size_t i = 0; i < predicted_prob.size(); ++i)
		{
			bin_size[predicted_prob[i]] += weights->at(i);
			bin_observed_prob[predicted_prob[i]] += int(y_label[i] > 0) * weights->at(i);
			total_cnt += weights->at(i);
		}
		for (auto &it : bin_observed_prob)
			it.second /= bin_size.at(it.first);

		//calulate ICI based on bin_observed_prob, bin_size:
		for (const auto &it : bin_size)
		{
			double prb_bin = it.second / total_cnt;
			res += prb_bin * abs(bin_observed_prob.at(it.first) - it.first);
		}

		return res;
	}
}
template double medial::performance::integrated_calibration_index<float>(const vector<float> &predicted_prob, const vector<float> &y_label, const vector<float>* weights);
template double medial::performance::integrated_calibration_index<double>(const vector<double> &predicted_prob, const vector<float> &y_label, const vector<float>* weights);

// Stats functions: various statistical utilities
//...............................................................................................................................

// Momments: mean
//...............................................................................................................................
template <typename T> double medial::stats::mean_without_cleaning(const vector<T> &v, const vector<float> *weights) {

	bool has_weights = weights != NULL && !weights->empty();

	double s = 0, c = 0;
	if (has_weights)
		for (size_t i = 0; i < v.size(); ++i) {
			s += static_cast<double>(v[i]) * static_cast<double>((*weights)[i]);
			c += (*weights)[i];
		}
	else {
		c = (double)v.size();
		for (size_t i = 0; i < v.size(); ++i)
			s += v[i];
	}

	if (c == 0)
		MTHROW_AND_ERR("No values (with weights!=0) given for mean_without_cleaning. Cannot return anything valid\n");

	return s / c;
}
template double medial::stats::mean_without_cleaning<float>(const vector<float> &v, const vector<float> *weights);
template double medial::stats::mean_without_cleaning<double>(const vector<double> &v, const vector<float> *weights);

template <typename T> double medial::stats::mean(const vector<T> &v, T missing_value, int& n, const vector<float> *weights) {

	bool has_weights = weights != NULL && !weights->empty();

	double s = 0, c = 0;
	n = 0;
	if (has_weights)
		for (size_t i = 0; i < v.size(); ++i) {
			s += v[i] != missing_value ? v[i] * (*weights)[i] : 0;
			c += (*weights)[i] * int(v[i] != missing_value);
			n += int(v[i] != missing_value);
		}
	else
		for (size_t i = 0; i < v.size(); ++i) {
			s += v[i] != missing_value ? v[i] : 0;
			c += int(v[i] != missing_value);
			n += int(v[i] != missing_value);
		}

	if (c == 0)
		return missing_value;
	return s / c;
}
template double medial::stats::mean<float>(const vector<float> &v, float missing_value, int& n, const vector<float> *weights);
template double medial::stats::mean<double>(const vector<double> &v, double missing_value, int& n, const vector<float> *weights);

// Momments: standard deviation
//...............................................................................................................................
template <typename T> double medial::stats::std_without_cleaning(const vector<T> &v, T mean, const vector<float> *weights) {

	bool has_weights = weights != NULL && !weights->empty();

	double s = 0, c = 0;
	if (has_weights)
		for (size_t i = 0; i < v.size(); ++i) {
			s += static_cast<double>((*weights)[i]) * (v[i] - mean) * (v[i] - mean);
			c += (*weights)[i];
		}
	else {
		c = (double)v.size();
		for (size_t i = 0; i < v.size(); ++i)
			s += static_cast<double>(v[i] - mean) * static_cast<double>(v[i] - mean);
	}

	if (c == 0)
		MTHROW_AND_ERR("No values (with weights!=0) given for std_without_cleaning. Cannot return anything valid\n")
	else if (c == 1)
		return 1.0;
	else
		return sqrt(s / c);
}
template double medial::stats::std_without_cleaning<float>(const vector<float> &v, float mean, const vector<float> *weights);
template double medial::stats::std_without_cleaning<double>(const vector<double> &v, double mean, const vector<float> *weights);

template <typename T> double medial::stats::std(const vector<T> &v, T mean, T missing_value, int& n, const vector<float> *weights) {

	bool has_weights = weights != NULL && !weights->empty();

	double s = 0, c = 0;
	n = 0;
	if (has_weights)
		for (size_t i = 0; i < v.size(); ++i) {
			s += v[i] != missing_value ? (*weights)[i] * (v[i] - mean) * (v[i] - mean) : 0;
			c += (*weights)[i] * int(v[i] != missing_value);
			n += int(v[i] != missing_value);
		}
	else
		for (size_t i = 0; i < v.size(); ++i) {
			s += v[i] != missing_value ? (v[i] - mean) * (v[i] - mean) : 0;
			c += int(v[i] != missing_value);
			n += int(v[i] != missing_value);
		}

	if (c == 0)
		return missing_value;
	else if (n == 1)
		return 1.0;
	else
		return sqrt(s / c);
}
template double medial::stats::std<float>(const vector<float> &v, float mean, float missing_value, int& n, const vector<float> *weights);
template double medial::stats::std<double>(const vector<double> &v, double mean, double missing_value, int& n, const vector<float> *weights);

// Momments : backward compatible version for get_mean_and_atd
//...............................................................................................................................
void medial::stats::get_mean_and_std(float *values, const float* wgts, int size, float missing_value, float& mean, float&sd, int& n, bool do_missing) {

	double c = 0;
	double s = 0;
	n = 0;

	for (int i = 0; i < size; i++) {
		if (!do_missing || values[i] != missing_value) {
			c += wgts[i];
			s += static_cast<double>(wgts[i]) * static_cast<double>(values[i]);
			n++;
		}
	}
	if (c == 0) {
		mean = 0;
		sd = 1.0;
		return;
	}

	mean = (float)(s / n);

	s = 0.0;
	for (int i = 0; i < size; i++) {
		if (!do_missing || values[i] != missing_value)
			s += static_cast<double>(wgts[i]) * static_cast<double>((values[i] - mean)) * static_cast<double>((values[i] - mean));
	}

	if (n > 1)
		sd = (float)sqrt((s / n));
	else
		sd = (float) 1.0;

	if (sd == 0) {
		MWARN("get_moments for all-zeros vector, fixing SD from 0.0 to 1.0\n");
		sd = 1.0;
	}
	return;
}


// Momments: median
//...............................................................................................................................
template<typename T> T medial::stats::median_without_cleaning(vector<T>& v, bool in_place) {

	if (v.size() == 0)
		MTHROW_AND_ERR("No values given for getting median\n");

	if (in_place) {
		nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
		return v[v.size() / 2];
	}
	else {
		vector<T> v_ = v;
		return medial::stats::median_without_cleaning(v_, true);
	}
}
template float medial::stats::median_without_cleaning<float>(vector<float>& v, bool in_place);
template double medial::stats::median_without_cleaning<double>(vector<double>& v, bool in_place);

template<typename T> T medial::stats::median(vector<T>& v, T missing_value, int& n) {

	vector<T> v_(v.size());

	n = 0;
	for (unsigned int idx = 0; idx < v.size(); idx++) {
		if (v[idx] != missing_value)
			v_[n++] = v[idx];
	}

	if (n == 0)
		return missing_value;

	v_.resize(n);
	nth_element(v_.begin(), v_.begin() + v_.size() / 2, v_.end());
	return v_[v_.size() / 2];
}
template float medial::stats::median<float>(vector<float>& v, float missing_value, int& n);
template double medial::stats::median<double>(vector<double>& v, double missing_value, int& n);

// Momments: most_common value
//...............................................................................................................................
template<typename T> T medial::stats::most_common_without_cleaning(vector<T>& v) {

	if (v.size() == 0)
		MTHROW_AND_ERR("No values given for getting most-common value\n");

	map<T, int> counter;
	for (T val : v)
		counter[val]++;

	int count = 0;
	T most_common = v[0];
	for (auto& rec : counter) {
		if (rec.second > count) {
			count = rec.second;
			most_common = rec.first;
		}
	}

	return most_common;
}
template float medial::stats::most_common_without_cleaning<float>(vector<float>& v);
template double medial::stats::most_common_without_cleaning<double>(vector<double>& v);

template<typename T> T medial::stats::most_common(vector<T>& v, T missing_value, int& n) {

	map<T, int> counter;
	for (T val : v)
		counter[val]++;

	int count = 0;
	T most_common = missing_value;
	for (auto& rec : counter) {
		if (rec.first != missing_value) {
			n += rec.second;
			if (rec.second > count) {
				count = rec.second;
				most_common = rec.first;
			}
		}
	}

	return most_common;
}
template float medial::stats::most_common<float>(vector<float>& v, float missing_value, int& n);
template double medial::stats::most_common<double>(vector<double>& v, double missing_value, int& n);

// Momments: histogram of values
//...............................................................................................................................
template<typename T> void medial::stats::get_histogram_without_cleaning(vector<T>& v, vector<pair<T, float> >& hist, bool in_place) {

	if (!in_place) {
		vector<T> v_ = v;
		get_histogram_without_cleaning(v_, hist, true);
	}

	sort(v.begin(), v.end());

	for (unsigned int i = 1; i < v.size(); i++) {
		if (v[i] != v[i - 1])
			hist.push_back({ v[i - 1] , ((float)i) / v.size() });
	}

	hist.push_back({ v.back() , (float)1.0 });
}
template void medial::stats::get_histogram_without_cleaning<float>(vector<float>& v, vector<pair<float, float> >& hist, bool in_place);
template void medial::stats::get_histogram_without_cleaning<double>(vector<double>& v, vector<pair<double, float> >& hist, bool in_place);

template<typename T> void medial::stats::get_histogram(vector<T>& v, T missing_value, int& n, vector<pair<T, float> >& hist) {

	vector<T> v_(v.size());

	n = 0;
	for (unsigned int idx = 0; idx < v.size(); idx++) {
		if (v[idx] != missing_value)
			v_[n++] = v[idx];
	}
	v_.resize(n);

	sort(v_.begin(), v_.end());

	for (int i = 1; i < n; i++) {
		if (v_[i] != v_[i - 1])
			hist.push_back({ v_[i - 1] , ((float)i) / n });
	}

	hist.push_back({ v_.back() , (float)1.0 });

}
template void medial::stats::get_histogram<float>(vector<float>& v, float missing_value, int& n, vector<pair<float, float> >& hist);
template void medial::stats::get_histogram<double>(vector<double>& v, double missing_value, int& n, vector<pair<double, float> >& hist);

template<typename T> T medial::stats::sample_from_histogram(vector<pair<T, float> >& hist) {

	// Generate a random number
	float r = (float)(globalRNG::rand() / (globalRNG::max() + 1.0));

	// Sample
	for (int i = 0; i < hist.size(); i++) {
		if (r < hist[i].second)
			return hist[i].first;
	}

	// Handle numeric issues
	return hist.back().first;
}
template float medial::stats::sample_from_histogram<float>(vector<pair<float, float> >& hist);
template double medial::stats::sample_from_histogram<double>(vector<pair<double, float> >& hist);

// Momments: percentiles
//...............................................................................................................................
template<typename T> void medial::stats::get_percentiles(vector<T> &vals, vector<float> &p, vector<T> &out_pvals, int only_positive_flag)
{
	sort(vals.begin(), vals.end());

	int n = (int)vals.size();
	int n0 = 0;
	if (only_positive_flag) {
		while (n0 < n && vals[n0] <= 0) n0++;
	}
	out_pvals.resize(p.size(), 0);
	for (int i = 0; i < p.size(); i++) {
		int k = n0 + (int)((float)(n - n0)*p[i]);
		if (k < 0) k = 0;
		if (k >= n) k = n - 1;
		out_pvals[i] = vals[k];
	}
}
template<typename T>
T inPlaceQuantile(T *vals, float *w, float wq, int length)
{
	int pIndex = length - 1;
	double weightToIndex = 0;
	for (int i = 0; i < pIndex; i++) {
		if (vals[i] > vals[pIndex]) {


			T dummy = vals[pIndex];
			vals[pIndex] = vals[i];
			vals[i] = vals[pIndex - 1];
			vals[pIndex - 1] = dummy;

			float dummy1 = w[pIndex];
			w[pIndex] = w[i];
			w[i] = w[pIndex - 1];
			w[pIndex - 1] = dummy1;

			pIndex--;
			i--;
		}
		else
			weightToIndex += w[i];
	}

	if (weightToIndex <= wq && weightToIndex + w[pIndex] >= wq)
		return(vals[pIndex]);
	if (wq < weightToIndex)return(inPlaceQuantile(vals, w, wq, pIndex));
	return(inPlaceQuantile(vals + pIndex, w + pIndex, wq - weightToIndex, length - pIndex));

}
template<typename T>
T medial::stats::get_quantile(vector<T> vals, vector<float> w, float q)
{
	if (vals.size() == 1)return(vals[0]);
	if (vals.size() != w.size())	MTHROW_AND_ERR("Length inconsistency in calculating Quantile\n");
	float sumW = 0;
	for (auto&& ww : w)sumW += ww;

	double wq = static_cast<double>(sumW) * static_cast<double>(q);
	return T(inPlaceQuantile(vals.data(), w.data(), wq, (int)vals.size()));
}
template void medial::stats::get_percentiles<float>(vector<float> &vals, vector<float> &p, vector<float> &out_pvals, int only_positive_flag);
template void medial::stats::get_percentiles<double>(vector<double> &vals, vector<float> &p, vector<double> &out_pvals, int only_positive_flag);

template float medial::stats::get_quantile<float>(vector<float> vals, vector<float> w, float q);
template double medial::stats::get_quantile<double>(vector<double> vals, vector<float> w, float q);

// Chi-Square
//.........................................................................................................................................
double medial::stats::chi2_n_x_m(vector<int> &cnts, int n, int m, vector<double> &exp)
{
	int i, j;

	if (cnts.size() != static_cast<size_t>(n) * static_cast<size_t>(m))
		return -1;

	vector<double> s_n(n, 0);
	vector<double> s_m(m, 0);
	double sum = 0;

	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++) {
			double c = (double)cnts[i*m + j];
			s_n[i] += c;
			s_m[j] += c;
			sum += c;
		}

	if (sum <= 0) return -1;

	exp.resize(static_cast<size_t>(n) * static_cast<size_t>(m));
	double score = 0;
	double epsilon = 1e-5;
	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++) {
			double e = (s_n[i] / sum)*s_m[j];
			exp[i*m + j] = e;
			if (e > epsilon) {
				double d = (double)cnts[i*m + j] - e;
				score += (d / e)*d;
			}
		}

	return score;
}

double medial::stats::chi2_n_x_m(vector<int> &cnts, int n, int m)
{
	vector<double> exp;
	return(chi2_n_x_m(cnts, n, m, exp));
}

// Utils
//...............................................................................................................................
// gets a vector of values, and checks the best way to round it at the 10^-3 to 10^3 (at 10x steps) range.
float medial::stats::get_best_rounding(vector<float> &vals, vector<float>& res, vector<int> &counts, float missing_value)
{
	int n_r = 7;
	res = { (float)0.001,(float)0.01,(float)0.1,1,10,100,1000 };
	vector<int> cnts(n_r, 0);

	// find best rounder for each value and count them
	for (int i = 0; i < vals.size(); i++) {
		for (int j = n_r - 1; j >= 0; j--) {
			if (vals[i] != 0 && vals[i] != missing_value) {
				float vr = ((float)((int)((vals[i] / res[j]))));
				if (abs((float)(vals[i] / res[j]) - vr) == 0) {
					cnts[j]++;
					break;
				}
			}
		}

	}


	// return most common rounder
	int best = 0, best_cnt = 0;
	for (int j = 0; j < n_r; j++) {
		if (cnts[j] > best_cnt) {
			best = j;
			best_cnt = cnts[j];
		}
	}
	counts = cnts;
	return res[best];
}

double medial::stats::chi_square_table(double grp1_cntrl, double grp1_cases, double grp2_cntrl, double grp2_cases,
	int smooth_balls, float allowed_error) {
	//calc over all ages
	double regScore = 0;
	double totCnt = 0;
	vector<double> R(2);
	vector<double> C(2);
	totCnt = grp1_cntrl + grp2_cntrl + grp1_cases + grp2_cases;

	vector<double> probs = { grp1_cntrl,grp1_cases,grp2_cntrl, grp2_cases }; //the forth numbers - float with fix
	if (smooth_balls > 0)
		for (size_t j = 0; j < 4; ++j)
			probs[j] = probs[j] + (smooth_balls * C[j % 2] / totCnt);  /* add smooth addition */

	totCnt = 0;
	R[0] = probs[0] + probs[1];
	R[1] = probs[2 + 0] + probs[2 + 1];
	C[0] = probs[0] + probs[2]; //how much controls
	C[1] = probs[1] + probs[1 + 2]; //how much cases
	for (size_t j = 0; j < probs.size(); ++j)
		totCnt += probs[j];

	for (size_t j = 0; j < probs.size(); ++j)
	{
		double	Qij = probs[j];
		double Eij = (R[j / 2] * C[j % 2]) / totCnt;
		double Dij = abs(Qij - Eij) - (allowed_error / 100) * Eij;
		if (Dij < 0)
			Dij = 0;

		if (Eij > 0)
			regScore += (Dij * Dij) / (Eij); //Chi-square
	}


	return regScore;
}

double get_t_test_pvalue(double t, int dof) {
	if (dof < 1 || t <= 0)
		return 1;
	else {
		boost::math::students_t dist(dof);
		return (1.0 - boost::math::cdf(dist, t));
	}
}

template<typename T> void medial::stats::t_test(const vector<T> &grp1, const vector<T> &grp2,
	double &t_value, double &degree_of_freedom, double &p_value) {
	if (grp1.size() < 2 || grp2.size() < 2)
		MTHROW_AND_ERR("Error medial::stats::t_test - both groups should have more than 1 obs\n");
	if (grp1.size() != grp2.size())
		MTHROW_AND_ERR("Error medial::stats::t_test - t test should accept samples with equal size\n");
	int sz = (int)grp1.size();
	degree_of_freedom = 2 * (sz - 1);

	T mean1, mean2;
	mean1 = mean_without_cleaning(grp1);
	mean2 = mean_without_cleaning(grp2);

	double var1 = pow(std_without_cleaning(grp1, mean1), 2) * sz / (sz - 1);
	double var2 = pow(std_without_cleaning(grp2, mean2), 2) * sz / (sz - 1);
	double s = sqrt((var1 + var2) / 2);

	t_value = abs(mean1 - mean2) / (s*sqrt(2 / sz));
	p_value = get_t_test_pvalue(t_value, degree_of_freedom);
}
template void medial::stats::t_test<float>(const vector<float> &grp1, const vector<float> &grp2, double &t_value, double &degree_of_freedom, double &p_value);
template void medial::stats::t_test<double>(const vector<double> &grp1, const vector<double> &grp2, double &t_value, double &degree_of_freedom, double &p_value);

template<typename T> void medial::stats::t_test_unequal_sample_size(const vector<T> &grp1, const vector<T> &grp2,
	double &t_value, double &degree_of_freedom, double &p_value) {
	if (grp1.size() < 2 || grp2.size() < 2)
		MTHROW_AND_ERR("Error medial::stats::t_test - both groups should have more than 1 obs\n");
	T mean1, mean2;
	mean1 = mean_without_cleaning(grp1);
	mean2 = mean_without_cleaning(grp2);
	double sz1 = (double)grp1.size();
	double sz2 = (double)grp2.size();
	degree_of_freedom = sz1 + sz2 - 2;

	double var1 = pow(std_without_cleaning(grp1, mean1), 2) * sz1 / (sz1 - 1);
	double var2 = pow(std_without_cleaning(grp2, mean2), 2) * sz2 / (sz2 - 1);
	double s = sqrt((((sz1 - 1)*var1) + ((sz2 - 1)*var2)) / degree_of_freedom);
	//calc t_test statstic value:
	t_value = abs(mean1 - mean2) / (s * sqrt((1 / sz1) + (1 / sz2)));

	p_value = get_t_test_pvalue(t_value, degree_of_freedom);
}
template void medial::stats::t_test_unequal_sample_size<float>(const vector<float> &grp1, const vector<float> &grp2, double &t_value, double &degree_of_freedom, double &p_value);
template void medial::stats::t_test_unequal_sample_size<double>(const vector<double> &grp1, const vector<double> &grp2, double &t_value, double &degree_of_freedom, double &p_value);

template<typename T> void medial::stats::welch_t_test(const vector<T> &grp1, const vector<T> &grp2,
	double &t_value, double &degree_of_freedom, double &p_value) {
	if (grp1.size() < 2 || grp2.size() < 2)
		MTHROW_AND_ERR("Error medial::stats::welch_t_test - both groups should have more than 1 obs\n");
	T mean1, std1, mean2, std2;
	get_mean_and_std_without_cleaning(grp1, mean1, std1);
	get_mean_and_std_without_cleaning(grp2, mean2, std2);
	double var1 = static_cast<double>(std1) * static_cast<double>(std1);
	double var2 = static_cast<double>(std2) * static_cast<double>(std2);
	double sz1 = (double)grp1.size();
	double sz2 = (double)grp2.size();

	double ele = (var1 / sz1) + (var2 / sz2);
	t_value = abs(mean1 - mean2) / sqrt(ele);
	degree_of_freedom = pow(ele, 2) / ((pow(var1, 2) / (pow(sz1, 2) * (sz1 - 1))) + (pow(var2, 2) / (pow(sz2, 2) * (sz2 - 1))));

	p_value = get_t_test_pvalue(t_value, degree_of_freedom);
}
template void medial::stats::welch_t_test<float>(const vector<float> &grp1, const vector<float> &grp2, double &t_value, double &degree_of_freedom, double &p_value);
template void medial::stats::welch_t_test<double>(const vector<double> &grp1, const vector<double> &grp2, double &t_value, double &degree_of_freedom, double &p_value);

// KL divergence (+ some heuristics for zeros)
template<typename T> double medial::stats::KL_divergence(const vector<T> &p, const vector<T> &q, T epsilon) {

	if (p.size() != q.size())
		MTHROW_AND_ERR("KL divergence requires probability vector of equal size\n");

	// Correct for zeros if required.
	bool req_corrections = false;
	for (size_t i = 0; i < p.size(); i++) {
		if ((p[i] == 0 && q[i] != 0) || (p[i] != 0 && q[i] == 0)) {
			req_corrections = true;
			break;
		}
	}

	double _epsilon = (req_corrections) ? (double)epsilon : 0.0;
	vector<T> _p(p.size()), _q(p.size());
	
	for (size_t i = 0; i < p.size(); i++) {
		_p[i] = (p[i] + _epsilon) / (1 + p.size()*_epsilon);
		_q[i] = (q[i] + _epsilon) / (1 + q.size()*_epsilon);
	}

	// Calculate
	double kl = 0;
	for (size_t i = 0; i < _p.size(); i++) {
		if (_p[i] > 0)
			kl += static_cast<double>(_p[i]) * log(static_cast<double>(_p[i]) / static_cast<double>(_q[i]));
	}

	return kl;
}
template double medial::stats::KL_divergence<float>(const vector<float> &p, const vector<float> &q, float epsilon);
template double medial::stats::KL_divergence<double>(const vector<double> &p, const vector<double> &q, double epsilon);

template<typename T> void medial::stats::get_z_transform(const vector<T> &v, T missing_value_in_v, T missing_value_z, vector<T> &z)
{
	int n;
	float m, s;
	medial::stats::get_mean_and_std<T>(v, missing_value_in_v, n, m, s, NULL);
	z.clear();
	if (s == 0) s = 1;
	for (auto val : v) {
		if (val == missing_value_in_v)
			z.push_back(missing_value_z);
		else
			z.push_back((val - m) / s);
	}
}
template void medial::stats::get_z_transform<float>(const vector<float> &v, float missing_value_in_v, float missing_value_z, vector<float> &z);
//template void medial::stats::get_z_transform<double>(const vector<double> &v, double missing_value_in_v, double missing_value_z, vector<double> &z);
