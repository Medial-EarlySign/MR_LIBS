#include "BinSplitOptimizer.h"
#include "Logger/Logger/Logger.h"
#include <algorithm>
#include "MedUtils/MedUtils/MedUtils.h"
#include "MedStat/MedStat/bootstrap.h"
#include <MedAlgo/MedAlgo/MedAlgo.h>
#include <MedAlgo/MedAlgo/MedQRF.h>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL

bool BinSplitOptimizer::evaluate(const vector<int> &indexes, float &totScore) {
	//Calc the error for this current model_params split rep = distance from avg in each partition
	int numOfElement = (int)_histElements.size();
	totScore = 0;
	auto it = _histElements.begin();
	int prevParam = -1;
	for (size_t i = 0; i < indexes.size(); ++i)
	{
		int currParam = indexes[i];
		if (currParam < 0) {
			currParam = 0;
		}
		if (currParam >= numOfElement) {
			currParam = numOfElement - 1;
		}
		float currentTh = _sortedArr[currParam];

		long clusterSize = 0;
		double clusterAvg = 0;
		while (it != _histElements.end() && it->first < currentTh) {
    clusterSize += it->second;
    clusterAvg += static_cast<double>(_histAvg[it->first]) * static_cast<double>(it->second);
    ++it;
}

		if (clusterSize == 0 || currParam < prevParam || indexes[i] < 0 || indexes[i] >= numOfElement
			|| (_minSamples > 0 && clusterSize < _minSamples)) {
			totScore = 0;
			return false; //ilegal combination
		}
		else {
			clusterAvg /= clusterSize;
			//cout << "avg=" << clusterAvg << endl;
			totScore += (float)clusterSize * (float)(clusterAvg * clusterAvg);
		}

		prevParam = currParam;
	}
	long clusterSize = 0;
	double clusterAvg = 0;
	while (it != _histElements.end()) {
		clusterSize += it->second;
		clusterAvg += static_cast<double>(it->first) * static_cast<double>(it->second);
		++it;
	}
	if (clusterSize == 0 || (_minSamples > 0 && clusterSize < _minSamples)) {
		totScore = 0;
		return false;
	}
	else {
		clusterAvg /= clusterSize;
		totScore += (float)clusterSize * (float)(clusterAvg * clusterAvg);
	}
	//totScore = -totScore;
	//Need to maximize sigma_all_cluster{size_of_cluster * avg_cluster_val^2}. so minimize minus
	return true;

}

vector<int> BinSplitOptimizer::SplitToBins(const vector<float> &vec, const vector<float> &y,
	size_t kBins, size_t min_samples, vector<float> &indToVal, vector<float> &partitionValues,
	bool &has_error) {
	has_error = false;
	_minSamples = (int)min_samples;
	int maxIters = 100;
	int maxDist = 10;
	vector<float> cp;
	vector<int> counts_asc;
	map<int, vector<int>> count2Inds;
	for (size_t i = 0; i < vec.size(); ++i)
	{
		if (_histElements.count(vec[i]) == 0) {
			_histElements[vec[i]] = 0;
			_histAvg[vec[i]] = 0;
		}
		++_histElements[vec[i]];
		_histAvg[vec[i]] += y[i];
	}
	int max_cnt = 0;

	for (auto it = _histElements.begin(); it != _histElements.end(); ++it)
	{
		cp.push_back(it->first);

		if (it->second > 0) {
			_histAvg[it->first] /= it->second;
		}
		if (it->second > max_cnt) {
			max_cnt = it->second;
		}
	}

	_sortedArr = cp.data();
	sort(_sortedArr, &_sortedArr[cp.size() - 1]);

	int minS = 10;
	if (min_samples > 0) {
		minS = (int)min_samples;
	}
	if (_histElements.size() == 1 || _histElements.size() <= kBins) {
		if (_histElements.size() == 1)
			MLOG_D("Warnning :: got signal with only 1 value - remove this signal\n");
		vector<int> emp;
		return emp;
		//throw invalid_argument("got signal with only 1 value - remove this signal");
	}
	if (vec.size() / (2 * minS) < kBins || kBins >= _histElements.size()) {
		if (kBins > _histElements.size()) {
			kBins = _histElements.size();
		}
		/*cout << "Doesn't have enogth values or enougth distinct values. has " << vec.size()
		<< " values and " << _histElements.size() << " distinct ones." << endl;*/
	}
	if (kBins == 0) {
		kBins = _histElements.size();
		vector<int> splittingIndexes((int)2 * kBins);
		for (size_t i = 0; i < splittingIndexes.size(); ++i)
		{
			splittingIndexes[i] = (int)(i / 2);
		}
		partitionValues = vector<float>((int)splittingIndexes.size());
		indToVal = vector<float>((int)splittingIndexes.size());
		indToVal[0] = _sortedArr[0];
		for (size_t i = 0; i < _histElements.size(); ++i)
		{
			float diff = 0;
			float diffPrev = 0;
			if (i > 0) {
				diffPrev = _sortedArr[i] - _sortedArr[i - 1];
			}
			if (i + 1 < _histElements.size()) {
				diff = _sortedArr[i + 1] - _sortedArr[i];
			}
			if (diff == 0 || (diffPrev != 0 && diff > diffPrev)) {
				diff = diffPrev;
			}
			indToVal[2 * i] = _sortedArr[i] - (diff / 2);
			indToVal[2 * i + 1] = _sortedArr[i] + (diff / 2);
			partitionValues[2 * i] = _sortedArr[i];
			partitionValues[2 * i + 1] = _sortedArr[i];
		}
		//indToVal[(int)indToVal.size() - 1] = _sortedArr[(int)cp.size() - 1] + (float)0.001;
		return splittingIndexes;
	}

	if (vec.size() < kBins * (min_samples + max_cnt)) {

		/*cout << "warnning, requested " << kBins << " bins for " << vec.size() << " elements and has "
		<< 100.0*max_cnt / float(vec.size()) << " percentage of biggest value " << max_val_hist << endl;*/
		//kBins = vec.size() / (min_samples + max_cnt);
	}

	//Find init legal solution - update splittingIndexes, kBins 
	int currentBinCnt = (int)_histElements.size();
	vector<int> splittingCnts(currentBinCnt);
	for (size_t i = 0; i < currentBinCnt; ++i)
	{
		float currVal = _sortedArr[i];
		splittingCnts[i] = _histElements[currVal];
		if (count2Inds.find(_histElements[currVal]) == count2Inds.end()) {
			counts_asc.push_back(_histElements[currVal]); //add only unique count items
		}
		count2Inds[_histElements[currVal]].push_back((int)i);
	}
	sort(counts_asc.begin(), counts_asc.end());

	if (_minSamples > vec.size() / 2)
		_minSamples = (int)vec.size() / 2;

	while (currentBinCnt > kBins || counts_asc[0] < _minSamples) { //stop only when all bins are bigger than _minSamples and has no more than kBins.
		vector<int> indWithCnt = count2Inds[counts_asc[0]];
		//merge those bins if needed (if bigger then minSample stop) and smaller than kBins

		int indCountPos = indWithCnt[0]; //fetch first index
										 //check right and left poses - update splittingIndexes, splittingCnts, counts_asc, count2Inds
		int newCount = 0;
		bool addNewKey;
		int prevCntUpdated = 0;
		int removeVal = 0;
		if ((indCountPos != 0) && (indCountPos >= splittingCnts.size() - 1
			|| splittingCnts[indCountPos - 1] < splittingCnts[indCountPos + 1])) { //merge right
			prevCntUpdated = splittingCnts[indCountPos - 1];
			splittingCnts[indCountPos - 1] += splittingCnts[indCountPos];
			newCount = splittingCnts[indCountPos - 1];
			addNewKey = count2Inds.find(newCount) == count2Inds.end();
			removeVal = indCountPos - 1;
			int countPos = 0;
			if (!addNewKey) {
				countPos = medial::process::binary_search_position(count2Inds[newCount].data(), count2Inds[newCount].data() + count2Inds[newCount].size() - 1, indCountPos - 1);
			}
			count2Inds[newCount].insert(count2Inds[newCount].begin() + countPos, indCountPos - 1);
		}
		else { //merge left
			prevCntUpdated = splittingCnts[indCountPos + 1];
			splittingCnts[indCountPos + 1] += splittingCnts[indCountPos];
			newCount = splittingCnts[indCountPos + 1];
			addNewKey = count2Inds.find(newCount) == count2Inds.end();
			removeVal = indCountPos + 1;
			int countPos = 0;
			if (!addNewKey) {
				countPos = medial::process::binary_search_position(count2Inds[newCount].data(), count2Inds[newCount].data() + count2Inds[newCount].size() - 1, indCountPos + 1);
			}
			count2Inds[newCount].insert(count2Inds[newCount].begin() + countPos, indCountPos + 1);
		}
		//int splitingIndexPos = binary_search_index(splittingIndexes.data(), splittingIndexes.data() + splittingIndexes.size() - 1, indCountPos);
		splittingCnts.erase(splittingCnts.begin() + indCountPos);
		--currentBinCnt;
		count2Inds[counts_asc[0]].erase(count2Inds[counts_asc[0]].begin());
		int delPos = medial::process::binary_search_index(count2Inds[prevCntUpdated].data(), count2Inds[prevCntUpdated].data() + count2Inds[prevCntUpdated].size() - 1, removeVal);
		count2Inds[prevCntUpdated].erase(count2Inds[prevCntUpdated].begin() + delPos);

		if (count2Inds[counts_asc[0]].size() == 0) {
			//remove it from count2Inds, counts_asc - if was last only 1 element with this size
			count2Inds.erase(counts_asc[0]);
			counts_asc.erase(counts_asc.begin());
		}
		if (count2Inds.find(prevCntUpdated) != count2Inds.end() &&
			count2Inds[prevCntUpdated].size() == 0) {
			//remove it from count2Inds, counts_asc - if was last only 1 element with this size
			count2Inds.erase(prevCntUpdated);
			delPos = medial::process::binary_search_index(counts_asc.data(), counts_asc.data() + counts_asc.size() - 1, prevCntUpdated);
			counts_asc.erase(counts_asc.begin() + delPos);
		}

		if (addNewKey) { //add new key, count2Inds - was already handled
			int newPos = medial::process::binary_search_position(counts_asc.data(), counts_asc.data() + counts_asc.size() - 1, newCount);
			counts_asc.insert(counts_asc.begin() + newPos, newCount);
		}
		//move all other indexes after this index
		for (size_t i = 0; i < counts_asc.size(); ++i)
		{
			vector<int> oldInds = count2Inds[counts_asc[i]];
			for (size_t j = 0; j < oldInds.size(); ++j)
			{
				if (oldInds[j] > indCountPos) {
					--oldInds[j];
				}
			}
			count2Inds[counts_asc[i]] = oldInds;
		}
	}
	if (kBins != currentBinCnt) {
		//cout << "changed bins from " << kBins << " to " << currentBinCnt << endl;
		if (currentBinCnt == 1)
			has_error = true;
	}
	kBins = currentBinCnt;

	//update splittingIndexes to fit convention - match splittingCnts sizes
	vector<int> splittingIndexes((int)kBins - 1);
	int indSorted = 0;
	for (size_t i = 0; i < splittingIndexes.size(); ++i)
	{
		int binSizeTillNow = 0;
		while (indSorted < _histElements.size() && binSizeTillNow < splittingCnts[i])
		{
			int currCnt = _histElements[_sortedArr[indSorted]];
			binSizeTillNow += currCnt;
			++indSorted;
		}
		splittingIndexes[i] = indSorted;
		if (binSizeTillNow != splittingCnts[i])
		{
			throw logic_error("problem in initiale splitting");
		}
	}

	//search for optimal solution by running evaluate:
	bool hasChange = true;
	int i = 0;
	float currentScore;
	float maxScore;
	int maxDiff_val = -1;
	int maxSpliting_ind = -1;
	if (!evaluate(splittingIndexes, currentScore)) {
		throw logic_error("invalid start solution, change min_samples");
	}
	maxScore = currentScore;
	while (i < maxIters && hasChange)
	{
		hasChange = false;
		//search for change:
		for (size_t k = 0; k < splittingIndexes.size(); ++k)
		{
			//search all changes in index:
			for (int diff = -maxDist; diff <= maxDist; ++diff)
			{
				splittingIndexes[k] += diff;
				if (evaluate(splittingIndexes, currentScore)) {
					if (currentScore > maxScore) {
						maxScore = currentScore;
						maxSpliting_ind = (int)k;
						maxDiff_val = diff;
						hasChange = true;
					}
				}
				splittingIndexes[k] -= diff;
			}
		}
		if (hasChange) {
			//commit the change:
			splittingIndexes[maxSpliting_ind] += maxDiff_val;
		}

		++i;
	}

	//set indToVal, partitionValues
	indToVal = vector<float>((int)splittingIndexes.size() + 2);
	partitionValues = vector<float>((int)splittingIndexes.size() + 1);
	indToVal[0] = _sortedArr[0];
	float prevTh = _sortedArr[0];
	int prevInd = 0;
	for (size_t i = 0; i < splittingIndexes.size(); ++i)
	{
		indToVal[i + 1] = _sortedArr[splittingIndexes[i]];
		float sVal = _histElements[prevTh] * prevTh;
		int totalCnt = _histElements[prevTh];
		for (size_t k = prevInd; k < splittingIndexes[i]; ++k)
		{
			sVal += _histElements[_sortedArr[k]] * _sortedArr[k];
			totalCnt += _histElements[_sortedArr[k]];
		}
		partitionValues[i] = sVal / totalCnt; //calc mean in partition
		prevTh = _sortedArr[splittingIndexes[i]];
		prevInd = splittingIndexes[i];
	}
	indToVal[(int)indToVal.size() - 1] = _sortedArr[(int)_histElements.size() - 1] + (float)0.001;

	float sVal = _histElements[prevTh] * prevTh;
	int totalCnt = _histElements[prevTh];
	for (size_t k = prevInd; k < _histElements.size(); ++k)
	{
		sVal += _histElements[_sortedArr[k]] * _sortedArr[k];
		totalCnt += _histElements[_sortedArr[k]];
	}
	partitionValues[(int)partitionValues.size() - 1] = sVal / totalCnt;

	return splittingIndexes;
}

float _max_vec(const vector<float> &vec) {
	if (vec.size() == 0) {
		throw invalid_argument("vector can't  be empty");
	}
	float m = MED_MAT_MISSING_VALUE;

	for (size_t i = 0; i < vec.size(); ++i)
	{
		if (vec[i] != MED_MAT_MISSING_VALUE && (m == MED_MAT_MISSING_VALUE || m < vec[i])) {
			m = vec[i];
		}
	}

	return m;
}

float _min_vec(const vector<float> &vec) {
	if (vec.size() == 0) {
		throw invalid_argument("vector can't  be empty");
	}
	float m = MED_MAT_MISSING_VALUE;

	for (size_t i = 0; i < vec.size(); ++i)
	{
		if (vec[i] != MED_MAT_MISSING_VALUE && (m == MED_MAT_MISSING_VALUE || m > vec[i])) {
			m = vec[i];
		}
	}

	return m;
}

void SplitToBinsSimple(vector<float> &x, int binCnt, const float *min_val_given = NULL
	, const float *max_val_given = NULL, bool weighted_score = false) {
	if (binCnt == 0) {
		return;
	}
	float min_val, max_val;
	if (min_val_given != NULL)
		min_val = *min_val_given;
	else
		min_val = _min_vec(x);
	if (max_val_given != NULL)
		max_val = *max_val_given;
	else
		max_val = _max_vec(x);
	if (max_val == min_val) {
		max_val = min_val + 1;
	}
	vector<double> sums, tots;
	if (weighted_score) {
		sums.resize(binCnt);
		tots.resize(binCnt);
	}
	for (size_t k = 0; k < x.size(); ++k)
	{
		if (x[k] != MED_MAT_MISSING_VALUE) {
			int index = int(binCnt*(x[k] - min_val) / (max_val - min_val));
			if (!weighted_score)
				x[k] = (max_val - min_val) * index / binCnt + min_val;
			else {
				//trimming:
				if (index >= binCnt)
					index = binCnt - 1;
				if (index < 0)
					index = 0;

				sums[index] += x[k];
				++tots[index];
				x[k] = index;
			}
		}
	}
	if (weighted_score) {
		for (size_t k = 0; k < binCnt; ++k)
			if (tots[k] > 0)
				sums[k] /= tots[k];

		for (size_t k = 0; k < x.size(); ++k)
			if (x[k] != MED_MAT_MISSING_VALUE) 
				x[k] = sums[int(x[k])];
	}

}

int split_using_optimizer(vector<float> &x, const vector<float> &y, int binCnt, int minSamples) {
	bool has_error;
	BinSplitOptimizer opt;
	vector<float> indToVal;
	vector<float> partitionValues;
	vector<int> splitIndex = opt.SplitToBins(x, y, binCnt, minSamples, indToVal, partitionValues, has_error);
	if (has_error) {
		x.clear();
		x.resize(y.size(), 0); //all have same value
		return 1;
	}
	if (splitIndex.empty()) {
		//no splitting
		return 1;
	}

	for (size_t i = 0; i < x.size(); ++i)
	{
		//test on ind2Val
		int ind = 0;
		while (ind + 1 < indToVal.size() && x[i] >= indToVal[ind + 1]) {
			++ind;
		}

		float centerBinValue = partitionValues[ind];

		x[i] = centerBinValue;
	}

	return binCnt;

}

void medial::process::split_feature_to_bins(const BinSettings &setting, vector<float> &feature,
	const vector<int> &sel_indexes, vector<float> &y) {
	vector<float> sel_y;
	vector<float> qrf_preds;
	ROC_Params splitingParams;
	splitingParams.score_resolution = 0;
	MedQRF qrf;
	qrf.params.ntrees = 1; qrf.params.maxq = 1000;  qrf.params.type = QRF_TreeType::QRF_REGRESSION_TREE;
	qrf.params.ntry = 1; qrf.params.get_only_this_categ = 0; qrf.qf.take_all_samples = true;
	if (setting.max_value_cutoff != (float)INT_MAX || setting.min_value_cutoff != (float)INT_MIN) {
		//trim values:
		for (size_t i = 0; i < feature.size(); ++i)
		{
			if (feature[i] == MED_MAT_MISSING_VALUE)
				continue;
			if (feature[i] > setting.max_value_cutoff)
				feature[i] = setting.max_value_cutoff;
			if (feature[i] < setting.min_value_cutoff)
				feature[i] = setting.min_value_cutoff;
		}
	}
	const float *max_val_given = NULL, *min_val_given = NULL;
	if (setting.max_value_cutoff != (float)INT_MAX)
		max_val_given = &setting.max_value_cutoff;
	if (setting.min_value_cutoff != (float)INT_MIN)
		min_val_given = &setting.min_value_cutoff;
	switch (setting.split_method)
	{
	case BinSplitMethod::SameValueWidth:
		SplitToBinsSimple(feature, setting.binCnt, min_val_given, max_val_given, setting.weighted);

		break;
	case BinSplitMethod::IterativeMerge:
		splitingParams.score_bins = setting.binCnt;
		splitingParams.score_min_samples = setting.min_bin_count;
		splitingParams.score_resolution = (float)setting.min_res_value;
		splitingParams.show_warns = false;
		preprocess_bin_scores(feature, (Measurement_Params *)&splitingParams); //will merge till converge to binCnt
		break;
	case BinSplitMethod::PartitaionMover:
		if (!sel_indexes.empty()) {
			sel_y.resize(sel_indexes.size());
			for (size_t j = 0; j < sel_y.size(); ++j)
				sel_y[j] = y[sel_indexes[j]];
			split_using_optimizer(feature, sel_y, setting.binCnt, setting.min_bin_count);
		}
		else
			split_using_optimizer(feature, y, setting.binCnt, setting.min_bin_count);
		break;
	case BinSplitMethod::DynamicSplit:
		qrf.params.min_node = setting.min_bin_count;
		qrf.params.spread = (float)setting.min_res_value;
		qrf.params.max_depth = setting.binCnt;
		if (!sel_indexes.empty()) {
			sel_y.resize(sel_indexes.size());
			for (size_t j = 0; j < sel_y.size(); ++j)
				sel_y[j] = y[sel_indexes[j]];
			qrf.learn(feature, sel_y, (int)sel_y.size(), 1);
			qrf.predict(feature, qrf_preds, (int)sel_y.size(), 1);
		}
		else {
			qrf.learn(feature, y, (int)y.size(), 1);
			qrf.predict(feature, qrf_preds, (int)y.size(), 1);
		}
		feature.swap(qrf_preds);
		break;
	default:
		MTHROW_AND_ERR("Unsupported split method");
	}

}

int BinSettings::init(map<string, string>& map) {
	for (auto it = map.begin(); it != map.end(); ++it)
	{
		if (it->first == "min_bin_count")
			min_bin_count = med_stoi(it->second);
		else if (it->first == "min_res_value")
			min_res_value = stod(it->second);
		else if (it->first == "binCnt")
			binCnt = stoi(it->second);
		else if (it->first == "min_value_cutoff")
			min_value_cutoff = stof(it->second);
		else if (it->first == "max_value_cutoff")
			max_value_cutoff = stof(it->second);
		else if (it->first == "split_method")
			split_method = bin_method_name_to_type(it->second);
		else if (it->first == "weighted")
			weighted = med_stoi(it->second) > 0;
		else
			MTHROW_AND_ERR("Unsupported param \"%s\"\n", it->first.c_str());
	}
	return  0;
}

const unordered_map<int, string> BinSettings::name_to_method = {
	{ BinSplitMethod::SameValueWidth, "same_width"},
	{ BinSplitMethod::PartitaionMover, "partition_mover" },
	{ BinSplitMethod::IterativeMerge, "iterative_merge" },
	{ BinSplitMethod::DynamicSplit, "dynamic_split" }
};

BinSplitMethod BinSettings::bin_method_name_to_type(const string& bin_method) {
	for (auto it = name_to_method.begin(); it != name_to_method.end(); ++it)
		if (it->second == bin_method) {
			return BinSplitMethod(it->first);
		}
	string opts = medial::io::get_list_op(name_to_method);
	MTHROW_AND_ERR("Not Supported \"%s\" bin method. options are: %s\n",
		bin_method.c_str(), opts.c_str());
}

void medial::process::normalize_feature_to_uniform(const BinSettings &setting, vector<float> &feature) {
	//split the feature to bins first for histogram - afterwards we will change the histogram to uniform:
	if (feature.empty())
		MTHROW_AND_ERR("normalize got empty vector\n");
	vector<float> copy_feature = feature;
	vector<int> sel;
	split_feature_to_bins(setting, copy_feature, sel, copy_feature);
	vector<float> binnned_feature = copy_feature; //a copy of binning
	sort(copy_feature.begin(), copy_feature.end()); //lets create hisogram of values:
	vector<int> ordValue_count;
	vector<float> ordValues;
	ordValue_count.reserve(copy_feature.size());
	ordValues.reserve(copy_feature.size());
	float prev = copy_feature[0];
	ordValue_count.push_back(1);
	ordValues.push_back(copy_feature[0]);
	for (size_t i = 1; i < copy_feature.size(); ++i)
	{
		if (prev != copy_feature[i]) {
			prev = copy_feature[i];
			ordValues.push_back(copy_feature[i]);
			ordValue_count.push_back(0);
		}
		++ordValue_count.back();
	}
	float min_val = ordValues.front();
	float max_val = ordValues.back();
	if (min_val == max_val) {
		MWARN("warning: after binning - all values are the same. doing nothing\n");
		return;
	}
	double tot_size = (double)feature.size();
	//let calc width's:
	vector<double> bin_widths_cumsum(ordValues.size());
	bin_widths_cumsum[0] = ordValue_count[0] / tot_size;
	for (size_t i = 1; i < ordValues.size(); ++i)
	{
		double bin_ratio_width = ordValue_count[i] / tot_size; //the current bin_width [min, max]
		bin_widths_cumsum[i] = bin_widths_cumsum[i - 1] + bin_ratio_width; //cumsum
	}
	vector<float> min_bin_val(ordValues.size(), (float)INT_MAX),
		max_bin_val(ordValues.size(), (float)INT_MIN);
	for (size_t i = 0; i < binnned_feature.size(); ++i)
	{
		float val = binnned_feature[i];
		int idx_val = medial::process::binary_search_index(ordValues.data(), ordValues.data() + ordValues.size() - 1, val);
		if (idx_val < 0)
			MTHROW_AND_ERR("ERROR in binary_search. i=%d, val=%f, full_size=%d\n",
			(int)i, val, (int)feature.size());
		if (min_bin_val[idx_val] > val)
			min_bin_val[idx_val] = val;
		if (max_bin_val[idx_val] < val)
			max_bin_val[idx_val] = val;

	}
	for (size_t i = 0; i < binnned_feature.size(); ++i)
	{
		float val = binnned_feature[i];
		int idx_val = medial::process::binary_search_index(ordValues.data(), ordValues.data() + ordValues.size() - 1, val);
		double start_idx = 0;
		if (idx_val > 0)
			start_idx = bin_widths_cumsum[idx_val - 1];
		double end_idx = bin_widths_cumsum[idx_val]; //need to get that width between [0,1]
		//spread val between start_idx to end_idx - linearly

		if (min_bin_val[idx_val] == max_bin_val[idx_val])
			val = float((start_idx + end_idx) / 2);
		else
			val = float(start_idx + (val - min_bin_val[idx_val]) /
			(max_bin_val[idx_val] - min_bin_val[idx_val]) * (end_idx - start_idx));

		feature[i] = val;
	}


}
