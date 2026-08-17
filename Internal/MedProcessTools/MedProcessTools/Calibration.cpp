#include "Calibration.h"
#include "MedAlgo/MedAlgo/MedAlgo.h"
#include "MedAlgo/MedAlgo/MedLinearModel.h"

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MED_MODEL
#define LOCAL_LEVEL	LOG_DEF_LEVEL

// *************************************************************************************************
// Functions for calibrators
// *************************************************************************************************


unordered_map<int, string> calibration_method_to_name = {
	{probability_time_window, "time_window"},
	{ probability_binning, "binning" },
	{ probability_platt_scale, "platt_scale" },
	{ probability_isotonic, "isotonic_regression" }
};

string calibration_entry::str() const {
	char buffer[5000];

	snprintf(buffer, sizeof(buffer), "Bin %d :: [%1.6f, %1.6f] => {kaplan_meier=%2.5f, mean_outcome=%2.5f, mean_pred=%2.5f} | cnt_cases=%2.1f(%d), cnt_controls=%2.1f(%d)}",
		bin, min_pred, max_pred, kaplan_meier, mean_outcome, mean_pred, cnt_cases, (int)cnt_cases_no_w, cnt_controls, (int)cnt_controls_no_w);

	return string(buffer);
}

CalibrationTypes clibration_name_to_type(const string& calibration_name) {
	for (auto it = calibration_method_to_name.begin(); it != calibration_method_to_name.end(); ++it)
		if (it->second == calibration_name)
			return CalibrationTypes(it->first);
	string opts = medial::io::get_list_op(calibration_method_to_name);
	MTHROW_AND_ERR("unknown calibration_name \"%s\"\nOptions Are:%s\n",
		calibration_name.c_str(), opts.c_str());
}

void Calibrator::get_input_fields(vector<Effected_Field> &fields) const {
	Effected_Field f(Field_Type::PREDICTION, "0");
	Effected_Field f_w(Field_Type::NUMERIC_ATTRIBUTE, weights_attr_name);
	
	fields.push_back(f); //prediction
	fields.push_back(f_w); //weight if exists
}

void Calibrator::get_output_fields(vector<Effected_Field> &fields) const {
	Effected_Field f(Field_Type::PREDICTION, "0");
	
	fields.push_back(f); //prediction
}

int Calibrator::init(map<string, string>& mapper) {
	for (auto entry : mapper) {
		string field = entry.first;
		//! [Calibrator::init]
		if (field == "calibration_type") calibration_type = clibration_name_to_type(entry.second);
		else if (field == "estimator_type") estimator_type = entry.second;
		else if (field == "binning_method") binning_method = entry.second;
		else if (field == "bins_num") bins_num = stoi(entry.second);
		else if (field == "pos_sample_min_time_before_case") pos_sample_min_time_before_case = stoi(entry.second);
		else if (field == "pos_sample_max_time_before_case") pos_sample_max_time_before_case = stoi(entry.second);
		else if (field == "km_time_resolution") km_time_resolution = stoi(entry.second);
		else if (field == "do_calibration_smoothing") do_calibration_smoothing = stoi(entry.second);
		else if (field == "min_cases_for_calibration_smoothing_pct") min_cases_for_calibration_smoothing_pct = stoi(entry.second);
		else if (field == "min_preds_in_bin") min_preds_in_bin = stoi(entry.second);
		else if (field == "min_score_res") min_score_res = stof(entry.second);
		else if (field == "min_prob_res") min_prob_res = stof(entry.second);
		else if (field == "fix_pred_order") fix_pred_order = stoi(entry.second) > 0;
		else if (field == "poly_rank") poly_rank = stoi(entry.second);
		else if (field == "control_weight_down_sample") control_weight_down_sample = stof(entry.second);
		else if (field == "censor_controls") censor_controls = stoi(entry.second);
		else if (field == "n_top_controls") n_top_controls = stoi(entry.second);
		else if (field == "n_bottom_cases") n_bottom_cases = stoi(entry.second);
		else if (field == "verbose") verbose = stoi(entry.second) > 0;
		else if (field == "use_split") use_split = stoi(entry.second);
		else if (field == "use_p") use_p = stof(entry.second);
		else if (field == "weights_attr_name") weights_attr_name = entry.second;
		else if (field == "min_control_bins") min_control_bins = stoi(entry.second);
		else if (field == "use_isotonic") use_isotonic = stoi(entry.second) > 0;
		else if (field == "pp_type") {} //ignore
		else MTHROW_AND_ERR("unknown init option [%s] for Calibrator\n", field.c_str());
		//! [Calibrator::init]
	}

	return 0;
}

void Calibrator::learn_isotonic_regression(const vector<float> &x, const vector<float> &y, const vector<float> &weights, vector<float> &min_range, vector<float> &max_range, vector<float> &map_prob, int n_top_controls, int n_bottom_cases,
	bool verbose) {

	int n = (int)x.size();
	vector<vector<float>> x2y(n);

	for (int i = 0; i < n; i++)
	{
		float curr_w = (y[i] > 0) ? weights[i] : weights[i] * control_weight_down_sample;
		x2y[i] = { x[i],y[i], curr_w };
	}


	sort(x2y.begin(), x2y.end(), [](const vector<float> &v1, const vector<float> &v2) {return (v1[0] < v2[0]); });

	// Add regularizers
	float min_score = x2y[0][0];
	float max_score = x2y.back()[0];
	float min_val = x2y[0][1], max_val = x2y[0][1];
	for (size_t i = 1; i < x2y.size(); i++) {
		if (x2y[i][1] > max_val)
			max_val = x2y[i][1];
		if (x2y[i][1] < min_val)
			min_val = x2y[i][1];
	}
	x2y.insert(x2y.begin(), n_bottom_cases, { min_score,max_val,1 });
	x2y.insert(x2y.end(), n_top_controls, { max_score,min_val,1 });

	n += n_bottom_cases + n_top_controls;

	// PAV
	vector<double> nag(n);
	vector<float> val(n);

	unsigned int i, j;
	nag[0] = x2y[0][2];
	val[0] = x2y[0][1];

	j = 0;
	vector<float> min_preds(n);
	vector<float> max_preds(n);

	vector<int> num_control_bins(n);
	num_control_bins[0] = 1 - x2y[0][1];
	for (i = 1; i < n; i++) {

		j += 1;
		val[j] = x2y[i][1];
		nag[j] = x2y[i][2];
		max_preds[j] = x2y[i][0];
		num_control_bins[j] = 1 - x2y[i][1];
		while ((j > 0) && ((val[j] <= val[j - 1]) || (num_control_bins[j - 1] < min_control_bins))) //change into val[j]>val[j-1] to have a non-increasing monotonic regression.
		{
			val[j - 1] = (nag[j] * val[j] + nag[j - 1] * val[j - 1]) / (nag[j] + nag[j - 1]);
			nag[j - 1] += nag[j];
			max_preds[j - 1] = max_preds[j];
			num_control_bins[j - 1] += num_control_bins[j];
			j--;
		}
	}

	// if needed, combine 2 last bins
	if (num_control_bins[j] < min_control_bins)
	{
		val[j - 1] = (nag[j] * val[j] + nag[j - 1] * val[j - 1]) / (nag[j] + nag[j - 1]);
		nag[j - 1] += nag[j];
		max_preds[j - 1] = max_preds[j];
		num_control_bins[j - 1] += num_control_bins[j];
		j--;
	}



	min_preds[0] = (float)INT32_MIN;
	max_preds[j] = (float)INT32_MAX;
	for (int i = 1; i <= j; i++)
		min_preds[i] = max_preds[i - 1];

	min_preds.resize(j + 1);
	max_preds.resize(j + 1);
	val.resize(j + 1);

	reverse(min_preds.begin(), min_preds.end());
	reverse(max_preds.begin(), max_preds.end());
	reverse(val.begin(), val.end());

	min_range = min_preds;
	max_range = max_preds;
	map_prob = val;

	if (verbose)
	{
		float sum_weights = 0;
		for (auto &xy : x2y)
			sum_weights += xy[2];

		nag.resize(j + 1);
		reverse(nag.begin(), nag.end()); //to match order of other vectors
		MLOG("Created %d bins for mapping prediction scores to probabilities\n", map_prob.size());
		for (size_t i = 0; i < map_prob.size(); ++i)
			MLOG("Range: [%2.4f, %2.4f] => %2.4f | %1.2f%%(%f / %f)\n",
				min_range[i], max_range[i], map_prob[i],
				100 * double(nag[i]) / sum_weights, nag[i], sum_weights);
	}

}


double Calibrator::calc_kaplan_meier(vector<double> controls_per_time_slot, vector<double> cases_per_time_slot,
	double controls_factor) {
	double prob = 1.0;
	double total_all = 0;
	for (int i = 0; i < controls_per_time_slot.size(); i++)
		total_all += (controls_per_time_slot[i] * controls_factor) + cases_per_time_slot[i];
	//MLOG("size %d total_controls_all %d\n", controls_per_time_slot.size(), total_controls_all);
	for (int i = 0; i < controls_per_time_slot.size(); i++) {
		if (total_all <= 0) {
			MWARN("Reached 0 samples at time slot [%d]. Quitting\n", i);
			break;
		}
		prob *= (1.0 - ((float)cases_per_time_slot[i]) / total_all);
		total_all -= (controls_per_time_slot[i] * controls_factor + cases_per_time_slot[i]);
	}
	return 1.0 - prob;
}

// expand to neighbor calibration entries, until finding enough cases
void Calibrator::smooth_calibration_entries(const vector<calibration_entry>& cals, vector<calibration_entry>& smooth_cals, double controls_factor) {
	smooth_cals.clear();
	int cases = 0;
	for (auto& c : cals)
		cases += c.cnt_cases;
	int min_cases_for_calibration = min_cases_for_calibration_smoothing_pct * cases / 100;
	MLOG("smooth_calibration_entries requiring min_cases_for_calibration_smoothing = [%d * %d / 100 = %d]\n", min_cases_for_calibration_smoothing_pct, cases,
		min_cases_for_calibration);

	for (int s = 0; s < cals.size(); s++) {
		int end = s, start = s;
		int controls = cals[start].cnt_controls;
		int cases = cals[start].cnt_cases;
		vector<double> controls_per_time_slot;
		vector<double> cases_per_time_slot;
		for (size_t j = 0; j < cals[start].controls_per_time_slot.size(); j++)
		{
			controls_per_time_slot.push_back(cals[start].controls_per_time_slot[j]);
			cases_per_time_slot.push_back(cals[start].cases_per_time_slot[j]);
		}

		while (cases < min_cases_for_calibration) {
			if (end == (cals.size() - 1) && start == 0)
				// the entire calibration table holds less than min_cases_for_calibration cases
				break;
			if (end < cals.size() - 1) {
				end++;
				controls += cals[end].cnt_controls;
				cases += cals[end].cnt_cases;
				for (size_t j = 0; j < cals[end].controls_per_time_slot.size(); j++)
				{
					controls_per_time_slot[j] += cals[end].controls_per_time_slot[j];
					cases_per_time_slot[j] += cals[end].cases_per_time_slot[j];
				}
			}
			if (start > 0) {
				start--;
				controls += cals[start].cnt_controls;
				cases += cals[start].cnt_cases;
				for (size_t j = 0; j < cals[start].controls_per_time_slot.size(); j++)
				{
					controls_per_time_slot[j] += cals[start].controls_per_time_slot[j];
					cases_per_time_slot[j] += cals[start].cases_per_time_slot[j];
				}
			}
		}
		calibration_entry res = cals[s];
		res.cnt_controls = controls;
		res.cnt_cases = cases;
		res.mean_outcome = 1.0f * cases / (cases + controls);
		res.cumul_pct = cals[start].cumul_pct;
		res.kaplan_meier = (float)calc_kaplan_meier(controls_per_time_slot, cases_per_time_slot, controls_factor);
		smooth_cals.push_back(res);
	}
}

void collect_preds_labels(const vector<MedSample>& orig_samples,
	vector<float> &preds, vector<float> &labels) {
	preds.resize(orig_samples.size());
	labels.resize(orig_samples.size());
	for (size_t i = 0; i < orig_samples.size(); ++i)
	{
		if (orig_samples[i].prediction.empty())
			MTHROW_AND_ERR("no prediction for samples %d\n", (int)i);
		preds[i] = orig_samples[i].prediction[0];
		labels[i] = orig_samples[i].outcome;
	}
}

inline float apply_binned_prob(float pred, const vector<float> &min_range, const vector<float> &max_range,
	const vector<float> &map_prob) {
	int pos = medial::process::binary_search_position(min_range.data(), min_range.data() + min_range.size() - 1, pred, true);

	if (pos >= map_prob.size())
		pos = (int)map_prob.size() - 1;
	return map_prob[pos];
}

void apply_binned_prob(const vector<float> &preds, const vector<float> &min_range,
	const vector<float> &max_range, const vector<float> &map_prob, vector<float> &probs) {
	probs.resize(preds.size());

	for (size_t i = 0; i < probs.size(); ++i)
		probs[i] = apply_binned_prob(preds[i], min_range, max_range, map_prob); //search for right range:
}

template<class T> float apply_platt_scale(T pred, const vector<double> &params) {
	double val = params[0];
	for (size_t k = 1; k < params.size(); ++k)
		val += params[k] * pow(double(pred), double(k));
	val = 1 / (1 + exp(val));//Platt Scale technique for probability calibaration
	return (float)val;
}
template<class T, class L> void apply_platt_scale(const vector<T> &preds, const vector<double> &params, vector<L> &converted) {
	converted.resize((int)preds.size());
	for (size_t i = 0; i < converted.size(); ++i)
		converted[i] = apply_platt_scale(preds[i], params);//Platt Scale technique for probability calibaration
}
template void apply_platt_scale<double, double>(const vector<double> &preds, const vector<double> &params, vector<double> &converted);
template void apply_platt_scale<double, float>(const vector<double> &preds, const vector<double> &params, vector<float> &converted);
template void apply_platt_scale<float, double>(const vector<float> &preds, const vector<double> &params, vector<double> &converted);
template void apply_platt_scale<float, float>(const vector<float> &preds, const vector<double> &params, vector<float> &converted);

int Calibrator::apply_time_window(vector<MedSample>& samples) const {

	int type;
	if (estimator_type == "kaplan_meier") {
		type = 1;
		MLOG("calibrating [%d] samples using kaplan_meier estimator\n", samples.size());
	}
	else if (estimator_type == "mean_cases") {
		type = 0;
		MLOG("calibrating [%d] samples using mean_cases estimator\n", samples.size());
	}
	else if (estimator_type == "bin") {
		type = 2;
		MLOG("calibrating [%d] samples using bin estimator\n", samples.size());
	}
	else MTHROW_AND_ERR("unknown estimator type [%s]", estimator_type.c_str());

	for (auto& s : samples)
		s.prediction[0] = calibrate_pred(s.prediction[0], type);


	return 0;
}


void write_to_predicition(vector<MedSample>& samples, vector<float> &probs) {
	int idx = 0;
	for (auto& s : samples) {
		s.prediction.resize(1);
		s.prediction[0] = probs[idx];
		++idx;
	}
}


int Calibrator::Apply(MedSamples& samples) {
	vector<MedSample> samples_vec;
	samples.export_to_sample_vec(samples_vec);
	int return_val = Apply(samples_vec);
	samples.import_from_sample_vec(samples_vec);
	return return_val;
}

void Calibrator::Apply(const vector<float> &preds, vector<float> &probs) const {
	switch (calibration_type)
	{
	case CalibrationTypes::probability_time_window:
		MTHROW_AND_ERR("Error Calibrator::Apply for single pred is not supported for probability_time_window\n");
	case CalibrationTypes::probability_binning:
	case CalibrationTypes::probability_isotonic:
		apply_binned_prob(preds, min_range, max_range, map_prob, probs);
		break;
	case CalibrationTypes::probability_platt_scale:
		apply_platt_scale(preds, platt_params, probs);
		break;
	default:
		MTHROW_AND_ERR("Unsupported implementation for applying calibration method %s\n",
			calibration_method_to_name[calibration_type].c_str());
	}
}

float Calibrator::Apply(float pred) const {
	switch (calibration_type)
	{
	case CalibrationTypes::probability_time_window:
		MTHROW_AND_ERR("Error Calibrator::Apply for single pred is not supported for probability_time_window\n");
	case CalibrationTypes::probability_binning:
	case CalibrationTypes::probability_isotonic:
		return apply_binned_prob(pred, min_range, max_range, map_prob);
	case CalibrationTypes::probability_platt_scale:
		return apply_platt_scale(pred, platt_params);
	default:
		MTHROW_AND_ERR("Unsupported implementation for applying calibration method %s\n",
			calibration_method_to_name[calibration_type].c_str());
	}
}

int Calibrator::Apply(vector <MedSample>& samples) {
	vector<float> preds, labels, probs;
	switch (calibration_type)
	{
	case CalibrationTypes::probability_time_window:
		return apply_time_window(samples);
		break;
	case CalibrationTypes::probability_binning:
	case CalibrationTypes::probability_isotonic:
		collect_preds_labels(samples, preds, labels);
		apply_binned_prob(preds, min_range, max_range, map_prob, probs);
		write_to_predicition(samples, probs);

		break;
	case CalibrationTypes::probability_platt_scale:
		collect_preds_labels(samples, preds, labels);
		apply_platt_scale(preds, platt_params, probs);
		write_to_predicition(samples, probs);
		break;
	default:
		MTHROW_AND_ERR("Unsupported implementation for applying calibration method %s\n",
			calibration_method_to_name[calibration_type].c_str());
	}
	return 0;
}

int Calibrator::Learn(const MedSamples& orig_samples) {
	vector<MedSample> samples;
	orig_samples.export_to_sample_vec(samples);
	return Learn(samples, orig_samples.time_unit);
}


void Calibrator::Apply(MedFeatures &matrix) {
	Apply(matrix.samples);
}

bool get_weights(const vector<MedSample>& orig_samples, const string &attr, vector<float> &weights) {
	bool has_miss = false, found = false;
	weights.resize(orig_samples.size(), 1); //give 1 weights to all
	for (size_t i = 0; i < orig_samples.size(); ++i) {
		if (orig_samples[i].attributes.find(attr) == orig_samples[i].attributes.end())
			has_miss = true;
		else {
			weights[i] = orig_samples[i].attributes.at(attr);
			found = true;
		}
	}
	if (has_miss && found)
		MWARN("Warning get_weights: has weights for some of the samples\n");
	else if (found)
		MLOG("Read Weights from samples attr %s\n", attr.c_str());
	return found;
}

int Calibrator::learn_time_window(const vector<MedSample>& orig_samples, const int samples_time_unit) {
	vector<float> weights;
	//read weights from samples:
	get_weights(orig_samples, weights_attr_name, weights);

	bool do_km;
	if (estimator_type == "kaplan_meier")
		do_km = true;
	else if (estimator_type == "mean_cases")
		do_km = false;
	else MTHROW_AND_ERR("unknown estimator type [%s]", estimator_type.c_str());

	cals.clear();
	float min_pred = 100000.0, max_pred = -100000.0;
	int cases = 0;
	double w_cases = 0;
	set<float> unique_preds;
	vector<MedSample> samples;
	for (int i = 0; i < orig_samples.size(); ++i) {
		MedSample e = orig_samples[i];

		if (unique_preds.find(e.prediction[0]) == unique_preds.end())
			unique_preds.insert(e.prediction[0]);
		if (e.prediction[0] < min_pred)
			min_pred = e.prediction[0];
		if (e.prediction[0] > max_pred)
			max_pred = e.prediction[0];
		int gap = med_time_converter.convert_times(samples_time_unit, time_unit, e.outcomeTime) - med_time_converter.convert_times(samples_time_unit, time_unit, e.time);
		if (gap < pos_sample_min_time_before_case)
			// too close to outcome date or censor date (chance for peeking, or even beyond the outcome date)
			continue;
		if (censor_controls && e.outcome == 0 && gap < pos_sample_max_time_before_case)
			// In censor_controls mode - remove controls without long-enough followup time
			continue;
		if (e.outcome >= 1 && gap > pos_sample_max_time_before_case)
			// too far case is considered as control
			e.outcome = 0;
		if (e.outcome >= 1) {
			++cases;
			w_cases += weights[i];
		}
		samples.push_back(e);
	}
	std::sort(samples.begin(), samples.end(), comp_sample_pred);
	weights.clear(); //reorder weigths to samples
	get_weights(samples, weights_attr_name, weights);

	MLOG("eligible samples [%d] cases [%d]\n", int(samples.size()), cases);
	int max_samples_per_bin = 0;
	int max_cases_per_bin = 0;
	float max_delta_in_bin = 0.0;
	if (binning_method == "unique_score_per_bin") {
		bins_num = (int)unique_preds.size();
		MLOG("unique_score_per_bin, bins_num [%d] \n", bins_num);
	}
	else if (binning_method == "equal_num_of_samples_per_bin") {
		max_samples_per_bin = max((int)samples.size() / bins_num, 10);
		MLOG("equal_num_of_samples_per_bin bins_num: %d max_samples_per_bin: %d \n",
			bins_num, max_samples_per_bin);
	}
	else if (binning_method == "equal_num_of_cases_per_bin") {
		//max_cases_per_bin = max((int)w_cases / bins_num, 10);
		max_cases_per_bin = max(cases / bins_num, 10);
		MLOG("equal_num_of_cases_per_bin bins_num: %d max_cases_per_bin: %d \n",
			bins_num, max_cases_per_bin);
	}
	else if (binning_method == "equal_score_delta_per_bin") {
		max_delta_in_bin = (max_pred - min_pred) / bins_num;
		MLOG("equal_score_delta_per_bin min_pred: %f max_pred: %f max_delta_in_bin: %f \n",
			min_pred, max_pred, (int)samples.size(), max_delta_in_bin);
	}
	else MTHROW_AND_ERR("unknown binning method [%s]\n", binning_method.c_str());

	vector<double> cnt_cases;
	vector<double> cnt_controls;
	vector<float> bin_max_preds;
	vector<float> bin_min_preds;
	vector<float> bin_sum_preds;
	vector<double> cnt_cases_no_w, cnt_ctrl_no_w;
	vector<vector<double>> bin_controls_per_time_slot;
	vector<vector<double>> bin_cases_per_time_slot;
	int km_time_slots = (pos_sample_max_time_before_case - pos_sample_min_time_before_case) / km_time_resolution;
	if (km_time_slots <= 0)
		km_time_slots = 1;
	MLOG("km_time_slots [%d] \n", km_time_slots);
	//init arrays:
	//int sz_loop = (bins_num + 1) * 2;
	int sz_loop = (bins_num + 1);
	cnt_cases.resize(sz_loop);
	cnt_controls.resize(sz_loop);
	cnt_cases_no_w.resize(sz_loop);
	cnt_ctrl_no_w.resize(sz_loop);
	bin_sum_preds.resize(sz_loop);
	bin_max_preds.resize(sz_loop);
	bin_min_preds.resize(sz_loop, min_pred);
	for (size_t i = 0; i < sz_loop; i++)
	{
		vector<double> controls_per_time_slot(km_time_slots + 1);
		vector<double> cases_per_time_slot(km_time_slots + 1);
		bin_controls_per_time_slot.push_back(controls_per_time_slot);
		bin_cases_per_time_slot.push_back(cases_per_time_slot);
	}
	int bin = 1;
	//end init

	//stats for all samples in time slots and general - iterating on scores (same bin, or move to next one)
	float prev_pred = min_pred;
	double tot_weight = 0;
	double controls_factor = 1;
	if (control_weight_down_sample > 0)
		controls_factor = control_weight_down_sample;
	for (int i = 0; i < samples.size(); ++i) {
		const MedSample &o = samples[i];
		int gap = med_time_converter.convert_times(samples_time_unit, time_unit, o.outcomeTime) - med_time_converter.convert_times(samples_time_unit, time_unit, o.time);
		if (
			(bin < bins_num)
			//&& (prev_pred != o.prediction[0]) //can't break prediction score
			&&
			(
			(binning_method == "unique_score_per_bin" && prev_pred != o.prediction[0]) ||
				(binning_method == "equal_num_of_samples_per_bin" && cnt_ctrl_no_w[bin] + cnt_cases_no_w[bin] >= max_samples_per_bin) ||
				(binning_method == "equal_score_delta_per_bin" && (o.prediction[0] - bin_min_preds[bin]) > max_delta_in_bin) ||
				(binning_method == "equal_num_of_cases_per_bin" && cnt_cases_no_w[bin] >= max_cases_per_bin))
			)
		{
			++bin;
			bin_min_preds[bin] = o.prediction[0];
		}
		int time_slot;
		if (gap > pos_sample_max_time_before_case)
			time_slot = km_time_slots;
		else
			time_slot = (gap - pos_sample_min_time_before_case) / km_time_resolution;

		if (o.outcome >= 1) {
			cnt_cases[bin] += weights[i];
			bin_cases_per_time_slot[bin][time_slot] += weights[i];
			++cnt_cases_no_w[bin];
		}
		else {
			cnt_controls[bin] += weights[i] * controls_factor;
			bin_controls_per_time_slot[bin][time_slot] += weights[i] * controls_factor;
			cnt_ctrl_no_w[bin] += controls_factor;
		}

		bin_max_preds[bin] = o.prediction[0];
		prev_pred = o.prediction[0];
		if (o.outcome > 0) {
			tot_weight += weights[i];
			bin_sum_preds[bin] += o.prediction[0] * weights[i];
		}
		else {
			tot_weight += controls_factor * weights[i];
			bin_sum_preds[bin] += static_cast<double>(o.prediction[0]) * static_cast<double>(weights[i]) * controls_factor;
		}
	}
	//end stats collection

	double cumul_cnt = 0;

	//calibration calc:
	for (int i = 1; i <= bin; i++)
	{
		calibration_entry ce;
		ce.bin = i;
		ce.min_pred = bin_min_preds[i];
		ce.max_pred = bin_max_preds[i];
		ce.cnt_controls = cnt_controls[i]; ce.cnt_cases = cnt_cases[i];
		ce.cnt_controls_no_w = cnt_ctrl_no_w[i]; ce.cnt_cases_no_w = cnt_cases_no_w[i];
		ce.mean_pred = 1.0f * bin_sum_preds[i] / (cnt_controls[i] + cnt_cases[i]);
		ce.cumul_pct = 1.0f * (cumul_cnt + ((cnt_controls[i] + cnt_cases[i]) / 2)) / (float)tot_weight;
		ce.controls_per_time_slot = bin_controls_per_time_slot[i];
		ce.cases_per_time_slot = bin_cases_per_time_slot[i];
		if (do_km) {
			ce.kaplan_meier = (float)calc_kaplan_meier(bin_controls_per_time_slot[i], bin_cases_per_time_slot[i], 1.0);
			ce.mean_outcome = 1.0F * cnt_cases[i] / (cnt_controls[i] + cnt_cases[i]);
		}
		else {
			ce.kaplan_meier = 0.0;
			ce.mean_outcome = 1.0F * cnt_cases[i] / (cnt_controls[i] + cnt_cases[i]);
		}
		cumul_cnt += (ce.cnt_controls) + ce.cnt_cases;
		cals.push_back(ce);
	}

	if (use_isotonic) {
		vector<float> collected_bin_idx(cals.size()), collected_probs(cals.size());
		for (size_t i = 0; i < cals.size(); ++i)
		{
			collected_bin_idx[i] = i;
			if (do_km)
				collected_probs[i] = cals[i].kaplan_meier;
			else
				collected_probs[i] = cals[i].mean_outcome;
		}
		vector<float> min_r, max_r, map_r;
		learn_isotonic_regression(collected_bin_idx, collected_probs, weights, min_r, max_r, map_r, n_top_controls, n_bottom_cases, verbose);
		//use new bins:
		vector<calibration_entry> new_cals(map_r.size());
		cumul_cnt = 0;
		for (int i = 0; i < new_cals.size(); ++i)
		{
			calibration_entry ce;
			ce.bin = i + 1;
			int min_idx_bin, max_idx_bin;
			if (max_r[i] >= cals.size())
				max_idx_bin = (int)cals.size() - 1;
			else
				max_idx_bin = (int)max_r[i];
			if (min_r[i] < 0)
				min_idx_bin = -1;
			else
				min_idx_bin = (int)min_r[i];
			++min_idx_bin;

			ce.min_pred = cals[min_idx_bin].min_pred;
			ce.max_pred = cals[max_idx_bin].max_pred;
			ce.cnt_controls = 0; ce.cnt_cases = 0;
			ce.cnt_controls_no_w = 0;  ce.cnt_cases_no_w = 0;
			ce.controls_per_time_slot.resize(km_time_slots + 1);
			ce.cases_per_time_slot.resize(km_time_slots + 1);
			ce.mean_pred = 0;
			int cnt = 0;
			for (int j = min_idx_bin; j <= max_idx_bin; ++j)
			{
				ce.cnt_controls += cals[j].cnt_controls;
				ce.cnt_cases += cals[j].cnt_cases;
				ce.cnt_controls_no_w += cals[j].cnt_controls_no_w;
				ce.cnt_cases_no_w += cals[j].cnt_cases_no_w;
				for (size_t k = 0; k < km_time_slots + 1; ++k)
					ce.controls_per_time_slot[k] += cals[j].controls_per_time_slot[k];
				for (size_t k = 0; k < km_time_slots + 1; ++k)
					ce.cases_per_time_slot[k] += cals[j].cases_per_time_slot[k];
				ce.mean_pred += cals[j].mean_pred;
				++cnt;
			}
			ce.mean_pred = ce.mean_pred / (float)cnt;
			ce.cumul_pct = 1.0f * (cumul_cnt + (((ce.cnt_controls) + ce.cnt_cases) / 2)) / (float)tot_weight;

			if (do_km) {
				ce.kaplan_meier = map_r[i];
				ce.mean_outcome = 1.0F * ce.cnt_cases / (ce.cnt_controls + ce.cnt_cases);
			}
			else {
				ce.kaplan_meier = 0.0;
				ce.mean_outcome = map_r[i];
			}
			cumul_cnt += (ce.cnt_controls) + ce.cnt_cases;
			new_cals[i] = ce;
		}
		reverse(new_cals.begin(), new_cals.end());
		for (size_t i = 0; i < new_cals.size(); ++i)
			new_cals[i].bin = (int)i + 1;

		cals = std::move(new_cals);
	}
	//smooth calc
	if (do_calibration_smoothing) {
		vector<calibration_entry> smooth_cals;
		smooth_calibration_entries(cals, smooth_cals, 1.0);
		cals = smooth_cals;
	}
	if (verbose)
		dprint("");

	return 0;
}

void get_counts(const vector<int> &inds, const vector<float> &y, const vector<float> &w, double &cases_cnt,
	double &cntls_cnt, double control_weight) {
	cases_cnt = 0, cntls_cnt = 0;
	if (control_weight <= 0)
		control_weight = 1;

	for (int ind : inds) {
		cases_cnt += int(y[ind] > 0) * w[ind];
		cntls_cnt += int(y[ind] <= 0)* w[ind] * control_weight;
	}
}

void learn_binned_probs(const vector<float> &x, const vector<float> &y, const vector<float> &weights,
	int min_bucket_size, float min_score_jump, float min_prob_jump, bool fix_prob_order,
	vector<float> &min_range, vector<float> &max_range, vector<float> &map_prob, double control_weight_down_sample, bool verbose) {
	unordered_map<float, vector<int>> score_to_indexes;
	vector<float> unique_scores;
	for (size_t i = 0; i < x.size(); ++i)
	{
		if (score_to_indexes.find(x[i]) == score_to_indexes.end())
			unique_scores.push_back(x[i]);
		score_to_indexes[x[i]].push_back((int)i);
	}
	sort(unique_scores.begin(), unique_scores.end());
	int sz = (int)unique_scores.size();

	float curr_max = (float)INT32_MAX; //unbounded
	float curr_min = curr_max;
	int pred_sum = 0;
	double curr_cnt = 0;
	vector<int> bin_cnts;
	for (int i = sz - 1; i >= 0; --i)
	{
		//update values curr_cnt, pred_avg
		double cases_cnt, cntls_cnt;
		get_counts(score_to_indexes[unique_scores[i]], y, weights, cases_cnt, cntls_cnt, control_weight_down_sample);

		pred_sum += cases_cnt;
		curr_cnt += (cases_cnt + cntls_cnt);

		if (curr_cnt > min_bucket_size && curr_max - unique_scores[i] > min_score_jump) {
			//flush buffer
			curr_min = unique_scores[i];
			max_range.push_back(curr_max);
			min_range.push_back(curr_min);
			map_prob.push_back(float(double(pred_sum) / curr_cnt));
			bin_cnts.push_back(curr_cnt);

			//init new buffer:
			curr_cnt = 0;
			curr_max = unique_scores[i];
			pred_sum = 0;
		}
	}
	if (curr_cnt > 0) {
		//flush last buffer
		curr_min = (float)INT32_MIN;
		max_range.push_back(curr_max);
		min_range.push_back(curr_min);
		map_prob.push_back(float(double(pred_sum) / curr_cnt));
		bin_cnts.push_back(curr_cnt);
	}

	//unite similar prob bins:
	vector<int> ind_to_unite;
	for (int i = (int)map_prob.size() - 1; i >= 1; --i)
		if (abs(map_prob[i] - map_prob[i - 1]) < min_prob_jump ||
			(fix_prob_order && map_prob[i] > map_prob[i - 1])) { //unite bins:
			ind_to_unite.push_back(i);
			int new_count = bin_cnts[i] + bin_cnts[i - 1];
			float new_prob = (map_prob[i] * bin_cnts[i] + map_prob[i - 1] * bin_cnts[i - 1]) / new_count;
			float max_th = max_range[i - 1];
			float min_th = min_range[i];
			min_range[i - 1] = min_th;
			max_range[i - 1] = max_th;
			map_prob[i - 1] = new_prob;
			bin_cnts[i - 1] = new_count;
		}

	//unite from end to start:
	for (int i = 0; i < ind_to_unite.size(); ++i)
	{
		int unite_index = ind_to_unite[i];
		//delete old records:
		min_range.erase(min_range.begin() + unite_index);
		max_range.erase(max_range.begin() + unite_index);
		map_prob.erase(map_prob.begin() + unite_index);
		bin_cnts.erase(bin_cnts.begin() + unite_index);
	}

	if (verbose)
		MLOG("Created %d bins for mapping prediction scores to probabilities\n", map_prob.size());
	if (verbose)
		for (size_t i = 0; i < map_prob.size(); ++i)
			MLOG("Range: [%2.4f, %2.4f] => %2.4f | %1.2f%%(%d / %d)\n",
				min_range[i], max_range[i], map_prob[i],
				100 * double(bin_cnts[i]) / y.size(), bin_cnts[i], (int)y.size());
}

void learn_platt_scale(const vector<float> x, const vector<float> &y, const vector<float> &weights,
	int poly_rank, vector<double> &params, int min_bucket_size, float min_score_jump
	, float min_prob_jump, bool fix_pred_order, double control_weight_down_sample, bool verbose) {
	vector<float> min_range, max_range, map_prob;

	learn_binned_probs(x, y, weights, min_bucket_size, min_score_jump, min_prob_jump, fix_pred_order,
		min_range, max_range, map_prob, control_weight_down_sample, verbose);

	vector<float> probs;
	apply_binned_prob(x, min_range, max_range, map_prob, probs);
	//probs is the new Y - lets learn A, B:
	MedLinearModel lm; //B is param[0], A is param[1]

	lm.loss_function = [](const vector<double> &prds, const vector<float> &y, const vector<float> *weights) {
		double res = 0;
		//L2 on 1 / (1 + exp(A*score + B)) vs Y. prds[i] = A*score+B: 1 / (1 + exp(prds))
		for (size_t i = 0; i < y.size(); ++i)
		{
			double conv_prob = 1 / (1 + exp(prds[i]));
			res += (conv_prob - y[i]) * (conv_prob - y[i]);
		}
		res /= y.size();
		res = sqrt(res);
		return res;
	};
	lm.loss_function_step = [](const vector<double> &prds, const vector<float> &y, const vector<double> &params, const vector<float> *weights) {
		double res = 0;
		double reg_coef = 0;
		//L2 on 1 / (1 + exp(A*score + B)) vs Y. prds[i] = A*score+B: 1 / (1 + exp(prds))
		for (size_t i = 0; i < y.size(); ++i)
		{
			double conv_prob = 1 / (1 + exp(prds[i]));
			res += (conv_prob - y[i]) * (conv_prob - y[i]);
		}
		res /= y.size();
		res = sqrt(res);
		//Reg A,B:
		if (reg_coef > 0)
			res += reg_coef * sqrt((params[0] * params[0] + params[1] * params[1]) / 2);
		return res;
	};
	lm.block_num = float(10 * sqrt(poly_rank + 1));
	lm.sample_count = 1000;
	lm.tot_steps = 500000;
	lm.learning_rate = 3 * 1e-1;

	vector<float> poly_preds_params(x.size() * poly_rank);
	for (size_t j = 0; j < poly_rank; ++j)
		for (size_t i = 0; i < x.size(); ++i)
			poly_preds_params[i * poly_rank + j] = (float)pow(x[i], j + 1);

	lm.learn(poly_preds_params, probs, (int)x.size(), poly_rank);
	vector<float> factors(poly_rank), mean_shifts(poly_rank);
	lm.get_normalization(mean_shifts, factors);

	//put normalizations inside params:
	params.resize(poly_rank + 1);
	params[0] = lm.model_params[0];
	for (size_t i = 1; i < params.size(); ++i) {
		params[i] = lm.model_params[i] / factors[i - 1];
		params[0] -= lm.model_params[i] * mean_shifts[i - 1] / factors[i - 1];
	}

	vector<double> converted((int)x.size()), prior_score((int)x.size());
	apply_platt_scale(x, params, converted);
	int tot_pos = 0;
	for (size_t i = 0; i < y.size(); ++i)
		tot_pos += int(y[i] > 0);
	for (size_t i = 0; i < converted.size(); ++i)
		prior_score[i] = double(tot_pos) / y.size();

	double loss_model = _linear_loss_target_rmse(converted, probs, NULL);
	double loss_prior = _linear_loss_target_rmse(prior_score, probs, NULL);

	if (verbose)
		MLOG("Platt Scale prior=%2.5f. loss_model=%2.5f, loss_prior=%2.5f\n",
			double(tot_pos) / y.size(), loss_model, loss_prior);
}

int Calibrator::Learn(const vector<MedSample>& orig_samples, int sample_time_unit) {

	MLOG_D("Learning calibration on %d ids\n", (int)orig_samples.size());

	vector<float> preds, labels, weights;
	switch (calibration_type)
	{
	case CalibrationTypes::probability_time_window:
		learn_time_window(orig_samples, sample_time_unit);
		break;
	case CalibrationTypes::probability_binning:
		collect_preds_labels(orig_samples, preds, labels);
		get_weights(orig_samples, weights_attr_name, weights);
		learn_binned_probs(preds, labels, weights, min_preds_in_bin,
			min_score_res, min_prob_res, fix_pred_order, min_range, max_range, map_prob, control_weight_down_sample, verbose);
		break;
	case CalibrationTypes::probability_platt_scale:
		collect_preds_labels(orig_samples, preds, labels);
		get_weights(orig_samples, weights_attr_name, weights);
		learn_platt_scale(preds, labels, weights, poly_rank, platt_params, min_preds_in_bin,
			min_score_res, min_prob_res, fix_pred_order, control_weight_down_sample, verbose);
		break;
	case CalibrationTypes::probability_isotonic:
		collect_preds_labels(orig_samples, preds, labels);
		get_weights(orig_samples, weights_attr_name, weights);
		learn_isotonic_regression(preds, labels, weights, min_range, max_range, map_prob, n_top_controls, n_bottom_cases, verbose);

		break;
	default:
		MTHROW_AND_ERR("Unsupported implementation for learning calibration method %s\n",
			calibration_method_to_name[calibration_type].c_str());
	}
	return 0;
}

void Calibrator::write_calibration_time_window(const string & calibration_table_file) {
	ofstream of;
	of.open(calibration_table_file, ios::out);
	if (!of) {
		MLOG("can't open file %s for write\n", calibration_table_file.c_str());
		throw exception();
	}
	of << "bin,min_pred,max_pred,cnt_controls,cnt_cases,mean_pred,mean_outcome,cumul_pct,kaplan_meier\n";
	for (calibration_entry& ce : cals)
	{
		of << ce.bin << ","
			<< ce.min_pred << ","
			<< ce.max_pred << ","
			<< ce.cnt_controls << ","
			<< ce.cnt_cases << ","
			<< ce.mean_pred << ","
			<< ce.mean_outcome << ","
			<< ce.cumul_pct << ","
			<< ce.kaplan_meier
			<< endl;
	}

	of.close();
	MLOG("wrote [%d] bins into [%s]\n", cals.size(), calibration_table_file.c_str());
}

void Calibrator::write_calibration_table(const string & calibration_table_file) {
	ofstream of;
	switch (calibration_type)
	{
	case probability_time_window:
		write_calibration_time_window(calibration_table_file);
		break;
	case probability_binning:
	case probability_isotonic:
		of.open(calibration_table_file, ios::out);
		if (!of)
			MTHROW_AND_ERR("can't open file %s for write\n", calibration_table_file.c_str());
		of << "bin,min_pred,max_pred,mean_outcome\n";
		for (size_t i = 0; i < min_range.size(); ++i)
			of << i << "," << min_range[i] << "," << max_range[i] << "," << map_prob[i] << "\n";
		of.close();
		MLOG("wrote [%d] bins into [%s]\n", (int)min_range.size(), calibration_table_file.c_str());
		break;
	case probability_platt_scale:
		of.open(calibration_table_file, ios::out);
		if (!of)
			MTHROW_AND_ERR("can't open file %s for write\n", calibration_table_file.c_str());
		of << "bin,coeff\n";
		for (size_t i = 0; i < platt_params.size(); ++i)
			of << i << "," << platt_params[i] << "\n";
		of.close();
		MLOG("wrote [%d] bins into [%s]\n", (int)platt_params.size(), calibration_table_file.c_str());
		break;
	default:
		MTHROW_AND_ERR("Unsupported implementation for writing calibration method %s\n",
			calibration_method_to_name[calibration_type].c_str());
	}
}

void Calibrator::read_calibration_time_window(const string& fname) {
	ifstream inf(fname);
	if (!inf) {
		MLOG("can't open file %s for read\n", fname.c_str());
		throw exception();
	}
	MLOG("reading from: [%s]\n", fname.c_str());

	string curr_line;
	while (getline(inf, curr_line)) {
		if (curr_line[curr_line.size() - 1] == '\r')
			curr_line.erase(curr_line.size() - 1);

		vector<string> fields;
		split(fields, curr_line, boost::is_any_of(","));
		if (fields[0] == "bin")
			continue; // header

		calibration_entry ce;
		istringstream ls(curr_line);
		char delim;
		ls >> ce.bin >> delim
			>> ce.min_pred >> delim
			>> ce.max_pred >> delim
			>> ce.cnt_controls >> delim
			>> ce.cnt_cases >> delim
			>> ce.mean_pred >> delim
			>> ce.mean_outcome >> delim
			>> ce.cumul_pct >> delim
			>> ce.kaplan_meier;
		cals.push_back(ce);
	}
	MLOG("read %d entries from [%s]\n", cals.size(), fname.c_str());
	inf.close();
}

void Calibrator::read_calibration_table(const string& fname) {
	ifstream f;
	string curr_line;
	vector<string> tokens;
	switch (calibration_type)
	{
	case probability_time_window:
		read_calibration_time_window(fname);
		break;
	case probability_binning:
	case probability_isotonic:
		f.open(fname, ios::in);
		if (!f)
			MTHROW_AND_ERR("can't open file %s for write\n", fname.c_str());
		while (getline(f, curr_line)) {
			boost::split(tokens, curr_line, boost::is_any_of(","));
			if (tokens.size() != 4)
				MTHROW_AND_ERR("Bad format in line:\n%s\nexpected 4 tokens in probability bining\n",
					curr_line.c_str());
			if (tokens[0] == "bin")
				continue; //skip header
			min_range.push_back(stof(tokens[1]));
			max_range.push_back(stof(tokens[2]));
			map_prob.push_back(stof(tokens[3]));
		}
		f.close();
		MLOG("read [%d] bins into [%s]\n", (int)min_range.size(), fname.c_str());
		break;
	case probability_platt_scale:
		f.open(fname, ios::in);
		if (!f)
			MTHROW_AND_ERR("can't open file %s for write\n", fname.c_str());
		while (getline(f, curr_line)) {
			boost::split(tokens, curr_line, boost::is_any_of(","));
			if (tokens.size() != 2)
				MTHROW_AND_ERR("Bad format in line:\n%s\nexpected 2 tokens in probability platt scale\n",
					curr_line.c_str());
			if (tokens[0] == "bin")
				continue; //skip header
			platt_params.push_back(stod(tokens[1]));
		}
		f.close();
		MLOG("read [%d] bins into [%s]\n", (int)platt_params.size(), fname.c_str());
		break;
	default:
		MTHROW_AND_ERR("Unsupported implementation for reading calibration method %s\n",
			calibration_method_to_name[calibration_type].c_str());
	}
}

float Calibrator::calibrate_pred(float pred, int type) const {

	int start = 0;
	for (int i = 0; i < cals.size(); i++) {
		if (pred >= cals[i].min_pred)
			start = i;
	}

	if (type == 0)
		return cals[start].mean_outcome;
	else if (type == 1)
		return cals[start].kaplan_meier;
	else
		return (float)start;
}

calibration_entry Calibrator::calibrate_pred(float pred) {
	if (calibration_type == probability_time_window) {
		int start = 0;
		for (int i = 0; i < cals.size(); i++) {
			if (pred >= cals[i].min_pred)
				start = i;
		}
		return cals[start];
	}
	else {
		MedSample smp;
		smp.prediction = { pred };
		smp.outcome = 0; //doesn't matter
		smp.id = 1; //doesn't matter
		MedIdSamples smp_id(1);
		smp_id.samples.push_back(smp);
		MedSamples samples;
		samples.idSamples = { smp_id };
		Apply(samples);
		calibration_entry res;
		res.mean_outcome = samples.idSamples[0].samples[0].prediction[0];
		res.min_pred = pred;
		res.max_pred = pred;
		res.mean_pred = pred;
		res.bin = 0;
		res.cnt_cases = 0;
		res.cnt_controls = 0;
		res.cnt_cases_no_w = 0;
		res.cnt_controls_no_w = 0;
		res.cumul_pct = 0;
		res.kaplan_meier = 0;
		return res;
	}
}

void Calibrator::dprint(const string &pref) const {
	MLOG("%s :: PP type %d(%s) - of type %s\n", pref.c_str(), processor_type, my_class_name().c_str()
		, calibration_method_to_name.at(calibration_type).c_str());

	switch (calibration_type)
	{
	case probability_time_window:
		for (size_t i = 0; i < cals.size(); ++i)
			MLOG("%s\n", cals[i].str().c_str());
		break;
	case probability_isotonic:
	case probability_binning:
		for (size_t i = 0; i < map_prob.size(); ++i)
			MLOG("Range: [%2.4f, %2.4f] => %2.4f\n",
				min_range[i], max_range[i], map_prob[i]);
		break;
	case probability_platt_scale:
		MLOG("Platt Scale params for poly_rank=%d:\n", poly_rank);
		for (size_t i = 0; i < platt_params.size(); ++i)
			MLOG("i=0 :: %2.3f\n", platt_params[i]);
		break;
	default:
		MWARN("Unsupported print type %d - Maybe memory error\n", calibration_type);
		break;
	}

}

