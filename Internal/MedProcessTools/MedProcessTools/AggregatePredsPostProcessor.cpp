#include "AggregatePredsPostProcessor.h"
#include <boost/algorithm/string.hpp>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MED_MODEL
#define LOCAL_LEVEL LOG_DEF_LEVEL

//defaults:
AggregatePredsPostProcessor::AggregatePredsPostProcessor() {
	feature_processor_type = "";
	feature_processor_args = "";
	use_median = false;
	resample_cnt = 50;
	batch_size = 10000;
	force_cancel_imputations = false;
	print_missing_cnt = false;
	processor_type = FTR_POSTPROCESS_AGGREGATE_PREDS;
}

int AggregatePredsPostProcessor::init(map<string, string> &mapper) {
	for (const auto &it : mapper)
	{
		if (it.first == "feature_processor_type")
			feature_processor_type = it.second;
		else if (it.first == "feature_processor_args")
			feature_processor_args = it.second;
		else if (it.first == "use_median")
			use_median = med_stoi(it.second) > 0;
		else if (it.first == "resample_cnt")
			resample_cnt = med_stoi(it.second);
		else if (it.first == "batch_size")
			batch_size = med_stoi(it.second);
		else if (it.first == "force_cancel_imputations")
			force_cancel_imputations = med_stoi(it.second) > 0;
		else if (it.first == "print_missing_cnt")
			print_missing_cnt = med_stoi(it.second) > 0;
		else if (it.first == "pp_type") {} //ignore
		else
			MTHROW_AND_ERR("Error AggregatePredsPostProcessor::init - unknown argument %s\n",
				it.first.c_str());
	}

	if (feature_processor_type.empty())
		MTHROW_AND_ERR("Error AggregatePredsPostProcessor::init - must provide feature_processor_type\n");
	if (resample_cnt <= 0)
		MTHROW_AND_ERR("Error AggregatePredsPostProcessor::init - resample_cnt > 0\n");
	if (batch_size <= 0)
		MTHROW_AND_ERR("Error AggregatePredsPostProcessor::init - batch_size > 0\n");

	if (feature_processor == NULL && !boost::starts_with(feature_processor_type, "MODEL::")) {
		feature_processor = FeatureProcessor::make_processor(feature_processor_type, feature_processor_args);
	}

	return 0;
}

void AggregatePredsPostProcessor::get_input_fields(vector<Effected_Field> &fields) const {
	Effected_Field f(Field_Type::FEATURE, "");
	//empty value means all features, not a specific one

	fields.push_back(f);
}
void AggregatePredsPostProcessor::get_output_fields(vector<Effected_Field> &fields) const {
	Effected_Field f(Field_Type::PREDICTION, "0");
	fields.push_back(f);

	//Add additional attributes:
	fields.push_back(Effected_Field (Field_Type::NUMERIC_ATTRIBUTE, "pred.std"));
	fields.push_back(Effected_Field (Field_Type::NUMERIC_ATTRIBUTE, "pred.ci_lower"));
	fields.push_back(Effected_Field (Field_Type::NUMERIC_ATTRIBUTE, "pred.ci_upper"));
	fields.push_back(Effected_Field (Field_Type::NUMERIC_ATTRIBUTE, "pred.mean"));
	fields.push_back(Effected_Field (Field_Type::NUMERIC_ATTRIBUTE, "pred.median"));
}

void AggregatePredsPostProcessor::init_post_processor(MedModel& model)
{
	p_model = &model;
	model_predictor = model.predictor;
	before_processors.clear();
	after_processors.clear();
	if (boost::starts_with(feature_processor_type, "MODEL::")) {
		feature_processor = NULL;
		//store feature processor if has or need from model.
		string feature_type = feature_processor_type.substr(7);
		vector<FeatureProcessor *> fps_flat;
		for (size_t i = 0; i < model.feature_processors.size(); ++i)
		{
			if (model.feature_processors[i]->processor_type == FTR_PROCESS_MULTI) {
				vector<FeatureProcessor *> &to_add = static_cast<MultiFeatureProcessor *>(model.feature_processors[i])->processors;
				fps_flat.insert(fps_flat.end(), to_add.begin(), to_add.end());
			}
			else
				fps_flat.push_back(model.feature_processors[i]);
		}

		for (FeatureProcessor *f : fps_flat)
		{
			if (f->my_class_name() == feature_type) {
				if (feature_processor != NULL)
					MTHROW_AND_ERR("Error AggregatePredsPostProcessor::init_post_processor - found multiple feature processors of type %s\n",
						feature_type.c_str());
				feature_processor = f;
			}
			else {
				if (feature_processor != NULL)
					after_processors.push_back(f);
				else
					before_processors.push_back(f);
			}
		}
		if (feature_processor == NULL)
			MTHROW_AND_ERR("Error AggregatePredsPostProcessor::init_post_processor - can't find feature processors of type %s\n",
				feature_type.c_str());
		if (!before_processors.empty())
			MWARN("WARN:: AggregatePredsPostProcessor :: found %zu processors before\n", before_processors.size());
		if (!after_processors.empty())
			MLOG("INFO:: AggregatePredsPostProcessor :: found %zu processors after\n", after_processors.size());
	}
	else if (force_cancel_imputations) {
		//find imputer and store all what happens after him:
		vector<bool> processors_tp(FTR_PROCESS_LAST);
		processors_tp[FTR_PROCESS_IMPUTER] = true;
		processors_tp[FTR_PROCESS_ITERATIVE_IMPUTER] = true;
		processors_tp[FTR_PROCESS_PREDICTOR_IMPUTER] = true;

		vector<FeatureProcessor *> fps_flat;
		for (size_t i = 0; i < model.feature_processors.size(); ++i)
		{
			if (model.feature_processors[i]->processor_type == FTR_PROCESS_MULTI) {
				vector<FeatureProcessor *> &to_add = static_cast<MultiFeatureProcessor *>(model.feature_processors[i])->processors;
				fps_flat.insert(fps_flat.end(), to_add.begin(), to_add.end());
			}
			else
				fps_flat.push_back(model.feature_processors[i]);
		}

		bool found = false;
		for (FeatureProcessor *f : fps_flat)
		{
			if (found) {
				if (!processors_tp[f->processor_type])
					after_processors.push_back(f);
			}
			else {
				if (processors_tp[f->processor_type])
					found = true;
				else
					before_processors.push_back(f);
			}
		}

		if (!before_processors.empty())
			MLOG("INFO:: AggregatePredsPostProcessor :: found %zu processors before\n", before_processors.size());
		if (!after_processors.empty())
			MLOG("INFO:: AggregatePredsPostProcessor :: found %zu processors after\n", after_processors.size());
	}
	else
	{
		before_processors.insert(before_processors.end(),
			model.feature_processors.begin(), model.feature_processors.end());
		if (!before_processors.empty())
			MLOG("INFO:: AggregatePredsPostProcessor :: found %zu processors before (fp added to end)\n", before_processors.size());
	}
}

void AggregatePredsPostProcessor::generate_matrix_till_feature_process(const MedFeatures &input_mat, MedFeatures &res) const {
	//get samples:
	MedSamples input_smps;
	input_smps.import_from_sample_vec(input_mat.samples);
	//erase attriubtes:
	for (size_t i = 0; i < input_smps.idSamples.size(); ++i)
		for (size_t j = 0; j < input_smps.idSamples[i].samples.size(); ++j) {
			input_smps.idSamples[i].samples[j].attributes.clear();
			input_smps.idSamples[i].samples[j].str_attributes.clear();
		}
	//apply p_model with p_rep till feature processors:
	//effect on features - disable it.
	MLOG("Generating matrix again, without feature processors (improve the process later)\n");
	MedModel copy_mdl;
	unsigned char *blob = new unsigned char[p_model->get_size()];
	p_model->serialize(blob);
	copy_mdl.deserialize(blob);
	delete[] blob;
	//MedFeatures copy_mat = input_mat;
	copy_mdl.apply(*p_model->p_rep, input_smps, MedModelStage::MED_MDL_LEARN_REP_PROCESSORS,
		MedModelStage::MED_MDL_APPLY_FTR_GENERATORS);
	res = std::move(copy_mdl.features);
	//apply before 
	for (FeatureProcessor *fp : before_processors)
		fp->apply(res, false);

}

void print_msn(const MedFeatures &f, float missing_value, const string &prefix) {
	if (f.data.empty())
		return;
	vector<string> names(f.data.size());
	vector<int> counts(f.data.size());
	int tot_count = (int)f.data.begin()->second.size();
	int feat_idx = 0;
	for (const auto &it : f.data)
	{
		const string &nm = it.first;
		names[feat_idx] = nm;
		for (float val : it.second)
			counts[feat_idx] += int(val == missing_value);
		++feat_idx;
	}

	//print names,counts,tot_count
	MLOG("%s :: Prints missing values count for %zu features (non missing are skipped):\n",
		prefix.c_str(), names.size());
	for (size_t i = 0; i < names.size(); ++i)
		if (counts[i] > 0)
			MLOG("%s :: %s :: %d / %d :: %2.2f%%\n", prefix.c_str(), names[i].c_str(), counts[i], tot_count,
				100 * double(counts[i]) / tot_count);

}

void AggregatePredsPostProcessor::Learn(const MedFeatures &train_mat) {
	if (!boost::starts_with(feature_processor_type, "MODEL::")) {
		unordered_set<int> empt;
		if (!p_model->feature_processors.empty() && force_cancel_imputations) { //need to cancel imputations
			MedFeatures train_mat_for_processor;
			generate_matrix_till_feature_process(train_mat, train_mat_for_processor);
			if (print_missing_cnt)
				print_msn(train_mat, MED_MAT_MISSING_VALUE, "Learn");
			feature_processor->Learn(train_mat_for_processor, empt);
		}
		else { //feature processors happens in the end - no need to do something
			MedFeatures mat = train_mat;
			feature_processor->Learn(mat, empt);
		}
	}
}

void clear_map(map<string, vector<float>> &data) {
	for (auto &it : data)
		data[it.first].clear();
}

void AggregatePredsPostProcessor::Apply(MedFeatures &matrix) {
	//Apply plan, do in batches:
	//1. resample input - apply feature_processor multiple times for each sample (if imputer and using existing in model. will get matrix without feature processors/ rerun model again)
	//2. predict with model_predictor
	//3. aggregate predictions - fetch mean,median,std,ci - the rest in attributes
	vector<float> prctile_list = { (float)0.05, (float)0.5, (float)0.95 };
	MedFeatures fixed_mat;
	MedFeatures *p_matrix = &matrix;
	if (!p_model->feature_processors.empty() && force_cancel_imputations) { //applied till current processor
		generate_matrix_till_feature_process(matrix, fixed_mat);
		p_matrix = &fixed_mat;
	}
	if (print_missing_cnt)
		print_msn(*p_matrix, MED_MAT_MISSING_VALUE, "Apply");

	//1. resample input - apply feature_processor multiple times for each sample	
	MedFeatures batch;
	batch.attributes = p_matrix->attributes;
	batch.tags = p_matrix->tags;
	batch.time_unit = p_matrix->time_unit;
	batch.medf_missing_value = p_matrix->medf_missing_value;
	batch.samples.resize(static_cast<size_t>(batch_size) * static_cast<size_t>(resample_cnt));
	for (int i = 0; i < batch.samples.size(); ++i) {
		batch.samples[i].id = int(i / resample_cnt);
		batch.samples[i].time = i % resample_cnt;
	}
	batch.init_pid_pos_len();
	//data, samples
	int i = 0;
	vector<unordered_map<string, float>> samples_res(p_matrix->samples.size());
	MedProgress progrees("AggregatePredsPostProcessor::Apply", int(p_matrix->samples.size() / batch_size) + 1, 30, 10);
	while (i < p_matrix->samples.size())
	{
		//prepate batch
		int start_idx_i = i;
		int curr_sz = 0;
		MedFeatures apply_batch = batch;
		for (auto &it : p_matrix->data)
			apply_batch.data[it.first].resize(static_cast<size_t>(resample_cnt) * static_cast<size_t>(batch_size));
		while (curr_sz < batch_size && i < p_matrix->samples.size()) {
			//add data from matrix
			for (auto &it : p_matrix->data) {
				for (size_t j = 0; j < resample_cnt; ++j)
					apply_batch.data[it.first][static_cast<size_t>(curr_sz) * static_cast<size_t>(resample_cnt) + j] = it.second[i];
			}
			++curr_sz;
			++i;
		}
		//by curr_sz:
		if (curr_sz < batch_size) {//last batch - remove last samples
			apply_batch.samples.resize(static_cast<size_t>(curr_sz) * static_cast<size_t>(resample_cnt));
			apply_batch.init_pid_pos_len();
			for (auto &it : p_matrix->data)
				apply_batch.data[it.first].resize(static_cast<size_t>(resample_cnt) * static_cast<size_t>(curr_sz));
		}
		//apply feature processor on all duplicated batch:
		feature_processor->apply(apply_batch);
		//Apply after batch
		for (FeatureProcessor *fp : after_processors)
			fp->apply(apply_batch, false);
		//apply batch with MedPredictor:
		model_predictor->predict(apply_batch);
		//apply calibrator if has?
		vector<bool> apply_pp_mask(PostProcessorTypes::FTR_POSTPROCESS_LAST);
		apply_pp_mask[FTR_POSTPROCESS_CALIBRATOR] = true;
		apply_pp_mask[FTR_POSTPROCESS_ADJUST] = true;
		apply_pp_mask[FTR_POSTPROCESS_FAIRNESS] = true;
		for (PostProcessor *pp : p_model->post_processors)
		{
			if (!apply_pp_mask[pp->processor_type])
				continue;
			pp->Apply(apply_batch);
		}
		//collect preds from samples: each row was duplicated resample_cnt times
		vector<vector<float>> collected_preds(curr_sz); //for each sample
		for (size_t j = 0; j < curr_sz; ++j)
		{
			vector<float> &v = collected_preds[j];
			v.resize(resample_cnt);
			//add preds:
			for (size_t k = 0; k < resample_cnt; ++k)
				v[k] = apply_batch.samples[j*resample_cnt + k].prediction[0];
		}
		//aggregate results using collected_preds for each original pred:

		for (size_t j = 0; j < curr_sz; ++j) {
			unordered_map<string, float> &dict = samples_res[start_idx_i + j];
			vector<float> &dt = collected_preds[j];
			float mean, std;
			medial::stats::get_mean_and_std_without_cleaning(dt, mean, std);
			dict["Mean"] = mean;
			dict["Std"] = std;
			vector<float> res;
			medial::stats::get_percentiles(dt, prctile_list, res);
			dict["CI_Lower"] = res[0];
			dict["Median"] = res[1];
			dict["CI_Upper"] = res[2];
		}

		progrees.update();
	}

	//store in final matrix:
	for (size_t j = 0; j < matrix.samples.size(); ++j)
	{
		const unordered_map<string, float> &dict = samples_res[j];
		MedSample &s = matrix.samples[j];

		/*if (use_median)
			s.prediction[0] = dict.at("Median");
		else
			s.prediction[0] = dict.at("Mean");*/

			//store in attributes the rest:
		s.attributes["pred.std"] = dict.at("Std");
		s.attributes["pred.ci_lower"] = dict.at("CI_Lower");
		s.attributes["pred.ci_upper"] = dict.at("CI_Upper");
		s.attributes["pred.mean"] = dict.at("Mean");
		s.attributes["pred.median"] = dict.at("Median");
	}

}

AggregatePredsPostProcessor::~AggregatePredsPostProcessor() {
	if (boost::starts_with(feature_processor_type, "MODEL::")) {
		if (feature_processor != NULL) {
			delete feature_processor;
			feature_processor = NULL;
		}
	}
}

void AggregatePredsPostProcessor::dprint(const string &pref) const {
	MLOG("%s using %s preidctor, feature_processor of %s\n", pref.c_str(), model_predictor->my_class_name().c_str(),
		feature_processor->my_class_name().c_str());
}