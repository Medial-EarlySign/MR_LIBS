#define _CRT_SECURE_NO_WARNINGS

#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>

#include "FeatureGenerator.h"
#include "SmokingGenerator.h"
#include "KpSmokingGenerator.h"
#include "UnifiedSmokingGenerator.h"
#include "DrugIntakeGenerator.h"
#include "AlcoholGenerator.h"
#include "EmbeddingGenerator.h"
#include "FeatureGenExtractTable.h"
#include "DiabetesFinderGenerator.h"

#include <MedProcessTools/MedProcessTools/MedModel.h>
#include <MedUtils/MedUtils/MedGenUtils.h>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_FTRGNRTR
#define LOCAL_LEVEL	LOG_DEF_LEVEL

//=======================================================================================
// FeatGenerator
//=======================================================================================
// Generator types
FeatureGeneratorTypes ftr_generator_name_to_type(const string& generator_name) {

	if (generator_name == "basic")
		return FTR_GEN_BASIC;
	else if (generator_name == "age")
		return FTR_GEN_AGE;
	else if (generator_name == "gender")
		return FTR_GEN_GENDER;
	else if (generator_name == "singleton")
		return FTR_GEN_SINGLETON;
	else if (generator_name == "binnedLmEstimates" || generator_name == "binnedLm" || generator_name == "binnedLM")
		return FTR_GEN_BINNED_LM;
	else if (generator_name == "smoking")
		return FTR_GEN_SMOKING;
	else if (generator_name == "kp_smoking")
		return FTR_GEN_KP_SMOKING;
	else if (generator_name == "unified_smoking")
		return FTR_GEN_UNIFIED_SMOKING;
	else if (generator_name == "alcohol")
		return FTR_GEN_ALCOHOL;
	else if (generator_name == "range")
		return FTR_GEN_RANGE;
	else if (generator_name == "drugIntake")
		return FTR_GEN_DRG_INTAKE;
	else if (generator_name == "model")
		return FTR_GEN_MODEL;
	else if (generator_name == "time")
		return FTR_GEN_TIME;
	else if (generator_name == "attribute")
		return FTR_GEN_ATTR;
	else if (generator_name == "category_depend")
		return FTR_GEN_CATEGORY_DEPEND;
	else if (generator_name == "embedding")
		return FTR_GEN_EMBEDDING;
	else if (generator_name == "extract_tbl")
		return FTR_GEN_EXTRACT_TBL;
	else if (generator_name == "diabetes_finder")
		return FTR_GEN_DIABETES_FINDER;
	else MTHROW_AND_ERR("unknown generator name [%s]", generator_name.c_str());
}

// Prepare for feature Generation
//.......................................................................................
void FeatureGenerator::prepare(MedFeatures &features, MedPidRepository& rep, MedSamples& samples) {

	if (!iGenerateWeights) {
		//MLOG("FeatureGenerator::init _features\n");
		if (names.size() == 0)
			set_names();

		// Attributes and data
		for (auto& name : names) {
			features.attributes[name].normalized = false;
			features.data[name].resize(0, 0);
		}

		// Tags
		for (auto& name : names) {
			for (string& tag : tags)
				features.tags[name].insert(tag);
		}
	}
	else
		features.weights.resize(0, 0);
}

// Get pointers to data vectors
//.......................................................................................
void FeatureGenerator::get_p_data(MedFeatures &features, vector<float *> &_p_data) {

	p_data.clear();
	if (iGenerateWeights)
		_p_data.push_back(&(features.weights[0]));
	else {
		for (string& name : names)
			_p_data.push_back(&(features.data[name][0]));
	}

	return;
}

//.......................................................................................
FeatureGenerator *FeatureGenerator::create_generator(string &params)
{
	string fg_type;
	get_single_val_from_init_string(params, "fg_type", fg_type);
	return (make_generator(fg_type, params));
}

// Initialization
//.......................................................................................
FeatureGenerator *FeatureGenerator::make_generator(string generator_name) {

	return make_generator(ftr_generator_name_to_type(generator_name));
}

//.......................................................................................
FeatureGenerator *FeatureGenerator::make_generator(string generator_name, string init_string) {

	//MLOG("making generator %s , %s\n", generator_name.c_str(), init_string.c_str());
	return make_generator(ftr_generator_name_to_type(generator_name), init_string);
}

//.......................................................................................
void *FeatureGenerator::new_polymorphic(string dname) {

	CONDITIONAL_NEW_CLASS(dname, BasicFeatGenerator);
	CONDITIONAL_NEW_CLASS(dname, AgeGenerator);
	CONDITIONAL_NEW_CLASS(dname, GenderGenerator);
	CONDITIONAL_NEW_CLASS(dname, SingletonGenerator);
	CONDITIONAL_NEW_CLASS(dname, BinnedLmEstimates);
	CONDITIONAL_NEW_CLASS(dname, SmokingGenerator);
	CONDITIONAL_NEW_CLASS(dname, KpSmokingGenerator);
	CONDITIONAL_NEW_CLASS(dname, UnifiedSmokingGenerator);
	CONDITIONAL_NEW_CLASS(dname, AlcoholGenerator);
	CONDITIONAL_NEW_CLASS(dname, RangeFeatGenerator);
	CONDITIONAL_NEW_CLASS(dname, DrugIntakeGenerator);
	CONDITIONAL_NEW_CLASS(dname, ModelFeatGenerator);
	CONDITIONAL_NEW_CLASS(dname, TimeFeatGenerator);
	CONDITIONAL_NEW_CLASS(dname, AttrFeatGenerator);
	CONDITIONAL_NEW_CLASS(dname, CategoryDependencyGenerator);
	CONDITIONAL_NEW_CLASS(dname, EmbeddingGenerator);
	CONDITIONAL_NEW_CLASS(dname, FeatureGenExtractTable);
	CONDITIONAL_NEW_CLASS(dname, DiabetesFinderGenerator);
	MWARN("Warning in FeatureGenerator::new_polymorphic - Unsupported class %s\n", dname.c_str());
	return NULL;
}


//.......................................................................................
FeatureGenerator *FeatureGenerator::make_generator(FeatureGeneratorTypes generator_type) {

	if (generator_type == FTR_GEN_BASIC)
		return new BasicFeatGenerator;
	else if (generator_type == FTR_GEN_AGE)
		return new AgeGenerator;
	else if (generator_type == FTR_GEN_GENDER)
		return new GenderGenerator;
	else if (generator_type == FTR_GEN_SINGLETON)
		return new SingletonGenerator;
	else if (generator_type == FTR_GEN_BINNED_LM)
		return new BinnedLmEstimates;
	else if (generator_type == FTR_GEN_SMOKING)
		return new SmokingGenerator;
	else if (generator_type == FTR_GEN_KP_SMOKING)
		return new KpSmokingGenerator;
	else if (generator_type == FTR_GEN_UNIFIED_SMOKING)
		return new UnifiedSmokingGenerator;
	else if (generator_type == FTR_GEN_ALCOHOL)
		return new AlcoholGenerator;
	else if (generator_type == FTR_GEN_RANGE)
		return new RangeFeatGenerator;
	else if (generator_type == FTR_GEN_DRG_INTAKE)
		return new DrugIntakeGenerator;
	else if (generator_type == FTR_GEN_MODEL)
		return new ModelFeatGenerator;
	else if (generator_type == FTR_GEN_TIME)
		return new TimeFeatGenerator;
	else if (generator_type == FTR_GEN_ATTR)
		return new AttrFeatGenerator;
	else if (generator_type == FTR_GEN_CATEGORY_DEPEND)
		return new CategoryDependencyGenerator;
	else if (generator_type == FTR_GEN_EMBEDDING)
		return new EmbeddingGenerator;
	else if (generator_type == FTR_GEN_EXTRACT_TBL)
		return new FeatureGenExtractTable;
	else if (generator_type == FTR_GEN_DIABETES_FINDER)
		return new DiabetesFinderGenerator;
	else MTHROW_AND_ERR("dont know how to make_generator for [%s]", to_string(generator_type).c_str());
}

//.......................................................................................
FeatureGenerator * FeatureGenerator::make_generator(FeatureGeneratorTypes generator_type, string init_string) {

	//MLOG("making generator %d , %s\n", (int)generator_type, init_string.c_str());
	FeatureGenerator *newFtrGenerator = make_generator(generator_type);
	if (newFtrGenerator->init_from_string(init_string) < 0)
		MTHROW_AND_ERR("Cannot init FeatureGenerator of type %d with init string \'%s\'\n", generator_type, init_string.c_str());
	return newFtrGenerator;
}

//.......................................................................................
// Add at end of feature vector
int FeatureGenerator::generate(PidDynamicRec& in_rep, MedFeatures& features) {
	//MLOG("gen [%s]\n", this->names[0].c_str());
	return _generate(in_rep, features, features.get_pid_pos(in_rep.pid), features.get_pid_len(in_rep.pid));
}

//.......................................................................................
// Add uncleaned data at end of feature vector
int FeatureGenerator::generate(MedPidRepository& rep, int id, MedFeatures& features) {

	int samples_size = (int)features.samples.size();
	int data_size;

	if (iGenerateWeights) {
		data_size = (int)features.weights.size();

		if (data_size > samples_size) {
			string name = "";
			if (names.size() > 0)
				name = names[0];
			MERR("Data (%d) is longer than Samples (%d) for %s. Cannot generate weights \n", data_size, samples_size, name.c_str());
			return -1;
		}
		features.weights.resize(samples_size);
		p_data[0] = &(features.weights[0]);
	}
	else {
		data_size = (int)features.data[names[0]].size();

		if (data_size > samples_size) {
			MERR("Data (%d) is longer than Samples (%d) for %s. Cannot generate feature \n", data_size, samples_size, names[0].c_str());
			return -1;
		}
		for (string& name : names)
			features.data[name].resize(samples_size);

		get_p_data(features);
	}
	return generate(rep, id, features, data_size, (int)(samples_size - data_size));
}

//.......................................................................................
// Add uncleaned data
int FeatureGenerator::generate(MedPidRepository& rep, int id, MedFeatures& features, int index, int num) {

	PidDynamicRec rec;
	rec.prealloc(DYNAMIC_REC_SIZE);

	rec.init_from_rep(std::addressof(rep), id, req_signal_ids, num);

	return _generate(rec, features, index, num);
}

//.......................................................................................
// Init 
int FeatureGenerator::init(map<string, string>& mapper) {

	for (auto entry : mapper) {
		string field = entry.first;
		if (field == "tags")
			boost::split(tags, entry.second, boost::is_any_of(","));
		else if (field == "weights_generator")
			iGenerateWeights = med_stoi(entry.second);
		else if (field != "fg_type")
			MLOG("Unknown parameter \'%s\' for FeatureGenerator\n", field.c_str());
	}
	set_names();
	return 0;
}

// (De)Serialize
//.......................................................................................
size_t FeatureGenerator::get_generator_size() {
	return sizeof(generator_type) + get_size();
}

//.......................................................................................
size_t FeatureGenerator::generator_serialize(unsigned char *blob) {

	size_t ptr = 0;
	memcpy(blob + ptr, &generator_type, sizeof(FeatureGeneratorTypes)); ptr += sizeof(FeatureGeneratorTypes);
	ptr += serialize(blob + ptr);

	return ptr;
}

// Set required signal ids
//.......................................................................................
void FeatureGenerator::set_required_signal_ids(MedDictionarySections& dict) {
	/*if (req_signals.empty()) {
		dprint("", 1);
		MTHROW_AND_ERR("FeatureGenerator::set_required_signal_ids got empty req_signals\n");
	}*/
	req_signal_ids.resize(req_signals.size());

	for (unsigned int i = 0; i < req_signals.size(); i++)
		req_signal_ids[i] = dict.id(req_signals[i]);
}


// Get Required Signals
//.......................................................................................
void FeatureGenerator::get_required_signal_names(unordered_set<string>& signalNames) {
	for (auto sig : req_signals)
		signalNames.insert(sig);
}

//.......................................................................................
void FeatureGenerator::get_required_signal_ids(unordered_set<int>& signalIds) {
	for (auto sig : req_signal_ids)
		signalIds.insert(sig);
}


//.......................................................................................
// Filter generated features according to a set. return number of valid features (does not affect single-feature genertors, just returns 1/0 if feature name in set)
int FeatureGenerator::filter_features(unordered_set<string>& validFeatures) {

	vector<string> names_new;
	names_new.reserve(names.size());
	for (int i = 0; i < names.size(); i++)
		if (validFeatures.find(names[i]) != validFeatures.end())
			names_new.push_back(names[i]);

	names = std::move(names_new);

	return ((int)names.size());
}

inline bool isInteger(const std::string & s)
{
	if (s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false;

	char * p;
	strtol(s.c_str(), &p, 10);

	return (*p == 0);
}


//.......................................................................................
void FeatureGenerator::dprint(const string &pref, int fg_flag)
{
	if (fg_flag > 0) {
		MLOG("%s :: FG type %d(%s) : serial_id %d : ", pref.c_str(), generator_type, my_class_name().c_str(), serial_id);
		MLOG("names(%d) : ", names.size());
		if (fg_flag > 1) for (auto &name : names) MLOG("%s,", name.c_str());
		MLOG(" tags(%d) : ", tags.size());
		if (fg_flag > 1) for (auto &t : tags) MLOG("%s,", t.c_str());
		MLOG(" req_signals(%d) : ", req_signals.size());
		if (fg_flag > 1) for (auto &rsig : req_signals) MLOG("%s,", rsig.c_str());
		MLOG("\n");
	}
}

//=======================================================================================
// Single signal features that do not require learning(e.g. last hemoglobin)
//=======================================================================================
// 
BasicFeatureTypes BasicFeatGenerator::name_to_type(const string &name)
{

	if (name == "last")				return FTR_LAST_VALUE;
	if (name == "last_nth")				return FTR_LAST_NTH_VALUE;
	if (name == "first")			return FTR_FIRST_VALUE;
	if (name == "last2")			return FTR_LAST2_VALUE;
	if (name == "avg")				return FTR_AVG_VALUE;
	if (name == "max")				return FTR_MAX_VALUE;
	if (name == "min")				return FTR_MIN_VALUE;
	if (name == "std")				return FTR_STD_VALUE;
	if (name == "sum")				return FTR_SUM_VALUE;
	if (name == "last_delta")		return FTR_LAST_DELTA_VALUE;
	if (name == "last_time")		return FTR_LAST_DAYS;
	if (name == "last2_time")		return FTR_LAST2_DAYS;
	if (name == "slope")			return FTR_SLOPE_VALUE;
	if (name == "win_delta")				return FTR_WIN_DELTA_VALUE;
	if (name == "category_set")				return FTR_CATEGORY_SET;
	if (name == "category_set_last_nth")				return FTR_CATEGORY_SET_LAST_NTH;
	if (name == "category_set_count")		return FTR_CATEGORY_SET_COUNT;
	if (name == "category_set_sum")			return FTR_CATEGORY_SET_SUM;
	if (name == "nsamples")			return FTR_NSAMPLES;
	if (name == "exists")			return FTR_EXISTS;
	if (name == "range_width")			return FTR_RANGE_WIDTH;
	if (name == "max_diff")			return FTR_MAX_DIFF;
	if (name == "first_time")		return FTR_FIRST_DAYS;
	if (name == "category_set_first")				return FTR_CATEGORY_SET_FIRST;
	if (name == "category_set_first_time")				return FTR_CATEGORY_SET_FIRST_TIME;
	if (name == "time_since_last_change")			return FTR_TIME_SINCE_LAST_CHANGE;


	if (isInteger(name))
		return (BasicFeatureTypes)med_stoi(name);
	else
		MTHROW_AND_ERR("unknown name [%s]\n", name.c_str());
}

TimeRangeTypes BasicFeatGenerator::time_range_name_to_type(const string &name)
{

	if (name == "current")			return TIME_RANGE_CURRENT;
	if (name == "before")			return TIME_RANGE_BEFORE;

	if (isInteger(name))
		return (TimeRangeTypes)med_stoi(name);
	else
		MTHROW_AND_ERR("unknown name [%s]\n", name.c_str());
}

string BasicFeatGenerator::time_range_type_to_name(TimeRangeTypes type)
{
	if (type == TIME_RANGE_CURRENT) return "current";
	if (type == TIME_RANGE_BEFORE) return "before";

	MTHROW_AND_ERR("unknown type [%d]\n", (int)type);
}

//.......................................................................................
void BasicFeatGenerator::set_names() {

	names.clear();
	string name = signalName + ".";
	if (!rename_signal.empty())
		name = rename_signal + ".";
	//string name = signalName + ".";
	string set_names = in_set_name;
	if (set_names == "" && this->sets.size() > 0)
		set_names = boost::algorithm::join(this->sets, "_");
	switch (type) {
	case FTR_LAST_VALUE:	name += "last"; break;
	case FTR_LAST_NTH_VALUE:	name += "last_" + to_string(N_th); break;
	case FTR_FIRST_VALUE:	name += "first"; break;
	case FTR_LAST2_VALUE:	name += "last2"; break;
	case FTR_AVG_VALUE:		name += "avg"; break;
	case FTR_MAX_VALUE:		name += "max"; break;
	case FTR_MIN_VALUE:		name += "min"; break;
	case FTR_STD_VALUE:		name += "std"; break;
	case FTR_SUM_VALUE:		name += "sum"; break;
	case FTR_LAST_DELTA_VALUE:		name += "last_delta"; break;
	case FTR_LAST_DAYS:				name += "last_time"; break;
	case FTR_LAST2_DAYS:			name += "last2_time"; break;
	case FTR_SLOPE_VALUE:			name += "slope"; break;
	case FTR_WIN_DELTA_VALUE:		name += "win_delta"; break;
	case FTR_CATEGORY_SET:			name += "category_set_" + set_names; break;
	case FTR_CATEGORY_SET_LAST_NTH:			name += "category_set_last_nth_" + to_string(N_th) + "_" + set_names; break;
	case FTR_CATEGORY_SET_COUNT:	name += "category_set_count_" + set_names; break;
	case FTR_CATEGORY_SET_SUM:		name += "category_set_sum_" + set_names; break;
	case FTR_CATEGORY_SET_FIRST:	name += "category_set_first_" + set_names; break;
	case FTR_CATEGORY_SET_FIRST_TIME:	name += "category_set_first_time_" + set_names; break;
	case FTR_NSAMPLES:			name += "nsamples"; break;
	case FTR_EXISTS:			name += "exists"; break;
	case FTR_RANGE_WIDTH:			name += "range_width"; break;
	case FTR_MAX_DIFF:			name += "max_diff"; break;
	case FTR_FIRST_DAYS:		name += "first_time"; break;
	case FTR_TIME_SINCE_LAST_CHANGE: name += "last_time_since_change"; break;

	default: {
		name += "ERROR";
		MTHROW_AND_ERR("Got a wrong type in basic feature generator %d\n", type);
	}
	}

	name += ".win_" + std::to_string(win_from) + "_" + std::to_string(win_to);
	if (type == FTR_WIN_DELTA_VALUE)
		name += "_" + std::to_string(d_win_from) + "_" + std::to_string(d_win_to);
	if (time_channel != 0 || val_channel != 0 || full_name)
		name += ".t" + std::to_string(time_channel) + "v" + std::to_string(val_channel);
	if (timeRangeSignalName != "")
		name += ".time_range_" + timeRangeSignalName + "_" + time_range_type_to_name(timeRangeType);
	if (min_value != -FLT_MAX)
		name += ".min_value_" + to_string(min_value);
	if (max_value != FLT_MAX)
		name += ".max_value_" + to_string(max_value);
	names.push_back("FTR_" + int_to_string_digits(serial_id, 6) + "." + name);
	// add the undecorated feature name as a tag, so we can later remove/select it with TagFeatureSelector
	tags.push_back(name);
	//MLOG("Created %s\n", name.c_str());

	//time_unit_sig = rep.sigs.Sid2Info[sid].time_unit; !! this is an issue to SOLVE !!
}

// Init
//.......................................................................................
void BasicFeatGenerator::init_defaults() {
	generator_type = FTR_GEN_BASIC;
	signalId = -1;
	time_unit_sig = MedTime::Undefined;
	time_unit_win = global_default_windows_time_unit;
	string _signalName = "";
	bound_outcomeTime = false;
	timeRangeSignalId = -1;
	//set(_signalName, FTR_LAST, 0, 360000);
	N_th = 0;
};

// Generate
//.......................................................................................
int BasicFeatGenerator::_generate(PidDynamicRec& rec, MedFeatures& features, int index, int num, vector<float *> &_p_data) {

	if (time_unit_sig == MedTime::Undefined)	time_unit_sig = rec.my_base_rep->sigs.Sid2Info[signalId].time_unit;
	if (timeRangeSignalName != "" && time_unit_range_sig == MedTime::Undefined)
		time_unit_range_sig = rec.my_base_rep->sigs.Sid2Info[timeRangeSignalId].time_unit;



	float *p_feat = _p_data[0] + index;
	MedSample *p_samples = &(features.samples[index]);

	for (int i = 0; i < num; i++) {
		p_feat[i] = get_value(rec, i, med_time_converter.convert_times(features.time_unit, time_unit_win, p_samples[i].time),
			med_time_converter.convert_times(features.time_unit, time_unit_sig, p_samples[i].outcomeTime));
		if (apply_categ_map && (p_feat[i] != missing_val)) p_feat[i] = categ_map[p_feat[i]];
		if (zero_missing && (p_feat[i] == missing_val)) p_feat[i] = zero_missing_val;
	}
	return 0;
}

void BasicFeatGenerator::set_signal_ids(MedSignals& sigs) {
	signalId = sigs.sid(signalName);
	timeRangeSignalId = sigs.sid(timeRangeSignalName);
	needs_categ_dict = sigs.is_categorical_channel(signalId, val_channel);
	if (!categ_value2id.empty() &&
		!sigs.is_categorical_channel(signalId, val_channel)) {
		categ_value2id.clear();
		needs_categ_dict = false;
	}
}

// Init look-up table
//.......................................................................................
void BasicFeatGenerator::init_tables(MedDictionarySections& dict) {

	// Look-up table for FTR_CATEGORY_SET types
	if (type == FTR_CATEGORY_SET || type == FTR_CATEGORY_SET_COUNT || type == FTR_CATEGORY_SET_SUM || type == FTR_CATEGORY_SET_FIRST || type == FTR_CATEGORY_SET_FIRST_TIME
		|| type == FTR_CATEGORY_SET_LAST_NTH) {
		if (lut.size() == 0) {
			int section_id = dict.section_id(signalName);
			//MLOG("BEFORE_LEARN:: signalName %s section_id %d sets size %d sets[0] %s\n", signalName.c_str(), section_id, sets.size(), sets[0].c_str());
			dict.prep_sets_lookup_table(section_id, sets, lut);
			//MLOG("AFTER_LEARN:: signalName %s section_id %d sets size %d sets[0] %s LUT %d\n", signalName.c_str(), section_id, sets.size(), sets[0].c_str(), lut.size());
		}
	}
	else
		lut.clear();

	// Look-up tables for categ_require_dict types
	if (categ_require_dict.find(type) != categ_require_dict.end() && needs_categ_dict) {
		MedDictionary& _dict = dict.dicts[dict.section_id(signalName)];
		if (_dict.Id2Name.size() == 0)
			MTHROW_AND_ERR("Empty dictionary for signal %s\n", signalName.c_str());

		// Before learning: Generate a map from name (in dictionary) to dictionary-independent value (this also usefull for backward compatability)
		if (categ_value2id.empty())
			categ_value2id = _dict.Name2Id;

		// For generation : Generate a map from dictionary value to dictionary-independent value
		categ_map.resize(_dict.Id2Name.rbegin()->first + 1);

		int maxId = 0;
		for (auto& rec : categ_value2id) {
			if (rec.second > maxId)
				maxId = rec.second;
		}

		for (auto& rec : _dict.Id2Name) {
			if (categ_value2id.find(rec.second) == categ_value2id.end())
				categ_map[rec.first] = maxId + 1;
			else
				categ_map[rec.first] = categ_value2id[rec.second];
		}
	}

	return;
}

void BasicFeatGenerator::prepare(MedFeatures &features, MedPidRepository& rep, MedSamples& samples) {
	FeatureGenerator::prepare(features, rep, samples);

	// Handle categorical data
	if (rep.sigs.is_categorical_channel(signalId, val_channel)) {
		if (categ_forbidden.find(type) != categ_forbidden.end())
			MTHROW_AND_ERR("name %s SignalId %d val_channel %d , Type %d not allowed on categorical data in BasicFeatureGenerator\n", names[0].c_str(), signalId, val_channel, type);
		if (categ_require_dict.find(type) != categ_require_dict.end()) {
			int section_id = rep.dict.SectionName2Id[signalName];
			for (auto& rec : rep.dict.dicts[section_id].Id2Name)
				features.attributes[names[0]].value2Name[rec.first] = rec.second;
		}
	}

	apply_categ_map = (rep.sigs.is_categorical_channel(signalId, val_channel) && (!categ_map.empty()));
}

void BasicFeatGenerator::get_required_signal_categories(unordered_map<string, vector<string>> &signal_categories_in_use) const {
	if (type == FTR_CATEGORY_SET || type == FTR_CATEGORY_SET_COUNT || type == FTR_CATEGORY_SET_SUM || type == FTR_CATEGORY_SET_FIRST || type == FTR_CATEGORY_SET_FIRST_TIME) {
		signal_categories_in_use[signalName] = sets;
	}
}

//.......................................................................................
float BasicFeatGenerator::get_value(PidDynamicRec& rec, int idx, int time, int outcomeTime) {

	rec.uget(signalId, idx);

	int updated_win_from = win_from, updated_win_to = win_to;
	int updated_d_win_from = d_win_from, updated_d_win_to = d_win_to;
	if (timeRangeSignalId != -1) {
		UniversalSigVec time_range_usv;
		rec.uget(timeRangeSignalId, idx, time_range_usv);
		get_updated_time_window(time_range_usv, timeRangeType, time_unit_range_sig, time_unit_win, time_unit_sig, time,
			win_from, updated_win_from, win_to, updated_win_to, (type == FTR_WIN_DELTA_VALUE), d_win_from, updated_d_win_from, d_win_to, updated_d_win_to);
	}

	switch (type) {
	case FTR_LAST_VALUE:	return uget_last(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_LAST_NTH_VALUE:	return uget_last_nth(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_FIRST_VALUE:	return uget_first(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_LAST2_VALUE:	return uget_last2(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_AVG_VALUE:		return uget_avg(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_MAX_VALUE:		return uget_max(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_MIN_VALUE:		return uget_min(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_STD_VALUE:		return uget_std(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_SUM_VALUE:		return uget_sum(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_LAST_DELTA_VALUE:	return uget_last_delta(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_LAST_DAYS:			return uget_last_time(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_LAST2_DAYS:		return uget_last2_time(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_SLOPE_VALUE:		return uget_slope(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_WIN_DELTA_VALUE:	return uget_win_delta(rec.usv, time, updated_win_from, updated_win_to, updated_d_win_from, updated_d_win_to, outcomeTime);
	case FTR_CATEGORY_SET:				return uget_category_set(rec, rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_CATEGORY_SET_LAST_NTH:				return uget_category_set_last_nth(rec, rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_CATEGORY_SET_COUNT:		return uget_category_set_count(rec, rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_CATEGORY_SET_SUM:			return uget_category_set_sum(rec, rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_NSAMPLES:			return uget_nsamples(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_EXISTS:			return uget_exists(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_RANGE_WIDTH:			return uget_range_width(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_MAX_DIFF:			return uget_max_diff(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_FIRST_DAYS:		return uget_first_time(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_CATEGORY_SET_FIRST:		return uget_category_set_first(rec, rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_CATEGORY_SET_FIRST_TIME:		return uget_category_set_first_time(rec, rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	case FTR_TIME_SINCE_LAST_CHANGE:		 return uget_time_since_last_change(rec.usv, time, updated_win_from, updated_win_to, outcomeTime);
	default:	return missing_val;
	}

	return missing_val;
}

// Init
//.......................................................................................
int BasicFeatGenerator::init(map<string, string>& mapper) {

	for (auto entry : mapper) {
		string field = entry.first;
		//! [BasicFeatGenerator::init]
		if (field == "type") { type = name_to_type(entry.second); }
		else if (field == "win_from") win_from = med_stoi(entry.second);
		else if (field == "win_to") win_to = med_stoi(entry.second);
		else if (field == "d_win_from") d_win_from = med_stoi(entry.second);
		else if (field == "d_win_to") d_win_to = med_stoi(entry.second);
		else if (field == "signalName" || field == "signal") signalName = entry.second;
		else if (field == "time_unit") time_unit_win = med_time_converter.string_to_type(entry.second);
		else if (field == "time_channel") time_channel = med_stoi(entry.second);
		else if (field == "val_channel") val_channel = med_stoi(entry.second);
		else if (field == "sum_channel") sum_channel = med_stoi(entry.second);
		else if (field == "sets") boost::split(sets, entry.second, boost::is_any_of(","));
		else if (field == "tags") boost::split(tags, entry.second, boost::is_any_of(","));
		else if (field == "in_set_name") in_set_name = entry.second;
		else if (field == "bound_outcomeTime") bound_outcomeTime = stoi(entry.second) > 0;
		else if (field == "weights_generator") iGenerateWeights = med_stoi(entry.second);
		else if (field == "time_range_signal") timeRangeSignalName = entry.second;
		else if (field == "time_range_signal_type") timeRangeType = time_range_name_to_type(entry.second);
		else if (field == "min_value") min_value = stof(entry.second);
		else if (field == "max_value") max_value = stof(entry.second);
		else if (field == "nth" || field == "Nth") N_th = stoi(entry.second);
		else if (field == "zero_missing") zero_missing = stoi(entry.second);
		else if (field == "zero_missing_val") zero_missing_val = med_stof(entry.second);
		else if (field == "missing_value") missing_val = stof(entry.second);
		else if (field == "full_name") full_name = stoi(entry.second);
		else if (field == "rename_signal") rename_signal = entry.second;
		else if (field != "fg_type")
			MLOG("Unknown parameter \'%s\' for BasicFeatGenerator\n", field.c_str());
		//! [BasicFeatGenerator::init]
	}

	// names for BasicFeatGenerator are set as a first step in the Learn call as we must have access to the MedRepository
	names.clear();

	set_names();

	req_signals.assign(1, signalName);
	if (timeRangeSignalName != "")
		req_signals.push_back(timeRangeSignalName);

	return 0;
}

//=======================================================================================
// Age
//=======================================================================================
int AgeGenerator::_generate(PidDynamicRec& rec, MedFeatures& features, int index, int num, vector<float *> &_p_data) {

	// Sanity check
	if (signalId == -1)
		MTHROW_AND_ERR("Uninitialized signalId in age generation\n");

	float *p_feat = _p_data[0] + index;


	UniversalSigVec usv;
	rec.uget(signalId, 0, usv);
	if (usv.len != 1) { MTHROW_AND_ERR("AgeGenerator: id %d , got len %d for signal %d [%s])...\n", rec.pid, usv.len, signalId, signalName.c_str()); }
	if (usv.len == 0) throw MED_EXCEPTION_NO_BYEAR_GIVEN;
	if (signalName == "BDATE") {
		int bdate;
		if (usv.n_val_channels() > 0)
			bdate = (int)(usv.Val(0));
		else
			bdate = (int)(usv.Time(0));
		for (int i = 0; i < num; i++) {
			int time = med_time_converter.convert_times(features.time_unit, MedTime::Date, features.samples[index + i].time);
			//int days_since_birth = get_day_approximate(time) - get_day_approximate(bdate);
			//p_feat[i] = (float)(1.0 * days_since_birth) / 365;
			p_feat[i] = int(time / 10000) - int(bdate / 10000);

		}
	}
	else if (signalName == "BYEAR") {
		int byear = usv.Val<int>(0);
		for (int i = 0; i < num; ++i)
			p_feat[i] = float(med_time_converter.convert_times(features.time_unit, MedTime::Date, features.samples[index + i].time) / 10000 - byear);
	}
	else
		MTHROW_AND_ERR("Unknown age signal [%s] \n", signalName.c_str());

	return 0;
}

int AgeGenerator::init(map<string, string>& mapper) {

	for (auto entry : mapper) {
		string field = entry.first;
		if (field == "signal")
			signalName = entry.second;
		else if (field == "tags")
			boost::split(tags, entry.second, boost::is_any_of(","));
		else if (field != "fg_type")
			MTHROW_AND_ERR("Unknown parameter [%s] for AgeGenerator\n", field.c_str());
	}
	set_names();

	req_signals.clear();
	req_signals.push_back(signalName);
	return 0;
}

//=======================================================================================
// Gender
//=======================================================================================
int GenderGenerator::_generate(PidDynamicRec& rec, MedFeatures& features, int index, int num, vector<float *> &_p_data) {

	// Sanity check
	if (genderId == -1) {
		MERR("Uninitialized genderId\n");
		return -1;
	}

	rec.uget(genderId, 0);
	if (rec.usv.len == 0) throw MED_EXCEPTION_NO_GENDER_GIVEN;
	int gender = (int)(rec.usv.Val(0));

	float *p_feat = _p_data[0] + index;
	for (int i = 0; i < num; i++)
		p_feat[i] = (float)gender;

	return 0;
}


// Init
//.......................................................................................
int GenderGenerator::init(map<string, string>& mapper) {

	for (auto entry : mapper) {
		string field = entry.first;
		//! [GenderGenerator::init]
		if (field == "tags") boost::split(tags, entry.second, boost::is_any_of(","));
		else if (field == "weights_generator") iGenerateWeights = med_stoi(entry.second);
		else if (field != "fg_type")
			MLOG("Unknown parameter \'%s\' for GenderGenerator\n", field.c_str());
		//! [GenderGenerator::init]
	}

	// naming and required signals

	names.clear();
	set_names();

	req_signals.assign(1, "GENDER");

	return 0;
}

void GenderGenerator::get_required_signal_categories(unordered_map<string, vector<string>> &signal_categories_in_use) const {
	if (category_values.empty())
		return;
	for (const string &categ : category_values)
		if (categ != "GENDER") //remove the name of the signal - if there is rename of the name of signal to SEX
			signal_categories_in_use["GENDER"].push_back(categ);
}

//=======================================================================================
// Singleton
//=======================================================================================

//.......................................................................................
int SingletonGenerator::_learn(MedPidRepository& rep, const MedSamples& samples, vector<RepProcessor *> processors) {

	// Learn mapping from string to value (not relying on dictionary ...)
	if (rep.sigs.Sid2Info[signalId].is_categorical_per_val_channel[0]) {
		int section_id = rep.dict.section_id(signalName);

		int idx = 0;
		for (auto& rec : rep.dict.dicts[section_id].Name2Id)
			name2Value[rec.first] = idx++;

		name2Value["SINGLETON_UNKNOWN"] = idx;
	}

	get_id2Value(rep.dict);

	return 0;
}

//.......................................................................................
int SingletonGenerator::_generate(PidDynamicRec& rec, MedFeatures& features, int index, int num, vector<float *> &_p_data) {

	// Sanity check
	if (signalId == -1) {
		MERR("SingletonGenerator::_generate - Uninitialized signalId(%s)\n", signalName.c_str());
		return -1;
	}

	rec.uget(signalId, 0);
	float value;
	if (rec.usv.len == 0)
		value = missing_val;
	else {
		if (sets.size() == 0)
		{
			// Normal Singleton
			if (id2Value.empty()) // Values as is
				value = (float)((int)(rec.usv.Val(0)));
			else {// dictionaries
				int idx_lut = (int)(rec.usv.Val(0));
				if (idx_lut < 0 || idx_lut >= id2Value.size())
					MTHROW_AND_ERR("Error SingletonGenerator::_generate - invalid value %d\n", idx_lut);
				value = id2Value[idx_lut];
			}
		}
		else
		{
			// Categorial Variable - check whether exists in LUT. Return 0/1
			value = (float)lut[((int)(rec.usv.Val(0)))];
		}
	}
	float *p_feat = _p_data[0] + index;
	for (int i = 0; i < num; i++)
		p_feat[i] = value;

	return 0;
}

//.......................................................................................
void SingletonGenerator::set_names()
{
	if (names.empty()) {
		string name = "FTR_" + int_to_string_digits(serial_id, 6) + "." + signalName;
		//string name = signalName + ".";
		string set_names = in_set_name;
		if (set_names == "" && this->sets.size() > 0)
			set_names = boost::algorithm::join(this->sets, "_");

		if (set_names != "")
			name += ".category_set_" + set_names;

		names.push_back(name);
		//MLOG("Created %s\n", name.c_str());
	}
}

//.......................................................................................
void SingletonGenerator::init_tables(MedDictionarySections& dict) {

	//MLOG("sets size = %d \n", lut.size());
	if (sets.size() > 0) {
		// This is a categorial variable.
		//if (lut.size() == 0) {
		int section_id = dict.section_id(signalName);
		//MLOG("BEFORE_LEARN:: signalName %s section_id %d sets size %d sets[0] %s\n", signalName.c_str(), section_id, sets.size(), sets[0].c_str());
		dict.prep_sets_lookup_table(section_id, sets, lut);
		//MLOG("AFTER_LEARN:: signalName %s section_id %d sets size %d sets[0] %s LUT %d\n", signalName.c_str(), section_id, sets.size(), sets[0].c_str(), lut.size());
	//}
	}
	else {
		lut.clear();
		get_id2Value(dict);
	}

	return;
}

void SingletonGenerator::get_required_signal_categories(unordered_map<string, vector<string>> &signal_categories_in_use) const {
	if (sets.size() > 0)
		signal_categories_in_use[signalName] = sets;
	else {
		for (auto &it : name2Value)
			signal_categories_in_use[signalName].push_back(it.first);
	}
}
//.......................................................................................
void SingletonGenerator::get_id2Value(MedDictionarySections& dict) {

	if (!name2Value.empty()) {
		int section_id = dict.section_id(signalName);

		int max_id = 1;
		if (dict.dicts[section_id].Id2Name.size() > 0)
			max_id = dict.dicts[section_id].Id2Name.rbegin()->first;
		else
			MTHROW_AND_ERR("SingletonGenerator::init_tables() : Got an empty Id2Name...\n");

		id2Value.resize(max_id + 1, (float)0);

		for (auto& rec : dict.dicts[section_id].Id2Name) {
			if (name2Value.find(rec.second) == name2Value.end()) {
				//Test if We can recognize other name:
				int found_id = -1;
				for (const string &other_nm : dict.dicts[section_id].Id2Names[rec.first])
				{
					if (name2Value.find(other_nm) != name2Value.end())
					{
						found_id = name2Value.at(other_nm);
						break;
					}
				}
				if ((found_id) >= 0)
					id2Value[rec.first] = found_id;
				else
					id2Value[rec.first] = name2Value["SINGLETON_UNKNOWN"];
			}
			else
				id2Value[rec.first] = name2Value[rec.second];
		}


	}
}
//.......................................................................................
int SingletonGenerator::init(map<string, string>& mapper) {

	for (auto entry : mapper) {
		string field = entry.first;
		//! [SingletonGenerator::init]
		if (field == "signalName" || field == "signal") signalName = entry.second;
		else if (field == "tags") boost::split(tags, entry.second, boost::is_any_of(","));
		else if (field == "weights_generator") iGenerateWeights = med_stoi(entry.second);
		else if (field == "sets") boost::split(sets, entry.second, boost::is_any_of(","));
		else if (field == "in_set_name") in_set_name = entry.second;
		else if (field != "fg_type")
			MLOG("Unknown parameter \'%s\' for SingletonGenerator\n", field.c_str());
		//! [SingletonGenerator::init]
	}

	// naming and required signals

	names.clear();
	set_names();

	req_signals.assign(1, signalName);

	return 0;
}

//.......................................................................................
void SingletonGenerator::prepare(MedFeatures & features, MedPidRepository& rep, MedSamples& samples) {
	FeatureGenerator::prepare(features, rep, samples);
	if (sets.size() == 0)
		for (auto& rec : name2Value)
			features.attributes[names[0]].value2Name[rec.second] = rec.first;
}

//=======================================================================================
// RangeFeatGenerator
//=======================================================================================

//................................................................................................................
void RangeFeatGenerator::set_names() {

	names.clear();

	string name = signalName + ".";

	switch (type) {
	case FTR_RANGE_CURRENT:	name += "current" + ((sets.size() > 0) ? "_" + sets[0] : ""); break;
	case FTR_RANGE_LATEST:	name += "latest" + ((sets.size() > 0) ? "_" + sets[0] : ""); break;
	case FTR_RANGE_MIN:		name += "min"; break;
	case FTR_RANGE_MAX:		name += "max"; break;
	case FTR_RANGE_EVER:	name += "ever" + ((sets.size() > 0) ? "_" + sets[0] : ""); break;
	case FTR_RANGE_TIME_DIFF: name += "time_diff_" + to_string(check_first) + ((sets.size() > 0) ? "_" + sets[0] : ""); break;
	case FTR_RANGE_RECURRENCE_COUNT: name += "recurrence_count"; break;
	case FTR_RANGE_TIME_COVERED: name += "time_covered" + ((sets.size() > 0) ? "_" + sets[0] : ""); break;
	case FTR_RANGE_LAST_NTH_TIME_LENGTH: name += "last_nth_time_len_" + to_string(N_th) + ((sets.size() > 0) ? "_" + sets[0] : "");; break;
	case FTR_RANGE_TIME_DIFF_START: name += "time_diff_start" + ((sets.size() > 0) ? "_" + sets[0] : ""); break;
	case FTR_RANGE_TIME_INSIDE: name += "time_inside" + ((sets.size() > 0) ? "_" + sets[0] : ""); break;
	default: {
		name += "ERROR";
		MTHROW_AND_ERR("Got a wrong type in range feature generator %d\n", type);
	}
	}

	name += ".win_" + std::to_string(win_from) + "_" + std::to_string(win_to);
	if (val_channel != 0)
		name += ".v" + std::to_string(val_channel);
	names.push_back("FTR_" + int_to_string_digits(serial_id, 6) + "." + name);

	if (timeRangeSignalName != "")
		name += ".time_range_" + timeRangeSignalName + "_" + time_range_type_to_name(timeRangeType);

	// add the undecorated feature name as a tag, so we can later remove/select it with TagFeatureSelector
	tags.push_back(name);
}

// Init
//.......................................................................................
int RangeFeatGenerator::init(map<string, string>& mapper) {

	for (auto entry : mapper) {
		string field = entry.first;
		//! [RangeFeatGenerator::init]
		if (field == "type") { type = name_to_type(entry.second); }
		else if (field == "win_from") win_from = med_stoi(entry.second);
		else if (field == "win_to") win_to = med_stoi(entry.second);
		else if (field == "signalName" || field == "signal") signalName = entry.second;
		else if (field == "time_unit" || field == "win_time_unit") time_unit_win = med_time_converter.string_to_type(entry.second);
		else if (field == "val_channel") val_channel = med_stoi(entry.second);
		else if (field == "sets") boost::split(sets, entry.second, boost::is_any_of(","));
		else if (field == "tags") boost::split(tags, entry.second, boost::is_any_of(","));
		else if (field == "weights_generator") iGenerateWeights = med_stoi(entry.second);
		else if (field == "check_first") check_first = med_stoi(entry.second);
		else if (field == "time_range_signal") timeRangeSignalName = entry.second;
		else if (field == "time_range_signal_type") timeRangeType = time_range_name_to_type(entry.second);
		else if (field == "recurrence_delta") recurrence_delta = med_stoi(entry.second);
		else if (field == "min_range_time") min_range_time = med_stoi(entry.second);
		else if (field == "div_factor") div_factor = med_stof(entry.second);
		else if (field == "Nth" || field == "nth") N_th = med_stoi(entry.second);
		else if (field == "zero_missing") zero_missing = med_stoi(entry.second);
		else if (field == "strict_times") strict_times = med_stoi(entry.second);
		else if (field == "conditional_channel") conditional_channel = med_stoi(entry.second);
		else if (field == "first_evidence_time_channel") first_evidence_time_channel = med_stoi(entry.second);
		else if (field == "regex_on_sets") regex_on_sets = (bool)med_stoi(entry.second);
		else if (field != "fg_type")
			MLOG("Unknown parameter \'%s\' for RangeFeatGenerator\n", field.c_str());
		//! [RangeFeatGenerator::init]
	}

	// set names and required signals
	set_names();

	req_signals.assign(1, signalName);
	if (timeRangeSignalName != "")
		req_signals.push_back(timeRangeSignalName);

	return 0;
}

void RangeFeatGenerator::init_tables(MedDictionarySections& dict) {

	if (type == FTR_RANGE_EVER || type == FTR_RANGE_TIME_DIFF || type == FTR_RANGE_TIME_DIFF_START || conditional_channel >= 0) {
		if (lut.size() == 0) {
			int section_id = dict.section_id(signalName);
			if (regex_on_sets)
			{
				vector<string> agg_sets;
				unordered_set<string> aggregated_values;
				for (auto& s : sets)
				{
					vector<string> curr_set;
					dict.dicts[section_id].get_regex_names(".*" + s + ".*", curr_set);
					aggregated_values.insert(curr_set.begin(), curr_set.end());
				}
				agg_sets.insert(agg_sets.begin(), aggregated_values.begin(), aggregated_values.end());
				dict.prep_sets_lookup_table(section_id, agg_sets, lut);
			}
			else
				dict.prep_sets_lookup_table(section_id, sets, lut);
		}
	}
	else
		lut.clear();
	return;
}

void RangeFeatGenerator::get_required_signal_categories(unordered_map<string, vector<string>> &signal_categories_in_use) const {
	if (type == FTR_RANGE_EVER || type == FTR_RANGE_TIME_DIFF || type == FTR_RANGE_TIME_DIFF_START || conditional_channel >= 0)
		signal_categories_in_use[signalName] = sets;
}

// Init
//.......................................................................................
void RangeFeatGenerator::init_defaults() {
	generator_type = FTR_GEN_RANGE;
	signalId = -1;
	sets.clear();
	time_unit_sig = MedTime::Undefined;
	time_unit_win = global_default_windows_time_unit;
	string _signalName = "";
	set(_signalName, FTR_RANGE_CURRENT, 0, 360000);
	timeRangeSignalId = -1;
	N_th = 0;
	zero_missing = 0;
	strict_times = 0;
	conditional_channel = -1;
};

// Get type from name
//.......................................................................................
RangeFeatureTypes RangeFeatGenerator::name_to_type(const string &name)
{

	if (name == "current")				return FTR_RANGE_CURRENT;
	if (name == "latest")			return FTR_RANGE_LATEST;
	if (name == "max")			return FTR_RANGE_MAX;
	if (name == "min")			return FTR_RANGE_MIN;
	if (name == "ever")			return FTR_RANGE_EVER;
	if (name == "time_diff")  return FTR_RANGE_TIME_DIFF;
	if (name == "recurrence_count")		return FTR_RANGE_RECURRENCE_COUNT;
	if (name == "time_covered")		return FTR_RANGE_TIME_COVERED;
	if (name == "last_nth_time_len")		return FTR_RANGE_LAST_NTH_TIME_LENGTH;
	if (name == "time_diff_start")  return FTR_RANGE_TIME_DIFF_START;
	if (name == "time_inside")  return FTR_RANGE_TIME_INSIDE;

	return (RangeFeatureTypes)med_stoi(name);
}

// Generate
//.......................................................................................
int RangeFeatGenerator::_generate(PidDynamicRec& rec, MedFeatures& features, int index, int num, vector<float *> &_p_data) {

	if (time_unit_sig == MedTime::Undefined)	time_unit_sig = rec.my_base_rep->sigs.Sid2Info[signalId].time_unit;
	if (timeRangeSignalName != "" && time_unit_range_sig == MedTime::Undefined)
		time_unit_range_sig = rec.my_base_rep->sigs.Sid2Info[timeRangeSignalId].time_unit;

	float *p_feat = _p_data[0] + index;
	MedSample *p_samples = &(features.samples[index]);

	for (int i = 0; i < num; i++) {
		p_feat[i] = get_value(rec, i, med_time_converter.convert_times(features.time_unit, time_unit_win, p_samples[i].time));
		if (zero_missing && (p_feat[i] == missing_val)) p_feat[i] = 0;
	}

	return 0;
}

//.......................................................................................
float RangeFeatGenerator::get_value(PidDynamicRec& rec, int idx, int time) {

	rec.uget(signalId, idx);

	int updated_win_from = win_from, updated_win_to = win_to;
	int updated_d_win_from, updated_d_win_to;
	if (timeRangeSignalId != -1) {
		UniversalSigVec time_range_usv;
		rec.uget(timeRangeSignalId, idx, time_range_usv);
		get_updated_time_window(time_range_usv, timeRangeType, time_unit_range_sig, time_unit_win, time_unit_sig, time,
			win_from, updated_win_from, win_to, updated_win_to, false, 0, updated_d_win_from, 0, updated_d_win_to);
	}

	switch (type) {
	case FTR_RANGE_CURRENT:	return uget_range_current(rec.usv, updated_win_from, updated_win_to, time);
	case FTR_RANGE_LATEST:	return uget_range_latest(rec.usv, updated_win_from, updated_win_to, time);
	case FTR_RANGE_MIN:	return uget_range_min(rec.usv, updated_win_from, updated_win_to, time);
	case FTR_RANGE_MAX:		return uget_range_max(rec.usv, updated_win_from, updated_win_to, time);
	case FTR_RANGE_EVER:		return uget_range_ever(rec.usv, updated_win_from, updated_win_to, time);
	case FTR_RANGE_TIME_DIFF: 	return uget_range_time_diff(rec.usv, updated_win_from, updated_win_to, time);
	case FTR_RANGE_RECURRENCE_COUNT: return uget_range_recurrence_count(rec.usv, updated_win_from, updated_win_to, time);
	case FTR_RANGE_TIME_COVERED: return uget_range_time_covered(rec.usv, win_from, win_to, time);
	case FTR_RANGE_LAST_NTH_TIME_LENGTH: return uget_range_last_nth_time_len(rec.usv, win_from, win_to, time);
	case FTR_RANGE_TIME_DIFF_START: 	return uget_range_time_diff_start(rec.usv, updated_win_from, updated_win_to, time);
	case FTR_RANGE_TIME_INSIDE: 	return uget_range_time_inside(rec.usv, updated_win_from, updated_win_to, time);


	default:	return missing_val;
	}

	return missing_val;
}


//................................................................................................................
// in all following uget funcs the relevant time window is [min_time, max_time] and time is given in time_unit_win
//................................................................................................................

// get the last value in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_last(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	//MLOG("min_time %d max_time %d usv.len %d time %d\n", min_time, max_time, usv.len, time);
	for (int i = usv.len - 1; i >= 0; i--) {
		int itime = usv.Time(i, time_channel);
		//MLOG("%d,%d,%f\n", i, itime, usv.Val(i, val_channel));
		if (itime <= max_time) {
			if (itime >= min_time)
				return usv.Val(i, val_channel);
			else
				return missing_val;
		}
	}

	return missing_val;
}

// get the last nth value in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_last_nth(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	//MLOG("min_time %d max_time %d usv.len %d time %d\n", min_time, max_time, usv.len, time);
	int nth = 0;
	for (int i = usv.len - 1; i >= 0; i--) {
		int itime = usv.Time(i, time_channel);
		//MLOG("%d,%d,%d(%d),%f\n", i, itime, nth, N_th, usv.Val(i, val_channel));
		if (itime <= max_time) {
			if (itime >= min_time) {
				if (nth == N_th)
					return usv.Val(i, val_channel);
				nth++;
			}
			else
				return missing_val;
		}
	}

	return missing_val;
}


//.......................................................................................
// get the first value in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_first(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime >= min_time) {
			if (itime > max_time)
				return missing_val;
			else
				return usv.Val(i, val_channel);
		}
	}
	return missing_val;
}

//.......................................................................................
// get the last2 value (the one before the last) in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_last2(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	for (int i = usv.len - 1; i >= 0; i--) {
		if (usv.Time(i, time_channel) <= max_time) {
			if (i > 0 && usv.Time(i - 1, time_channel) >= min_time)
				return usv.Val(i - 1, val_channel);
			else
				return missing_val;
		}
	}

	return missing_val;
}

//.......................................................................................
// get the average value in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_avg(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	double sum = 0, nvals = 0;

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) break;
		if (itime >= min_time) {
			sum += usv.Val(i, val_channel);
			nvals++;
		}
	}

	if (nvals > 0)
		return (float)(sum / nvals);

	return missing_val;
}

//.......................................................................................
// get the max value in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_max(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	float max_val = -1e10;

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) break;
		if (itime >= min_time && usv.Val(i, val_channel) > max_val)	max_val = usv.Val(i, val_channel);
	}

	if (max_val > -1e10)
		return max_val;

	return missing_val;
}


//.......................................................................................
// get max_val - min_val in the window
float BasicFeatGenerator::uget_range_width(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	float max_val = uget_max(usv, time, _win_from, _win_to, outcomeTime);
	float min_val = uget_min(usv, time, _win_from, _win_to, outcomeTime);

	if (max_val == missing_val || min_val == missing_val)
		return missing_val;

	return max_val - min_val;
}

//.......................................................................................
// get the min value in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_min(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	float min_val = (float)1e20;

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) break;
		if (itime >= min_time && usv.Val(i, val_channel) < min_val) 	min_val = usv.Val(i, val_channel);
	}

	if (min_val < (float)1e20)
		return min_val;

	return missing_val;
}


//.......................................................................................
// get the sum of values in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_sum(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	float sum_val = (float)0;

	for (int i = usv.len - 1; i >= 0; i--) {
		int itime = usv.Time(i, time_channel);
		if (itime < min_time) break;
		if (itime <= max_time) sum_val += usv.Val(i, val_channel);
	}

	return sum_val;
}


//.......................................................................................
// get the std in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_std(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	double sum = 0, sum_sq = 0, nvals = 0;

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) break;
		if (itime >= min_time) {
			float ival = usv.Val(i, val_channel);
			sum += ival;
			sum_sq += static_cast<double>(ival) * static_cast<double>(ival);
			nvals++;
		}
	}

	if (nvals > 1) {
		double avg = sum / nvals;
		double var = sum_sq / nvals - avg * avg;
		if (var < 0.0001) var = 0.0001;
		return (float)sqrt(var);
	}

	return missing_val;
}

//.......................................................................................
float BasicFeatGenerator::uget_last_delta(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	for (int i = usv.len - 1; i >= 0; i--) {
		if (usv.Time(i, time_channel) <= max_time) {
			if (i > 0 && usv.Time(i - 1, time_channel) >= min_time)
				return (usv.Val(i, val_channel) - usv.Val(i - 1, val_channel));
			else
				return missing_val;
		}
	}
	return missing_val;
}

//.......................................................................................
float BasicFeatGenerator::uget_last_time(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	for (int i = usv.len - 1; i >= 0; i--) {
		int itime = usv.Time(i, time_channel);
		if (itime <= max_time) {
			if (itime >= min_time) {
				float val = usv.Val(i, val_channel);
				if (val >= min_value && val <= max_value)
					return (float)(time - usv.TimeU(i, time_channel, time_unit_win));
			}
			else
				return missing_val;
		}
	}

	return missing_val;
}

//.......................................................................................
float BasicFeatGenerator::uget_first_time(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime >= min_time) {
			if (itime > max_time)
				return missing_val;
			else {
				float val = usv.Val(i, val_channel);
				if (val >= min_value && val <= max_value)
					return (float)(time - usv.TimeU(i, time_channel, time_unit_win));
			}
		}
	}
	return missing_val;
}

//.......................................................................................
float BasicFeatGenerator::uget_time_since_last_change(UniversalSigVec &usv, int time_point, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time_point, min_time, max_time, bound_outcomeTime, outcomeTime);
	float time_since_last_change = missing_val;
	if (usv.len > 1)
	{
		float val = usv.Val(0, val_channel);
		for (int i = 1; i < usv.len; i++) {
			float last_val = val;
			int itime = usv.Time(i, time_channel);
			if (itime >= min_time) {
				if (itime > max_time)
					break;
				else {
					val = usv.Val(i, val_channel);
					if (val != last_val)
						time_since_last_change = (float)(time_point - usv.TimeU(i, time_channel, time_unit_win));
				}
			}
		}
	}
	return time_since_last_change;
}


//.......................................................................................
float BasicFeatGenerator::uget_last2_time(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	int cnt = 0;
	for (int i = usv.len - 1; i >= 0; i--) {
		int itime = usv.Time(i, time_channel);
		if (itime <= max_time) {
			if (itime >= min_time) {
				float val = usv.Val(i, val_channel);
				if (val >= min_value && val <= max_value) {
					if (cnt > 0)
						return (float)(time - usv.TimeU(i, time_channel, time_unit_win));
					cnt++;
				}
			}
			else
				return missing_val;
		}
	}

	return missing_val;
}

//.......................................................................................
// get the slope in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_slope(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{

	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	double sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
	double t_start = -1;

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) break;
		if (itime >= min_time) {
			if (t_start < 0) t_start = usv.TimeU(i, time_channel, time_unit_win);
			double t_curr = usv.TimeU(i, time_channel, time_unit_win);
			double x = (t_curr - t_start) / 500.0;
			double ival = usv.Val(i, val_channel);
			sx += x;
			sy += ival;
			sxx += x * x;
			sxy += x * ival;
			n++;
		}
	}

	if (n < 2) return missing_val;

	double cov = sxy - sx * (sy / n);
	double var = sxx - sx * (sx / n);

	if (var < 0.1)		return 0;

	return ((float)(cov / var));

}

//.......................................................................................
float BasicFeatGenerator::uget_win_delta(UniversalSigVec &usv, int time, int _win_from, int _win_to, int _d_win_from, int _d_win_to, int outcomeTime)
{
	float val1 = uget_last(usv, time, _win_from, _win_to, outcomeTime);
	if (val1 == missing_val) return missing_val;

	float val2 = uget_last(usv, time, _d_win_from, _d_win_to, outcomeTime);
	if (val2 == missing_val) return missing_val;


	return (val1 - val2);
}

//.......................................................................................
float BasicFeatGenerator::uget_category_set(PidDynamicRec &rec, UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{

	//#pragma omp critical
	//if (lut.size() == 0) {
	//		int section_id = rec.my_base_rep->dict.section_id(signalName);
	//		rec.my_base_rep->dict.prep_sets_lookup_table(section_id, sets, temp_lut);
	//	}

	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) break;
		if (itime >= min_time && lut[(int)usv.Val(i, val_channel)]) 	return 1;
	}

	return 0;
}

//.......................................................................................
float BasicFeatGenerator::uget_category_set_last_nth(PidDynamicRec &rec, UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{

	float val = uget_last_nth(usv, time, _win_from, _win_to, outcomeTime);
	if (val == missing_val) return missing_val;
	if (lut[(int)val]) return 1;
	return 0;
}

float BasicFeatGenerator::uget_category_set_first_time(PidDynamicRec &rec, UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	float diff = missing_val;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) return missing_val; // passed window
		if (lut[(int)usv.Val(i, val_channel)]) {// what we look for
			if (itime >= min_time) {
				diff = (float)time - med_time_converter.convert_times(MedTime::Date, MedTime::Days, itime);
				return diff; // inside window
			}
		}
	}

	return missing_val;
}


float BasicFeatGenerator::uget_category_set_first(PidDynamicRec &rec, UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) return 0; // passed window
		if (lut[(int)usv.Val(i, val_channel)]) // what we look for
			return itime >= min_time; // inside window
	}

	return 0;
}

//.......................................................................................
float BasicFeatGenerator::uget_category_set_count(PidDynamicRec &rec, UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	//#pragma omp critical
	//	if (lut.size() == 0) {
	//		int section_id = rec.my_base_rep->dict.section_id(signalName);
	//		rec.my_base_rep->dict.prep_sets_lookup_table(section_id, sets, lut);
	//	}

	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	int cnt = 0;
	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) break;
		if (itime >= min_time && lut[(int)usv.Val(i, val_channel)]) 	cnt++;
	}

	return (float)cnt;
}

//.......................................................................................
float BasicFeatGenerator::uget_category_set_sum(PidDynamicRec &rec, UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	//#pragma omp critical
	//	if (lut.size() == 0) {
	//		int section_id = rec.my_base_rep->dict.section_id(signalName);
	//		rec.my_base_rep->dict.prep_sets_lookup_table(section_id, sets, lut);
	//	}

	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	float sum = 0;
	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime > max_time) break;
		if (itime >= min_time && lut[(int)usv.Val(i, val_channel)]) 	sum += usv.Val(i, sum_channel);
	}

	return sum;
}

//.......................................................................................
// get the number of samples in [win_to, win_from] before time
float BasicFeatGenerator::uget_nsamples(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	if (usv.len <= 0)
		return 0;
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);
	float jVal;
	int jTime;
	int count_samples = 0;
	for (int j = 0; j < usv.len; ++j) {
		jTime = usv.Time(j, time_channel);
		if (jTime < min_time) { continue; }
		if (jTime > max_time) { break; }
		jVal = usv.Val(j, val_channel);
		if (jVal >= min_value && jVal <= max_value) { count_samples++; }
	}
	return (float)count_samples;
}

//.......................................................................................
// get 1.0 if there were any samples in [win_to, win_from] before time, else 0.0
float BasicFeatGenerator::uget_exists(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);
	int i, j;
	for (i = usv.len - 1; i >= 0 && usv.Time(i, time_channel) > max_time; i--);
	for (j = 0; j < usv.len && usv.Time(j, time_channel) < min_time; j++);
	if (i >= 0 && j < usv.len && usv.Time(i, time_channel) <= max_time && usv.Time(j, time_channel) >= min_time && i - j >= 0)
		return 1.0;
	else return 0.0;
}

//.......................................................................................
// get the max difference in values in the window [win_to, win_from] before time
float BasicFeatGenerator::uget_max_diff(UniversalSigVec &usv, int time, int _win_from, int _win_to, int outcomeTime)
{
	int min_time, max_time;
	get_window_in_sig_time(_win_from, _win_to, time_unit_win, time_unit_sig, time, min_time, max_time, bound_outcomeTime, outcomeTime);

	float max_diff = missing_val;
	vector<float> _vals_vec;
	for (int i = 0; i < usv.len; i++) {
		int itime = usv.Time(i, time_channel);
		if (itime >= min_time) {
			if (itime > max_time)
				break;
			else {
				if (_vals_vec.size() > 0) {
					nth_element(_vals_vec.begin(), _vals_vec.begin() + _vals_vec.size() / 2, _vals_vec.end());
					//float prev_val = median_prev_val;
					float prev_val = _vals_vec.back();
					float diff = usv.Val(i, val_channel) - prev_val;
					if (diff > max_diff || max_diff == missing_val)
						max_diff = diff;
				}
				_vals_vec.push_back(usv.Val(i, val_channel));
			}
		}
	}
	return max_diff;
}

//.......................................................................................
// get values for RangeFeatGenerator
//.......................................................................................
// get the value in a range that includes time - win_from, if available
float RangeFeatGenerator::uget_range_current(UniversalSigVec &usv, int updated_win_from, int updated_win_to, int time)
{
	int dummy_time, time_to_check;
	get_window_in_sig_time(updated_win_from, updated_win_from, time_unit_win, time_unit_sig, time, dummy_time, time_to_check, false);

	for (int i = 0; i < usv.len; i++) {
		int fromTime = usv.Time(i, 0);
		int toTime = usv.Time(i, 1);

		if (fromTime > time_to_check)
			break;
		else if (toTime >= time_to_check)
			return usv.Val(i, val_channel);
	}

	return missing_val;
}


//.......................................................................................
// get the value in a range that includes time - win_from, if available
float RangeFeatGenerator::uget_range_time_inside(UniversalSigVec &usv, int updated_win_from, int updated_win_to, int time)
{
	//int dummy_time, time_to_check;
	//get_window_in_sig_time(updated_win_from, updated_win_from, time_unit_win, time_unit_sig, time, dummy_time, time_to_check, false);

	//convert time back to signal time format
	time = med_time_converter.convert_times(time_unit_win, time_unit_sig, time);

	int time_inside = 0;
	for (int i = 0; i < usv.len; i++) {

		int fromTime = usv.Time(i, 0);
		int toTime = usv.Time(i, 1);
		int firstKnowTime = toTime;
		if (first_evidence_time_channel >= 0)
			firstKnowTime = usv.Time(i, first_evidence_time_channel);

		if (fromTime > time)
			break;

		if ((conditional_channel < 0) || lut[usv.Val<int>(i, conditional_channel)]) {
			if (time >= fromTime && time <= toTime && time >= firstKnowTime) {
				time_inside = 1 + med_time_converter.diff_times(time, fromTime, time_unit_sig, time_unit_win);
				break;
			}
		}

	}

	return (float)time_inside;
}

//.......................................................................................
// get the value in the latest range that intersets with time-win_to to time-win_from
float RangeFeatGenerator::uget_range_latest(UniversalSigVec &usv, int updated_win_from, int updated_win_to, int time)
{
	int min_time, max_time;
	get_window_in_sig_time(updated_win_from, updated_win_to, time_unit_win, time_unit_sig, time, min_time, max_time, false);

	float val = missing_val;
	for (int i = 0; i < usv.len; i++) {
		int fromTime = usv.Time(i, 0);
		int toTime = usv.Time(i, 1);

		if (fromTime > max_time)
			break;
		else if (toTime < min_time)
			continue;

		val = usv.Val(i, val_channel);
	}

	return val;
}

//.......................................................................................
// get the minimal value in a range that intersets with time-win_to to time-win_from
float RangeFeatGenerator::uget_range_min(UniversalSigVec &usv, int updated_win_from, int updated_win_to, int time)
{
	int min_time, max_time;
	get_window_in_sig_time(updated_win_from, updated_win_to, time_unit_win, time_unit_sig, time, min_time, max_time, false);

	float min_val = (float)1e20;

	for (int i = 0; i < usv.len; i++) {
		int fromTime = usv.Time(i, 0);
		int toTime = usv.Time(i, 1);

		if (fromTime > max_time)
			break;
		else if (toTime < min_time)
			continue;
		else if ((fromTime >= min_time || toTime <= max_time) && usv.Val(i, val_channel) < min_val)
			min_val = usv.Val(i, val_channel);
	}

	if (min_val < (float)1e20)
		return min_val;
	else
		return missing_val;
}

//.......................................................................................
// get the maximal value in a range that intersets with time-win_to to time-win_from
float RangeFeatGenerator::uget_range_max(UniversalSigVec &usv, int updated_win_from, int updated_win_to, int time)
{
	int min_time, max_time;
	get_window_in_sig_time(updated_win_from, updated_win_to, time_unit_win, time_unit_sig, time, min_time, max_time, false);

	float max_val = (float)-1e10;

	for (int i = 0; i < usv.len; i++) {
		int fromTime = usv.Time(i, 0);
		int toTime = usv.Time(i, 1);

		if (fromTime > max_time)
			break;
		else if (toTime < min_time)
			continue;
		else if ((fromTime >= min_time || toTime <= max_time) && usv.Val(i, val_channel) > max_val)
			max_val = usv.Val(i, val_channel);
	}

	if (max_val > (float)-1e10)
		return max_val;
	else
		return missing_val;
}

//.......................................................................................
// returns 1 if the range ever (up to time) had the value signalValue
float RangeFeatGenerator::uget_range_ever(UniversalSigVec &usv, int updated_win_from, int updated_win_to, int time)
{
	int min_time, max_time;
	get_window_in_sig_time(updated_win_from, updated_win_to, time_unit_win, time_unit_sig, time, min_time, max_time, false);

	for (int i = 0; i < usv.len; i++) {
		int fromTime = usv.Time(i, 0);
		int toTime = usv.Time(i, 1);

		if (fromTime > max_time)
			break;
		else if (toTime < min_time)
			continue;
		else if (lut[(int)usv.Val(i, val_channel)]) 	return 1;
	}
	return 0.0;
}

//.......................................................................................
// returns time diff if the range ever (up to time) had the value signalValue
float RangeFeatGenerator::uget_range_time_diff(UniversalSigVec &usv, int updated_win_from, int updated_win_to, int time)
{
	int min_time, max_time;
	get_window_in_sig_time(updated_win_from, updated_win_to, time_unit_win, time_unit_sig, time, min_time, max_time, false);

	int no_lut_ind = 0;
	float time_diff = missing_val;
	for (int i = 0; i < usv.len; i++) {
		int fromTime = usv.Time(i, 0);
		int toTime = usv.Time(i, 1);
		if (fromTime > max_time)
			break;
		else if (toTime < min_time)
			continue;
		else if (lut[(int)usv.Val(i, val_channel)]) {
			if (check_first == 1) {
				//in case of first range
				int max_time = fromTime;
				if (win_from > max_time) max_time = win_from;

				time_diff = (float)time - med_time_converter.convert_times(MedTime::Date, MedTime::Days, max_time);
				//fprintf(stderr, "max_time: %i time :%i from_time:%i win_from:%i time_diff:%i\n", max_time, time, fromTime, win_from, time_diff);
				return time_diff;
			}
			else {
				//in case of last range
				int time_to_diff = toTime;
				if (win_to < toTime) time_to_diff = win_to;
				time_diff = (float)+time - med_time_converter.convert_times(MedTime::Date, MedTime::Days, time_to_diff);
			}
		}
		else
			no_lut_ind = 1;
	}

	//in case of last range
	if (check_first == 0 && time_diff != missing_val)
		return time_diff;

	//in case of range exists but no lut
	if (no_lut_ind == 1) {
		time_diff = -1.0F* win_to;
		return time_diff;
	}
	//in case of no range in the time window
	else {
		return missing_val;
	}
}

//.......................................................................................

float RangeFeatGenerator::uget_range_time_diff_start(UniversalSigVec &usv, int updated_win_from, int updated_win_to, int time)
{
	int min_time, max_time;
	get_window_in_sig_time(updated_win_from, updated_win_to, time_unit_win, time_unit_sig, time, min_time, max_time, false);

	float time_diff = missing_val;
	for (int i = 0; i < usv.len; i++) {
		int fromTime = usv.Time(i, 0);
		if (fromTime > max_time)
			break;
		if (fromTime < min_time)
			continue;
		else if (lut[(int)usv.Val(i, val_channel)]) {
			time_diff = (float)time - med_time_converter.convert_times(MedTime::Date, MedTime::Days, fromTime);
		}
	}
	return time_diff;
}

//.......................................................................................
// get the number of samples in [win_to, win_from] before time that occur within recurrence_delta of the last sample
float RangeFeatGenerator::uget_range_recurrence_count(UniversalSigVec &usv, int updated_win_from, int updated_win_to, int time)
{
	int min_time, max_time;
	get_window_in_sig_time(updated_win_from, updated_win_to, time_unit_win, time_unit_sig, time, min_time, max_time, false);

	int last_end = 0;
	int num_recurrence = 0;
	for (int i = 0; i < usv.len; i++) {

		if (min_range_time != -1 && med_time_converter.diff_times(usv.Time(i, 1), usv.Time(i, 0), time_unit_sig, time_unit_sig) < min_range_time) {
			continue;
		}
		if (usv.Time(i, 1) > max_time) {
			break;
		}
		if (usv.Time(i, 0) >= min_time && med_time_converter.diff_times(usv.Time(i, 0), last_end, time_unit_sig, time_unit_sig) <= recurrence_delta) {
			num_recurrence++;
		}
		last_end = usv.Time(i, 1);
	}

	return (float)num_recurrence;
}


//.......................................................................................
float RangeFeatGenerator::uget_range_time_covered(UniversalSigVec &usv, int win_from, int win_to, int time)
{
	int min_time, max_time;
	get_window_in_sig_time(win_from, win_to, time_unit_win, time_unit_sig, time, min_time, max_time, false);

	int time_sum = 0;
	for (int i = 0; i < usv.len; i++) {

		int curr_from = usv.Time(i, 0);
		int curr_to = usv.Time(i, 1);
		int firstKnowTime = curr_to;
		if (first_evidence_time_channel >= 0)
			firstKnowTime = usv.Time(i, first_evidence_time_channel);

		if (curr_from > max_time) break;
		if (strict_times && firstKnowTime > max_time) continue; //continue and not break since not sorted by this field
		if (curr_to < min_time) continue;

		if (curr_from < min_time) curr_from = min_time;
		if (curr_to > max_time) curr_to = max_time;

		if ((conditional_channel < 0) || lut[usv.Val<int>(i, conditional_channel)])
			time_sum += med_time_converter.diff_times(curr_to, curr_from, time_unit_sig, time_unit_win);
	}

	return (float)time_sum / div_factor;
}

// calculate the length in time (in win_time unit) of the last nth range in the given window
// major usages: with N_th=0 , get the last range length in days, etc
//............................................................................................
float RangeFeatGenerator::uget_range_last_nth_time_len(UniversalSigVec &usv, int win_from, int win_to, int time)
{
	int min_time, max_time;
	get_window_in_sig_time(win_from, win_to, time_unit_win, time_unit_sig, time, min_time, max_time, false);

	int nth = 0;
	for (int i = usv.len - 1; i >= 0; i--) {
		int curr_from = usv.Time(i, 0);
		int curr_to = usv.Time(i, 1);
		int firstKnowTime = curr_to;
		if (first_evidence_time_channel >= 0)
			firstKnowTime = usv.Time(i, first_evidence_time_channel);

		if (curr_from > max_time) continue; // skip cases 
		if (curr_to > max_time) {
			if (strict_times && firstKnowTime > max_time) continue;
			curr_to = max_time;
		}

		if ((conditional_channel < 0) || lut[usv.Val<int>(i, conditional_channel)]) {
			// we are at the right window, find the n-th
			if (nth == N_th)
				return (float)med_time_converter.diff_times(curr_to, curr_from, time_unit_sig, time_unit_win);

			nth++;
			if (nth > N_th)
				return missing_val;
		}
	}

	return missing_val;
}

//=======================================================================================
// TimeFeatGenerator: creating sample-time features (e.g. differentiate between 
//	times of day, season of year, days of the week, etc.)
//=======================================================================================

// Set name
//................................................................................................................
void TimeFeatGenerator::set_names() {
	names.clear();

	string name = "FTR_" + int_to_string_digits(serial_id, 6) + ".Time." + time_unit_to_string(time_unit);
	for (string bin_name : time_bin_names)
		name += "." + bin_name;
	names.push_back(name);
}

// Time Unit Name
//................................................................................................................
string TimeFeatGenerator::time_unit_to_string(TimeFeatTypes time_unit) {

	switch (time_unit) {
	case FTR_TIME_YEAR: return "Year";
	case FTR_TIME_DATE: return "Date";
	case FTR_TIME_MONTH: return "Month";
	case FTR_TIME_DAY_IN_MONTH: return "Day_in_Month";
	case FTR_TIME_DAY_IN_WEEK: return "Day_in_Week";
	case FTR_TIME_HOUR: return "Hour";
	case FTR_TIME_MINUTE: return "Minute";
	default: return "Undef";
	}
}

// Init
//.......................................................................................
int TimeFeatGenerator::init(map<string, string>& mapper) {

	string time_bins_string = "";
	for (auto entry : mapper) {
		string field = entry.first;
		//! [TimeFeatGenerator::init]
		if (field == "time_unit") {
			if (get_time_unit(entry.second) < 0) {
				MERR("Cannot parse time unit \'%s\'\n", entry.second.c_str());
				return -1;
			}
		}
		else if (field == "tags") { boost::split(tags, entry.second, boost::is_any_of(",")); }
		// For example: "Jan:1,Feb:2,March:3,..." To give names to month in features
		else if (field == "time_bins") time_bins_string = entry.second;
		else if (field != "fg_type")
			MLOG("Unknown parameter \'%s\' for TimeFeatGenerator\n", field.c_str());
		//! [TimeFeatGenerator::init]
	}

	// Parse time_bins_string
	if (time_bins_string != "" && get_time_bins(time_bins_string) < 0) {
		MERR("Cannot parse time bins \'%s\'\n", time_bins_string.c_str());
		return -1;
	}

	// set names and required signals
	set_names();

	return 0;
}

// Parse time-binning info.
// Required format = name:val1,val2,...,val,name:val,...
//.......................................................................................
int TimeFeatGenerator::get_time_bins(string& binsInfo) {

	int nBins = get_nBins();
	time_bins.assign(nBins, -1);

	int start = 0;
	string name = "";
	int bin = -1;
	for (int i = 0; i < binsInfo.size(); i++) {
		if (binsInfo[i] == ':') {
			time_bin_names.push_back(binsInfo.substr(start, i - start));
			start = i + 1;
			bin++;
		}
		else if (binsInfo[i] == ',') {
			int index = stoi(binsInfo.substr(start, i - start));
			if (bin == -1 || index < 0 || index >= nBins)
				return -1;
			time_bins[index] = bin;
			start = i + 1;
		}
	}

	int index = stoi(binsInfo.substr(start, binsInfo.size() - start));
	if (bin == -1 || index < 0 || index >= nBins)
		return -1;
	time_bins[index] = bin;

	if (bin == 0) {
		MERR("Only one bin given for TimeFeatGenerator\n");
		return -1;
	}

	for (int i = 0; i < nBins; i++) {
		if (time_bins[i] == -1) {
			MERR("%d not coverd in time_bins for TimeFeatGenerator\n", i);
			return -1;
		}
	}

	return 0;
}

//.......................................................................................
int TimeFeatGenerator::get_time_unit(string name) {

	boost::algorithm::to_lower(name);

	if (name == "year") time_unit = FTR_TIME_YEAR;
	else if (name == "month") time_unit = FTR_TIME_MONTH;
	else if (name == "day_in_month") time_unit = FTR_TIME_DAY_IN_MONTH;
	else if (name == "day_in_week") time_unit = FTR_TIME_DAY_IN_WEEK;
	else if (name == "hour") time_unit = FTR_TIME_HOUR;
	else if (name == "minute") time_unit = FTR_TIME_MINUTE;
	else if (name == "date") time_unit = FTR_TIME_DATE;

	if (time_unit != FTR_TIME_LAST) {
		set_default_bins();
		return 0;
	}
	else
		return -1;
}

//.......................................................................................
int TimeFeatGenerator::get_nBins() {

	switch (time_unit) {
	case FTR_TIME_YEAR: return 0;
	case FTR_TIME_DATE: return 0;
	case FTR_TIME_MONTH: return 12;
	case FTR_TIME_DAY_IN_MONTH: return 31;
	case FTR_TIME_DAY_IN_WEEK: return 7;
	case FTR_TIME_HOUR: return 24;
	case FTR_TIME_MINUTE: return 60;
	default: return 0;
	}
}

//.......................................................................................
void TimeFeatGenerator::set_default_bins() {

	int nBins = get_nBins();

	time_bins.resize(nBins);
	time_bin_names.resize(nBins);

	for (int i = 0; i < nBins; i++) {
		time_bins[i] = i;
		time_bin_names[i] = to_string(i);
	}
}

// Generate
//.......................................................................................
int TimeFeatGenerator::_generate(PidDynamicRec& rec, MedFeatures& features, int index, int num, vector<float *> &_p_data) {

	float *p_feat = _p_data[0] + index;

	// Special care of year
	if ((time_unit == FTR_TIME_YEAR) || (time_unit == FTR_TIME_DATE)) {
		int dest_type = (time_unit == FTR_TIME_YEAR) ? MedTime::Years : MedTime::Date;
		for (int i = 0; i < num; i++)
		{
			p_feat[i] = med_time_converter.convert_times(features.time_unit, dest_type, features.samples[index + i].time);
			if (time_unit == FTR_TIME_YEAR)
				p_feat[i] += 1900;
		}
		return 0;
	}

	int target_time_unit, mod, shift;
	switch (time_unit) {
	case FTR_TIME_MONTH: target_time_unit = MedTime::Months; mod = 12; shift = 0; break;
	case FTR_TIME_DAY_IN_MONTH: target_time_unit = MedTime::Date; mod = 100; shift = 0; break;
	case FTR_TIME_DAY_IN_WEEK: target_time_unit = MedTime::Days; mod = 7; shift = 1; break; // 01/01/1900 was monday (==1)
	case FTR_TIME_HOUR: target_time_unit = MedTime::Hours; mod = 24; shift = 0; break;
	case FTR_TIME_MINUTE: target_time_unit = MedTime::Minutes; mod = 60; shift = 0; break;
	default:
		MTHROW_AND_ERR("TimeFeatGenerator: Unknown time-unit = %d\n", time_unit);
	}

	for (int i = 0; i < num; i++)
		p_feat[i] = time_bins[(med_time_converter.convert_times(features.time_unit, target_time_unit, features.samples[index + i].time) + shift) % mod];

	return 0;
}

//=======================================================================================
// AttrFeatGenerator: creating a feature from samples' attribute
//=======================================================================================

// Set name
//................................................................................................................
void AttrFeatGenerator::set_names() {
	names.clear();
	names.push_back("FTR_" + int_to_string_digits(serial_id, 6) + ".Attr." + (ftr_name.empty() ? attribute : ftr_name));
}

// Init
//.......................................................................................
int AttrFeatGenerator::init(map<string, string>& mapper) {

	string time_bins_string = "";
	for (auto entry : mapper) {
		string field = entry.first;
		//! [RangeFeatGenerator::init]
		if (field == "attribute") attribute = entry.second;
		else if (field == "name") ftr_name = entry.second;
		else if (field != "fg_type")
			MLOG("Unknown parameter \'%s\' for AttrFeatGenerator\n", field.c_str());
		//! [RangeFeatGenerator::init]
	}

	// set names and required signals
	set_names();

	return 0;
}

// Generate
//.......................................................................................
int AttrFeatGenerator::_generate(PidDynamicRec& rec, MedFeatures& features, int index, int num, vector<float *> &_p_data) {

	float *p_feat = p_data[0] + index;
	for (int i = 0; i < num; i++) {
		if (features.samples[index + i].attributes.find(attribute) != features.samples[index + i].attributes.end())
			p_feat[i] = features.samples[index + i].attributes[attribute];
		else
			p_feat[i] = missing_val;
	}

	return 0;
}

// ModelFeatureGenerator
//=======================================================================================

//................................................................................................................
void ModelFeatGenerator::set_names() {
	MLOG("In ModelFeatGenerator::set_names()\n");
	names.clear();

	string name;
	if (modelName != "")
		name = modelName;
	else if (modelFile != "")
		name = modelFile;
	else
		name = "ModelPred";

	if (impute_existing_feature)
		names.push_back(modelName);
	else {
		for (int t = 0; t < times.size(); t++) {
			for (int i = 0; i < n_preds; i++)
				names.push_back("FTR_" + int_to_string_digits(serial_id, 6) + "." + name + "_t_" + to_string(times[t]) + "." + to_string(i + 1));
		}
	}


}

// Init from map
//.......................................................................................
int ModelFeatGenerator::init(map<string, string>& mapper) {
	//MLOG("In ModelFeatGenerator::init()\n");
	for (auto entry : mapper) {
		string field = entry.first;
		//! [ModelFeatGenerator::init]
		if (field == "name") modelName = entry.second;
		else if (field == "file") modelFile = entry.second;
		else if (field == "impute_existing_feature") impute_existing_feature = med_stoi(entry.second);
		else if (field == "n_preds") n_preds = med_stoi(entry.second);
		else if (field == "model_json") model_json = entry.second;
		else if (field == "model_train_samples") model_train_samples = entry.second;
		else if (field == "ensure_patient_ids") ensure_patient_ids = med_stoi(entry.second) > 0;
		else if (field == "tags") { boost::split(tags, entry.second, boost::is_any_of(",")); }
		else if (field == "time_unit_win") time_unit_win = med_time_converter.string_to_type(entry.second);
		else if (field == "time_unit_sig") time_unit_sig = med_time_converter.string_to_type(entry.second);
		else if (field == "times") {
			vector<string> stringTimes;
			boost::split(stringTimes, entry.second, boost::is_any_of(","));
			for (auto stringTime : stringTimes) times.push_back(med_stoi(stringTime));
		}
		else if (field != "fg_type")
			MTHROW_AND_ERR("Unknown parameter \'%s\' for ModelFeatureGenerator\n", field.c_str());
		//! [ModelFeatGenerator::init]
	}
	if (times.empty()) times.push_back(0); // if no times were given, predict at sample time
	if (impute_existing_feature != 0 && times.size() > 1) // it does not make sense to impute existing features at multiple time points
		MTHROW_AND_ERR("cannot use impute_existing_feature and more than one time in ModelFeatureGenerator\n")
		// set names
		set_names();
	// {name} is magical
	boost::replace_all(modelFile, "{name}", modelName);
	// Read Model and get required signal
	model = new MedModel;
	if (!modelFile.empty()) {
		if (model->read_from_file(modelFile) != 0)
			MTHROW_AND_ERR("Cannot read model from binary file %s\n", modelFile.c_str());
		init_from_model();
	}
	else {
		if (model_json.empty() || model_train_samples.empty())
			MTHROW_AND_ERR("Error must supply trained modelFile or model_json+model_train_samples\n");
		model->init_from_json_file(model_json);
		init_from_model();
	}


	generator_type = FTR_GEN_MODEL;

	return 0;
}

// Copy Model and get required signals
//.......................................................................................
int ModelFeatGenerator::init_from_model() {
	MLOG("In ModelFeatGenerator::init_from_model()\n");
	generator_type = FTR_GEN_MODEL;

	unordered_set<string> required;
	model->get_required_signal_names(required);
	for (string signal : required)
		req_signals.push_back(signal);

	return 0;
}

int ModelFeatGenerator::_learn(MedPidRepository& rep, const MedSamples& samples, vector<RepProcessor *> processors) {
	if (model_json.empty() || model_train_samples.empty() || !modelFile.empty())
		return 0; //no need for learn
	MedSamples train_sample_cp;
	train_sample_cp.read_from_file(model_train_samples);
	MedPidRepository rep_feat;
	vector<int> train_pids, intersection_ids;
	train_sample_cp.get_ids(train_pids);
	if (ensure_patient_ids) {
		samples.get_ids(intersection_ids);
		unordered_set<int> accepted_ids(intersection_ids.begin(), intersection_ids.end());
		vector<int> final_ids;
		final_ids.reserve(min(train_pids.size(), accepted_ids.size()));
		//Do Intersection
		for (size_t i = 0; i < train_pids.size(); ++i)
			if (accepted_ids.find(train_pids[i]) != accepted_ids.end())
				final_ids.push_back(train_pids[i]);
		train_pids = std::move(final_ids);
	}

	model->load_repository(rep.config_fname, train_pids, rep_feat, true);

	//Need to set outcome - use other samples
	int rc = model->learn(rep, &train_sample_cp);
	return rc;
}

/// Load predictions from a MedSamples object. Compare to the models MedSamples (unless empty)
//.......................................................................................
void ModelFeatGenerator::override_predictions(MedSamples& inSamples, MedSamples& modelSamples) {
	MLOG("In ModelFeatGenerator::override_predictions()\n");
	// Note: when using this feature with multiple time points make sure the predictions in the given sample are setup in the correct order
	// Sanity check ...
	if (modelSamples.idSamples.size() && !inSamples.same_as(modelSamples, 0))
		MTHROW_AND_ERR("inSamples is not identical to model samples in ModelFeatGenerator::load\n");
	int t_size = (int)times.size();
	preds.resize(t_size, vector<vector<float>>(n_preds, vector<float>(inSamples.nSamples())));

	for (int t = 0; t < t_size; t++) {
		for (int i = 0; i < n_preds; i++) {
			int idx = 0;
			for (auto& idSamples : inSamples.idSamples) {
				for (auto& sample : idSamples.samples) {
					if (sample.prediction.size() < static_cast<size_t>(n_preds) * static_cast<size_t>(t_size))
						MTHROW_AND_ERR("Cannot extract %d predictions from sample in ModelFeatGenerator::load\n", n_preds*t_size);
					preds[t][i][idx++] = sample.prediction[t*n_preds + i];

				}
			}
		}
	}

	use_overriden_predictions = 1;

}

void ModelFeatGenerator::modifySampleTime(MedSamples& samples, int time) {
	for (int s = 0; s < samples.idSamples.size(); s++) {
		for (int m = 0; m < samples.idSamples[s].samples.size(); m++) {
			samples.idSamples[s].samples[m].time += time;
		}

	}
}

// Do the actual prediction prior to feature generation , only if vector is empty
//.......................................................................................
void ModelFeatGenerator::prepare(MedFeatures & features, MedPidRepository& rep, MedSamples& samples) {
	MLOG("In ModelFeatGenerator::prepare()\n");
	if (!use_overriden_predictions) {
		MedSamples modifiedSamples = samples;
		int t_size = (int)times.size();
		preds.resize(t_size);
		for (int t = 0; t < t_size; t++) {
			// modify sample time
			int sig_time = med_time_converter.convert_times(time_unit_win, time_unit_sig, times[t]);
			modifySampleTime(modifiedSamples, -sig_time);

			// Predict
			if (model->apply(rep, modifiedSamples, MED_MDL_APPLY_FTR_GENERATORS, MED_MDL_APPLY_PREDICTOR) != 0)
				MTHROW_AND_ERR("ModelFeatGenerator::prepare feature %s failed to apply model\n", modelName.c_str());

			// Extract predictions
			if (model->features.samples[0].prediction.size() < n_preds)
				MTHROW_AND_ERR("ModelFeatGenerator::prepare cannot generate feature %s\n", modelName.c_str());
			preds[t].resize(n_preds, vector<float>(model->features.samples.size()));
			for (int i = 0; i < model->features.samples.size(); i++) {
				for (int j = 0; j < n_preds; j++) {
					float new_val = model->features.samples[i].prediction[j];
					if (!isfinite(new_val))
						MTHROW_AND_ERR("ModelFeatGenerator::prepare feature %s nan in row %d\n", modelName.c_str(), i);
					preds[t][j][i] = new_val;
				}
			}

			// release some memory
			model->features.clear();
			modifySampleTime(modifiedSamples, +sig_time);
		}
	}
	// Generate a new feature or identify feature to impute.
	if (impute_existing_feature) {
		FeatureProcessor p;
		string res = p.resolve_feature_name(features, modelName);
		names.clear();
		names.push_back(res);
	}
	else {
		FeatureGenerator::prepare(features, rep, samples);
	}
}

// Put relevant predictions in place
//.......................................................................................
int ModelFeatGenerator::_generate(PidDynamicRec& rec, MedFeatures& features, int index, int num, vector<float *> &_p_data) {
	for (int t = 0; t < times.size(); t++) {
		for (int n = 0; n < n_preds; n++) {
			float *p_feat = _p_data[t*n_preds + n] + index;
			for (int i = 0; i < num; i++) {
				if (!impute_existing_feature || p_feat[i] == missing_val) {
					float new_val = preds[t][n][index + i];
					if (!isfinite(new_val))
						MTHROW_AND_ERR("ModelFeatGenerator::_generate nan in row %d\n", index + i);
					p_feat[i] = new_val;
				}
			}
		}
	}

	return 0;

}


ModelFeatGenerator::~ModelFeatGenerator() {
	MLOG("In ModelFeatGenerator::~ModelFeatGenerator()\n");
	if (model != NULL) delete model;
	model = NULL;
}
ADD_SERIALIZATION_FUNCS_CPP(ModelFeatGenerator, generator_type, tags, modelFile, model, modelName, n_preds, names, req_signals, impute_existing_feature, use_overriden_predictions, time_unit_win, time_unit_sig, times)
//................................................................................................................
// Helper function for time conversion
//................................................................................................................
void get_window_in_sig_time(int _win_from, int _win_to, int _time_unit_win, int _time_unit_sig, int _win_time, int &_min_time, int &_max_time,
	bool boundOutcomeTime, int outcome_time)
{
	if (_win_time - _win_to > 0)
		_min_time = med_time_converter.convert_times(_time_unit_win, _time_unit_sig, _win_time - _win_to);
	else
		_min_time = 0;
	if (_win_time - _win_from > 0)
		_max_time = med_time_converter.convert_times(_time_unit_win, _time_unit_sig, _win_time - _win_from);
	else
		_max_time = 0;
	if (boundOutcomeTime && outcome_time < _max_time)
		_max_time = outcome_time;
}

// Helper functions for time-range signals
//.......................................................................................
string time_range_type_to_name(TimeRangeTypes type)
{
	if (type == TIME_RANGE_CURRENT) return "current";
	if (type == TIME_RANGE_BEFORE) return "before";

	MTHROW_AND_ERR("unknown type [%d]\n", (int)type);
}

TimeRangeTypes time_range_name_to_type(const string& name) {

	if (name == "current")			return TIME_RANGE_CURRENT;
	if (name == "before")			return TIME_RANGE_BEFORE;

	if (isInteger(name))
		return (TimeRangeTypes)med_stoi(name);
	else
		MTHROW_AND_ERR("unknown name [%s]\n", name.c_str());
}

//.......................................................................................
// update time window according to time-range signal
void get_updated_time_window(UniversalSigVec& time_range_usv, TimeRangeTypes type, int time_unit_range_sig, int time_unit_win, int time_unit_sig, int time,
	int win_from, int& updated_win_from, int win_to, int& updated_win_to, bool delta_win, int d_win_from, int& updated_d_win_from, int d_win_to, int& updated_d_win_to) {

	// Identify relevant range
	int range_from = -1, range_to = -1;
	int time_to_check = med_time_converter.convert_times(time_unit_range_sig, time_unit_win, med_time_converter.convert_times(time_unit_sig, time_unit_win, time));

	for (int i = 0; i < time_range_usv.len; i++) {
		int fromTime = time_range_usv.Time(i, 0);
		int toTime = time_range_usv.Time(i, 1);

		if (fromTime > time_to_check)
			break;
		else if (toTime >= time_to_check) {
			range_from = fromTime;
			range_to = toTime;
			break;
		}
	}

	// Handle cases
	get_updated_time_window(type, range_from, range_to, time_to_check, win_from, win_to, updated_win_from, updated_win_to);
	if (delta_win)
		get_updated_time_window(type, range_from, range_to, time_to_check, d_win_from, d_win_to, updated_d_win_from, updated_d_win_to);
}

void get_updated_time_window(TimeRangeTypes type, int range_from, int range_to, int time, int _win_from, int _win_to, int& updated_win_from, int& updated_win_to) {

	if (type == TIME_RANGE_CURRENT) {
		// Intersection is empty
		if (range_from == -1 || _win_from > time - range_from || _win_to < time - range_to) {
			updated_win_from = _win_from;
			updated_win_to = updated_win_from - 1; // Empty window ...
		}
		else {
			if (_win_to > time - range_from) // win_to points before range
				updated_win_to = time - range_from;
			else
				updated_win_to = _win_to;

			if (_win_from < time - range_to) // win_from is negative (points to the future) and after range-to
				updated_win_from = time - range_to;
			else
				updated_win_from = _win_from;
		}
	}
	else if (type == TIME_RANGE_BEFORE) {
		if (range_from == -1) {
			updated_win_from = _win_from;
			updated_win_to = _win_to;
		}
		else {
			if (_win_from < time - range_from)
				updated_win_from = time - range_from;
			else
				updated_win_from = _win_from;

			if (_win_to < time - range_from)
				updated_win_to = time - range_from;
			else
				updated_win_to = _win_to;

		}
	}
}

