#define _CRT_SECURE_NO_WARNINGS

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_REPCLEANER
#define LOCAL_LEVEL LOG_DEF_LEVEL

#include "RepProcess.h"
#include <MedUtils/MedUtils/MedUtils.h>
#include "RepCreateRegistry.h"
#include "RepCategoryDescenders.h"
#include "RepFilterByChannels.h"
#include "RepClearSignalByDiag.h"
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <cassert>
#include <iostream>

//=======================================================================================
// RepProcessors
//=======================================================================================
// Processors types
RepProcessorTypes rep_processor_name_to_type(const string &processor_name)
{

	if (processor_name == "multi_processor" || processor_name == "multi")
		return REP_PROCESS_MULTI;
	else if (processor_name == "basic_outlier_cleaner" || processor_name == "basic_cln")
		return REP_PROCESS_BASIC_OUTLIER_CLEANER;
	else if (processor_name == "nbrs_outlier_cleaner" || processor_name == "nbrs_cln")
		return REP_PROCESS_NBRS_OUTLIER_CLEANER;
	else if (processor_name == "configured_outlier_cleaner" || processor_name == "conf_cln")
		return REP_PROCESS_CONFIGURED_OUTLIER_CLEANER;
	else if (processor_name == "rulebased_outlier_cleaner" || processor_name == "rule_cln")
		return REP_PROCESS_RULEBASED_OUTLIER_CLEANER;
	else if (processor_name == "calc_signals" || processor_name == "calculator")
		return REP_PROCESS_CALC_SIGNALS;
	else if (processor_name == "complete")
		return REP_PROCESS_COMPLETE;
	else if (processor_name == "req" || processor_name == "requirements")
		return REP_PROCESS_CHECK_REQ;
	else if (processor_name == "sim_val" || processor_name == "sim_val_handler")
		return REP_PROCESS_SIM_VAL;
	else if (processor_name == "signal_rate")
		return REP_PROCESS_SIGNAL_RATE;
	else if (processor_name == "combine")
		return REP_PROCESS_COMBINE;
	else if (processor_name == "split")
		return REP_PROCESS_SPLIT;
	else if (processor_name == "aggregation_period")
		return REP_PROCESS_AGGREGATION_PERIOD;
	else if (processor_name == "basic_range_cleaner" || processor_name == "range_cln")
		return REP_PROCESS_BASIC_RANGE_CLEANER;
	else if (processor_name == "aggregate")
		return REP_PROCESS_AGGREGATE;
	else if (processor_name == "limit_history" || processor_name == "history_limit")
		return REP_PROCESS_HISTORY_LIMIT;
	else if (processor_name == "create_registry")
		return REP_PROCESS_CREATE_REGISTRY;
	else if (processor_name == "bit_signal")
		return REP_PROCESS_CREATE_BIT_SIGNAL;
	else if (processor_name == "category_descenders")
		return REP_PROCESS_CATEGORY_DESCENDERS;
	else if (processor_name == "reorder_channels")
		return REP_PROCESS_REODER_CHANNELS;
	else if (processor_name == "filter_channels")
		return REP_PROCESS_FILTER_BY_CHANNELS;
	else if (processor_name == "numeric_noiser" || processor_name == "noiser")
		return REP_PROCESS_NUMERIC_NOISER;
	else if (processor_name == "filter_by_diag")
		return REP_PROCESS_FILTER_BY_DIAG;
	else
		return REP_PROCESS_LAST;
}

// rep processors get a new derived class
//.......................................................................................
void *RepProcessor::new_polymorphic(string dname)
{
	CONDITIONAL_NEW_CLASS(dname, RepMultiProcessor);
	CONDITIONAL_NEW_CLASS(dname, RepBasicOutlierCleaner);
	CONDITIONAL_NEW_CLASS(dname, RepNbrsOutlierCleaner);
	CONDITIONAL_NEW_CLASS(dname, RepConfiguredOutlierCleaner);
	CONDITIONAL_NEW_CLASS(dname, RepRuleBasedOutlierCleaner);
	CONDITIONAL_NEW_CLASS(dname, RepCalcSimpleSignals);
	CONDITIONAL_NEW_CLASS(dname, RepPanelCompleter);
	CONDITIONAL_NEW_CLASS(dname, RepCheckReq);
	CONDITIONAL_NEW_CLASS(dname, RepSimValHandler);
	CONDITIONAL_NEW_CLASS(dname, RepSignalRate);
	CONDITIONAL_NEW_CLASS(dname, RepCombineSignals);
	CONDITIONAL_NEW_CLASS(dname, RepSplitSignal);
	CONDITIONAL_NEW_CLASS(dname, RepAggregationPeriod);
	CONDITIONAL_NEW_CLASS(dname, RepBasicRangeCleaner);
	CONDITIONAL_NEW_CLASS(dname, RepAggregateSignal);
	CONDITIONAL_NEW_CLASS(dname, RepHistoryLimit);
	CONDITIONAL_NEW_CLASS(dname, RepCreateRegistry);
	CONDITIONAL_NEW_CLASS(dname, RepCreateBitSignal);
	CONDITIONAL_NEW_CLASS(dname, RepCategoryDescenders);
	CONDITIONAL_NEW_CLASS(dname, RepReoderChannels);
	CONDITIONAL_NEW_CLASS(dname, RepFilterByChannel);
	CONDITIONAL_NEW_CLASS(dname, RepNumericNoiser);
	CONDITIONAL_NEW_CLASS(dname, RepClearSignalByDiag);
	MWARN("Warning in RepProcessor::new_polymorphic - Unsupported class %s\n", dname.c_str());
	return NULL;
}

// Create processor from params string (type must be given within string)
//.......................................................................................
RepProcessor *RepProcessor::create_processor(string &params)
{
	string rp_type;
	get_single_val_from_init_string(params, "rp_type", rp_type);
	return (make_processor(rp_type, params));
}

// Initialization : given processor name
//.......................................................................................
RepProcessor *RepProcessor::make_processor(string processor_name)
{

	return make_processor(rep_processor_name_to_type(processor_name));
}

// Initialization : given processor name and intialization string
//.......................................................................................
RepProcessor *RepProcessor::make_processor(string processor_name, string init_string)
{

	return make_processor(rep_processor_name_to_type(processor_name), init_string);
}

// Initialization : given processor type
//.......................................................................................
RepProcessor *RepProcessor::make_processor(RepProcessorTypes processor_type)
{

	if (processor_type == REP_PROCESS_MULTI)
		return new RepMultiProcessor;
	else if (processor_type == REP_PROCESS_BASIC_OUTLIER_CLEANER)
		return new RepBasicOutlierCleaner;
	else if (processor_type == REP_PROCESS_NBRS_OUTLIER_CLEANER)
		return new RepNbrsOutlierCleaner;
	else if (processor_type == REP_PROCESS_CONFIGURED_OUTLIER_CLEANER)
		return new RepConfiguredOutlierCleaner;
	else if (processor_type == REP_PROCESS_RULEBASED_OUTLIER_CLEANER)
		return new RepRuleBasedOutlierCleaner;
	else if (processor_type == REP_PROCESS_CALC_SIGNALS)
		return new RepCalcSimpleSignals;
	else if (processor_type == REP_PROCESS_COMPLETE)
		return new RepPanelCompleter;
	else if (processor_type == REP_PROCESS_CHECK_REQ)
		return new RepCheckReq;
	else if (processor_type == REP_PROCESS_SIM_VAL)
		return new RepSimValHandler;
	else if (processor_type == REP_PROCESS_SIGNAL_RATE)
		return new RepSignalRate;
	else if (processor_type == REP_PROCESS_COMBINE)
		return new RepCombineSignals;
	else if (processor_type == REP_PROCESS_SPLIT)
		return new RepSplitSignal;
	else if (processor_type == REP_PROCESS_AGGREGATION_PERIOD)
		return new RepAggregationPeriod;
	else if (processor_type == REP_PROCESS_BASIC_RANGE_CLEANER)
		return new RepBasicRangeCleaner;
	else if (processor_type == REP_PROCESS_AGGREGATE)
		return new RepAggregateSignal;
	else if (processor_type == REP_PROCESS_HISTORY_LIMIT)
		return new RepHistoryLimit;
	else if (processor_type == REP_PROCESS_CREATE_REGISTRY)
		return new RepCreateRegistry;
	else if (processor_type == REP_PROCESS_CREATE_BIT_SIGNAL)
		return new RepCreateBitSignal;
	else if (processor_type == REP_PROCESS_CATEGORY_DESCENDERS)
		return new RepCategoryDescenders;
	else if (processor_type == REP_PROCESS_REODER_CHANNELS)
		return new RepReoderChannels;
	else if (processor_type == REP_PROCESS_FILTER_BY_CHANNELS)
		return new RepFilterByChannel;
	else if (processor_type == REP_PROCESS_NUMERIC_NOISER)
		return new RepNumericNoiser;
	else if (processor_type == REP_PROCESS_FILTER_BY_DIAG)
		return new RepClearSignalByDiag;
	else
		return NULL;
}

// Initialization : given processor type and intialization string
//.......................................................................................
RepProcessor *RepProcessor::make_processor(RepProcessorTypes processor_type, string init_string)
{

	// MLOG("Processor type is %d init_string is %s\n", (int)processor_type, init_string.c_str());
	RepProcessor *newRepProcessor = make_processor(processor_type);
	if (newRepProcessor->init_from_string(init_string) < 0)
		MTHROW_AND_ERR("Cannot init RepProcessor of type %d with init string \'%s\'\n", processor_type, init_string.c_str());

	return newRepProcessor;
}

// learn on all pids in repository, using fake samples - works only for repProcessors that ignore sample dates
//.......................................................................................
int RepProcessor::learn(MedPidRepository &rep)
{
	MedSamples fakeSamples;
	for (int pid : rep.pids)
		fakeSamples.insertRec(pid, 0);
	this->learn(rep, fakeSamples);
	return 0;
}

// Learn processing parameters only if affecting any of the signals given in neededSignalIds
//.......................................................................................
int RepProcessor::_conditional_learn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_processors, unordered_set<int> &neededSignalIds)
{
	for (int signalId : neededSignalIds)
	{
		if (is_signal_affected(signalId))
			return _learn(rep, samples, prev_processors);
	}
	return 0;
}

// Check if processor can be filtered
//...................................
bool RepProcessor::filter(unordered_set<string> &neededSignals)
{

	if (unconditional)
		return false;

	for (string signal : neededSignals)
	{
		if (is_signal_affected(signal))
			return false;
	}

	MLOG_D("RepProcessor::filter filtering out processor of type %d(%s), affected signals: ", processor_type, my_class_name().c_str());
	for (string signal : aff_signals)
		MLOG_D("[%s] ", signal.c_str());
	MLOG_D("\n");

	return true;
}

// Apply processing on a single PidDynamicRec at a set of time-points given by samples
//.......................................................................................
int RepProcessor::apply(PidDynamicRec &rec, MedIdSamples &samples)
{

	vector<int> time_points(samples.samples.size());
	for (unsigned int i = 0; i < time_points.size(); i++)
		time_points[i] = samples.samples[i].time;

	vector<vector<float>> attributes_mat(time_points.size(), vector<float>(attributes.size(), 0));
	int rc = apply(rec, time_points, attributes_mat);

	if (rc == 0)
	{
		for (unsigned int i = 0; i < time_points.size(); i++)
		{
			for (int j = 0; j < attributes.size(); j++)
				samples.samples[i].attributes[attributes[j]] += attributes_mat[i][j];
		}
	}

	return rc;
}

// Apply processing on a single PidDynamicRec at a set of time-points given by samples,
// only if affecting any of the signals given in neededSignalIds
//.......................................................................................
int RepProcessor::conditional_apply(PidDynamicRec &rec, MedIdSamples &samples, unordered_set<int> &neededSignalIds)
{

	vector<int> time_points;
	samples.get_times(time_points);

	vector<vector<float>> attributes_mat(time_points.size(), vector<float>(attributes.size(), 0));
	int rc = conditional_apply(rec, time_points, neededSignalIds, attributes_mat);

	if (rc == 0)
	{
		for (unsigned int i = 0; i < time_points.size(); i++)
		{
			for (int j = 0; j < attributes.size(); j++)
			{
				samples.samples[i].attributes[attributes[j]] += attributes_mat[i][j];
			}
		}
	}

	return rc;
}

// Apply processing on a single PidDynamicRec at a set of time-points given by samples,
// only if affecting any of the signals given in neededSignalIds. Do not affect attributes
//.......................................................................................
int RepProcessor::conditional_apply_without_attributes(PidDynamicRec &rec, const MedIdSamples &samples, unordered_set<int> &neededSignalIds)
{

	vector<int> time_points;
	samples.get_times(time_points);

	vector<vector<float>> attributes_mat(time_points.size(), vector<float>(attributes.size(), 0));
	return conditional_apply(rec, time_points, neededSignalIds, attributes_mat);
}

// Apply processing on a single PidDynamicRec at a set of time-points given by time-points,
// only if affecting any of the signals given in neededSignalIds
//.......................................................................................
int RepProcessor::_conditional_apply(PidDynamicRec &rec, vector<int> &time_points, unordered_set<int> &neededSignalIds, vector<vector<float>> &attributes_mat)
{

	if (unconditional)
		return apply(rec, time_points, attributes_mat);

	for (int signalId : neededSignalIds)
	{
		if (is_signal_affected(signalId))
			return apply(rec, time_points, attributes_mat);
	}

	return 0;
}

// Fill req_signal_ids from req_signals
//.......................................................................................
void RepProcessor::set_required_signal_ids(MedDictionarySections &dict)
{

	for (string signal : req_signals)
		req_signal_ids.insert(dict.id(signal));
}

// Add req_signals to set
//.......................................................................................
void RepProcessor::get_required_signal_names(unordered_set<string> &signalNames)
{

	for (auto sig : req_signals)
		signalNames.insert(sig);
}

// Add req_signals to set only if processor is required for any of preReqSignalNames
//.......................................................................................
void RepProcessor::get_required_signal_names(unordered_set<string> &signalNames, unordered_set<string> preReqSignalNames)
{

	if (unconditional)
		get_required_signal_names(signalNames);
	else
	{
		for (string signal : preReqSignalNames)
		{
			if (is_signal_affected(signal))
			{
				get_required_signal_names(signalNames);
				return;
			}
		}
	}
}

//.......................................................................................
void RepProcessor::get_required_signal_ids(unordered_set<int> &signalIds)
{

	for (auto sig : req_signal_ids)
		signalIds.insert(sig);
}

// Add req_signals to set only if processor is required for any of preReqSignalNames
//.......................................................................................
void RepProcessor::get_required_signal_ids(unordered_set<int> &signalIds, unordered_set<int> preReqSignals)
{

	for (int signal : preReqSignals)
	{
		if (is_signal_affected(signal))
		{
			get_required_signal_ids(signalIds);
			return;
		}
	}
}

// Affected signals - set aff_signal_ids from aff_signals (id->name)
//.......................................................................................
void RepProcessor::set_affected_signal_ids(MedDictionarySections &dict)
{

	for (string signalName : aff_signals)
		aff_signal_ids.insert(dict.id(signalName));
}

//.......................................................................................
void RepProcessor::dprint(const string &pref, int rp_flag)
{
	if (rp_flag > 0)
	{
		MLOG("%s :: RP type %d(%s) : required(%d): ", pref.c_str(), processor_type, my_class_name().c_str(), req_signals.size());
		if (rp_flag > 1)
			for (auto &rsig : req_signals)
				MLOG("%s,", rsig.c_str());
		MLOG(" affected(%d): ", aff_signals.size());
		if (rp_flag > 1)
			for (auto &asig : aff_signals)
				MLOG("%s, ", asig.c_str());
		MLOG(" virtual(%d): ", virtual_signals.size());
		if (rp_flag > 1)
			for (auto &vsig : virtual_signals)
				MLOG("%s ", vsig.first.c_str());
		MLOG("\n");
	}
}

// (De)Serialize
//.......................................................................................
size_t RepProcessor::get_processor_size()
{
	return sizeof(processor_type) + get_size();
}

//.......................................................................................
size_t RepProcessor::processor_serialize(unsigned char *blob)
{

	size_t ptr = 0;
	memcpy(blob + ptr, &processor_type, sizeof(RepProcessorTypes));
	ptr += sizeof(RepProcessorTypes);
	ptr += serialize(blob + ptr);

	return ptr;
}

//=======================================================================================
// RepMultiProcessor
//=======================================================================================
//.......................................................................................
void RepMultiProcessor::clear()
{
	for (auto p : processors)
	{
		if (p != NULL)
		{
			delete p;
			p = NULL;
		}
	}
	processors.clear();
}

// Required Signals ids : Fill the member vector - req_signal_ids
//.......................................................................................
void RepMultiProcessor::set_required_signal_ids(MedDictionarySections &dict)
{

	req_signal_ids.clear();
	for (auto &processor : processors)
	{
		req_signal_ids.clear();
		processor->set_required_signal_ids(dict);

		for (int signalId : processor->req_signal_ids)
			req_signal_ids.insert(signalId);
	}
}

// Affected Signals : Fill the member vector aff_signal_ids
//.......................................................................................
void RepMultiProcessor::set_affected_signal_ids(MedDictionarySections &dict)
{

	aff_signal_ids.clear();
	for (auto &processor : processors)
	{
		processor->aff_signal_ids.clear();
		processor->set_affected_signal_ids(dict);

		for (int signalId : processor->aff_signal_ids)
			aff_signal_ids.insert(signalId);
	}
}

// Check if processor can be filtered
//...................................
bool RepMultiProcessor::filter(unordered_set<string> &neededSignals)
{

	vector<RepProcessor *> filtered;
	bool did_something = false;
	for (auto &processor : processors)
	{
		if (!processor->filter(neededSignals))
			filtered.push_back(processor);
		else
		{
			delete processor;
			processor = NULL;
			did_something = true;
		}
	}
	if (did_something)
		MLOG_D("Filtering uneeded rep_processors in RepMultiProcessor. left with %zu processors out of %zu\n",
			   filtered.size(), processors.size());

	if (filtered.empty())
	{
		MLOG_D("RepMultiProcessor::filter filtering out processor of type %d(%s)\n", processor_type, my_class_name().c_str());
		processors.clear();
		return true;
	}
	else
	{
		processors.swap(filtered);
		return false;
	}
}

// Make changes to RepProcessor according to available signals in Repository
//.......................................................................................
void RepMultiProcessor::fit_for_repository(MedPidRepository &rep)
{

	for (auto &processor : processors)
		processor->fit_for_repository(rep);
}

// Set signal-ids for all linked signals
//.......................................................................................
void RepMultiProcessor::set_signal_ids(MedSignals &sigs)
{

	for (auto &processor : processors)
		processor->set_signal_ids(sigs);
}

// Required Signals names : Fill the unordered set signalNames
//.......................................................................................
void RepMultiProcessor::get_required_signal_names(unordered_set<string> &signalNames)
{
	for (auto &processor : processors)
		processor->get_required_signal_names(signalNames);
}

// Add req_signals to set only if processor is required for any of preReqSignalNames
// Note that preReq is copied so it is not affected by enlarging signalNames
//.......................................................................................
void RepMultiProcessor::get_required_signal_names(unordered_set<string> &signalNames, unordered_set<string> preReqSignalNames)
{
	for (auto &processor : processors)
		processor->get_required_signal_names(signalNames, preReqSignalNames);
}

// Required Signals names : Fill the unordered set signalIds
//.......................................................................................
void RepMultiProcessor::get_required_signal_ids(unordered_set<int> &signalIds)
{
	for (auto &processor : processors)
		processor->get_required_signal_ids(signalIds);
}

// Virtual Signals names : Get the virtual signals map
//.......................................................................................
void RepMultiProcessor::add_virtual_signals(map<string, int> &_virtual_signals, map<string, string> &_virtual_signals_generic) const
{
	for (auto &processor : processors)
		processor->add_virtual_signals(_virtual_signals, _virtual_signals_generic);
}

// Add req_signals to set only if processor is required for any of preReqSignalNames
// Note that preReq is copied so it is not affected by enlarging signalNames
//.......................................................................................
void RepMultiProcessor::get_required_signal_ids(unordered_set<int> &signalIds, unordered_set<int> preReqSignals)
{
	for (auto &processor : processors)
		processor->get_required_signal_ids(signalIds, preReqSignals);
}

// Learn processors
//.......................................................................................
int RepMultiProcessor::_learn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_processors)
{

	vector<int> rc(processors.size(), 0);

#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < processors.size(); j++)
	{
		rc[j] = processors[j]->learn(rep, samples, prev_processors);
	}

	for (int r : rc)
		if (r < 0)
			return -1;
	return 0;
}

// Learn processing parameters only if affecting any of the signals given in neededSignalIds
//.......................................................................................
int RepMultiProcessor::_conditional_learn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_processors, unordered_set<int> &neededSignalIds)
{

	vector<int> rc(processors.size(), 0);

#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < processors.size(); j++)
	{
		rc[j] = processors[j]->conditional_learn(rep, samples, prev_processors, neededSignalIds);
	}

	for (int r : rc)
		if (r < 0)
			return -1;
	return 0;
}

// Apply processors
//.......................................................................................
int RepMultiProcessor::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	vector<int> rc(processors.size(), 0);

	// If attributes are required, prepare space for collecting them
	vector<vector<vector<float>>> all_attributes_mats(processors.size());
	if (!attributes_mat.empty())
	{
		all_attributes_mats.resize(processors.size());
		for (int j = 0; j < processors.size(); j++)
		{
			all_attributes_mats[j].resize(time_points.size());
			for (int i = 0; i < time_points.size(); i++)
				all_attributes_mats[j][i].resize(processors[j]->attributes.size(), 0);
		}
	}

	// ??? chances are this next parallelization is not needed, as we parallel before on recs...
	// NO PARALLEL - we are using rec.usv inside processors, generators which is not thread safe!
	for (int j = 0; j < processors.size(); j++)
	{
		rc[j] = processors[j]->apply(rec, time_points, all_attributes_mats[j]);
	}

	for (int r : rc)
		if (r < 0)
			return -1;

	// If attributes are required, collect them
	if (!attributes_mat.empty())
	{
		for (int j = 0; j < processors.size(); j++)
		{
			for (int i = 0; i < time_points.size(); i++)
			{
				for (int k = 0; k < processors[j]->attributes.size(); k++)
					attributes_mat[i][attributes_map[j][k]] += all_attributes_mats[j][i][k];
			}
		}
	}

	return 0;
}

//.......................................................................................
int RepMultiProcessor::_apply_simple(PidDynamicRec &rec, vector<int> &time_points)
{
	for (auto p : processors)
	{
		if ((p->_apply_simple(rec, time_points)) < 0)
			return -1;
	}
	return 0;
}

// Apply processors that affect any of the needed signals
//.......................................................................................
int RepMultiProcessor::_conditional_apply(PidDynamicRec &rec, vector<int> &time_points, unordered_set<int> &neededSignalIds, vector<vector<float>> &attributes_mat)
{

	vector<int> rc(processors.size(), 0);

	// If attributes are required, prepare space for collecting them
	vector<vector<vector<float>>> all_attributes_mats(processors.size());
	if (!attributes_mat.empty())
	{
		all_attributes_mats.resize(processors.size());
		for (int j = 0; j < processors.size(); j++)
		{
			all_attributes_mats[j].resize(time_points.size());
			for (int i = 0; i < time_points.size(); i++)
				all_attributes_mats[j][i].resize(processors[j]->attributes.size(), 0);
		}
	}

	// ??? chances are this next parallelization is not needed, as we parallel before on recs...
	// Can't parallel - we may use rec.usv in procesors
	// #pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < processors.size(); j++)
	{
		if (unconditional)
			rc[j] = processors[j]->apply(rec, time_points, all_attributes_mats[j]);
		else
			rc[j] = processors[j]->conditional_apply(rec, time_points, neededSignalIds, all_attributes_mats[j]);
	}

	for (int r : rc)
		if (r < 0)
			return -1;

	// If attributes are required, collect them
	if (!attributes_mat.empty())
	{
		for (int j = 0; j < processors.size(); j++)
		{
			for (int i = 0; i < time_points.size(); i++)
			{
				for (int k = 0; k < processors[j]->attributes.size(); k++)
					attributes_mat[i][attributes_map[j][k]] += all_attributes_mats[j][i][k];
			}
		}
	}

	return 0;
}

// Add processors
//.......................................................................................
void RepMultiProcessor::add_processors_set(RepProcessorTypes type, vector<string> &signals)
{

	for (string &signal : signals)
	{
		RepProcessor *processor = RepProcessor::make_processor(type);
		processor->set_signal(signal);
		processors.push_back(processor);
	}
}

// Add processors with initialization string
//.......................................................................................
void RepMultiProcessor::add_processors_set(RepProcessorTypes type, vector<string> &signals, string init_string)
{

	for (string &signal : signals)
	{
		RepProcessor *processor = RepProcessor::make_processor(type, init_string);
		processor->set_signal(signal);
		processors.push_back(processor);
	}
}

// init attributes list and attributes map
//.......................................................................................
void RepMultiProcessor::init_attributes()
{

	attributes.clear();
	map<string, int> attributes_pos;

	attributes_map.resize(processors.size());
	for (int i = 0; i < processors.size(); i++)
	{
		processors[i]->init_attributes();
		attributes_map[i].resize(processors[i]->attributes.size());
		for (int j = 0; j < processors[i]->attributes.size(); j++)
		{
			if (attributes_pos.find(processors[i]->attributes[j]) == attributes_pos.end())
			{
				attributes.push_back(processors[i]->attributes[j]);
				attributes_pos[attributes.back()] = (int)attributes.size() - 1;
			}
			attributes_map[i][j] = attributes_pos[processors[i]->attributes[j]];
		}
	}
}

//.......................................................................................
void RepMultiProcessor::dprint(const string &pref, int rp_flag)
{
	if (rp_flag > 0)
	{
		MLOG("%s :: RP MULTI(%d) -->\n", pref.c_str(), processors.size());
		int ind = 0;
		for (auto &proc : processors)
		{
			proc->dprint("\t" + pref + "->Multi[" + to_string(ind) + "]", rp_flag);
			++ind;
		}
	}
}

void RepMultiProcessor::make_summary()
{
	for (auto &proc : processors)
		proc->make_summary();
}

void RepMultiProcessor::get_required_signal_categories(unordered_map<string, vector<string>> &signal_categories_in_use) const
{
	for (auto &proc : processors)
	{
		unordered_map<string, vector<string>> local_use;
		proc->get_required_signal_categories(local_use);
		// do merge with signal_categories_in_use:
		for (auto &it : local_use)
		{
			if (signal_categories_in_use.find(it.first) == signal_categories_in_use.end())
				signal_categories_in_use[it.first] = std::move(it.second);
			else
			{
				// merge with existing:
				unordered_set<string> existing_sets(signal_categories_in_use.at(it.first).begin(),
													signal_categories_in_use.at(it.first).end());
				existing_sets.insert(it.second.begin(), it.second.end());
				;
				vector<string> uniq_vec(existing_sets.begin(), existing_sets.end());
				signal_categories_in_use[it.first] = std::move(uniq_vec);
			}
		}
	}
}

void RepMultiProcessor::register_virtual_section_name_id(MedDictionarySections &dict)
{
	for (size_t i = 0; i < processors.size(); ++i)
		processors[i]->register_virtual_section_name_id(dict);
}

void RepBasicOutlierCleaner::set_signal_ids(MedSignals &sigs)
{
	signalId = sigs.sid(signalName);
	is_categ = sigs.is_categorical_channel(signalId, val_channel);
	if (is_categ)
		MWARN("Warning Signal %s is categorical - no cleaning\n", signalName.c_str());
}

//=======================================================================================
// BasicOutlierCleaner
//=======================================================================================
// Fill req- and aff-signals vectors
//.......................................................................................
void RepBasicOutlierCleaner::init_lists()
{

	req_signals.insert(signalName);
	aff_signals.insert(signalName);

	if (!verbose_file.empty() && !log_file.is_open())
	{
		log_file.open(verbose_file, ios::app);
		if (!log_file.good())
			MWARN("Warnning in RepRuleBasedOutlierCleaner - verbose_file %s can't be opened\n", verbose_file.c_str());
	}
}

// Init from map
//.......................................................................................
int RepBasicOutlierCleaner::init(map<string, string> &mapper)
{
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepBasicOutlierCleaner::init]
		if (field == "signal")
		{
			signalName = entry.second;
		}
		else if (field == "time_channel")
			time_channel = med_stoi(entry.second);
		else if (field == "val_channel")
			val_channel = med_stoi(entry.second);
		else if (field == "nrem_attr")
			nRem_attr = entry.second;
		else if (field == "ntrim_attr")
			nTrim_attr = entry.second;
		else if (field == "nrem_suff")
			nRem_attr_suffix = entry.second;
		else if (field == "ntrim_suff")
			nTrim_attr_suffix = entry.second;
		else if (field == "verbose_file")
			verbose_file = entry.second;
		else if (field == "unconditional")
			unconditional = stoi(entry.second) > 0;
		else if (field == "print_summary")
			print_summary = stoi(entry.second) > 0;
		else if (field == "print_summary_critical_cleaned")
			print_summary_critical_cleaned = stof(entry.second);
		else if (field == "rp_type")
		{
		}
		//! [RepBasicOutlierCleaner::init]
	}

	if (!verbose_file.empty())
	{
		verbose_file += "." + signalName + (val_channel == 0 ? "" : "_ch_" + to_string(val_channel));
		ofstream fw(verbose_file);
		fw.close(); // rewrite empty file
	}

	init_lists();
	map<string, string> &mapper_p = mapper;
	vector<string> remove_fl = {"verbose_file", "fp_type", "rp_type", "unconditional", "signal", "time_channel", "val_channel", "nrem_attr", "ntrim_attr", "nrem_suff",
								"ntrim_suff", "time_unit", "nbr_time_unit", "nbr_time_width", "tag", "conf_file", "clean_method", "signals",
								"addRequiredSignals", "consideredRules", "print_summary", "print_summary_critical_cleaned"};

	for (const string &fl : remove_fl)
		if (mapper_p.find(fl) != mapper_p.end())
			mapper_p.erase(fl);
	return MedValueCleaner::init(mapper_p);
}

// init attributes list
//.......................................................................................
void RepBasicOutlierCleaner::init_attributes()
{

	string _signal_name = signalName;
	if (val_channel != 0)
		_signal_name += "_" + to_string(val_channel);

	attributes.clear();
	if (!nRem_attr.empty())
		attributes.push_back(nRem_attr);
	if (!nRem_attr_suffix.empty())
		attributes.push_back(_signal_name + "_" + nRem_attr_suffix);

	if (!nTrim_attr.empty())
		attributes.push_back(nTrim_attr);
	if (!nTrim_attr_suffix.empty())
		attributes.push_back(_signal_name + "_" + nTrim_attr_suffix);
}

// Learn cleaning boundaries
//.......................................................................................
int RepBasicOutlierCleaner::_learn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_cleaners)
{

	if (params.type == VAL_CLNR_ITERATIVE)
		return iterativeLearn(rep, samples, prev_cleaners);
	else if (params.type == VAL_CLNR_QUANTILE)
		return quantileLearn(rep, samples, prev_cleaners);
	else
	{
		MERR("Unknown cleaning type %d\n", params.type);
		return -1;
	}
}

// Learning : learn cleaning boundaries using MedValueCleaner's iterative approximation of moments
//.......................................................................................
int RepBasicOutlierCleaner::iterativeLearn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_cleaners)
{

	// Sanity check
	if (signalId == -1)
	{
		MERR("RepBasicOutlierCleaner::iterativeLearn - Uninitialized signalId(%s)\n", signalName.c_str());
		return -1;
	}

	// Get all values
	vector<float> values;
	get_values(rep, samples, signalId, time_channel, val_channel, params.range_min, params.range_max, values, prev_cleaners);
	// MLOG("basic Iterative clean Learn: signalName %s signalId %d :: got %d values()\n", signalName.c_str(), signalId, values.size());

	// Iterative approximation of moments
	int rc = get_iterative_min_max(values);

	return rc;
}

// Learning : learn cleaning boundaries using MedValueCleaner's quantile approximation of moments
//.......................................................................................
int RepBasicOutlierCleaner::quantileLearn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_cleaners)
{

	// Sanity check
	if (signalId == -1)
	{
		MERR("RepBasicOutlierCleaner::quantileLearn - Uninitialized signalId(%s)\n", signalName.c_str());
		return -1;
	}

	// Get all values
	vector<float> values;
	get_values(rep, samples, signalId, time_channel, val_channel, params.range_min, params.range_max, values, prev_cleaners);

	if (values.empty())
	{
		MWARN("RepBasicOutlierCleaner::quantileLearn WARNING signal [%d] = [%s] is empty, will not clean outliers\n", signalId,
			  this->signalName.c_str());
		return 0;
	}
	// Quantile approximation of moments
	return get_quantile_min_max(values);
}

// Apply cleaning model
//.......................................................................................
int RepBasicOutlierCleaner::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	// MLOG("basic cleaner _apply: signalName %s signalId %d\n", signalName.c_str(), signalId);

	// Sanity check
	if (signalId == -1)
	{
		MERR("RepBasicOutlierCleaner::_apply - Uninitialized signalId(%s)\n", signalName.c_str());
		return -1;
	}

	// Check that we have the correct number of dynamic-versions : one per time-point (if given)
	if (time_points.size() != 0 && time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}

	if (is_categ)
		return 0;
	int len;

	differentVersionsIterator vit(rec, signalId);
	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{

		// Clean
		rec.uget(signalId, iver, rec.usv); // get into the internal usv obeject - this statistically saves init time

		len = rec.usv.len;
		vector<int> remove(len);
		vector<pair<int, float>> change(len);
		int nRemove = 0, nChange = 0;

		// Collect
		for (int i = 0; i < len; i++)
		{
			int itime = rec.usv.Time(i, time_channel);
			float ival = rec.usv.Val(i, val_channel);

			// No need to clean past the latest relevant time-point
			if (time_points.size() != 0 && itime > time_points[iver])
				break;

			// MLOG("Checking pid %d with remove %d %f %f trim %d %f %f\n", rec.pid, params.doRemove, removeMin, removeMax, params.doTrim, trimMin, trimMax);
			//  Identify values to change or remove
			if (params.doRemove && (ival < removeMin - NUMERICAL_CORRECTION_EPS || ival > removeMax + NUMERICAL_CORRECTION_EPS))
			{
				// MLOG("pid %d ver %d time %d %s channel %d %f removed\n", rec.pid, iver, itime, signalName.c_str(), val_channel, ival);
				remove[nRemove++] = i;
			}
			else if (params.doTrim)
			{
				if (ival < trimMin)
				{
					//					MLOG("pid %d ver %d time %d %s %f trimmed\n", rec.pid, iver, itime, signalName.c_str(), ival);
					change[nChange++] = pair<int, float>(i, trimMin);
				}
				else if (ival > trimMax)
				{
					//					MLOG("pid %d ver %d time %d %s %f trimmed\n", rec.pid, iver, itime, signalName.c_str(), ival);
					change[nChange++] = pair<int, float>(i, trimMax);
				}
			}
		}

		// collect stats for summary:
#pragma omp critical
		{
			++_stats.total_pids;
			_stats.total_records += len;
			_stats.total_removed += nRemove;
			_stats.total_pids_touched += nRemove > 0;
		}

		// Apply removals + changes
		change.resize(nChange);
		remove.resize(nRemove);
		if (!verbose_file.empty())
		{
			if (remove.size() > 0)
			{
				string sigName = signalName;
				if (val_channel > 0)
					sigName += "_ch_" + to_string(val_channel);
				string collect_cleaning_time = to_string(rec.usv.Time(remove[0], time_channel));
				for (int rem = 0; rem < remove.size(); ++rem)
					collect_cleaning_time += "," + to_string(rec.usv.Time(remove[rem], time_channel));
#pragma omp critical
				{
					log_file << "GLOBAL_STATS: signal " << sigName << " pid " << rec.pid << " removed "
							 << nRemove << " changed " << nChange << "\n";
					for (int rem = 0; rem < remove.size(); ++rem)
						log_file << "FILTER: signal " << sigName << " pid " << rec.pid << " removed/changed\t"
								 << rec.usv.Time(remove[rem], time_channel) << "\t" << rec.usv.Val(remove[rem], val_channel) << "\n";
				}
				//<< collect_cleaning_time << "\n";
			}
		}
		if (rec.update(signalId, iver, val_channel, change, remove) < 0)
			return -1;

		// Collect atttibutes
		int idx = 0;
		if (!nRem_attr.empty() && !attributes_mat.empty())
		{
			for (int pVersion = vit.block_first(); pVersion <= vit.block_last(); pVersion++)
				attributes_mat[pVersion][idx] = (float)nRemove;
			idx++;
		}

		if (!nRem_attr_suffix.empty() && !attributes_mat.empty())
		{
			for (int pVersion = vit.block_first(); pVersion <= vit.block_last(); pVersion++)
				attributes_mat[pVersion][idx] = (float)nRemove;
			idx++;
		}

		if (!nTrim_attr.empty() && !attributes_mat.empty())
		{
			for (int pVersion = vit.block_first(); pVersion <= vit.block_last(); pVersion++)
				attributes_mat[pVersion][idx] = (float)nChange;
			idx++;
		}

		if (!nTrim_attr_suffix.empty() && !attributes_mat.empty())
		{
			for (int pVersion = vit.block_first(); pVersion <= vit.block_last(); pVersion++)
				attributes_mat[pVersion][idx] = (float)nChange;
			idx++;
		}
	}

	return 0;
}

void remove_stats::restart()
{
	total_removed = 0;
	total_pids_touched = 0;
	total_records = 0;
	total_pids = 0;
}

void remove_stats::print_summary(const string &cleaner_info, const string &signal_name, int minimal_pid_cnt, float print_summary_critical_cleaned, bool prnt_flg) const
{
	if (total_records == 0)
		return; // nothing happend
	float rmv_ratio = float(total_removed) / total_records;
	bool is_critical = total_pids > minimal_pid_cnt && rmv_ratio > print_summary_critical_cleaned;
	// build msg:
	string msg;
	if (is_critical || prnt_flg)
	{
		char buffer[5000];
		snprintf(buffer, sizeof(buffer), "Removed Records: (%d / %d) %2.1f%%, Touched Patients: (%d / %d) %2.1f%%",
				 total_removed, total_records, 100 * rmv_ratio,
				 total_pids_touched, total_pids, 100 * float(total_pids_touched) / total_pids);
		msg = string(buffer);
	}

	if (is_critical)
	{
		MWARN("Warning: %s(%s): %s\n", cleaner_info.c_str(), signal_name.c_str(), msg.c_str());
	}
	else
	{
		if (prnt_flg)
			MLOG("Info: %s(%s): %s\n", cleaner_info.c_str(), signal_name.c_str(), msg.c_str());
	}
}

void RepBasicOutlierCleaner::make_summary()
{
	_stats.print_summary(my_class_name(), signalName, 100, print_summary_critical_cleaned, print_summary);
	_stats.restart();
}

//.......................................................................................
void RepBasicOutlierCleaner::dprint(const string &pref, int rp_flag)
{
	if (rp_flag > 0)
		MLOG("%s :: BasicOutlierCleaner: signal: %d %s : v_channel %d : doTrim %d trimMax %f trimMin %f : doRemove %d : removeMax %f removeMin %f\n",
			 pref.c_str(), signalId, signalName.c_str(), val_channel, params.doTrim, trimMax, trimMin, params.doRemove, removeMax, removeMin);
}

//=======================================================================================
// ConfiguredOutlierCleaner
//=======================================================================================
//.......................................................................................
int readConfFile(string confFileName, map<string, confRecord> &outlierParams)
// read from outlierParamFile into outlierParams map
{
	ifstream infile;
	confRecord thisRecord;
	string thisLine;
	infile.open(confFileName.c_str(), ifstream::in);
	if (!infile.is_open())
		MTHROW_AND_ERR("Cannot open %s for reading\n", confFileName.c_str());
	getline(infile, thisLine); // consume title line.
	while (getline(infile, thisLine))
	{
		boost::trim(thisLine);
		if (thisLine.empty() || thisLine.at(0) == '#')
			continue; // skip empty line

		vector<string> f;
		boost::split(f, thisLine, boost::is_any_of(","));
		if (f.size() != 8 && f.size() != 10)
		{
			infile.close();
			MTHROW_AND_ERR("Wrong field count in  %s (%s : %zd) \n", confFileName.c_str(), thisLine.c_str(), f.size());
		}

		thisRecord.confirmedLow = thisRecord.logicalLow = atof(f[1].c_str());
		thisRecord.confirmedHigh = thisRecord.logicalHigh = atof(f[2].c_str());

		thisRecord.distLow = f[4];
		thisRecord.distHigh = f[6];
		thisRecord.val_channel = stoi(f[7]);
		if (thisRecord.distLow != "none")
			thisRecord.confirmedLow = atof(f[3].c_str());
		if (thisRecord.distHigh != "none")
			thisRecord.confirmedHigh = atof(f[5].c_str());
		string sigName = f[0];
		if (thisRecord.val_channel > 0)
			sigName += "_ch_" + to_string(thisRecord.val_channel);
		thisRecord.trimLow = -1e30F;
		thisRecord.trimHigh = 1e30F;
		if (f.size() > 8)
		{
			thisRecord.trimLow = atof(f[8].c_str());
			thisRecord.trimHigh = atof(f[9].c_str());
		}
		outlierParams[sigName] = thisRecord;
	}

	infile.close();
	return (0);
}
int RepConfiguredOutlierCleaner::init(map<string, string> &mapper)
{
	map<string, confRecord> outlierParams_dict;
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepConfiguredOutlierCleaner::init]
		if (field == "signal")
		{
			signalName = entry.second;
			req_signals.insert(signalName);
		}
		else if (field == "time_channel")
			time_channel = med_stoi(entry.second);
		else if (field == "val_channel")
			val_channel = med_stoi(entry.second);
		else if (field == "nrem_attr")
			nRem_attr = entry.second;
		else if (field == "ntrim_attr")
			nTrim_attr = entry.second;
		else if (field == "nrem_suff")
			nRem_attr_suffix = entry.second;
		else if (field == "ntrim_suff")
			nTrim_attr_suffix = entry.second;
		else if (field == "conf_file")
			readConfFile(entry.second, outlierParams_dict);
		else if (field == "clean_method")
			cleanMethod = entry.second;
		else if (field == "unconditional")
			unconditional = stoi(entry.second) > 0;
		else if (field == "rp_type")
		{
		}
		else if (field == "verbose_file")
			verbose_file = entry.second;
		else if (field == "print_summary")
			print_summary = stoi(entry.second) > 0;
		else if (field == "print_summary_critical_cleaned")
			print_summary_critical_cleaned = stof(entry.second);
		//! [RepConfiguredOutlierCleaner::init]
	}

	string sig_search = signalName;
	if (val_channel > 0)
		sig_search += "_ch_" + to_string(val_channel);

	if (outlierParams_dict.find(sig_search) != outlierParams_dict.end())
		outlierParam = outlierParams_dict.at(sig_search);
	else
		MTHROW_AND_ERR("Error in RepConfiguredOutlierCleaner::init - Unknown signal %s in configure rules\n", sig_search.c_str());

	if (!verbose_file.empty())
	{
		verbose_file += "." + sig_search;
		ofstream fw(verbose_file);
		fw.close(); // rewrite empty file
	}

	init_lists();
	map<string, string> &mapper_p = mapper;
	vector<string> remove_fl = {"verbose_file", "clean_method", "conf_file", "rp_type", "unconditional",
								"signal", "time_channel", "val_channel", "print_summary", "print_summary_critical_cleaned"};
	for (const string &fl : remove_fl)
		if (mapper_p.find(fl) != mapper_p.end())
			mapper_p.erase(fl);
	return MedValueCleaner::init(mapper_p);
}

void RepConfiguredOutlierCleaner::set_signal_ids(MedSignals &sigs)
{
	RepBasicOutlierCleaner::set_signal_ids(sigs); // call base class init
}

// Learn bounds
//.......................................................................................
int RepConfiguredOutlierCleaner::_learn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_cleaners)
{
	trimMax = outlierParam.trimHigh;
	trimMin = outlierParam.trimLow;
	// fetch val_channel from file
	val_channel = outlierParam.val_channel;

	if (cleanMethod == "logical")
	{
		removeMax = outlierParam.logicalHigh;
		removeMin = outlierParam.logicalLow;
		return (0);
	}
	else if (cleanMethod == "confirmed")
	{
		removeMax = outlierParam.confirmedHigh;
		removeMin = outlierParam.confirmedLow;
		return (0);
	}
	else if (cleanMethod == "learned")
	{
		removeMax = outlierParam.logicalHigh;
		removeMin = outlierParam.logicalLow;
		string thisDistHi = outlierParam.distHigh;
		string thisDistLo = outlierParam.distLow;
		if (thisDistHi == "none" && thisDistLo == "none")
			return (0); // nothing to learn

		else
		{

			vector<float> values, filteredValues;

			float borderHi = numeric_limits<float>::max(), borderLo = -99999, logBorderHi = 9999, logBorderLo = -99999;
			get_values(rep, samples, signalId, time_channel, val_channel, removeMin, removeMax, values, prev_cleaners);
			for (auto &el : values)
				if (el != 0)
					filteredValues.push_back(el);
			sort(filteredValues.begin(), filteredValues.end());
			if (thisDistHi == "norm" || thisDistLo == "norm")
				learnDistributionBorders(borderHi, borderLo, filteredValues);
			if (thisDistHi == "lognorm" || thisDistLo == "lognorm")
			{
				/*	ofstream dFile;
				dFile.open("DFILE");
				for (auto& el : filteredValues)dFile << el << "\n";
				dFile.close();
				*/

				for (auto &el : filteredValues)
					if (el > 0)
						el = log(el);
					else
						return (-1);

				learnDistributionBorders(logBorderHi, logBorderLo, filteredValues);
			}
			if (thisDistHi == "norm")
				removeMax = borderHi;
			else if (thisDistHi == "lognorm")
				removeMax = expf(logBorderHi);
			else if (thisDistHi == "manual")
				removeMax = outlierParam.confirmedHigh;
			if (thisDistLo == "norm")
				removeMin = borderLo;
			else if (thisDistLo == "lognorm")
				removeMin = expf(logBorderLo);
			else if (thisDistLo == "manual")
				removeMin = outlierParam.confirmedLow;

			return (0);
		}
	}

	else
	{
		MERR("Unknown cleaning method %s\n", cleanMethod.c_str());
		return -1;
	}
}

void learnDistributionBorders(float &borderHi, float &borderLo, vector<float> filteredValues)
// a function that takes sorted vector of filtered values and estimates the +- 7 sd borders based on the center of distribution
// predefined calibration constants are used for estimation of the borders.
{
	double sum = 0;
	double sumsq = 0;
	const float margin[] = {0.01F, 0.99F}; // avoid tails of distribution
	const float varianceFactor = 0.8585F;
	const float meanShift = 0; // has value when margins are asymetric
	const float sdNums = 7;	   // how many standard deviation on each side of the mean.

	int start = (int)round(filteredValues.size() * margin[0]);
	int stop = (int)round(filteredValues.size() * margin[1]);
	for (vector<float>::iterator el = filteredValues.begin() + start; el < filteredValues.begin() + stop; el++)
	{

		sum += *el;
		sumsq += static_cast<double>(*el) * static_cast<double>(*el);
	}
	double mean = sum / (stop - start);
	double var = sumsq / (stop - start) - mean * mean;
	// printf("sum %f sumsq %f  stop %d start %d\n", sum, sumsq, stop, start);
	var = var / varianceFactor;
	mean = mean - meanShift * sqrt(var);
	borderHi = (float)(mean + sdNums * sqrt(var));
	borderLo = (float)(mean - sdNums * sqrt(var));
}

//.......................................................................................

void RepConfiguredOutlierCleaner::print()
{
	MLOG("BasicOutlierCleaner: signal: %d : doTrim %d trimMax %f trimMin %f : doRemove %d : removeMax %f removeMin %f\n",
		 signalId, params.doTrim, trimMax, trimMin, params.doRemove, removeMax, removeMin, confFileName.c_str(), cleanMethod.c_str());
}
//=======================================================================================
// RuleBasedOutlierCleaner
//=======================================================================================

// Defulat initialization
void RepRuleBasedOutlierCleaner::init_defaults()
{

	processor_type = REP_PROCESS_RULEBASED_OUTLIER_CLEANER;
	// consider all rules if none is given specifically
	for (const auto &rule : rules2Signals)
		consideredRules.push_back(rule.first);

	// Set rules2Apply
	select_rules_to_apply();
	init_lists();
}

// Read rules from file
void RepRuleBasedOutlierCleaner::parse_rules_signals(const string &path)
{
	ifstream fr(path);
	if (!fr.good())
		MTHROW_AND_ERR("Error RepRuleBasedOutlierCleaner::parse_rules_signals - can't read file %s\n", path.c_str());
	string line;
	while (getline(fr, line))
	{
		boost::trim(line);
		if (line.empty() || line[0] == '#')
			continue;
		vector<string> tokens, list_of_sigs;
		boost::split(tokens, line, boost::is_any_of("\t"));
		if (tokens.size() != 2 && tokens.size() != 3)
			MTHROW_AND_ERR("Error RepRuleBasedOutlierCleaner::parse_rules_signals - line should contain 2-3 tokens with TAB. got line:\n%s\n",
						   line.c_str());
		int rule_id = med_stoi(tokens[0]);
		boost::split(list_of_sigs, tokens[1], boost::is_any_of(","));
		if (rules2Signals[rule_id].size() != list_of_sigs.size())
			MTHROW_AND_ERR("Error RepRuleBasedOutlierCleaner::parse_rules_signals - rule %d contains %zu signals, got %zu signals\n",
						   rule_id, rules2Signals[rule_id].size(), list_of_sigs.size());
		rules2Signals[rule_id] = list_of_sigs;
		if (tokens.size() == 3) // has rules2RemoveSignal value
			rules2RemoveSignal[rule_id] = boost::trim_copy(tokens[2]);
	}
	fr.close();
}

// Read channels information from file
void RepRuleBasedOutlierCleaner::parse_sig_channels(const string &path)
{
	ifstream fr(path);
	if (!fr.good())
		MTHROW_AND_ERR("Error RepRuleBasedOutlierCleaner::parse_sig_channels - can't read file %s\n", path.c_str());
	string line;
	while (getline(fr, line))
	{
		boost::trim(line);
		if (line.empty() || line[0] == '#')
			continue;
		vector<string> tokens, list_of_sigs;
		boost::split(tokens, line, boost::is_any_of("\t"));
		if (tokens.size() != 3)
			MTHROW_AND_ERR("Error RepRuleBasedOutlierCleaner::parse_sig_channels - line should contain 3 tokens with TAB (signal name, time channel, val channel). got line:\n%s\n",
						   line.c_str());
		string sigName = tokens[0];
		int time_channel = med_stoi(tokens[1]);
		int val_channel = med_stoi(tokens[1]);
		signal_channels[sigName].first = time_channel;
		signal_channels[sigName].second = val_channel;
	}
	fr.close();
}

// Adjust rulesToApply: Check if some required signals are missing and remove corresponding rules
void RepRuleBasedOutlierCleaner::fit_for_repository(MedPidRepository &rep)
{

	int index = 0;
	for (auto &rule : rulesToApply)
	{
		const vector<string> &all_sigs = rules2Signals[rule];
		string miss_sig = "";
		for (size_t i = 0; i < all_sigs.size() && miss_sig.empty(); ++i)
		{
			if (rep.sigs.Name2Sid.find(all_sigs[i]) == rep.sigs.Name2Sid.end())
			{
				miss_sig = all_sigs[i];
				break;
			}
		}
		if (!miss_sig.empty())
			MWARN("RepRuleBasedOutlierCleaner: Signal %s does not exists - removing rule %d\n", miss_sig.c_str(), rule);
		else
			index++;
	}

	rulesToApply.resize(index);
	rulesToApply.shrink_to_fit();

	init_lists();
}

/// select which rules to apply according to consideredRules
void RepRuleBasedOutlierCleaner::select_rules_to_apply()
{

	rulesToApply = consideredRules;
}

// Init affected and required signals lists according to rules
void RepRuleBasedOutlierCleaner::init_lists()
{

	aff_signals.clear();
	req_signals.clear();

	// Loop on rules to apply
	for (int rule : rulesToApply)
	{
		// Affected Signals
		if (rules2RemoveSignal.find(rule) != rules2RemoveSignal.end())
			aff_signals.insert(rules2RemoveSignal[rule]);
		else
		{
			for (string signal : rules2Signals[rule])
				aff_signals.insert(signal);
		}

		// Required Signals
		for (string signal : rules2Signals[rule])
			req_signals.insert(signal);
	}
}

// Init from map
int RepRuleBasedOutlierCleaner::init(map<string, string> &mapper)
{

	processor_type = REP_PROCESS_RULEBASED_OUTLIER_CLEANER;
	set<string> rulesStrings;

	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepRuleBasedOutlierCleaner::init]
		if (field == "rules2Signals")
			parse_rules_signals(entry.second); // each line is rule_id [TAB] list of signals with ","  optional [TAB] for which signal to remove when contradiction
		else if (field == "signal_channels")
			parse_sig_channels(entry.second); // each line is signal_name [TAB] time_channel [TAB] val_channel
		else if (field == "time_window")
			time_window = med_stoi(entry.second);
		else if (field == "nrem_attr")
			nRem_attr = entry.second;
		else if (field == "verbose_file")
			verbose_file = entry.second;
		else if (field == "nrem_suff")
			nRem_attr_suffix = entry.second;
		else if (field == "tolerance")
			tolerance = med_stof(entry.second);
		else if (field == "calc_res")
			calc_res = med_stof(entry.second);
		else if (field == "unconditional")
			unconditional = stoi(entry.second) > 0;
		else if (field == "rp_type")
		{
		}
		else if (field == "print_summary")
			print_summary = stoi(entry.second) > 0;
		else if (field == "print_summary_critical_cleaned")
			print_summary_critical_cleaned = stof(entry.second);
		else if (field == "consideredRules")
		{
			consideredRules.clear();
			boost::split(rulesStrings, entry.second, boost::is_any_of(","));
			for (auto &rule : rulesStrings)
			{
				int ruleNum = med_stoi(rule);
				consideredRules.push_back(ruleNum);
				if (ruleNum == 0)
					break;
			}
		}
		//! [RepRuleBasedOutlierCleaner::init]
	}

	// consider all rules if none is given specifically
	if (consideredRules.empty())
	{
		// init deafault to use all:
		for (const auto &rule : rules2Signals)
			consideredRules.push_back(rule.first);
	}

	select_rules_to_apply();
	init_lists();

	if (!verbose_file.empty())
	{
		ofstream fw(verbose_file);
		fw.close(); // rewrite empty file
	}

	return 0;
}

// Initialization of reporting attributes
void RepRuleBasedOutlierCleaner::init_attributes()
{

	attributes.clear();
	if (!nRem_attr_suffix.empty())
	{
		for (string signalName : aff_signals)
			attributes.push_back(signalName + "_" + nRem_attr_suffix);
	}

	if (!nRem_attr.empty())
		attributes.push_back(nRem_attr);
}

// Get req/aff signal ids
void RepRuleBasedOutlierCleaner::set_signal_ids(MedSignals &sigs)
{
	for (const auto &reqSig : req_signals)
		reqSignalIds.insert(sigs.sid(reqSig));
	for (const auto &affSig : aff_signals)
		affSignalIds.insert(sigs.sid(affSig));

	// Keep names for logging.
	for (int affSig_id : affSignalIds)
		affected_ids_to_name[affSig_id] = sigs.name(affSig_id);
	if (!verbose_file.empty() && !log_file.is_open())
	{
		log_file.open(verbose_file, ios::app);
		if (!log_file.good())
			MWARN("Warnning in RepRuleBasedOutlierCleaner - verbose_file %s can't be opened\n", verbose_file.c_str());
	}
}

// Initialization of tables
void RepRuleBasedOutlierCleaner::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{

	rules_sids.clear();
	affected_by_rules.clear();

	for (int i = 0; i < rulesToApply.size(); i++)
	{

		// build set of the participating signals
		string removal_signal;
		if (rules2RemoveSignal.find(rulesToApply[i]) != rules2RemoveSignal.end())
			removal_signal = rules2RemoveSignal.at(rulesToApply[i]);

		for (auto &sname : rules2Signals[rulesToApply[i]])
		{
			int thisSid = dict.id(sname);
			rules_sids[rulesToApply[i]].push_back(thisSid);
			bool affect_sig = affSignalIds.find(thisSid) != affSignalIds.end();
			affect_sig = affect_sig && (removal_signal.empty() || removal_signal == sname);
			affected_by_rules[rulesToApply[i]].push_back(affect_sig);

			if (signal_channels.find(sname) != signal_channels.end())
			{
				signal_id_channels[thisSid] = signal_channels[sname];
				// check channels exists:
				if (signal_id_channels[thisSid].first >= sigs.Sid2Info.at(thisSid).n_time_channels ||
					signal_id_channels[thisSid].second >= sigs.Sid2Info.at(thisSid).n_val_channels)
					MTHROW_AND_ERR("Error in RepRuleBasedOutlierCleaner::init_tables - signal %s reffer to channel that not exists\n"
								   "existed time_channels %d, requested %d, existed val_channels %d, request %d\n",
								   sname.c_str(), sigs.Sid2Info.at(thisSid).n_time_channels, signal_id_channels[thisSid].first,
								   sigs.Sid2Info.at(thisSid).n_val_channels, signal_id_channels[thisSid].second);
			}
		}
	}
}

// Apply cleaners
int RepRuleBasedOutlierCleaner::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	// get the signals
	map<int, UniversalSigVec> usvs; // from signal to its USV
	// map <int, vector <int>> removePoints; // from signal id to its remove points
	if (rulesToApply.empty())
		return 0; // removed rule

	// Check that we have the correct number of dynamic-versions : one per time-point
	if (time_points.size() != 0 && time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}

	differentVersionsIterator vit(rec, reqSignalIds);
	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{

		map<int, set<int>> removePoints;				   // from sid to indices to be removed
		unordered_map<int, vector<int>> removePoints_Time; // from sid to Time to be removed - for printings

		// Clean
		for (auto reqSigId : reqSignalIds)
		{
			rec.uget(reqSigId, iver, usvs[reqSigId]);
			removePoints[reqSigId] = {};
		}

		// Now loop on rules
		// printf("removePointsSize:%d\n", removePoints.size());

		for (int iRule = 0; iRule < rulesToApply.size(); iRule++)
		{
			int rule = rulesToApply[iRule];
			// MLOG("Apply Rule %d\n", rule);
			vector<UniversalSigVec> ruleUsvs;
			vector<int> &mySids = rules_sids[rule];
			const vector<string> &rule_signals = rules2Signals[rule];

			// build set of the participating signals
			for (int sid : mySids)
				ruleUsvs.push_back(usvs[sid]);

			bool signalEmpty = false;
			for (auto &thisUsv : ruleUsvs) // look for empty signals and skip the rule
				if (thisUsv.len == 0)
					signalEmpty = true;
			if (signalEmpty)
				continue; // skip this rule. one of the signals is empty (maybe was cleaned in earlier stage )

			// loop and find times where you have all signals
			vector<int> sPointer(mySids.size(), 0);
			int thisTime;
			pair<int, int> first_chan(0, 0);
			if (signal_channels.find(rule_signals.front()) != signal_channels.end())
				first_chan = signal_channels[rule_signals.front()];
			for (sPointer[0] = 0; sPointer[0] < ruleUsvs[0].len; ++sPointer[0])
			{
				// printf("start loop %d %d \n", sPointer[0], ruleUsvs[0].len);
				// skip sPointer[0] to last with same time:
				while (sPointer[0] + 1 < ruleUsvs[0].len && ruleUsvs[0].Time(sPointer[0], first_chan.first) ==
																ruleUsvs[0].Time(sPointer[0] + 1, first_chan.first))
					++sPointer[0]; // If can advance forward and same time, then advance!

				thisTime = ruleUsvs[0].Time(sPointer[0], first_chan.first);
				if (time_points.size() != 0 && thisTime > time_points[iver])
					break;
				bool ok = true;
				for (int i = 1; i < mySids.size(); i++)
				{
					pair<int, int> sig_channels(0, 0);
					if (signal_channels.find(rule_signals[i]) != signal_channels.end())
						sig_channels = signal_channels[rule_signals[i]];
					while (ruleUsvs[i].Time(sPointer[i], sig_channels.first) < thisTime - time_window && sPointer[i] < ruleUsvs[i].len - 1)
						++sPointer[i];
					// find closest (or exact):
					int try_more = 1;
					while (sPointer[i] + try_more < ruleUsvs[i].len &&
						   ruleUsvs[i].Time(sPointer[i] + try_more, sig_channels.first) <= thisTime)
						++try_more;
					if (try_more - 1 > 0)
						sPointer[i] += (try_more - 1); // better pointer so use it's best

					// printf("before ok_check: %d %d %d %d %d %d\n", i, sPointer[0], sPointer[1], sPointer[2],thisTime, ruleUsvs[i].Time(sPointer[i], time_channel));
					int time_diff = abs(ruleUsvs[i].Time(sPointer[i], sig_channels.first) - thisTime);
					// if (ruleUsvs[i].Time(sPointer[i], sig_channels.first) != thisTime) {
					if (time_diff > time_window)
					{ // not found any candidate
						// printf("before ok_0: %d %d %d %d %d\n", rule, sPointer[0], sPointer[1], sPointer[2]);
						ok = 0;
						break;
					}
				}
				if (ok)
				{
					// if found all signals from same date eliminate doubles and take the last one for comparison
					vector<int> rule_val_channels, rule_time_channels;
					for (int i = 0; i < mySids.size(); i++)
					{
						pair<int, int> sig_channels(0, 0);
						if (signal_channels.find(rule_signals[i]) != signal_channels.end())
							sig_channels = signal_channels[rule_signals[i]];
						rule_val_channels.push_back(sig_channels.second);
						rule_time_channels.push_back(sig_channels.first);
					}
					for (int i = 0; i < mySids.size(); i++)
					{

						int remove_same_time = 1; // try remove for signal with same time value - Can use SimValHandler before, it's better
						while (sPointer[i] - remove_same_time >= 0 &&
							   ruleUsvs[i].Time(sPointer[i], rule_time_channels[i]) == ruleUsvs[i].Time(sPointer[i] - remove_same_time, rule_time_channels[i]))
						{
							if (affected_by_rules[rule][i])
							{
								// MLOG("Remove SAME:: sig %s, time_idx=%d (sPointer[0]=%d)\n", affected_ids_to_name[mySids[i]].c_str(), sPointer[i] - remove_same_time, sPointer[0]);
								removePoints[mySids[i]].insert(sPointer[i] - remove_same_time);
								if (!verbose_file.empty())
									removePoints_Time[mySids[i]].push_back(ruleUsvs[i].Time(sPointer[i] - remove_same_time, rule_time_channels[i]));
							}
							++remove_same_time;
						}
					}
					// check rule and mark for removement
					// printf("before apply: %d %d %d %d\n", rule, sPointer[0],sPointer[1],sPointer[2]);
					bool ruleFlagged = applyRule(rule, ruleUsvs, rule_val_channels, sPointer);
					/*
					printf("%d R: %d P: %d t: %d   ",ruleFlagged, rule, rec.pid, thisTime);
					for (int k = 0; k < sPointer.size(); k++)printf(" %f", ruleUsvs[k].Val(sPointer[k]));
					printf("\n");
					*/
					if (ruleFlagged)
					{
						for (int sIndex = 0; sIndex < mySids.size(); sIndex++)
							if (affected_by_rules[rulesToApply[iRule]][sIndex])
							{
								removePoints[mySids[sIndex]].insert(sPointer[sIndex]);
								if (!verbose_file.empty())
									removePoints_Time[mySids[sIndex]].push_back(ruleUsvs[sIndex].Time(sPointer[sIndex], rule_time_channels[sIndex]));
							}
					}
				}
			}
		}

		// Apply removals
		size_t nRemove = 0;
		int idx = 0;

		for (auto sig : affSignalIds)
		{
			vector<int> toRemove(removePoints[sig].begin(), removePoints[sig].end());
			string sig_name = affected_ids_to_name[sig];
			if (!verbose_file.empty() && !toRemove.empty())
			{
				string time_points = removePoints_Time[sig].empty() ? "" : to_string(removePoints_Time[sig].front());
				for (size_t i = 1; i < removePoints_Time[sig].size(); ++i)
					time_points += "," + to_string(removePoints_Time[sig][i]);
				log_file << "signal " << sig_name << " pid " << rec.pid << " removed "
						 << toRemove.size() << " int_times " << time_points << "\n";
			}

#pragma omp critical
			{
				++_rmv_stats[sig_name].total_pids;
				_rmv_stats[sig_name].total_pids_touched += int(!removePoints[sig].empty());
				_rmv_stats[sig_name].total_records += usvs[sig].len;
				_rmv_stats[sig_name].total_removed += (int)removePoints[sig].size();
			}

			vector<pair<int, float>> noChange;
			pair<int, int> sig_channels(0, 0);
			if (signal_id_channels.find(sig) != signal_id_channels.end())
				sig_channels = signal_id_channels[sig];
			if (rec.update(sig, iver, sig_channels.second, noChange, toRemove) < 0)
				return -1;
			if (!nRem_attr_suffix.empty() && !attributes_mat.empty())
			{
				for (int pVersion = vit.block_first(); pVersion <= vit.block_last(); pVersion++)
					attributes_mat[pVersion][idx] = (float)toRemove.size();
				idx++;
			}
			nRemove += toRemove.size();
		}

		// Collect atttibutes
		if (!nRem_attr.empty() && !attributes_mat.empty())
		{
			for (int pVersion = vit.block_first(); pVersion <= vit.block_last(); pVersion++)
				attributes_mat[pVersion][idx] = (float)nRemove;
		}
	}

	return 0;
}

// Reporting.
void RepRuleBasedOutlierCleaner::make_summary()
{
	for (auto it = _rmv_stats.begin(); it != _rmv_stats.end(); ++it)
		it->second.print_summary(my_class_name(), it->first.c_str(), 100, print_summary_critical_cleaned, print_summary);

	_rmv_stats.clear(); // prepare for next apply
}

void RepRuleBasedOutlierCleaner::dprint(const string &pref, int rp_flag)
{
	if (rp_flag > 0)
	{
		MLOG("%s :: RP type %d(%s) : required(%d): ", pref.c_str(), processor_type, my_class_name().c_str(), req_signals.size());
		if (rp_flag > 1)
			for (auto &rsig : req_signals)
				MLOG("%s,", rsig.c_str());
		MLOG(" affected(%d): ", aff_signals.size());
		if (rp_flag > 1)
			for (auto &asig : aff_signals)
				MLOG("%s, ", asig.c_str());
		if (!rulesToApply.empty())
			MLOG(" Rules: [%d", rulesToApply[0]);
		for (size_t i = 1; i < rulesToApply.size(); ++i)
			MLOG(", %d", rulesToApply[i]);
		MLOG("]\n");
	}
}

// Utility
bool test_diff(float origianl, float calculated, float tolerance, float resulotion)
{
	float df = abs(origianl - calculated);
	return (df > tolerance * calculated && df > resulotion);
}

// Apply rule
bool RepRuleBasedOutlierCleaner::applyRule(int rule, const vector<UniversalSigVec> &ruleUsvs,
										   const vector<int> &val_channels, const vector<int> &sPointer)
// apply the rule and return true if data is consistent with the rule
// ruleUsvs hold the signals in the order they appear in the rule in the rules2Signals above
{

	float left, right, right2, right3; // sides of the equality or inequality of the rule
	float res_factor = 1;
	if (calc_res > 0)
		res_factor = 1.0 / calc_res;
	bool test_1;

	switch (rule)
	{
	case 1: // BMI=Weight/Height^2*1e4
		if (ruleUsvs[2].Val(sPointer[2], val_channels[2]) == 0)
			return (true);
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]);
		right = ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * (float)1e4;
		// printf("inputs %f %f\n", ruleUsvs[1].Val(sPointer[1]), ruleUsvs[2].Val(sPointer[2]));
		return test_diff(left, right, tolerance, calc_res);

	case 2: // MCH=Hemoglobin/RBC*10
	case 3: // MCV=Hematocrit/RBC*10
		if (ruleUsvs[2].Val(sPointer[2], val_channels[2]) == 0)
			return (true);
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]);
		right2 = ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * 10;
		right = round(ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * 10 * res_factor) / (float)res_factor; // resolution in THIN is 0.1
		right3 = int(ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * 10 * res_factor) / (float)res_factor;	// resolution in THIN is 0.1
		if (calc_res > 0)
			return (test_diff(left, right, tolerance, calc_res) && test_diff(left, right2, tolerance, calc_res) &&
					test_diff(left, right3, tolerance, calc_res));
		else
			return test_diff(left, right2, tolerance, calc_res);

	case 4: // MCHC-M=MCH/MCV*100
		if (ruleUsvs[2].Val(sPointer[2], val_channels[2]) == 0)
			return (true);
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]);
		right2 = ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * 100;
		right = round(ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * 100 * res_factor) / (float)res_factor; // resolution in THIN is 0.1
		right3 = int(ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * 100 * res_factor) / (float)res_factor;	 // resolution in THIN is 0.1
		if (calc_res > 0)
			return (test_diff(left, right, tolerance, calc_res) && test_diff(left, right2, tolerance, calc_res) &&
					test_diff(left, right3, tolerance, calc_res));
		else
			return test_diff(left, right2, tolerance, calc_res);

	case 11: // HDL_over_nonHDL=HDL/NonHDLCholesterol
	case 12: // HDL_over_Cholesterol=HDL/Cholesterol
		if (ruleUsvs[2].Val(sPointer[2], val_channels[2]) == 0)
			return (true);
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]);
		right2 = ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]);
		right = round(ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * res_factor) / (float)res_factor; // resolution in THIN is 0.1
		right3 = int(ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * res_factor) / (float)res_factor;  // resolution in THIN is 0.1
		if (calc_res > 0)
			return (test_diff(left, right, tolerance, calc_res) && test_diff(left, right2, tolerance, calc_res) &&
					test_diff(left, right3, tolerance, calc_res));
		else
			return test_diff(left, right2, tolerance, calc_res);

	case 6: // MPV=Platelets_Hematocrit/Platelets
		if (ruleUsvs[2].Val(sPointer[2], val_channels[2]) == 0)
			return (true);
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]);
		right = 100 * ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]);
		return test_diff(left, right, tolerance, calc_res);

	case 8:	 // UrineAlbumin_over_Creatinine = UrineAlbumin / UrineCreatinine
	case 13: // HDL_over_LDL=HDL/LDL
	case 15: // Cholesterol_over_HDL=Cholesterol/HDL
	case 18: // LDL_over_HDL=LDL/HDL
		if (ruleUsvs[2].Val(sPointer[2], val_channels[2]) == 0)
			return (true);
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]);
		right2 = ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]);									// calc no resolution
		right = round(ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * res_factor) / res_factor; // resolution in THIN is 0.1
		right3 = int(ruleUsvs[1].Val(sPointer[1], val_channels[1]) / ruleUsvs[2].Val(sPointer[2], val_channels[2]) * res_factor) / res_factor;	// resolution in THIN is 0.1
		if (calc_res > 0)
			return (test_diff(left, right, tolerance, calc_res) && test_diff(left, right2, tolerance, calc_res) &&
					test_diff(left, right3, tolerance, calc_res));
		else
			return test_diff(left, right2, tolerance, calc_res);

	case 5: // Eosinophils#+Monocytes#+Basophils#+Lymphocytes#+Neutrophils#<=WBC
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]) + ruleUsvs[1].Val(sPointer[1], val_channels[1]) + ruleUsvs[2].Val(sPointer[2], val_channels[2]) + ruleUsvs[3].Val(sPointer[3], val_channels[3]) + ruleUsvs[4].Val(sPointer[4], val_channels[4]);
		right = ruleUsvs[5].Val(sPointer[5], val_channels[5]);
		return (left * (1 - tolerance) >= right);

	case 19: // Albumin<=Protein_Total
	case 21: // NRBC<=RBC
	case 22: // CHADS2<=CHADS2_VASC
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]);
		right = ruleUsvs[1].Val(sPointer[1], val_channels[1]);
		return (left * (1 - tolerance) >= right);

	case 7:	 // UrineAlbumin <= UrineTotalProtein
	case 20: // FreeT4<=T4
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]);
		right = 1000 * ruleUsvs[1].Val(sPointer[1], val_channels[1]);
		return (left * (1 - tolerance) >= right); // T4 is nmol/L free T4 is pmol/L ;  Albumin mg/L versus protein g/L

	case 9: // LDL+HDL<=Cholesterol
		left = ruleUsvs[0].Val(sPointer[0]) + ruleUsvs[1].Val(sPointer[1]);
		right = ruleUsvs[2].Val(sPointer[2]);
		return (left * (1 - tolerance) > right);

	case 10: // NonHDLCholesterol + HDL = Cholesterol
		left = ruleUsvs[0].Val(sPointer[0], val_channels[0]) + ruleUsvs[1].Val(sPointer[1], val_channels[1]);
		right = ruleUsvs[2].Val(sPointer[2], val_channels[2]);
		return test_diff(left, right, tolerance, calc_res);

	case 14: // HDL_over_LDL=1/LDL_over_HDL
	case 17: // = HDL_over_Cholestrol = 1 / Cholesterol_over_HDL (opposite)
		if (ruleUsvs[0].Val(sPointer[0], val_channels[0]) == 0 &&
			ruleUsvs[1].Val(sPointer[1], val_channels[1]) == 0)
			return (true);
		test_1 = false;
		if (ruleUsvs[0].Val(sPointer[0], val_channels[0]) != 0)
		{
			left = ruleUsvs[1].Val(sPointer[1], val_channels[1]);
			right = (float)1. / ruleUsvs[0].Val(sPointer[0], val_channels[0]);
			test_1 = test_diff(left, right, tolerance, calc_res);
		}
		if (ruleUsvs[1].Val(sPointer[1], val_channels[1]) != 0)
		{
			left = ruleUsvs[0].Val(sPointer[0], val_channels[0]);
			right = (float)1. / ruleUsvs[1].Val(sPointer[1], val_channels[1]);
			test_1 = test_1 && test_diff(left, right, tolerance, calc_res);
		}
		return test_1; // filter of both tests(when available failed)
	case 23:		   // Check BP 2nd channel is bigger
		return ruleUsvs[0].Val(sPointer[0], val_channels[0]) < ruleUsvs[0].Val(sPointer[0], val_channels[0] + 1);
	default:
		assert(0);
		return false; // return is never executed but eliminates warning
	}
}

//=======================================================================================
// NbrsOutlierCleaner
//=======================================================================================
void RepNbrsOutlierCleaner::init_lists()
{

	req_signals.insert(signalName);
	aff_signals.insert(signalName);
}

//.......................................................................................
int RepNbrsOutlierCleaner::init(map<string, string> &mapper)
{
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepNbrsOutlierCleaner::init]
		if (field == "signal")
		{
			signalName = entry.second;
		}
		else if (field == "time_channel")
			time_channel = med_stoi(entry.second);
		else if (field == "val_channel")
			val_channel = med_stoi(entry.second);
		else if (field == "nbr_time_unit")
			nbr_time_unit = med_time_converter.string_to_type(entry.second);
		else if (field == "nbr_time_width")
			nbr_time_width = med_stoi(entry.second);
		else if (field == "nrem_attr")
			nRem_attr = entry.second;
		else if (field == "ntrim_attr")
			nTrim_attr = entry.second;
		else if (field == "nrem_suff")
			nRem_attr_suffix = entry.second;
		else if (field == "ntrim_suff")
			nTrim_attr_suffix = entry.second;
		//! [RepNbrsOutlierCleaner::init]
	}

	init_lists();
	return MedValueCleaner::init(mapper);
}

// init attributes list
//.......................................................................................
void RepNbrsOutlierCleaner::init_attributes()
{

	string _signal_name = signalName;
	if (val_channel != 0)
		_signal_name += "_" + to_string(val_channel);

	attributes.clear();
	if (!nRem_attr.empty())
		attributes.push_back(nRem_attr);
	if (!nRem_attr_suffix.empty())
		attributes.push_back(_signal_name + "_" + nRem_attr_suffix);

	if (!nTrim_attr.empty())
		attributes.push_back(nTrim_attr);
	if (!nTrim_attr_suffix.empty())
		attributes.push_back(_signal_name + "_" + nTrim_attr_suffix);
}

// Learn bounds
//.......................................................................................
int RepNbrsOutlierCleaner::_learn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_cleaners)
{

	if (params.type == VAL_CLNR_ITERATIVE)
		return iterativeLearn(rep, samples, prev_cleaners);
	else if (params.type == VAL_CLNR_QUANTILE)
		return quantileLearn(rep, samples, prev_cleaners);
	else
	{
		MERR("Unknown cleaning type %d\n", params.type);
		return -1;
	}
}

//.......................................................................................
int RepNbrsOutlierCleaner::iterativeLearn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_cleaners)
{

	// Sanity check
	if (signalId == -1)
	{
		MERR("RepNbrsOutlierCleaner::iterativeLearn - Uninitialized signalId(%s)\n", signalName.c_str());
		return -1;
	}

	// Get all values
	vector<float> values;
	get_values(rep, samples, signalId, time_channel, val_channel, params.range_min, params.range_max, values, prev_cleaners);

	int rc = get_iterative_min_max(values);
	return rc;
}

//.......................................................................................
int RepNbrsOutlierCleaner::quantileLearn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_cleaners)
{

	// Sanity check
	if (signalId == -1)
	{
		MERR("RepNbrsOutlierCleaner::quantileLearn - Uninitialized signalId(%s)\n", signalName.c_str());
		return -1;
	}

	// Get all values
	vector<float> values;
	get_values(rep, samples, signalId, time_channel, val_channel, params.range_min, params.range_max, values, prev_cleaners);

	return get_quantile_min_max(values);
}

// Clean
//.......................................................................................
int RepNbrsOutlierCleaner::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	// Sanity check
	if (signalId == -1)
	{
		MERR("RepNbrsOutlierCleaner::_apply - Uninitialized signalId(%s)\n", signalName.c_str());
		return -1;
	}

	if (time_points.size() != 0 && time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}

	int len;
	allVersionsIterator vit(rec, signalId);
	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{

		rec.uget(signalId, iver, rec.usv);

		len = rec.usv.len;
		vector<int> remove(len);
		vector<pair<int, float>> change(len);
		int nRemove = 0, nChange = 0;

		// Clean
		int verLen = len;
		vector<int> candidates(len, 0);
		vector<int> removed(len, 0);

		for (int i = 0; i < len; i++)
		{
			int itime = rec.usv.Time(i, time_channel);

			if (time_points.size() != 0 && itime > time_points[iver])
			{
				verLen = i;
				break;
			}

			float ival = rec.usv.Val(i, val_channel);

			// Remove ?
			if (params.doRemove && (ival < removeMin - NUMERICAL_CORRECTION_EPS || ival > removeMax + NUMERICAL_CORRECTION_EPS))
			{
				remove[nRemove++] = i;
				removed[i] = 1;
			}
			else if (params.doTrim)
			{
				if (ival < trimMin - NUMERICAL_CORRECTION_EPS)
				{
					candidates[i] = -1;
				}
				else if (ival > trimMax + NUMERICAL_CORRECTION_EPS)
				{
					candidates[i] = 1;
				}
			}
		}

		// Check candidates
		for (int i = 0; i < verLen; i++)
		{
			if (candidates[i] != 0)
			{
				int dir = candidates[i];

				// Get weighted values from neighbours
				double sum = 0, norm = 0;
				double priorSum = 0, priorNorm = 0;
				double postSum = 0, postNorm = 0;

				int time_i = rec.usv.TimeU(i, nbr_time_unit);

				for (int j = 0; j < verLen; j++)
				{

					if (j != i && !removed[j])
					{
						int diff = abs(rec.usv.TimeU(j, nbr_time_unit) - time_i) / nbr_time_width;
						double w = 1.0 / (diff + 1);

						float jval = rec.usv.Val(j, val_channel);

						sum += w * jval;
						norm += w;

						if (j > i)
						{
							postSum += w * jval;
							postNorm += w;
						}
						else
						{
							priorSum += w * jval;
							priorNorm += w;
						}
					}
				}

				// Check it up
				int found_nbr = 0;
				if (norm > 0)
				{
					double win_val = sum / norm;
					if ((dir == 1 && win_val > nbrsMax) || (dir == -1 && win_val < nbrsMin))
						found_nbr = 1;
				}

				if (!found_nbr && priorNorm > 0)
				{
					double win_val = priorSum / priorNorm;
					if ((dir == 1 && win_val > nbrsMax) || (dir == -1 && win_val < nbrsMin))
						found_nbr = 1;
				}

				if (!found_nbr && postNorm > 0)
				{
					double win_val = postSum / postNorm;
					if ((dir == 1 && win_val > nbrsMax) || (dir == -1 && win_val < nbrsMin))
						found_nbr = 1;
				}

				// Should we clip ?
				if (!found_nbr)
				{
					float cval = (dir == 1) ? trimMax : trimMin;
					change[nChange++] = pair<int, float>(i, cval);
				}
			}
		}

		// Apply removals + changes
		change.resize(nChange);
		remove.resize(nRemove);
		if (rec.update(signalId, iver, val_channel, change, remove) < 0)
			return -1;

		// Collect atttibutes
		int idx = 0;
		if (!nRem_attr.empty() && !attributes_mat.empty())
			attributes_mat[iver][idx++] = (float)nRemove;
		if (!nRem_attr_suffix.empty() && !attributes_mat.empty())
			attributes_mat[iver][idx++] = (float)nRemove;
		if (!nTrim_attr.empty() && !attributes_mat.empty())
			attributes_mat[iver][idx++] = (float)nChange;
		if (!nTrim_attr_suffix.empty() && !attributes_mat.empty())
			attributes_mat[iver][idx++] = (float)nChange;
	}

	return 0;
}

//.......................................................................................
void RepNbrsOutlierCleaner::print()
{
	MLOG("RepNbrsOutlierCleaner: signal: %d : doTrim %d trimMax %f trimMin %f : doRemove %d : removeMax %f : removeMin %f : nbrsMax %f : nbrsMin %f\n",
		 signalId, params.doTrim, trimMax, trimMin, params.doRemove, removeMax, removeMin, nbrsMax, nbrsMin);
}

//=======================================================================================
// RepMinimalReq - check requirement and set attributes accordingly
//=======================================================================================
//.......................................................................................
int RepCheckReq::init(map<string, string> &mapper)
{

	time_channels.clear();

	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepCheckReq::init]
		if (field == "signals")
			boost::split(signalNames, entry.second, boost::is_any_of(","));
		else if (field == "time_channels")
		{
			vector<string> channels;
			boost::split(channels, entry.second, boost::is_any_of(","));
			for (string &channel : channels)
				time_channels.push_back(stoi(channel));
		}
		else if (field == "time_unit")
			window_time_unit = med_time_converter.string_to_type(entry.second);
		else if (field == "win_from")
			win_from = med_stoi(entry.second);
		else if (field == "win_to")
			win_to = med_stoi(entry.second);
		else if (field == "attr")
			attrName = entry.second;
		//! [RepCheckReq::init]
	}

	// Take care of time-channels
	if (time_channels.size() == 0)
		time_channels.push_back(0);
	if (time_channels.size() == 1)
	{
		int channel = time_channels[0];
		time_channels.resize(signalNames.size(), channel);
	}

	init_lists();

	return 0;
}

//.......................................................................................
void RepCheckReq::set_signal_ids(MedSignals &sigs)
{

	signalIds.resize(signalNames.size());
	for (int i = 0; i < signalNames.size(); i++)
		signalIds[i] = (sigs.sid(signalNames[i]));
}

//.......................................................................................
void RepCheckReq::init_lists()
{
	req_signals.insert(signalNames.begin(), signalNames.end());
}

//.......................................................................................
void RepCheckReq::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{

	sig_time_units.resize(signalIds.size());
	for (int i = 0; i < signalIds.size(); i++)
		sig_time_units[i] = sigs.Sid2Info[signalIds[i]].time_unit;
}

//.......................................................................................
int RepCheckReq::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	// Should we do anything ?
	if (attributes_mat.empty())
		return 0;

	// Sanity checks
	if (signalIds.size() != signalNames.size())
	{
		MERR("RepCheckReq::_apply - Uninitialized signalId(bad size)\n");
		return -1;
	}

	int id_i = 0;
	for (int signalId : signalIds)
	{
		if (signalId == -1)
		{
			MERR("RepCheckReq::_apply - Uninitialized signalId(%s)\n", signalNames[id_i].c_str());
			return -1;
		}
		++id_i;
	}

	if (time_points.size() != 0 && time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}

	// Loop on versions
	set<int> _signalIds(signalIds.begin(), signalIds.end());
	allVersionsIterator vit(rec, _signalIds);
	vector<UniversalSigVec> usvs(signalIds.size());

	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{

		// NOTE : This is not perfect : we assume that the samples' time-unit is the default time unit
		int time_point = med_time_converter.convert_times(global_default_time_unit, window_time_unit, time_points[iver]);

		int nMissing = 0;
		for (int i = 0; i < signalIds.size(); i++)
		{
			rec.uget(signalIds[i], iver, usvs[i]);

			bool found = false;
			for (int j = 0; j < usvs[i].len; j++)
			{
				if (usvs[i].Time(j, time_channels[i]) > med_time_converter.convert_times(window_time_unit, sig_time_units[i], time_point - win_from))
					break;
				if (usvs[i].Time(j, time_channels[i]) >= med_time_converter.convert_times(window_time_unit, sig_time_units[i], time_point - win_to))
				{
					found = true;
					break;
				}
			}

			if (!found)
				nMissing++;
		}

		// Set attribute
		attributes_mat[iver][0] = (float)nMissing;
	}

	return 0;
}

//=======================================================================================
// RepSimValHandler - handle multiple simultanous values
//=======================================================================================
// Fill req- and aff-signals vectors
//.......................................................................................
void RepSimValHandler::init_lists()
{

	req_signals.insert(signalName);
	aff_signals.insert(signalName);
}

// Init from map
//.......................................................................................
int RepSimValHandler::init(map<string, string> &mapper)
{
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepSimValHandler::init]
		if (field == "signal")
		{
			signalName = entry.second;
		}
		else if (field == "type")
			handler_type = get_sim_val_handle_type(entry.second);
		else if (field == "debug")
			debug = stoi(entry.second) > 0;
		else if (field == "unconditional")
			unconditional = stoi(entry.second) > 0;
		else if (field == "time_channels")
		{
			vector<string> channels;
			boost::split(channels, entry.second, boost::is_any_of(","));
			for (auto &channel : channels)
				time_channels.push_back(stoi(channel));
		}
		else if (field == "attr")
			nHandle_attr = entry.second;
		else if (field == "suff")
			nHandle_attr_suffix = entry.second;
		//! [RepSimValHandler::init]
	}

	init_lists();
	return 0;
}

// init attributes list
//.......................................................................................
void RepSimValHandler::init_attributes()
{

	string _signal_name = signalName;

	if (time_channels.size() > 1 || (time_channels.size() == 1 && time_channels[0] != 0))
	{
		vector<string> channels_s(time_channels.size());
		for (unsigned int i = 0; i < channels_s.size(); i++)
			channels_s[i] = to_string(time_channels[i]);
		_signal_name += "_" + boost::join(channels_s, "_");
	}

	attributes.clear();
	if (!nHandle_attr.empty())
		attributes.push_back(nHandle_attr);
	if (!nHandle_attr_suffix.empty())
		attributes.push_back(_signal_name + "_" + nHandle_attr_suffix);
}

// name to SimValHandleTypes
//.......................................................................................
SimValHandleTypes RepSimValHandler::get_sim_val_handle_type(string &name)
{
	//! [RepSimValHandler::get_sim_val_handle_type]
	if (name == "first" || name == "first_val")
		return SIM_VAL_FIRST_VAL;
	else if (name == "last" || name == "last_val")
		return SIM_VAL_LAST_VAL;
	else if (name == "mean" || name == "avg")
		return SIM_VAL_MEAN;
	else if (name == "rem" || name == "remvoe")
		return SIM_VAL_REM;
	else if (name == "rem_diff" || name == "remove_diff")
		return SIM_VAL_REM_DIFF;
	else if (name == "min")
		return SIM_VAL_MIN;
	else if (name == "max")
		return SIM_VAL_MAX;
	else
		MTHROW_AND_ERR("Unkwnon sim_val_hand_type \'%s\'\n", name.c_str());
	//! [RepSimValHandler::get_sim_val_handle_type]
}

// Get time-channels (if empty)
//.......................................................................................
void RepSimValHandler::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{
	if (signalId < 0)
		return;
	if (time_channels.empty())
	{
		int n = sigs.Sid2Info[signalId].n_time_channels;
		time_channels.resize(n);
		for (int i = 0; i < n; i++)
			time_channels[i] = i;
	}

	nValChannels = sigs.Sid2Info[signalId].n_val_channels;
}

// Apply
//.......................................................................................
int RepSimValHandler::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	// Sanity check
	if (signalId == -1)
	{
		MERR("RepSimValHandler::_apply - Uninitialized signalId(%s)\n", signalName.c_str());
		return -1;
	}

	// Check that we have the correct number of dynamic-versions : one per time-point (if given)
	if (time_points.size() != 0 && time_points.size() != rec.get_n_versions())
	{
		MERR("RepSimValHandler::_apply - nversions mismatch\n");
		return -1;
	}

	int len;
	differentVersionsIterator vit(rec, signalId);
	int total_nTimes = 0;
	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{

		// Do it
		rec.uget(signalId, iver, rec.usv); // get into the internal usv obeject - this statistically saves init time

		len = rec.usv.len;
		vector<int> remove(len);
		vector<pair<int, vector<float>>> change(len);
		int nRemove = 0, nChange = 0, nTimes = 0;

		// Collect
		int start = 0, end = 0;
		for (int i = 1; i < len; i++)
		{

			// No need to clean past the latest relevant time-point (valid only when using a single time-channel == 0)
			if (time_points.size() != 0 && time_channels.size() == 1 && time_channels[0] == 0 && rec.usv.Time(i) > time_points[iver])
				break;

			// Are we simultanous ?
			bool sim = true;
			for (int channel : time_channels)
			{
				if (rec.usv.Time(i, channel) != rec.usv.Time(i - 1, channel))
				{
					sim = false;
					break;
				}
			}

			if (!sim)
			{
				if (end > start)
					handle_block(start, end, rec.usv, remove, nRemove, change, nChange, nTimes);
				start = end = i;
			}
			else
				end++;
		}

		// Handle last block
		if (end > start)
			handle_block(start, end, rec.usv, remove, nRemove, change, nChange, nTimes);

		// Apply removals + changes
		change.resize(nChange);
		remove.resize(nRemove);
		if (rec.update(signalId, iver, change, remove) < 0)
			return -1;

		// Collect atttibutes
		int idx = 0;
		if (!nHandle_attr.empty() && !attributes_mat.empty())
		{
			for (int pVersion = vit.block_first(); pVersion <= vit.block_last(); pVersion++)
				attributes_mat[pVersion][idx] = (float)nTimes;
			idx++;
		}

		if (!nHandle_attr_suffix.empty() && !attributes_mat.empty())
		{
			for (int pVersion = vit.block_first(); pVersion <= vit.block_last(); pVersion++)
				attributes_mat[pVersion][idx] = (float)nTimes;
			idx++;
		}
		total_nTimes += nTimes;
	}

	if (debug && total_nTimes > 0 && verbose_cnt < 1)
	{
		MLOG("RepSimValHandler for %s - patient %d handled %d samples\n",
			 signalName.c_str(), rec.pid, total_nTimes);
		++verbose_cnt;
	}
	return 0;
}

// Utility : handle a block
//.......................................................................................
void RepSimValHandler::handle_block(int start, int end, UniversalSigVec &usv, vector<int> &remove, int &nRemove, vector<pair<int, vector<float>>> &change, int &nChange, int &nTimes)
{

	if (handler_type == SIM_VAL_FIRST_VAL)
	{
		for (int j = start + 1; j <= end; j++)
			remove[nRemove++] = j;
		nTimes++;
	}
	else if (handler_type == SIM_VAL_LAST_VAL)
	{
		for (int j = start; j < end; j++)
			remove[nRemove++] = j;
		nTimes++;
	}
	else if (handler_type == SIM_VAL_MEAN)
	{
		vector<float> sums(nValChannels, 0);
		for (int j = start; j < end; j++)
		{
			for (int iChannel = 0; iChannel < nValChannels; iChannel++)
				sums[iChannel] += usv.Val(j, iChannel);
			remove[nRemove++] = j;
		}
		pair<int, vector<float>> newChange;
		newChange.first = end;
		newChange.second.resize(nValChannels);
		for (int iChannel = 0; iChannel < nValChannels; iChannel++)
			newChange.second[iChannel] = (sums[iChannel] + usv.Val(end, iChannel)) / (end + 1 - start);
		change[nChange++] = newChange;
		nTimes++;
	}
	else if (handler_type == SIM_VAL_MIN)
	{
		vector<float> mins(nValChannels);
		for (int iChannel = 0; iChannel < nValChannels; iChannel++)
			mins[iChannel] = usv.Val(start, iChannel);
		remove[nRemove++] = start;

		for (int j = start + 1; j <= end; j++)
		{
			for (int iChannel = 0; iChannel < nValChannels; iChannel++)
			{
				if (usv.Val(j, iChannel) < mins[iChannel])
					mins[iChannel] = usv.Val(j, iChannel);
			}
			if (j != end)
				remove[nRemove++] = j;
		}

		pair<int, vector<float>> newChange;
		newChange.first = end;
		newChange.second.resize(nValChannels);
		for (int iChannel = 0; iChannel < nValChannels; iChannel++)
			newChange.second[iChannel] = mins[iChannel];
		change[nChange++] = newChange;
		nTimes++;
	}
	else if (handler_type == SIM_VAL_MAX)
	{
		vector<float> maxs(nValChannels);
		for (int iChannel = 0; iChannel < nValChannels; iChannel++)
			maxs[iChannel] = usv.Val(start, iChannel);
		remove[nRemove++] = start;

		for (int j = start + 1; j <= end; j++)
		{
			for (int iChannel = 0; iChannel < nValChannels; iChannel++)
			{
				if (usv.Val(j, iChannel) > maxs[iChannel])
					maxs[iChannel] = usv.Val(j, iChannel);
			}
			if (j != end)
				remove[nRemove++] = j;
		}
		pair<int, vector<float>> newChange;
		newChange.first = end;
		newChange.second.resize(nValChannels);
		for (int iChannel = 0; iChannel < nValChannels; iChannel++)
			newChange.second[iChannel] = maxs[iChannel];
		change[nChange++] = newChange;
		nTimes++;
	}
	else if (handler_type == SIM_VAL_REM)
	{
		for (int j = start; j <= end; j++)
			remove[nRemove++] = j;
		nTimes++;
	}
	else if (handler_type == SIM_VAL_REM_DIFF)
	{
		bool rem = false;
		for (int j = start + 1; j <= end; j++)
		{
			for (int channel = 0; channel < nValChannels; channel++)
			{
				if (usv.Val(j, channel) != usv.Val(j - 1, channel))
				{
					rem = true;
					break;
				}
			}
			if (rem)
				break;
		}

		if (rem)
		{
			for (int j = start; j <= end; j++)
				remove[nRemove++] = j;
			nTimes++;
		}
		else
		{
			// remove all but keep only one copy (last for example)
			for (int j = start; j < end; j++)
				remove[nRemove++] = j;
			nTimes++;
		}
	}
}

//=======================================================================================
// RepCalcSimpleSignals - calculators with no learning stage, can be parametric.
//=======================================================================================
//.......................................................................................
int RepCalcSimpleSignals::init(map<string, string> &mapper)
{
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepCalcSimpleSignals::init]
		if (field == "calculator")
			calculator = entry.second;
		else if (field == "output_signal_type")
			output_signal_type = entry.second;
		else if (field == "missing_value")
			missing_value = stof(entry.second);
		else if (field == "work_channel")
			work_channel = stoi(entry.second);
		else if (field == "time_channel")
			time_channel = stoi(entry.second);
		else if (field == "max_time_search_range")
			max_time_search_range = stoi(entry.second);
		else if (field == "calculator_init_params")
			calculator_init_params = entry.second;
		else if (field == "names")
			boost::split(V_names, entry.second, boost::is_any_of(",:"));
		else if (field == "signals")
			boost::split(signals, entry.second, boost::is_any_of(",:"));
		else if (field == "signals_time_unit")
			signals_time_unit = med_time_converter.string_to_type(entry.second);
		else if (field == "unconditional")
			unconditional = stoi(entry.second) > 0;
		else if (field == "rp_type")
		{
		}
		else
			MTHROW_AND_ERR("Error in RepCalcSimpleSignals::init - Unsupported param \"%s\"\n", field.c_str());
		//! [RepCalcSimpleSignals::init]
	}
	if (signals_time_unit == -1 || signals_time_unit == MedTime::Undefined)
	{
		MWARN("Warning in RepCalcSimpleSignals::init - using signals_time_unit = Days as defualt time unit\n");
		signals_time_unit = MedTime::Days;
	}

	MLOG_D("DBG===> in RepCalcSimpleSignals init: calculator %s , time %d\n", calculator.c_str(), signals_time_unit);
	calculator_logic = SimpleCalculator::make_calculator(calculator);

	if (!calculator_init_params.empty())
	{
		if (calculator_logic->init_from_string(calculator_init_params) < 0)
			return -1;
	}

	calculator_logic->missing_value = missing_value;
	calculator_logic->work_channel = work_channel;
	calculator_logic->output_signal_names = V_names;

	req_signals.clear();
	if (signals.empty() && calc2req_sigs.find(calculator) != calc2req_sigs.end())
		signals = calc2req_sigs.at(calculator);
	if (signals.empty())
		MTHROW_AND_ERR("Error in RepCalcSimpleSignals::init please provide input signals for \"%s\" calculator. no defaluts\n",
					   calculator.c_str());

	for (auto &req_s : signals)
		req_signals.insert(req_s);

	// add V_names
	vector<pair<string, string>> default_virtual_signals;
	calculator_logic->list_output_signals(signals, default_virtual_signals, output_signal_type);
	if (V_names.size() == 0)
	{
		// fetch from default:
		V_names.resize(default_virtual_signals.size());
		for (size_t i = 0; i < default_virtual_signals.size(); ++i)
			V_names[i] = default_virtual_signals[i].first;
	}

	// add names to required, affected and virtual_signals
	aff_signals.clear();
	virtual_signals.clear();
	virtual_signals_generic.clear();

	for (int i = 0; i < V_names.size(); i++)
	{
		aff_signals.insert(V_names[i]);
		virtual_signals_generic.push_back({V_names[i], default_virtual_signals[i].second});
	}
	for (int i = 0; i < signals.size(); i++)
		req_signals.insert(signals[i]);

	calculator_logic->validate_arguments(signals, V_names);
	pass_time_last = calculator_logic->need_time;

	return 0;
}

SimpleCalculator *SimpleCalculator::make_calculator(const string &calc_type)
{
	//! [SimpleCalculator::make_calculator]
	if (calc_type == "ratio" || calc_type == "calc_ratio")
		return new RatioCalculator();
	else if (calc_type == "eGFR" || calc_type == "calc_eGFR")
		return new eGFRCalculator();
	else if (calc_type == "log" || calc_type == "calc_log")
		return new logCalculator();
	else if (calc_type == "sum" || calc_type == "calc_sum")
		return new SumCalculator();
	else if (calc_type == "range" || calc_type == "calc_range")
		return new RangeCalculator();
	else if (calc_type == "multiply" || calc_type == "calc_multiply")
		return new MultiplyCalculator();
	else if (calc_type == "set" || calc_type == "calc_set")
		return new SetCalculator();
	else if (calc_type == "exists" || calc_type == "calc_exists")
		return new ExistsCalculator();
	else if (calc_type == "empty")
		return new EmptyCalculator();
	else if (calc_type == "kfre" || calc_type == "calc_kfre")
		return new KfreCalculator();
	else if (calc_type == "constant" || calc_type == "constant_value")
		return new ConstantValueCalculator();
	else
		HMTHROW_AND_ERR("Error: SimpleCalculator::make_calculator - unsupported calculator: %s\n",
						calc_type.c_str());
	//! [SimpleCalculator::make_calculator]
}

RepCalcSimpleSignals::~RepCalcSimpleSignals()
{
	if (calculator_logic != NULL)
	{
		delete calculator_logic;
		calculator_logic = NULL;
	}
}

mutex RepCalcSimpleSignals_init_tables_mutex;
//.......................................................................................
void RepCalcSimpleSignals::fit_for_repository(MedPidRepository &rep)
{
	if (calculator_logic == NULL)
	{ // recover from serialization
		calculator_logic = SimpleCalculator::make_calculator(calculator);

		if (!calculator_init_params.empty())
		{
			if (calculator_logic->init_from_string(calculator_init_params) < 0)
				MTHROW_AND_ERR("Cannot init calculator from \'%s\'\n", calculator_init_params.c_str());
		}
		calculator_logic->missing_value = missing_value;
	}
	calculator_logic->fit_for_repository(rep, virtual_signals_generic);
}
void RepCalcSimpleSignals::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{
	lock_guard<mutex> guard(RepCalcSimpleSignals_init_tables_mutex);
	static_input_signals.resize(signals.size());

	V_ids.clear();
	sigs_ids.clear();
	for (auto &vsig : V_names)
		V_ids.push_back(sigs.sid(vsig));
	aff_signal_ids.clear();
	aff_signal_ids.insert(V_ids.begin(), V_ids.end());

	// In the next loop it is VERY important to go over items in the ORDER they are given in calc2req
	// This is since we create a vector of sids (sigs_ids) that matches it exactly, and enables a much
	// more efficient code without going to this map for every pid. (See for example the egfr calc function)
	for (const string &rsig : signals)
		sigs_ids.push_back(sigs.sid(rsig));
	req_signal_ids.clear();
	req_signal_ids.insert(sigs_ids.begin(), sigs_ids.end());
	vector<bool> all_sigs_static(T_Last);
	// all_sigs_static[T_TimeStamp] = true;
	all_sigs_static[T_Value] = true;
	all_sigs_static[T_ValShort2] = true;
	all_sigs_static[T_ValShort4] = true;
	unordered_set<string> static_names;
	static_names.insert("BDATE");
	for (size_t i = 0; i < signals.size(); ++i)
		static_input_signals[i] = all_sigs_static[sigs.Sid2Info[sigs_ids[i]].type] || sigs.Sid2Info[sigs_ids[i]].n_time_channels == 0 || static_names.find(signals[i]) != static_names.end();
	if (calculator_logic == NULL)
	{ // recover from serialization
		calculator_logic = SimpleCalculator::make_calculator(calculator);

		if (!calculator_init_params.empty())
		{
			if (calculator_logic->init_from_string(calculator_init_params) < 0)
				MTHROW_AND_ERR("Cannot init calculator from \'%s\'\n", calculator_init_params.c_str());
		}
		calculator_logic->missing_value = missing_value;
	}
	calculator_logic->init_tables(dict, sigs, signals);
	vector<pair<string, string>> default_virtual_signals;
	calculator_logic->list_output_signals(signals, default_virtual_signals, output_signal_type); // init calculator
	pass_time_last = calculator_logic->need_time;

	const SignalInfo &out_si = sigs.Sid2Info[V_ids.front()];
	out_n_val_ch = out_si.n_val_channels;
	out_n_time_ch = out_si.n_time_channels;
	if (out_n_val_ch < work_channel + 1)
		MTHROW_AND_ERR("Error RepCalcSimpleSignals::init_tables- output signal should contain at least %d val channels\n",
					   work_channel + 1);
	for (int i = 0; i < signals.size(); ++i)
	{
		if (sigs_ids[i] < 0)
			MTHROW_AND_ERR("Error RepCalcSimpleSignals::init_tables - can't find input signal %s\n",
						   signals[i].c_str());
		const SignalInfo &si = sigs.Sid2Info[sigs_ids[i]];
		if (si.n_time_channels < time_channel)
			MTHROW_AND_ERR("Error RepCalcSimpleSignals::init_tables - input signal %s should contain %d time channels\n",
						   signals[i].c_str(), time_channel);
		if (si.n_time_channels < out_n_time_ch && !static_input_signals[i])
			MWARN("WARN RepCalcSimpleSignals::init_tables - input signal %s should contain %d time channels\n",
				  signals[i].c_str(), out_n_time_ch);
		if (si.n_val_channels < out_n_val_ch && !static_input_signals[i])
			MWARN("WARN RepCalcSimpleSignals::init_tables - input signal %s should contain %d val channels\n",
				  signals[i].c_str(), out_n_val_ch);
	}
}

void RepCalcSimpleSignals::get_required_signal_categories(unordered_map<string, vector<string>> &signal_categories_in_use) const
{
	// should be called when calculator_logic is not null and list_output_signals was called
	SimpleCalculator *p = calculator_logic;
	if (p == NULL)
	{
		// if after deserialization without call to init_tables
		// realloc and close:
		p = SimpleCalculator::make_calculator(calculator);
		if (!calculator_init_params.empty())
		{
			if (p->init_from_string(calculator_init_params) < 0)
				MTHROW_AND_ERR("Cannot init calculator from \'%s\'\n", calculator_init_params.c_str());
		}
		p->missing_value = missing_value;
		vector<pair<string, string>> default_virtual_signals;
		p->list_output_signals(signals, default_virtual_signals, output_signal_type); // init calculator
	}
	p->get_required_signal_categories(signal_categories_in_use);
}

void RepCalcSimpleSignals::register_virtual_section_name_id(MedDictionarySections &dict)
{
	if (!signals.empty())
	{
		int section_id = dict.section_id(signals.front());
		for (size_t i = 0; i < V_names.size(); ++i)
		{
			if (dict.section_id(V_names[i]) > 0)
				continue; // already defined
			dict.connect_to_section(V_names[i], section_id);
		}
	}
}
//.......................................................................................

bool is_in_time_range(vector<UniversalSigVec> &usvs, vector<int> idx, int active_id,
					  int time_range, int time_unit, int &sum_diff)
{
	int time = usvs[active_id].TimeU(idx[active_id] - 1, time_unit);
	sum_diff = 0; // if not found
	for (size_t i = 0; i < idx.size(); ++i)
	{
		if (i == active_id)
			continue;	 // skip current
		if (idx[i] == 0) // one signal is not yet started - will be happen in future, so waiting for it!
			return false;

		int ref_time = usvs[i].TimeU(idx[i] - 1, time_unit);
		if (time - ref_time > time_range)
			return false;
		sum_diff += time - ref_time;
	}
	return true;
}

bool no_missings(const vector<float> &vals, float missing_value)
{
	for (size_t i = 0; i < vals.size(); ++i)
		if (vals[i] == missing_value)
			return false;
	return true;
}
int RepCalcSimpleSignals::apply_calc_in_time(PidDynamicRec &rec, vector<int> &time_points)
{
	// Check that we have the correct number of dynamic-versions : one per time-point
	if (time_points.size() != 0 && time_points.size() != rec.get_n_versions())
	{
		MERR("RepCalcSimpleSignals::apply_calc_in_time nversions mismatch\n");
		return -1;
	}
	int v_out_sid = V_ids[0];
	if (v_out_sid < 0)
		MTHROW_AND_ERR("Error in RepCalcSimpleSignals::apply_calc_in_time - V_ids is not initialized - bad call\n");
	int n_vals = out_n_val_ch;
	// first lets fetch "static" signals without Time field:

	// MLOG("DBG3===>: apply_calc_in_time: pid %d\n", rec.pid);

	set<int> iteratorSignalIds;
	vector<int> timed_sigs;
	for (size_t i = 0; i < sigs_ids.size(); ++i)
		if (!static_input_signals[i])
		{
			iteratorSignalIds.insert(sigs_ids[i]);
			timed_sigs.push_back(sigs_ids[i]);
		}

	allVersionsIterator vit(rec, iteratorSignalIds);
	rec.usvs.resize(timed_sigs.size());

	int first_ver = vit.init();
	vector<float> static_signals_values(sigs_ids.size(), missing_value);
	for (size_t i = 0; i < static_signals_values.size(); ++i)
		if (static_input_signals[i])
		{
			UniversalSigVec usv;
			rec.uget(sigs_ids[i], first_ver, usv);
			if (usv.len == 0)
				MTHROW_AND_ERR("Error in signal %s - empty. patient %d\n", signals[i].c_str(),
							   rec.pid);
			if (usv.n_val_channels() > 0)
				static_signals_values[i] = usv.Val(0);
			else
				static_signals_values[i] = usv.Time(0);
		}

	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{
		for (size_t i = 0; i < timed_sigs.size(); ++i)
			rec.uget(timed_sigs[i], iver, rec.usvs[i]);
		bool all_non_empty = true;
		for (size_t i = 0; i < rec.usvs.size() && all_non_empty; ++i)
			all_non_empty = rec.usvs[i].len > 0;
		int last_time = -1;
		if (all_non_empty)
		{
			vector<int> idx(timed_sigs.size());
			int active_id = medial::repository::fetch_next_date(rec.usvs, idx);
			int final_size = 0;
			vector<float> v_vals;
			vector<int> v_times;
			int max_diff = -1;
			while (active_id >= 0)
			{
				// iterate on time ordered of signals - Let's try to calc signal:
				bool can_calc = is_in_time_range(rec.usvs, idx, active_id, max_time_search_range, signals_time_unit, max_diff);
				if (can_calc)
				{
					vector<float> collected_vals(sigs_ids.size() + int(pass_time_last));
					int time_idx = 0;
					for (size_t i = 0; i < sigs_ids.size(); ++i)
					{
						if (static_input_signals[i])
						{
							collected_vals[i] = static_signals_values[i];
						}
						else
						{
							if (work_channel >= 0)
							{
								int sel_chanl = work_channel;
								if (sel_chanl >= rec.usvs[time_idx].n_val_channels())
									sel_chanl = rec.usvs[time_idx].n_val_channels() - 1;
								if (sel_chanl < 0)
									collected_vals[i] = rec.usvs[time_idx].Time(idx[time_idx] - 1, 0);
								else
									collected_vals[i] = rec.usvs[time_idx].Val(idx[time_idx] - 1, sel_chanl);
							}
							++time_idx;
						}
					}
					if (pass_time_last)
					{
						if (rec.usvs.empty())
							collected_vals.back() = missing_value;
						else
							collected_vals.back() = rec.usvs[0].Time(idx[0] - 1, time_channel);
					}

					if (no_missings(collected_vals, missing_value))
					{
						float prev_val = missing_value;
						float res_calc;
						bool legal_val = calculator_logic->do_calc(collected_vals, res_calc);
						if (legal_val)
						{
							if (last_time == rec.usvs[active_id].Time(idx[active_id] - 1))
							{
								--final_size; // override last value
								prev_val = v_vals[final_size];
							}
							if (v_times.size() < final_size + out_n_time_ch)
							{
								v_times.resize(static_cast<size_t>(final_size + 1) * out_n_time_ch);
								v_vals.resize(static_cast<size_t>(n_vals) * (final_size + 1));
							}
							for (int kk = 0; kk < out_n_time_ch; ++kk)
							{
								int k_id = kk;
								if (k_id > rec.usvs[active_id].n_time_channels())
									k_id = rec.usvs[active_id].n_time_channels() - 1;
								v_times[final_size * out_n_time_ch + kk] = rec.usvs[active_id].Time(idx[active_id] - 1, k_id);
							}

							v_vals[n_vals * final_size + n_vals - 1] = res_calc;
							for (int kk = 0; kk < n_vals - 1; ++kk) // copy from first
								v_vals[n_vals * final_size + kk] = rec.usvs[0].Val(idx[0] - 1, kk);

							if (v_vals[n_vals * final_size + n_vals - 1] != missing_value)
							{ // insert only legal values (missing_value when ilegal)!
								++final_size;
								last_time = rec.usvs[active_id].Time(idx[active_id] - 1);
							}
							else if (last_time == rec.usvs[active_id].Time(idx[active_id] - 1))
							{
								v_vals[n_vals * final_size + n_vals - 1] = prev_val; // return previous val that was not missing
								// Pay attention it still update rest value channels
								++final_size;
							}
						}
					}
				}

				active_id = medial::repository::fetch_next_date(rec.usvs, idx);
			}
			// pushing virtual data into rec (into orig version)
			rec.set_version_universal_data(v_out_sid, iver, &v_times[0], &v_vals[0], final_size);
		}
	}

	return 0;
}

void RepCalcSimpleSignals::print()
{
	MLOG("RepCalcSimpleSignals: calculator: %s : calculator_init_params %s : max_time_search_range %d signals_time_unit %s signals: %s, V_names: %s, req_signals: %s, aff_signals: %s, work_channel: %d\n",
		 calculator.c_str(), calculator_init_params.c_str(), max_time_search_range, med_time_converter.type_to_string(signals_time_unit).c_str(),
		 medial::io::get_list(signals).c_str(), medial::io::get_list(V_names).c_str(),
		 medial::io::get_list(req_signals).c_str(), medial::io::get_list(aff_signals).c_str(),
		 work_channel);
}

//.......................................................................................
int RepCalcSimpleSignals::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{
	// handle special calculations
	apply_calc_in_time(rec, time_points);

	return 0;
}

int RepCombineSignals::init(map<string, string> &mapper)
{
	vector<string> tokens;
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepCombineSignals::init]
		if (field == "names")
			output_name = entry.second;
		else if (field == "signals")
			signals = boost::split(signals, entry.second, boost::is_any_of(","));
		else if (field == "unconditional")
			unconditional = stoi(entry.second) > 0;
		else if (field == "signal_type")
			signal_type = entry.second;
		else if (field == "rp_type")
		{
		}
		else if (field == "factors")
		{
			boost::split(tokens, entry.second, boost::is_any_of(","));
			factors.resize(tokens.size());
			for (size_t i = 0; i < tokens.size(); ++i)
				factors[i] = stof(tokens[i]);
		}
		else if (field == "factor_channel")
			factor_channel = med_stoi(entry.second);
		else
			MTHROW_AND_ERR("Error in RepCombineSignals::init - Unsupported param \"%s\"\n", field.c_str());
		//! [RepCombineSignals::init]
	}
	if (signals.empty())
		MTHROW_AND_ERR("Error in RepCombineSignals::init - parameter \"signals\" should be passed.\n");
	factors.resize(signals.size(), 1);
	if (output_name.empty())
	{
		output_name = "COMBO_" + signals[0];
		for (size_t i = 1; i < signals.size(); ++i)
			output_name += "_" + signals[i];
	}

	aff_signals.clear();
	aff_signals.insert(output_name);
	req_signals.clear();
	req_signals.insert(signals.begin(), signals.end());
	virtual_signals.clear();
	virtual_signals_generic.clear();
	string const_str = signal_type;
	virtual_signals_generic.push_back(pair<string, string>(output_name, const_str));

	return 0;
}

int RepCombineSignals::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{
	// uses each time point - If have only drug amount  (2nd signal) so using second signal value
	if (time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}
	if (v_out_sid < 0)
		MTHROW_AND_ERR("Error in RepCombineSignals::_apply - v_out_sid is not initialized - bad call\n");
	// first lets fetch "static" signals without Time field:

	set<int> set_ids(sigs_ids.begin(), sigs_ids.end());
	allVersionsIterator vit(rec, set_ids);
	rec.usvs.resize(sigs_ids.size());

	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{
		for (size_t i = 0; i < sigs_ids.size(); ++i)
			rec.uget(sigs_ids[i], iver, rec.usvs[i]);

		vector<int> idx(sigs_ids.size());
		int active_id = medial::repository::fetch_next_date(rec.usvs, idx);
		int final_size = 0;
		vector<float> v_vals;
		vector<int> v_times;
		vector<int> last_time(v_out_n_time_ch, -1);
		vector<float> last_val_ch1(v_out_n_val_ch, MED_MAT_MISSING_VALUE);
		while (active_id >= 0)
		{
			bool not_the_same = false; // remove duplicats from the same signal
			for (int i = 0; i < v_out_n_time_ch; ++i)
				not_the_same |= last_time[i] != rec.usvs[active_id].Time(idx[active_id] - 1, i);
			for (int i = 0; i < v_out_n_val_ch; ++i)
				not_the_same |= last_val_ch1[i] != rec.usvs[active_id].Val(idx[active_id] - 1, i);

			if (!not_the_same)
			{
				active_id = medial::repository::fetch_next_date(rec.usvs, idx);
				continue; // skip same time, and same first channel value
			}

			if (v_times.size() < v_out_n_time_ch * final_size + 1)
			{
				v_times.resize(static_cast<size_t>(v_out_n_time_ch) * (final_size + 1));
				v_vals.resize(static_cast<size_t>(v_out_n_val_ch) * (final_size + 1));
			}
			for (int k = 0; k < v_out_n_time_ch; ++k)
				v_times[v_out_n_time_ch * final_size + k] = rec.usvs[active_id].Time(idx[active_id] - 1, k);
			for (int k = 0; k < v_out_n_val_ch; ++k)
				v_vals[v_out_n_val_ch * final_size + k] = (k == factor_channel ? factors[active_id] : 1) * rec.usvs[active_id].Val(idx[active_id] - 1, k);
			++final_size;
			for (int k = 0; k < v_out_n_time_ch; ++k)
				last_time[k] = rec.usvs[active_id].Time(idx[active_id] - 1, k);
			for (int k = 0; k < v_out_n_val_ch; ++k)
				last_val_ch1[k] = rec.usvs[active_id].Val(idx[active_id] - 1, k);
			active_id = medial::repository::fetch_next_date(rec.usvs, idx);
		}
		// pushing virtual data into rec (into orig version)
		rec.set_version_universal_data(v_out_sid, iver, &v_times[0], &v_vals[0], final_size);
	}

	return 0;
}

void RepCombineSignals::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{
	v_out_sid = sigs.sid(output_name);
	if (v_out_sid < 0)
		MTHROW_AND_ERR("Error in RepCombineSignals::init_tables - virtual output signal %s not found\n",
					   output_name.c_str());
	aff_signal_ids.clear();
	aff_signal_ids.insert(v_out_sid);
	sigs_ids.resize(signals.size());
	const SignalInfo &info = sigs.Sid2Info[v_out_sid];
	v_out_n_time_ch = info.n_time_channels;
	v_out_n_val_ch = info.n_val_channels;

	for (size_t i = 0; i < signals.size(); ++i)
	{
		sigs_ids[i] = sigs.sid(signals[i]);
		if (sigs_ids[i] < 0)
			MTHROW_AND_ERR("Error in RepCombineSignals::init_tables - input signal %s not found\n",
						   signals[i].c_str());
		const SignalInfo &si = sigs.Sid2Info[sigs_ids[i]];

		if (si.n_val_channels < v_out_n_val_ch)
			MTHROW_AND_ERR("ERROR in RepCombineSignals::init_tables - input signal %s should contain %d val channels\n",
						   signals[i].c_str(), v_out_n_val_ch);
		if (si.n_time_channels < v_out_n_time_ch)
			MTHROW_AND_ERR("ERROR in RepCombineSignals::init_tables - input signal %s should contain %d time channels\n",
						   signals[i].c_str(), v_out_n_time_ch);
	}
	req_signal_ids.clear();
	req_signal_ids.insert(sigs_ids.begin(), sigs_ids.end());
}

void RepCombineSignals::fit_for_repository(MedPidRepository &rep)
{
	bool is_virtual = false;
	if (rep.sigs.sid(output_name) > 0)
	{
		const SignalInfo &si = rep.sigs.Sid2Info[rep.sigs.sid(output_name)];
		if (!si.virtual_sig)
			virtual_signals_generic.clear(); // not virtual signal
		else
			is_virtual = true;
	}
	else
		is_virtual = true;

	if (is_virtual && virtual_signals_generic.empty())
		virtual_signals_generic.push_back(pair<string, string>(output_name, signal_type));
}

void RepCombineSignals::register_virtual_section_name_id(MedDictionarySections &dict)
{
	dict.connect_to_section(output_name, dict.section_id(signals.front()));
}

void RepCombineSignals::print()
{
	MLOG("RepCombineSignals: output_name: %s : signals %s : req_signals %s aff_signals %s\n",
		 output_name.c_str(), medial::io::get_list(signals).c_str(), medial::io::get_list(req_signals).c_str(), medial::io::get_list(aff_signals).c_str());
}

int RepSignalRate::init(map<string, string> &mapper)
{
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepSignalRate::init]
		if (field == "names")
			output_name = entry.second;
		else if (field == "input_name")
			input_name = entry.second;
		else if (field == "output_signal_type")
			output_signal_type = entry.second;
		else if (field == "unconditional")
			unconditional = stoi(entry.second) > 0;
		else if (field == "work_channel")
			work_channel = stoi(entry.second);
		else if (field == "factor")
			factor = stof(entry.second);
		else if (field == "rp_type")
		{
		}
		else
			MTHROW_AND_ERR("Error in RepSignalRate::init - Unsupported param \"%s\"\n", field.c_str());
		//! [RepSignalRate::init]
	}
	if (input_name.empty())
		MTHROW_AND_ERR("Error in RepSignalRate::init - input signal should be passed\n");
	if (work_channel > 1)
		MTHROW_AND_ERR("Error in RepSignalRate::init - unsupported work_channel > 1\n");
	aff_signals.clear();
	aff_signals.insert(output_name);
	req_signals.clear();
	req_signals.insert(input_name);
	virtual_signals.clear();
	virtual_signals_generic.clear();
	string const_str = output_signal_type;
	virtual_signals_generic.push_back(pair<string, string>(output_name, const_str));

	return 0;
}

void RepSignalRate::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{
	v_out_sid = sigs.sid(output_name);
	if (v_out_sid < 0)
		MTHROW_AND_ERR("Error in RepSignalRate::init_tables - virtual output signal %s not found\n",
					   output_name.c_str());
	aff_signal_ids.clear();
	aff_signal_ids.insert(v_out_sid);
	in_sid = sigs.sid(input_name);
	if (in_sid < 0)
		MTHROW_AND_ERR("Error in RepSignalRate::init_tables - input signal %s not found\n",
					   input_name.c_str());
	req_signal_ids.clear();
	req_signal_ids.insert(in_sid);

	const SignalInfo &si = sigs.Sid2Info[in_sid];
	const SignalInfo &out_si = sigs.Sid2Info[v_out_sid];
	v_out_n_time_ch = out_si.n_time_channels;
	v_out_n_val_ch = out_si.n_val_channels;
	if (si.n_time_channels < 2)
		MTHROW_AND_ERR("ERROR in RepSignalRate::init_tables - input signal %s should contain at least 2 time channels\n",
					   input_name.c_str());
	if (si.n_val_channels < work_channel + 1)
		MTHROW_AND_ERR("ERROR in RepSignalRate::init_tables - input signal %s should contain %d val channels\n",
					   input_name.c_str(), work_channel + 1);

	if (si.n_time_channels < out_si.n_time_channels)
		MTHROW_AND_ERR("ERROR in RepSignalRate::init_tables - input signal %s should contain at least %d time channels (as output)\n",
					   input_name.c_str(), out_si.n_time_channels);
	if (si.n_val_channels < out_si.n_val_channels)
		MTHROW_AND_ERR("ERROR in RepSignalRate::init_tables - input signal %s should contain at least %d val channels (as output)\n",
					   input_name.c_str(), out_si.n_val_channels);
}

void RepSignalRate::register_virtual_section_name_id(MedDictionarySections &dict)
{
	dict.connect_to_section(output_name, dict.section_id(input_name));
}

int RepSignalRate::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{
	// uses each time point - I have only signal value need to tranform into signal_rate divide by time unit
	if (time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}
	if (v_out_sid < 0)
		MTHROW_AND_ERR("Error in RepCombineSignals::_apply - v_out_sid is not initialized - bad call\n");
	// first lets fetch "static" signals without Time field:

	set<int> set_ids;
	set_ids.insert(in_sid);
	allVersionsIterator vit(rec, set_ids);
	rec.usvs.resize(1);

	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{
		rec.uget(in_sid, iver, rec.usvs[0]);

		vector<float> v_vals;
		vector<int> v_times;
		int out_len = 0;
		for (int i = 0; i < rec.usvs[0].len; ++i)
		{
			int start_time = rec.usvs[0].Time(i);
			int end_time = rec.usvs[0].Time(i, 1);
			int diff_time = med_time_converter.diff_times(end_time, start_time, rec.usvs[0].time_unit(),
														  global_default_time_unit);
			if (diff_time == 0 || start_time == 0 || end_time == 0)
				continue;

			for (int k = 0; k < v_out_n_time_ch; ++k)
				v_times.push_back(rec.usvs[0].Time(i, k));
			// add previous channels
			for (int k = 0; k < v_out_n_val_ch; ++k)
			{
				if (k != work_channel)
					v_vals.push_back(rec.usvs[0].Val(i, k));
				else // update current channel to be rate
					v_vals.push_back(factor * rec.usvs[0].Val(i, k) / diff_time);
			}
			++out_len;
		}
		// pushing virtual data into rec (into orig version)
		if (out_len > 0)
			rec.set_version_universal_data(v_out_sid, iver, &v_times[0], &v_vals[0], out_len);
	}

	return 0;
}

void RepSignalRate::print()
{
	MLOG("RepSignalRate: input_name: %s, output_name: %s : factor: %f, work_channel: %d, req_signals %s aff_signals %s\n",
		 input_name.c_str(), output_name.c_str(), factor, work_channel, medial::io::get_list(req_signals).c_str(), medial::io::get_list(aff_signals).c_str());
}

int RepSplitSignal::init(map<string, string> &mapper)
{
	vector<string> tokens;
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepSplitSignal::init]
		if (field == "input_name")
			input_name = entry.second;
		else if (field == "names")
			boost::split(names, entry.second, boost::is_any_of(","));
		else if (field == "sets")
			boost::split(sets, entry.second, boost::is_any_of(","));
		else if (field == "val_channel")
			val_channel = med_stoi(entry.second);
		else if (field == "output_signal_type")
			output_signal_type = entry.second;
		else if (field == "factors")
		{
			boost::split(tokens, entry.second, boost::is_any_of(","));
			factors.resize(tokens.size());
			for (size_t i = 0; i < tokens.size(); ++i)
				factors[i] = stof(tokens[i]);
		}
		else if (field == "unconditional")
			unconditional = stoi(entry.second) > 0;
		else if (field == "rp_type")
		{
		}
		else
			MTHROW_AND_ERR("Error in RepSplitSignal::init - Unsupported param \"%s\"\n", field.c_str());
		//! [RepSplitSignal::init]
	}
	if (input_name.empty())
		MTHROW_AND_ERR("ERROR in RepSplitSignal::init - must provide input_name\n");
	if (names.size() != 2)
		MTHROW_AND_ERR("ERROR in RepSplitSignal::init - must provide only 2 names for output\n");
	if (sets.empty())
		MTHROW_AND_ERR("ERROR in RepSplitSignal::init - must provide sets\n");
	factors.resize(names.size(), 1);

	aff_signals.clear();
	aff_signals.insert(names.begin(), names.end());
	req_signals.clear();
	req_signals.insert(input_name);
	virtual_signals.clear();

	virtual_signals_generic.clear();
	/*in_sid = rep.sigs.sid(input_name);
	if (in_sid < 0)
		MTHROW_AND_ERR("Error can't find signal %s in rep\n", input_name.c_str());
	const SignalInfo &s_info = rep.sigs.Sid2Info[in_sid];
	string type_gen = generate_signal_sig(s_info);*/
	// string type_gen = "T(i),T(i),V(f),V(f)";

	for (size_t i = 0; i < names.size(); ++i)
		virtual_signals_generic.push_back(pair<string, string>(names[i], output_signal_type));

	return 0;
}

string generate_signal_sig(const SignalInfo &s_info)
{
	stringstream str;

	if (s_info.n_time_channels > 0)
		str << "T(" << s_info.time_channel_types[0] << ")";
	for (size_t i = 1; i < s_info.n_time_channels; ++i)
		str << ",T(" << GenericSigVec::type_enc::decode(s_info.time_channel_types[i]) << ")";
	if (s_info.n_val_channels > 0)
		str << "V(" << s_info.val_channel_types[0] << ")";
	for (size_t i = 1; i < s_info.n_val_channels; ++i)
		str << ",V(" << GenericSigVec::type_enc::decode(s_info.val_channel_types[i]) << ")";

	return str.str();
}

void RepSplitSignal::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{
	// init Flags:
	int section_id = dict.section_id(input_name);

	dict.prep_sets_lookup_table(section_id, sets, Flags);

	in_sid = sigs.sid(input_name);
	if (in_sid < 0)
		MTHROW_AND_ERR("Error in RepSplitSignal::init_tables - input signal %s not found\n",
					   input_name.c_str());
	req_signal_ids.clear();
	req_signal_ids.insert(in_sid);

	V_ids.resize(names.size());
	for (size_t i = 0; i < V_ids.size(); ++i)
	{
		V_ids[i] = sigs.sid(names[i]);
		if (V_ids[i] < 0)
			MTHROW_AND_ERR("Error in RepSplitSignal::init_tables - virtual output signal %s not found\n",
						   names[i].c_str());
	}
	aff_signal_ids.clear();
	aff_signal_ids.insert(V_ids.begin(), V_ids.end());

	const SignalInfo &out_si = sigs.Sid2Info[V_ids.front()];
	const SignalInfo &si = sigs.Sid2Info[in_sid];
	v_out_n_time_ch = out_si.n_time_channels;
	v_out_n_val_ch = out_si.n_val_channels;

	if (si.n_val_channels < val_channel)
		MTHROW_AND_ERR("ERROR in RepSplitSignal::init_tables - input signal %s should contain  at least %d val channels\n",
					   input_name.c_str(), val_channel + 1);
	if (si.n_val_channels < v_out_n_val_ch)
		MTHROW_AND_ERR("ERROR in RepSplitSignal::init_tables - input signal %s should contain at least %d val channels (as defined in output)\n",
					   input_name.c_str(), v_out_n_val_ch);
	if (si.n_val_channels < v_out_n_time_ch)
		MTHROW_AND_ERR("ERROR in RepSplitSignal::init_tables - input signal %s should contain at least %d time channels (as defined in output)\n",
					   input_name.c_str(), v_out_n_time_ch);
}

void RepSplitSignal::register_virtual_section_name_id(MedDictionarySections &dict)
{
	for (size_t i = 0; i < names.size(); ++i)
		dict.connect_to_section(names[i], dict.section_id(input_name));
}

int RepSplitSignal::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{
	if (time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}
	for (size_t i = 0; i < V_ids.size(); ++i)
		if (V_ids[i] < 0)
			MTHROW_AND_ERR("Error in RepSplitSignal::_apply - V_ids is not initialized - bad call\n");

	set<int> set_ids;
	set_ids.insert(in_sid);
	allVersionsIterator vit(rec, set_ids);
	rec.usvs.resize(1);

	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{
		rec.uget(in_sid, iver, rec.usvs[0]);

		vector<vector<int>> v_times(names.size());
		vector<vector<float>> v_vals(names.size());
		vector<int> new_len(names.size());
		for (int i = 0; i < rec.usvs[0].len; ++i)
		{
			int kv_idx = (int)rec.usvs[0].Val(i, val_channel);
			// if (kv_idx >= Flags.size())
			//	MTHROW_AND_ERR("Error got value %d outside dict(%zu)\n", kv_idx, Flags.size());
			int idx = Flags[kv_idx];

			for (int j = 0; j < v_out_n_time_ch; ++j)
				v_times[idx].push_back(rec.usvs[0].Time(i, j));
			for (int j = 0; j < v_out_n_val_ch; ++j)
				v_vals[idx].push_back(rec.usvs[0].Val(i, j) * factors[j]);
			++new_len[idx];
		}
		// pushing virtual data into rec (into orig version)
		for (size_t i = 0; i < names.size(); ++i)
			if (!v_times[i].empty())
				rec.set_version_universal_data(V_ids[i], iver, &v_times[i][0], &v_vals[i][0], new_len[i]);
	}

	return 0;
}

void RepSplitSignal::print()
{
	MLOG("RepSplitSignal: input_name: %s, names: %s, req_signals %s aff_signals %s\n",
		 input_name.c_str(), medial::io::get_list(names).c_str(), medial::io::get_list(req_signals).c_str(), medial::io::get_list(aff_signals).c_str());
}

void RepSplitSignal::get_required_signal_categories(unordered_map<string, vector<string>> &signal_categories_in_use) const
{
	signal_categories_in_use[input_name] = sets;
}

//=======================================================================================
// RepAggregationPeriod
//=======================================================================================

int RepAggregationPeriod::init(map<string, string> &mapper)
{
	vector<string> tokens;
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepAggregationPeriod::init]
		if (field == "input_name")
			input_name = entry.second;
		else if (field == "output_name")
			output_name = entry.second;
		else if (field == "sets")
			boost::split(sets, entry.second, boost::is_any_of(","));
		else if (field == "rp_type")
		{
		}
		else if (field == "period")
			period = med_stoi(entry.second);
		else if (field == "time_unit_sig")
			time_unit_sig = med_time_converter.string_to_type(entry.second);
		else if (field == "time_unit_win")
			time_unit_win = med_time_converter.string_to_type(entry.second);
		else
			MTHROW_AND_ERR("Error in RepAggregationPeriod::init - Unsupported param \"%s\"\n", field.c_str());
		//! [RepAggregationPeriod::init]
	}
	if (input_name.empty())
		MTHROW_AND_ERR("ERROR in RepAggregationPeriod::init - must provide input_name\n");
	if (output_name.empty())
		MTHROW_AND_ERR("ERROR in RepAggregationPeriod::init - must provide output_name\n");
	if (sets.empty())
		MTHROW_AND_ERR("ERROR in RepAggregationPeriod::init - must provide sets\n");
	if (period == 0)
		MLOG("WARNING in RepAggregationPeriod::init  - period set to default value: %d\n", period);

	aff_signals.insert(output_name);
	req_signals.insert(input_name);
	virtual_signals.clear();
	virtual_signals_generic.clear();
	virtual_signals_generic.push_back(pair<string, string>(output_name, "T(l,l)"));

	return 0;
}

void RepAggregationPeriod::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{
	// init:
	int section_id = dict.section_id(input_name);
	dict.prep_sets_lookup_table(section_id, sets, lut);

	in_sid = sigs.sid(input_name);
	if (in_sid < 0)
		MTHROW_AND_ERR("Error in RepAggregationPeriod::init_tables - input signal %s not found\n",
					   input_name.c_str());
	req_signal_ids.insert(in_sid);

	V_ids.resize(1);
	V_ids[0] = sigs.sid(output_name);
	if (V_ids[0] < 0)
		MTHROW_AND_ERR("Error in RepAggregationPeriod::init_tables - virtual output signal %s not found\n",
					   output_name.c_str());

	aff_signal_ids.insert(V_ids.begin(), V_ids.end());
}

int RepAggregationPeriod::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{
	if (time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}
	for (size_t i = 0; i < V_ids.size(); ++i)
		if (V_ids[i] < 0)
			MTHROW_AND_ERR("Error in RepAggregationPeriod::_apply - V_ids is not initialized - bad call\n");

	set<int> set_ids;
	set_ids.insert(in_sid);
	allVersionsIterator vit(rec, set_ids);
	rec.usvs.resize(1);

	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{
		rec.uget(in_sid, iver, rec.usvs[0]);

		vector<int> v_times;
		vector<float> v_vals;
		if (rec.usvs[0].len < 1) // in case this version of the signal is empty
			continue;

		int start_time_u = 0, end_time_u = 0;
		bool first = true;
		for (int i = 0; i < rec.usvs[0].len; ++i)
		{									  // find remaining valid values
			if (lut[rec.usvs[0].Val(i)] == 0) // value not in set
				continue;

			int time = med_time_converter.convert_times(time_unit_sig, time_unit_win, rec.usvs[0].Time(i));
			if (first)
			{
				start_time_u = time;
				end_time_u = start_time_u + period;

				first = false;
			}
			else
			{
				if (time <= end_time_u)
				{ // extend end period, Not reach end, no hole.
					end_time_u = max(end_time_u, time + period);
				}
				else
				{ // found a signal that is not included in the current period, close old period and open new one
					int start_time = med_time_converter.convert_times(time_unit_win, time_unit_sig, start_time_u);
					int end_time = med_time_converter.convert_times(time_unit_win, time_unit_sig, end_time_u);
					v_times.push_back(start_time);
					v_times.push_back(end_time);

					start_time_u = time;
					end_time_u = time + period;
				}
			}
		}
		if (!first)
		{ // else - no valid set values were found
			int start_time = med_time_converter.convert_times(time_unit_win, time_unit_sig, start_time_u);
			int end_time = med_time_converter.convert_times(time_unit_win, time_unit_sig, end_time_u);
			v_times.push_back(start_time);
			v_times.push_back(end_time);
			// pushing virtual data into rec (into orig version)
			rec.set_version_universal_data(V_ids[0], iver, &v_times[0], &v_vals[0], (int)v_times.size() / 2);
		}
	}

	return 0;
}

void RepAggregationPeriod::print()
{
	MLOG("RepAggregationPeriod: input_name: %s, output_name: %s, req_signals %s aff_signals %s\n",
		 input_name.c_str(), output_name.c_str(), medial::io::get_list(req_signals).c_str(), medial::io::get_list(aff_signals).c_str());
}

void RepAggregationPeriod::get_required_signal_categories(unordered_map<string, vector<string>> &signal_categories_in_use) const
{
	signal_categories_in_use[input_name] = sets;
}

//=======================================================================================
// BasicRangeCleaner
//=======================================================================================

bool _is_numeric(const std::string &s)
{
	return !s.empty() && std::find_if(s.begin(),
									  s.end(), [](char c)
									  { return !std::isdigit(c); }) == s.end();
}

static unordered_map<string, int> range_op_enum_map = {
	{"all", (int)range_op_type::all},
	{"first", (int)range_op_type::first},
	{"last", (int)range_op_type::last}};

range_op_type get_range_op(const string &op)
{
	if (range_op_enum_map.find(op) == range_op_enum_map.end())
	{
		string opts = medial::io::get_list(range_op_enum_map, ",");
		MTHROW_AND_ERR("Error can't find option %s, available options are: [%s]\n",
					   op.c_str(), opts.c_str());
	}
	return range_op_type(range_op_enum_map.at(op));
}

int RepBasicRangeCleaner::init(map<string, string> &mapper)
{
	output_type = -1;
	string output_type_s = "";
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepBasicRangeCleaner::init]
		if (field == "signal_name")
		{
			signal_name = entry.second;
		}
		else if (field == "rp_type")
		{
		}
		else if (field == "ranges_sig_name")
		{
			ranges_name = entry.second;
		}
		else if (field == "output_name")
		{
			output_name = entry.second;
		}
		else if (field == "time_channel")
			time_channel = med_stoi(entry.second);
		else if (field == "range_time_channel")
			range_time_channel = med_stoi(entry.second);
		else if (field == "get_values_in_range")
			get_values_in_range = med_stoi(entry.second);
		else if (field == "range_operator")
			range_operator = get_range_op(entry.second);
		else if (field == "range_val_channel")
			range_val_channel = med_stoi(entry.second);
		else if (field == "regex_on_sets")
			regex_on_sets = (bool)med_stoi(entry.second);
		else if (field == "last_n")
		{
			last_n = med_stoi(entry.second);
			do_on_last_n = true;
		}
		else if (field == "sets")
		{
			boost::split(sets, entry.second, boost::is_any_of(",;"));
		}
		else if (field == "output_type")
		{
			if (_is_numeric(entry.second))
				output_type = med_stoi(entry.second);
			else
				output_type_s = entry.second;
		} // needs to match the input signal type! defaults to range-value signal (3)
		else
			MTHROW_AND_ERR("Error in RepBasicRangeCleaner::init - Unsupported param \"%s\"\n", field.c_str());
		//! [RepBasicRangeCleaner::init]
	}
	if (signal_name.empty())
		MTHROW_AND_ERR("ERROR in RepBasicRangeCleaner::init - must provide signal_name\n");
	if (ranges_name.empty())
		MTHROW_AND_ERR("ERROR in RepBasicRangeCleaner::init - must provide ranges_sig_name\n");
	if (output_name.empty())
	{
		output_name = signal_name + "_" + ranges_name;
		if (sets.size() > 0)
		{
			for (int i = 0; i < sets.size(); i++)
				output_name += "_" + sets[i];
		}

		if (do_on_last_n)
			output_name += "_last_" + to_string(last_n);

		// MLOG("WARNING in RepBasicRangeCleaner::init - no output_name provided, using input signal combination: %s\n", output_name.c_str());
	}

	req_signals.insert(signal_name);
	req_signals.insert(ranges_name);
	aff_signals.insert(output_name);

	virtual_signals.clear();
	virtual_signals_generic.clear();

	if (output_type_s.empty())
		virtual_signals_generic.push_back(pair<string, string>(output_name, GenericSigVec::get_type_generic_spec(SigType(output_type))));
	else
		virtual_signals_generic.push_back(pair<string, string>(output_name, output_type_s));
	return 0;
}

void RepBasicRangeCleaner::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{

	signal_id = sigs.sid(signal_name);
	ranges_id = sigs.sid(ranges_name);
	output_id = sigs.sid(output_name);
	req_signal_ids.insert(signal_id);
	req_signal_ids.insert(ranges_id);
	aff_signal_ids.insert(output_id);

	if (range_val_channel >= 0 && !sets.empty())
	{
		int sec_id = dict.section_id(ranges_name);
		if (regex_on_sets)
		{
			unordered_set<string> aggregated_values;
			for (auto &s : sets)
			{
				vector<string> curr_set;
				dict.dicts[sec_id].get_regex_names(".*" + s + ".*", curr_set);
				aggregated_values.insert(curr_set.begin(), curr_set.end());
			}
			sets.clear();
			sets.insert(sets.begin(), aggregated_values.begin(), aggregated_values.end());
		}

		dict.prep_sets_lookup_table(sec_id, sets, lut);
	}

	const SignalInfo &in_s = sigs.Sid2Info.at(signal_id);
	const SignalInfo &out_s = sigs.Sid2Info.at(output_id);

	if (in_s.n_time_channels != out_s.n_time_channels)
		MTHROW_AND_ERR("Error RepBasicRangeCleaner::init_tables - output signal can't have less/more time channels than input signal\n");
	if (in_s.n_val_channels != out_s.n_val_channels)
		MTHROW_AND_ERR("Error RepBasicRangeCleaner::init_tables - output signal can't have less/more val channels than input signal\n");
}

void RepBasicRangeCleaner::register_virtual_section_name_id(MedDictionarySections &dict)
{
	int sec_id = dict.section_id(signal_name);
	dict.connect_to_section(output_name, sec_id);
}

bool RepBasicRangeCleaner::get_last_n_value(int time, const UniversalSigVec &range_sig, float &last_value)
{
	bool found_flag = false;
	int i = 0;
	if (range_sig.len > 0)
	{
		// find first i before time
		for (i = range_sig.len - 1; i >= 0; i--)
		{
			if (range_sig.Time(i) <= time)
				break;
		}
	}
	if (i - last_n >= 0)
	{
		last_value = range_sig.Val(i - last_n, range_val_channel);
		found_flag = true;
	}
	return found_flag;
}

int RepBasicRangeCleaner::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	if (signal_id == -1)
	{
		MERR("Uninitialized signal_id\n");
		return -1;
	}
	if (ranges_id == -1)
	{
		MERR("Uninitialized ranges_id\n");
		return -1;
	}
	if (output_id == -1)
	{
		MERR("Uninitialized output_id\n");
		return -1;
	}
	// Check that we have the correct number of dynamic-versions : one per time-point (if given)
	if (time_points.size() != 0 && time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}
	int len;
	set<int> set_ids;
	set_ids.insert(signal_id);
	set_ids.insert(ranges_id);
	allVersionsIterator vit(rec, set_ids);
	rec.usvs.resize(2);
	int tp_idx = 0;
	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{
		// setup
		rec.uget(signal_id, iver, rec.usvs[0]); // original signal
		rec.uget(ranges_id, iver, rec.usvs[1]); // range signal
		int time_channels = rec.usvs[0].n_time_channels();
		int val_channels = rec.usvs[0].n_val_channels();
		len = rec.usvs[0].len;
		vector<int> v_times(static_cast<size_t>(len) * static_cast<size_t>(time_channels));
		vector<float> v_vals(static_cast<size_t>(len) * static_cast<size_t>(val_channels));

		float last_value_n;
		bool found_last_n = false;
		if (do_on_last_n)
			found_last_n = get_last_n_value(time_points[tp_idx], rec.usvs[1], last_value_n);

		// Collect elements to keep
		int nKeep = 0;
		int j = 0;
		for (int i = 0; i < len; i++)
		{ // iterate over input signal
			int time = rec.usvs[0].Time(i, time_channel);
			// remove element only if it doesn't appear in any range
			bool doRemove = true;
			switch (range_operator)
			{
			case all:
				// increase till end or till end_time of signal passed time (sorted so no need to search after)
				// or if has filter on sets - skip is not in set
				while ((j < rec.usvs[1].len) && ((time > rec.usvs[1].Time(j, 1)) ||
												 ((!lut.empty() && lut[(int)rec.usvs[1].Val(j, range_val_channel)]) || (found_last_n && (rec.usvs[1].Val(j, range_val_channel) == last_value_n)))))
				{
					++j;
				}
				if (j < rec.usvs[1].len && rec.usvs[1].Time(j, range_time_channel) > time_points[tp_idx])
					j = -1; // mark as no match, passed prediction time
				break;
			case first:
				if (i == 0)
				{
					j = 0;
					// find first occourence if has set:

					while (j < rec.usvs[1].len && !((!lut.empty() && lut[(int)rec.usvs[1].Val(j, range_val_channel)]) || (found_last_n && (rec.usvs[1].Val(j, range_val_channel) == last_value_n))))
						++j;
					if (j < rec.usvs[1].len && rec.usvs[1].Time(j, range_time_channel) > time_points[tp_idx])
						j = -1; // mark as no match
				}
				break;
			case last:
				if (rec.usvs[1].len > 0 && i == 0)
				{ // can do once - only in first time
					j = rec.usvs[1].len - 1;
					// last till time_point:
					while (j >= 0 && rec.usvs[1].Time(j, range_time_channel) > time_points[tp_idx])
						--j;

					while (j >= 0 && !((!lut.empty() && lut[(int)rec.usvs[1].Val(j, range_val_channel)]) || (found_last_n && (rec.usvs[1].Val(j, range_val_channel) == last_value_n))))
						--j;
				}
				break;
			default:
				MTHROW_AND_ERR("Not Supported\n");
			}

			if (j >= 0 && j < rec.usvs[1].len && time >= rec.usvs[1].Time(j, 0) && time <= rec.usvs[1].Time(j, 1))
				doRemove = false; // if signal time in range - keep item. doRemove=false

			if ((doRemove && get_values_in_range) || ((!doRemove) && (!get_values_in_range)))
				doRemove = true;
			else
				doRemove = false;
			// MLOG("remove : %d , i: %d, j :%d, time: %d, time_0: %d, time_1: %d, lut: %d last: %d \n",doRemove,i,j,time, rec.usvs[1].Time(j, 0), rec.usvs[1].Time(j, 1), (!lut.empty() && lut[(int)rec.usvs[1].Val(j, range_val_channel)]), (found_last_n && (rec.usvs[1].Val(j, range_val_channel) == last_value_n)));
			if (!doRemove)
			{
				for (int t = 0; t < time_channels; t++)
					v_times[nKeep * time_channels + t] = rec.usvs[0].Time(i, t);
				for (int v = 0; v < val_channels; v++)
					v_vals[nKeep * val_channels + v] = rec.usvs[0].Val(i, v);
				++nKeep;
			}
		}

		v_times.resize(static_cast<size_t>(nKeep) * static_cast<size_t>(time_channels));
		v_vals.resize(static_cast<size_t>(nKeep) * static_cast<size_t>(time_channels));
		// v_times and v_vals are likely longer than necessary, it's ok because nKeep defines which part of the vector is used.
		rec.set_version_universal_data(output_id, iver, &v_times[0], &v_vals[0], nKeep);
		++tp_idx;
	}

	return 0;
}

void RepBasicRangeCleaner::fit_for_repository(MedPidRepository &rep)
{
	bool is_virtual = false;
	if (rep.sigs.sid(output_name) > 0)
	{
		const SignalInfo &si = rep.sigs.Sid2Info[rep.sigs.sid(output_name)];
		if (!si.virtual_sig)
			virtual_signals_generic.clear(); // not virtual signal
		else
			is_virtual = true;
	}
	else
		is_virtual = true;

	if (is_virtual && virtual_signals_generic.empty())
		virtual_signals_generic.push_back(pair<string, string>(output_name, GenericSigVec::get_type_generic_spec(SigType(output_type))));
}

void RepBasicRangeCleaner::print()
{
	MLOG("RepBasicRangeCleaner: signal: %d %s : t_channel %d : ranges_signal: %d %s : output_signal: %d %s\n",
		 signal_id, signal_name.c_str(), time_channel, ranges_id, ranges_name.c_str(), output_id, output_name.c_str());
}

int RepAggregateSignal::init(map<string, string> &mapper)
{
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepAggregateSignal::init]
		if (field == "output_name")
			output_name = entry.second;
		else if (field == "signal")
			signalName = entry.second;
		else if (field == "unconditional")
			unconditional = med_stoi(entry.second) > 0;
		else if (field == "work_channel")
			work_channel = med_stoi(entry.second);
		else if (field == "start_time_channel")
			start_time_channel = med_stoi(entry.second);
		else if (field == "end_time_channel")
			end_time_channel = med_stoi(entry.second);
		else if (field == "time_window")
			time_window = med_stoi(entry.second);
		else if (field == "time_unit")
			time_unit = med_time_converter.string_to_type(entry.second);
		else if (field == "factor")
			factor = med_stof(entry.second);
		else if (field == "drop_missing_rate")
			drop_missing_rate = med_stof(entry.second);
		else if (field == "buffer_first")
			buffer_first = med_stoi(entry.second) > 0;
		else if (field == "output_signal_type")
			output_signal_type = entry.second;
		else if (field == "rp_type")
		{
		}
		else
			MTHROW_AND_ERR("Error in RepAggregateSignal::init - Unsupported param \"%s\"\n", field.c_str());
		//! [RepAggregateSignal::init]
	}
	if (signalName.empty())
		MTHROW_AND_ERR("Error in RepAggregateSignal::init - signal should be passed\n");
	if (time_window == 0)
		MTHROW_AND_ERR("Error in RepAggregateSignal::init - time_window should be passed\n");
	if (output_name.empty())
		MTHROW_AND_ERR("Error in RepAggregateSignal::init - output_name should be passed\n");

	aff_signals.clear();
	aff_signals.insert(output_name);
	req_signals.clear();
	req_signals.insert(signalName);
	virtual_signals.clear();
	virtual_signals_generic.clear();
	virtual_signals_generic.push_back(pair<string, string>(output_name, output_signal_type));

	return 0;
}

void RepAggregateSignal::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{
	v_out_sid = sigs.sid(output_name);
	if (v_out_sid < 0)
		MTHROW_AND_ERR("Error in RepAggregateSignal::init_tables - virtual output signal %s not found\n",
					   output_name.c_str());
	aff_signal_ids.clear();
	aff_signal_ids.insert(v_out_sid);
	in_sid = sigs.sid(signalName);
	if (in_sid < 0)
		MTHROW_AND_ERR("Error in RepAggregateSignal::init_tables - input signal %s not found\n",
					   signalName.c_str());
	req_signal_ids.clear();
	req_signal_ids.insert(in_sid);

	const SignalInfo &si = sigs.Sid2Info[in_sid];

	if (si.n_time_channels <= max(end_time_channel, start_time_channel))
		MTHROW_AND_ERR("ERROR in RepAggregateSignal::init_tables - input signal %s should contain [%d, %d] time channels\n",
					   signalName.c_str(), start_time_channel, end_time_channel);
	if (si.n_val_channels < work_channel + 1)
		MTHROW_AND_ERR("ERROR in RepAggregateSignal::init_tables - input signal %s should contain %d val channels\n",
					   signalName.c_str(), work_channel + 1);

	const SignalInfo &out_si = sigs.Sid2Info[v_out_sid];
	v_out_n_time_ch = out_si.n_time_channels;
	v_out_n_val_ch = out_si.n_val_channels;
	if (si.n_val_channels < v_out_n_val_ch)
		MTHROW_AND_ERR("ERROR in RepAggregateSignal::init_tables - input signal %s should contain %d val channels\n",
					   signalName.c_str(), v_out_n_val_ch);
	if (si.n_time_channels < v_out_n_time_ch)
		MTHROW_AND_ERR("ERROR in RepAggregateSignal::init_tables - input signal %s should contain %d time channels\n",
					   signalName.c_str(), v_out_n_time_ch);
}

void RepAggregateSignal::register_virtual_section_name_id(MedDictionarySections &dict)
{
	dict.connect_to_section(output_name, dict.section_id(signalName));
}

//-------------------------------------------------------------------------------------------------------
// RepCreateBitSignal
//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------
int RepCreateBitSignal::init(map<string, string> &mapper)
{
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepAggregateSignal::init]
		if (field == "in_sig")
			in_sig = entry.second;
		else if (field == "out_virtual")
			out_virtual = entry.second;
		else if (field == "t_chan")
			t_chan = med_stoi(entry.second);
		else if (field == "c_chan")
			c_chan = med_stoi(entry.second);
		else if (field == "duration_chan")
			duration_chan = med_stoi(entry.second);
		else if (field == "min_jitter")
			get_min_jitters(entry.second);
		else if (field == "min_duration")
			min_duration = med_stoi(entry.second);
		else if (field == "max_duration")
			max_duration = med_stoi(entry.second);
		else if (field == "duration_add")
			duration_add = med_stof(entry.second);
		else if (field == "duration_mult")
			duration_mult = med_stof(entry.second);
		else if (field == "dont_look_back")
			dont_look_back = med_stoi(entry.second);
		else if (field == "min_clip_time")
			min_clip_time = med_stoi(entry.second);
		else if (field == "last_clip_period")
			last_clip_period = med_stoi(entry.second);
		else if (field == "time_unit_sig")
			time_unit_sig = med_time_converter.string_to_type(entry.second);
		else if (field == "time_unit_duration")
			time_unit_duration = med_time_converter.string_to_type(entry.second);
		else if (field == "print_dict")
			print_dict = entry.second;
		else if (field == "time_channels")
			time_channels = med_stoi(entry.second);
		else if (field == "min_durations")
		{
			vector<string> fs;
			boost::split(fs, entry.second, boost::is_any_of(","));
			min_durations.clear();
			for (auto f : fs)
				min_durations.push_back(stoi(f));
		}
		else if (field == "categories")
		{

			// format is for example: Metformin:ATC_A10B_A__,ATC_A10B_D03,ATC_A10B_D07:Sulfonylureas:ATC_A10B_B__:SGLT2:ATC_A10B_K__,ATC_A10B_D15:Insulins:ATC_A10A____

			vector<string> s1;
			boost::split(s1, entry.second, boost::is_any_of(":"));
			categories_names.clear();
			categories_sets.clear();
			for (int i = 0; i < s1.size(); i += 2)
			{
				categories_names.push_back(s1[i]);
				vector<string> s2, s3;
				boost::split(s2, s1[i + 1], boost::is_any_of(","));
				for (string &s : s2) // remove empty values (can happen due to comma additions when using list file)
				{
					if (s.size() > 0)
						s3.push_back(s);
				}
				categories_sets.push_back(s3);
			}
		}
		else if (field == "change_at_prescription_mode")
			change_at_prescription_mode = med_stoi(entry.second);
		else if (field == "rp_type")
		{
		}
		else
			MTHROW_AND_ERR("Error in RepCreateBitSignal::init - Unsupported param \"%s\"\n", field.c_str());
		//! [RepCreateBitSignal::init]
	}
	if (in_sig.empty())
		MTHROW_AND_ERR("Error in RepCreateBitSignal::init - in_sig must be passed\n");
	if (out_virtual.empty())
		MTHROW_AND_ERR("Error in RepCreateBitSignal::init - out_virtual must be passed\n");
	if (categories_names.empty())
		MTHROW_AND_ERR("Error in RepCreateBitSignal::init - empty categories is not allowed\n");
	if (min_durations.size() > 0 && min_durations.size() != categories_names.size())
		MTHROW_AND_ERR("Error in RepCreateBitSignal::init - got min_durations of length %d, and %d categories. They must be the same (or use empty min_durations)\n", (int)min_durations.size(), (int)categories_names.size());

	aff_signals.clear();
	aff_signals.insert(out_virtual);
	req_signals.clear();
	req_signals.insert(in_sig);
	virtual_signals.clear();
	virtual_signals_generic.clear();
	if (time_channels == 1)
		virtual_signals_generic.push_back(pair<string, string>(out_virtual, "T(i),V(f)"));
	else if (time_channels == 2)
		virtual_signals_generic.push_back(pair<string, string>(out_virtual, "T(i),T(i),V(f)"));
	else
		MTHROW_AND_ERR("Error in RepCreateBitSignal::init - %d time channels not allowed. maximum of 2 \n", time_channels);

	return 0;
}

//-------------------------------------------------------------------------------------------------------
void RepCreateBitSignal::get_min_jitters(string &jitters_s)
{

	vector<string> jitters_v;
	boost::split(jitters_v, jitters_s, boost::is_any_of(","));

	if (jitters_v.size() == min_jitters.size())
	{
		for (size_t i = 0; i < min_jitters.size(); i++)
			min_jitters[i] = stoi(jitters_v[i]);
	}
	else if (jitters_v.size() == 1)
	{
		for (size_t i = 0; i < min_jitters.size(); i++)
			min_jitters[i] = stoi(jitters_v[0]);
	}
	else
		MTHROW_AND_ERR("min_jitter initialization must have 1 or %d entries\n", (int)min_jitters.size());
}

//-------------------------------------------------------------------------------------------------------
void RepCreateBitSignal::init_tables(MedDictionarySections &dict, MedSignals &sigs)
{

	v_out_sid = sigs.sid(out_virtual);
	if (v_out_sid < 0)
		MTHROW_AND_ERR("Error in RepAggregateSignal::init_tables - virtual output signal %s not found\n", out_virtual.c_str());

	aff_signal_ids.clear();
	aff_signal_ids.insert(v_out_sid); // ??? : is this needed ???

	in_sid = sigs.sid(in_sig);
	if (in_sid < 0)
		MTHROW_AND_ERR("Error in RepAggregateSignal::init_tables - input signal %s not found\n", in_sig.c_str());
	req_signal_ids.clear();
	req_signal_ids.insert(in_sid); // ??? : is this needed ???

	// preparing lut tables
	int section_id = dict.section_id(in_sig);
	vector<string> categories_sets_aggregated;
	for (int i = 0; i < categories_names.size(); i++)
	{
		categories_luts.push_back({});
		dict.dicts[section_id].prep_sets_lookup_table(categories_sets[i], categories_luts.back());
		categories_sets_aggregated.insert(categories_sets_aggregated.end(), categories_sets[i].begin(), categories_sets[i].end());
	}
	dict.dicts[section_id].prep_sets_lookup_table(categories_sets_aggregated, all_cat_lut);
	_dict = &dict;

	// making sure our virtual signal is marked as categorical on channel 0
	sigs.Sid2Info[v_out_sid].is_categorical_per_val_channel[0] = 1;

	// Dictionary for virtual signal (if needed)
	if (dict.section_id(out_virtual) == 0)
	{
		dict.add_section(out_virtual);
		int newSectionId = dict.section_id(out_virtual);

		// The dictionary contains 2^N raw values (N = number of categories)
		// + N categories for sets
		int N = (int)categories_names.size();
		registry_values.clear();
		int n_combinations = (int)(1 << N);
		for (int i = 0; i < n_combinations; i++)
		{
			stringstream stream;
			stream << "BITS_0x" << setfill('0') << setw(sizeof(int) * 2) << hex << i;
			string s(stream.str());
			registry_values.push_back(s);
			dict.dicts[newSectionId].push_new_def(s, i);
			string better_name = "CATEGS";
			if (i == 0)
				better_name += "_NONE";
			else
			{
				for (int j = 0; j < N; j++)
					if (i & (1 << j))
						better_name += "_" + categories_names[j];
			}
			dict.dicts[newSectionId].push_new_def(better_name, i);
		}
		for (int i = 0; i < N; i++)
			registry_values.push_back(categories_names[i]);

		// insert new defs
		for (int i = n_combinations; i < (int)registry_values.size(); i++)
			dict.dicts[newSectionId].push_new_def(registry_values[i], (int)i);

		// insert sets
		for (int i = 0; i < n_combinations; i++)
		{
			for (int j = 0; j < N; j++)
				if (i & (1 << j))
				{
					dict.dicts[newSectionId].push_new_set(n_combinations + j, i);
				}
		}

		if (print_dict != "")
			dict.dicts[newSectionId].write_to_file(print_dict);
	}

	// Maximal min_jitters ;
	max_min_jitters = min_jitters[0];
	for (size_t i = 1; i < min_jitters.size(); i++)
	{
		if (min_jitters[i] > max_min_jitters)
			max_min_jitters = min_jitters[i];
	}
}

void RepCreateBitSignal::get_required_signal_categories(unordered_map<string, vector<string>> &signal_categories_in_use) const
{
	unordered_set<string> uniq_set;
	for (const vector<string> &e : categories_sets)
		uniq_set.insert(e.begin(), e.end());
	vector<string> uniq_ls(uniq_set.begin(), uniq_set.end());
	signal_categories_in_use[in_sig] = std::move(uniq_ls);
}
//-------------------------------------------------------------------------------------------------------
void RepCreateBitSignal::register_virtual_section_name_id(MedDictionarySections &dict)
{
	// dict.SectionName2Id[out_virtual] = dict.section_id(in_sig);
}

// #define _APPLY_VERBOSE
int RepCreateBitSignal::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	if (time_points.size() != 0 && time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch: %d time-pints vs %d versions\n", time_points.size(), rec.get_n_versions());
		return -1;
	}
	if (v_out_sid < 0)
		MTHROW_AND_ERR("Error in RepCreateBitSignal::_apply - v_out_sid is not initialized - bad call\n");

	vector<int> actual_min_durations = min_durations;
	if (actual_min_durations.size() == 0)
		actual_min_durations.resize(categories_names.size(), min_duration);

	// plan:
	// Go over versions:
	// For each version, calculate a list of time intervals in which the category is contained.
	// Then at the end , unite them to the proper states and push as a signal
	//

	int N = (int)categories_names.size();
	allVersionsIterator vit(rec, {in_sid});
	UniversalSigVec usv;
	int maskN = (1 << N) - 1;
	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{
		rec.uget(in_sid, iver, usv);

		// MLOG("working on iver %d , time %d dont_look_back %d\n", iver, time_points[iver], dont_look_back);

		// max_look_at_time : a safety parameter enabling to make sure we are NOT looking at presctiptions from the current (or very near to) days.
		int max_look_at_time = (time_points.size() > 0) ? med_time_converter.add_subtract_time(time_points[iver], time_unit_sig, -dont_look_back, time_unit_duration) : -1;
		// MLOG("max_look_at_time %d\n", max_look_at_time);

		// Collect durations per category
		vector<vector<pair<int, int>>> collected_info(N);
		for (int i = 0; i < usv.len; i++)
		{
			int i_time = (int)usv.Time(i, t_chan);
			if (max_look_at_time == -1 || i_time <= max_look_at_time)
			{
				int i_val = (int)usv.Val(i, c_chan);
				int duration = (int)usv.Val(i, duration_chan);
				for (int j = 0; j < N; j++)
				{
					if (categories_luts[j][i_val])
					{
						if (collected_info[j].empty() || collected_info[j].back().first != i_time)
							collected_info[j].push_back({i_time, duration});
						else if (duration > collected_info[j].back().second)
							collected_info[j].back().second = duration;
					}
				}
			}
		}

		// calculating a time interval for each category
		vector<vector<category_time_interval>> time_intervals(N, {category_time_interval()});
		for (int j = 0; j < N; j++)
		{
			for (size_t i = 0; i < collected_info[j].size(); i++)
			{
				int i_time = collected_info[j][i].first;
				int j_duration = collected_info[j][i].second;

				// if (duration < min_duration) duration = min_duration;
				// if (duration > max_duration) duration = max_duration;
				// int to_time = med_time_converter.add_subtract_time(i_time, time_unit_sig, duration, time_unit_duration);

				// at this point we have i_val and we assume it happens in the time range [i_time, to_time]
				// we used the duration and fixed it to its min/max values.

				if (j_duration < actual_min_durations[j])
					j_duration = actual_min_durations[j];
				j_duration += (int)(duration_add + duration_mult * (float)j_duration);
				if (j_duration > max_duration)
					j_duration = max_duration;
				int to_time = med_time_converter.add_subtract_time(i_time, time_unit_sig, j_duration, time_unit_duration);

				// MLOG("go over drugs : time %d drug %d urationd %d j=%d j_duration %d\n", i_time, i_val, duration, j, j_duration);

				if (time_intervals[j].back().first_appearance == 0)
				{
					// case it is the first interval
					time_intervals[j].back().set(i_time, i_time, 1, to_time);
				}
				else if (i_time > time_intervals[j].back().last_time)
				{
					// starting too long after current interval, hence starting a new one
					time_intervals[j].push_back(category_time_interval(i_time, i_time, 1, to_time));
				}
				else if (to_time > time_intervals[j].back().last_time)
				{
					// means we need to extend the time_interval
					time_intervals[j].back().last_appearance = i_time;
					time_intervals[j].back().n_appearances++;
					time_intervals[j].back().last_time = to_time;
				}
			}
		}

		// doing the last period clips
		for (int j = 0; j < N; j++)
		{

			for (auto &e : time_intervals[j])
			{
				int clip_time = med_time_converter.add_subtract_time(e.last_appearance, time_unit_sig, last_clip_period, time_unit_duration);
				if (e.last_time > clip_time)
					e.last_time = clip_time;
			}
		}

		// now packing these into states
		// first step : get a single chain of events
		// we encode the events with a +1 on the index, 1 for start, and 0 for end
		// general idea: create an event for every start and end of an interval and later sort it and pass over it.
		vector<category_event_state> ev;
		for (int j = 0; j < N; j++)
		{
			for (auto &e : time_intervals[j])
			{
#ifdef _APPLY_VERBOSE
				if (e.first_appearance > 0)
					MLOG("ID=%d\ttime intervals: j=%d first_a %d n %d last_a %d last_t %d\n", rec.pid, j, e.first_appearance, e.n_appearances, e.last_appearance, e.last_time);
#endif
				if (e.first_appearance > 0)
				{

					// start of an interval
					ev.push_back(category_event_state(e.first_appearance, e.last_time, j, 1));

					// end of an interval : we have some uncertainty regarding the EXACT time to code the drug OFF in the range [e.last_appearance, e.last_time]
					ev.push_back(category_event_state(e.last_time, e.last_appearance, j, 0));
				}
			}
		}

		// sorting the pairs, by date, and within each date: first the ends , then the starts
		sort(ev.begin(), ev.end());

		// now clipping last parts of intervals in cases where a change in category happened.
		// We define a change in category in the following situation:
		// (1) It happened at least min_clip_time after the last appearance.
		// (2) It happened before the the last time
		// (3) A different category started at the exact same time.

		for (int i = 0; i < ev.size(); i++)
		{
			if (ev[i].type == 0)
			{
				int min_time = med_time_converter.add_subtract_time(ev[i].appear_time, time_unit_sig, min_clip_time, time_unit_duration);
				if (min_time < ev[i].time)
				{
					for (int j = i - 1; j > 0; j--)
					{
						if (ev[j].type)
						{
							// this is a start event

							if (ev[j].time < min_time)
								break; // no need to keep testing, all times will be lower (since sorted).

							if (ev[j].categ != ev[i].categ)
							{
								ev[i].time = ev[j].time; // new stop time is shortened to the start time of the starting event. earliest one will survive.
#ifdef _APPLY_VERBOSE
								MLOG("ID=%d\tClipping period %d of category %d to %d\n", rec.pid, i, ev[i].categ, ev[i].time);
#endif
							}
						}
					}
				}
			}
		}

#ifdef _APPLY_VERBOSE
		for (int j = 0; j < N; j++)
		{
			for (auto &e : time_intervals[j])
			{
				if (e.first_appearance > 0)
					MLOG("ID=%d\ttime intervals2: j=%d first_a %d n %d last_a %d last_t %d\n", rec.pid, j, e.first_appearance, e.n_appearances, e.last_appearance, e.last_time);
			}
		}
#endif

		// sorting again as we may have touched times
		sort(ev.begin(), ev.end());

		// actually creating the states
		vector<combination_state> states;

		if (usv.len > 0)
		{
			int first_date = usv.Time(0);
			states.push_back(combination_state(first_date, 0, N));

			for (auto &e : ev)
			{
				if (max_look_at_time != -1 && e.time > max_look_at_time)
					break;

				if (states.back().start < e.time)
					states.push_back(combination_state(e.time, states.back().state, N));

				if (e.type == 1)
					states.back().state |= (1 << e.categ);
				else
					states.back().state &= (maskN ^ (1 << e.categ));
			}

			if (change_at_prescription_mode)
			{
				vector<combination_state> updated_states; // date , encoded N bits state
				updated_states.push_back(combination_state(first_date, 0, N));

				vector<combination_state>::iterator curr_state = states.begin();
				combination_state prev;
				for (int i = 0; i < usv.len; i++)
				{
					int i_time = (int)usv.Time(i, t_chan);
					if (max_look_at_time != -1 && i_time > max_look_at_time)
						break;
					int i_val = (int)usv.Val(i, c_chan);
					if (all_cat_lut[i_val] && updated_states.back().start != i_time)
					{
						// MLOG("%d:%s is relevant: %d, duration: %f \n", i_time, (_dict->dicts[_dict->SectionName2Id["Drug"]].Id2Names[i_val][0]).c_str(), all_cat_lut[i_val], (float)usv.Val(i, 1));
						while ((curr_state->start <= i_time) && (curr_state != states.end()))
						{
							prev = *curr_state;
							curr_state++;
						}
						updated_states.push_back(prev);
						updated_states.back().start = i_time;
					}
				}
				states = std::move(updated_states);
			}

			// Roll back last-appearances information
			if (states.size() > 1)
			{
				for (int i = (int)states.size() - 2; i >= 0; i--)
				{
					for (int j = 0; j < N; j++)
					{
						if (states[i].last[j] == -1 && states[i + 1].last[j] != -1)
							states[i].last[j] = states[i + 1].last[j];
					}
				}
			}
		}

#ifdef _APPLY_VERBOSE
		for (size_t j = 0; j < states.size(); j++)
			MLOG("ID=%d\tJittered state %d - %d %d\n", rec.pid, j, states[j].start, states[j].state);
#endif

		// fixing jitters : remove all jitters that are too short and add to next state
		vector<combination_state> unjittered_states;
		int j = 0;
		int last_taken = 0;

		while (j < states.size())
		{

			// search for max jitter at this point
			int max_k = -1;
			bool take_it = true;
			if (j > 0 && j < states.size() - 1)
			{
				// we have at least 3 states, hence we can check for jitters.
				// we need to decide if (j-1) , (j,..,j+k) , (j+k+1) are in a state of jitter and look for the max k for which it happens
				int v1 = states[last_taken].state;
				int v2 = 0;
				for (int k = 0; k < states.size() - j - 2; k++)
				{

					// We can't add another drug at k>0 - otherwise we would be backtracking that drug start day
					if (k > 0 && ((states[j + k + 1].state | v2) != v2))
						break;

					v2 = v2 | states[j + k].state;
					int v3 = states[j + k + 1].state;
					int len = med_time_converter.diff_times(states[j + k + 1].start, states[j].start, time_unit_sig, time_unit_duration);
					if (len >= max_min_jitters)
						break;

					// the A-AB-B case
					if ((len < min_jitters[0]) && ((v1 | v3) == v2))
					{
						max_k = k;
#ifdef _APPLY_VERBOSE
						MLOG("ID=%d\tJitter at j=%d len = %d last_taken=%d v1=%d k=%d v2=%d and v3=%d : A-AB-B\n", rec.pid, j, len, last_taken, v1, k, v2, v3);
#endif
					}

					// the case of ABC-AB-A
					if ((len < min_jitters[1]) && ((v1 | v2) == v1) && ((v2 | v3) == v2) && ((v1 | v3) == v1))
					{
						max_k = k;
#ifdef _APPLY_VERBOSE
						MLOG("ID=%d\tJitter at j=%d len = %d last_taken=%d v1=%d k=%d v2=%d and v3=%d : ABC-AB-A\n", rec.pid, j, len, last_taken, v1, k, v2, v3);
#endif
					}

					// the case of AB-A-AC
					if ((len < min_jitters[2]) && ((v1 | v2) == v1) && ((v2 | v3) == v3))
					{
						take_it = false;
#ifdef _APPLY_VERBOSE
						MLOG("ID=%d\tJitter at j=%d len = %d last_taken=%d v1=%d k=%d v2=%d and v3=%d : AB-A-AC\n", rec.pid, j, len, last_taken, v1, k, v2, v3);
#endif
					}
				}
			}

			if (max_k >= 0)
			{
				// there was a jitter, hence we push nothing, but fix the state after the jitter, and jump to it
#ifdef _APPLY_VERBOSE
				MLOG("ID=%d\tJitter fixing : j = %d max_k = %d\n", rec.pid, j, max_k);
#endif
				states[j + max_k + 1].start = states[j].start;
				j += max_k;
			}
			else if (take_it)
			{
				// all is well there was no jitter, hence we push the current state
				if (unjittered_states.size() == 0 || unjittered_states.back().state != states[j].state)
				{
					unjittered_states.push_back(states[j]);
					last_taken = j;
				}
			}
			j++;
		}

		// packing and pushing new virtual signal
		vector<int> v_times;
		vector<float> v_vals;
		if (states.size() > 0)
		{
			for (auto &e : unjittered_states)
			{
				if (!v_times.empty() && (time_channels == 2))
				{
					int end_time = med_time_converter.add_subtract_time(e.start, time_unit_sig, -1, time_unit_duration);
					v_times.push_back(end_time);
				}
				v_times.push_back(e.start);
				v_vals.push_back((float)e.state);
				// MLOG("Final: state: %d %d\n", e.first, e.second);
			}
			// handle last range
			if (time_channels == 2)
				v_times.push_back(MAX_DATE);

			rec.set_version_universal_data(v_out_sid, iver, &v_times[0], &v_vals[0], (int)v_vals.size());
		}
	}

	return 0;
}

//-------------------------------------------------------------------------------------------------------

void update_collected(vector<float> &collected, vector<int> collected_times[], int start_time, int end_time)
{
	// iterate throght collected and remove indexes with no intersect with start_time->end_time
	vector<float> sel_vals;
	vector<int> sel_times[2];
	for (int i = 0; i < collected.size(); ++i)
	{
		int start = collected_times[0][i];
		int end = collected_times[1][i];
		if (start <= end_time && end > start_time)
		{
			sel_vals.push_back(collected[i]);
			sel_times[0].push_back(start);
			sel_times[1].push_back(end);
		}
	}
	// comit selection:
	collected.swap(sel_vals);
	collected_times[0].swap(sel_times[0]);
	collected_times[1].swap(sel_times[1]);
}

float calc_value(const vector<int> collected_times[], const vector<float> &collected,
				 int start_time, int end_time, float threshold)
{
	float res = 0;

	int window_len = end_time - start_time;
	// asuume sorted by collected_times[0] which is start_time
	int coverage = 0;
	int prev_end = 0;
	for (int i = 0; i < collected.size(); ++i)
	{
		int start = collected_times[0][i];
		int end = collected_times[1][i];
		int curr_len = end - start;
		float v = collected[i];

		int real_end = end_time;
		int real_start = start_time;
		if (end < end_time)
			real_end = end;
		if (start > start_time)
			real_start = start;
		int peroid_len = real_end - real_start;
		float ratio_weight = 1;
		if (curr_len > 0)
			ratio_weight = peroid_len / float(curr_len);

		res += ratio_weight * v;
		if (i == 0)
			coverage += peroid_len;
		else
		{
			if (real_start >= prev_end)
				coverage += peroid_len;
			else if (real_end >= prev_end)
				coverage += real_end - prev_end;
		}
		prev_end = real_end;
	}
	float missing_rate = 0;
	if (window_len > 0)
		missing_rate = 1 - coverage / float(window_len);
	if (missing_rate > threshold)
		return MED_MAT_MISSING_VALUE;
	return res;
}

int RepAggregateSignal::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	if (time_points.size() != rec.get_n_versions())
	{
		MERR("nversions mismatch\n");
		return -1;
	}
	if (v_out_sid < 0)
		MTHROW_AND_ERR("Error in RepAggregateSignal::_apply - v_out_sid is not initialized - bad call\n");
	// first lets fetch "static" signals without Time field:

	set<int> set_ids;
	set_ids.insert(in_sid);
	allVersionsIterator vit(rec, set_ids);
	rec.usvs.resize(1);

	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{
		rec.uget(in_sid, iver, rec.usvs[0]);

		vector<float> v_vals;
		vector<int> v_times;
		vector<float> collected;
		vector<int> collected_times[2];
		int first_time = 0;
		for (int i = 0; i < rec.usvs[0].len; ++i)
		{
			int end_time = rec.usvs[0].Time(i, end_time_channel);
			int start_time = rec.usvs[0].Time(i, start_time_channel);
			if (start_time <= 0 || end_time <= 0)
				continue;
			if (first_time == 0)
				first_time = med_time_converter.convert_times(global_default_time_unit, time_unit, rec.usvs[0].Time(i, start_time_channel));

			int end_win_time = med_time_converter.convert_times(global_default_time_unit, time_unit, end_time);
			int start_win_time = end_win_time - time_window;

			collected.push_back(rec.usvs[0].Val(i, work_channel));
			collected_times[0].push_back(med_time_converter.convert_times(global_default_time_unit, time_unit, start_time));
			collected_times[1].push_back(end_win_time);

			update_collected(collected, collected_times, start_win_time, end_win_time);
			if (buffer_first && collected_times[0].back() - first_time < time_window)
				continue; // do not add - wait till buffer filled
			// get value:
			float val = calc_value(collected_times, collected, start_win_time, end_win_time, drop_missing_rate);
			if (val != MED_MAT_MISSING_VALUE)
			{
				val *= factor;
				// if (v_out_n_time_ch > 0)
				//	v_times.push_back(end_time);
				for (int k = 0; k < v_out_n_time_ch; ++k) // copy rest
					v_times.push_back(rec.usvs[0].Time(i, k));
				for (int k = 0; k < v_out_n_val_ch; ++k)
					if (k != work_channel)
						v_vals.push_back(rec.usvs[0].Val(i, k));
					else
						v_vals.push_back(val);
			}
		}
		// pushing virtual data into rec (into orig version)
		if (rec.usvs[0].len > 0)
			rec.set_version_universal_data(v_out_sid, iver, &v_times[0], &v_vals[0], (int)v_vals.size());
	}

	return 0;
}

void RepAggregateSignal::print()
{
	MLOG("RepAggregateSignal:: signal:%s, output_name:%s, work_channel=%d, factor=%2.4f, time_window=%d, time_unit=%s"
		 ", start_time_channel=%d, end_time_channel=%d, drop_missing_rate=%2.4f, buffer_first=%d\n",
		 signalName.c_str(), output_name.c_str(),
		 work_channel, factor, time_window, med_time_converter.type_to_string(time_unit).c_str(),
		 start_time_channel, end_time_channel, drop_missing_rate, buffer_first);
}

//----------------------------------------------------------------------------------------
// RepHistoryLimit : given a signal : chomps history to be at a given window relative
//                   to prediction points
//----------------------------------------------------------------------------------------

// Fill req- and aff-signals vectors
//.......................................................................................
void RepHistoryLimit::init_lists()
{

	req_signals.insert(signalName);
	aff_signals.insert(signalName);
}

// Init from map
//.......................................................................................
int RepHistoryLimit::init(map<string, string> &mapper)
{
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepHistoryLimit::init]
		if (field == "signal")
		{
			signalName = entry.second;
		}
		else if (field == "time_channel")
			time_channel = med_stoi(entry.second);
		else if (field == "truncate_time_channel")
			truncate_time_channel = med_stoi(entry.second);
		else if (field == "win_from")
			win_from = med_stoi(entry.second);
		else if (field == "win_to")
			win_to = med_stoi(entry.second);
		else if (field == "delete_sig")
			delete_sig = med_stoi(entry.second);
		else if (field == "rep_time_unit")
			rep_time_unit = med_time_converter.string_to_type(entry.second);
		else if (field == "win_time_unit")
			win_time_unit = med_time_converter.string_to_type(entry.second);
		else if (field == "take_last_events")
			take_last_events = med_stoi(entry.second);
		else if (field == "unconditional")
			unconditional = med_stoi(entry.second) > 0;
		else if (field != "rp_type")
			MWARN("WARN :: RepHistoryLimit::init - unknown parameter %s - ignored\n", field.c_str());
		//! [RepHistoryLimit::init]
	}

	init_lists();

	if (take_last_events > 0 && (win_to!=0 || win_from!=0))
		MTHROW_AND_ERR("Error in history limit. Can't use take_last_events and win_from, win_from\n");

	return 0;
}

int RepHistoryLimit::get_sub_usv_data(UniversalSigVec &usv, int from_time, int to_time, vector<char> &data, int &len)
{
	data.clear();
	len = 0;
	if (truncate_time_channel < 0 || truncate_time_channel >= usv.n_time_channels())
	{
		char *udata = (char *)usv.data;
		int element_size = (int)usv.size();
		for (int i = 0; i < usv.len; i++)
		{
			int i_time = usv.Time(i, time_channel);
			if (i_time > from_time && i_time <= to_time)
			{
				for (int j = element_size * i; j < element_size * (i + 1); j++)
					data.push_back(udata[j]);
				len++;
			}
		}
	}
	else
	{
		char *udata = (char *)usv.data;
		int element_size = (int)usv.size();
		for (int i = 0; i < usv.len; i++)
		{
			int i_time = usv.Time(i, time_channel);
			int tr_time = usv.Time(i, truncate_time_channel);
			if (i_time > from_time && i_time <= to_time)
			{
				if (tr_time > to_time)
					usv.setTime(i, truncate_time_channel, to_time);
				for (int j = element_size * i; j < element_size * (i + 1); j++)
					data.push_back(udata[j]);
				++len;
			}
		}
	}
	return 0;
}

//---------------------------------------------------------------------------------------------------------------
int RepHistoryLimit::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{

	// goal : for each time_points[i] generate the signal i version to contain only data within the given time window

	int len = 0;
	UniversalSigVec usv;
	vector<char> data;

	if (delete_sig == 0)
	{
		if (win_from == 0 && win_to == 0 && take_last_events > 0)
		{
			for (int ver = 0; ver < time_points.size(); ver++)
			{
				int current_time = time_points[ver];
				rec.uget(signalId, ver, usv);
				int start_ind = usv.len - 1;
				char *udata = (char *)usv.data;
				int element_size = (int)usv.size();
				// Iterate backward
				len = 0;
				int start_cp = -1;
				int end_cp = -1;
				for (int i = start_ind; i >= 0; --i)
				{
					// Test that we respect current_time - not after it
					if (usv.Time(i, time_channel) > current_time)
						continue;
					if (end_cp < 0)
						end_cp = i;
					start_cp = i;

					++len;
					if (len >= take_last_events)
						break;
				}
				//Do the copy in right order from start_cp to end_cp:
				for (int i = start_cp; i <= end_cp; ++i)
				{
					// Copy all element data
					for (int j = element_size * i; j < element_size * (i + 1); ++j)
						data.push_back(udata[j]);
				}

				rec.set_version_data(signalId, ver, &data[0], len);
			}
		}
		else
		{
			for (int ver = 0; ver < time_points.size(); ver++)
			{
				rec.uget(signalId, ver, usv);
				int curr_time = med_time_converter.convert_times(rep_time_unit, win_time_unit, time_points[ver]);
				int from_time = med_time_converter.convert_times(win_time_unit, rep_time_unit, curr_time - win_to);
				int to_time = med_time_converter.convert_times(win_time_unit, rep_time_unit, curr_time - win_from);
				get_sub_usv_data(usv, from_time, to_time, data, len);
				if (len < usv.len)
				{
					rec.set_version_data(signalId, ver, &data[0], len);
				}
			}
		}
	}
	else
	{
		// simply delete signal and point all versions to the deleted signal
		rec.uget(signalId, 0, usv);
		rec.set_version_data(signalId, 0, &data[0], 0);
		for (int ver = 1; ver < time_points.size(); ver++)
			rec.point_version_to(signalId, 0, ver);
	}

	return 0;
}

//----------------------------------------------------------------------------------------
// RepNumericNoiser : given a numeric signal : adds gaussian noise to each value, with std as user-determined
// fraction of signal std
//----------------------------------------------------------------------------------------

// Fill req- and aff-signals vectors
//.......................................................................................
void RepNumericNoiser::init_lists()
{

	req_signals.insert(signalName);
	aff_signals.insert(signalName);
}

// Init from map
//.......................................................................................
int RepNumericNoiser::init(map<string, string> &mapper)
{
	for (auto entry : mapper)
	{
		string field = entry.first;
		//! [RepHistoryLimit::init]
		if (field == "signal")
		{
			signalName = entry.second;
		}
		else if (field == "time_channel")
			time_channel = med_stoi(entry.second);
		else if (field == "val_channel")
			val_channel = med_stoi(entry.second);
		else if (field == "time_noise")
			time_noise = med_stoi(entry.second);
		else if (field == "value_noise")
			value_noise = med_stof(entry.second);
		else if (field == "truncation")
			truncation = med_stoi(entry.second);
		else if (field == "drop_probability")
			drop_probability = med_stof(entry.second);
		else if (field == "apply_in_test")
			apply_in_test = med_stof(entry.second);

		else if (field != "rp_type")
			MWARN("WARN :: RepNumericNoiser::init - unknown parameter %s - ignored\n", field.c_str());
		//! [RepHistoryLimit::init]
	}

	init_lists();

	return 0;
}

//---------------------------------------------------------------------------------------------------------------

int RepNumericNoiser::_learn(MedPidRepository &rep, MedSamples &samples, vector<RepProcessor *> &prev_cleaners)
{

	// Sanity check
	if (signalId == -1)
	{
		MERR("RepNumericNoiser::_learn - Uninitialized signalId(%s)\n", signalName.c_str());
		return -1;
	}

	// Get all values
	vector<float> v;
	get_values(rep, samples, signalId, time_channel, val_channel, -FLT_MAX, FLT_MAX, v, prev_cleaners);

	if (v.empty())
	{
		MTHROW_AND_ERR("RepNumericNoiser::_learn WARNING signal [%d] = [%s] is empty, will not calculate std\n", signalId,
					   this->signalName.c_str());
	}
	else
	{
		double sum = std::accumulate(v.begin(), v.end(), 0.0);
		double mean = sum / v.size();

		std::vector<double> diff(v.size());
		std::transform(v.begin(), v.end(), diff.begin(), [mean](double x)
					   { return x - mean; });
		double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
		stdev = std::sqrt(sq_sum / v.size());
	}

	on_learning = true;
	return 0;
}
//---------------------------------------------------------------------------------------------------------------

int RepNumericNoiser::_apply(PidDynamicRec &rec, vector<int> &time_points, vector<vector<float>> &attributes_mat)
{
	if (!on_learning && !apply_in_test)
		return 0;

	UniversalSigVec usv;

	set<int> iteratorSignalIds;
	iteratorSignalIds.insert(signalId);
	allVersionsIterator vit(rec, iteratorSignalIds);

	int n_th = omp_get_thread_num();

	int iii = 0;

	for (int iver = vit.init(); !vit.done(); iver = vit.next())
	{
		rec.uget(signalId, iver, usv);
		int final_size = usv.len;
		vector<int> times;
		vector<float> vals;

		vector<pair<int, float>> vect;

		int current_date = time_points[iii];

		for (int i = 0; i < usv.len; i++)
		{

			int i_time = usv.Time(i, time_channel);

			if (i_time > current_date)
			{

				final_size -= 1;
				/*
				MLOG("curr date: %d, filtered date %d\n", current_date, i_time); */
				continue;
			}
			/*			else {
				MLOG("curr date: %d, nonfiltered date %d\n", current_date, i_time);
			}*/

			float val = usv.Val(i, val_channel);

			uniform_real_distribution<> dist_prob(0, 1);
			float random_sample = dist_prob(gens[3 * n_th]);
			if (random_sample < drop_probability)
			{
				final_size -= 1;
				continue;
			}

			uniform_int_distribution<> distrib_uni(0, time_noise);
			int years = i_time / 10000;
			int months = (i_time % 10000) / 100;
			int days = i_time % 100;

			int shift = distrib_uni(gens[3 * n_th + 1]);

			int years_to_remove = shift / 365;
			int months_to_remove = (shift % 365) / 30;
			int days_to_remove = (shift % 365) % 30;

			days = days - days_to_remove;
			if (days < 1)
			{
				months = months - 1;
				days = days + 30;
			}

			months = months - months_to_remove;
			if (months < 1)
			{
				months = months + 12;
				years = years - 1;
			}

			years = years - years_to_remove;

			int new_time = years * 10000 + months * 100 + days;

			// Add noise to vals

			normal_distribution<float> distrib_norm(0.0, stdev * value_noise);
			float new_val = val + distrib_norm(gens[3 * n_th + 2]);

			new_val = round(new_val * pow(10, truncation)) / pow(10, truncation);
			vect.push_back(make_pair(new_time, new_val));
		}

		sort(vect.begin(), vect.end());

		for (int i = 0; i < final_size; i++)
		{
			times.push_back(vect[i].first);
			vals.push_back(vect[i].second);
		}

		/*
				if (final_size > 7) {
			MLOG("non-filtered dates: \n");

			for (int i = 0; i< final_size; i++)
			{
				MLOG("%d %d %d\n", current_date, i, times[i]);

			}
		}
		*/

		rec.set_version_universal_data(signalId, iver, &times[0], &vals[0], final_size);
		++iii;
	}

	return 0;
}

//=======================================================================================
// Utility Functions
//=======================================================================================
//.......................................................................................
// Get values of a signal from a set of ids
int get_values(MedRepository &rep, MedSamples &samples, int signalId, int time_channel, int val_channel, float range_min, float range_max, vector<float> &values, vector<RepProcessor *> &prev_processors)
{

	// Required signals
	vector<int> req_signal_ids_v;
	vector<unordered_set<int>> current_required_signal_ids(prev_processors.size());
	vector<FeatureGenerator *> noGenerators;
	unordered_set<int> extra_req_signal_ids = {signalId};
	handle_required_signals(prev_processors, noGenerators, extra_req_signal_ids, req_signal_ids_v, current_required_signal_ids);

	PidDynamicRec rec;
	UniversalSigVec usv;

	bool signalIsVirtual = (bool)(rep.sigs.Sid2Info[signalId].virtual_sig != 0);

	for (MedIdSamples &idSamples : samples.idSamples)
	{

		int id = idSamples.id;

		vector<int> time_points;
		// Special care for virtual signals - use samples
		if (signalIsVirtual)
		{

			time_points.resize(idSamples.samples.size());
			for (size_t i = 0; i < time_points.size(); i++)
			{
				time_points[i] = idSamples.samples[i].time;
			}
		}
		else
		{
			// Get signal
			rep.uget(id, signalId, usv);

			time_points.resize(usv.len);
			for (int i = 0; i < usv.len; i++)
				time_points[i] = usv.Time(i, time_channel);

			// Nothing to do if empty ...
			if (time_points.empty())
				continue;

			if (prev_processors.size())
			{

				// Init Dynamic Rec
				rec.init_from_rep(std::addressof(rep), id, req_signal_ids_v, (int)time_points.size());

				// Process at all time-points
				vector<vector<float>> dummy_attributes_mat;
				for (size_t i = 0; i < prev_processors.size(); i++)
					prev_processors[i]->conditional_apply(rec, time_points, current_required_signal_ids[i], dummy_attributes_mat);

				// If virtual - we need to get the signal now
				if (signalIsVirtual)
					rec.uget(signalId, 0, usv);

				// Collect
				int iVersion = 0;
				rec.uget(signalId, iVersion, rec.usv);

				for (int i = 0; i < usv.len; i++)
				{
					// Get a new version if we past the current one
					if (usv.Time(i) > time_points[iVersion])
					{
						iVersion++;
						if (iVersion == rec.get_n_versions())
							break;
						rec.uget(signalId, iVersion, rec.usv);
					}

					float ival = rec.usv.Val(i, val_channel);
					if (ival >= range_min && ival <= range_max)
						values.push_back(ival);
				}
			}
			else
			{
				// Collect
				for (int i = 0; i < usv.len; i++)
				{
					float ival = usv.Val(i, val_channel);
					if (ival >= range_min && ival <= range_max)
						values.push_back(ival);
				}
			}
		}
	}
	return 0;
}
//.......................................................................................
int get_values(
	MedRepository &rep,
	MedSamples &samples,
	int signalId,
	int time_channel,
	int val_channel,
	float range_min,
	float range_max,
	vector<float> &values)
{
	vector<RepProcessor *> temp;
	return get_values(rep, samples, signalId, time_channel, val_channel, range_min, range_max, values, temp);
}
