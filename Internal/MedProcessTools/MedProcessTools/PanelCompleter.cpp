#define _CRT_SECURE_NO_WARNINGS

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_REPCLEANER
#define LOCAL_LEVEL	LOG_DEF_LEVEL

#include "RepProcess.h"
#include <MedUtils/MedUtils/MedUtils.h>
#include <MedUtils/MedUtils/MedMedical.h>

//=======================================================================================
// RepPanelCompleter fills-in calculatable signal values. Enriching existing signals
//=======================================================================================
//.......................................................................................
// Init from map
int RepPanelCompleter::init(map<string, string>& mapper)
{

	for (auto entry : mapper) {
		string field = entry.first;
		//! [RepPanelCompleter::init]
		if (panel2type.find(field) != panel2type.end()) {
			if (update_signal_names(field, entry.second) != 0) return -1;
		}
		else if (field == "missing") missing_val = stof(entry.second);
		else if (field == "config" || field == "metadata") metadata_file = entry.second;
		else if (field == "unconditional") unconditional = stoi(entry.second) > 0;
		else if (field == "sim_val" || field == "sim_val_handler") sim_val_handler = RepSimValHandler::get_sim_val_handle_type(entry.second);
		else if (field == "panels") {
			if (update_panels(entry.second) != 0) return -1;
		}
		//! [RepPanelCompleter::init]
	}

	init_lists();
	return 0;
}

//.......................................................................................
// Update signal names
int RepPanelCompleter::update_signal_names(string panel, string& names) {

	vector<string> signals;
	boost::split(signals, names, boost::is_any_of(","));

	if (signals.size() != 0 && signals.size() != panel2signals[panel].size()) {
		MERR("Wrong number of signals given for panel %s", panel.c_str());
		return -1;
	}
	else {
		panel_signal_names[panel2type[panel]] = signals;
		return 0;
	}
}

//.......................................................................................
// Update panels to work on.
int RepPanelCompleter::update_panels(string& panels) {

	vector<string> list;
	boost::split(list, panels, boost::is_any_of(","));

	unordered_set<string> selected;
	for (string& panel : list) {
		if (panel2signals.find(panel) == panel2signals.end()) {
			string panel_opts = "";
			for (auto it = panel2signals.begin(); it != panel2signals.end(); ++it)
				if (it == panel2signals.begin())
					panel_opts += it->first;
				else
					panel_opts += "," + it->first;
			MTHROW_AND_ERR("Required unknown panel %s for completion. Options:%s\n", panel.c_str(), panel_opts.c_str());
			return -1;
		}
		else
			selected.insert(panel);
	}

	for (auto& panel : panel2signals) {
		if (selected.find(panel.first) == selected.end())
			panel_signal_names[panel2type[panel.first]].clear();
	}
	return 0;
}

//.......................................................................................
// Set signal names to default
void RepPanelCompleter::init_defaults() {

	panel_signal_names.resize(panel2type.size());
	for (auto& panel : panel2type)
		panel_signal_names[panel.second] = panel2signals[panel.first];

	genderSignalName = "GENDER";
}

//.......................................................................................
// initialize signal ids
void RepPanelCompleter::set_signal_ids(MedSignals& sigs) {

	panel_signal_ids.resize(panel_signal_names.size());
	for (int iPanel = 0; iPanel < panel_signal_ids.size(); iPanel++) {
		panel_signal_ids[iPanel].resize(panel_signal_names[iPanel].size());
		for (int iSig = 0; iSig < panel_signal_ids[iPanel].size(); iSig++) {
			panel_signal_ids[iPanel][iSig] = sigs.sid(panel_signal_names[iPanel][iSig]);
			if (panel_signal_ids[iPanel][iSig] == -1)
				MTHROW_AND_ERR("Cannot find signal-id for %s\n", panel_signal_names[iPanel][iSig].c_str());
		}
	}

	// EGFR Requires age and gender
	if (panel_signal_names[REP_CMPLT_EGFR_PANEL].size()) {
		genderId = sigs.sid(genderSignalName);
		bdateId = sigs.sid("BDATE");

	}


}

bool has_valid_panel(const vector<vector<int>> &mini_panel,
	const vector<string> &input_signals, MedPidRepository& rep,
	unordered_set<string> &missing_sigs) {
	bool has_valid_minipanel = false;
	for (size_t i = 0; i < mini_panel.size(); ++i)
	{
		vector<string> mini_panel_missing;
		const vector<int> &input_sig_idx = mini_panel[i];
		for (size_t j = 0; j < input_sig_idx.size(); ++j)
			if (rep.sigs.Name2Sid.find(input_signals[input_sig_idx[j]]) == rep.sigs.Name2Sid.end())
				mini_panel_missing.push_back(input_signals[input_sig_idx[j]]);

		if (mini_panel_missing.size() == 1) {
			missing_sigs.insert(mini_panel_missing[0]); //at least we have partial match
			has_valid_minipanel = true;
		}
		else {
			if (mini_panel_missing.empty())
				has_valid_minipanel = true;
		}
	}

	return has_valid_minipanel;
}

// Check if some required signals are missing and make them virtual or remove relevant panel completer
void RepPanelCompleter::fit_for_repository(MedPidRepository& rep) {

	unordered_set<string> missing_sigs;
	for (int iPanel = 0; iPanel < panel_signal_names.size(); iPanel++) {
		vector<string> panel_missing;
		const vector<string> &input_signals = panel_signal_names[iPanel];
		for (int iSig = 0; iSig < input_signals.size(); iSig++) {
			const string &sig_name = input_signals[iSig];
			if (rep.sigs.Name2Sid.find(sig_name) == rep.sigs.Name2Sid.end())
				panel_missing.push_back(sig_name);
		}

		// If one signal is missing, we can declare it virtual and use the panel completer
		// Otherwise, we have to remove the completer
		if (panel_missing.size() == 1)
			missing_sigs.insert(panel_missing[0]);
		else {
			bool has_valid_minipanel = false;
			if (!input_signals.empty()) {
				if (iPanel == REP_CMPLT_RED_LINE_PANEL) {
					//break into mini-panels:
					vector<vector<int>> mini_panel(3);
					mini_panel[0] = { RED_PNL_MCV , RED_PNL_HCT , RED_PNL_RBC };
					mini_panel[1] = { RED_PNL_MCH , RED_PNL_HGB , RED_PNL_RBC };
					mini_panel[2] = { RED_PNL_MCHC , RED_PNL_HGB , RED_PNL_HCT };

					has_valid_minipanel = has_valid_panel(mini_panel, input_signals, rep, missing_sigs);
				}
				else if (iPanel == REP_CMPLT_WHITE_LINE_PANEL) {
					vector<vector<int>> mini_panel(white_panel_nums.size() + 1);
					mini_panel[0] = white_panel_nums;
					mini_panel[0].push_back(WHITE_PNL_WBC);
					for (int j = 0; j < white_panel_nums.size(); j++) {
						int num_idx = white_panel_nums[j];
						int perc_idx = white_panel_precs[j];
						mini_panel[j + 1] = { num_idx , perc_idx , WHITE_PNL_WBC };
					}

					has_valid_minipanel = has_valid_panel(mini_panel, input_signals, rep, missing_sigs);
				}
				else if (iPanel == REP_CMPLT_LIPIDS_PANEL) {
					vector<vector<int>> mini_panel(10);
					mini_panel[0] = { LIPIDS_PNL_HDL_OVER_CHOL, LIPIDS_PNL_HDL, LIPIDS_PNL_CHOL };
					mini_panel[1] = { LIPIDS_PNL_CHOL_OVER_HDL, LIPIDS_PNL_CHOL, LIPIDS_PNL_HDL };
					mini_panel[2] = { LIPIDS_PNL_HDL_OVER_LDL, LIPIDS_PNL_HDL, LIPIDS_PNL_LDL };
					mini_panel[3] = { LIPIDS_PNL_LDL_OVER_HDL, LIPIDS_PNL_LDL, LIPIDS_PNL_HDL };
					mini_panel[4] = { LIPIDS_PNL_HDL_OVER_NON_HDL, LIPIDS_PNL_HDL, LIPIDS_PNL_NON_HDL_CHOL };
					mini_panel[5] = { LIPIDS_PNL_HDL_OVER_LDL, LIPIDS_PNL_LDL_OVER_HDL };
					mini_panel[6] = { LIPIDS_PNL_HDL_OVER_CHOL, LIPIDS_PNL_CHOL_OVER_HDL };
					mini_panel[7] = chol_types1;
					mini_panel[7].push_back(LIPIDS_PNL_CHOL);
					mini_panel[8] = chol_types2;
					mini_panel[8].push_back(LIPIDS_PNL_CHOL);
					mini_panel[9] = { LIPIDS_PNL_VLDL, LIPIDS_PNL_TRGS };

					has_valid_minipanel = has_valid_panel(mini_panel, input_signals, rep, missing_sigs);
				}
			}

			if (has_valid_minipanel) {
				//TODO: break into mini-panels and only add needed signals.
				// Currently, if will not add all signals in the panel, it will not work.
				// the code expects all signals in the panel to be defined (phisical, or virtually)
				missing_sigs.insert(panel_missing.begin(), panel_missing.end());
			}

			if (!panel_missing.empty() && !has_valid_minipanel) {
				//remove panel
				panel_signal_names[iPanel].clear();
				string panel_name = "";
				for (const auto &it : panel2type)
					if (it.second == iPanel)
						panel_name = it.first;
				MLOG("RepPanelCompleter::fit_for_repository - removed panel %s\n", panel_name.c_str());
			}
		}
	}

	// Missing sigs are virtual signals
	virtual_signals.clear();
	virtual_signals_generic.clear();
	for (const string &sig : missing_sigs) {
		virtual_signals_generic.push_back(pair<string, string>(sig, "T(i),V(f)"));
		MWARN("Warning: RepPanelCompleter:: add virtual signal %s\n", sig.c_str());
	}

	//analyze required and affected signals again:
	init_lists();
}
//.......................................................................................
void RepPanelCompleter::init_tables(MedDictionarySections &dict, MedSignals& sigs)
{
	// TBD: Should be improved, this is way too specific 

	if (panel_signal_names[REP_CMPLT_GCS].size()) {
		int section_id = dict.section_id(panel_signal_names[REP_CMPLT_GCS].back());
		if (section_id < 0)
			MTHROW_AND_ERR("unable to find GCS Section. search for %s signal\n",
				panel_signal_names[REP_CMPLT_GCS].back().c_str());
		eye_vals = {
		{ dict.id(section_id, "GCS_Eye:none") , 1 },
		{ dict.id(section_id, "GCS_Eye:to_pain"), 2 },
		{ dict.id(section_id, "GCS_Eye:to_speech"), 3 },
		{ dict.id(section_id, "GCS_Eye:spontaneously"), 4 }
		};
		verbal_vals = {
		{ dict.id(section_id, "GCS_Verbal:no_response"), 1 },
		{ dict.id(section_id, "GCS_Verbal:no_response-ett"), 1 },
		{ dict.id(section_id, "GCS_Verbal:incomprehensible_sounds"), 2 },
		{ dict.id(section_id, "GCS_Verbal:inappropriate_words"), 3 },
		{ dict.id(section_id, "GCS_Verbal:confused"), 4 },
		{ dict.id(section_id, "GCS_Verbal:oriented"), 5 }
		};
		motor_vals = {
		{ dict.id(section_id, "GCS_Motor:no_response"), 1 },
		{ dict.id(section_id, "GCS_Motor:abnormal_extension"), 2 },
		{ dict.id(section_id, "GCS_Motor:abnormal_flexion"), 3 },
		{ dict.id(section_id, "GCS_Motor:flex-withdraws"), 4 },
		{ dict.id(section_id, "GCS_Motor:localizes_pain"), 5 },
		{ dict.id(section_id, "GCS_Motor:obeys_commands"), 6 }
		};

	}

}

//.......................................................................................
// Fill required and affected signals
void RepPanelCompleter::init_lists() {

	req_signals.clear();
	aff_signals.clear();

	for (int iPanel = 0; iPanel < panel_signal_names.size(); iPanel++) {
		for (int iSig = 0; iSig < panel_signal_names[iPanel].size(); iSig++) {
			req_signals.insert(panel_signal_names[iPanel][iSig]);
			aff_signals.insert(panel_signal_names[iPanel][iSig]);
		}
	}

	if (panel_signal_names[REP_CMPLT_EGFR_PANEL].size()) {
		req_signals.insert(genderSignalName);
		req_signals.insert("BDATE");

	}
}

// Apply completions (no relevant attributes)
//.......................................................................................
int RepPanelCompleter::_apply(PidDynamicRec& rec, vector<int>& time_points, vector<vector<float>>& attributes_mat) {

	// Check that we have the correct number of dynamic-versions : one per time-point (if given)
	if (time_points.size() != 0 && time_points.size() != rec.get_n_versions()) {
		MERR("nversions mismatch\n");
		return -1;
	}

	int rc = 0;
	if (panel_signal_ids[REP_CMPLT_RED_LINE_PANEL].size())
		rc = apply_red_line_completer(rec, time_points);
	if (rc < 0)
		return -1;

	if (panel_signal_ids[REP_CMPLT_WHITE_LINE_PANEL].size())
		rc = apply_white_line_completer(rec, time_points);
	if (rc < 0)
		return -1;

	if (panel_signal_ids[REP_CMPLT_PLATELETS_PANEL].size())
		rc = apply_platelets_completer(rec, time_points);
	if (rc < 0)
		return -1;

	if (panel_signal_ids[REP_CMPLT_LIPIDS_PANEL].size())
		rc = apply_lipids_completer(rec, time_points);
	if (rc < 0)
		return -1;

	if (panel_signal_ids[REP_CMPLT_EGFR_PANEL].size())
		rc = apply_eGFR_completer(rec, time_points);
	if (rc < 0)
		return -1;

	if (panel_signal_ids[REP_CMPLT_BMI_PANEL].size())
		rc = apply_BMI_completer(rec, time_points);
	if (rc < 0)
		return -1;

	if (panel_signal_ids.size() > REP_CMPLT_GCS && panel_signal_ids[REP_CMPLT_GCS].size())
		rc = apply_GCS_completer(rec, time_points);
	if (rc < 0)
		return -1;

	return 0;
}

// Completion of red blood line panels
//.......................................................................................
int RepPanelCompleter::apply_red_line_completer(PidDynamicRec& rec, vector<int>& time_points) {

	vector<int>& sigs_ids = panel_signal_ids[REP_CMPLT_RED_LINE_PANEL];
	int n_sigs = (int)sigs_ids.size();

	vector<float>& orig_res = original_sig_res[REP_CMPLT_RED_LINE_PANEL];
	vector<float>& final_res = final_sig_res[REP_CMPLT_RED_LINE_PANEL];
	vector<float>& conv = sig_conversion_factors[REP_CMPLT_RED_LINE_PANEL];
	if (orig_res.empty() || final_res.empty() || conv.empty())
		MTHROW_AND_ERR("Error in RepPanelCompleter::apply_red_line_completer - please provide config/metadata file\n");

	// Loop on versions
	set<int> iteratorSignalIds(sigs_ids.begin(), sigs_ids.end());

	allVersionsIterator vit(rec, iteratorSignalIds);
	for (int iver = vit.init(); !vit.done(); iver = vit.next()) {

		int time_limit = (time_points.size()) ? time_points[iver] : -1;

		// Get Signals
		rec.usvs.resize(n_sigs);
		for (size_t i = 0; i < n_sigs; ++i)
			rec.uget(sigs_ids[i], iver, rec.usvs[i]);

		// Get Panels
		vector<int> panel_times;
		vector<vector<float> > panels;
		get_panels(rec.usvs, panel_times, panels, time_limit, RED_PNL_LAST);

		// Complete
		vector<int> changed_signals(n_sigs, 0);
		for (size_t i = 0; i < panels.size(); i++) {
			int complete = 1;
			while (complete) {
				complete = 0;

				// MCV = 10*HCT/RBC
				complete += triplet_complete(panels[i], 10, RED_PNL_MCV, RED_PNL_HCT, RED_PNL_RBC, orig_res, final_res, conv, changed_signals);

				// MCH = 10 * HGB / RBC
				complete += triplet_complete(panels[i], 10, RED_PNL_MCH, RED_PNL_HGB, RED_PNL_RBC, orig_res, final_res, conv, changed_signals);

				// MCHC = 100 * HGB / HCT
				complete += triplet_complete(panels[i], 100, RED_PNL_MCHC, RED_PNL_HGB, RED_PNL_HCT, orig_res, final_res, conv, changed_signals);
			}
		}

		// Update changed signals
		if (update_signals(rec, iver, panels, panel_times, sigs_ids, changed_signals) < 0)
			return -1;
	}

	return 0;
}

// Completion of white blood line panels
//.......................................................................................
int RepPanelCompleter::apply_white_line_completer(PidDynamicRec& rec, vector<int>& time_points) {

	vector<int>& sigs_ids = panel_signal_ids[REP_CMPLT_WHITE_LINE_PANEL];
	int n_sigs = (int)sigs_ids.size();

	vector<float>& orig_res = original_sig_res[REP_CMPLT_WHITE_LINE_PANEL];
	vector<float>& final_res = final_sig_res[REP_CMPLT_WHITE_LINE_PANEL];
	vector<float>& conv = sig_conversion_factors[REP_CMPLT_WHITE_LINE_PANEL];
	if (orig_res.empty() || final_res.empty() || conv.empty())
		MTHROW_AND_ERR("Error in RepPanelCompleter::apply_white_line_completer - please provice config/metadata file\n");

	// Loop on versions
	set<int> iteratorSignalIds(sigs_ids.begin(), sigs_ids.end());

	allVersionsIterator vit(rec, iteratorSignalIds);
	for (int iver = vit.init(); !vit.done(); iver = vit.next()) {

		int time_limit = (time_points.size()) ? time_points[iver] : -1;

		// Get Signals
		rec.usvs.resize(n_sigs);
		for (size_t i = 0; i < n_sigs; ++i)
			rec.uget(sigs_ids[i], iver, rec.usvs[i]);

		// Get Panels
		vector<int> panel_times;
		vector<vector<float> > panels;
		get_panels(rec.usvs, panel_times, panels, time_limit, WHITE_PNL_LAST);

		// Complete
		vector<int> changed_signals(n_sigs, 0);
		for (size_t i = 0; i < panels.size(); i++) {
			int complete = 1;
			while (complete) {
				complete = 0;

				// WBC = SUM(#s)
				complete += sum_complete(panels[i], WHITE_PNL_WBC, white_panel_nums, orig_res, final_res, conv, changed_signals);

				// White subtypes - 
				for (int j = 0; j < white_panel_nums.size(); j++) {
					int num_idx = white_panel_nums[j];
					int perc_idx = white_panel_precs[j];

					// Perc = 100 * Num/WBC ;
					complete += triplet_complete(panels[i], 100, perc_idx, num_idx, WHITE_PNL_WBC, orig_res, final_res, conv, changed_signals);
				}
			}
		}

		// Update changed signals
		if (update_signals(rec, iver, panels, panel_times, sigs_ids, changed_signals) < 0)
			return -1;

	}

	return 0;
}

// Completion of platelets panels
//.......................................................................................
int RepPanelCompleter::apply_platelets_completer(PidDynamicRec& rec, vector<int>& time_points) {
	vector<int>& sigs_ids = panel_signal_ids[REP_CMPLT_PLATELETS_PANEL];
	int n_sigs = (int)sigs_ids.size();

	vector<float>& orig_res = original_sig_res[REP_CMPLT_PLATELETS_PANEL];
	vector<float>& final_res = final_sig_res[REP_CMPLT_PLATELETS_PANEL];
	vector<float>& conv = sig_conversion_factors[REP_CMPLT_PLATELETS_PANEL];
	if (orig_res.empty() || final_res.empty() || conv.empty())
		MTHROW_AND_ERR("Error in RepPanelCompleter::apply_platelets_completer - please provice config/metadata file\n");

	// Loop on versions
	set<int> iteratorSignalIds(sigs_ids.begin(), sigs_ids.end());

	allVersionsIterator vit(rec, iteratorSignalIds);
	for (int iver = vit.init(); !vit.done(); iver = vit.next()) {

		int time_limit = (time_points.size()) ? time_points[iver] : -1;

		// Get Signals
		rec.usvs.resize(n_sigs);
		for (size_t i = 0; i < n_sigs; ++i)
			rec.uget(sigs_ids[i], iver, rec.usvs[i]);

		// Get Panels
		vector<int> panel_times;
		vector<vector<float> > panels;
		get_panels(rec.usvs, panel_times, panels, time_limit, PLT_PNL_LAST);

		// Complete
		vector<int> changed_signals(n_sigs, 0);
		for (size_t i = 0; i < panels.size(); i++)
			triplet_complete(panels[i], 100, PLT_PNL_MPV, PLT_PNL_PLT_HCT, PLT_PNL_PLTS, orig_res, final_res, conv, changed_signals);


		// Update changed signals
		if (update_signals(rec, iver, panels, panel_times, sigs_ids, changed_signals) < 0)
			return -1;
	}

	return 0;
}

// Completion of lipids panels
//.......................................................................................
int RepPanelCompleter::apply_lipids_completer(PidDynamicRec& rec, vector<int>& time_points) {

	vector<int>& sigs_ids = panel_signal_ids[REP_CMPLT_LIPIDS_PANEL];
	int n_sigs = (int)sigs_ids.size();

	vector<float>& orig_res = original_sig_res[REP_CMPLT_LIPIDS_PANEL];
	vector<float>& final_res = final_sig_res[REP_CMPLT_LIPIDS_PANEL];
	vector<float>& conv = sig_conversion_factors[REP_CMPLT_LIPIDS_PANEL];
	if (orig_res.empty() || final_res.empty() || conv.empty())
		MTHROW_AND_ERR("Error in RepPanelCompleter::apply_lipids_completer - please provice config/metadata file\n");

	// Loop on versions
	set<int> iteratorSignalIds(sigs_ids.begin(), sigs_ids.end());

	allVersionsIterator vit(rec, iteratorSignalIds);
	for (int iver = vit.init(); !vit.done(); iver = vit.next()) {
		int time_limit = (time_points.size()) ? time_points[iver] : -1;

		// Get Signals
		rec.usvs.resize(n_sigs);
		for (size_t i = 0; i < n_sigs; ++i)
			rec.uget(sigs_ids[i], iver, rec.usvs[i]);

		// Get Panels
		vector<int> panel_times;
		vector<vector<float> > panels;
		get_panels(rec.usvs, panel_times, panels, time_limit, LIPIDS_PNL_LAST);

		// Tryglicerids -> VLDL
		for (int iPanel = 0; iPanel < panels.size(); iPanel++) {
			if (panels[iPanel][LIPIDS_PNL_TRGS] != missing_val)
				panels[iPanel][LIPIDS_PNL_VLDL] = panels[iPanel][LIPIDS_PNL_TRGS] / 5.0F;
		}

		// Complete
		vector<int> changed_signals(LIPIDS_PNL_LAST, 0);
		for (size_t i = 0; i < panels.size(); i++) {
			int complete = 1;
			while (complete) {
				complete = 0;

				complete += triplet_complete(panels[i], 1, LIPIDS_PNL_HDL_OVER_CHOL, LIPIDS_PNL_HDL, LIPIDS_PNL_CHOL, orig_res, final_res, conv, changed_signals);
				complete += triplet_complete(panels[i], 1, LIPIDS_PNL_CHOL_OVER_HDL, LIPIDS_PNL_CHOL, LIPIDS_PNL_HDL, orig_res, final_res, conv, changed_signals);
				complete += triplet_complete(panels[i], 1, LIPIDS_PNL_HDL_OVER_LDL, LIPIDS_PNL_HDL, LIPIDS_PNL_LDL, orig_res, final_res, conv, changed_signals);
				complete += triplet_complete(panels[i], 1, LIPIDS_PNL_LDL_OVER_HDL, LIPIDS_PNL_LDL, LIPIDS_PNL_HDL, orig_res, final_res, conv, changed_signals);
				complete += triplet_complete(panels[i], 1, LIPIDS_PNL_HDL_OVER_NON_HDL, LIPIDS_PNL_HDL, LIPIDS_PNL_NON_HDL_CHOL, orig_res, final_res, conv, changed_signals);

				complete += reciprocal_complete(panels[i], 1, LIPIDS_PNL_HDL_OVER_LDL, LIPIDS_PNL_LDL_OVER_HDL, orig_res, final_res, conv, changed_signals);
				complete += reciprocal_complete(panels[i], 1, LIPIDS_PNL_HDL_OVER_CHOL, LIPIDS_PNL_CHOL_OVER_HDL, orig_res, final_res, conv, changed_signals);

				complete += sum_complete(panels[i], LIPIDS_PNL_CHOL, chol_types1, orig_res, final_res, conv, changed_signals);
				complete += sum_complete(panels[i], LIPIDS_PNL_CHOL, chol_types2, orig_res, final_res, conv, changed_signals);
			}
		}

		// VLDL -> Tryglicerids
		if (changed_signals[LIPIDS_PNL_VLDL]) {
			changed_signals[LIPIDS_PNL_TRGS] = 1;
			for (int iPanel = 0; iPanel < panels.size(); iPanel++) {
				if (panels[iPanel][LIPIDS_PNL_TRGS] == missing_val && panels[iPanel][LIPIDS_PNL_VLDL] != missing_val)
					panels[iPanel][LIPIDS_PNL_TRGS] = 5.0F * panels[iPanel][LIPIDS_PNL_VLDL];
			}
		}

		// Update changed signals
		if (update_signals(rec, iver, panels, panel_times, sigs_ids, changed_signals) < 0)
			return -1;

	}

	return 0;
}

// Completion of eGFR panels
//.......................................................................................
int RepPanelCompleter::apply_eGFR_completer(PidDynamicRec& rec, vector<int>& time_points) {

	vector<int>& sigs_ids = panel_signal_ids[REP_CMPLT_EGFR_PANEL];
	int n_sigs = (int)sigs_ids.size();

	vector<float>& orig_res = original_sig_res[REP_CMPLT_EGFR_PANEL];
	vector<float>& final_res = final_sig_res[REP_CMPLT_EGFR_PANEL];
	vector<float>& conv = sig_conversion_factors[REP_CMPLT_EGFR_PANEL];
	if (orig_res.empty() || final_res.empty() || conv.empty())
		MTHROW_AND_ERR("Error in RepPanelCompleter::apply_eGFR_completer - please provice config/metadata file\n");

	//  Age & Gender
	int age, bYear, gender;
	if (perpare_for_age_and_gender(rec, age, bYear, gender) < 0)
		return -1;


	// Loop on versions
	set<int> iteratorSignalIds(sigs_ids.begin(), sigs_ids.end());

	allVersionsIterator vit(rec, iteratorSignalIds);
	for (int iver = vit.init(); !vit.done(); iver = vit.next()) {

		int time_limit = (time_points.size()) ? time_points[iver] : -1;

		// Get Signals
		rec.usvs.resize(n_sigs);
		for (size_t i = 0; i < n_sigs; ++i)
			rec.uget(sigs_ids[i], iver, rec.usvs[i]);

		// Get Panels
		vector<int> panel_times;
		vector<vector<float> > panels;
		get_panels(rec.usvs, panel_times, panels, time_limit, EGFR_PNL_LAST);

		// Complete
		vector<int> changed_signals(n_sigs, 0);
		float current_age;
		for (size_t i = 0; i < panels.size(); i++) {
			current_age = (float)(1900 + med_time_converter.convert_date(MedTime::Years, panel_times[i]) - bYear);
			egfr_complete(panels[i], current_age, gender, orig_res, final_res, conv, changed_signals);
		}


		// Update changed signals
		if (update_signals(rec, iver, panels, panel_times, sigs_ids, changed_signals) < 0)
			return -1;
	}

	return 0;
}

// Age/Gender util
//.......................................................................................
int RepPanelCompleter::perpare_for_age_and_gender(PidDynamicRec& rec, int& age, int& bYear, int& gender) {

	rec.uget(genderId, 0);
	if (rec.usv.len == 0) {
		MERR("No Gender given for %d\n", rec.pid);
		return -1;
	}
	gender = (int)(rec.usv.Val(0));
	rec.uget(bdateId, 0);
	if (rec.usv.len == 0) {
		MERR("No BDATE given for %d\n", rec.pid);
		return -1;
	}
	int bdate_v = (int)(rec.usv.Val(0));
	bYear = int(bdate_v / 10000);

	return 0;
}

// Completion of BMI panels
//.......................................................................................
int RepPanelCompleter::apply_BMI_completer(PidDynamicRec& rec, vector<int>& time_points) {

	vector<int>& sigs_ids = panel_signal_ids[REP_CMPLT_BMI_PANEL];
	int n_sigs = (int)sigs_ids.size();

	vector<float>& orig_res = original_sig_res[REP_CMPLT_BMI_PANEL];
	vector<float>& final_res = final_sig_res[REP_CMPLT_BMI_PANEL];
	vector<float>& conv = sig_conversion_factors[REP_CMPLT_BMI_PANEL];
	if (orig_res.empty() || final_res.empty() || conv.empty())
		MTHROW_AND_ERR("Error in RepPanelCompleter::apply_BMI_completer - please provice config/metadata file\n");

	// Loop on versions
	set<int> iteratorSignalIds(sigs_ids.begin(), sigs_ids.end());

	allVersionsIterator vit(rec, iteratorSignalIds);
	for (int iver = vit.init(); !vit.done(); iver = vit.next()) {

		int time_limit = (time_points.size()) ? time_points[iver] : -1;

		// Get Signals
		rec.usvs.resize(n_sigs);
		for (size_t i = 0; i < n_sigs; ++i)
			rec.uget(sigs_ids[i], iver, rec.usvs[i]);

		// Get Panels
		vector<int> panel_times;
		vector<vector<float> > panels;
		get_panels(rec.usvs, panel_times, panels, time_limit, BMI_PNL_LAST);

		// Get Square of height (in meters)
		for (int iPanel = 0; iPanel < panels.size(); iPanel++) {
			if (panels[iPanel][BMI_PNL_HGT] != missing_val)
				panels[iPanel][BMI_PNL_HGT_SQR] = (panels[iPanel][BMI_PNL_HGT] / 100.0F)*(panels[iPanel][BMI_PNL_HGT] / 100.0F);
		}

		// Complete
		vector<int> changed_signals(BMI_PNL_LAST, 0);
		for (size_t i = 0; i < panels.size(); i++)
			triplet_complete(panels[i], 1, BMI_PNL_BMI, BMI_PNL_WGT, BMI_PNL_HGT_SQR, orig_res, final_res, conv, changed_signals);


		// Get height in cm
		if (changed_signals[BMI_PNL_HGT_SQR]) {
			changed_signals[BMI_PNL_HGT] = 1;
			for (int iPanel = 0; iPanel < panels.size(); iPanel++) {
				if (panels[iPanel][BMI_PNL_HGT] == missing_val && panels[iPanel][BMI_PNL_HGT_SQR] != missing_val)
					panels[iPanel][BMI_PNL_HGT] = completer_round(sqrt(panels[iPanel][BMI_PNL_HGT_SQR]) * 100.0F, orig_res[BMI_PNL_HGT],
						final_res[BMI_PNL_HGT], conv[BMI_PNL_HGT]);
			}
		}

		// Update changed signals
		if (update_signals(rec, iver, panels, panel_times, sigs_ids, changed_signals) < 0)
			return -1;
	}

	return 0;
}

void RepPanelCompleter::convert_gcs_signals(vector<float> &panel) {
	//convert channels 1,2,3 (Eye, Motor, Verbal) to values if not missing values
	if (panel[1] != MED_MAT_MISSING_VALUE) {
		if (eye_vals.find(panel[1]) == eye_vals.end()) {
			MTHROW_AND_ERR("Missing value %d in GCS_Eye\n", (int)panel[1]);
		}
		else {
			panel[1] = eye_vals[panel[1]];
		}
	}

	if (panel[2] != MED_MAT_MISSING_VALUE) {
		if (motor_vals.find(panel[2]) == motor_vals.end()) {
			MTHROW_AND_ERR("Missing value %d in GCS_Motor\n", (int)panel[2]);
		}
		else {
			panel[2] = motor_vals[panel[2]];
		}
	}

	if (panel[3] != MED_MAT_MISSING_VALUE) {
		if (verbal_vals.find(panel[3]) == verbal_vals.end()) {
			MTHROW_AND_ERR("Missing value %d in GCS_Motor\n", (int)panel[3]);
		}
		else {
			panel[3] = verbal_vals[panel[3]];
		}
	}

}

int RepPanelCompleter::apply_GCS_completer(PidDynamicRec& rec, vector<int>& time_points) {

	vector<int>& sigs_ids = panel_signal_ids[REP_CMPLT_GCS];
	int n_sigs = (int)sigs_ids.size();

	vector<float>& orig_res = original_sig_res[REP_CMPLT_GCS];
	vector<float>& final_res = final_sig_res[REP_CMPLT_GCS];
	vector<float>& conv = sig_conversion_factors[REP_CMPLT_GCS];

	// Loop on versions
	set<int> iteratorSignalIds(sigs_ids.begin(), sigs_ids.end());

	allVersionsIterator vit(rec, iteratorSignalIds);
	for (int iver = vit.init(); !vit.done(); iver = vit.next()) {

		int time_limit = (time_points.size()) ? time_points[iver] : -1;

		// Get Signals
		rec.usvs.resize(n_sigs);
		for (size_t i = 0; i < n_sigs; ++i)
			rec.uget(sigs_ids[i], iver, rec.usvs[i]);

		// Get Panels
		vector<int> panel_times;
		vector<vector<float> > panels;
		get_panels(rec.usvs, panel_times, panels, time_limit, GCS_PNL_LAST);
		for (size_t i = 0; i < panels.size(); ++i)
			convert_gcs_signals(panels[i]);

		// Complete
		vector<int> changed_signals(GCS_PNL_LAST, 0);
		for (size_t i = 0; i < panels.size(); i++)
			sum_complete(panels[i], GCS_PNL, gcs_panel_parts, orig_res, final_res, conv, changed_signals);

		// Update changed signals
		if (update_signals(rec, iver, panels, panel_times, sigs_ids, changed_signals) < 0)
			return -1;
	}

	return 0;
}


// Utilities
// Generate panels from signals
//.......................................................................................

struct candidate_compare {
	bool operator() (const pair<int, int>& lhs, const pair<int, int>& rhs) const {
		return (lhs.first < rhs.first) || (lhs.first == rhs.first && lhs.second < rhs.second);
	}
};

void RepPanelCompleter::get_panels(vector<UniversalSigVec>& usvs, vector<int>& panel_times, vector<vector<float>> &panels, int time_limit, int panel_size) {

	// Prepare
	panels.clear();
	panel_times.clear();
	vector<vector<int> > val_counters;

	set<pair<int, int>, candidate_compare> candidates; // Which signal should we insert next ;

	vector<int> idx(usvs.size(), 0);
	for (int i = 0; i < usvs.size(); i++) {
		if (usvs[i].len > 0)
			candidates.insert({ usvs[i].Time(0),i });
	}

	int currentTime = -1;
	while (!candidates.empty()) {
		// Get the next element
		std::set<pair<int, int>, candidate_compare>::iterator it = candidates.begin();

		// Have we reached the time-limit ?
		if (time_limit != -1 && it->first > time_limit)
			break;

		// New Time Point
		if (it->first != currentTime) {
			currentTime = it->first;
			panel_times.push_back(it->first);
			panels.push_back(vector<float>(panel_size, missing_val));
			if (sim_val_handler == SIM_VAL_MEAN || sim_val_handler == SIM_VAL_REM || sim_val_handler == SIM_VAL_REM_DIFF)
				val_counters.push_back(vector<int>(panel_size, 0));
		}
		int sig_idx = it->second;
		candidates.erase(it);

		// Update panel
		if (sim_val_handler == SIM_VAL_MEAN) {
			if (panels.back()[sig_idx] == missing_val)
				panels.back()[sig_idx] = usvs[sig_idx].Val(idx[sig_idx]);
			else
				panels.back()[sig_idx] += usvs[sig_idx].Val(idx[sig_idx]);
			val_counters.back()[sig_idx]++;
		}
		else if (sim_val_handler == SIM_VAL_REM) {
			if (val_counters.back()[sig_idx] != 0)
				panels.back()[sig_idx] = missing_val;
			else
				panels.back()[sig_idx] = usvs[sig_idx].Val(idx[sig_idx]);
			val_counters.back()[sig_idx]++;
		}
		else if (sim_val_handler == SIM_VAL_REM_DIFF) {
			if (val_counters.back()[sig_idx] != 0 && panels.back()[sig_idx] != usvs[sig_idx].Val(idx[sig_idx]))
				panels.back()[sig_idx] = missing_val;
			else
				panels.back()[sig_idx] = usvs[sig_idx].Val(idx[sig_idx]);
			val_counters.back()[sig_idx]++;
		}
		else if (sim_val_handler == SIM_VAL_LAST_VAL || panels.back()[sig_idx] == missing_val)
			panels.back()[sig_idx] = usvs[sig_idx].Val(idx[sig_idx]);
		idx[sig_idx]++;

		// New candidate
		if (usvs[sig_idx].len > idx[sig_idx]) {
			int time = usvs[sig_idx].Time(idx[sig_idx]);
			if (time_limit == -1 || time <= time_limit)
				candidates.insert({ time ,sig_idx });
		}
	}

	// Get Means
	if (sim_val_handler == SIM_VAL_MEAN) {
		for (unsigned int i = 0; i < panels.size(); i++) {
			for (int j = 0; j < panel_size; j++) {
				if (val_counters[i][j] > 0)
					panels[i][j] /= val_counters[i][j];
			}
		}
	}

	/* Print Panels
	cerr << "Time Limit = " << time_limit << "\n";
	for (int i = 0; i < panels.size(); i++) {
		cerr << panel_times[i];
		for (int j = 0; j < panels[i].size(); j++)
			cerr << " " << panels[i][j];
			cerr << "\n";
	}
	*/

}

// Complete X = factor*Y/Z ; Y = X*Z/factor ; Z = factor*Y/X
//.......................................................................................
int RepPanelCompleter::triplet_complete(vector<float>& panel, float factor, int x_idx, int y_idx, int z_idx, vector<float>& orig_res, vector<float>& final_res, vector<float>& conv, vector<int>& changed) {

	// Try completing ...
	if (panel[x_idx] == missing_val && panel[y_idx] != missing_val && panel[z_idx] != missing_val && panel[z_idx] != 0.0) {
		panel[x_idx] = factor * panel[y_idx] / panel[z_idx];
		if (x_idx < orig_res.size())
			panel[x_idx] = completer_round(panel[x_idx], orig_res[x_idx], final_res[x_idx], conv[x_idx]);
		changed[x_idx] = 1;
		return 1;
	}
	else if (panel[y_idx] == missing_val && panel[x_idx] != missing_val && panel[z_idx] != missing_val) {
		panel[y_idx] = panel[x_idx] * panel[z_idx] / factor;
		if (y_idx < orig_res.size())
			panel[y_idx] = completer_round(panel[y_idx], orig_res[y_idx], final_res[y_idx], conv[y_idx]);
		changed[y_idx] = 1;
		return 1;
	}
	else if (panel[z_idx] == missing_val && panel[y_idx] != missing_val && panel[x_idx] != missing_val && panel[x_idx] != 0) {
		panel[z_idx] = factor * panel[y_idx] / panel[x_idx];
		if (z_idx < orig_res.size())
			panel[z_idx] = completer_round(panel[z_idx], orig_res[z_idx], final_res[z_idx], conv[z_idx]);
		changed[z_idx] = 1;
		return 1;
	}
	else
		return 0;
}

// Complete sum = Sum of summands
//.......................................................................................
int RepPanelCompleter::sum_complete(vector<float>& panel, int sum, vector<int>& summands, vector<float>& orig_res, vector<float>& final_res, vector<float>& conv, vector<int>& changed) {

	int npresent = 0;
	float sumVal = 0.0;
	int missing = -1;
	for (int i = 0; i < summands.size(); i++) {
		if (panel[summands[i]] != missing_val) {
			npresent++;
			sumVal += panel[summands[i]];
		}
		else
			missing = summands[i];
	}

	// Can we complete ?
	if (npresent == summands.size() && panel[sum] == missing_val) {
		panel[sum] = sumVal;
		if (sum < orig_res.size())
			panel[sum] = completer_round(panel[sum], orig_res[sum], final_res[sum], conv[sum]);
		changed[sum] = 1;
		return 1;
	}
	else if (npresent == summands.size() - 1 && panel[sum] != missing_val) {
		float val = panel[sum] - sumVal;
		if (val >= 0) {
			panel[missing] = val;
			if (missing < orig_res.size())
				panel[missing] = completer_round(panel[missing], orig_res[missing], final_res[missing], conv[missing]);
			changed[missing] = 1;
			return 1;
		}
	}

	return 0;
}

// Complete x = factor/y
//.......................................................................................
int RepPanelCompleter::reciprocal_complete(vector<float>& panel, float factor, int x_idx, int y_idx, vector<float>& orig_res, vector<float>& final_res, vector<float>& conv, vector<int>& changed) {

	// Can We complete ?
	if (panel[x_idx] == missing_val && panel[y_idx] != missing_val && panel[y_idx] != 0.0) {
		panel[x_idx] = factor / panel[y_idx];
		if (x_idx < orig_res.size())
			panel[x_idx] = completer_round(panel[x_idx], orig_res[x_idx], final_res[x_idx], conv[x_idx]);
		changed[x_idx] = 1;
		return 1;
	}
	else if (panel[y_idx] == missing_val && panel[x_idx] != missing_val && panel[x_idx] != 0.0) {
		panel[y_idx] = factor / panel[x_idx];
		if (y_idx < orig_res.size())
			panel[y_idx] = completer_round(panel[y_idx], orig_res[y_idx], final_res[y_idx], conv[y_idx]);
		changed[y_idx] = 1;
		return 1;
	}

	return 0;
}

// Complete eGFRs
//.......................................................................................
int RepPanelCompleter::egfr_complete(vector<float>& panel, float age, int gender, vector<float>& orig_res, vector<float>& final_res, vector<float>& conv, vector<int>& changed) {

	int complete = 0;
	float egfr;
	if (panel[EGFR_PNL_CRT] != missing_val && panel[EGFR_PNL_CRT] != 0.0) {
		egfr = completer_round(get_eGFR_CKD_EPI(age, panel[EGFR_PNL_CRT], gender), orig_res[EGFR_PNL_CKD_EPI], final_res[EGFR_PNL_CKD_EPI], conv[EGFR_PNL_CKD_EPI]);
		if (isfinite(egfr)) {
			panel[EGFR_PNL_CKD_EPI] = egfr;
			changed[EGFR_PNL_CKD_EPI] = 1;
			complete = 1;
		}

		egfr = completer_round(get_eGFR_MDRD(age, panel[EGFR_PNL_CRT], gender), orig_res[EGFR_PNL_MDRD], final_res[EGFR_PNL_MDRD], conv[EGFR_PNL_MDRD]);
		if (isfinite(egfr)) {
			panel[EGFR_PNL_MDRD] = egfr;
			changed[EGFR_PNL_MDRD] = 1;
			complete = 1;
		}
	}

	return complete;
}

// Updating signals in dynamic-rec
//.......................................................................................
int RepPanelCompleter::update_signals(PidDynamicRec& rec, int iver, vector<vector<float>>& panels, vector<int>& panel_times, vector<int>& sigs_ids, vector<int>& changed) {

	for (int iSig = 0; iSig < sigs_ids.size(); iSig++) {
		if (changed[iSig]) {

			// We need to take care of the (rare) events of multiple values at the same time.
			// Look back at the original signal :
			rec.uget(sigs_ids[iSig], iver);
			int nEXtra = 0;
			unordered_map<int, vector<float> > multiple_values;
			for (unsigned int i = 1; i < rec.usv.len; i++) {
				int time = rec.usv.Time(i);
				if (time == rec.usv.Time(i - 1)) {
					if (multiple_values[time].empty())
						multiple_values[time].push_back(rec.usv.Val(i - 1));
					multiple_values[time].push_back(rec.usv.Val(i));
					nEXtra++;
				}
			}
			int val_ch_sz = rec.usv.n_val_channels();
			int time_ch_sz = rec.usv.n_time_channels();
			// Generate new data
			int trueSize = 0, data_idx = 0;
			vector<float> values(val_ch_sz * (panels.size() + nEXtra), missing_val);
			vector<int> times(time_ch_sz * (panels.size() + nEXtra));
			if (nEXtra == 0) { // Easy case - no multiple values 			
				for (int iPanel = 0; iPanel < panels.size(); iPanel++) {
					if (panels[iPanel][iSig] != missing_val) {
						values[static_cast<size_t>(trueSize) * static_cast<size_t>(val_ch_sz)] = panels[iPanel][iSig];
						times[static_cast<size_t>(trueSize) * static_cast<size_t>(time_ch_sz)] = panel_times[iPanel];
						if (data_idx < rec.usv.len && rec.usv.Time(data_idx) == panel_times[iPanel]) {
							for (int i = 1; i < val_ch_sz; ++i)
								values[trueSize * val_ch_sz + i] = rec.usv.Val(data_idx, i);
							for (int i = 1; i < time_ch_sz; ++i)
								times[trueSize * time_ch_sz + i] = rec.usv.Time(data_idx, i);
						}
						++trueSize;
						data_idx += data_idx < rec.usv.len && rec.usv.Time(data_idx) == panel_times[iPanel]; //move next when reach panel time
					}
				}
			}
			else { // In case of multiple values, take them
				for (int iPanel = 0; iPanel < panels.size(); iPanel++) {
					int time = panel_times[iPanel];
					if (multiple_values.find(time) != multiple_values.end()) {
						for (float value : multiple_values[time]) {
							values[static_cast<size_t>(trueSize) * static_cast<size_t>(val_ch_sz)] = value;
							times[static_cast<size_t>(trueSize) * static_cast<size_t>(time_ch_sz)] = time;
							if (data_idx < rec.usv.len && rec.usv.Time(data_idx) == time) {
								for (int i = 1; i < val_ch_sz; ++i)
									values[trueSize * val_ch_sz + i] = rec.usv.Val(data_idx, i);
								for (int i = 1; i < time_ch_sz; ++i)
									times[trueSize * time_ch_sz + i] = rec.usv.Time(data_idx, i);
							}
							trueSize++;
							data_idx += data_idx < rec.usv.len && rec.usv.Time(data_idx) == time; //move next when reach panel time
						}
					}
					else if (panels[iPanel][iSig] != missing_val) {
						values[static_cast<size_t>(trueSize) * static_cast<size_t>(val_ch_sz)] = panels[iPanel][iSig];
						times[static_cast<size_t>(trueSize) * static_cast<size_t>(time_ch_sz)] = time;
						if (data_idx < rec.usv.len && rec.usv.Time(data_idx) == time) {
							for (int i = 1; i < val_ch_sz; ++i)
								values[trueSize * val_ch_sz + i] = rec.usv.Val(data_idx, i);
							for (int i = 1; i < time_ch_sz; ++i)
								times[trueSize * time_ch_sz + i] = rec.usv.Time(data_idx, i);
						}
						trueSize++;
						data_idx += data_idx < rec.usv.len && rec.usv.Time(data_idx) == time; //move next when reach panel time
					}
				}
			}

			if (rec.set_version_universal_data(sigs_ids[iSig], iver, &(times[0]), &(values[0]), trueSize) < 0)
				return -1;
		}
	}

	return 0;
}

// Read Signals metadata - extract resolutions and conversions from a csv
//.......................................................................................
void RepPanelCompleter::read_metadata() {

	if (metadata_file.empty()) {
		original_sig_res.resize(REP_CMPLT_LAST);
		final_sig_res.resize(REP_CMPLT_LAST);
		sig_conversion_factors.resize(REP_CMPLT_LAST);
		MWARN("Warning: No metadata file given for RepPanelCompleter\n");
		return;
	}

	// Open
	ifstream infile;
	infile.open(metadata_file.c_str(), ifstream::in);
	if (!infile.is_open())
		MTHROW_AND_ERR("Cannot open %s for reading\n", metadata_file.c_str());

	// Read
	int header = 1;
	string thisLine;
	map<string, int> columns;
	map<string, float> all_original_res, all_final_res, all_conversion_factors;
	vector<string> required = { "Name","FinalFactor","OrigResolution","FinalResolution" };

	while (!infile.eof()) {
		getline(infile, thisLine);
		if (thisLine.empty() || thisLine.substr(0, 1) == "#")
			continue;

		vector<string> fields;
		boost::split(fields, thisLine, boost::is_any_of(","));

		if (header == 1) {
			for (int iCol = 0; iCol < fields.size(); iCol++)
				columns[fields[iCol]] = iCol;

			for (string& req : required) {
				if (columns.find(req) == columns.end())
					MTHROW_AND_ERR("Cannot find %s in meta-data file \'%s\'", req.c_str(), metadata_file.c_str());
			}

			header = 0;
		}
		else {
			string sigName = fields[columns["Name"]];
			all_original_res[sigName] = med_stof(fields[columns["OrigResolution"]]);
			all_final_res[sigName] = med_stof(fields[columns["FinalResolution"]]);
			all_conversion_factors[sigName] = med_stof(fields[columns["FinalFactor"]]);
		}
	}

	// Fill in
	original_sig_res.resize(panel_signal_names.size());
	final_sig_res.resize(panel_signal_names.size());
	sig_conversion_factors.resize(panel_signal_names.size());

	for (int iPanel = 0; iPanel < panel_signal_names.size(); iPanel++) {
		original_sig_res[iPanel].resize(panel_signal_names[iPanel].size());
		final_sig_res[iPanel].resize(panel_signal_names[iPanel].size());
		sig_conversion_factors[iPanel].resize(panel_signal_names[iPanel].size());

		for (int iSig = 0; iSig < panel_signal_names[iPanel].size(); iSig++) {

			if (all_original_res.find(panel_signal_names[iPanel][iSig]) == all_original_res.end())
				MTHROW_AND_ERR("Cannot find metadata for signal %s\n", panel_signal_names[iPanel][iSig].c_str());

			original_sig_res[iPanel][iSig] = all_original_res[panel_signal_names[iPanel][iSig]];
			final_sig_res[iPanel][iSig] = all_final_res[panel_signal_names[iPanel][iSig]];
			sig_conversion_factors[iPanel][iSig] = all_conversion_factors[panel_signal_names[iPanel][iSig]];
		}
	}


	infile.close();
}

void RepPanelCompleter::print() {
	string panels = "";
	vector<string> reverse_map(REP_CMPLT_LAST);
	for (auto it = panel2type.begin(); it != panel2type.end(); ++it)
		reverse_map[it->second] = it->first;

	for (int i = 0; i < panel_signal_names.size(); ++i)
		if (!panel_signal_names[i].empty()) {
			if (!panels.empty())
				panels += ",";
			panels += reverse_map[i];
		}
	string req_ls = medial::io::get_list(req_signals);
	string aff_ls = medial::io::get_list(aff_signals);
	MLOG("RepPanelCompleter: panels=%s, missing_val=%2.4f, sim_val_handler=%d, metadata_file=%s\n",
		panels.c_str(), missing_val, sim_val_handler, metadata_file.c_str(), req_ls.c_str(), aff_ls.c_str());
}