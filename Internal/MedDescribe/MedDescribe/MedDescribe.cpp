#include "MedDescribe.h"

#include <MedUtils/MedUtils/MedGenUtils.h>
#include <Logger/Logger/Logger.h>
#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_APP
#define LOCAL_LEVEL	LOG_DEF_LEVEL

//====================================================================================================
int MedMutualDist::init(map<string, string>& mapper)
{
	for (auto entry : mapper) {
		string field = entry.first;

		if (field == "sig1") { sig1 = entry.second; }
		else if (field == "sig2") { sig2 = entry.second; }
		else if (field == "sig1_time_ch") { sig1_time_ch = stoi(entry.second); }
		else if (field == "sig1_val_ch") { sig1_val_ch = stoi(entry.second); }
		else if (field == "sig2_time_ch") { sig2_time_ch = stoi(entry.second); }
		else if (field == "sig2_val_ch") { sig2_val_ch = stoi(entry.second); }
		else if (field == "sig1_categ") { sig1_is_categorial = stoi(entry.second); }
		else if (field == "min_val1") { min_val1 = stof(entry.second); }
		else if (field == "max_val1") { max_val1 = stof(entry.second); }
		else if (field == "min_val2") { min_val2 = stof(entry.second); }
		else if (field == "max_val2") { max_val2 = stof(entry.second); }
		else if (field == "bin_size") { bin_size = stof(entry.second); }
		else if (field == "min_time") { min_time = stoi(entry.second); }
		else if (field == "max_time") { max_time = stoi(entry.second); }
		else if (field == "gender_mask") { gender_mask = stoi(entry.second); }
		else if (field == "jump1") { jump1_time = stoi(entry.second); }
		else if (field == "win_from") { win_from = stoi(entry.second); }
		else if (field == "win_to") { win_to = stoi(entry.second); }
		else if (field == "win_time_unit") { win_time_unit = med_time_converter.string_to_type(entry.second); }
		else
			MLOG("Unknown parameter \'%s\' for MedMutialDist\n", field.c_str());
	}

	return 0;
}

//====================================================================================================
int MedMutualDist::collect_values(const string &fname)
{
	MedRepository rep;

	if (rep.read_all(fname, pids_to_check, { "BDATE","GENDER", sig1, sig2 }) < 0) {
		MERR("ERROR: can't open repository %s\n", fname.c_str());
		return -1;
	}

	return collect_values(rep);
}


//====================================================================================================
struct index_times {
	int ind;
	int time;
	int win_time;
	float val;
};

struct sampling_candidate {
	int ind1;
	int ind2;
	float val1;
	float val2;
	int time1;
	int time2;
	int dtime;
	int wtime1;
};

//====================================================================================================
int MedMutualDist::collect_values(MedRepository &rep)
{
	int gender_sid = rep.sigs.sid("GENDER");
	int sig1_sid = rep.sigs.sid(sig1);
	int sig2_sid = rep.sigs.sid(sig2);

	int len;
	UniversalSigVec usv1;
	UniversalSigVec usv2;

	int sig1_time_unit = rep.sigs.Sid2Info[sig1_sid].time_unit;
	int sig2_time_unit = rep.sigs.Sid2Info[sig2_sid].time_unit;

	for (auto pid : rep.pids) {

		//MLOG("Working on pid %d\n", pid);

		int gender = medial::repository::get_value(rep, pid, gender_sid);
		if (!(gender & gender_mask)) continue;

		//MLOG("gender is %d\n", gender);
		// ToDo: add age test

		rep.uget(pid, sig1_sid, usv1);
		rep.uget(pid, sig2_sid, usv2);

		//MLOG("usv1.len %d usv2.len %d\n", usv1.len, usv2.len);
		if (usv1.len > 0 && usv2.len > 0) {

			vector<index_times> sig2_times;
			for (int i=0; i<usv2.len; i++) {
				index_times it;
				it.val = usv2.Val(i, sig2_val_ch);
				if (it.val >= min_val2 && it.val <= max_val2) {
					it.ind = i;
					it.time = usv2.Time(i, sig2_time_ch);
					it.win_time = med_time_converter.convert_times(sig2_time_unit, win_time_unit, it.time);
				//	MLOG("sig2 i %d ind %d time %d win_time %d val %f\n", i, it.ind, it.time, it.win_time, it.val);
					sig2_times.push_back(it);
				}

				//MLOG("sig2 i %d ind %d time %d win_time %d\n", i, sig2_times[i].ind, sig2_times[i].time, sig2_times[i].win_time);
			}

			//MLOG("sig2_times size %d\n", sig2_times.size());

			vector<sampling_candidate> candidates; // index, time in sig1 , possible times in sig2

			for (int i=0; i<usv1.len; i++) {
				//MLOG("usv1 i=%d\n", i);
				int itime1 = usv1.Time(i, sig1_time_ch);

				// check if time in range
				if (itime1 >= min_time && itime1 <= max_time) {

					float ival = usv1.Val(i, sig1_val_ch);

					//MLOG("i=%d itime1 %d ival %f\n", i, itime1, ival);
					// check value is in range
					if (ival >= min_val1 && ival <= max_val1) {

						int iwtime1 = med_time_converter.convert_times(sig1_time_unit, win_time_unit, itime1);

						//MLOG("i=%d iwtime1 %d\n", i, iwtime1);
						// now have to find closest value in sig2 that's in the window frame
						int min_dist = -1;
						int min_j = -1;
						for (int j=0; j<sig2_times.size(); j++) {
							int dtime = iwtime1 - sig2_times[j].win_time;
							//MLOG("j=%d dtime %d win[%d,%d]\n", j, dtime, win_from, win_to);
							if (dtime >= win_to && dtime <= win_from) {
								//MLOG("dtime in range\n");
								if (min_dist < 0 || abs(dtime) < min_dist) {
									min_dist = abs(dtime);
									min_j = j;
									//MLOG("min_j is %d\n", min_j);
								}
							}
							if (dtime < win_to) break; // all next ones will be over it too, as dtime only decreases.
						}

						if (min_j >= 0) {
							sampling_candidate sc;
							sc.ind1 = i;
							sc.ind2 = min_j;
							sc.time1 = itime1;
							sc.time2 = sig2_times[min_j].time;
							sc.dtime = min_dist;
							sc.wtime1 = iwtime1;
							sc.val1 = usv1.Val(i, sig1_val_ch);
							//sc.val2 = usv2.Val(sig2_times[min_j].ind, sig2_val_ch);
							sc.val2 = sig2_times[min_j].val;
							//MLOG("min_j %d ind1 %d ind2 %d time1 %d time2 %d dtime %d wtime1 %d val1 %f val2 %f\n",
							//	min_j, sc.ind1, sc.ind2, sc.time1, sc.time2, sc.dtime, sc.wtime1, sc.val1, sc.val2);
							candidates.push_back(sc);
						}

					}

				}
			}

			//MLOG("candidates size %d\n", candidates.size());
			// now we have all out candidates and are ready to push them into the collected values
			if (candidates.size() > 0) {

				int base_win_time = med_time_converter.convert_date(win_time_unit, 19000101);

				int i = 0;
				while (i < candidates.size()) {
					int curr_cell = (candidates[i].wtime1-base_win_time)/jump1_time;
					int last_j = i;
					for (int j=i+1; j<candidates.size(); j++) {
						int jcell = (candidates[j].wtime1-base_win_time)/jump1_time;
						if (jcell == curr_cell)
							last_j = j;
						else
							break;
					}
					
					// choose a random candidate from all those in cell
					int k = rand_N(last_j - i + 1) + i;

					// collect it
					if (sig1_is_categorial) {
						values[candidates[k].val1].push_back(candidates[k].val2);
					}
					else {
						float bin = min_val1 + bin_size*((int)((candidates[k].val1-min_val1)/bin_size));
						//MLOG("Inserting to bin %f :: %f\n", bin, candidates[k].val2);
						values[bin].push_back(candidates[k].val2);
						//MLOG("bin %f :: %d\n", bin, values[bin].size());
					}

					// next
					i = last_j + 1;
				}



			}

		}

	}

	return 0;
}


//====================================================================================================
int VecMoments::get_for_vec(vector<float> &v)
{
	// init moments
	N = 0; N_pos = 0; N_neg = 0; N0 = 0; N_diff_vals = 0; N_out_of_range = 0;
	vmin = (float)1e10; vmax = -(float)1e10;
	mean = 0; median = 0; std = 1;
	quantiles_vals.clear();

	if (v.size() == 0) return 0;

	sort(v.begin(), v.end());
	double sum = 0;
	double sum_sq = 0;
	double n = 0;
	N_diff_vals = 1;
	float prev = v[0];
	for (auto val : v) {
		N++;
		if (val == 0) N0++;
		if (val > 0) N_pos++;
		if (val < 0) N_neg++;
		if (val != prev) { N_diff_vals++; prev = val; }
		if (val > max_val || val < min_val) 
			N_out_of_range++;
		else {
			sum += val;
			sum_sq += static_cast<double>(val) * static_cast<double>(val);
			n++;
		}
	}

	if (n > 0) {
		sum /= n;
		sum_sq /= n;
		mean = (float)sum;
		std = (float)sqrt(sum_sq - sum*sum);
	}

	quantiles_vals.resize(quantiles.size());
	for (int i=0; i<quantiles.size(); i++)
		quantiles_vals[i] = v[(int)(quantiles[i]*(float)v.size())];

	median = v[(int)(0.5*(float)v.size())];
	return 0;
}

//====================================================================================================
void VecMoments::print(const string prefix)
{

}