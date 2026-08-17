#include "InfraMed.h"

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_REP
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

int InMemRepData::insertData_to_buffer(int pid, int sid, int *time_data, float *val_data, int n_time, int n_val, 
	const MedSignals &sigs, map<pair<int, int>, pair<int, vector<char>>> &data) {
	int n_time_ch = sigs.Sid2Info[sid].n_time_channels;
	int n_val_ch = sigs.Sid2Info[sid].n_val_channels;

	int n_elem_by_time = 0;
	int n_elem_by_val = 0;
	if (n_time_ch > 0) n_elem_by_time = n_time/n_time_ch;
	if (n_val_ch > 0) n_elem_by_val = n_val/n_val_ch;


	//MLOG(">>>> sid %d ch(%d,%d) elem(%d,%d) n(%d,%d)\n", sid, n_time_ch, n_val_ch, n_elem_by_time, n_elem_by_val, n_time, n_val);
	// a few sanity tests
	if ((n_time != n_elem_by_time*n_time_ch) || (n_val != n_elem_by_val*n_val_ch) || 
	    (n_time_ch == 0 && n_time > 0) || (n_val_ch == 0 && n_val > 0) ||
	    (n_time_ch > 0 && n_time == 0) || (n_val_ch > 0 && n_val == 0) || 
		(n_time_ch > 0 && n_val_ch > 0 && n_elem_by_time != n_elem_by_val))
	{
		MERR("ERROR: InMemRepData: non matching time/value numbers for pid %d, sid %d : needed (%d,%d) per element , got (%d,%d) ...\n",
			pid, sid, n_time_ch, n_val_ch, n_time, n_val);
		return -1;
	}
	int n_elem = max(n_elem_by_time, n_elem_by_val);

	int len_bytes = sigs.Sid2Info[sid].bytes_len;

	vector<char> elem(static_cast<size_t>(len_bytes) * static_cast<size_t>(n_elem));

	int type = sigs.Sid2Info[sid].type;

	//MLOG("pid %d sid %d type %d len_bytes %d nelem %d\n", pid, sid, type, len_bytes, n_elem);
	int *tdata = time_data;
	float *vdata = val_data;

	if (n_time_ch == 0) tdata = NULL;
	if (n_val_ch == 0) vdata = NULL;

	//MLOG("pid %d sid %d type %d len_bytes %d nelem %d\n", pid, sid, type, len_bytes, n_elem);
	if (type == T_Generic) {
		GenericSigVec gsv;
		gsv.init(sigs.Sid2Info[sid]);
		gsv.set_data(&elem[0], n_elem);
		for (int i = 0; i < n_elem; i++) {
			gsv.Set(i, tdata, vdata);
			if (tdata) tdata += n_time_ch;
			if (vdata) vdata += n_val_ch;
		}
	}
	else {
		for (int i = 0; i < n_elem; i++) {
			if (MedSignalsSingleElemFill(type, &elem[static_cast<size_t>(len_bytes) * static_cast<size_t>(i)], tdata, vdata) < 0) {
				MERR("ERROR: InMemRepData::insertData failed fill element %d/%d.", i, n_elem);
				return -1;
			}

			if (tdata) tdata += n_time_ch;
			if (vdata) vdata += n_val_ch;
		}
	}
	// all is ready -> we push it to the map
	pair<int, int> pid_sid(pid, sid);
	pair<int, vector<char>> n_data;
	if (data.find(pid_sid) == data.end()) {
		data[pid_sid] = n_data;
		data[pid_sid].second = elem;
		data[pid_sid].first = n_elem;
		//MLOG("inserted pid %d sid %d nelem %d bytes %d size %d\n", pid, sid, n_elem, len_bytes, elem.size());
	}
	else {
		data[pid_sid].second.insert(data[pid_sid].second.end(), elem.begin(), elem.end());
		data[pid_sid].first += n_elem;
	}

	return 0;
}

//-------------------------------------------------------------------------------------------------------------------
int InMemRepData::insertData(int pid, const char *sig, int *time_data, float *val_data, int n_time, int n_val)
{
	int sid = my_rep->sigs.sid(string(sig));
	if (sid < 0) {
		MERR("ERROR: InMemRepData: sig %s in not is signals file (did you read the file\?\?)\n", sig);
		return -1;
	}
	return insertData(pid, sid, time_data, val_data, n_time, n_val);

}

//-------------------------------------------------------------------------------------------------------------------
int InMemRepData::insertData(int pid, int sid, int *time_data, float *val_data, int n_time, int n_val)
{
	return InMemRepData::insertData_to_buffer(pid, sid, time_data, val_data, n_time, n_val, my_rep->sigs, data);
}

//-------------------------------------------------------------------------------------------------------------------
int InMemRepData::sort_pid_sid(int pid, int sid)
{
	pair<int, int> pid_sid(pid, sid);

	if (data.find(pid_sid) == data.end()) return 0; // nothing to do.

	if (data[pid_sid].first <= 1) return 0; // no need to sort a single variable

	int (*compare_func)(const void *, const void *);
	GenericSigVec gsv;
	switch (my_rep->sigs.Sid2Info[sid].type) {
	case T_Value:				compare_func = &MedSignalsCompareSig<SVal>;				break;
	case T_DateVal:				compare_func = &MedSignalsCompareSig<SDateVal>;			break;
	case T_TimeVal:				compare_func = &MedSignalsCompareSig<STimeVal>;			break;
	case T_DateRangeVal:		compare_func = &MedSignalsCompareSig<SDateRangeVal>;	break;
	case T_TimeStamp:			compare_func = &MedSignalsCompareSig<STimeStamp>;		break;
	case T_TimeRangeVal:		compare_func = &MedSignalsCompareSig<STimeRangeVal>;	break;
	case T_DateVal2:			compare_func = &MedSignalsCompareSig<SDateVal2>;		break;
	case T_TimeLongVal:			compare_func = &MedSignalsCompareSig<STimeLongVal>;		break;
	case T_DateShort2:			compare_func = &MedSignalsCompareSig<SDateShort2>;		break;
	case T_ValShort2:			compare_func = &MedSignalsCompareSig<SValShort2>;		break;
	case T_ValShort4:			compare_func = &MedSignalsCompareSig<SValShort4>;		break;
	case T_DateRangeVal2:		compare_func = &MedSignalsCompareSig<SDateRangeVal2>;	break;
	case T_DateFloat2:			compare_func = &MedSignalsCompareSig<SDateFloat2>;		break;
	case T_TimeShort4:			compare_func = &MedSignalsCompareSig<STimeShort4>;		break;
	case T_Generic:
		gsv.init(my_rep->sigs.Sid2Info[sid]);
		gsv.inplace_sort_data(&data[pid_sid].second[0], data[pid_sid].first);
		return 0;
	//case T_CompactDateVal:		break; // not fully supported yet
	default: MERR("ERROR:sort_pid_sid Unknown sig_type %d\n", my_rep->sigs.Sid2Info[sid].type);
		return -1;
	}

	//MLOG("pid %d sid %d len %d bytes %d first_val %f\n", pid, sid, data[pid_sid].first, my_rep->sigs.Sid2Info[sid].bytes_len, data[pid_sid].second[0]);
	qsort(&data[pid_sid].second[0], data[pid_sid].first, my_rep->sigs.Sid2Info[sid].bytes_len, compare_func);

	return 0;
}

//-------------------------------------------------------------------------------------------------------------------
int InMemRepData::sortData()
{
	for (auto &data_elem : data) {
		//MLOG("Now sorting pid %d sid %d (%s)\n", data_elem.first.first, data_elem.first.second, my_rep->sigs.Sid2Name[data_elem.first.second].c_str());
		if (sort_pid_sid(data_elem.first.first, data_elem.first.second) < 0) {
			MERR("FAILED:: InMemRepData::sortData() failed sorting pid %d sid %d\n", data_elem.first.first, data_elem.first.second);
			return -1;
		}
	}
	return 0;
}

//-------------------------------------------------------------------------------------------------------------------
void *InMemRepData::get_from_buffer(int pid, int sid, int &len, const map<pair<int, int>, pair<int, vector<char>>> &data) {
	pair<int, int> pid_sid(pid, sid);
	if (data.find(pid_sid) == data.end()) {
		len = 0;
		return NULL;
	}

	len = data.at(pid_sid).first;
	return (void *)&data.at(pid_sid).second[0];
}

void * InMemRepData::get(int pid, int sid, int &len)
{
	return InMemRepData::get_from_buffer(pid, sid, len, data);
}

void InMemRepData::erase_pid_data(int pid) {
	//Erase from data all (pid, sid):
	for ( auto &it : my_rep->sigs.Name2Sid) {
		pair<int, int> pid_sid(pid, it.second);
		if (data.find(pid_sid) != data.end()) {
			data.erase(pid_sid);
		}
	}

}