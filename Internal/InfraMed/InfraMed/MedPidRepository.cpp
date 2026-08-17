#include "MedPidRepository.h"
#include "Utils.h"
#include <Logger/Logger/Logger.h>
#include <string.h>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_REP
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

#define MAX_HEADER_SIZE	1000000
//#define MAX_PID_DATA_SIZE	10000000
#define PID_REC_MAGIC_NUM	0x0248ace13579bdf

#define PID_REC_MUTEX_POOL_SIZE	2048
#define PID_REC_MUTEX_MASK (2047)
mutex dynamic_pid_rec_mutex_pool[PID_REC_MUTEX_POOL_SIZE];

//------------------------------------------------------------------------------------------------------------
// creating the "by pid" index and data files for a range of given pids with at most "jump" pids in each file
int MedPidRepository::create(string &rep_fname, int from_pid, int to_pid, int jump)
{
	pids_idx.clear();
	pids_idx.set_min(from_pid);

	if (jump <= 0)
		jump = (to_pid - from_pid) / 4 + 1;

	int curr_chunk = 0;

	vector<unsigned char> pid_header(MAX_HEADER_SIZE);
	vector<unsigned char> pid_data(MAX_PID_DATA_SIZE);
	string path_and_pref;

	// looping over original repository several times, in order to allow not loading all of it to memory.
	for (int pid_s = from_pid; pid_s <= to_pid; pid_s += jump) {

		// init our current list of pids to read
		vector<int> pids_to_load(jump);
		for (int i = 0; i < jump; i++) pids_to_load[i] = pid_s + i;

		// read this portion of the repository
		MedRepository rep;
		vector<int> sids;
		MLOG("Before read_all()\n");
		if (rep.read_all(rep_fname, pids_to_load, sids) < 0) {
			MERR("MedPidRepository: ERROR reading %s for pid range %d - %d\n", rep_fname.c_str(), pid_s, pid_s + jump);
			return -1;
		}


		MLOG("read rep chunk %d with %d/%d pids\n", curr_chunk, rep.pids.size(), pids_to_load.size());
		// open current pids data file
		path_and_pref = rep.path + "/" + rep.rep_files_prefix;
		string curr_out_fname = path_and_pref + "__pids__" + to_string(curr_chunk) + ".pid_data";
		ofstream of;
		of.open(curr_out_fname, ios::out | ios::binary);
		if (!of) {
			MERR("MedPidRepository: ERROR chunk %d : can't open file %s for write\n", curr_chunk, curr_out_fname.c_str());
			return -1;
		}
		unsigned long long fpos = 0;

		MLOG("MedPidRepository: working on chunk %d for file %s\n", curr_chunk, curr_out_fname.c_str());
		// go over pids, prepare record, write data to file, and add a record to index
		for (int i = 0; i < rep.pids.size(); i++) {
			int pid = rep.pids[i];
			int header_len = 0;
			int data_len = 0;
			int nsigs = 0;
			//MLOG("Packing pid %d\n", pid);
			if (i % 10000 == 0) MLOG(".");
			//MLOG("%d\n", i);

			// add magic number to header
			*((unsigned long long *)&pid_header[header_len]) = (unsigned long long)PID_REC_MAGIC_NUM; header_len += sizeof(long long);
			*((int *)&pid_header[header_len]) = pid; header_len += sizeof(int);
			int *n_sigs = (int *)&pid_header[header_len]; header_len += sizeof(int);
			int *sig_list = (int *)&pid_header[header_len];
			for (int j = 0; j < rep.sigs.signals_ids.size(); j++) {
				int sid = rep.sigs.signals_ids[j];
				int len;
				unsigned char *sig_data = (unsigned char *)rep.get(pid, sid, len);
				if (len > 0) {
					int byte_len = len * rep.sigs.Sid2Info[sid].bytes_len;

					// data copying
					int pos = data_len;
					if (data_len + byte_len >= pid_data.size()) {
						MLOG("Resizing pid_data : pid %d sid %d len %d byte_len %d data_len %d max size %d\n", pid, sid, len, byte_len, data_len, pid_data.size());
						pid_data.resize(data_len + byte_len + pid_data.size() / 4);
						//exit(-1);
					}

					memcpy(&pid_data[data_len], sig_data, byte_len); data_len += byte_len;

					// header 
					// adding sid, pos (from start of data), len (in units of type)
					*((int *)&pid_header[header_len]) = sid; header_len += sizeof(int);
					*((int *)&pid_header[header_len]) = pos; header_len += sizeof(int);
					*((int *)&pid_header[header_len]) = len; header_len += sizeof(int);
					nsigs++;

				}
			}

			// plugin nsigs to header , now that we know it.
			*n_sigs = nsigs;
			//MLOG("%d pid %d nsigs %d header_len %d data_len %d\n", i,pid,nsigs,header_len,data_len);

			// add header len to all sig records, making it relative position from start
			for (int j = 0; j < nsigs; j++)
				sig_list[3 * j + 1] += header_len;

			// writer header
			if (header_len > 0)
				of.write((char *)&pid_header[0], header_len);

			// write data
			if (data_len > 0)
				of.write((char *)&pid_data[0], data_len);

			// adding a record to pids_idx
			PidIdxRec pir;
			pir.byte_len = header_len + data_len;
			pir.fnum = curr_chunk;
			pir.pos = fpos;
			fpos += pir.byte_len;
			pir.idx = -1;
			pids_idx.insert(pid, pir);

		}
		MLOG("\n");
		// close current file, advance to next
		of.close();
		MLOG("Wrote data chunk %d\n", curr_chunk);
		curr_chunk++;
	}

	// serializing and writing pids_idx
	vector<unsigned char> serialized;
	unsigned long long slen = pids_idx.get_size();
	serialized.resize(slen);
	pids_idx.serialize(&serialized[0]);
	// write it to file
	string curr_out_fname = path_and_pref + "__pids__all.pid_idx";
	ofstream of;
	of.open(curr_out_fname, ios::out | ios::binary);
	if (!of) {
		MERR("MedPidRepository: ERROR chunk %d : can't open file %s for write\n", curr_chunk, curr_out_fname.c_str());
		return -1;
	}
	of.write((char *)&serialized[0], slen);

	return 0;
}


//------------------------------------------------------------------------------------------------------------
int MedPidRepository::init(const string &conf_fname)
{
	// regular init, without reading data

	// read config
	if (read_config(conf_fname) < 0) {
		MERR("MedPidRepository: init: error: read_config %s failed\n", conf_fname.c_str());
		return -1;
	}

	MLOG_D("MedPidRepository: read config file %s\n", conf_fname.c_str());

	// read dictionaries
	if (dict.read_state == 0) {
		if (dict.read(dictionary_fnames) < 0) {
			MERR("MedPidRepository: init: error: read dictionary failed\n");
			return -1;
		}
	}
	dict.read_state = 2;
	MLOG_D("MedPidRepository: read %d dictionary files\n", dictionary_fnames.size());

	// read signals
	if (signal_fnames.size() == 0) {
		MERR("MedPidRepository: init: error: no signals def file given, this is mandatory\n");
		return -1;
	}
	if (sigs.read(signal_fnames) < 0) {
		MERR("MedPidRepository: init: error: read signal files failed\n");
		return -1;
	}

	// reading pid_idx file and openning all pid_data files

	if (!in_mem_mode_active()) {
		string idx_fname = path + "/" + rep_files_prefix + "__pids__all.pid_idx";
		unsigned char *serialized;
		unsigned long long size;
		if (read_bin_file_IM_parallel(idx_fname, serialized, size) < 0) {
			MERR("MedPidRepository::init() ERROR failed reading file %s\n", idx_fname.c_str());
			return -1;
		}

		pids_idx.deserialize(serialized);
		delete[] serialized;
		MLOG_D("Read and deserialized %s\n", idx_fname.c_str());

		for (int i = 0; ; i++) {

			string data_fname = path + "/" + rep_files_prefix + "__pids__" + to_string(i) + ".pid_data";
			if (file_exists_IM(data_fname)) {
				MedBufferedFile mbf;
				in_files.push_back(mbf);
				if (in_files.back().open(data_fname) < 0) {
					MERR("MedPidRepository::init() ERROR failed openning file %s\n", data_fname.c_str());
					return -1;
				}
				MLOG_D("Read %s\n", data_fname.c_str());
			}
			else
				break;

		}
	}
	// calling the base class init to allow for load() operations to work well
	if (MedRepository::init(conf_fname) < 0) {
		MERR("MedPidRepository: init: failed init of MedRepository side\n");
		return -1;
	}

	return 0;

}

MedPidRepository::~MedPidRepository() {
	for (size_t i = 0; i < in_files.size(); ++i)
		if (in_files[i].inf != NULL && in_files[i].inf->is_open())
			in_files[i].close();
}
//------------------------------------------------------------------------------------------------------------
// get data size of a pid (0 => pid not in data)
unsigned int MedPidRepository::get_data_size(int pid)
{
	PidIdxRec *pir = pids_idx.get((unsigned int)pid);
	if (pir == NULL)
		return 0;
	return pir->byte_len;

}

// simpler API, will use the data and data_len inside the prec instead
// it is recommended for use with pre allocation of enough space in prec.data when going to reuse the same prec for reads.
int MedPidRepository::get_pid_rec(int pid, PidRec &prec)
{
	prec.data_len = prec.data_size;
	return get_pid_rec(pid, prec.data, prec.data_len, prec);
}

mutex pid_files_lock;

//------------------------------------------------------------------------------------------------------------
// if data is NULL it will be allocated and data_size will be the allocated size
// if data is not NULL data_size should contain the max size allowed on input, and on output contains the actual size used
// general error : -1
// error due to insufficient data_size (non NULL data) : -2
int MedPidRepository::get_pid_rec(int pid, unsigned char *&_data, unsigned int &data_size, PidRec &prec)
{
	prec.pid = pid;
	prec.my_rep = this;
	prec.my_base_rep = (MedRepository *)prec.my_rep;
	prec.sv.clear();

	unsigned int size = get_data_size(pid);

	if (size == 0) {
		MERR("get_pid_rec(): pid %d not in data\n", pid);
		return -1;
	}


	if (size > 0) {
		if (_data == NULL || (data_size < size && prec.allow_realloc)) {
			prec.data = new unsigned char[size];
			prec.is_allocated = 1;
			prec.data_size = size;
			prec.data_len = size;
			data_size = size;
		}
		else if (data_size < size) {
			prec.prealloc(size + 8);
			//MERR("get_pid_rec(): ERROR: Not enough space to read pid %d : need %d and got %d\n", pid, size, data_size);
			//return -2;
		}

		PidIdxRec *pir = pids_idx.get((unsigned int)pid);

		{
			//lock_guard<mutex> guard(pid_files_lock); // locking here in order to allow for different threads read using the same in_file
			if (in_files[pir->fnum].read(_data, pir->pos, (unsigned long long)size) < size) {
				MERR("get_pid_rec(): ERROR: couldn't read %d bytes for pid %d\n", pir->byte_len, pid);
				return -1;
			}
		}

		prec.data = _data;
		prec.data_len = size;
		prec.init_sv();
	}

	//	MLOG("### Read pid %d : data_size %d : entries in sv: %d\n", prec.pid, size, prec.sv.data.size());
	return 0;

}

//------------------------------------------------------------------------------------------------------------
// after reading the data to *data this operation is needed to build the sparse vec from (serial) sid to PosLen
int PidRec::init_sv()
{
	if (data == NULL || data_len <= 0)
		return 0; // nothing to do - empty pid

	if (my_rep == NULL) {
		MERR("PidRec::init_sv(): ERROR my_rep not initialized.... can't load pid\n");
		return -1;
	}

	unsigned char *buf = data;
	int len = 0;

	// verify magic number
	unsigned long long magic = *((unsigned long long *)&buf[len]); len += sizeof(long long);
	if (magic != (unsigned long long)PID_REC_MAGIC_NUM) {
		MERR("PidRec::init_sv() ERROR: wrong magic number for pid %d : %llx vs %llx\n", pid, magic, (unsigned long long)PID_REC_MAGIC_NUM);
		return -1;
	}

	// verify pid
	int my_pid = *((int *)&buf[len]); len += sizeof(int);
	if (my_pid != pid) {
		MERR("PidRec::init_sv() ERROR pid not matching: in rec %d, in data %d\n", pid, my_pid);
		return -1;
	}

	// read sigs and initiate sparse vec
	int n_sigs = *((int *)&buf[len]); len += sizeof(int);
	//	MLOG("### pid %d n_sigs %d\n", pid, n_sigs);
	for (int i = 0; i < n_sigs; i++) {
		PosLen pl;
		int sid = *((int *)&buf[len]); len += sizeof(int);
		int sid_serial = my_base_rep->sigs.sid2serial[sid];
		pl.pos = *((int *)&buf[len]); len += sizeof(int);
		pl.len = *((int *)&buf[len]); len += sizeof(int);
		//MLOG("pid %d : sid %d : pos %d : len %d\n", pid, sid, pl.pos, pl.len);
		//MLOG("### pid %d : sid %d : pos %d : len %d : sid_serial %d\n", pid, sid, pl.pos, pl.len, sid_serial);

		if (sid_serial < 0) {
			MERR("PidRec::init_sv() ERROR: sid %d with no serial.\n", sid);
			return -1;
		}

		sv.insert((unsigned int)sid_serial, pl);
	}

	return 0;

}

//------------------------------------------------------------------------------------------------------------
void *PidRec::get(const string &sig_name, int &len)
{
	len = 0;
	int sid = my_base_rep->sigs.sid(sig_name);
	if (sid < 0)
		return NULL;
	return(get(sid, len));
}

//------------------------------------------------------------------------------------------------------------
void *PidRec::get(int sid, int &len)
{
	len = 0;
	int sid_serial = my_base_rep->sigs.sid2serial[sid];

	PosLen *pl = sv.get((unsigned int)sid_serial);

	if (pl == NULL)
		return NULL;
	len = pl->len;
	return ((void *)&data[pl->pos]);
}

//------------------------------------------------------------------------------------------------------------
void PidRec::prealloc(unsigned int len)
{
	free();
	data_buffer.resize(len);
	data = &data_buffer[0];
	data_len = 0;
	data_size = len;
	is_allocated = 0;
}

//------------------------------------------------------------------------------------------------------------
int PidRec::realloc(unsigned int len)
{
	if (is_allocated) return -1;
	data_buffer.resize(len);
	data = &data_buffer[0];
	data_size = len;
	return 0;
}

//------------------------------------------------------------------------------------------------------------
void PidRec::free()
{
	if (is_allocated && data != NULL)
		delete[] data;
	data_buffer.clear();
	if (data != NULL)
		data = NULL;
	data_len = 0;
	data_size = 0;
	is_allocated = 0;
}


//..................................................................................................................
int PidRec::init_from_rep(MedRepository *rep, int _pid, vector<int> &sids_to_use)
{
	// clear what was before and init basics
	sv.clear();
	my_rep = NULL;
	my_base_rep = rep;
	pid = _pid;

	data_len = 0;
	if (data_size < 8) { realloc(1024); }	// making sure we have some work space
	((int *)(&data[data_len]))[0] = _pid;		// it's here for debugging , and in order to make sure sv's don't start at 0 addresses
	data_len += sizeof(int);

	// copy data for sids to our data_buffer, and init the relevant records in sv
	vector<int> sids = sids_to_use;
	sort(sids.begin(), sids.end()); // sorting sids in preparation for inserts into sv.

	for (auto sid : sids) {
		int len;
		unsigned char *sig_data = (unsigned char *)my_base_rep->get(pid, sid, len);
		//unsigned char *sig_data = (unsigned char *)rep->get(pid, sid, len);
		if (sig_data != NULL) {
			int sid_serial = my_base_rep->sigs.sid2serial[sid];
			int sid_byte_len = my_base_rep->sigs.Sid2Info[sid].bytes_len;
			int slen = len * sid_byte_len;
			if (data_len + slen >= data_size) { realloc(2 * (data_len + slen)); }
			memcpy(&data[data_len], sig_data, slen);
			PosLen pl;
			pl.pos = data_len;
			pl.len = len;
			sv.insert(sid_serial, pl);
			data_len += slen;
		}
	}

	return 0;

}


#if 1
//==================================================================================================================
// PidDynamicRec
//==================================================================================================================

//..................................................................................................................
// a version is always a positive (>0) number. 0 is kept as the version number of the original data.
// this method should always be called AFTER the original version had been read.
int PidDynamicRec::set_n_versions(int n_ver)
{
	//	lock_guard<mutex> guard(dynamic_pid_rec_mutex_pool[pid & PID_REC_MUTEX_MASK]);
	if (n_versions > 0)
		clear_vers();

	if (n_ver < 1) {
		MERR("PidDynamicRec::set_n_versions :: ERROR n_versions must be at least 1, remember that version 0 is the original version\n");
		return -1;
	}

	n_versions = n_ver;

	// keys to the new index will be (serial sid)*(n_ver)+ver, originally we want them all to point to the original
	vector<unsigned int> orig_keys;
	sv.get_all_keys(orig_keys);
	sv_vers.init();

	int do_split = 0;
	if (n_versions > 1) do_split = 1;

	if (orig_keys.size() > 0) {

		sv_vers.set_min(orig_keys[0]);

		for (auto key : orig_keys) {
			PosLen *pl = sv.get(key);
			(*pl).do_split = do_split;
			for (int j = 0; j < n_versions; j++)
				sv_vers.insert(key*n_versions + j, *pl); // at start all versions point to the original one
		}
	}

	curr_len = data_len;

	return 0;
}


//..................................................................................................................
// This code assumes time_points are sorted in time !
// It also assumes that the versions are to be set according to the 0 time channel, 
// and that the time_points are given in a unit that is the same for all the needed sids.
int PidDynamicRec::set_n_versions(vector<int> &time_points)
{
	int n_ver = (int)time_points.size();
	//	lock_guard<mutex> guard(dynamic_pid_rec_mutex_pool[pid & PID_REC_MUTEX_MASK]);
	if (n_versions > 0)
		clear_vers();

	if (n_ver < 1) {
		MERR("PidDynamicRec::set_n_versions :: ERROR n_versions must be at least 1, remember that version 0 is the original version\n");
		return -1;
	}

	n_versions = n_ver;

	// keys to the new index will be (serial sid)*(n_ver)+ver, originally we want them all to point to the original
	vector<unsigned int> orig_keys;
	sv.get_all_keys(orig_keys);
	sv_vers.init();

	int do_split = 0;
	if (n_versions > 1) do_split = 1;

	if (orig_keys.size() > 0) {

		sv_vers.set_min(orig_keys[0]);

		for (auto key : orig_keys) {
			PosLen *pl = sv.get(key);
			PosLen my_pl = *pl;
			// key is serial_sid, we need the actual sid now
			int sid = my_base_rep->sigs.signals_ids[key];

			// universal get for sid, in order to be able to get times
			PidRec::uget(sid);

			int tlen = 0;
			for (int j = 0; j < n_versions; j++) {
				if (usv.n_time_channels() > 0) {
					while (tlen < usv.len && usv.Time(tlen) <= time_points[j]) tlen++;
					//len = tlen;
				}
				my_pl.len = tlen;
				my_pl.do_split = do_split;
				sv_vers.insert(key*n_versions + j, my_pl); // at start all versions point to the original one
			}
		}
	}

	curr_len = data_len;

	return 0;
}


//..................................................................................................................
void *PidDynamicRec::get(const string &sig_name, int version, int &len)
{
	len = 0;
	int sid = my_base_rep->sigs.sid(sig_name);
	if (sid < 0)
		return NULL;
	return(get(sid, version, len));
}

//..................................................................................................................
void *PidDynamicRec::get(int sid, int version, int &len)
{
	len = 0;
	PosLen *pl = get_poslen(sid, version);
	if (pl == NULL)
		return NULL;
	len = pl->len;
	return ((void *)&data[pl->pos]);
}

//..................................................................................................................
// deletes all versions and remains just with the original one.
void PidDynamicRec::clear_vers()
{
	n_versions = 0;
	sv_vers.clear();
	curr_len = data_len;
}

//..................................................................................................................
int PidDynamicRec::set_version_data(int sid, int version, void *datap, int len)
{

	lock_guard<mutex> guard(dynamic_pid_rec_mutex_pool[pid & PID_REC_MUTEX_MASK]);
	PosLen *pl = get_poslen(sid, version);

	if (pl == NULL)
		return -1;

	int size = my_base_rep->sigs.Sid2Info[sid].bytes_len * len;

	if (((unsigned int)pl->pos < data_len) || (len > pl->len) || (pl->do_split)) {
		// need to create a new place for this version
		//MLOG("In here : pos %d len %d curr_len %d data_len %d len %d size %d data_size %d\n", pl->pos, pl->len, curr_len, data_len, len, size, data_size);
		if (curr_len + size > data_size)
			resize_data(2 * (data_size + size));
		PosLen new_pl;
		new_pl.pos = curr_len;
		new_pl.len = len;
		new_pl.do_split = 0; // since it is now single
		curr_len += size;
		//MLOG("In here : NEW pos %d len %d curr_len %d data_len %d len %d size %d version %d\n", new_pl.pos, new_pl.len, curr_len, data_len, len, size, version);
		memcpy((char *)&data[new_pl.pos], (char *)datap, size);
		set_poslen(sid, version, new_pl);
	}
	else {
		// current place is enough for it, and also after the original
		memcpy((char *)&data[pl->pos], (char *)datap, size);
		PosLen new_pl = *pl;
		new_pl.len = len;
		set_poslen(sid, version, new_pl);
	}
	return 0;
}


//..................................................................................................................
int PidDynamicRec::set_version_universal_data(int sid, int version, int *_times, float *_vals, int len)
{
	UniversalSigVec usv;

	SignalInfo &info = this->my_base_rep->sigs.Sid2Info[sid];

	usv.init(info);

	vector<char> packed_data(static_cast<size_t>(len) * static_cast<size_t>(info.bytes_len));

	usv.data = &packed_data[0];
	usv.len = len;

	float *_curr_vals = _vals;
	int *_curr_times = _times;
	int inc_time = info.n_time_channels;
	int inc_vals = info.n_val_channels;

	for (int i = 0; i < len; i++) {

		usv.Set(i, _curr_times, _curr_vals, usv.data);

		_curr_times += inc_time;
		_curr_vals += inc_vals;

	}

	return set_version_data(sid, version, usv.data, usv.len);
}


//..................................................................................................................
int PidDynamicRec::set_version_off_orig(int sid, int version)
{

	lock_guard<mutex> guard(dynamic_pid_rec_mutex_pool[pid & PID_REC_MUTEX_MASK]);
	PosLen *pl = get_poslen(sid, version);


	if (pl == NULL)
		return -1;

	if ((unsigned int)pl->pos < data_len && pl->len > 0) {
		// This means our current version is in the original section (below data_len)
		// Hence we will make a copy of it in the versions working area

		int len = pl->len;
		int size = my_base_rep->sigs.Sid2Info[sid].bytes_len * len;

		if (curr_len + size > data_size)
			resize_data(2 * (data_size + size));

		PosLen new_pl;
		new_pl.pos = curr_len;
		new_pl.len = len;
		new_pl.do_split = 0; // since we allocated a new space for it
		curr_len += size;

		void *datap = (void *)&data[pl->pos];
		memcpy((char *)&data[new_pl.pos], (char *)datap, size);
		set_poslen(sid, version, new_pl);
	}

	return 0;
}

//..................................................................................................................
// will point version v_dst to the data of version v_src
int PidDynamicRec::point_version_to(int sid, int v_src, int v_dst)
{
	lock_guard<mutex> guard(dynamic_pid_rec_mutex_pool[pid & PID_REC_MUTEX_MASK]);
	PosLen *pl_src = get_poslen(sid, v_src);

	if (pl_src == NULL)
		return -1;
	if (v_dst >= n_versions || v_dst < 0)
		return -1;

	if (pl_src->do_split == 0) {
		pl_src->do_split = 1;
		set_poslen(sid, v_src, *pl_src); // to make sure it is do_split=1 now
	}

	set_poslen(sid, v_dst, *pl_src); // will always be with do_split=1
	return 0;
}

//..................................................................................................................
// removing element idx from version
int PidDynamicRec::remove(int sid, int version, int idx)
{
	return remove(sid, version, idx, version);
}

//..................................................................................................................
// removing element idx from version v_in and putting it in v_out
int PidDynamicRec::remove(int sid, int v_in, int idx, int v_out)
{
	lock_guard<mutex> guard(dynamic_pid_rec_mutex_pool[pid & PID_REC_MUTEX_MASK]);

	PosLen *pl_in = get_poslen(sid, v_in);
	PosLen *pl_out = get_poslen(sid, v_out);

	if (pl_in == NULL || pl_out == NULL)
		return -1;

	if (idx < 0 || idx >= pl_in->len || pl_in->len <= 0)
		return -1;

	int sid_byte_len = my_base_rep->sigs.Sid2Info[sid].bytes_len;
	int size1 = sid_byte_len * idx;
	int size2 = sid_byte_len * (pl_in->len - 1 - idx);
	int size = size1 + size2;

	PosLen new_pl;
	if (((unsigned int)pl_out->pos < data_len) || (pl_out->len < pl_in->len - 1) || (pl_out->do_split)) {
		// need to create a new place for this version
		if (curr_len + size > data_size)
			resize_data(2 * (data_size + size));
		new_pl.pos = curr_len;
		new_pl.do_split = 0;
		curr_len += size;

		memcpy(&data[new_pl.pos], &data[pl_in->pos], size1);
		memcpy(&data[new_pl.pos + size1], &data[pl_in->pos + size1 + sid_byte_len], size2);
	}
	else {
		// current place is enough for it, and also positioned after the original
		if (v_in != v_out) {
			memcpy(&data[pl_out->pos], &data[pl_in->pos], size1);
			memcpy(&data[pl_out->pos + size1], &data[pl_in->pos + size1 + sid_byte_len], size2);
		}
		else {
			unsigned char *d = &data[pl_out->pos + size1];
			for (int i = 0; i < size2; i++)
				d[i] = d[i + sid_byte_len];
		}
		new_pl = *pl_out;

	}

	new_pl.len = pl_in->len - 1;
	set_poslen(sid, v_out, new_pl);

	return 0;
}

//..................................................................................................................
// changing element idx in version to hold *new_elem
int PidDynamicRec::change(int sid, int version, int idx, void *new_elem)
{
	return change(sid, version, idx, new_elem, version);
}

//..................................................................................................................
// changing element idx in v_in to *new_elem, and putting it all in v_out
int PidDynamicRec::change(int sid, int v_in, int idx, void *new_elem, int v_out)
{
	lock_guard<mutex> guard(dynamic_pid_rec_mutex_pool[pid & PID_REC_MUTEX_MASK]);

	PosLen *pl_in = get_poslen(sid, v_in);
	PosLen *pl_out = get_poslen(sid, v_out);

	if (pl_in == NULL || pl_out == NULL)
		return -1;

	if (idx < 0 || idx >= pl_in->len || pl_in->len <= 0)
		return -1;

	int sid_byte_len = my_base_rep->sigs.Sid2Info[sid].bytes_len;
	int pos_to_change = sid_byte_len * idx;
	int size = sid_byte_len * pl_in->len;

	PosLen new_pl;
	if (((unsigned int)pl_out->pos < data_len) || (pl_out->len < pl_in->len - 1) || (pl_out->do_split)) {
		// need to create a new place for this version
		if (curr_len + size > data_size)
			resize_data(2 * (data_size + size));
		new_pl.pos = curr_len;
		new_pl.do_split = 0;
		curr_len += size;

		memcpy(&data[new_pl.pos], &data[pl_in->pos], size);
		memcpy((char *)&data[new_pl.pos + pos_to_change], (char *)new_elem, sid_byte_len);
	}
	else {
		// current place is enough for it, and also positioned after the original
		if (v_in != v_out) {
			memcpy(&data[pl_out->pos], &data[pl_in->pos], size);
		}
		memcpy((char *)&data[pl_out->pos + pos_to_change], (char *)new_elem, sid_byte_len);
		new_pl = *pl_out;
	}

	new_pl.len = pl_in->len;
	set_poslen(sid, v_out, new_pl);

	return 0;

}

//..................................................................................................................
int PidDynamicRec::update(int sid, int v_in, vector<pair<int, void *>>& changes, vector<int>& removes) {
	for (unsigned int iChange = 0; iChange < changes.size(); iChange++) {
		//if (change(sid, v_in, changes[iChange].first, &(changes[iChange].second)) < 0)
		if (change(sid, v_in, changes[iChange].first, changes[iChange].second) < 0)
			return -1;
	}

	for (int iRemove = 0; iRemove < removes.size(); iRemove++) {
		if (remove(sid, v_in, removes[iRemove] - iRemove) < 0)
			return -1;
	}

	return 0;
}

//..................................................................................................................
int PidDynamicRec::update(int sid, int v_in, int val_channel, vector<pair<int, float>>& changes, vector<int>& removes) {
	// first we make sure we get our copy if we have changes
	if (changes.size() > 0 || removes.size() > 0) {

		if (set_version_off_orig(sid, v_in) < 0)
			return -1;

		if (changes.size() > 0) {
			UniversalSigVec usv;
			uget(sid, v_in, usv);

			for (auto &change : changes)
				usv.SetVal_ch_vec(change.first, val_channel, change.second, usv.data);

		}

		for (int iRemove = 0; iRemove < removes.size(); iRemove++) {
			if (remove(sid, v_in, removes[iRemove] - iRemove) < 0)
				return -1;
		}

	}

	return 0;
}

//..................................................................................................................
int PidDynamicRec::update(int sid, int v_in, vector<pair<int, vector<float>>>& changes, vector<int>& removes) {
	// first we make sure we get our copy if we have changes
	if (changes.size() > 0 || removes.size() > 0) {

		if (set_version_off_orig(sid, v_in) < 0)
			return -1;

		if (changes.size() > 0) {
			UniversalSigVec usv;
			uget(sid, v_in, usv);

			for (auto &change : changes) {
				for (unsigned int iChannel = 0; iChannel < change.second.size(); iChannel++)
					usv.SetVal_ch_vec(change.first, iChannel, change.second[iChannel], usv.data);
			}

		}

		for (int iRemove = 0; iRemove < removes.size(); iRemove++) {
			if (remove(sid, v_in, removes[iRemove] - iRemove) < 0)
				return -1;
		}

	}

	return 0;
}

//..................................................................................................................
int PidDynamicRec::print_ver(int sid, int ver)
{

	int len;
	void *data = get(sid, ver, len);
	MLOG("VER %d :: ", ver);
	my_base_rep->print_vec_dict(data, len, pid, sid);
	//MLOG("Record size: data_size %d data_len %d curr_len %d\n", data_size, data_len, curr_len);
	return 0;
}

//..................................................................................................................
int PidDynamicRec::print_all_vers(int sid)
{
	MLOG("ORIG  :: ");
	int len;
	void *data = PidRec::get(sid, len);
	my_base_rep->print_vec_dict(data, len, pid, sid);

	for (int ver = 0; ver < n_versions; ver++)
		print_ver(sid, ver);

	//MLOG("Record size: data_size %d data_len %d curr_len %d\n", data_size, data_len, curr_len);
	return 0;
}


//..................................................................................................................
int PidDynamicRec::print_all()
{
	for (auto sid : my_base_rep->sigs.signals_ids) {
		int len;
		get(sid, 0, len);
		if (len > 0)
			print_ver(sid, 0);
	}

	return 0;
}


//..................................................................................................................
int PidDynamicRec::print_sigs(const vector<string> &sigs)
{
	for (auto sig : sigs) {
		int sid = my_base_rep->sigs.sid(sig);
		print_ver(sid, 0);
	}

	return 0;
}
//..................................................................................................................
int PidDynamicRec::init_from_rep(MedRepository *rep, int _pid, vector<int> &sids_to_use, int _n_versions)
{
	// clear what was before and init basics
	sv.clear();
	sv_vers.clear();
	//	sv_vers.init();
	my_rep = NULL;
	my_base_rep = rep;
	pid = _pid;
	//	n_versions = _n_versions;

	data_len = 0;
	if (data_size < 8) { realloc(1024); }	// making sure we have some work space
	((int *)(&data[data_len]))[0] = _pid;		// it's here for debugging , and in order to make sure sv's don't start at 0 addresses
	data_len += sizeof(int);

	// copy data for sids to our data_buffer, and init the relevant records in sv
	vector<int> sids = sids_to_use;
	//	sort(sids_to_use.begin(), sids_to_use.end()); // sorting sids in preparation for inserts into sv.
	sort(sids.begin(), sids.end()); // sorting sids in preparation for inserts into sv.
	for (auto sid : sids) {
		int len;
		unsigned char *sig_data = (unsigned char *)my_base_rep->get(pid, sid, len);
		//unsigned char *sig_data = (unsigned char *)rep->get(pid, sid, len);
		if (sig_data != NULL) {
			int sid_serial = my_base_rep->sigs.sid2serial[sid];
			int sid_byte_len = my_base_rep->sigs.Sid2Info[sid].bytes_len;
			int slen = len * sid_byte_len;
			if (data_len + slen >= data_size) { realloc(2 * (data_len + slen)); }
			memcpy(&data[data_len], sig_data, slen);
			PosLen pl;
			pl.pos = data_len;
			pl.len = len;
			pl.do_split = 0;
			sv.insert(sid_serial, pl);
			data_len += slen;
		}
		else {
			int sid_serial = my_base_rep->sigs.sid2serial[sid];
			PosLen pl;
			pl.pos = 0;
			pl.len = 0;
			pl.do_split = 0;
			sv.insert(sid_serial, pl);
		}
	}
	//curr_len = data_len;
	//return 0;

	// now setting the version number
	if (_n_versions >= 0)
		return (set_n_versions(_n_versions));

	return 0;
}

//..................................................................................................................
int PidDynamicRec::init_from_rep(MedRepository *rep, int _pid, vector<int> &sids_to_use, vector<int> &time_points)
{
	init_from_rep(rep, _pid, sids_to_use, -1);

	// now setting the version number
	return (set_n_versions(time_points));

}

//..................................................................................................................
// VersionsIterator
//..................................................................................................................
int differentVersionsIterator::init() {

	// Last Version
	iVersion = my_rec->get_n_versions() - 1;
	jVersion = iVersion;
	return next();

}

//..................................................................................................................
int differentVersionsIterator::next() {

	// Point data
	for (int pVersion = iVersion - 1; pVersion > jVersion; pVersion--) {
		for (int signalId : signalIds)
			my_rec->point_version_to(signalId, iVersion, pVersion);
	}

	// Iterate
	iVersion = jVersion;

	// Next Version
	jVersion = iVersion - 1;
	while (jVersion >= 0 && my_rec->versions_are_the_same(signalIds, iVersion, jVersion))
		jVersion--;

	return iVersion;
};


#endif