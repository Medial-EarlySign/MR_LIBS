//
// MedIndex.c
//
#define __INFRAMED_DLL
#define _CRT_SECURE_NO_WARNINGS
#include "InfraMed.h"
#include <fstream>
#include <cstring>
#include "Logger/Logger/Logger.h"
#include "Utils.h"


#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_INDEX
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

//===========================================================
// MedBufferedFile
//===========================================================

//-----------------------------------------------------------
int MedBufferedFile::open(const string &fname, const int bsize)
{

	name = fname;

	inf = (ifstream *)new ifstream;
	//MLOG("MedBufferedFile::open inf [%d] [%s]\n", inf, name.c_str());
	inf->open(name,ios::in | ios::binary);
	if (!(inf->is_open())) {
		MTHROW_AND_ERR("MedBufferedFile: open: Can't open file %s\n",name.c_str());
		return -1;
	}

	if (bsize > 0)
		buf_size = bsize;
	else
		buf_size = MedBufferedFile::default_buf_size;

	buf = new unsigned char[buf_size];
	//MLOG("MedBufferedFile::open buf [%d] [%s]\n", buf, name.c_str());
	if (buf == NULL) {
		inf->close();
		MTHROW_AND_ERR("MedBufferedFile: open: Can't allocate buffer of size %d for file %s\n",buf_size,name.c_str());
		return -1;
	}
	buf_len = 0;

	return 0;
}

//-----------------------------------------------------------
int MedBufferedFile::open(const string &fname)
{
	return(open(fname,MedBufferedFile::default_buf_size));
}

//-----------------------------------------------------------
void MedBufferedFile::close()
{	
	if (buf != NULL) {
		//MLOG("MedBufferedFile::close before buf [%d] delete [%s]\n", buf, name.c_str());
		delete[] buf;
		//MLOG("MedBufferedFile::close after buf [%d] delete [%s]\n", buf, name.c_str()); 
	}
	if (inf != NULL) {
		//MLOG("MedBufferedFile::close before inf [%d] inf->is_open() [%s]\n", inf, name.c_str());
		if (inf->is_open())
			inf->close();
		//MLOG("MedBufferedFile::close before inf [%d] delete [%s]\n", inf, name.c_str());
		delete inf;
		//MLOG("MedBufferedFile::close after inf [%d] delete [%s]\n", inf, name.c_str());
	}
	buf_len = 0;
}

//-----------------------------------------------------------
unsigned long long MedBufferedFile::read(unsigned char *outb, unsigned long long pos, unsigned long long len)
{
	unsigned long long rlen = 0;
	unsigned long long read_len = 0;
	// if pos is inside the buffer start copy data to output
	if (buf_len>0 && pos>=buf_start_in_file && pos<buf_start_in_file+buf_len) {

		rlen = min(len, (buf_start_in_file+buf_len)-pos);
		std::memcpy(outb,&buf[pos-buf_start_in_file],rlen);
		read_len += rlen;
		if (rlen == len)
			return read_len;
		pos = pos + rlen;
		len = len - rlen;
		outb += rlen;
	}

	// pos is out of buffer either initially or since we have leftovers
	while (len > 0) {
		if (pos < buf_start_in_file) inf->clear();
		inf->seekg(pos,ios::beg); 
		inf->read((char *)buf,buf_size);
		buf_start_in_file = pos;
		buf_len = (int)inf->gcount();
		rlen = min(len, (buf_start_in_file+buf_len)-pos);
		std::memcpy(outb,buf,rlen);
		pos = pos + rlen;
		len = len - rlen;
		outb += rlen;
		read_len += rlen;
	}

	return read_len;
}

//===========================================================
// MedIndex
//===========================================================

//-----------------------------------------------------------
int MedIndex::get_mode(const string &fname)
{
	ifstream inf;
	
	inf.open(fname,ios::in | ios::binary);

	if (!inf) {
		MERR("MedIndex: get_mode: Can't open file %s\n",fname.c_str());
		return -1;
	}

	unsigned long long magic;
	int mode;
	inf.read((char *)&magic,sizeof(unsigned long long));
	if (magic != MED_MAGIC_NUM) {
		MERR("MedIndex: get_mode: got wrong magic number %016x\n",magic);
		return -1;
	}

	inf.read((char *)&mode,sizeof(int));
	if (mode != 0) {
		MERR("MedIndex: get_mode: unsuported index mode %d\n",mode);
		return -1;
	}

	inf.close();

	return mode;
}

//-----------------------------------------------------------
int MedIndex::read_index(vector<string> &fnames)
{
	vector<int> dummy1, dummy2;
	// return (read_sub_index(fnames, vector<int>(), vector<int>()));
	 return (read_sub_index(fnames, dummy1, dummy2));
}

//-----------------------------------------------------------
int MedIndex::read_sub_index(vector<string> &fnames, const vector<int> &pids_to_include, const vector<int> &signals_to_include)
{
	int i;


	idx.clear();
	idx_pid.clear();
	idx_sid.clear();

	unsigned long long tot_len = 0;
	for (i=0; i<fnames.size(); i++)
		tot_len += get_file_size_IM(fnames[i]);
	// reserving the maximal number of records that idx will hold, to allow for a much faster load into it.
	tot_len /= (int)(sizeof(int)+sizeof(short)+sizeof(unsigned long long)+sizeof(int)); 
	int estimate = (int)pids_to_include.size() * (int)signals_to_include.size();
	if (estimate > 0 && tot_len > estimate) tot_len = estimate;
	MLOG_D("read_sub_index: reserving %d index records\n",tot_len);
	MedTimer t("Index preps");

	map<int, int> seen;
	vector<int> fnos;
	int len_s = (int)fnames.size();
	vector<int> f_factor(len_s,-1); // if file contains signals of the SAME type we can keep more records by multiplying with the record size.
	for (auto it=my_rep->sigs.Name2Sid.begin(); it!=my_rep->sigs.Name2Sid.end(); it++) {
		int sid = it->second;
		//MLOG("sid %s %d\n", it->first.c_str(), sid);
		int fno = my_rep->sigs.Sid2Info[sid].fno;
		int s_size = my_rep->sigs.Sid2Info[sid].bytes_len;
		if (fno < len_s && fno>=0) {
			if (f_factor[fno] < 0) f_factor[fno] = s_size;
			else if (f_factor[fno] != s_size) f_factor[fno] = 1;
		}
		//MLOG("fno %d/%d s_size %d f_factor %d\n", fno, f_factor.size(), s_size, f_factor[fno]);
	}


	int take_all = 0;
	for (i=0; i<signals_to_include.size(); i++) {
		int fno = my_rep->sigs.Sid2Info[signals_to_include[i]].fno;

		if (fno < 0) {
			// we have a signal for which we do not know the file number at this stage...
			// hence we are forced to go over all files.
			take_all = 1;
			break;
		}
		if (seen.find(fno) == seen.end()) {
			fnos.push_back(fno);
			seen[fno] = 1;
		}
	}

	if (take_all || signals_to_include.size()==0) {
		fnos.clear();
		f_factor.clear();
		for (i=0; i<fnames.size(); i++) {
			fnos.push_back(i);
			//f_factor[i] = 1;
		}
	}

	MLOG_D("MedIndex: read_sub_index: take_all %d reading %d/%d files.\n", take_all, (int)fnos.size(), (int)fnames.size());

	t.start();
	//for (i=0; i<fnames.size(); i++) {
	for (int j=0; j<fnos.size(); j++) {
		i = fnos[j];
		int mode;

		mode = get_mode(fnames[i]);
	
		if (mode != 0) {
			MERR("MedIndex: read_sub_index: wrong mode: only mode 0 supported at the moment\n");
			return -1;
		}

		switch (mode) {
			case 0:
				MLOG_D("MedIndex: read_sub_index: reading index file %s , factor %d, (include (%d pids, %d sids))\n",fnames[i].c_str(),f_factor[i],pids_to_include.size(),signals_to_include.size());
				if (read_index_mode0_new_direct(fnames[i], f_factor[i], pids_to_include, signals_to_include) < 0) //????????
				//if (read_index_mode0_new_direct(fnames[i], 1, pids_to_include, signals_to_include) < 0) //????????
						return -1;
		}
	}

	//if (mode == 0)
	//	for (unsigned int i=0; i<n_signals_in_index; i++)
	//		idx_i_add[i].resize(n_pids_in_index, 0);

	t.take_curr_time(); MLOG_D("Index read time %f sec\n",t.diff_sec()); t.start();
	n_pids = n_pids_in_index;
	n_signals = n_signals_in_index;

	t.take_curr_time(); MLOG_D("Index prep_idx time %f sec\n",t.diff_sec());

	return 0;
}

//-----------------------------------------------------------
int MedIndex::read_index_mode0(const string &fname, const vector<int> &pids_to_include, const vector<int> &signals_to_include)
{
	ifstream inf;
	char *buffer;

	unsigned long long buflen,bufp;
	buflen = get_file_size_IM(fname);

	buffer = new char[buflen];

	if (buffer == NULL) {
		MERR("MedIndex: read_index_mode0: Can't allocate buffer of size %lld for index\n",buflen);
		return -1;
	}

	inf.rdbuf()->pubsetbuf(0, 0);
	inf.open(fname,ios::in | ios::binary);
	if (!inf) {
		MERR("MedIndex: read_index_mode0: Can't open file %s\n",fname.c_str());
		delete [] buffer;
		return -1;
	}

	inf.read(buffer,buflen);
	bufp = 0;

	unsigned long long magic;
	int mode;
	//inf.read((char *)&magic,sizeof(unsigned long long));
	magic = ((unsigned long long *)(buffer+bufp))[0]; bufp+=sizeof(unsigned long long);
	if (magic != MED_MAGIC_NUM) {
		MERR("MedIndex: read_index_mode0: got wrong magic number %016x\n",magic);
		delete [] buffer;
		return -1;
	}

	//inf.read((char *)&mode,sizeof(int));
	mode = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
	if (mode != 0) {
		MERR("MedIndex: read_index_mode0: unsuported index mode %d\n",mode);
		delete [] buffer;
		return -1;
	}

	// prepare maps to take into account only "to include" elements
	int take_all_pids = 1;
	int take_all_sids = 1;
	map<int,bool> use_pid;
	map<int,bool> use_sid;

	if (pids_to_include.size() > 0) {
		take_all_pids = 0;
		n_pids = (int)pids_to_include.size();
		for (int i=0; i<pids_to_include.size(); i++)
			use_pid[pids_to_include[i]] = true;
	}

	if (signals_to_include.size() > 0) {
		take_all_sids = 0;
		n_signals = (int)signals_to_include.size();
		for (int i=0; i<signals_to_include.size(); i++)
			use_sid[signals_to_include[i]] = true;
	}

	// now go over rest of packets in index file and load them into internal arrays

//	while (!inf.eof()) {
	while (bufp<buflen) {
		//inf.read((char *)&magic,sizeof(unsigned long long));
		magic = ((unsigned long long *)(buffer+bufp))[0]; bufp+=sizeof(unsigned long long);
		if (magic != MED_MAGIC_NUM) {
			MERR("MedIndex: read_index_mode0: got wrong magic number (in patient packet) %016x\n",magic);
			delete [] buffer;
			return -1;
		}

		int pid = 0,n_sig = 0,sid = 0,len = 0;
		short fno;
		unsigned long long fpos;

		//inf.read((char *)&pid,sizeof(int));
		pid = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
		//inf.read((char *)&n_sig,sizeof(int));
		n_sig = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);

		for (int i=0; i<n_sig; i++) {
			//inf.read((char *)&sid, sizeof(int));
			sid = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
			//inf.read((char *)&fno, sizeof(short));
			fno = ((short *)(buffer+bufp))[0]; bufp+=sizeof(short);
			//inf.read((char *)&fpos, sizeof(unsigned long long));
			fpos = ((unsigned long long *)(buffer+bufp))[0]; bufp+=sizeof(unsigned long long);
			//inf.read((char *)&len, sizeof(int));
			len = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);

			if ((take_all_pids || use_pid.find(pid) != use_pid.end()) && 
				(take_all_sids || use_sid.find(sid) != use_sid.end())) 
			{
				IndexElem ie;
				ie.data = NULL;
				ie.file_num = fno;
				ie.len = len;
				ie.pos_in_file = fpos;
				idx.push_back(ie);
				idx_pid.push_back(pid);
				idx_sid.push_back(sid);
			}

		}
	}

	ifnames.push_back(fname);
	MLOG_D("MedIndex: read_index: read index file %s. Currently holding %d index records\n",fname.c_str(),idx.size());
	inf.close();
	delete [] buffer;
	return 0;
}
//-----------------------------------------------------------
// a more efficient (4x ...) index reading mechanism
int MedIndex::read_index_mode0_direct(const string &fname, const vector<int> &pids_to_include, const vector<int> &signals_to_include)
{
	ifstream inf;
	char *buffer;

	unsigned long long buflen,bufp;
	buflen = get_file_size_IM(fname);

	buffer = new char[buflen];

	if (buffer == NULL) {
		MERR("MedIndex: read_index_mode0: Can't allocate buffer of size %lld for index\n",buflen);
		return -1;
	}

	inf.open(fname,ios::in | ios::binary);
	if (!inf) {
		MERR("MedIndex: read_index_mode0: Can't open file %s\n",fname.c_str());
		delete [] buffer;
		return -1;
	}

	inf.read(buffer,buflen);
	bufp = 0;

	unsigned long long magic;
	int mode;
	//inf.read((char *)&magic,sizeof(unsigned long long));
	magic = ((unsigned long long *)(buffer+bufp))[0]; bufp+=sizeof(unsigned long long);
	if (magic != MED_MAGIC_NUM) {
		MERR("MedIndex: read_index_mode0: got wrong magic number %016x\n",magic);
		delete [] buffer;
		return -1;
	}

	//inf.read((char *)&mode,sizeof(int));
	mode = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
	if (mode != 0) {
		MERR("MedIndex: read_index_mode0: unsuported index mode %d\n",mode);
		delete [] buffer;
		return -1;
	}

	if (pid_idx.size() == 0) {
		min_pid = 0;	// TODO:: automatically detect range to minimize size of pid_idx.
		max_pid = MAX_PID_NUMBER;
		pid_idx.resize(max_pid-min_pid);
		sid_idx.resize(1000000);
		fill(pid_idx.begin(), pid_idx.end(), MAX_ID_NUM+1);
		fill(sid_idx.begin(), sid_idx.end(), MAX_ID_NUM+1);
		pids.clear();
		signals.clear();
		n_pids_in_index = 0;
		n_signals_in_index = 0;
		if (pids_to_include.size() > 0) {
			for (int i=0; i<pids_to_include.size(); i++)
				pid_idx[pids_to_include[i]] = MAX_ID_NUM;
		}

		if (signals_to_include.size() > 0) {
			for (int i=0; i<signals_to_include.size(); i++)
				sid_idx[signals_to_include[i]] = MAX_ID_NUM;
		}

	}
	// prepare maps to take into account only "to include" elements
	int take_all_pids = 1;
	int take_all_sids = 1;

	if (pids_to_include.size() > 0)
		take_all_pids = 0;

	if (signals_to_include.size() > 0)
		take_all_sids = 0;

	// now go over rest of packets in index file and load them into internal arrays
	while (bufp<buflen) {
		//inf.read((char *)&magic,sizeof(unsigned long long));
		magic = ((unsigned long long *)(buffer+bufp))[0]; bufp+=sizeof(unsigned long long);
		if (magic != MED_MAGIC_NUM) {
			MERR("MedIndex: read_index_mode0: got wrong magic number (in patient packet) %016x\n",magic);
			delete [] buffer;
			return -1;
		}

		int pid = 0,n_sig = 0,sid = 0,len = 0;
		short fno;
		unsigned long long fpos;

		pid = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
		n_sig = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);

		for (int i=0; i<n_sig; i++) {
			sid = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
			fno = ((short *)(buffer+bufp))[0]; bufp+=sizeof(short);
			fpos = ((unsigned long long *)(buffer+bufp))[0]; bufp+=sizeof(unsigned long long);
			len = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);

			if ((take_all_pids || pid_idx[pid] <= MAX_ID_NUM) && 
				(take_all_sids || sid_idx[sid] <= MAX_ID_NUM)) 
			{
				IndexElem ie;
				ie.data = NULL;
				ie.file_num = fno;
				ie.len = len;
				ie.pos_in_file = fpos;
				idx.push_back(ie);
				idx_pid.push_back(pid);
				idx_sid.push_back(sid);
				if (pid_idx[pid] >= MAX_ID_NUM) {
					pids.push_back(pid);
					pid_idx[pid] = n_pids_in_index++;
				}
				if (sid_idx[sid] >= MAX_ID_NUM) {
					signals.push_back(sid);
					sid_idx[sid] = n_signals_in_index++;
				}
			}

		}
	}
	
	MLOG_D("MedIndex: read_index: read index file %s. Currently holding %d index records\n",fname.c_str(),idx.size());
	inf.close();
	delete [] buffer;
	return 0;
}


//-----------------------------------------------------------
// a more efficient (4x ...) index reading mechanism, this version tries to save LOTS of memory usage
int MedIndex::read_index_mode0_new_direct(const string &fname, int f_factor, const vector<int> &pids_to_include, const vector<int> &signals_to_include)
{
	ifstream inf;
	char *buffer;

	unsigned long long buflen, bufp;
	buflen = get_file_size_IM(fname);

	buffer = new char[buflen];


	if (buffer == NULL) {
		MERR("MedIndex: read_index_mode0: Can't allocate buffer of size %lld for index\n", buflen);
		return -1;
	}

	inf.open(fname, ios::in | ios::binary);
	if (!inf) {
		MERR("MedIndex: read_index_mode0: Can't open file %s\n", fname.c_str());
		delete[] buffer;
		return -1;
	}

	inf.read(buffer, buflen);
	bufp = 0;

	unsigned long long magic;
	int mode;
	//inf.read((char *)&magic,sizeof(unsigned long long));
	magic = ((unsigned long long *)(buffer+bufp))[0]; bufp+=sizeof(unsigned long long);
	if (magic != MED_MAGIC_NUM) {
		MERR("MedIndex: read_index_mode0: got wrong magic number %016x in file %s\n", magic,fname.c_str());
		delete[] buffer;
		return -1;
	}

	//inf.read((char *)&mode,sizeof(int));
	mode = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
	if (mode != 0) {
		MERR("MedIndex: read_index_mode0: unsuported index mode %d\n", mode);
		delete[] buffer;
		return -1;
	}

	if (pid_idx.size() == 0) {
		min_pid = 0;	// TODO:: automatically detect range to minimize size of pid_idx.
		max_pid = MAX_PID_NUMBER;
		pid_idx.resize(max_pid);
		pid_seen.resize(max_pid, 0);
		sid_idx.resize(MAX_SID_NUMBER);
		idx_recs.clear();
		idx_recs.clear();
		idx_recs.resize(MAX_SIGNALS); 
		idx_recs_base.resize(MAX_SIGNALS);
		fill(pid_idx.begin(), pid_idx.end(), MAX_ID_NUM+1);
		fill(sid_idx.begin(), sid_idx.end(), MAX_ID_NUM+1);
		pids.clear();
		signals.clear();
		n_pids_in_index = 0;
		n_signals_in_index = 0;
		if (pids_to_include.size() > 0) {
			vector<int> sorted = pids_to_include;
			sort(sorted.begin(), sorted.end());
			for (int i=0; i<sorted.size(); i++) {
				pid_idx[sorted[i]] = i; //MAX_ID_NUM;
			}
			n_pids_in_index = (int)pids_to_include.size();
		}

		if (signals_to_include.size() > 0) {
			for (int i=0; i<signals_to_include.size(); i++)
				sid_idx[signals_to_include[i]] = MAX_ID_NUM;
		}

		idx_i_base.clear();
		idx_i_base.resize(MAX_SIGNALS);
		idx_i_add.clear();
		idx_i_add.resize(MAX_SIGNALS);
		i_sid_type_byte_len.resize(MAX_SIGNALS, 0);
		i_sid_factor.resize(MAX_SIGNALS, 0);

	}
	// prepare maps to take into account only "to include" elements
	int take_all_pids = 1;
	int take_all_sids = 1;

	if (pids_to_include.size() > 0)
		take_all_pids = 0;

	if (signals_to_include.size() > 0)
		take_all_sids = 0;


	// we are now holding the index for one or more signals in memory
	// in order to save allocation times (due to very slow reallocations) we do a first pass over the index
	// in the first pass we calculate the sizes for each signal
	vector<int> sig_size(MAX_SID_NUMBER, 0);
	vector<int> sig_cnt(MAX_SID_NUMBER, -1);
	unsigned long long bufp_orig = bufp;
	int min_pid_appearing = MAX_PID_NUMBER;
	int max_pid_appearing = 0;
	while (bufp<buflen) {
		bufp+=sizeof(unsigned long long); // skip magic

		int pid, n_sig, sid;
		pid = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
		if (pid < min_pid_appearing) min_pid_appearing = pid;
		if (pid > max_pid_appearing) max_pid_appearing = pid;

		n_sig = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);

		for (int i=0; i<n_sig; i++) {
			sid = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
			bufp+=sizeof(short) + sizeof(unsigned long long) + sizeof(int); // skip fno + fpos + len

			if ((take_all_pids || pid_idx[pid] <= MAX_ID_NUM) && (take_all_sids || sid_idx[sid] <= MAX_ID_NUM))
				sig_size[sid]++;
		}
	}

	if (max_pid_appearing < min_pid_appearing) {
		min_pid_appearing = 0;
		max_pid_appearing = MAX_PID_NUMBER;
	}
	if (min_pid_num < 0 && pids_to_include.size()==0) {
		min_pid_appearing = 1000000*(min_pid_appearing/1000000);
		max_pid_appearing = min(1000000*(max_pid_appearing/1000000)+1000000, MAX_PID_NUMBER);
		//MLOG("read_index_new_direct:: min_pid_appearing <= %d , max_pid_appearing >= %d\n", min_pid_appearing, max_pid_appearing);
		for (int i=min_pid_appearing; i<=max_pid_appearing; i++) {
			pid_idx[i] = i-min_pid_appearing;
		}
		min_pid_num = min_pid_appearing;
	}

	vector<int> shifting(4097, -1);
	for (int i=0; i<=12; i++) shifting[((size_t)1)<<i] = i;
	if (f_factor <= 0) f_factor = 1;
	int shift_f = shifting[f_factor];
	//MLOG("fname %s f_factor=%d shift_f=%d \n", fname.c_str(), f_factor, shift_f);
	//MLOG_D("Going over index buffer....\n");
	// now go over rest of packets in index file and load them into internal arrays
	bufp = bufp_orig;
	while (bufp<buflen) {
		//inf.read((char *)&magic,sizeof(unsigned long long));
		magic = ((unsigned long long *)(buffer+bufp))[0]; bufp+=sizeof(unsigned long long);

		int pid = 0, n_sig = 0, sid = 0, len = 0;
		int i_sid, i_pid, i_rec, i_base;
		short fno;
		unsigned long long fpos, dpos, maxd;
		maxd = 1;
		maxd = maxd << 32;
		pid = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
		n_sig = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);


		if (magic != MED_MAGIC_NUM) {
			MERR("MedIndex: read_index_mode0_new_direct: got wrong magic number (in patient packet) %016x file %s pid %d %0x bufp %ld\n",
				magic, fname.c_str(), pid, pid, bufp);
			delete[] buffer;
			return -1;
		}

		//MLOG_D("MedIndex: read_index_mode0_new_direct: pid %d n_sig %d bufp %ld\n",
		//	 pid, n_sig, bufp);

		for (int i=0; i<n_sig; i++) {
			sid = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);
			fno = ((short *)(buffer+bufp))[0]; bufp+=sizeof(short);
			fpos = ((unsigned long long *)(buffer+bufp))[0]; bufp+=sizeof(unsigned long long);
			len = ((int *)(buffer+bufp))[0]; bufp+=sizeof(int);

			//MLOG_D("MedIndex: read_index_mode0_new_direct %d %d: pid %d (%x) n_sig %d bufp %ld sid %d (%x) fno %d fpos %ld len %d\n",
			//	take_all_pids, take_all_sids, pid, pid_idx[pid], n_sig, bufp, sid, sid_idx[sid], fno, fpos, len);


			if ((take_all_pids || pid_idx[pid] <= MAX_ID_NUM) &&
				(take_all_sids || sid_idx[sid] <= MAX_ID_NUM))
			{
				
				//if (pid_idx[pid] >= MAX_ID_NUM) {
				//	pid_idx[pid] = n_pids_in_index++;
				//}
				if (pid_seen[pid] == 0) {
					pid_seen[pid] = 1;
					pids.push_back(pid);
					n_pids_in_index = (unsigned int)pids.size();
				}
				if (sid_idx[sid] >= MAX_ID_NUM) {
					signals.push_back(sid);
					sid_idx[sid] = n_signals_in_index;
					i_sid_type_byte_len[n_signals_in_index] = my_rep->sigs.Sid2Info[sid].bytes_len;
					i_sid_factor[n_signals_in_index] = f_factor;
					n_signals_in_index++;
				}

				i_pid = pid_idx[pid];
				i_base = i_pid >> 7;
				i_sid = sid_idx[sid];
				//i_rec = (int)idx_recs[i_sid].size();
				if (sig_cnt[sid] < 0) {
					sig_cnt[sid] = 0;
					idx_recs[i_sid].resize(sig_size[sid]);
				}
				i_rec = sig_cnt[sid];

				if (idx_recs_base[i_sid].file_num < 0) {
					if (rep_mode < 2)
						idx_recs_base[i_sid].file_num = fno; // fno read from file
					else
						idx_recs_base[i_sid].file_num = my_rep->sigs.Sid2Info[sid].fno; // fno is virtual, we make sure to point our signal file
					idx_recs_base[i_sid].data = NULL;
					idx_recs_base[i_sid].pos_in_file = fpos;
				}

				//MLOG_D("i_pid %d i_base %d i_sid %d i_rec %d i_len %d\n", i_pid, i_base, i_sid, i_rec, i_sid_type_byte_len[i_sid]);
				//CompactIndexElem cie;
				CompactIndexElem *cie = &idx_recs[i_sid][sig_cnt[sid]];
				sig_cnt[sid]++;
				int i_bytes = i_sid_type_byte_len[i_sid];
				if (shifting[i_bytes] >= 0)
					(*cie).len = len>>shifting[i_bytes];
				else
					(*cie).len = len/i_bytes;

				dpos = fpos - idx_recs_base[i_sid].pos_in_file;

#if 0
				if ((dpos % f_factor) != 0) {
					MERR("ERROR: dpos is %ld, f_factor is %d ... not divisible...", dpos, f_factor);
					exit(-1);
				} 
#endif

				if (shift_f >= 0)
					dpos = dpos >> shift_f;
				else
					dpos = dpos/f_factor;


				if (dpos > maxd - 1) {
					MERR("ERROR: dpos larger than 4GB !!!! : dpos %ld fpos %ld pos %ld sid %d i_sid %d pid %d i_pid %d fno %d\n",
						dpos, fpos, idx_recs_base[i_sid].pos_in_file, sid, i_sid, pid, i_pid, fno);
					exit(-1);
				}
				(*cie).pos_add = (unsigned int)(dpos); // This forces same signal to be in span of 4GB inside a data file.... better always have less than 4GB data files...
				(*cie).data_ptr_add = 0;

				//idx_recs[i_sid].push_back(cie);
				//idx_recs[i_sid][sig_cnt[sid]++] = cie;

				if (i_pid >= idx_i_add[i_sid].size()) {
					//int curr_size_add = (int)idx_i_add[i_sid].size();
					//int curr_size_base = (int)idx_i_base[i_sid].size();

					int new_add_len = (i_pid & 0xffffff00) + 256;
					int new_base_len = (i_base+2);

					idx_i_add[i_sid].resize(new_add_len, 0);
					idx_i_base[i_sid].resize(new_base_len, 0xffffffff);
				}

				if (idx_i_base[i_sid][i_base] != 0xffffffff) {
					idx_i_add[i_sid][i_pid] = 1 + i_rec - idx_i_base[i_sid][i_base];
				}
				else {
					idx_i_base[i_sid][i_base] = i_rec;
					idx_i_add[i_sid][i_pid] = 1;
				}

				//if (i_pid == 0) {
				//	MLOG("_new_direct: i_pid %d pid %d i_sid %d sid %d i_rec %d cie.pos_add %d recs_base %ld i_base %ld i_add %d fno%d\n",
				//		i_pid, pid, i_sid, sid, i_rec, cie.pos_add, idx_recs_base[i_sid].pos_in_file, idx_i_base[i_sid][i_base], idx_i_add[i_sid][i_pid],fno);
				//}
			}

		}
	}

	for (unsigned int i=0; i<n_signals_in_index; i++) {
		//MLOG("idx_i_add size i=%d size=%d\n", i, idx_i_add[i].size());
		idx_i_add[i].resize(max_pid_appearing, 0);
	}

	MLOG_D("MedIndex: read_index: read index file %s. Currently holding %d index records\n", fname.c_str(), idx.size());
	inf.close();
	delete[] buffer;
	return 0;
}


//-----------------------------------------------------------
int MedIndex::prep_idx_i()
{
	map<int,bool> pid_appeared;
	map<int,bool> sid_appeared;

	n_pids = 0;
	n_signals = 0;

	pids.clear();
	signals.clear();

	// first we calculate n_signals and n_pids in order to be able to allocate idx_i
	for (unsigned int i=0; i<idx.size(); i++) {
		//MLOG("Index: idx %d : pid %d sid %d fno %d len %d pos %d\n",i,idx_pid[i],idx_sid[i],idx[i].file_num,idx[i].len,idx[i].pos_in_file);
		if (pid_appeared.find(idx_pid[i]) == pid_appeared.end()) {
			pid_appeared[idx_pid[i]] = true;
			pid2idx[idx_pid[i]] = n_pids++;
			pids.push_back(idx_pid[i]);
		}
		if (sid_appeared.find(idx_sid[i]) == sid_appeared.end()) {
			sid_appeared[idx_sid[i]] = true;
			sid2idx[idx_sid[i]] = n_signals++;
			signals.push_back(idx_sid[i]);
		}
	}

	pid_appeared.clear();
	sid_appeared.clear();

	idx_i.resize(n_signals);
	for (int i=0; i<n_signals; i++) {
		idx_i[i].resize(n_pids);
		fill(idx_i[i].begin(), idx_i[i].end(), MAX_ID_NUM);
	}
	unsigned int i_p, i_s;
	for (unsigned int i=0; i<idx.size(); i++) {
		i_p = pid2idx[idx_pid[i]];
		i_s = sid2idx[idx_sid[i]];
		idx_i[i_s][i_p] = i;
	}

	idx_pid.clear();
	idx_sid.clear();
	return 0;
}


//-----------------------------------------------------------
int MedIndex::prep_idx_i_direct()
{
	n_pids = n_pids_in_index;
	n_signals = n_signals_in_index;

	//	idx_i.resize((n_signals)*(n_pids));
	//	fill(idx_i.begin(), idx_i.end(), -1);
	idx_i.resize(n_signals);
	for (int i=0; i<n_signals; i++) {
		idx_i[i].resize(n_pids);
		fill(idx_i[i].begin(), idx_i[i].end(), MAX_ID_NUM);
	}

	unsigned int i_p, i_s;
	for (unsigned int i=0; i<idx.size(); i++) {
		i_p = pid_idx[idx_pid[i]];
		i_s = sid_idx[idx_sid[i]];
		//		idx_i[i_p*n_signals + i_s] = i;
		idx_i[i_s][i_p] = i;
	}
	idx_pid.clear();
	idx_sid.clear();
	return 0;
}

//-----------------------------------------------------------
unsigned long long MedIndex::get_index_max_data_size()
{
	unsigned long long size = 0;

	for (int i=0; i<idx_recs.size(); i++)
		for (int j=0; j<idx_recs[i].size(); j++)
			size += static_cast<size_t>(idx_recs[i][j].len) * static_cast<size_t>(i_sid_type_byte_len[i]);
//	for (int i=0; i<idx.size(); i++)
//		size += idx[i].len;
	return size;
}

//-----------------------------------------------------------
void MedIndex::set_mem_ptrs_off()
{
	for (int i=0; i<idx.size(); i++)
		idx[i].data = NULL;
}


//-----------------------------------------------------------
/*int MedIndex::read_all_data(unsigned char *&work_area, unsigned long long &wlen, vector<string> &data_fnames, vector<int> pids_to_take, vector<int> sids_to_take)
{
	return(read_all_data(work_area,wlen, data_fnames, vector<int>(), vector<int>()));
}
*/
//-----------------------------------------------------------
int MedIndex::read_all_data(unsigned char *&work_area, unsigned long long &wlen, vector<string> &data_fnames)
{
	if (work_area != NULL || wlen != 0) {
		MERR("MedIndex: read_all_data: error: got a non empty work_area.\n");
		return -1;
	}

	set_mem_ptrs_off();

	// open files
	int n_files = (int)data_fnames.size();
	vector<MedBufferedFile> inf(n_files);
	for (int i=0; i<n_files; i++) {
		if (inf[i].open(data_fnames[i].c_str()) < 0) {
			for (int j=0; j<i; j++)
				inf[j].close();
			MERR("MedIndex: read_all_data: error: can't open file %s\n",data_fnames[i].c_str());
			return -1;
		}
	}

	// allocating needed size
	wlen = get_index_max_data_size();
	MLOG_D("MedRepository: read_all_data: opened %d data files. wlen = %lld = %5.2fGB\n",n_files,wlen,(double)wlen/(double)(1<<30));

	if (wlen == 0)
		return 0;
	work_area = new unsigned char[wlen];
	MLOG_D("MedRepository: read_all_data: work_area allocated %x\n", work_area);
	if (work_area == NULL) {
		MERR("MedIndex: read_all_data: error: can't allocate %d bytes for work_area\n",wlen);
		for (int i=0; i<n_files; i++) inf[i].close();
		wlen = 0;
		return -1;
	}

	// go over idx and do the reads
	unsigned long long curr_w = 0;
	MLOG_D("MedRepository: read_all_data: actually reading data\n");
	int n_rec = 0;
	int len;
	unsigned long long size_idx_recs = 0;
	unsigned long long size_idx_recs_base = idx_recs_base.size() * sizeof(IndexElem);
	unsigned long long size_idx_i_base = 0;
	unsigned long long size_idx_i_add = 0;
	for (int i=0; i<idx_recs.size(); i++) {
		
		size_idx_recs += idx_recs[i].size() * sizeof(CompactIndexElem);
		size_idx_i_base += idx_i_base[i].size() * sizeof(unsigned int);
		size_idx_i_add += idx_i_add[i].size() * sizeof(char);
		
		if (idx_recs[i].size() > 0)
		MLOG_D("read_data: i=%d/%d(%d) size %d fname = %s fno = %d \n", i, idx_recs.size(), n_files, idx_recs[i].size(), inf[idx_recs_base[i].file_num].name.c_str(), idx_recs_base[i].file_num);
		
		for (int j=0; j<idx_recs[i].size(); j++) {


			if (idx_recs_base[i].file_num >= n_files) {
				MERR("MedIndex: read_all_data: error: index contains unspecified data file num: %d\n", idx_recs_base[i].file_num);
				for (int k=0; k<n_files; k++) inf[k].close();
				delete[] work_area;
				set_mem_ptrs_off();
				work_area = NULL;
				wlen = 0;
				return -1;
			}

			len = idx_recs[i][j].len*i_sid_type_byte_len[i];
			unsigned long long p_add = static_cast<size_t>(idx_recs[i][j].pos_add) * static_cast<size_t>(i_sid_factor[i]);
			inf[idx_recs_base[i].file_num].read(&work_area[curr_w], idx_recs_base[i].pos_in_file + p_add, len);
			if (idx_recs_base[i].data == NULL) {
				idx_recs_base[i].data = (void *)(&work_area[curr_w]);
			}
			unsigned long long d_add = (unsigned long long)(&work_area[curr_w]) - (unsigned long long)idx_recs_base[i].data;
			d_add = d_add/i_sid_factor[i];
			//idx_recs[i][j].data_ptr_add = (unsigned int)((unsigned long long)(&work_area[curr_w]) - (unsigned long long)idx_recs_base[i].data);
			idx_recs[i][j].data_ptr_add = (unsigned int)d_add;

			curr_w += len;
			n_rec++;
#if 0
			if ((n_rec)%100 == 0 || j>1279300) {
				MLOG_D("MedRepository: read_all_data: read %d index records i %d j %d\n", n_rec, i, j);
			}
#endif
		}
	}

	unsigned long long size_idx_i = size_idx_i_add + size_idx_i_base;
	unsigned long long size_idx = size_idx_recs + size_idx_recs_base;

	double data_gb = (double)wlen/(double)(1<<30);
	double idx_i_gb = (double)size_idx_i/(double)(1<<30);
	double idx_gb = (double)size_idx/(double)(1<<30);
	double i_gb = idx_i_gb + idx_gb;
	double tot_gb = data_gb + i_gb;

	MLOG("Medrepository: read_all_data(): data %5.2fGB : index %5.2fGB (recs %5.2fGB idx_i %5.2fGB) : total %5.2fGB\n",
		data_gb, i_gb, idx_gb, idx_i_gb, tot_gb);

	for (int i=0; i<n_files; i++) inf[i].close();
	inf.clear();

	return 0;
}

//-----------------------------------------------------------
// next is faster, as it reads whole files directly to memory, then sets the index pointers
// drawback is that it reads all data to memory rather than just the subset we need.
int MedIndex::read_full_data(unsigned char *&work_area, unsigned long long &wlen, vector<string> &data_fnames)
{
	if (work_area != NULL || wlen != 0) {
		MERR("MedIndex: read_full_data: error: got a non empty work_area.\n");
		return -1;
	}

	set_mem_ptrs_off();

	// get total size
	wlen = 0;
	vector<unsigned long long> flen(data_fnames.size());
	for (int i=0; i<data_fnames.size(); i++) {
		flen[i] = get_file_size_IM(data_fnames[i]);
		wlen += flen[i];
	}
	// allocating needed size
	MLOG("MedRepository: read_full_data:  %d data files. wlen = %lld = %5.2fGB\n",data_fnames.size(),wlen,(double)wlen/(double)(1<<30));
	if (wlen == 0)
		return 0;
	

	
	work_area = new unsigned char[wlen];
	MLOG_D("MedRepository: read_all_data: work_area allocated %x\n", work_area);

	if (work_area == NULL) {
		MERR("MedRepository: read_full_data: error: can't allocate %lld bytes for work_area\n",wlen);
		wlen = 0;
		return -1;
	}
	// read data from files
	unsigned long long pos = 0;
	vector<unsigned long long> fpos(data_fnames.size());
	for (int i=0; i<data_fnames.size(); i++) {
		MLOG_D("Before reading %s\n", data_fnames[i].c_str());
		FILE *in_f;
		in_f = fopen(data_fnames[i].c_str(), "rb");
		setvbuf(in_f, (char *)NULL, _IONBF, 0);
		//ifstream inf;
		//inf.rdbuf()->pubsetbuf(0, 0);
		//inf.open(data_fnames[i],ios::in|ios::binary);
		//if (!inf) {
		if (!in_f) {
			MERR("MedRepository: read_full_data: can't open input data file %s\n",data_fnames[i].c_str());
			delete [] work_area;
			wlen = 0;
			work_area = NULL;
			return -1;
		}
		unsigned long long nread = (unsigned long long)fread((char *)&work_area[pos], 1, flen[i], in_f);
		//inf.read((char *)&work_area[pos],flen[i]);
		//if (inf.fail()) {
		if (nread != flen[i]) {
			MERR("MedRepository: read_full_data: failed reading %ld bytes from %s (nread=%ld i=%d pos=%d)\n",flen[i],data_fnames[i].c_str(),nread,i,pos);
			delete [] work_area;
			wlen = 0;
			work_area = NULL;
			return -1;
		}
		fpos[i] = pos;
		pos += flen[i];
		fclose(in_f);

		MLOG_D("Read file %s , nread %ld fpos %ld pos %ld\n", data_fnames[i].c_str(), nread, fpos[i], pos);
		//inf.close();
	}

	// setting the index pointers
	for (int i=0; i<idx_recs.size(); i++) {
		if (idx_recs[i].size() > 0)
			idx_recs_base[i].data = work_area + fpos[idx_recs_base[i].file_num] + idx_recs_base[i].pos_in_file;
		for (int j=0; j<idx_recs[i].size(); j++) {
			idx_recs[i][j].data_ptr_add = idx_recs[i][j].pos_add; //(unsigned int)(fpos[idx_recs[i][j].file_num] + idx_recs[i][j].pos_in_file);
		}
	}


	MLOG_D("Set all recs\n");
	// printing memory used
	unsigned long long idx_size, idx_i_size, data_size, index_size, tot_size;
	int n_recs = 0;
	for (int i=0; i<idx_recs.size(); i++)
		n_recs += (int)idx_recs[i].size();
	idx_size = (unsigned long long)n_recs*(int)sizeof(CompactIndexElem);

	idx_i_size = 0;
	for (int i=0; i<idx_i_base.size(); i++)
		idx_i_size += (int)idx_i_base[i].size()*sizeof(unsigned int);
	for (int i=0; i<idx_i_add.size(); i++)
		idx_i_size += (int)idx_i_add[i].size()*sizeof(unsigned char);
	//idx_i_size = idx_i.size()*idx_i[0].size()*sizeof(unsigned int);

	data_size = wlen; //sizeof(work_area);
	index_size = idx_size + idx_i_size;
	tot_size = data_size + index_size;

	double idx_s, idx_i_s, data_s, index_s, tot_s;

	//idx_s = (double)idx_size / (double)(1<<30);
	idx_s = (double)(idx_size>>20)/1024.0;
	idx_i_s = (double)idx_i_size / (double)(1<<30);
	data_s = (double)data_size / (double)(1<<30);
	index_s = (double)index_size / (double)(1<<30);
	tot_s = (double)tot_size / (double)(1<<30);

	MLOG("MedRepository: read_full_data: sizes : npids %d nsigs %d : nrecs %d (%5.3f) (%d) : tot %5.2fGB data %5.2fGB , index %5.2fGB (idx %5.2fGB, idx_i %5.2fGB)\n",
			n_pids, n_signals, n_recs, 
			(double)n_recs/((double)n_pids*n_signals), sizeof(unsigned short), //(int)sizeof(CompactIndexElem),
			tot_s,data_s,index_s,idx_s,idx_i_s);

	return 0;
}

//---------------------------------------------------------------------------------------------
int MedIndex::read_index_table_and_data(int sid, string &idx_fname, string &data_fname, const vector<int> &pids_to_include) {
	unsigned long long data_size = 0;
	return read_index_table_and_data(sid, idx_fname, data_fname, pids_to_include, NULL, data_size);
}

//---------------------------------------------------------------------------------------------
int MedIndex::read_index_table_and_data(int sid, string &idx_fname, string &data_fname,
										const vector<int> &pids_to_include, unsigned char *w_area, unsigned long long &data_size)
{
	string prefix = "MedIndex::read_index_table_and_data() : sid : " + to_string(sid) + " :: ";
	//MLOG("%s start reading index and data\n", prefix.c_str());
#pragma omp critical
	{
		if (index_table.size() == 0)
			index_table.resize(MAX_SID_NUMBER);
	}

	if (sid > MAX_SID_NUMBER) {
		MERR("%s ERROR: sid too big (max sid num is %d)\n", prefix.c_str(), MAX_SID_NUMBER);
		return -1;
	}

	if (w_area != NULL) {
		MERR("%s Preallocated mode not supported yet....\n", prefix.c_str());
		return -1;
	}

	index_table[sid].sid = sid; // setting the sid number inside its container

	if (index_table[sid].read_index_and_data(idx_fname, data_fname, pids_to_include) < 0) {
		MERR("%s Error reading index and data\n", prefix.c_str());
		return -1;
	}

#pragma omp critical
	{
		if (sid_in.find(sid) == sid_in.end()) {
			//MLOG("Inserting sid %d to sid_in\n", sid);
			sid_in[sid] = 1;
			signals.push_back(sid);
		}
		data_size = index_table[sid].w_size;
	}

	return 0;
}

//-------------------------------------------------------------------------------
int MedIndex::update_pids()
{
	pids.clear();
	vector<unsigned int> pid_list;
	vector<unsigned int> pid_in;
	
	for (auto it : sid_in) {
		if (it.second) {
			int sid = it.first;
			index_table[sid].sv.get_all_keys(pid_list);
//			MLOG("found sid %d : %d pids : last %d\n", sid, pid_list.size(), pid_list.back());
			if (pid_list.size() > 0) {
				pid_in.resize(max((unsigned int)pid_in.size(),pid_list.back()+1), 0);
				for (auto i_pid : pid_list) {
					pid_in[i_pid] = 1;
				}
			}
		}
	}

//	MLOG("pid_list size %d pid_in size %d\n", pid_list.size(), pid_in.size());
	for (unsigned int i=0; i<pid_in.size(); i++)
		if (pid_in[i]) {
			pids.push_back(i);
		}
//	MLOG("pids size %d\n", pids.size());

	return 0;

}