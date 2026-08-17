//
// templated code for MedMat class, included after class definition
//

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#define MED_FLT_EPSILON     1.192092896e-07F

//...........................................................................................
template <class T> template <class S> void MedMat<T>::load(S *x, int n_rows, int n_cols)
{
	nrows = n_rows;
	ncols = n_cols;
	m.resize(nrows*ncols);

	for (size_t i=0; i<nrows; i++)
		for (size_t j=0; j<ncols; j++)
			set(i,j) = (T)x[i*ncols + j];
	transposed_flag = 0;
}

//...........................................................................................
template <class T> template <class S> void MedMat<T>::load_transposed(S *x, int n_rows, int n_cols)
{
	nrows = n_cols;
	ncols = n_rows;
	m.resize(nrows*ncols);

	for (size_t i=0; i<n_rows; i++)
		for (size_t j=0; j<n_cols; j++)
			set(j,i) = (T)x[i*n_cols + j];
	transposed_flag = 1;
}

//...........................................................................................
template <class T> template <class S> void MedMat<T>::load(const vector<S> &x, int n_cols)
{
	ncols = n_cols;
	if (ncols == 0)
		return;
	nrows = x.size()/ncols;
	m.resize(nrows*ncols);

	for (size_t i=0; i<nrows; i++)
		for (size_t j=0; j<ncols; j++)
			set(i,j) = (T)x[i*ncols + j];
}

//...........................................................................................
// also copies metadata
template <class T> template <class S> void MedMat<T>::load(MedMat<S> &x)
{
	ncols = x.ncols;
	nrows = x.nrows;
	m.resize(nrows*ncols);

	for (size_t i=0; i<nrows; i++)
		for (size_t j=0; j<ncols; j++)
			set(i,j) = (T)x(i,j);

	signals.clear();
	signals = x.signals;
	recordsMetadata.clear();
	recordsMetadata = x.recordsMetadata;

}

//...........................................................................................
template <class T> void MedMat<T>::transpose()
{
	vector<T> m_orig = m;
	size_t ncols_orig = ncols;

	swap(ncols,nrows);

	for (size_t i=0; i<nrows; i++)
		for (size_t j=0; j<ncols; j++)
			set(i,j) = m_orig[j*ncols_orig+i];
	transposed_flag = 1 - transposed_flag;
}

//...........................................................................................
// empty list means - take them  all
// taking rows/cols in the order they were given
// duplicated numbers will be duplicated in rows/cols.
// also sublists metadata
template <class T> void MedMat<T>::get_sub_mat(vector<int> &rows_to_take, vector<int> &cols_to_take)
{
	size_t new_n_rows = (rows_to_take.size() == 0 ? nrows : (int) rows_to_take.size());
	size_t new_n_cols = (cols_to_take.size() == 0 ? ncols : (int) cols_to_take.size());

	vector<T> m_orig = m; // copying 
	vector<RecordData> r_orig = recordsMetadata;
	vector<string> c_orig = signals;
	recordsMetadata.clear(); signals.clear();

	m.resize(new_n_rows*new_n_cols);

	for (size_t i=0; i<new_n_rows; i++) {
		size_t r = (rows_to_take.size() == 0 ? i : rows_to_take[i]);
		if (r_orig.size() > 0)
			recordsMetadata.push_back(r_orig[r]);
		for (size_t j=0; j<new_n_cols; j++) {
			size_t c = (cols_to_take.size() == 0 ? j : cols_to_take[j]);
			if ((i == 0) && (c_orig.size() > 0))
				signals.push_back(c_orig[c]);

			m[i*new_n_cols + j] = m_orig[r*ncols+c];
		}
	}

	ncols = new_n_cols;
	nrows = new_n_rows;
}

//...........................................................................................
template <class T> void MedMat<T>::get_sub_mat_by_flags(vector<int> &rows_to_take_flag, vector<int> &cols_to_take_flag)
{
	vector<int> rows_to_take;
	vector<int> cols_to_take;

	flags_to_indexes(rows_to_take_flag, rows_to_take);
	flags_to_indexes(cols_to_take_flag, cols_to_take);

	get_sub_mat(rows_to_take, cols_to_take);
}

//...........................................................................................
template <class T> template <class S> void MedMat<T>::add_rows(MedMat<S> &m_add)
{
	if (recordsMetadata.size() > 0 || m_add.recordsMetadata.size() > 0)
		throw runtime_error("concating matrices with metadata is not supported yet");
	if (ncols != m_add.ncols)
		throw runtime_error("can not concat matrices with different number of cols");

	add_rows(&m_add.m[0], m_add.nrows);
}

//...........................................................................................
template <class T> template <class S> void MedMat<T>::add_rows(S *m_add, int nrows_to_add)
{
	if (nrows_to_add <= 0)
		return;
	
	m.resize((nrows+(size_t)nrows_to_add)*ncols);
	for (size_t j=0; j<nrows_to_add*ncols; j++)
		m[ncols*nrows+j] = (T)m_add[j];
	nrows += nrows_to_add;
}

//...........................................................................................
template <class T> template <class S> void MedMat<T>::add_rows(vector<S> &m_add)
{
	if (ncols == 0 || (int)(m_add.size() % ncols) != 0)
		return;

	int nrows_to_add =(int) (m_add.size()/ncols);

	add_rows(&m_add[0], nrows_to_add);

}

//...........................................................................................
template <class T> template <class S> void MedMat<T>::add_cols(MedMat<S> &m_add)
{
	if (signals.size() > 0 || m_add.signals.size() > 0)
		throw runtime_error("concating matrices with metadata is not supported yet");
	if (m_add.nrows != nrows)
		throw runtime_error("can not concat matrices with different number of rows");
	add_cols(&m_add.m[0], m_add.ncols);
}

//...........................................................................................
// packed as nrows x ncols_to_add 
template <class T> template <class S> void MedMat<T>::add_cols(S *m_add, int ncols_to_add)
{
	if (ncols_to_add == 0)
		return;

	vector<T> m_orig = m;

	size_t new_ncols = ncols+ncols_to_add;

	m.resize(nrows*new_ncols);
	for (size_t i=0; i<nrows; i++) {
		for (size_t j=0; j<ncols; j++)
			m[i*new_ncols+j] = (T)m_orig[i*ncols+j];
		for (size_t j=0; j<ncols_to_add; j++)
			m[i*new_ncols+ncols+j] = (T)m_add[i*(size_t)ncols_to_add + j];
	}

	ncols = new_ncols;
}

//...........................................................................................
template <class T> template <class S> void MedMat<T>::add_cols(vector<S> &m_add)
{
	if (nrows == 0 || (int)(m_add.size() % nrows) != 0)
		return;

	int ncols_to_add = (int)(m_add.size()/nrows);
	add_cols(&m_add[0], ncols_to_add);
}

//...........................................................................................
template <class T> void MedMat<T>::reorder_by_row(vector<int> &row_order)
{
	get_sub_mat(row_order, vector<int>());
}

//...........................................................................................
template <class T> void MedMat<T>::reorder_by_col(vector<int> &col_order)
{
	get_sub_mat(vector<int>(), col_order);
}

//...........................................................................................
template <class T> void MedMat<T>::get_row(int i_row, vector<T> &rowv) const
{
	rowv.resize(ncols);
	if (ncols > 0 && nrows > 0)
		memcpy(&(rowv[0]),&(m[(size_t)i_row*ncols]),ncols*sizeof(T));
}

//...........................................................................................
template <class T> void MedMat<T>::get_col(int i_col, vector<T> &colv) const
{
	colv.resize(nrows);
	if (ncols > 0 && nrows > 0) {
		for (size_t i=0; i<nrows; i++)
			colv[i] = m[i*ncols + (size_t)i_col];
	}

}

/*
#define MEDMAT_MAGIC_NUMBER 0x0011223344556677
//...........................................................................................
template <class T> size_t MedMat<T>::get_size()
{
	size_t size = 0;
	
	// 2 magic nums
	size += 2*sizeof(unsigned long long);

	// sizeof(T)
	size += sizeof(int);

	// matrix nrows, ncols, normalized_flag, transposed_flag, missing value
	size += 4*sizeof(int);
	size += sizeof(T);

	// matrix
	size += sizeof(T)*nrows*ncols;

	// avg
	size += sizeof(int);
	size += sizeof(T)*avg.size();

	// std
	size += sizeof(int);
	size += sizeof(T)*std.size();

	// recordsMetadata
	size += sizeof(int);
	size += sizeof(RecordData) * recordsMetadata.size();

	// signals
	size += sizeof(int);
	for (int i=0; i<(int)signals.size(); i++)
		size += signals[i].length();

	return size;
}

//...........................................................................................
template <class T> size_t MedMat<T>::serialize(unsigned char *buf)
{
	// start with 2 magic numbers (first for identification, second for future use/versions)
	size_t size = 0;

	*((unsigned long long *)&buf[size]) = (unsigned long long)MEDMAT_MAGIC_NUMBER; size += sizeof(unsigned long long);
	*((unsigned long long *)&buf[size]) = (unsigned long long)MEDMAT_MAGIC_NUMBER; size += sizeof(unsigned long long);

	// writing sizeof(T) for debugging
	*((int *)&buf[size]) = (int)sizeof(T); size += sizeof(int);


	// matrix nrows, ncols, normalized_flag, transposed_flag, missing value
	*((int *)&buf[size]) = nrows; size += sizeof(int);
	*((int *)&buf[size]) = ncols; size += sizeof(int);
	*((int *)&buf[size]) = normalized_flag; size += sizeof(int);
	*((int *)&buf[size]) = transposed_flag; size += sizeof(int);
	*((T *)&buf[size]) = missing_value; size += sizeof(T);

	// matrix itself
	memcpy(&buf[size], &m[0], (size_t)nrows*ncols*sizeof(T));
	size += (unsigned long long)nrows * ncols * sizeof(T);

	// avg size followed by avg elements, then std size followed by std elements
	*((int *)&buf[size]) = (int)avg.size(); size += sizeof(int);
	memcpy(&buf[size], &avg[0], avg.size()*sizeof(T));
	*((int *)&buf[size]) = (int)std.size(); size += sizeof(int);
	memcpy(&buf[size], &std[0], std.size()*sizeof(T));

	// recordsMetadata size followed by records
	int r_size = (int)recordsMetadata.size();
	*((int *)&buf[size]) = r_size; size += sizeof(int);
	memcpy(&buf[size], &recordsMetadata[0], (size_t)r_size * sizeof(RecordData));
	size += (unsigned long long)r_size * sizeof(RecordData);


	// names of columns 
	int s_size = (int)signals.size();
	*((int *)&buf[size]) = s_size; size += sizeof(int);
	for (int i=0; i<s_size; i++) {
		// size of string, followed by string
		int slen = (int)signals[i].length();
		*((int *)&buf[size]) = slen; size += sizeof(int);
		memcpy(&buf[size], signals[i].c_str(), slen);
		size += slen;
	}

	// Done !
	return size;
}

//...........................................................................................
template <class T> size_t MedMat<T>::deserialize(unsigned char *buf)
{
	size_t size = 0;
	// 2 magic nums
	unsigned long long magic1 = *((unsigned long long *)&buf[size]); size += sizeof(unsigned long long);
	unsigned long long magic2 = *((unsigned long long *)&buf[size]); size += sizeof(unsigned long long);

	if (magic1 != (unsigned long long)MEDMAT_MAGIC_NUMBER) {
		fprintf(stderr, "MedMat deserialize error: Wrong magic number %llx instead of %llx\n", magic1, (unsigned long long)MEDMAT_MAGIC_NUMBER);
		return (size_t)-1;
	}

	if (magic2 == (unsigned long long)MEDMAT_MAGIC_NUMBER) {

		// sizeof (T) 
		int sizeT = *((int *)&buf[size]); size += sizeof(int);
		if (sizeT != (int)sizeof(T)) {
			fprintf(stderr, "MedMat deserialize error: sizeT not matching %d vs. %d\n", sizeT, (int)sizeof(T));
			return (size_t)-1;
		}

		// matrix nrows, ncols, normalized_flag, transposed_flag, missing value
		nrows = *((int *)&buf[size]); size += sizeof(int);
		ncols = *((int *)&buf[size]); size += sizeof(int);
		normalized_flag = *((int *)&buf[size]); size += sizeof(int);
		transposed_flag = *((int *)&buf[size]); size += sizeof(int);
		missing_value = *((T *)&buf[size]); size += sizeof(T);

		//cerr << "desrialize: nrows " << nrows << " ncols " << ncols << " sizeT " << sizeT << "\n";
		// matrix itself
		m.resize((size_t)nrows*ncols);
		memcpy(&m[0], &buf[size], (size_t)nrows*ncols*sizeof(T));
		size += (unsigned long long)nrows * ncols * sizeof(T);

		// avg size followed by avg elements, then std size followed by std elements
		int a_size = *((int *)&buf[size]); size += sizeof(int);
		avg.resize(a_size);
		memcpy(&avg[0], &buf[size], avg.size()*sizeof(T));
		int std_size = *((int *)&buf[size]); size += sizeof(int);
		std.resize(a_size);
		memcpy(&std[0], &buf[size], std.size()*sizeof(T));

		// recordsMetadata size followed by records
		int r_size = *((int *)&buf[size]); size += sizeof(int);
		recordsMetadata.resize(r_size);
		memcpy(&recordsMetadata[0], &buf[size], (size_t)r_size * sizeof(RecordData));
		size += (unsigned long long)r_size * sizeof(RecordData);


		// names of columns 
		int s_size = *((int *)&buf[size]); size += sizeof(int);
		signals.resize(s_size);
		for (int i=0; i<s_size; i++) {
			// size of string, followed by string
			int slen = *((int *)&buf[size]); size += sizeof(int);
			signals[i].assign((char *)&buf[size], slen);
			size += slen;
		}

	}
	else {
		fprintf(stderr, "MedMat deserialize error: unsupported mode %llx\n", magic2);
		return (size_t)-1;
	}

	return size;
}
*/

//...........................................................................................
template <class T> int MedMat<T>::read_from_bin_file(const string &fname)
{
#if 1
	unsigned char *data;
	unsigned long long size;
	if (read_binary_data_alloc(fname, data, size) < 0) {
		fprintf(stderr, "Error reading file %s\n", fname.c_str()); fflush(stderr);
		return -1;
	}

//	cerr << "before serialize size is " << size << "\n";
	if (deserialize(data) == (size_t)-1)
		return -1;

//	cerr << "after deserialize\n";

	delete[] data;
	return 0;
#endif
#if 0
	// OLDER code - kept for a while

	if (!file_exists(fname)) {
		cerr << "File " << fname << " doesn't exist\n";
		return -1;
	}
	cerr << "reading binary data from " << fname << "\n";

	ifstream inf;
	
	inf.open(fname,ios::in|ios::binary);
	if (!inf)
		return -1;




	// read nrows,ncols
	inf.read((char *)(&nrows),sizeof(int));
	inf.read((char *)(&ncols),sizeof(int));

	m.resize(nrows*ncols);

	char *d = (char *)&m[0];
	inf.read(d, nrows*ncols*sizeof(T));

	int r_size;
	inf.read((char *)(&r_size), sizeof(int));
	for (int i = 0; i < r_size; i++) {
		RecordData r;
		inf.read((char *)(&r.id), sizeof(r.id));
		inf.read((char *)(&r.date), sizeof(r.date));
		inf.read((char *)(&r.time), sizeof(r.time));
		inf.read((char *)(&r.split), sizeof(r.split));
		inf.read((char *)(&r.weight), sizeof(r.weight));
		inf.read((char *)(&r.label), sizeof(r.label));
		inf.read((char *)(&r.pred), sizeof(r.pred));
		recordsMetadata.push_back(r);
	}

	int c_size;
	inf.read((char *)(&c_size), sizeof(int));
	for (int i = 0; i < c_size; i++) {
		int len;
		inf.read((char *)(&len), sizeof(int));
		std::vector<char> tmp(len);
		inf.read(tmp.data(), len); //deserialize characters of string
		string name;
		name.assign(tmp.data(), len);
		signals.push_back(name);
	}

	inf.close();
	
	return 0;

#endif

}

//...........................................................................................
template <class T> int MedMat<T>::write_to_bin_file(const string &fname)
{
	vector<unsigned char> serialized;
	size_t size = get_size();
	serialized.resize(size+1);
	serialize(&serialized[0]);
	if (write_binary_data(fname, &serialized[0], size) < 0) {
		fprintf(stderr, "MedMat write_to_bon_file ERROR: failed writing to %s\n", fname.c_str());
		return -1;
	}

	return 0;

#if 0

	ofstream of;
	of.open(fname, ios::out|ios::binary);
	if (!of) {
		fprintf(stderr, "Can not write to %s\n", fname.c_str());
		throw exception();
	}

	// OLDER code - kept for a while
	cerr << "writing binary " << fname << " with " << nrows << "X" << ncols <<" :: elem size " << sizeof(T) << "\n";
	of.write((char *)(&nrows),sizeof(int));
	of.write((char *)(&ncols),sizeof(int));
	of.write((char *)(&m[0]),sizeof(T)*nrows*ncols);
	int r_size = (int)recordsMetadata.size();
	of.write((char *)(&r_size), sizeof(int));
	cerr << "writing additional data for " << r_size << " records\n";
	for (RecordData r : recordsMetadata) {
		of.write((char *)(&r.id), sizeof(r.id));
		of.write((char *)(&r.date), sizeof(r.date));
		of.write((char *)(&r.time), sizeof(r.time));
		of.write((char *)(&r.split), sizeof(r.split));
		of.write((char *)(&r.weight), sizeof(r.weight));
		of.write((char *)(&r.label), sizeof(r.label));
		of.write((char *)(&r.pred), sizeof(r.pred));
	}

	int c_size = (int)signals.size();
	of.write((char *)(&c_size), sizeof(int));
	cerr << "writing additional data for " << c_size << " columns\n";
	for (string name: signals) {
		int len = (int)name.size();
		of.write((char *)(&len), sizeof(len));
		of << name;
	}

	of.close();

	return 0;
#endif
}


//...........................................................................................
// expected format for titles line: id, date, time, split, weight, <signals>, label, pred
// if no titles line, then only the naked matrix is expected
//template <class T> int MedMat<T>::read_from_csv_file(const string &fname, int titles_line_flag, vector<string>& fields_out)
template <class T> int MedMat<T>::read_from_csv_file(const string &fname, int titles_line_flag)
{
	clear();
	if (!file_exists(fname)) {
		fprintf(stderr, "File %s doesn't exist\n",fname.c_str());
		throw std::exception();
	}
	fprintf(stderr, "reading data from %s\n", fname.c_str());
	ifstream inf;
	inf.open(fname, ios::in);
	if (!inf) {
		cerr << "can not open file\n";
		throw std::exception();
	}
	ncols = -1;
	string curr_line;
	int METADATA_COLUMNS_PREFIX = 0;
	int METADATA_COLUMNS_SUFFIX = 0;
	if (titles_line_flag == 1) {	
		METADATA_COLUMNS_PREFIX = 5;
		METADATA_COLUMNS_SUFFIX = 2;
	}
	int METADATA_COLUMNS = METADATA_COLUMNS_PREFIX + METADATA_COLUMNS_SUFFIX;

	while (getline(inf, curr_line)) {
		boost::trim(curr_line);
		vector<string> fields;
		boost::split(fields, curr_line, boost::is_any_of(","));
		if (ncols == -1) {
			if (titles_line_flag) {
				assert(fields[0].compare("pid") == 0);
				assert(fields[1].compare("date") == 0);
				assert(fields[2].compare("outcomeTime") == 0);
				assert(fields[3].compare("split") == 0);
				assert(fields[4].compare("weight") == 0);
				for (int i = METADATA_COLUMNS_PREFIX; i < fields.size() - METADATA_COLUMNS_SUFFIX; i++)
					signals.push_back(fields[i]);
				
				assert(fields.end()[-2].compare("label") == 0);
				assert(fields.end()[-1].compare("pred") == 0);
				ncols = (int)fields.size() - METADATA_COLUMNS;
				assert(ncols >= 0);
				continue;
			}
			else {
				ncols = (int)fields.size();
				assert(ncols >= 0);
			}
		}
		if (fields.size() != ncols + METADATA_COLUMNS) {
			//char msg[200];
			string msg = "expected " + to_string(ncols + METADATA_COLUMNS) + " fields, got " + to_string((int)fields.size()) + "fields in line: " + curr_line.c_str() + "\n";
			//sprintf(msg, "expected %d fields, got %d fields in line: %s\n", ncols + METADATA_COLUMNS, (int)fields.size(), curr_line.c_str());
			throw runtime_error(msg.c_str());
		}
		if (METADATA_COLUMNS > 0) {
			RecordData sample(stoi(fields[0]), stoi(fields[1]), stol(fields[2]), stoi(fields[3]), stof(fields[4]),
				stof(fields.end()[-2]), stof(fields.end()[-1]));
			recordsMetadata.push_back(sample);
		}
		vector<T> new_row(ncols);
		for (int i = 0; i < ncols; i++)
			new_row[i] = (T)stof(fields[i + METADATA_COLUMNS_PREFIX]);
		add_rows(new_row);		
	}

	inf.close();
	fprintf(stderr, "read %lldX%lld data\n", nrows, ncols);
	return 0;
}

template <class T> int MedMat<T>::write_to_csv_file(const string &fname) {
	fprintf(stderr, "writing %s with %lldX%lld data\n", fname.c_str(), nrows, ncols);
	ofstream of;
	of.open(fname, ios::out);
	if (!of) {
		fprintf(stderr, "Error: failed opening file %s\n", fname.c_str());
		//cerr << "Error: " << strerror(errno);
		throw std::exception();
	}
	bool with_signals = (signals.size() == ncols);
	bool with_records = (recordsMetadata.size() == nrows);

	if (signals.size() != ncols) 
		cerr << "ncols: " << ncols << " number of column names: " << signals.size() << ", not writing column names\n";
	if (recordsMetadata.size() != nrows)
		cerr << "nrows: " << nrows << " number of records metadata entries: " << recordsMetadata.size() << ", not writing record metadata\n";

	if (with_records && with_signals)
		of << "pid,date,outcomeTime,split,weight,";
	if (with_signals)
		for (int j = 0; j < ncols; j++) {
			of << signals[j] << ",";
		}
	if (with_records && with_signals)
		of << "label,pred\n";
	else if (with_signals)
			of << "\n";


	for (int i = 0; i < nrows; i++) {
		if (with_records)
			of << recordsMetadata[i].id << "," << med_time_converter.convert_times_S(global_default_time_unit, MedTime::DateTimeString, recordsMetadata[i].date)
			<< "," << recordsMetadata[i].outcomeTime << "," << recordsMetadata[i].split << "," << 
				recordsMetadata[i].weight << ",";
		for (int j = 0; j < ncols; j++) {
			of << get(i, j) << ",";
		}
		if (with_records)
			of << recordsMetadata[i].label << "," << recordsMetadata[i].pred;
		of << "\n";
	}
	of.close();
	return 0;
}


// normalization
//..............................................................................................................
inline void calculate_moments(int num, double sum, double sum2, float& mean, float& std, float missing_val) {

	if (num == 0) {
		mean = std = missing_val ;
	} else {
		mean = (float)(sum/(double)num);
		if (num > 1) {
			float val = (float)((sum2 - sum*mean)/(double)(num-1)) ;
			if (val > MED_FLT_EPSILON)
				std = sqrt(val) ;
			else
				std = 1 ; // Dummy std for constant value
		} else {
			std = 1 ;
		}
	}
}

//..............................................................................................
template <class T> void MedMat<T>::normalize(int norm_type, float *wgts) {

	double sum,sum2; // square sums become large fast...need doubles for this
	int num ;
	float val;

	if (norm_type == Normalize_Cols) { // Column-wise moments

		avg.resize(ncols) ;
		std.resize(ncols) ;

		for (int j=0; j<ncols; j++) {

			sum = sum2 = 0 ;
			num = 0 ;

			for (int i=0; i<nrows; i++) {
				val = (float)get(i,j);
				if (val != missing_value) {
					if (wgts != NULL) val *= wgts[i];
					num ++ ;
					sum += val ;
					sum2 += static_cast<double>(val) * static_cast<double>(val);
				}
			}

			float av,sd;
			calculate_moments(num,sum,sum2,av,sd,(float)missing_value) ;
			avg[j] = (T)av;
			std[j] = (T)sd;
		}

	} else { // Row-wise moments

		avg.resize(nrows) ;
		std.resize(nrows) ;

		for (int i=0; i<nrows; i++) {
			sum = sum2 = 0 ;
			num = 0 ;

			for (int j=0; j<ncols; j++) {
				val = (float)get(i,j);
				if (val != missing_value) {
					if (wgts != NULL) val *= wgts[i];
					num ++ ;
					sum += val ;
					sum2 += static_cast<double>(val) * static_cast<double>(val);
				}
			}

			float av,sd;
			calculate_moments(num,sum,sum2,av,sd,(float)missing_value) ;
			avg[i] = (T)av;
			std[i] = (T)sd;
		}
	}

	// Normalize
	normalize(avg,std,norm_type) ;
}

template <class T> void MedMat<T>::get_cols_avg_std(vector<T>& _avg, vector<T>& _std)
{
	_avg.resize(ncols);
	_std.resize(ncols);

	for (int j=0; j<ncols; j++) {

		double _sum = 0 , _sum2 = 0;
		int _num = 0;

		for (int i=0; i<nrows; i++) {
			float val = (float)get(i, j);
			if (val != missing_value) {
				_num++;
				_sum += val;
			}
		}


		if (_num > 0)
			_avg[j] = (T)(_sum/_num);
		else
			_avg[j] = (T)0;

		_num = 0;
		for (int i=0; i<nrows; i++) {
			T val = get(i, j);
			if (val != missing_value) {
				_num++;
				_sum2 += (double)(val - _avg[j])*(val - _avg[j]);
			}
		}

		if (_num > 0)
			_std[j] = (T)sqrt((double)_sum2/_num);
		else
			_std[j] = (T)1;
		//float av, sd;
		//calculate_moments(_num, _sum, _sum2, av, sd, (float)missing_value);
		//_avg[j] = (T)av;
		//_std[j] = (T)sd;
	}
}

template <class T> template <class S> void MedMat<T>::normalize (const vector<S>& external_mean, const vector<S>& external_std, int norm_type) {
	
	normalized_flag = norm_type ;
	vector<S> internal_std(external_std.size());// to go with the const attr
	for (int i=0; i<external_std.size(); i++) {
		if (external_std[i] == 0)
			internal_std[i] = 1;
		else
			internal_std[i] = external_std[i];
	}

	for (size_t i=0; i<nrows; i++) {
		for (size_t j=0; j<ncols; j++) {
			if (normalized_flag == Normalize_Cols) {
				if (m[i*ncols +j] == missing_value)
					m[i*ncols +j]  = 0 ;
				else if (internal_std.size())
					m[i*ncols + j] = (m[i*ncols + j] - external_mean[j])/internal_std[j] ;
				else
					m[i*ncols + j] = (m[i*ncols + j] - external_mean[j]) ;
			} else {
				if (m[i*ncols +j] == missing_value)
					m[i*ncols +j]  = 0 ;
				else if (internal_std.size())
					m[i*ncols + j] = (m[i*ncols + j] - external_mean[i])/internal_std[i] ;
				else
					m[i*ncols + j] = (m[i*ncols + j] - external_mean[i]) ;
			}
		}
	}
}

//..............................................................................................................................
template <class T> void MedMat<T>::print_row(FILE *fout, const string &prefix, const string &format, int i_row)
{
	fprintf(fout, "%s :: [%d,:] :", prefix.c_str(),i_row);
	for (int i=0; i<ncols; i++)
		fprintf(fout, format.c_str(), get(i_row, i));
	fprintf(fout, "\n");
}


//..............................................................................................................................
template <class T> void MedMat<T>::set_signals(vector<string> & sigs)
{
	signals.clear();
	for (auto &sig : sigs)
	{
		signals.push_back(sig);
	}
}

//..............................................................................................................................
template <class T> void MedMat<T>::random_split_mat_by_ids(MedMat<T> &mat_0, MedMat<T> &mat_1, float p0, vector<int> &inds0, vector<int> &inds1)
{
	if (recordsMetadata.size() != nrows)
		HMTHROW_AND_ERR("ERROR: MedMat : Can't split a matrix by id with a non matching recordsMetadata (%d records != %d rows)\n", (int)recordsMetadata.size(), (int)nrows);

	// collect ids and randomize the 0 group
	unordered_set<int> all_ids, ids_0;
	for (int i = 0; i < nrows; i++)
		all_ids.insert(recordsMetadata[i].id);
	for (auto id : all_ids)
		if (rand_1() < p0)
			ids_0.insert(id);

	// calculate sizes for matrices
	mat_0.clear();
	mat_0.copy_header(*this);
	mat_1.copy_header(*this);
	int nrows0 = 0, nrows1 = 0;
	inds0.clear();
	inds1.clear();
	vector<int> assignment(nrows, 0);
	for (int i = 0; i < nrows; i++) {
		if (ids_0.find(recordsMetadata[i].id) != ids_0.end()) {
			nrows0++;
			inds0.push_back(i);
		}
		else {
			nrows1++;
			assignment[i] = 1;
			inds1.push_back(i);
		}
	}

	mat_0.nrows = nrows0;
	mat_0.m.resize(mat_0.nrows*mat_0.ncols);
	mat_0.recordsMetadata.resize(mat_0.nrows);
	mat_0.row_ids.resize(mat_0.nrows);

	mat_1.nrows = nrows1;
	mat_1.m.resize(mat_1.nrows*mat_1.ncols);
	mat_1.recordsMetadata.resize(mat_1.nrows);
	mat_1.row_ids.resize(mat_1.nrows);

	int i0 = 0, i1 = 0;
	for (int i = 0; i < nrows; i++) {
		if (assignment[i]) {
			for (int j = 0; j < ncols; j++)
				mat_1(i1, j) = m[i*ncols + j];
			mat_1.recordsMetadata[i1] = recordsMetadata[i];
			mat_1.row_ids[i1] = recordsMetadata[i].id;
			i1++;
		}
		else {
			for (int j = 0; j < ncols; j++)
				mat_0(i0, j) = m[i*ncols + j];
			mat_1.recordsMetadata[i0] = recordsMetadata[i];
			mat_1.row_ids[i0] = recordsMetadata[i].id;
			i0++;
		}
	}

}
