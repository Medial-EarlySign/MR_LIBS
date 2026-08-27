#include "MedPlot.h"
#include <MedIO/MedIO/MedIO.h>
#include <fstream>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
//define something for Windows
string BaseResourcePath = "W:\\Graph_Infra";
#else
string BaseResourcePath = "/nas1/Work/Graph_Infra";
#endif
#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MED_UTILS

using namespace std;

inline string get_html_template() {
	ifstream file(BaseResourcePath + path_sep() + "Graph_HTML.txt");
	string content = "";
	if (file.is_open()) {
		string ln;
		while (getline(file, ln))
			content += "\n" + ln;
		file.close();
	}
	else {
		content = "<html>\n<head>\n  <!-- Plotly.js -->\n   <script src=\"plotly-latest.min.js\"></script>\n</head>\n\n<body>\n  <div id=\"myDiv\" align=\"center\"><!-- Plotly chart will be drawn inside this DIV --></div>\n  <script>\n    <!-- JAVASCRIPT CODE GOES HERE -->\n\n\t{0}\n\nPlotly.newPlot(\'myDiv\', data, layout);\n\tfunction toggleMissingAll(data) {\n\t\tfor (var i=0;i<data.length;i++) {\n\t\t\ttoggleMissing(data[i]);\n\t\t}\n\t\tPlotly.redraw(\'myDiv\', data, layout);\n\t}\n\t\n\tfunction toggleMissing(trace) {\n\t\tvar show_msn=true;\n\t\tif (\'show_msn\' in trace) {\n\t\t\tshow_msn=trace.show_msn;\n\t\t\tif (show_msn) {\n\t\t\t\tif (trace.msn_state) {\n\t\t\t\t\ttrace.msn_state=false;\n\t\t\t\t\ttrace.x = trace.x.slice(1);\n\t\t\t\t\ttrace.y = trace.y.slice(1);\n\t\t\t\t\ttrace.text = [];\n\t\t\t\t\ttrace.mode = trace.old_mode;\n\t\t\t\t}\n\t\t\t\telse {\n\t\t\t\t\ttrace.msn_state=true;\n\t\t\t\t\ttrace.x.unshift(trace.msn_x);\n\t\t\t\t\ttrace.y.unshift(trace.msn_y);\n\t\t\t\t\ttrace.text=[\'Missing_value\'];\n\t\t\t\t\ttrace.mode = trace.old_mode + \'+text\';\n\t\t\t\t}\n\t\t\t}\n\t\t\treturn;\n\t\t}\n\t\tvar msn_x=trace.x[0];\n\t\tif (show_msn && (trace.x[0] <= -65336)) {\n\t\t\tif (trace.x.length > 2) {\n\t\t\t\tmsn_x=trace.x[1]-Math.abs(trace.x[2]-trace.x[1]);\n\t\t\t}\n\t\t\telse {\n\t\t\t\tshow_msn=true;\n\t\t\t\tmsn_x=-1; \n\t\t\t}\n\t\t}\n\t\telse { //first value is not missing_value - no missing value\n\t\t\tshow_msn=false;\n\t\t}\n\t\t\n\t\t\n\t\ttrace.x[0] = msn_x;\n\t\ttrace[\'show_msn\']=show_msn;\n\t\ttrace[\'msn_x\']=msn_x;\n\t\ttrace[\'msn_y\']=trace.y[0];\n\t\ttrace[\'msn_state\']=show_msn;\n\t\ttrace[\'old_mode\']=trace.mode;\n\t\tif (show_msn) {\n\t\t\ttrace[\'text\']=[\'Missing_value\'];\n\t\t\ttrace.mode = trace.mode + \'+text\';\n\t\t}\n\t}\n\tfunction searchXY() {\n\t\tsNum = parseInt(document.getElementById(\'seriesNum\').value);\n\t\tx = parseFloat(document.getElementById(\'xVal\').value);\n\t\ty = parseFloat(document.getElementById(\'yVal\').value);\n\t\tvar ser = data[sNum];\n\t\t\n\t\tif (\'z\' in ser) { //search in 3D Graph\n\t\t\tvar ind = ser.x.indexOf(x) + (ser.y.indexOf(y) % ser.x.length);\n\t\t\tdocument.getElementById(\'res\').innerHTML = ser.z[ind];\n\t\t\treturn ser.z[ind];\n\t\t}\n\t\telse {\n\t\tvar minDistance = NaN;\n\t\tvar minDistanceInd = NaN;\n\t\t\tif (!isNaN(x)) {\n\t\t\t\tfor (i=0; i< ser.x.length; ++i) {\n\t\t\t\t\tif (isNaN(minDistance) || Math.abs(ser.x[i] - x) < minDistance) {\n\t\t\t\t\t\tminDistance = Math.abs(ser.x[i] - x);\n\t\t\t\t\t\tminDistanceInd = i;\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t\tdocument.getElementById(\'xVal\').value = ser.x[minDistanceInd];\n\t\t\t\tdocument.getElementById(\'yVal\').value = ser.y[minDistanceInd];\n\t\t\t}\n\t\t\telse if(!isNaN(y)) {\n\t\t\t\tfor (i=0; i< ser.y.length; ++i) {\n\t\t\t\t\tif (isNaN(minDistance) || Math.abs(ser.y[i] - y) < minDistance) {\n\t\t\t\t\t\tminDistance = Math.abs(ser.y[i] - y);\n\t\t\t\t\t\tminDistanceInd = i;\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t\tdocument.getElementById(\'xVal\').value = ser.x[minDistanceInd];\n\t\t\t\tdocument.getElementById(\'yVal\').value = ser.y[minDistanceInd];\n\t\t\t}\n\t\t\t\n\t\t}\n\t}\n\t\n  </script>\n  \n  <h3 align=\"center\">Search for values in graph (leave x or y empty to search for the other value)</h3>\n   <p align=\"center\"> \n   <label for \"seriesNum\">series number (top is 0)</label>\n   <input type=\"number\" name=\"seriesNum\" id=\"seriesNum\" value=\"0\"></input>\n  <label for=\"xVal\">x</label>\n  <input type=\"number\" name=\"xVal\" id=\"xVal\"></input>\n  <label for=\"xVal\">y</label>\n  <input type=\"number\" name=\"yVal\" id=\"yVal\"></input>\n  <input type=\"button\" value=\"search\" onclick=\"searchXY();\"></input>\n  </p>\n <p><input type=\"button\" value=\"Toggle missing value\" onclick=\"toggleMissingAll(data);\"></input></p>\n <span id=\"res\"></span>\n  \n</body>\n</html>";
	}
	return content;
}

string float2Str(float num) {
	//return to_string((round(num * 1000) / 1000));
	char res[50];
	snprintf(res, sizeof(res), "%2.4f", num);
	return string(res);
}

map<float, float> BuildHist(vector<float> featNums) {
	map<float, float> hist;
	for (float n : featNums)
	{
		if (hist.find(n) == hist.end()) {
			hist[n] = 0;
		}
		++hist[n];
	}
	return hist;
}

map<float, float> BuildAggeration(const vector<vector<float>> &vec_x, const vector<float> &y,
	float(*aggFunction)(const vector<float> &),
	float(*combineFeat)(const vector<float>&)) {
	map<float, float> res;
	map<float, vector<float>> data;

	for (size_t i = 0; i < vec_x.size(); ++i)
	{
		if (y.size() != vec_x[i].size()) {
			std::cout << "y_size=" << y.size() << " vec_" << i << "_size=" << vec_x[i].size() << endl;
			throw logic_error("not all feature vectors has same length of target. there is no matching between them");
		}
	}

	vector<float> rowData(vec_x.size());
	float bucket;
	for (int i = 0; i < y.size(); ++i)
	{
		for (int k = 0; k < vec_x.size(); ++k) {
			rowData[k] = vec_x[k][i];
		}
		if (combineFeat == NULL && vec_x.size() == 1) {
			bucket = vec_x[0][i];
		}
		else {
			bucket = combineFeat(rowData);
		}

		data[bucket].push_back(y[i]);
	}

	for (auto it = data.begin(); it != data.end(); ++it) {
		res[it->first] = aggFunction(it->second);
	}

	return res;
}

void Build3Data(const vector<float> &x1, const vector<float> &x2,
	const vector<float> &y,
	float(*aggFunction)(const vector<float> &), vector<vector<float>> &data, int min_filter_cnt) {
	//aggregate for each tuples of x1,x2 aggFucntion on y list results
	if (x1.size() != x2.size() || x1.size() != y.size()) {
		throw invalid_argument("arrays must have same size");
	}
	data = vector<vector<float>>(3);
	map<float, map<float, vector<float>>> d;
	for (size_t i = 0; i < x1.size(); ++i)
	{
		d[x1[i]][x2[i]].push_back(y[i]);
	}
	for (auto it = d.begin(); it != d.end(); ++it)
	{
		for (auto jt = it->second.begin(); jt != it->second.end(); ++jt)
		{
			if (jt->second.size() < min_filter_cnt)
				continue; //filtered out
			data[0].push_back(it->first);
			data[1].push_back(jt->first);
			data[2].push_back(aggFunction(jt->second));
		}
	}

	if (data[0].size() == 0)
		throw invalid_argument("filtered all points - min_filter_cnt is too high or axis bining is needed");
}

void createHtmlGraph(const string &outPath, const vector<map<float, float>> &data, const string &title, const string &xName
	, const string &yName, const vector<string> &seriesNames, int refreshTime, const string &chart_type, const string &mode, const string &template_str)
{
	vector<vector<pair<float, float>>> x_data(data.size());
	for (size_t i = 0; i < data.size(); ++i)
	{
		x_data[i].resize(data[i].size());
		int j = 0;
		for (const auto &it : data[i])
		{
			x_data[i][j].first = it.first;
			x_data[i][j].second = it.second;
			++j;
		}
	}
	createScatterHtmlGraph(outPath, x_data, title, xName, yName, seriesNames, refreshTime, chart_type, mode, template_str);
}

void createScatterHtmlGraph(const string &outPath, const vector<vector<pair<float, float>>> &data, const string &title,
	const string &xName, const string &yName, const vector<string> &seriesNames,
	int refreshTime, const string &chart_type, const string &mode, const string &template_str) {

	string x_name = "x";
	string y_name = "y";
	if (chart_type == "pie") {
		x_name = "labels";
		y_name = "values";
	}

	std::filesystem::path p(outPath);
	std::filesystem::path outDir = p.parent_path();

	string content = template_str;
	if (template_str.empty())
		content = get_html_template();

	size_t ind = content.find("{0}");
	if (ind == string::npos) {
		throw invalid_argument("Not Found in template. need to contain string {0}\n");
	}

	string rep = "";

	for (size_t i = 0; i < data.size(); ++i)
	{
		const vector<pair<float, float>> &dmap = data[i];

		rep += "var series" + to_string(i) + " = {\n type: '" + chart_type + "',\n mode: '" + mode + "',\n " + x_name + ": [";
		for (auto it = dmap.begin(); it != dmap.end(); ++it) {
			rep += float2Str(it->first) + ", ";
		}
		if (rep[rep.size() - 2] == ',') {
			rep = rep.substr(0, rep.size() - 2);
		}
		rep += "], \n";

		rep += y_name + ": [";
		for (auto it = dmap.begin(); it != dmap.end(); ++it) {
			rep += float2Str(it->second) + ", ";
		}
		if (rep[rep.size() - 2] == ',') {
			rep = rep.substr(0, rep.size() - 2);
		}
		rep += "] \n";
		if (seriesNames.size() > 0) {
			if (seriesNames.size() != data.size()) {
				throw invalid_argument("wrong number of series names passed with data graphs to plot");
			}
			rep += ", name: '";
			rep += seriesNames[i];
			rep += "' \n";
		}
		rep += "};\n";
	}

	if (refreshTime > 0) {
		char buf[100];
		snprintf(buf, 100, "setTimeout(function() { window.location.reload(1); }, %d);", refreshTime);
		rep += buf;
	}


	rep += "var data = [";
	for (size_t i = 0; i < data.size(); ++i)
		rep += " series" + to_string(i) + ", ";

	rep = rep.substr(0, rep.size() - 2);
	rep += " ]; \n";

	rep += "var layout = { \n  title:'";
	rep += title + "',\n ";

	if (chart_type != "pie") {
		rep += "xaxis: { title : '";
		rep += xName;
		rep += "'}, \n yaxis: { title: '";
		rep += yName + "'},\n ";
	}


	rep += "height: 800, \n    width: 1200 \n }; ";

	content.replace(ind, 3, rep);
	if (template_str.empty())
		content.replace(content.find("\"plotly-latest.min.js\""), 22, "\"/nas1/Work/Graph_Infra/plotly-latest.min.js\"");

	ofstream myfile;
	cerr << "writing: [" << outPath << "]\n";
	myfile.open(outPath);
	if (!myfile.good())
		cerr << "IO Error: can't write " << outPath << endl;
	myfile << content;
	myfile.close();

}

void createHtml3D(const string &outPath, const vector<vector<vector<float>>> &vec3d,
	const vector<string> &seriesNames, bool heatmap, const string &title, const string &xName, const string &yName, const string &zName) {
	if (vec3d.empty())
		throw invalid_argument("please pass at least one graph data");
	if (vec3d.size() != seriesNames.size())
		throw invalid_argument("seriesNames size and vec3d first dim should be same size");
	for (const vector<vector<float>> &v : vec3d)
		if (v.size() != 3)
			throw invalid_argument("please pass 3 signal vectors as input");

	vector<string> ind2axis = { "x", "y", "z" };

	/*ifstream jsFile;
	jsFile.open(BaseResourcePath + separator() + "plotly-latest.min.js");
	if (!jsFile.is_open()) {
		throw logic_error("Unable to open js file");
	}
	string jsData((istreambuf_iterator<char>(jsFile)),
		istreambuf_iterator<char>());
	ofstream jsOut;
	size_t lastDirPos = outPath.find_last_of("/\\");
	string outDir = outPath.substr(0, lastDirPos) + separator();
	if (lastDirPos == string::npos)
	{
		outDir = "";
	}

	jsOut.open(outDir + "plotly-latest.min.js");
	if (!jsOut.good())
		cerr << "IO Error: can't write " << outDir + "plotly-latest.min.js" << endl;
	jsOut << jsData;
	jsOut.close();*/

	string content = get_html_template();

	size_t ind = content.find("{0}");
	if (ind == string::npos) {
		throw invalid_argument("Not Found in template");
	}

	string type = "scatter3d";
	if (heatmap)
		type = "heatmap";

	string rep = "";
	for (size_t graph_id = 0; graph_id < vec3d.size(); ++graph_id)
	{
		rep += "var series" + to_string(graph_id) + " = {\n type: '" + type + "', \n mode: 'markers'";
		for (size_t i = 0; i < vec3d[graph_id].size(); ++i) {
			rep += ",\n" + ind2axis[i] + ": [";
			rep += float2Str(vec3d[graph_id][i][0]);
			for (size_t j = 1; j < vec3d[graph_id][i].size(); ++j)
				rep += ", " + float2Str(vec3d[graph_id][i][j]);
			rep += "]";
		}
		rep += "\n";
		rep += ", name: '";
		rep += seriesNames[graph_id];
		rep += "' \n";
		rep += "};\n";
	}

	rep += "var data = [";
	for (size_t i = 0; i < vec3d.size(); ++i)
		rep += " series" + to_string(i) + ", ";
	rep = rep.substr(0, rep.size() - 2);
	rep += " ]; \n";
	rep += "var layout = { \n  title:'";
	rep += title;
	if (!heatmap)
		rep += "', \n scene: {\n xaxis: { title : '";
	else
		rep += "', \n xaxis: { title : '";
	rep += xName;
	rep += "'}, \n yaxis: { title: '";
	rep += yName;
	rep += "'}, \n zaxis: { title: '";
	rep += zName;
	if (!heatmap)
		rep += "' }\n }, \n height: 800, \n    width: 1200 \n }; ";
	else
		rep += "'}, \n height: 800, \n    width: 1200 \n }; ";

	content.replace(ind, 3, rep);
	content.replace(content.find("\"plotly-latest.min.js\""), 22, "\"/nas1/Work/Graph_Infra/plotly-latest.min.js\"");

	ofstream myfile;
	myfile.open(outPath);
	if (!myfile.good())
		cerr << "IO Error: can't write " << outPath << endl;
	myfile << content;
	myfile.close();
}

string createCsvFile(const map<float, float> &data)
{
	string out = "X, Y\n";
	for (auto it = data.begin(); it != data.end(); ++it)
		out += to_string(it->first) + ", " + to_string(it->second) + "\n";

	return out;
}

string createCsvFile(const vector<vector<float>> &data, const vector<string> &headers)
{
	string out = "ROWS";
	for (size_t i = 0; i < headers.size(); ++i)
	{
		out += ", " + headers[i];
	}
	out += "\n";

	for (size_t i = 0; i < data.size(); ++i)
	{
		out += "ROW_" + to_string(i);
		for (size_t k = 0; k < data[i].size(); ++k)
		{
			out += ", " + float2Str(data[i][k]);
		}
		out += "\n";

	}

	return out;
}

vector<bool> empty_bool_arr;
void get_ROC_working_points(const vector<float> &preds, const vector<float> &y, const vector<float> &weights,
	vector<float> &pred_threshold, vector<float> &true_rate, vector<float> &false_rate, vector<float> &ppv, vector<float> &pr,
	const vector<bool> &indexes) {
	bool censor_removed = true;
	if (y.size() != preds.size())
		MTHROW_AND_ERR("Error in get_ROC_working_points - preds.size()=%zu, y.size()=%zu\n",
			preds.size(), y.size());
	if (!weights.empty() && y.size() != weights.size())
		MTHROW_AND_ERR("Error in get_ROC_working_points - y.size()=%zu, weights.size()=%zu\n",
			y.size(), weights.size());

	map<float, vector<int>> pred_indexes;
	double tot_true_labels = 0, tot_false_labels = 0;
	double tot_obj = 0;
	for (size_t i = 0; i < preds.size(); ++i)
		if (indexes.empty() || indexes[i]) {
			float weight = 1;
			if (!weights.empty())
				weight = weights[i];
			pred_indexes[preds[i]].push_back((int)i);
			tot_true_labels += static_cast<double>(y[i]) * static_cast<double>(weight);
			tot_false_labels += static_cast<double>(weight) * static_cast<double>((!censor_removed ? (1 - y[i]) : int(y[i] <= 0)));
			++tot_obj;
		}
	//tot_false_labels = tot_obj - tot_true_labels;
	if (tot_false_labels == 0 || tot_true_labels == 0) {
		cerr << "only controls or cases are given." << endl;
		throw logic_error("only controls or cases are given.\n");
	}
	pred_threshold = vector<float>((int)pred_indexes.size());
	map<float, vector<int>>::iterator it = pred_indexes.begin();
	for (size_t i = 0; i < pred_threshold.size(); ++i)
	{
		pred_threshold[i] = it->first;
		++it;
	}
	sort(pred_threshold.begin(), pred_threshold.end());
	//From up to down sort:
	double t_sum = 0;
	double f_sum = 0;
	//double n_samples = (double)preds.size();
	true_rate = vector<float>((int)pred_indexes.size());
	false_rate = vector<float>((int)pred_indexes.size());
	ppv = vector<float>((int)pred_indexes.size());
	pr = vector<float>((int)pred_indexes.size());
	for (int i = (int)pred_threshold.size() - 1; i >= 0; --i)
	{
		const vector<int> &indexes = pred_indexes[pred_threshold[i]];
		//calc AUC status for this step:
		for (int ind : indexes)
		{
			float weight = 1;
			if (!weights.empty())
				weight = weights[ind];
			float true_label = y[ind];
			t_sum += static_cast<double>(true_label) * static_cast<double>(weight);
			if (!censor_removed)
				f_sum += static_cast<double>(1 - true_label) * static_cast<double>(weight);
			else
				f_sum += int(true_label <= 0) * weight;
		}
		true_rate[i] = float(t_sum / tot_true_labels);
		false_rate[i] = float(f_sum / tot_false_labels);
		ppv[i] = float(t_sum / (t_sum + f_sum));
		pr[i] = float((t_sum + f_sum) / (tot_true_labels + tot_false_labels));
	}
}
void down_sample_graph(map<float, float> &points, int points_count) {
	if (points_count == 0 || points_count >= points.size())
		return;
	//linear interpolate points to points_count sample count:
	vector<pair<float, float>> xy((int)points.size());
	int i = 0;
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		xy[i] = pair<float, float>(it->first, it->second);
		++i;
	}
	sort(xy.begin(), xy.end()); //should be already sorted - it's map
	double jmp_size = double(xy.size() - 1) / (points_count - 1);
	points.clear();
	for (size_t j = 0; j < points_count; ++j)
	{
		double newPos = j * jmp_size;
		//find interpolation of 2 close integers:
		int basePos = (int)newPos;
		int endPos = basePos + 1;
		double beforeFactor = 1 - (newPos - basePos);
		double afterFactor = 1 - ((basePos + 1) - newPos);
		if (endPos >= xy.size()) {
			endPos = (int)xy.size() - 1;
			beforeFactor = 1;
			afterFactor = 0;
		}
		if (basePos >= xy.size()) {
			basePos = (int)xy.size() - 1;
			beforeFactor = 1;
			afterFactor = 0;
		}
		double interp_y = beforeFactor * xy[basePos].second + afterFactor * xy[endPos].second;
		double interp_x = beforeFactor * xy[basePos].first + afterFactor * xy[endPos].first;
		points[(float)interp_x] = (float)interp_y;
	}
}

void plotAUC(const vector<vector<float>> &all_preds, const vector<vector<float>> &y, const vector<vector<float>> &weights,
	const vector<string> &modelNames, string baseOut, bool print_y) {
	vector<float> pred_threshold;
	vector<float> true_rate;
	vector<float> false_rate;
	vector<float> ppv, pr;

	std::filesystem::create_directories(baseOut);
	vector<map<float, float>> allData;
	vector<map<float, float>> allPPV;
	vector<map<float, float>> allSensPPV;
	vector<map<float, float>> allSensPR, allPRPPV, allPRSens , allLiftScr;
	vector<double> auc((int)all_preds.size());
	vector<float> empty_vec;
	map<float, float> ref_graph;
	for (size_t i = 0; i < all_preds.size(); ++i)
	{
		const vector<float> &preds = all_preds[i];
		const vector<float> *w = &empty_vec;
		if (!weights.empty())
			w = &weights[i];
		get_ROC_working_points(preds, y[i], *w, pred_threshold, true_rate, false_rate, ppv, pr);
		map<float, float> false_true;
		map<float, float> th_false;
		map<float, float> false_ppv;
		map<float, float> xy;
		map<float, float> sens_ppv, sens_pr;
		map<float, float> pr_ppv, pr_sens;
		map<float, float> lift_by_score;
		float epsilon = (float)1e-8;
		for (size_t k = 0; k < true_rate.size(); ++k)
		{
			false_true[100 * false_rate[k]] = 100 * true_rate[k];
			th_false[pred_threshold[k]] = 100 * false_rate[k];
			false_ppv[100 * false_rate[k]] = 100 * ppv[k];
			if (i == 0) {
				xy[100 * false_rate[k]] = 100 * false_rate[k];
			}
			sens_ppv[float((int)(100 * true_rate[k]))] = ppv[k];
			sens_pr[float((int)(100 * true_rate[k]))] = pr[k];
			float rounded_pr = (float)((int)(1000 * pr[k])) / 10;
			pr_ppv[rounded_pr] = 100 * ppv[k];
			pr_sens[rounded_pr] = 100 * true_rate[k];

			float rounded_score = (float)((int)(10000.0f*pred_threshold[k])) / 10000.0f;
			lift_by_score[rounded_score] = ppv[k] / (ppv[0] + epsilon);

/*
			if (ppv[k] < (1.0f-epsilon) && (pr[k] < (1.0f-epsilon)) && (true_rate[k] > epsilon)) {
				float ppos = (ppv[k]*pr[k]*(1.0f/true_rate[k] - 1.0f))/(1.0 - pr[k]);
				if (ppos > epsilon && ppos < (1.0f - epsilon)) {
					float ods_nom = ppv[k] / (1.0f - ppv[k]);
					float ods_denom = ppos / (1.0f - ppos);
					float rounded_score = (float)((int)(10000.0f*pred_threshold[k])) / 10000.0f;

					ods_r_score[rounded_score] = ppos; // ppv[k]; // ods_nom / ods_denom;
				}
			}
*/
		}
		auc[i] = false_rate.back() * true_rate.back() / 2; //"auc" - saved in reversed order from smallest score to highest score)
		for (int k = (int)true_rate.size() - 1; k > 0; --k)
			auc[i] += (false_rate[k - 1] - false_rate[k]) * (true_rate[k - 1] + true_rate[k]) / 2;


		if (i == 0) {
			ref_graph = xy;
			down_sample_graph(ref_graph);
			//allData.push_back(xy);
		}
		down_sample_graph(false_true);
		down_sample_graph(false_ppv);
		allData.push_back(false_true);
		allPPV.push_back(false_ppv);
		allSensPPV.push_back(sens_ppv);
		allSensPR.push_back(sens_pr);
		allPRPPV.push_back(pr_ppv);
		allPRSens.push_back(pr_sens);
		//allORScr.push_back(ods_r_score);
		allLiftScr.push_back(lift_by_score);
		vector<map<float, float>> model_false_scores;
		down_sample_graph(th_false);
		model_false_scores.push_back(th_false);
		string fname = modelNames[i];
		fix_filename_chars(&fname);
		createHtmlGraph(baseOut + path_sep() + fname + "_False_Thresholds.html", model_false_scores,
			"False rate as function of thresholds", "Prediction Threshold score value", "False Positive Rate");
	}
	if (!all_preds.empty())
		allData.push_back(ref_graph);
	vector<string> data_titles(modelNames);
	//append Auc to titles
	char buff[200];
	for (size_t i = 0; i < data_titles.size(); i++) {
		snprintf(buff, sizeof(buff), "%s (AUC=%1.3f)", data_titles[i].c_str(), auc[i]);
		data_titles[i] = string(buff);
	}
	data_titles.push_back("x=y reference");
	createHtmlGraph(baseOut + path_sep() + "ROC.html", allData, "ROC curve", "False Positive Rate", "True Positive Rate", data_titles);
	data_titles = vector<string>(modelNames);
	createHtmlGraph(baseOut + path_sep() + "PPV.html", allPPV, "PPV curve", "False Positive Rate", "Positive Predictive Value", data_titles);
	createHtmlGraph(baseOut + path_sep() + "SensPPV.html", allSensPPV, "PPV by Sensitivity", "Sensitivity", "Positive Predictive Value", data_titles);
	createHtmlGraph(baseOut + path_sep() + "SensPR.html", allSensPR, "PR by Sensitivity", "Sensitivity", "Positivity Rate", data_titles);
	createHtmlGraph(baseOut + path_sep() + "PRPPV.html", allPRPPV, "PPV by PR", "PR", "Positive Predictive Value", data_titles);
	createHtmlGraph(baseOut + path_sep() + "PRSens.html", allPRSens, "Sensitivity by PR", "PR", "Sensitivity", data_titles);
	createHtmlGraph(baseOut + path_sep() + "ORScr.html", allLiftScr, "OR by score", "Score", "Lift", data_titles);


	if (print_y)
		for (size_t i = 0; i < y.size(); ++i) {
			allData.clear();
			allData.push_back(BuildHist(y[i]));
			string fname = modelNames[i];
			fix_filename_chars(&fname);
			createHtmlGraph(baseOut + path_sep() + "y_labels_" + fname + ".html", allData, "Y Labels", "Y",
				"Count", {}, 0, "pie");
		}
}

void plotAUC(const vector<vector<float>> &all_preds, const vector<float> &y, const vector<string> &modelNames,
	string baseOut, const vector<bool> &indexes, const vector<float> *weights) {
	vector<vector<float>> all_y(all_preds.size());
	vector<vector<float>> all_preds_filtered(all_preds.size());
	vector<vector<float>> all_weights_filtered(all_preds.size());
	for (size_t k = 0; k < all_preds.size(); ++k)
	{
		all_y[k].reserve((int)y.size());
		all_preds_filtered[k].reserve((int)y.size());
		all_weights_filtered[k].reserve(weights != NULL ? weights->size() : 0);
	}
	for (size_t i = 0; i < y.size(); ++i)
	{
		if (indexes.empty() || indexes[i])
			for (size_t k = 0; k < all_preds.size(); ++k)
			{
				all_preds_filtered[k].push_back(all_preds[k][i]);
				all_y[k].push_back(y[i]);
				if (weights != NULL && !weights->empty())
					all_weights_filtered[k].push_back(weights->at(i));
			}
	}

	plotAUC(all_preds_filtered, all_y, all_weights_filtered, modelNames, baseOut, false);
	vector<float> filty;
	filty.reserve((int)y.size());
	for (size_t i = 0; i < y.size(); ++i)
		if (indexes.empty() || indexes[i])
			filty.push_back(y[i]);

	vector<map<float, float>> allData;
	allData.push_back(BuildHist(filty));
	createHtmlGraph(baseOut + path_sep() + "y_labels.html", allData, "Y Labels", "Y",
		"Count", {}, 0, "pie");

}