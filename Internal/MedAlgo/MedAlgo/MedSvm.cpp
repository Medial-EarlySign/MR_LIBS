#include <MedAlgo/MedAlgo/MedAlgo.h>
#include <MedAlgo/MedAlgo/MedSVM.h>

#undef LOCAL_SECTION
#define LOCAL_SECTION LOG_MEDALGO
#define LOCAL_LEVEL	LOG_DEF_LEVEL
extern MedLogger global_logger;

void MedSvm::init_defaults() {
	transpose_for_learn = false;
	transpose_for_predict = false;
	normalize_for_learn = false;
	normalize_y_for_learn = false;
	normalize_for_predict = false;

	classifier_type = MODEL_SVM;

	params.C = 1.0;
	params.cache_size = 30000;
	params.kernel_type = SVM_LINEAR; //all options: LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
	params.svm_type = C_SVC; // all options: C_SVC, NU_SVC, ONE_CLASS(no labels, one class), EPSILON_SVR (regression), NU_SVR (Regression)
	params.degree = 3; //only when kernel = POLY 
	params.coef0 = 0; //only when kernel = POLY 
	params.gamma = 1; //kernel when POLY, RBF, SIGOID
	params.shrinking = 1; //for optimizatio process
	params.eps = 0.001; //for optimizatio process - termination crietria
	params.probability = 0; //won't calc and use probability
	params.nr_weight = 0;
}

int MedSvm::set_params(map<string, string>& mapper) {

	for (auto it = mapper.begin(); it != mapper.end(); ++it)
	{
		//! [MedSvm::init]
		if (it->first == "C") params.C = stod(it->second);
		else if (it->first == "cache_size")  params.cache_size = stod(it->second);
		else if (it->first == "coef0")  params.coef0 = stod(it->second);
		else if (it->first == "degree")  params.degree = stoi(it->second);
		else if (it->first == "eps")  params.eps = stod(it->second);
		else if (it->first == "gamma")  params.gamma = stod(it->second);
		else if (it->first == "kernel_type")  params.kernel_type = stoi(it->second);
		else if (it->first == "nr_weight")  params.nr_weight = stoi(it->second);
		else if (it->first == "nu")  params.nu = stod(it->second);
		else if (it->first == "p")  params.p = stod(it->second);
		else if (it->first == "probability")  params.probability = stoi(it->second);
		else if (it->first == "shrinking")  params.shrinking = stoi(it->second);
		else if (it->first == "svm_type")  params.svm_type = stoi(it->second);
		//else if (it->first == "weight") params->weight = stod(it->second);
		//else if (it->first == "weight_label")  params->p = stod(it->second);
		else MLOG("Unknown parameter \'%s\' for QRF\n", it->first.c_str());
		//! [MedSvm::init]
	}

	return 0;
}
int MedSvm::init(struct svm_parameter &params) {
	init_defaults();

	this->params = params;

	return 0;
}
int MedSvm::init(void *params) {
	init_defaults();
	struct svm_parameter *p = (struct svm_parameter *)params;

	init(*p);
	return 0;
}
MedSvm::MedSvm() {
	model = NULL;

	init_defaults();
};
MedSvm::~MedSvm() {
	svm_free_and_destroy_model(&model);
};

MedSvm::MedSvm(void *params) {
	init(params);
}

MedSvm::MedSvm(struct svm_parameter &params) {
	init(params);
}

int MedSvm::Learn(float *x, float *y, const float *w, int nsamples, int nftrs) {

	struct svm_problem problem;

	problem.l = nsamples;
	problem.x = new svm_node*[nsamples];
	problem.y = new double[nsamples]; // couldn't do y = y because our y is float *
	for (int i = 0; i < nsamples; ++i)
	{
		svm_node *node = new svm_node[nftrs+1];
		for (size_t j = 0; j < nftrs; ++j)
		{
			//if (x[i*nftrs + j] == MISSINg_Value)
			//	continue;
			node[j].index = (int)j;
			node[j].value = x[static_cast<size_t>(i) * static_cast<size_t>(nftrs) + j];
		}
		node[nftrs].index = -1;

		problem.x[i] = node;
		problem.y[i] = y[i]; // no need to convert to -1,+1 from 0,1. svm suports both formats
	}
	model = svm_train(&problem, &params);

	//free memory:
	for (int i = 0; i < nsamples; ++i)
		delete[] problem.x[i];
	delete[] problem.x;
	delete[] problem.y;

	return 0;
}

int MedSvm::Predict(float *x, float *&preds, int nsamples, int nftrs) const {

#pragma omp parallel for
	for (int i = 0; i < nsamples; ++i)
	{
		svm_node *node = new svm_node[nftrs+1];
		for (size_t j = 0; j < nftrs; ++j)
		{
			node[j].index = (int)j;
			node[j].value = x[static_cast<size_t>(i) * static_cast<size_t>(nftrs) + j];
		}
		node[nftrs].index = -1;

		preds[i] = (float)svm_predict(model, node);
		delete[] node;
	}

	return 0;
}

size_t MedSvm::get_size() {
	return 0;
}
size_t MedSvm::serialize(unsigned char *blob) {
	throw invalid_argument("plaeas use model.svm_save_model function directly");
}
size_t MedSvm::deserialize(unsigned char *blob) {
	throw invalid_argument("plaeas use model.svm_load_model function directly");
}