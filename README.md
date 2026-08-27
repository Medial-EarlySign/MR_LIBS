# Medial EarlySign Libraries

![Pepy Total Downloads](https://img.shields.io/pepy/dt/medpython)
![PyPI - License](https://img.shields.io/pypi/l/medpython)
![GitHub contributors](https://img.shields.io/github/contributors-anon/Medial-EarlySign/medpython)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/Medial-EarlySign/medpython)

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/medpython/badges/version.svg)](https://anaconda.org/conda-forge/medpython)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/medpython/badges/platforms.svg)](https://anaconda.org/conda-forge/medpython)
![Conda Downloads](https://img.shields.io/conda/d/conda-forge/medpython)


## Overview of Medial Infrastructure
Medial EarlySign provides an infrastructure to convert **Electronic Medical Records (EMR)** - a complex, semi-structured time-series dataset into **machine-learning-ready** data and reproducible model pipelines. The library is optimized for sparse time-series EMR data and is designed for low memory usage and fast processing at scale.
Unlike images or free text, EMR data can be stored in complex format. The infrastructure standardize both the storage and the processing of time-series signals. We can think about this infrastructure as "TensorFlow" of medical data machine learning.

Key benefits at a glance:

- **Fast and memory-efficient processing** for large-scale EMR sparse time series where general-purpose libraries (e.g., pandas) are often impractical.
- Shareable, tested pipelines and methods that save engineering time and reduce duplicated effort.
- Built-in safeguards to reduce **data leakage** and time-series-specific overfitting.
- **Production-ready**: easily deployable in Docker or minimal Linux images.

This framework is deployed in production across multiple healthcare sites and played a key role in our award-winning submission to the [CMS AI Health Outcomes Challenge](https://www.cms.gov/priorities/innovation/innovation-models/artificial-intelligence-health-outcomes-challenge).

### Main contributers from recent years:
- [Avi Shoshan](https://www.linkedin.com/in/avi-shoshan-a684933b/)
- [Yaron Kinar](https://www.linkedin.com/in/yaron-kinar-il/)
- [Alon Lanyado](https://www.linkedin.com/in/lanyado/)

### Challenges
- **Variety of Questions**: Risk prediction (e.g., cancer, CKD), compliance, diagnostics, treatment recommendations
- **Medical Data Complexity**: Temporal irregularity, high dimensionality (>100k categories), sparse signals, multiple data types
- **Retrospective Data Issues**: Noise, bias, spurious patterns, policy sensitivity

### Platform Requirements
- **Performance**: Ultra-efficient in memory & time (>100x compare to native python pandas in some cases, mainly in preprocessing)
- **Extensibility**: Rich APIs, configurable pipelines, support new data types
- **Minimal Rewriting & Ease Of Usage**: JSON‑driven configs, unified codebase, python API to the C library
- **Comprehensive**: From "raw" data to model deployment
- **Reproducible & Versioned**: Track data, code, models, and parameters

## Documentation

Please refer to [MR_WIKI](https://medial-earlysign.github.io/MR_Wiki/) for full documentation
- [Installation](https://medial-earlysign.github.io/MR_Wiki/Installation/)


## Setup
To install the python module:
```bash
pip install medpython
```

Build from source:
```bash
python -m pip install -v "medpython @ git+https://github.com/Medial-EarlySign/medpython.git/#subdirectory=Internal/MedPyExport/generate_binding"
```

To explore the code:

* `Internal/MedPyExport` - for the python module. The folder `generate_bindings` contains the `setup.py` to install the module.
* `Internal/AlgoMarker` - the folder to compile the minimal shared library for production and running it in distroless container.

