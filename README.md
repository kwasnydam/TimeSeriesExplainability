# TimeSeriesExplainability

Review of methods for time-series explainability of Neural Networks.

The demo presented here uses pretrained models for speech-based age group classification.

# Installation
In order to install, run the following chain of commands:
0. First, get the submodules
```
git submodule update --init --recursive
```

1. Create python virtualenv and install the requirements:
```
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Inside the virtualnevironment install the ftdnn submodule:
```
./install.sh
```

# Overview
For all the details about the algorithms used please check the demo notebook `time_series_xplain.ipynb`.
There is little point in copy-pasting its contents here.