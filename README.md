# Context-Aware V2X Message Classification

A deep-learning prototype for classifying Vehicle-to-Everything (V2X)
message text into six categories and estimating whether a message is
spam.

> **Research status:** This project currently uses a realistic-inspired
> synthetic dataset. The reported results are preliminary synthetic-data
> results and should not be interpreted as validated performance on real
> vehicle-generated V2X traffic.

## Overview

The system accepts: - A V2X message as text - A priority value (`Low`,
`Medium`, `High`, or `Unknown`) - A selected model architecture

It returns: - A predicted message category - The predicted category
confidence - A spam / non-spam prediction and spam probability - A
probability breakdown across all six categories

### Message categories

1.  Emergency
2.  Hazard
3.  Roadwork
4.  Traffic
5.  Weather
6.  Spam

## Model architectures

The project includes four CNN-based hybrid recurrent variants:

-   CNN-BiGRU
-   CNN-BiLSTM
-   CNN-StackLSTM
-   CNN-BiRNN

The text branch uses text vectorization, an embedding layer, a 1D
convolution and max pooling, followed by a recurrent layer. A separate
priority branch embeds the priority value. The branches are combined
before the classification and spam-prediction output heads.

## Repository structure

``` text
V2X-Project/
├── data/
│   ├── V2X_demo_dataset.csv
│   ├── V2X_expanded_synthetic_dataset_12000.csv
│   ├── V2X_realistic_inspired_dataset_12000.csv
│   └── V2X_synthetic_training_dataset.csv
├── models_realistic/
│   ├── CNN_BiGRU.keras
│   ├── CNN_BiLSTM.keras
│   ├── CNN_BiRNN.keras
│   ├── CNN_StackLSTM.keras
│   └── comparison_results.json
├── V2X_All_In_One.py
├── V2X_Run.bat
├── requirements.txt
├── README_RUN.txt
├── V2X_Project_PPT.pptx
└── V2X_Research_Paper_IEEE.pdf
```

File names may change as the project is updated.

## Requirements

-   Python 3.10 or a compatible version supported by the packages in
    `requirements.txt`
-   pip
-   The dependencies listed in `requirements.txt`

## Setup

Clone the repository:

``` bash
git clone https://github.com/sumit-code5849/V2X-Project.git
cd V2X-Project
```

Create and activate a virtual environment (recommended):

**Windows PowerShell**

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the application

The consolidated application entry point is `V2X_All_In_One.py`.

If the script launches the Streamlit interface, run:

``` bash
streamlit run V2X_All_In_One.py
```

On Windows, the repository also includes `V2X_Run.bat`; you can run it
from File Explorer or from the project directory:

``` powershell
.\V2X_Run.bat
```

Refer to `README_RUN.txt` for the project-specific run instructions. If
the commands above differ from that file, follow the instructions in
`README_RUN.txt`.

## Dataset and evaluation notes

The current 12,000-row realistic-inspired dataset is synthetic; it is
not raw traffic collected from deployed vehicles. The documented
experiment used a stratified 70% / 15% / 15% train-validation-test split
with random seed 42.

The project has recorded preliminary perfect classification metrics on
the synthetic dataset. Such results can be affected by synthetic
templates, label construction, and possible shortcut learning. In
particular, the spam target is aligned with the `Spam` category, so the
spam output may learn the category distinction rather than an
independent spam signal.

Before making real-world performance claims, further work is needed: -
Evaluate on independently collected or appropriately licensed real V2X
data. - Audit near-duplicate and template leakage across splits. - Test
across multiple random seeds and report variability. - Check calibration
of predicted probabilities. - Measure model size and inference
latency. - Review padding and sequence-handling behavior.

## Research paper

The repository includes `V2X_Research_Paper_IEEE.pdf`, a draft prepared
in an IEEE-style conference layout for review. Its presence does not
imply IEEE or Springer acceptance, publication, or endorsement.

## Reproducibility

For reproducible experiments, record the Python and package versions,
dataset version, split seed, model configuration, and training settings
used for each run. Do not commit credentials, API keys, private data, or
local virtual-environment folders.

## License

No license is currently specified. All rights and reuse permissions
remain subject to the project owner's decision until a license is added.
