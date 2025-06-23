Lung Cancer Survivability Prediction

This project aims to predict the survivability of lung cancer patients using machine learning techniques. The goal is to assist healthcare professionals in assessing patient prognosis and supporting clinical decision-making.

Table of Contents

Project Overview

Dataset

Project Structure

Technologies Used

Installation

Usage

Results

Future Work



Project Overview

Lung cancer is one of the most prevalent and deadly forms of cancer worldwide. Early prediction of patient survivability based on clinical and demographic factors can be valuable for treatment planning.
This project builds a predictive model that uses patient data to estimate survivability outcomes.

Dataset

The project uses the SEER Dataset (or another public lung cancer dataset — update accordingly).
The dataset includes features such as:

Age

Gender

Cancer stage

Histology type

Tumor size

Treatment type

Survival months

... and other clinical variables.


Note: The dataset is preprocessed to handle missing values and normalize the features.

Project Structure

lung-cancer-survivability-prediction/
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks for EDA and modeling
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
├── models/                 # Saved models
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
└── main.py                 # Main script to run the project

Technologies Used

Python

NumPy

Pandas

Matplotlib / Seaborn

Scikit-learn

XGBoost / LightGBM (optional, if used)

Jupyter Notebook


Installation

1. Clone the repository:



git clone https://github.com/your-username/lung-cancer-survivability-prediction.git
cd lung-cancer-survivability-prediction

2. Create a virtual environment and install dependencies:



python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows

pip install -r requirements.txt

Usage

1. Preprocess the data:



python src/data_preprocessing.py

2. Train the model:



python src/model_training.py

3. Evaluate the model:



python src/model_evaluation.py

Alternatively, you can run everything from main.py.

Results

The model achieves an accuracy of XX% and an AUC score of YY on the test dataset.

Visualizations of model performance, feature importance, and ROC curves are included in the notebooks/ directory.


Future Work

Explore deep learning models (e.g., neural networks)

Integrate additional clinical data

Deploy the model as a web application using Flask / Streamlit

Continuous model updating with new data
