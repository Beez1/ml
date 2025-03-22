
# Insurance Cost Prediction

This project analyzes medical insurance costs and builds a linear regression model to predict insurance charges based on various factors.

## Dataset

The dataset (`data/Insurance.csv`) contains information about insurance charges and the following features:
- age: Age of primary beneficiary
- sex: Gender of insurance contractor (female/male)
- bmi: Body mass index
- children: Number of children covered by health insurance
- smoker: Smoking status (yes/no)
- region: Beneficiary's residential area (northeast/southeast/southwest/northwest)
- charges: Individual medical costs billed by health insurance

## Project Structure

```
.
├── data/
│   └── Insurance.csv
├── notebooks/
│   └── assignment1.ipynb 
├── scripts/
│   └── linear_regression.py
└── README.md
```

## Analysis

The Jupyter notebook `notebooks/assignment1.ipynb` contains:
- Exploratory data analysis
- Feature engineering
- Model training and evaluation
- Visualizations of results

The `scripts/linear_regression.py` implements the linear regression model for predicting insurance costs.

## Getting Started

1. Clone this repository
2. Install required dependencies
3. Run the Jupyter notebook to see the analysis
4. Use the linear regression script to make predictions

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
