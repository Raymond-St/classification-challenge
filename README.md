# Spam Detector

A machine learning project that compares Logistic Regression and Random Forest classifiers for email spam detection.

## Overview

This project implements and compares two different machine learning models for email spam classification:
- Logistic Regression
- Random Forest Classifier

The models are trained on a dataset containing various email features and evaluated based on their accuracy in distinguishing between spam and non-spam emails.

## Dataset

The dataset used in this project is from the [UCI Machine Learning Repository (Spambase)](https://archive.ics.uci.edu/dataset/94/spambase) and includes features like:
- Word frequency metrics
- Character frequency metrics
- Capital letter statistics
- Binary classification (spam/not spam)

## Results

The comparison between the two models showed:
- Logistic Regression Accuracy: 92.79%
- Random Forest Accuracy: 96.70%

The Random Forest classifier outperformed the Logistic Regression model by approximately 3.9 percentage points, confirming the initial hypothesis that Random Forest would be better suited for this classification task.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy
- Jupyter Notebook

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```
pip install pandas scikit-learn numpy jupyter
```

## Usage

1. Start Jupyter Notebook:
```
jupyter notebook
```

2. Open the `spam_detector.ipynb` notebook

3. Run the cells to:
   - Load and explore the dataset
   - Preprocess the data
   - Train the Logistic Regression model
   - Train the Random Forest Classifier
   - Compare model performance

## Project Structure

- `spam_detector.ipynb`: Main Jupyter notebook containing the analysis
- `README.md`: This file
- `.gitignore`: Git ignore file for Python projects

## Future Improvements

- Implement hyperparameter tuning for both models
- Add feature importance analysis
- Explore additional models like SVM or Neural Networks

## License

This project is licensed under the Creative Commons Zero v1.0 Universal License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- scikit-learn for the implementation of machine learning algorithms
