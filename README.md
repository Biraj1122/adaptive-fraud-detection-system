# Adaptive Hybrid AI Verification System for Web Transactions

This repository contains the implementation of my ST5001CMD Artificial Intelligence individual coursework. The project combines machine learning and heuristic search to build an adaptive verification system for web transactions. Its goal is to improve security while reducing unnecessary verification steps for legitimate users.

---

## Project Overview

Traditional authentication systems often rely on fixed rules that apply the same verification process to every user. This can either increase user friction by requiring unnecessary verification or reduce security by allowing risky transactions to proceed without additional checks.

This project addresses that problem by combining two AI techniques:

- **Machine Learning:** A Random Forest Classifier analyzes transaction data (`Time`, `Amount`, `V1`–`V28`) and predicts the probability that a transaction is fraudulent.
- **Heuristic Search:** An A* Search algorithm uses the predicted fraud probability to determine the most appropriate verification path for each transaction.

Instead of making decisions based only on predefined rules, the system adapts its verification process according to the estimated level of risk.

### How It Works

1. A transaction is provided as input.
2. The Random Forest model predicts the likelihood of fraud.
3. The prediction is converted into a risk score.
4. The A* Search algorithm uses this score as part of its heuristic function.
5. The algorithm selects the most suitable verification route.

For example:

- **Low-risk transactions** are processed through a fast verification path with minimal user interruption.
- **High-risk transactions** are routed through additional security steps such as multi-factor authentication before approval.

This approach allows the system to balance security and user experience by making verification decisions based on predicted risk rather than static rules.

---

## Repository Structure

```
.
├── Kaggle's_credit_card_fraud_repository.py   # Main application containing data processing, model training, prediction, and A* search
├── README.md                                  # Project documentation
└── requirements.txt                           # Project dependencies (optional)
```

---

## Installation

### Prerequisites

- Python 3.10 or later

Install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

If a `requirements.txt` file is included, you can install everything with:

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the Credit Card Fraud Detection dataset available on Kaggle.
"https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download"

Place the `creditcard.csv` file in the project directory before running the application.

The dataset is not included in this repository because it exceeds GitHub's file size limit.

---

## Running the Project

Run the application using:

```bash
python "Kaggle's_credit_card_fraud_repository.py"
```

The program will:

- Load and preprocess the dataset.
- Train the Random Forest classifier.
- Predict fraud probability for a transaction.
- Use A* Search to determine the appropriate verification path.
- Display the prediction results and selected verification route.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Random Forest Classifier
- A* Search Algorithm

---

## Project Features

- Credit card fraud detection using machine learning
- Adaptive verification based on transaction risk
- A* Search for decision-making
- Dynamic verification routing
- Simple command-line implementation

---

## Author

**Biraj Sharma Chapagain**

ST5001CMD – Artificial Intelligence Coursework"# ai_project" 
