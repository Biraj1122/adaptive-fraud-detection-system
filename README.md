# 🛡️ Adaptive Hybrid AI Verification System for Web Transactions

# PROJECT OVERVIEW

# This repository contains the complete implementation of the ST5001CMD 
# Artificial Intelligence individual coursework submission for the March 
# Intake 2026 session. The project engineers a resilient Hybrid AI 
# Architecture that combines statistical ensemble classification 
# mechanics with an informed graph-search framework to dynamically manage 
# web transaction verification pipelines. The platform balances institutional 
# transaction security against consumer onboarding drop-off risks by 
# optimizing user verification pathways under real-world behavioral 
# uncertainty.

# Traditional fraud prevention architectures rely on rigid, rule-based 
# authentication workflows or standalone machine learning inference blocks 
# operating in total isolation. Static rules introduce excessive user 
# friction, leading to high cart abandonment rates, while standalone 
# predictive classifiers lack the context required to dynamically budget 
# latency, external API execution expenses, and cumulative user friction 
# constraints.

# This codebase overcomes these modular limitations by bridging two highly 
# optimized AI paradigms:

# 1. Machine Learning Layer: A robust Random Forest Classifier augmented 
#    with an in-memory over-sampling protocol (SMOTE) to evaluate transaction 
#    patterns and generate a continuous fraud probability matrix 
#    (0.0 <= P_Fraud <= 1.0).
#
# 2. Heuristic Search Layer: An Informed A* Search framework that ingests 
#    the continuous threat probability vector to dynamically warp its 
#    heuristic cost estimation topology h(n), calculating the path of 
#    absolute lowest user friction necessary to clear the session securely.

# TECHNICAL FEATURES & IN-MEMORY ARCHITECTURES

# 1. Feature Selection and Schema Layout:
#    The ingestion pipeline targets an explicit structural feature matrix 
#    extracted directly from raw data logs:
#    * Time: Continuous time log sequence capturing seconds elapsed since 
#      the dataset baseline.
#    * Amount: Absolute monetary size of active transaction to trap rapid 
#      velocity spikes.
#    * V1 & V2: Anonymized behavioral vectors derived via Principal Component 
#      Analysis (PCA) tracking interaction characteristics.
#    * Class: Binary validation target label (1 = Fraud, 0 = Legitimate).

# 2. Resolving Class Imbalance via In-Memory SMOTE:

#    Real-world transactional fraud records are heavily skewed (>99% 
#    legitimate transactions). To eliminate the accuracy paradox, the 
#    codebase integrates a Synthetic Minority Over-sampling Technique 
#    (SMOTE) preprocessing block. Running strictly on the training partition 
#    within volatile RAM, SMOTE synthesizes mathematically distinct 
#    intermediate fraud vectors. This pushes the internal training matrix 
#    to a clean 50/50 balance ratio, ensuring highly precise decision trees 
#    while leaving the test partition un-sampled to guarantee realistic 
#    academic evaluation metrics.

# 3. A* Heuristic Path Planning:

#    The A* engine handles sequential business logic workflow using the 
#    standard graph optimization equation:
#    f(n) = g(n) + h(n)
#    When P_Fraud breaches a calibrated post-SMOTE threshold (>0.40), the 
#    heuristic evaluation function h(n) for the automated bypass lane 
#    scales dynamically to infinity. This mathematically warps the search 
#    space topology, forcing the A* priority queue to drop the frictionless 
#    clearance node and explore alternative multi-stage verification routes 
#    (e.g., initiating Multi-Factor MFA challenges or administrative holds) 
#    based on live system latency vectors.


# REPOSITORY STRUCTURE


# ├── creditcard.csv                             # Local gold-standard dataset file
# ├── app.py                                     # Interactive Streamlit Web Application
# ├── Kaggle's_credit_card_fraud_repository.py   # Integrated Hybrid System backend script
# ├── data_cleaning_comparison.png              # Figure A: Ingestion & Cleaning Breakdown
# ├── smote_rebalance_comparison.png             # Figure B: Pre vs Post-SMOTE Distribution
# ├── confusion_matrix.png                      # Figure 1: Post-SMOTE Confusion Matrix
# ├── feature_importance.png                    # Figure 2: Relative Feature Importances
# ├── table_b1.png                               # Table B.1: Predictive Performance Matrix
# ├── table_b2.png                               # Table B.2: Traversal & Latency Audit Log
# └── README.md                                  # Up-to-date academic project documentation


# INSTALLATION & ENVIRONMENT SETUP

# Prerequisites:
# - Python 3.10+
# - Access to a terminal environment with privileges to install packages

# Install Dependencies via Pip:
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn streamlit


# DATASET CONFIGURATION

# 1. Download the Kaggle Credit Card Fraud Detection dataset (284,807 
#    European cardholder records) directly from: 
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 2. Extract the archive and ensure the file is named creditcard.csv.
# 3. Place creditcard.csv directly into your workspace directory:
#    C:\Users\Asus Tuf\Desktop\3rd Sem AI project\

# Note: Source CSV remains completely unaltered on disk; all cleaning and 
# SMOTE rebalancing occur dynamically in-memory at runtime.


# SYSTEM EXECUTION OPTIONS


# Option A: Launch Interactive Streamlit Dashboard
streamlit run app.py

# Option B: Execute CLI Backend & Automated Visualization Extraction
python "Kaggle's_credit_card_fraud_repository.py"


# SYSTEM EXECUTION SEQUENCE

# 1. Data Ingestion Layer: Loads dense 150,000-record partition from disk.
# 2. SMOTE Layer: Isolates training split in RAM and synthesizes minority instances to 50/50 balance.
# 3. Model Training: Fits 100 Random Forest decision trees capped at depth 10.
# 4. Predictive Inference: Extracts threat probabilities for live transaction checks.
# 5. A* Routing Engine: Computes graph-search optimizations across network nodes.
# 6. Report Visualizer: Auto-generates and saves figures (data_cleaning_comparison.png, 
#    smote_rebalance_comparison.png, confusion_matrix.png, feature_importance.png, 
#    table_b1.png, table_b2.png) into local project folder.


# AUTHOR & ACADEMIC AFFILIATION

# Author: Biraj Sharma Chapagain
# Student Identification Number (UID): 250125
# Module Details: ST5001CMD - Artificial Intelligence Individual Coursework
# Academic Institution: Softwarica College of IT and E-Commerce in partnership with Coventry University
# Module Leader / Evaluator: Suman Shrestha
