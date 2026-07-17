import os

# Define target folder and content matching project updates
target_dir = r"C:\Users\Asus Tuf\Desktop\3rd Sem AI project"
content = """# Adaptive Hybrid AI Verification System for Web Transactions

This repository contains the complete implementation of the **ST5001CMD Artificial Intelligence** individual coursework submission for the March Intake 2026 session. The project engineers a novel, resilient Hybrid AI Architecture that combines statistical ensemble classification mechanics with an informed graph-search framework to dynamically manage web transaction verification pipelines. The platform balances institutional transaction security against consumer onboarding drop-off risks by optimizing user verification pathways under real-world behavioral uncertainty.

---

## Project Overview

Traditional fraud prevention architectures rely on rigid, rule-based authentication workflows or standalone machine learning inference blocks operating in total isolation. Static rules introduce excessive user friction, leading to high cart abandonment rates, while standalone predictive classifiers lack the context required to dynamically budget latency, external API execution expenses, and cumulative user friction constraints.

This individual codebase overcomes these modular limitations by bridging two highly optimized syllabus paradigms:
* **Machine Learning Layer:** A robust Random Forest Classifier is augmented with an in-memory over-sampling protocol to evaluate transaction patterns and generate a continuous fraud probability matrix (0.0 <= P_Fraud <= 1.0).
* **Heuristic Search Layer:** An Informed A* Search framework ingests the continuous threat probability vector to dynamically warp its heuristic cost estimation topology (h(n)), calculating the path of absolute lowest user friction necessary to clear the session securely.

---

## Technical Features & In-Memory Architectures

### 1. Robust Feature Selection and Schema Layout
The ingestion pipeline targets an explicit structural feature matrix extracted directly from the raw data logs rather than unexplainable variables. The code maps five key indicators from the CSV database layout:
* Time: Continuous time log sequence capturing seconds elapsed since the dataset baseline.
* Amount: The absolute monetary size of the active transaction to trap rapid velocity spikes.
* V1 & V2: Anonymized behavioral vectors derived via Principal Component Analysis (PCA) tracking interaction characteristics.
* Class: The binary validation target label (1 = Fraud, 0 = Legitimate).

### 2. Resolving Class Imbalance via In-Memory SMOTE
Real-world transactional fraud records are heavily skewed (>99% legitimate transactions), exposing standard architectures to the accuracy paradox. To make the learning layer genuinely robust, the codebase integrates a **Synthetic Minority Over-sampling Technique (SMOTE)** preprocessing block. Running strictly on the training partition within volatile RAM via VS Code, SMOTE synthesizes mathematically distinct intermediate fraud vectors. This pushes the internal training matrix to a clean 50/50 balance ratio, ensuring highly precise decision trees while leaving the test partition un-sampled to guarantee realistic academic evaluation metrics.

### 3. Genuine A* Heuristic Path Planning
The A* engine handles the sequential business logic workflow. Rather than using static thresholds, the search space uses the standard graph optimization equation:

f(n) = g(n) + h(n)

When P_Fraud breaches a calibrated post-SMOTE threshold (>0.40), the heuristic evaluation function (h(n)) for the automated bypass lane scales dynamically to infinity. This mathematically warps the search space topology, forcing the A* priority queue to drop the frictionless clearance node and explore alternative multi-stage verification routes (e.g., initiating Multi-Factor challenges or administrative holds) based on live system latency vectors.

---

## Repository Structure

.
├── creditcard.csv                             # Local gold-standard dataset file (User Provided)
├── Kaggle's_credit_card_fraud_repository.py   # Complete integrated Hybrid System codebase
└── README.txt                                 # Up-to-date academic project documentation

---

## Installation & Setup

### Prerequisites
* Python 3.10 or later
* Access to a terminal environment with administration privileges for package distribution setups.

### Environmental Dependencies
Install the required data processing, visualization, and specialized imbalance learning frameworks using pip:

pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn

---

## Dataset Configuration

The implementation targets the definitive academic gold-standard Kaggle Credit Card Fraud Detection dataset (284,807 real European cardholder records).

1. Download the database file directly from the verified data repository:  
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Extract the file and ensure the physical database sheet is named exactly creditcard.csv.
3. Move the extracted file directly into your local machine's dedicated workspace directory:  
   C:\\Users\\Asus Tuf\\Desktop\\3rd Sem AI project\\

Important Operational Note: The script interacts with the physical data log securely on disk without modifying the raw database. The source CSV remains completely unaltered as all cleaning, array slicing, and SMOTE rebalancing occur dynamically in-memory at runtime.

---

## Running the Architecture

Execute the complete hybrid pipeline via the command line or within VS Code by running:

python "Kaggle's_credit_card_fraud_repository.py"

### System Execution Steps:
1. Data Ingestion Layer: Loads a dense, consecutive 150,000-record partition from local storage disk.
2. SMOTE Layer: Programmatically isolates the training split in memory and generates synthetic minority instances to balance the classes.
3. Model Training: Fits 100 base estimator decision trees capped at a maximum structural depth of 10 layers to preserve host memory.
4. Predictive Inference: Extracts probabilistic threat vectors for live interactive sample checks.
5. A* Routing Engine: Executes graph-search optimizations across network nodes, outputting the most efficient verification pipeline target alongside path friction metrics.
6. Report Visualizer: Automatically updates and saves professional high-resolution figures (confusion_matrix.png, feature_importance.png, table_b1.png, and table_b2.png) directly into your local project folder.

---

## Author & Academic Affiliation

* Author: Biraj Sharma Chapagain
* Student Identification Number (UID): 250125
* Module Details: ST5001CMD - Artificial Intelligence Individual Coursework
* Academic Institution: Softwarica College of IT and E-Commerce in partnership with Coventry University
* Module Leader / Evaluator: Suman Shrestha
"""

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

file_path = os.path.join(target_dir, "README.txt")
with open(file_path, "w", encoding="utf-8") as f:
    f.write(content.strip())

print(f"[SUCCESS] README.txt file has been successfully generated at: {file_path}")