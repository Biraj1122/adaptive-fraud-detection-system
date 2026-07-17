import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Configure standard academic chart styles
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# =====================================================================
# 1. DATABASE INGESTION LAYER
# =====================================================================

class DatabaseLoader:
    """
    Handles the retrieval and preprocessing of the credit card fraud database
    directly from the host machine's storage disk.
    """
    def __init__(self, explicit_path=r"C:\Users\Asus Tuf\Desktop\3rd Sem AI project\creditcard.csv"):
        self.file_path = explicit_path

    def load_csv(self):
        """
        Ingests the CSV database file, cleans data gaps, 
        and filters down structural feature vectors to optimize memory.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"[CRITICAL ERROR] Database file not found at path: '{self.file_path}'. "
            )
            
        print(f" -> Accessing Storage Database: '{self.file_path}'...")
        
        # Reading first 50,000 rows to optimize execution speeds on standard laptop hardware
        df = pd.read_csv(self.file_path, nrows=50000)
        
        # Target column schema matching the file structure
        required_columns = ['Time', 'Amount', 'V1', 'V2', 'Class']
        
        # Data Preprocessing: Drop row entries with missing/NaN feature attributes
        df = df.dropna(subset=required_columns)
        
        print(f" -> Database Ingest Complete. Successfully loaded {len(df)} entries from disk.")
        return df, required_columns


# =====================================================================
# 2. MACHINE LEARNING & ENSEMBLE CONFIGURATION
# =====================================================================

class AcademicRiskClassifier:
    """
    Ingests database features to train a Random Forest ensemble,
    providing continuous fraud probability values for the planning framework.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.db_loader = DatabaseLoader()
        self.is_fitted = False
        self.X_test = None
        self.y_test = None

    def execute_model_training(self):
        """Loads files, splits arrays, and trains the estimators."""
        df, cols = self.db_loader.load_csv()
        
        # Feature columns excluding target labels
        X = df[['Time', 'Amount', 'V1', 'V2']].values
        y = df['Class'].values
        
        # Train-test split calculation to prevent predictive data leakage
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(" -> Training Random Forest Ensemble on records...")
        self.model.fit(X_train, y_train)
        
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(self.X_test, self.y_test)
        print(f" -> Model Fitted successfully. Train Accuracy: {train_acc*100:.2f}% | Test Accuracy: {test_acc*100:.2f}%")
        self.is_fitted = True

    def predict_fraud_likelihood(self, feature_vector):
        """Yields continuous confidence scores used directly by the search layer."""
        if not self.is_fitted:
            self.execute_model_training()
        vector = np.array([feature_vector])
        proba = self.model.predict_proba(vector)[0][1]
        return float(proba)


# =====================================================================
# 3. STATE TRAVERSAL & DECISION MAKING (A* SEARCH MECHANICS)
# =====================================================================

class SystemVerificationState:
    def __init__(self, name, execution_friction_cost):
        self.name = name
        self.friction_cost = float(execution_friction_cost)

    def __eq__(self, other):
        return self.name == other.name if isinstance(other, SystemVerificationState) else False

    def __hash__(self):
        return hash(self.name)


class InformedRoutePlanner:
    """
    Evaluates risk probability outputs using a dynamic A* search state graph.
    """
    def __init__(self, probability_score, system_latency=45):
        self.p_fraud = probability_score
        self.latency = system_latency
        
        self.START = SystemVerificationState("UNVERIFIED_ENTRY_QUEUE", 0)
        self.BYPASS = SystemVerificationState("FRICTIONLESS_AUTO_CLEARANCE", 10)
        self.CHALLENGE = SystemVerificationState("MFA_IDENTITY_CHALLENGE", 60)
        self.TERMINATE = SystemVerificationState("ACCOUNT_SUSPENSION_HOLD", 300)
        self.GOAL = SystemVerificationState("RESOLVED_COMPLIANCE_TARGET", 0)

    def fetch_valid_transitions(self, state: SystemVerificationState):
        if state == self.START:
            return [self.BYPASS, self.CHALLENGE, self.TERMINATE]
        if state in [self.BYPASS, self.CHALLENGE, self.TERMINATE]:
            return [self.GOAL]
        return []

    def evaluate_heuristic(self, state: SystemVerificationState):
        """Maintains reasoning under uncertainty dynamically based on risk scores."""
        # This is where A* search utility is generated over a simple classifier.
        # The heuristic dynamic shapes the search landscape based on ML probability.
        
        if state == self.GOAL:
            return 0.0
            
        if state == self.BYPASS:
            # If probability is high, the cost to bypass becomes mathematically infinite,
            # forcing the A* search loop to explore other paths.
            if self.p_fraud > 0.50:
                return float('inf') 
            return self.p_fraud * 200.0

        if state == self.CHALLENGE:
            # Moderate risk routes here; cost decreases as fraud probability increases.
            return (1.0 - self.p_fraud) * 80.0

        if state == self.TERMINATE:
            # High risk routes. Factor in system constraints (latency).
            if self.latency > 100:
                return float('inf') # Constraint violation
            return (1.0 - self.p_fraud) * 600.0

        return 50.0

    def run_pathfinding_optimization(self):
        open_list = [self.START]
        came_from = {}
        g_cost = {self.START: 0.0}
        f_cost = {self.START: self.evaluate_heuristic(self.START)}

        while open_list:
            current = min(open_list, key=lambda n: f_cost.get(n, float('inf')))
            
            if current == self.GOAL:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, g_cost[self.GOAL]

            open_list.remove(current)
            
            for neighbor in self.fetch_valid_transitions(current):
                tentative_g = g_cost[current] + neighbor.friction_cost
                
                if tentative_g < g_cost.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g
                    f_cost[neighbor] = tentative_g + self.evaluate_heuristic(neighbor)
                    if neighbor not in open_list:
                        open_list.append(neighbor)
                        
        return None, float('inf')


# =====================================================================
# 4. REPORT VISUALIZATION MODULE
# =====================================================================

class AnalyticalVisualizer:
    """
    Generates required academic figures and plots directly from data frames
    and model training sessions, explicitly saving to the designated workspace.
    """
    @staticmethod
    def generate_report_figures(classifier: AcademicRiskClassifier):
        print("\n [Generating Visualizations] Creating academic figures...")
        
        # Explicit definition of the exact destination path
        target_directory = r"C:\Users\Asus Tuf\Desktop\3rd Sem AI project"
        
        # Ensure directory exists before saving (robustness)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # --- Figure 1: Confusion Matrix ---
        plt.figure(figsize=(6, 4.5))
        y_pred = classifier.model.predict(classifier.X_test)
        cm = confusion_matrix(classifier.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
        plt.title('Figure 1: Risk Classifier Confusion Matrix', fontweight='bold', fontsize=12)
        plt.xlabel('Predicted System Designation')
        plt.ylabel('True Historical Profile')
        plt.tight_layout()
        
        # Enforcing explicit absolute path routing
        plt.savefig(os.path.join(target_directory, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print(f"   -> 'confusion_matrix.png' saved explicitly to: {target_directory}")

        # --- Figure 2: Feature Importance Histogram ---
        plt.figure(figsize=(7, 4))
        # Note: Must match features used in training (execute_model_training)
        feature_names = ['Time', 'Amount', 'V1', 'V2']
        importances = classifier.model.feature_importances_
        
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(feature_names)), importances[indices], color='#2b5c8f', edgecolor='black', width=0.5)
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices])
        plt.title('Figure 2: Ensemble Feature Importance Gini Weight Summary', fontweight='bold', fontsize=12)
        plt.ylabel('Relative Weight Score')
        plt.xlabel('Profile Attribute Vector')
        plt.tight_layout()
        
        # Enforcing explicit absolute path routing
        plt.savefig(os.path.join(target_directory, 'feature_importance.png'), dpi=300)
        plt.close()
        print(f"   -> 'feature_importance.png' saved explicitly to: {target_directory}")


# =====================================================================
# 5. EXECUTION DRIVER WITH FORCE PLOT EXTRACTION
# =====================================================================

if __name__ == "__main__":
    print("=====================================================================")
    print(" INITIALIZING INTEGRATED HYBRID AI SYSTEM WITH DATABASE CORES")
    print("=====================================================================\n")
    
    # 1. Initialize classifier layer
    classifier = AcademicRiskClassifier()
    
    # 2. Define interactive query data sample parameters [Time, Amount, V1, V2]
    # Sample format matching dataset schema
    sample_safe_txn = [0.0, 149.62, -1.359807, -0.072781]
    sample_fraud_txn = [406.0, 0.00, -2.312227, 1.951992]
    
    # 3. Run evaluation paths
    print("--- EVALUATION RUN A: STANDARD ACCOUNT TRANSACTION ---")
    prob_a = classifier.predict_fraud_likelihood(sample_safe_txn)
    planner_a = InformedRoutePlanner(probability_score=prob_a)
    path_a, cost_a = planner_a.run_pathfinding_optimization()
    print(f" -> Selected Verification Pipeline Target: {path_a[1].name} (Path Friction Cost: {cost_a})")
    
    print("\n--- EVALUATION RUN B: MALICIOUS TRANSFER INCIDENT ---")
    prob_b = classifier.predict_fraud_likelihood(sample_fraud_txn)
    planner_b = InformedRoutePlanner(probability_score=prob_b)
    path_b, cost_b = planner_b.run_pathfinding_optimization()
    print(f" -> Selected Verification Pipeline Target: {path_b[1].name} (Path Friction Cost: {cost_b})")
    
    # 4. CRITICAL: Explicitly force plot generation to absolute directory path
    print("\n--- TRIGGERING AUTOMATED VISUALIZATION REPORT EXTRACTION ---")
    AnalyticalVisualizer.generate_report_figures(classifier)
    print("=====================================================================")