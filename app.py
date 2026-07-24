import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# Page Configuration
st.set_page_config(
    page_title="Enterprise Hybrid AI Fraud Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Matplotlib styles
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

# Initialize Session State History
if 'audit_history' not in st.session_state:
    st.session_state.audit_history = []

# =====================================================================
# 1. DATABASE INGESTION & DATA CLEANING LAYER
# =====================================================================

class DatabaseLoader:
    def __init__(self, explicit_path=r"C:\Users\Asus Tuf\Desktop\3rd Sem AI project\creditcard.csv"):
        self.file_path = explicit_path
        self.raw_rows = 0
        self.cleaned_rows = 0
        self.dropped_rows = 0

    def load_csv(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Database file not found at path: '{self.file_path}'")
        
        raw_df = pd.read_csv(self.file_path, nrows=150000)
        self.raw_rows = len(raw_df)
        
        required_columns = ['Time', 'Amount', 'V1', 'V2', 'Class']
        clean_df = raw_df.dropna(subset=required_columns)
        self.cleaned_rows = len(clean_df)
        self.dropped_rows = self.raw_rows - self.cleaned_rows
        
        return clean_df, required_columns

# =====================================================================
# 2. MACHINE LEARNING ENSEMBLE WITH SMOTE
# =====================================================================

class AcademicRiskClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.db_loader = DatabaseLoader()
        self.is_fitted = False
        self.X_test = None
        self.y_test = None
        self.raw_counts = {}
        self.smote_counts = {}
        self.X_train_raw_shape = None
        self.X_train_smote_shape = None

    def execute_model_training(self):
        df, cols = self.db_loader.load_csv()
        X = df[['Time', 'Amount', 'V1', 'V2']].values
        y = df['Class'].values
        
        X_train_raw, self.X_test, y_train_raw, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.X_train_raw_shape = X_train_raw.shape
        self.raw_counts = {
            "Legitimate (Class 0)": int(sum(y_train_raw == 0)), 
            "Fraud (Class 1)": int(sum(y_train_raw == 1))
        }
        
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
        
        self.X_train_smote_shape = X_train.shape
        self.smote_counts = {
            "Legitimate (Class 0)": int(sum(y_train == 0)), 
            "Fraud (Class 1)": int(sum(y_train == 1))
        }
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict_fraud_likelihood(self, feature_vector):
        if not self.is_fitted:
            self.execute_model_training()
        vector = np.array([feature_vector])
        proba = self.model.predict_proba(vector)[0][1]
        return float(proba)

    def generate_documentation_figures(self, output_dir=r"C:\Users\Asus Tuf\Desktop\3rd Sem AI project"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        categories = ['Raw Ingested Rows', 'Cleaned Active Dataset', 'Dropped Null Rows']
        counts = [self.db_loader.raw_rows, self.db_loader.cleaned_rows, self.db_loader.dropped_rows]
        colors = ['#2b5c8f', '#27ae60', '#e74c3c']
        
        bars = ax.bar(categories, counts, color=colors, edgecolor='black', width=0.5)
        ax.set_ylabel('Record Count')
        ax.set_title('Figure A: Data Cleaning & Preprocessing Breakdown', fontweight='bold', pad=15)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_cleaning_comparison.png'), dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        labels = ['Class 0 (Legitimate)', 'Class 1 (Fraud)']
        pre_smote = [self.raw_counts["Legitimate (Class 0)"], self.raw_counts["Fraud (Class 1)"]]
        post_smote = [self.smote_counts["Legitimate (Class 0)"], self.smote_counts["Fraud (Class 1)"]]
        
        x = np.arange(len(labels))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, pre_smote, width, label='Pre-SMOTE (Raw Train)', color='#e74c3c', edgecolor='black')
        rects2 = ax.bar(x + width/2, post_smote, width, label='Post-SMOTE (Balanced Train)', color='#27ae60', edgecolor='black')
        
        ax.set_ylabel('Sample Count')
        ax.set_title('Figure B: Pre vs Post-SMOTE Class Distribution Comparison', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'smote_rebalance_comparison.png'), dpi=300)
        plt.close()

# =====================================================================
# 3. STATE TRAVERSAL & INFORMED A* SEARCH (WITH DYNAMIC THRESHOLD)
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
    def __init__(self, probability_score, decision_threshold=0.40, system_latency=45):
        self.p_fraud = probability_score
        self.threshold = decision_threshold
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
        if state == self.GOAL:
            return 0.0
        if state == self.BYPASS:
            if self.p_fraud > self.threshold:
                return float('inf')  
            return self.p_fraud * 200.0
        if state == self.CHALLENGE:
            return (1.0 - self.p_fraud) * 80.0
        if state == self.TERMINATE:
            if self.latency > 100:
                return float('inf')
            return (1.0 - self.p_fraud) * 600.0
        return 50.0

    def get_node_breakdown(self):
        nodes = [self.BYPASS, self.CHALLENGE, self.TERMINATE]
        breakdown = []
        for n in nodes:
            g = n.friction_cost
            h = self.evaluate_heuristic(n)
            f = g + h if h != float('inf') else float('inf')
            breakdown.append({
                "State Node": n.name,
                "Friction Cost g(n)": g,
                "Risk Heuristic h(n)": "∞ (Blocked)" if h == float('inf') else round(h, 2),
                "Total Cost f(n)": "∞ (Pruned)" if f == float('inf') else round(f, 2)
            })
        return pd.DataFrame(breakdown)

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
# 4. STREAMLIT FRONTEND
# =====================================================================

st.title("🛡️ Enterprise Hybrid AI Verification System")
st.markdown("**Dynamic Fraud Prevention and Friction-Optimized User Routing Platform")

st.divider()

# Cache Model
@st.cache_resource
def get_trained_classifier():
    clf = AcademicRiskClassifier()
    clf.execute_model_training()
    clf.generate_documentation_figures()
    return clf

with st.spinner("Initializing Storage Engine, Cleaning Data, & Fitting SMOTE Ensemble..."):
    classifier = get_trained_classifier()

# SIDEBAR CONTROLS & NEW THRESHOLD SLIDER
st.sidebar.header("🕹️ Transaction Simulator")

st.sidebar.markdown("### **System Policy Tuning**")
custom_threshold = st.sidebar.slider(
    "A* Dynamic Risk Threshold", 
    0.10, 0.90, 0.40, 0.05,
    help="Default is 0.40. Adjust to see how A* reroutes transactions under stricter/looser security policies."
)

st.sidebar.markdown("### **Quick Load Presets**")
col_p1, col_p2, col_p3 = st.sidebar.columns(3)

if col_p1.button("🟢 Safe"):
    st.session_state.time_val = 184.0
    st.session_state.amount_val = 75.10
    st.session_state.v1_val = -0.89
    st.session_state.v2_val = 0.21

if col_p2.button("🟡 Medium"):
    st.session_state.time_val = 312.0
    st.session_state.amount_val = 890.45
    st.session_state.v1_val = -2.01
    st.session_state.v2_val = 1.15

if col_p3.button("🔴 Fraud"):
    st.session_state.time_val = 406.0
    st.session_state.amount_val = 0.00
    st.session_state.v1_val = -2.31
    st.session_state.v2_val = 1.95

st.sidebar.markdown("---")
st.sidebar.markdown("### **Custom Parameter Vectors**")

time_input = st.sidebar.number_input("Time Elapsed (Seconds)", value=st.session_state.get('time_val', 184.0), step=10.0)
amount_input = st.sidebar.number_input("Transaction Amount ($)", value=st.session_state.get('amount_val', 75.10), step=10.0)
v1_input = st.sidebar.slider("Behavioral Indicator V1", -5.0, 5.0, st.session_state.get('v1_val', -0.89), 0.01)
v2_input = st.sidebar.slider("Behavioral Indicator V2", -5.0, 5.0, st.session_state.get('v2_val', 0.21), 0.01)

# Inference & A* Search with Dynamic Threshold
sample_vector = [time_input, amount_input, v1_input, v2_input]
p_fraud = classifier.predict_fraud_likelihood(sample_vector)
planner = InformedRoutePlanner(probability_score=p_fraud, decision_threshold=custom_threshold)
path, friction_cost = planner.run_pathfinding_optimization()
resolved_node = path[1].name if path else "N/A"

# Append to History Session State
history_entry = {
    "Time": time_input,
    "Amount": amount_input,
    "V1": v1_input,
    "V2": v2_input,
    "Threshold": custom_threshold,
    "P(Fraud)": round(p_fraud, 4),
    "Resolved Route": resolved_node,
    "Path Cost": friction_cost
}
if not st.session_state.audit_history or st.session_state.audit_history[-1] != history_entry:
    st.session_state.audit_history.append(history_entry)

# MAIN DASHBOARD TOP METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("Calculated P(Fraud)", f"{p_fraud * 100:.2f}%")
m2.metric("Target Route Node", resolved_node)
m3.metric("Path Friction Cost", f"{friction_cost:.1f}")
m4.metric("Active Threshold", f"{custom_threshold:.2f}")

# STATUS DECISION BANNER
if p_fraud > custom_threshold:
    st.error(
        f"🚨 **HIGH RISK DETECTED ($P_{{\\text{{Fraud}}}} = {p_fraud:.4f} > {custom_threshold:.2f}$)**\n\n"
        f"The Random Forest model flagged an anomaly above the active threshold ({custom_threshold:.2f}). "
        f"The $A^*$ Pathfinder warped the heuristic cost of `FRICTIONLESS_AUTO_CLEARANCE` to $\\infty$, routing to **{resolved_node}**."
    )
else:
    st.success(
        f"✅ **LEGITIMATE TRANSACTION SESSION ($P_{{\\text{{Fraud}}}} = {p_fraud:.4f} \\le {custom_threshold:.2f}$)**\n\n"
        f"The risk profile is within the safety threshold ({custom_threshold:.2f}). "
        f"The $A^*$ Pathfinder selected the lowest cost path directly to **{resolved_node}**."
    )

st.divider()

# TABS SECTION
tab_cleaning, tab_inspector, tab_analytics, tab_batch, tab_history = st.tabs([
    "🧹 Data Cleaning & SMOTE Audit",
    "🧮 A* Mathematical Inspector", 
    "📊 Model Analytics & Charts", 
    "📁 Batch CSV Tester",
    "📜 Session Audit History"
])

# TAB 1: DATA CLEANING & SMOTE
with tab_cleaning:
    st.subheader("Data Preprocessing & Balancing Documentation Metrics")
    col_dc1, col_dc2 = st.columns(2)
    with col_dc1:
        st.markdown("### 1. Data Ingestion & Cleaning Summary")
        cleaning_data = {
            "Metric Stage": ["Raw Ingested Records", "Cleaned Active Records", "Dropped Null Gaps"],
            "Row Count": [f"{classifier.db_loader.raw_rows:,}", f"{classifier.db_loader.cleaned_rows:,}", f"{classifier.db_loader.dropped_rows:,}"]
        }
        st.table(pd.DataFrame(cleaning_data))
        img_dir = r"C:\Users\Asus Tuf\Desktop\3rd Sem AI project"
        dc_img_path = os.path.join(img_dir, "data_cleaning_comparison.png")
        if os.path.exists(dc_img_path):
            st.image(dc_img_path, caption="Figure A: Data Cleaning Results", use_container_width=True)

    with col_dc2:
        st.markdown("### 2. Pre vs. Post-SMOTE Class Distribution")
        smote_summary = {
            "Class Category": ["Legitimate User (Class 0)", "Fraudulent Incident (Class 1)", "Total Training Array"],
            "Pre-SMOTE (Raw Partition)": [f"{classifier.raw_counts['Legitimate (Class 0)']:,}", f"{classifier.raw_counts['Fraud (Class 1)']:,}", f"{classifier.X_train_raw_shape[0]:,}"],
            "Post-SMOTE (Balanced RAM)": [f"{classifier.smote_counts['Legitimate (Class 0)']:,}", f"{classifier.smote_counts['Fraud (Class 1)']:,}", f"{classifier.X_train_smote_shape[0]:,}"]
        }
        st.table(pd.DataFrame(smote_summary))
        smote_img_path = os.path.join(img_dir, "smote_rebalance_comparison.png")
        if os.path.exists(smote_img_path):
            st.image(smote_img_path, caption="Figure B: Class Rebalancing via SMOTE", use_container_width=True)

# TAB 2: A* INSPECTOR
with tab_inspector:
    st.subheader("Step-by-Step State Graph Cost Breakdown")
    node_df = planner.get_node_breakdown()
    st.dataframe(node_df, use_container_width=True)
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### **Heuristic Evaluation Logic**")
        st.latex(rf"h(\text{{BYPASS}}) = \begin{{cases}} \infty & \text{{if }} P(\text{{Fraud}}) > {custom_threshold:.2f} \\ P(\text{{Fraud}}) \times 200 & \text{{otherwise}} \end{cases}")
        st.latex(r"h(\text{CHALLENGE}) = (1 - P(\text{Fraud})) \times 80")
    with col_g2:
        st.markdown("#### **Current Active Values**")
        st.write(f"* **Current $P(\\text{{Fraud}}):$** `{p_fraud:.4f}`")
        st.write(f"* **Active Threshold:** `{custom_threshold:.2f}`")
        st.write(f"* **Auto-Bypass Cost $f(\\text{{BYPASS}}):$** `{node_df.iloc[0]['Total Cost f(n)']}`")
        st.write(f"* **MFA Challenge Cost $f(\\text{{CHALLENGE}}):$** `{node_df.iloc[1]['Total Cost f(n)']}`")

# TAB 3: MODEL ANALYTICS & CHARTS
with tab_analytics:
    st.subheader("Performance Metrics & Visualizations")
    img_dir = r"C:\Users\Asus Tuf\Desktop\3rd Sem AI project"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### **Figure 1: Post-SMOTE Confusion Matrix**")
        cm_path = os.path.join(img_dir, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
    with c2:
        st.markdown("#### **Figure 2: Relative Feature Importances**")
        fi_path = os.path.join(img_dir, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### **Table B.1: Predictive Metrics Matrix**")
        b1_path = os.path.join(img_dir, "table_b1.png")
        if os.path.exists(b1_path):
            st.image(b1_path, use_container_width=True)
    with c4:
        st.markdown("#### **Table B.2: Traversal & Latency Log**")
        b2_path = os.path.join(img_dir, "table_b2.png")
        if os.path.exists(b2_path):
            st.image(b2_path, use_container_width=True)

# NEW TAB 4: BATCH CSV TESTER
with tab_batch:
    st.subheader("📁 Batch Transaction Verification Suite")
    st.markdown("Upload a custom CSV file containing columns `Time`, `Amount`, `V1`, `V2` to evaluate multiple transactions simultaneously through the Hybrid AI pipeline.")
    
    uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        required = ['Time', 'Amount', 'V1', 'V2']
        if all(col in batch_df.columns for col in required):
            results = []
            for idx, row in batch_df.iterrows():
                vec = [row['Time'], row['Amount'], row['V1'], row['V2']]
                prob = classifier.predict_fraud_likelihood(vec)
                plan = InformedRoutePlanner(probability_score=prob, decision_threshold=custom_threshold)
                pth, cst = plan.run_pathfinding_optimization()
                results.append({
                    "Row ID": idx + 1,
                    "Amount ($)": row['Amount'],
                    "P(Fraud)": round(prob, 4),
                    "Action Route": pth[1].name if pth else "N/A",
                    "Path Cost": cst
                })
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
        else:
            st.error(f"Uploaded CSV must contain all required columns: {required}")

# TAB 5: SESSION AUDIT HISTORY
with tab_history:
    st.subheader("Live Tested Transaction History")
    history_df = pd.DataFrame(st.session_state.audit_history)
    st.dataframe(history_df, use_container_width=True)
    
    csv_buffer = io.StringIO()
    history_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Session Audit Log (CSV)",
        data=csv_buffer.getvalue(),
        file_name="hybrid_ai_session_audit.csv",
        mime="text/csv"
    )