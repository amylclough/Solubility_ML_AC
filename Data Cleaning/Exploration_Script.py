import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from io import BytesIO
from scipy.stats import norm
from rdkit import Chem
from rdkit.Chem import Descriptors

# --- 1. CONFIGURATION ---
INPUT_FILE = 'cleaned_solubility.csv'
OUTPUT_DIR = 'plots'
OUTPUT_CSV = f'{OUTPUT_DIR}/features_included.csv'
HTML_REPORT = f'{OUTPUT_DIR}/exploration_report.html'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calculate_ap(mol):
    if mol is None: return 0.0
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0: return 0.0
    aromatic = len([at for at in mol.GetAtoms() if at.GetIsAromatic()])
    return aromatic / heavy

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Your requested formatting logic adapted for HTML export (No titles)
def create_distribution_plot(y, xlabel, color, normal_overlay=False):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Base histogram (normalised) 
    sns.histplot(y, kde=False, stat='density', bins=30, color=color, alpha=0.6, ax=ax)
    
    mean = np.mean(y)
    median = np.median(y)
    std = np.std(y)
    n = len(y)
    
    # Normal distribution overlay logic
    if normal_overlay and std > 0:
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std)
        ax.plot(x, p, color=color, linewidth=2, label='Normal Distribution')

    # Add vertical lines for mean and median
    ax.axvline(mean, color='black', linestyle='--', linewidth=1.2, label=f"Mean = {mean:.2f}")
    ax.axvline(median, color='green', linestyle=':', linewidth=1.2, label=f"Median = {median:.2f}")

    # Set specific labels (No titles as requested)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalised frequency')

    # Annotate with text box 
    textstr = '\n'.join((
        f'n = {n}',
        f'Mean = {mean:.3f}',
        f'Median = {median:.3f}',
        f'SD = {std:.3f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.legend()
    plt.tight_layout()
    return fig_to_base64(fig)


# --- 2. DATA PROCESSING & FEATURE ENGINEERING ---
print("Loading data and calculating features...")
df = pd.read_csv(INPUT_FILE).dropna(subset=['SMILES', 'LogS_Median'])
results = []

for _, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    if not mol: continue
    
    # Standard Descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    mr = Descriptors.MolMR(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    ap = calculate_ap(mol)
    
    # Standard 4 Lipinski Rules
    rules_failed = 0
    if mw > 500: rules_failed += 1
    if logp > 5: rules_failed += 1
    if hbd > 5: rules_failed += 1
    if hba > 10: rules_failed += 1
    
    logs = row['LogS_Median']
    in_target_range = -5 <= logs <= -1

    results.append({
        'SMILES': row['SMILES'], 'LogS': logs, 
        'MW': mw, 'LogP': logp, 'MolMR': mr, 'HBD': hbd, 'HBA': hba,
        'TPSA': tpsa, 'RB': rb, 'AP': ap, 
        'Lipinski_Violations': rules_failed,
        'Target_LogS_Range': in_target_range
    })

feat_df = pd.DataFrame(results)
feat_df.to_csv(OUTPUT_CSV, index=False)

# --- 3. STATISTICS & METRICS ---
print("Calculating statistics and correlations...")
total_mols = len(feat_df)
logs_target_count = feat_df['Target_LogS_Range'].sum()
logs_target_perc = (logs_target_count / total_mols) * 100

lipinski_0_violations = len(feat_df[feat_df['Lipinski_Violations'] == 0])
lipinski_1_violation = len(feat_df[feat_df['Lipinski_Violations'] == 1])

lipinski_0_perc = (lipinski_0_violations / total_mols) * 100
lipinski_1_perc = (lipinski_1_violation / total_mols) * 100

numerical_cols = ['LogS', 'MW', 'LogP', 'MolMR', 'HBD', 'HBA', 'TPSA', 'RB', 'AP']

# --- 4. VISUALIZATIONS ---
print("Generating visualizations...")
plots_b64 = {}

# A. Histograms using requested format
features_to_plot = [
    ('LogS', '#3498db', True), 
    ('LogP', '#2ecc71', True), 
    ('MW', '#9b59b6', True), 
    ('AP', '#e74c3c', True), 
    ('RB', '#f1c40f', False), 
    ('TPSA', '#e67e22', True), 
    ('MolMR', '#1abc9c', True), 
    ('HBA', '#34495e', False), 
    ('HBD', '#95a5a6', False)
]

for feat, color, overlay in features_to_plot:
    plots_b64[feat] = create_distribution_plot(feat_df[feat].dropna(), feat, color, normal_overlay=overlay)

# B. Correlation Matrix (No Title)
fig_corr, ax = plt.subplots(figsize=(10, 8))
corr_matrix = feat_df[numerical_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, 
            mask=mask, square=True, linewidths=.5, ax=ax, annot_kws={"size": 10})
plt.tight_layout()
plots_b64['corr_matrix'] = fig_to_base64(fig_corr)

# C. Lipinski Violin Plot (No Title)
fig_vio, ax = plt.subplots(figsize=(8, 6))
feat_df['Lipinski_Status'] = feat_df['Lipinski_Violations'].apply(lambda x: '0 Violations' if x == 0 else ('1 Violation' if x == 1 else '>1 Violations'))
sns.violinplot(
    data=feat_df, x='Lipinski_Status', y='LogS', hue='Lipinski_Status', 
    order=['0 Violations', '1 Violation', '>1 Violations'],
    palette={'0 Violations': '#2ecc71', '1 Violation': '#f1c40f', '>1 Violations': '#e74c3c'}, 
    legend=False, ax=ax
)
ax.set_xlabel('Lipinski Violations')
ax.set_ylabel('LogS')
plt.tight_layout()
plots_b64['lipinski_vio'] = fig_to_base64(fig_vio)


# --- 5. HTML REPORT GENERATION ---
print("Building HTML Report...")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Exploration Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f4f7f9; color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .summary-box {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #3498db; font-size: 16px; line-height: 1.6;}}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; }}
        .grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Initial Exploration Report</h1>
        
        <div class="summary-box">
            <b>Key Metrics:</b><br>
            • Total Molecules: <b>{total_mols}</b><br>
            • Molecules in -1 to -5 LogS Range: <b>{logs_target_perc:.1f}%</b> ({logs_target_count})<br>
            • Molecules with 0 Lipinski Violations: <b>{lipinski_0_perc:.1f}%</b> ({lipinski_0_violations})<br>
            • Molecules with exactly 1 Lipinski Violation: <b>{lipinski_1_perc:.1f}%</b> ({lipinski_1_violation})
        </div>

        <h2>Feature Distributions</h2>
        <div class="grid-container">
            <div><img src="data:image/png;base64,{plots_b64['LogS']}"></div>
            <div><img src="data:image/png;base64,{plots_b64['LogP']}"></div>
            <div><img src="data:image/png;base64,{plots_b64['MW']}"></div>
            <div><img src="data:image/png;base64,{plots_b64['AP']}"></div>
            <div><img src="data:image/png;base64,{plots_b64['RB']}"></div>
            <div><img src="data:image/png;base64,{plots_b64['TPSA']}"></div>
            <div><img src="data:image/png;base64,{plots_b64['MolMR']}"></div>
            <div><img src="data:image/png;base64,{plots_b64['HBA']}"></div>
            <div><img src="data:image/png;base64,{plots_b64['HBD']}"></div>
        </div>

        <h2>Lipinski Violations vs LogS</h2>
        <img src="data:image/png;base64,{plots_b64['lipinski_vio']}" style="max-width: 700px;">
        
        <h2>Correlation Matrix</h2>
        <img src="data:image/png;base64,{plots_b64['corr_matrix']}" style="max-width: 900px;">
        
    </div>
</body>
</html>
"""

with open(HTML_REPORT, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\nSUCCESS! Workflow complete. HTML report generated: {HTML_REPORT}")