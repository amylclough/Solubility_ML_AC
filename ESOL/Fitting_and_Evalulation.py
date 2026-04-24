import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import base64
from io import BytesIO

# --- 1. SETUP & DEFINITIONS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, 'train_data_raw.csv')
test_path = os.path.join(script_dir, 'test_data_raw.csv')
html_report_path = os.path.join(script_dir, 'ESOL_Report_with_Outliers.html')

def calculate_esol_features(smiles):
    """Calculates the 4 ESOL descriptors using RDKit."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None: return None
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        rb = Descriptors.NumRotatableBonds(mol)
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        heavy_atoms = mol.GetNumHeavyAtoms()
        ap = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0
        return [logp, mw, rb, ap]
    except:
        return None

def process_dataset(filepath):
    """Loads CSV, calculates features, drops invalid SMILES."""
    dataframe = pd.read_csv(filepath)
    features, valid_indices = [], []
    for index, row in dataframe.iterrows():
        result = calculate_esol_features(row['SMILES'])
        if result:
            features.append(result)
            valid_indices.append(index)
    
    df_clean = dataframe.loc[valid_indices].copy()
    feature_df = pd.DataFrame(features, columns=['LogP', 'MW', 'RotatableBonds', 'AromaticProportion'], index=df_clean.index)
    return pd.concat([df_clean, feature_df], axis=1)

def original_esol_equation(logp, mw, rb, ap):
    """The original Delaney ESOL equation."""
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap

def calculate_within_stats(y_true, y_pred):
    """Calculates percentage of predictions within 0.7 and 1.0 log units."""
    errors = np.abs(y_true - y_pred)
    w07 = (errors <= 0.7).sum() / len(errors) * 100
    w10 = (errors <= 1.0).sum() / len(errors) * 100
    return w07, w10

def get_base64_plot(figure):
    """Converts a matplotlib figure to a base64 string."""
    buffer = BytesIO()
    figure.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(figure)
    return img_str

def get_base64_mol_image(smiles):
    """Generates a 2D RDKit image to base64."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None: return ""
        img = Draw.MolToImage(mol, size=(200, 200))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except:
        return ""

# --- 2. DATA PROCESSING & TRAINING ---
print("Processing training data...")
df_train = process_dataset(train_path)
print("Processing testing data...")
df_test = process_dataset(test_path)

feature_cols = ['LogP', 'MW', 'RotatableBonds', 'AromaticProportion']
X_train, y_train = df_train[feature_cols], df_train['LogS_Median']
X_test, y_test = df_test[feature_cols], df_test['LogS_Median']

# Fit New Model
model = LinearRegression()
model.fit(X_train, y_train)

# --- 3. PREDICTIONS & METRICS ---
y_pred_new = model.predict(X_test)
r2_new = r2_score(y_test, y_pred_new)
rmse_new = np.sqrt(mean_squared_error(y_test, y_pred_new))
mae_new = mean_absolute_error(y_test, y_pred_new)
w07_new, w10_new = calculate_within_stats(y_test, y_pred_new)

y_pred_orig = original_esol_equation(X_test['LogP'], X_test['MW'], X_test['RotatableBonds'], X_test['AromaticProportion'])
r2_orig = r2_score(y_test, y_pred_orig)
rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_orig))
mae_orig = mean_absolute_error(y_test, y_pred_orig)
w07_orig, w10_orig = calculate_within_stats(y_test, y_pred_orig)

# --- 4. OUTLIERS ---
df_test['Predicted_LogS'] = y_pred_new
df_test['Absolute_Error'] = np.abs(df_test['LogS_Median'] - df_test['Predicted_LogS'])
top_10_outliers = df_test.nlargest(10, 'Absolute_Error')[['SMILES', 'LogS_Median', 'Predicted_LogS', 'Absolute_Error']].copy()

image_tags = []
for smiles in top_10_outliers['SMILES']:
    b64_img = get_base64_mol_image(smiles)
    image_tags.append(f'<img src="data:image/png;base64,{b64_img}" width="150"/>' if b64_img else 'Failed')

top_10_outliers.insert(0, 'Structure', image_tags)
outliers_html_table = top_10_outliers.to_html(index=False, float_format=lambda value: f"{value:.3f}", border=0, escape=False)

# --- 5. VISUALIZATIONS ---
sns.set_style("whitegrid")
plot_low = min(y_test.min(), min(y_pred_new.min(), y_pred_orig.min()))
plot_high = max(y_test.max(), max(y_pred_new.max(), y_pred_orig.max()))

# Helper for consistent plotting (Updated to accept a dynamic color parameter)
def finalize_plot(axis, true_values, pred_values, plot_color):
    axis.scatter(true_values, pred_values, alpha=0.5, marker='o', s=60, color=plot_color)
    axis.plot([plot_low, plot_high], [plot_low, plot_high], 'r--')
    axis.fill_between([plot_low, plot_high], [plot_low-0.7, plot_high-0.7], [plot_low+0.7, plot_high+0.7], color='green', alpha=0.05)
    axis.set_xlabel('Experimental LogS')
    axis.set_ylabel('Predicted LogS')

# 5a. New Model Plot
fig_new, ax_new = plt.subplots(figsize=(7, 6))
finalize_plot(ax_new, y_test, y_pred_new, plot_color='purple')
plt.tight_layout()
fig_new.savefig(os.path.join(script_dir, 'scatter_new_model.png'), dpi=150)
plot_new_b64 = get_base64_plot(fig_new)

# 5b. Original ESOL Plot
fig_orig, ax_orig = plt.subplots(figsize=(7, 6))
finalize_plot(ax_orig, y_test, y_pred_orig, plot_color='steelblue')
plt.tight_layout()
fig_orig.savefig(os.path.join(script_dir, 'scatter_original_model.png'), dpi=150)
plot_orig_b64 = get_base64_plot(fig_orig)

# 5c. Distribution plot
fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
sns.kdeplot(y_pred_new - y_test, fill=True, label='New Model', color='purple', ax=ax_dist)
sns.kdeplot(y_pred_orig - y_test, fill=True, label='Original Equation', color='steelblue', ax=ax_dist)
ax_dist.axvline(0, color='black', linestyle='--')
ax_dist.set_xlabel('Prediction Error (Predicted - Experimental)')
ax_dist.legend()
plt.tight_layout()
fig_dist.savefig(os.path.join(script_dir, 'error_distribution.png'), dpi=150)
plot_dist_b64 = get_base64_plot(fig_dist)

# --- 6. HTML REPORT ---
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: sans-serif; margin: 40px; background: #f4f7f6; }}
        .container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .table-container {{ overflow-x: auto; width: 100%; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; min-width: 600px; }}
        th, td {{ padding: 12px; border: 1px solid #eee; text-align: center; }}
        th {{ background: #2c3e50; color: white; }}
        .equation {{ background: #eef; padding: 15px; border-left: 5px solid #2c3e50; font-family: monospace; }}
        .plot-container {{ display: flex; gap: 20px; margin-top: 20px; margin-bottom: 20px; }}
        .plot-box {{ flex: 1; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ESOL Comparison Report</h1>
        <div class="equation">New Fit: {model.intercept_:.3f} + ({model.coef_[0]:.3f})LogP + ({model.coef_[1]:.3f})MW + ({model.coef_[2]:.3f})RB + ({model.coef_[3]:.3f})AP</div>
        
        <h2>Metrics Table</h2>
        <div class="table-container">
            <table>
                <tr><th>Metric</th><th>New Model</th><th>Original ESOL</th></tr>
                <tr><td>R² Score</td><td>{r2_new:.4f}</td><td>{r2_orig:.4f}</td></tr>
                <tr><td>RMSE</td><td>{rmse_new:.4f}</td><td>{rmse_orig:.4f}</td></tr>
                <tr><td>MAE</td><td>{mae_new:.4f}</td><td>{mae_orig:.4f}</td></tr>
                <tr><td><strong>% Within 0.7</strong></td><td><strong>{w07_new:.1f}%</strong></td><td><strong>{w07_orig:.1f}%</strong></td></tr>
                <tr><td><strong>% Within 1.0</strong></td><td><strong>{w10_new:.1f}%</strong></td><td><strong>{w10_orig:.1f}%</strong></td></tr>
            </table>
        </div>

        <h2>Visual Analysis</h2>
        <div class="plot-container">
            <div class="plot-box">
                <h3>New Fitted Model</h3>
                <img src="data:image/png;base64,{plot_new_b64}" width="100%">
            </div>
            <div class="plot-box">
                <h3>Original ESOL</h3>
                <img src="data:image/png;base64,{plot_orig_b64}" width="100%">
            </div>
        </div>
        
        <div style="text-align: center;">
            <h3>Error Distribution</h3>
            <img src="data:image/png;base64,{plot_dist_b64}" width="80%">
        </div>

        <h2>Outlier Analysis</h2>
        <div class="table-container">
            {outliers_html_table}
        </div>
    </div>
</body>
</html>
"""

with open(html_report_path, 'w', encoding='utf-8') as html_file:
    html_file.write(html_content)

print(f"Success! Report generated at: {html_report_path}")
print("Saved scatter_new_model.png, scatter_original_model.png, and error_distribution.png to disk.")