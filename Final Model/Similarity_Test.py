import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

# --- Configuration ---
TRAIN_FILE = 'train_data_final.csv'
TEST_FILE = 'test_data_final.csv'
MODEL_FILE = 'trained_model.joblib'
SCALER_FILE = 'scaler_final.joblib'
LINKS_FILE = 'path/to/DrugBankStructurelinks'
HTML_OUT = 'Similarity_AD_Report.html'
PLOTS_DIR = 'similarity_plots'

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def save_and_get_b64(fig, filename):
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, format='png', bbox_inches='tight', dpi=120)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def calculate_bin_stats(df, sim_col):
    df = df.copy() 
    bins = [0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["<40%", "40-60%", "60-70%", "70-80%", "80-90%", "90%+"]
    df['bin'] = pd.cut(df[sim_col], bins=bins, labels=labels, right=False)
    
    stats = []
    for label in labels:
        subset = df[df['bin'] == label]
        if len(subset) > 0: 
            pct = np.mean(subset['abs_err'] <= 0.7) * 100
            rmse = np.sqrt(mean_squared_error(subset['LogS_Median'], subset['y_pred']))
            r2 = r2_score(subset['LogS_Median'], subset['y_pred']) if len(subset) > 1 else np.nan
            stats.append({'Bin': label, 'N': len(subset), 'R2': max(0, r2) if not np.isnan(r2) else np.nan, 'RMSE': rmse, 'Pct_07': pct})
        else:
            stats.append({'Bin': label, 'N': 0, 'R2': np.nan, 'RMSE': np.nan, 'Pct_07': np.nan})
    return pd.DataFrame(stats)

def generate_single_bar_plot(plot_data, x_col, y_col, color, y_lim_tuple, filename, is_pct=False):
    if plot_data.empty: return ""
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=plot_data, x=x_col, y=y_col, color=color, ax=ax)
    
    # Add some headroom for the text labels
    ax.set_ylim(y_lim_tuple[0], y_lim_tuple[1] * 1.15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # Add values on top of the bars
    for i, row in plot_data.reset_index(drop=True).iterrows():
        val = row[y_col]
        if pd.notna(val):
            text_val = f"{val:.1f}%" if is_pct else f"{val:.3f}"
            # Position text slightly above the bar
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.text(i, val + offset, text_val, ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)
            
    plt.tight_layout()
    return save_and_get_b64(fig, filename)

def generate_density_plot(df, sim_col, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    high = df[df[sim_col] > 0.8]
    med = df[(df[sim_col] <= 0.8) & (df[sim_col] > 0.6)]
    low = df[df[sim_col] <= 0.6]
    
    if len(high) > 2: sns.kdeplot(high['raw_err'], fill=True, color='#2ecc71', label='High Sim (>80%)', alpha=0.4, ax=ax)
    if len(med) > 2: sns.kdeplot(med['raw_err'], fill=True, color='#f1c40f', label='Med Sim (60-80%)', alpha=0.4, ax=ax)
    if len(low) > 2: sns.kdeplot(low['raw_err'], fill=True, color='#e74c3c', label='Low Sim (<60%)', alpha=0.4, ax=ax)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Prediction Error (Pred - Actual)"); ax.set_xlim(-3, 3)
    ax.legend()
    plt.tight_layout()
    return save_and_get_b64(fig, filename)

def main():
    print("Loading Data & Model...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    features = [c for c in train_df.columns if c not in ['SMILES', 'LogS_Median']]
    maccs_feats = [f for f in features if f.startswith('MACCS_')]
    cont_feats = [f for f in features if not f.startswith('MACCS_')]

    # Predict
    X_test_sc = scaler.transform(test_df[features].values)
    
    new_cols = {}
    new_cols['y_pred'] = model.predict(X_test_sc)
    new_cols['abs_err'] = np.abs(test_df['LogS_Median'] - new_cols['y_pred'])
    new_cols['raw_err'] = new_cols['y_pred'] - test_df['LogS_Median']

    # Drug Matching
    links_df = pd.read_csv(LINKS_FILE)
    key_map = dict(zip(links_df['InChIKey'], links_df['Name']))
    new_cols['InChIKey'] = test_df['SMILES'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None)
    new_cols['Drug_Name'] = new_cols['InChIKey'].map(key_map)
    new_cols['is_drug'] = new_cols['Drug_Name'].notna()

    print("Calculating Similarities...")
    
    # 1. Continuous Descriptor Similarity
    sim_scaler = StandardScaler()
    train_cont_sc = sim_scaler.fit_transform(train_df[cont_feats].values)
    test_cont_sc = sim_scaler.transform(test_df[cont_feats].values)
    
    nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(train_cont_sc)
    dists, _ = nbrs.kneighbors(test_cont_sc)
    
    normalized_dists = dists / np.sqrt(len(cont_feats))
    new_cols['sim_desc'] = np.mean(1 / (1 + normalized_dists), axis=1)

    # 2. MACCS Fingerprint Similarity
    train_maccs = train_df[maccs_feats].values.astype(bool)
    test_maccs = test_df[maccs_feats].values.astype(bool)
    jaccard_dists = pairwise_distances(test_maccs, train_maccs, metric='jaccard')
    tanimoto_sims = 1 - jaccard_dists
    
    top5_sims = np.sort(tanimoto_sims, axis=1)[:, -5:]
    new_cols['sim_fp'] = np.mean(top5_sims, axis=1)

    # 3. Average Similarity
    new_cols['sim_avg'] = (new_cols['sim_desc'] + new_cols['sim_fp']) / 2

    # Combine
    new_cols_df = pd.DataFrame(new_cols, index=test_df.index)
    test_df = pd.concat([test_df, new_cols_df], axis=1)

    print("Generating Bin Statistics & Plots...")
    stats_avg = calculate_bin_stats(test_df, 'sim_avg')
    stats_fp = calculate_bin_stats(test_df, 'sim_fp')
    stats_desc = calculate_bin_stats(test_df, 'sim_desc')

    # Generate individual plots instead of grouped plots
    def gen_three_bars(stats, prefix):
        data = stats[stats['N'] > 1].copy()
        r2_img = generate_single_bar_plot(data, 'Bin', 'R2', '#3498db', (0, 1.0), f"{prefix}_r2.png")
        rmse_img = generate_single_bar_plot(data, 'Bin', 'RMSE', '#e74c3c', (0, max(2.5, data['RMSE'].max() if not data.empty else 1)), f"{prefix}_rmse.png")
        pct_img = generate_single_bar_plot(data, 'Bin', 'Pct_07', '#2ecc71', (0, 100), f"{prefix}_pct.png", is_pct=True)
        return r2_img, rmse_img, pct_img

    b64_avg_r2, b64_avg_rmse, b64_avg_pct = gen_three_bars(stats_avg, "avg_sim")
    b64_fp_r2, b64_fp_rmse, b64_fp_pct = gen_three_bars(stats_fp, "fp_sim")
    b64_desc_r2, b64_desc_rmse, b64_desc_pct = gen_three_bars(stats_desc, "desc_sim")

    b64_dens_avg = generate_density_plot(test_df, 'sim_avg', "density_avg_sim.png")
    b64_dens_fp = generate_density_plot(test_df, 'sim_fp', "density_fp_sim.png")
    b64_dens_desc = generate_density_plot(test_df, 'sim_desc', "density_desc_sim.png")

    print("Extracting Targeted Drug Simlarities...")
    drug_df = test_df[test_df['is_drug']].sort_values('abs_err').reset_index(drop=True)
    best_idx = list(range(0, min(7, len(drug_df))))
    worst_idx = list(range(max(0, len(drug_df)-7), len(drug_df)))
    mid_start = max(0, (len(drug_df)//2) - 3)
    mid_idx = list(range(mid_start, min(len(drug_df), mid_start+7)))

    def build_drug_rows(indices, label, color):
        rows = ""
        for i in indices:
            row = drug_df.iloc[i]
            rows += f"<tr><td><b><span style='color:{color}'>{label}</span>: {row['Drug_Name']}</b></td><td>{row['abs_err']:.3f}</td><td>{row['sim_avg']*100:.1f}%</td><td>{row['sim_fp']*100:.1f}%</td><td>{row['sim_desc']*100:.1f}%</td></tr>"
        return rows

    drug_html = build_drug_rows(best_idx, "TOP 7", "#2ecc71") + build_drug_rows(mid_idx, "MEDIAN", "#f39c12") + build_drug_rows(worst_idx, "WORST", "#e74c3c")

    print("Saving HTML Report...")
    html = f"""
    <html><head><meta charset="UTF-8"><style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f4f7f9; color: #333; margin: 0; padding: 30px; }}
        h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.95em; }}
        th, td {{ padding: 12px; border-bottom: 1px solid #eee; text-align: center; }}
        th {{ background: #2c3e50; color: white; }}
        td:first-child {{ text-align: left; }}
        .flex-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; align-items: flex-start; margin-bottom: 20px; }}
        .plot-box {{ flex: 1; min-width: 300px; text-align: center; background: #fff; padding: 15px; border-radius: 8px; border: 1px solid #eee; }}
        .plot-box h4 {{ margin-top: 0; margin-bottom: 10px; color: #333; }}
        .dens-box {{ width: 100%; max-width: 800px; margin: 0 auto; display: block; text-align: center; }}
        img {{ max-width: 100%; height: auto; }}
    </style></head><body>

    <h1>Applicability Domain & Similarity Report</h1>
    <p>Evaluating test set performance against the 5-Nearest Neighbors in the Training Set.</p>

    <div class="section">
        <h2>1. Average Similarity (Fingerprints + Descriptors)</h2>
        <div class="flex-container">
            <div class="plot-box"><h4>R² Score</h4><img src="data:image/png;base64,{b64_avg_r2}"></div>
            <div class="plot-box"><h4>RMSE</h4><img src="data:image/png;base64,{b64_avg_rmse}"></div>
            <div class="plot-box"><h4>% Error ≤ 0.7</h4><img src="data:image/png;base64,{b64_avg_pct}"></div>
        </div>
        <div class="dens-box"><img src="data:image/png;base64,{b64_dens_avg}"></div>
        <div style="margin-top: 20px;">
            {stats_avg.to_html(index=False, classes='table', float_format="%.3f", na_rep="-")}
        </div>
    </div>

    <div class="section">
        <h2>2. MACCS Keys Fingerprint Similarity (Tanimoto)</h2>
        <div class="flex-container">
            <div class="plot-box"><h4>R² Score</h4><img src="data:image/png;base64,{b64_fp_r2}"></div>
            <div class="plot-box"><h4>RMSE</h4><img src="data:image/png;base64,{b64_fp_rmse}"></div>
            <div class="plot-box"><h4>% Error ≤ 0.7</h4><img src="data:image/png;base64,{b64_fp_pct}"></div>
        </div>
        <div class="dens-box"><img src="data:image/png;base64,{b64_dens_fp}"></div>
        <div style="margin-top: 20px;">
            {stats_fp.to_html(index=False, classes='table', float_format="%.3f", na_rep="-")}
        </div>
    </div>

    <div class="section">
        <h2>3. Descriptor Similarity (Continuous Features Euclidean)</h2>
        <div class="flex-container">
            <div class="plot-box"><h4>R² Score</h4><img src="data:image/png;base64,{b64_desc_r2}"></div>
            <div class="plot-box"><h4>RMSE</h4><img src="data:image/png;base64,{b64_desc_rmse}"></div>
            <div class="plot-box"><h4>% Error ≤ 0.7</h4><img src="data:image/png;base64,{b64_desc_pct}"></div>
        </div>
        <div class="dens-box"><img src="data:image/png;base64,{b64_dens_desc}"></div>
        <div style="margin-top: 20px;">
            {stats_desc.to_html(index=False, classes='table', float_format="%.3f", na_rep="-")}
        </div>
    </div>

    <div class="section">
        <h2>4. Drug Inspector: Similarity Deep-Dive</h2>
        <p>Similarity breakdown for the 7 Best, Median, and Worst predicted approved drugs in the test set.</p>
        <table>
            <tr><th>Drug Name</th><th>Absolute Error</th><th>Average Sim</th><th>MACCS Sim</th><th>Descriptor Sim</th></tr>
            {drug_html}
        </table>
    </div>

    </body></html>
    """
    with open(HTML_OUT, "w", encoding='utf-8') as f: f.write(html)
    print(f"Done! Dashboard saved to {HTML_OUT}")

if __name__ == "__main__":
    main()