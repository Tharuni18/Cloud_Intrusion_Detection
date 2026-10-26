# Hybrid ML Intrusion Detection for Cloud Environments

Detects cyberattacks (DDoS, port scans, brute-force / unauthorised access, data exfiltration,
botnet C2) with a **hybrid of supervised and unsupervised models**:

| Component | Role |
|---|---|
| Random Forest, XGBoost | Multi-class classifiers for *known* attack patterns |
| Autoencoder (benign-only) | Flags traffic that reconstructs poorly -> *novel* attacks |
| **Hybrid-OR** | Alert if classifier OR autoencoder fires; alert tagged `known-attack` / `novel-anomaly` |
| **Hybrid-Stacked** | Autoencoder reconstruction error fed to the classifier as an extra feature |
| Rule baseline | Frozen thresholds on rate/flag features, mimicking a static rule set |

## Quick start
```bash
pip install -r requirements.txt
python run_experiment.py --dataset synthetic --n-samples 40000          # pipeline demo
python run_experiment.py --dataset cicids2017 --data-path ./data/MachineLearningCVE
python run_experiment.py --dataset unsw-nb15  --data-path ./data/UNSW-NB15
python predict.py --model results/model.joblib --input flows.csv        # score new flows
```
Outputs land in `results/`: `report.md`, CSV tables, `figures/*.png`, `model.joblib`.
Useful flags: `--fast`, `--holdouts botnet,dos`, `--ae-percentile 99`, `--cap-per-class 40000`,
`--skip-zero-day`.

## Datasets
* **CIC-IDS2017** (download the *MachineLearningCVE* CSVs) - labels mapped to families: dos, portscan, bruteforce, webattack, botnet, infiltration, heartbleed.
* **UNSW-NB15** (`UNSW_NB15_training-set.csv` + `testing-set.csv`) - attack categories kept as-is.
* **synthetic** - cloud-flavoured generator for development; **not** a substitute for real data.

Loaders strip identifiers (IPs, timestamps, flow IDs), remove inf/NaN, **de-duplicate**, and cap rows per class for memory.

## Evaluation protocol (what makes the results defensible)
1. De-duplicate, then stratified 60/20/20 train/val/test split (no row in two splits).
2. Preprocessor (signed-log + z-score) fit on **benign train only**.
3. AE threshold = p99 of benign **validation** error (target ~1% FPR, calibrated without attacks).
4. **Closed-world**: precision / recall / F1 / **FPR** / PR-AUC / ROC-AUC, per-class recall, confusion matrix.
5. **Zero-day test**: each attack family is removed from training *and* validation, models are retrained, and recall on that unseen family is reported.
6. Ablation by construction: Rules vs RF vs XGB vs AE vs both hybrids.
7. Latency (batch throughput and single-flow ms) and permutation feature importance.

## Project layout
```
cloudids/data.py         loaders, synthetic generator, splitting
cloudids/models.py       preprocessor, rules, supervised, autoencoder, hybrids
cloudids/metrics.py      metrics, permutation importance
cloudids/plots.py        figures
cloudids/experiments.py  closed-world / zero-day / latency / report
run_experiment.py        CLI        predict.py   inference CLI        tests/  smoke tests
```

## Backends and known limitations
* If `xgboost` / `torch` are not installed, the code falls back to scikit-learn HistGradientBoosting / MLP autoencoder and **records the backend used in `report.md`**. Install both for final results. The PyTorch autoencoder path was written but could not be executed in the development sandbox (no torch there), so run it once and check.
* The bundled `results/` come from the synthetic generator with the fallback backends. They validate the code, **not** real-world performance. Do not quote them as findings.
* No dataset here is truly cloud-native; say so in your write-up. Cross-dataset testing (train CIC-IDS2017, test UNSW-NB15) is not implemented because the feature schemas differ and need a manual mapping.
* Rules baseline is a simplified stand-in, not Snort/Suricata.
* Permutation importance is used instead of SHAP to avoid a dependency; SHAP's `TreeExplainer` is a straightforward add-on for the XGBoost model.


## Web frontend (Stitch design)

This project includes a responsive dark cybersecurity dashboard based on the supplied Stitch design. It uses Flask to serve the UI and the existing `results/model.joblib` detector for CSV inference.

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. Upload a CSV containing all feature columns used by the trained model. The dashboard displays a summary and up to 250 rows; **Download CSV** exports all analyzed rows. Keep this local demo bound to localhost unless you add authentication, request limits, and deployment hardening. The bundled model was trained/evaluated on synthetic demo data, so do not treat its outputs as validated real-world security findings.
