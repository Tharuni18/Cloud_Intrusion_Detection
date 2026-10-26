from flask import Flask, render_template, request, jsonify
from pathlib import Path
import io
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "results" / "model.joblib"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/detect")
def detect():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename.lower().endswith(".csv"):
        return jsonify(error="Please upload a .csv file."), 400
    try:
        df = pd.read_csv(uploaded)
        if df.empty:
            return jsonify(error="The CSV has no data rows."), 400
        bundle = joblib.load(MODEL_PATH)
        detector, columns = bundle["detector"], bundle["columns"]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " …" if len(missing) > 10 else ""
            return jsonify(error=f"CSV is missing {len(missing)} required feature column(s): {preview}{suffix}"), 400
        X = df[columns].copy()
        flags, scores = detector.decision(X)
        reasons = detector.explain(X)
        out = df.copy()
        out["alert"] = flags.astype(int)
        out["score"] = scores
        out["reason"] = reasons
        csv_text = out.to_csv(index=False)
        rows = [{"alert": bool(flags[i]), "score": float(scores[i]), "reason": str(reasons[i])} for i in range(min(len(df), 250))]
        alerts = int(flags.sum())
        return jsonify(total=len(df), alerts=alerts, rate=round(alerts * 100 / len(df), 2), rows=rows, csv=csv_text,
                       note="Table shows up to 250 rows; download contains all rows.")
    except Exception as exc:
        return jsonify(error=f"Could not analyze this CSV: {exc}"), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
