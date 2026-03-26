import os
import uuid
import tempfile
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

from parser import parse_invoices_to_df
from anomaly_detector import detect_anomalies

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def df_to_response(df: pd.DataFrame) -> dict:
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "invoice_number":   row.get("invoice_number", "UNKNOWN"),
            "invoice_date":     row.get("invoice_date", "UNKNOWN"),
            "vendor_name":      row.get("vendor_name", "UNKNOWN"),
            "grand_total":      None if pd.isna(row.get("grand_total")) else round(float(row["grand_total"]), 2),
            "calculated_total": None if pd.isna(row.get("calculated_total")) else round(float(row["calculated_total"]), 2),
            "num_line_items":   int(row.get("num_line_items", 0)),
            "z_score":          None if pd.isna(row.get("z_score")) else round(float(row["z_score"]), 4),
            "is_anomaly":       bool(row.get("is_anomaly", False)),
            "anomaly_reason":   row.get("anomaly_reason", ""),
        })

    total_value = df["grand_total"].sum()
    summary = {
        "total_invoices":    len(df),
        "total_anomalies":   int(df["is_anomaly"].sum()),
        "total_value":       None if pd.isna(total_value) else round(float(total_value), 2),
        "unique_vendors":    int(df["vendor_name"].nunique()),
    }

    anomaly_reasons = {}
    flagged = df[df["is_anomaly"]]
    for reason_str in flagged["anomaly_reason"]:
        for part in str(reason_str).split(","):
            part = part.strip()
            if part:
                anomaly_reasons[part] = anomaly_reasons.get(part, 0) + 1

    return {"summary": summary, "invoices": rows, "anomaly_reasons": anomaly_reasons}


@app.route("/api/process", methods=["POST"])
def process_invoices():
    files = request.files.getlist("invoices")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded."}), 400

    invalid = [f.filename for f in files if not allowed_file(f.filename)]
    if invalid:
        return jsonify({"error": f"Unsupported file type(s): {', '.join(invalid)}"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for f in files:
            safe_name = f"{uuid.uuid4().hex}_{f.filename}"
            path = os.path.join(tmpdir, safe_name)
            f.save(path)
            image_paths.append(path)

        try:
            df = parse_invoices_to_df(image_paths)
            df = detect_anomalies(df)
        except Exception:
            return jsonify({"error": "Pipeline failed.", "detail": traceback.format_exc()}), 500

    return jsonify(df_to_response(df))


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>", methods=["GET"])
def static_files(path: str):
    full_path = os.path.join(app.root_path, path)
    if os.path.isfile(full_path):
        return send_from_directory(".", path)
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
