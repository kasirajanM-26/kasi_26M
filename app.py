import os
import sqlite3
import uuid
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import cv2
from PIL import Image
from flask import (
    Flask,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    send_file,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stored in project root as required.
DB_PATH = os.path.join(BASE_DIR, "database.db")

STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
PRESCRIPTIONS_DIR = os.path.join(STATIC_DIR, "prescriptions")
CSS_DIR = os.path.join(STATIC_DIR, "css")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PRESCRIPTIONS_DIR, exist_ok=True)
os.makedirs(CSS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "models", "binary_classifier.h5")
SEGMENTER_MODEL_PATH = os.path.join(BASE_DIR, "models", "brats_unet_final.keras")

# Default admin (single account). Credentials documented in README.md only — never shown in UI.
ADMIN_EMAIL = "admin@neurascan.com"
ADMIN_PASSWORD = "admin123"

DISEASES = {
    "diabetes": "Diabetes",
    "bp": "BP",
    "thyroid": "Thyroid",
    "sinus": "Sinus",
    "none": "None",
}

# Personalized food tips per disease only (general wellness, not medical advice).
DISEASE_FOOD_LISTS = {
    "diabetes": [
        "Low glycemic / low sugar foods (non-starchy vegetables, legumes)",
        "High fiber diet (whole grains, oats, beans)",
        "Limit sugary drinks and refined carbs",
        "Lean proteins in moderate portions",
    ],
    "bp": [
        "Low salt foods; use herbs and spices for flavor",
        "Fruits and vegetables (potassium-rich options as advised by your clinician)",
        "Limit processed and canned foods",
        "Whole grains over refined grains",
    ],
    "thyroid": [
        "Iodine-appropriate foods as directed by your doctor (e.g., eggs, dairy if suitable)",
        "Selenium sources in moderation (Brazil nuts, fish) if advised",
        "Avoid excessive iodine supplements unless prescribed",
    ],
    "sinus": [
        "Warm fluids (broths, herbal teas)",
        "Anti-inflammatory foods (ginger, turmeric, leafy greens)",
        "Stay hydrated; limit very spicy foods if they trigger symptoms",
    ],
    "none": [
        "Balanced diet with varied fruits and vegetables",
        "Whole grains and lean proteins",
        "Adequate hydration",
        "Limit ultra-processed foods and excess sugar",
    ],
}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-change-me")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB
DB_INITIALIZED = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

MEDICAL_SYSTEM_DISCLAIMER = (
    "This system provides AI-based suggestions and is not a substitute for professional medical advice."
)


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def _reset_db_file_if_needed() -> None:
    """
    Fresh database option:
    - Set env var RESET_DB=1 to delete database.db on each run.
    - Set env var CLEAR_OLD_USERS=1 to clear non-admin users + their history.
    """
    reset = os.environ.get("RESET_DB", "0").strip() == "1"
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    clear_users = os.environ.get("CLEAR_OLD_USERS", "0").strip() == "1"
    # Clearing will be applied after init_db.
    return clear_users


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r["name"] for r in rows}


def _ensure_single_admin(conn: sqlite3.Connection) -> None:
    """
    Enforce exactly one admin account (ADMIN_EMAIL). Remove any other admin rows.
    """
    conn.execute("DELETE FROM users WHERE role = 'admin' AND email != ?", (ADMIN_EMAIL,))
    row = conn.execute(
        "SELECT id FROM users WHERE email = ? AND role = 'admin'",
        (ADMIN_EMAIL,),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO users (name, email, password, role, disease) VALUES (?, ?, ?, 'admin', 'none')",
            ("Administrator", ADMIN_EMAIL, generate_password_hash(ADMIN_PASSWORD)),
        )
    conn.commit()


def init_db() -> None:
    # Ensure database existence and schema correctness.
    clear_users = _reset_db_file_if_needed()

    # If database exists but schema is outdated, we auto-reset to avoid runtime errors.
    # This matches the "fresh database OR clear old users automatically" requirement.
    auto_reset_on_schema_mismatch = os.environ.get("AUTO_RESET_ON_SCHEMA_MISMATCH", "1").strip() == "1"

    need_reset = False
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row

            if "users" not in {t["name"] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}:
                need_reset = True
            else:
                user_cols = _table_columns(conn, "users")
                required_user_cols = {
                    "id",
                    "name",
                    "email",
                    "password",
                    "role",
                    "disease",
                }
                if not required_user_cols.issubset(user_cols):
                    need_reset = True

                if not need_reset:
                    if "history" not in {t["name"] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}:
                        need_reset = True
                    else:
                        hist_cols = _table_columns(conn, "history")
                        required_hist_cols = {
                            "id",
                            "user_id",
                            "image",
                            "mask",
                            "result",
                            "confidence",
                            "timestamp",
                        }
                        if not required_hist_cols.issubset(hist_cols):
                            need_reset = True
            conn.close()
        except Exception:
            need_reset = True

    if need_reset and auto_reset_on_schema_mismatch and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # (Re)create schema. Column `password` stores werkzeug password hash.
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'admin')),
            disease TEXT NOT NULL DEFAULT 'none'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image TEXT NOT NULL,
            mask TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()

    _ensure_single_admin(conn)

    # Optional clearing of old users.
    if clear_users:
        conn.execute("DELETE FROM history WHERE user_id IN (SELECT id FROM users WHERE role='user')")
        conn.execute("DELETE FROM users WHERE role='user'")
        conn.commit()


@app.before_request
def _ensure_db():
    global DB_INITIALIZED
    if not DB_INITIALIZED:
        init_db()
        DB_INITIALIZED = True


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Admin access required.", "error")
            return redirect(url_for("home"))
        return view(*args, **kwargs)

    return wrapped


def _load_keras_model(path: str):
    """
    Load Keras model with compile=False, robust to environments where tf.keras is missing.
    """
    try:
        import tensorflow as tf  # local import prevents startup crashes

        if hasattr(tf, "keras"):
            return tf.keras.models.load_model(path, compile=False)
    except Exception:
        pass

    # Fallback to standalone keras
    try:
        import keras  # type: ignore

        return keras.models.load_model(path, compile=False)
    except Exception:
        return None


def load_models():
    classifier = None
    segmenter = None
    if os.path.exists(CLASSIFIER_MODEL_PATH):
        classifier = _load_keras_model(CLASSIFIER_MODEL_PATH)
    if os.path.exists(SEGMENTER_MODEL_PATH):
        segmenter = _load_keras_model(SEGMENTER_MODEL_PATH)
    return classifier, segmenter


CLASSIFIER_MODEL, SEGMENTER_MODEL = load_models()


def read_image_rgb_cv(image_path: str, size=(128, 128)) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(size)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def preprocess_classification(image_path: str) -> np.ndarray:
    arr = read_image_rgb_cv(image_path, size=(128, 128))
    return np.expand_dims(arr, axis=0)


def preprocess_segmentation(image_path: str) -> np.ndarray:
    arr = read_image_rgb_cv(image_path, size=(128, 128))
    # Convert 3 -> 4 channels by duplicating one channel.
    extra = arr[:, :, 0:1]
    arr4 = np.concatenate([arr, extra], axis=-1)
    return np.expand_dims(arr4, axis=0)


def create_segmentation_mask_overlay_bgr(input_image_path: str, mask_pred: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """
    Strict center-only segmentation: ignore skull edges / outer regions.
    """
    original = cv2.imread(input_image_path)
    if original is None:
        original_rgb = Image.open(input_image_path).convert("RGB").resize((128, 128))
        original = cv2.cvtColor(np.asarray(original_rgb), cv2.COLOR_RGB2BGR)
    else:
        original = cv2.resize(original, (128, 128))

    mask = np.squeeze(mask_pred)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = (mask > threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_mask[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)] = 1
    mask = mask * center_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)

    red_mask = np.zeros_like(original)
    red_mask[:, :, 2] = mask * 255
    overlay = cv2.addWeighted(original, 0.75, red_mask, 0.4, 0)
    return overlay


def predict_and_save(image_path: str) -> tuple[str, float, str]:
    if CLASSIFIER_MODEL is None:
        raise RuntimeError("Classification model not available. Place models/binary_classifier.h5 and install TensorFlow/Keras.")
    if SEGMENTER_MODEL is None:
        raise RuntimeError("Segmentation model not available. Place models/brats_unet_final.keras and install TensorFlow/Keras.")

    cls_in = preprocess_classification(image_path)
    prob_tumor = float(CLASSIFIER_MODEL.predict(cls_in, verbose=0)[0][0])

    if prob_tumor >= 0.5:
        result = "Tumor Detected"
        confidence = prob_tumor * 100.0
    else:
        result = "No Tumor"
        confidence = (1.0 - prob_tumor) * 100.0

    confidence = round(confidence, 2)

    # Always generate segmented image:
    # - Tumor Detected: mask-only overlay (RED tumor on black)
    # - No Tumor: blank black mask
    if result == "Tumor Detected":
        seg_in = preprocess_segmentation(image_path)
        mask_pred = SEGMENTER_MODEL.predict(seg_in, verbose=0)[0]
        overlay_bgr = create_segmentation_mask_overlay_bgr(image_path, mask_pred, threshold=0.7)
    else:
        overlay_bgr = np.zeros((128, 128, 3), dtype=np.uint8)

    unique_seg = f"seg_{uuid.uuid4().hex}.png"
    seg_abs_path = os.path.join(UPLOAD_DIR, unique_seg)
    cv2.imwrite(seg_abs_path, overlay_bgr)

    return result, confidence, unique_seg


def get_user() -> sqlite3.Row | None:
    if not session.get("user_id"):
        return None
    db = get_db()
    return db.execute("SELECT id, name, email, role, disease FROM users WHERE id = ?", (session["user_id"],)).fetchone()


def food_list_for_disease(disease_key: str) -> list[str]:
    """Return only the logged-in user's disease-based recommendations."""
    k = disease_key if disease_key in DISEASE_FOOD_LISTS else "none"
    return list(DISEASE_FOOD_LISTS[k])


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user is None or not check_password_hash(user["password"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        session["role"] = user["role"]
        session["email"] = user["email"]
        session["name"] = user["name"]
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        disease = (request.form.get("disease") or "none").strip().lower()
        if disease not in DISEASES:
            disease = "none"

        if not name or not email or not password:
            flash("Name, email, and password are required.", "error")
            return render_template("register.html")

        if email == ADMIN_EMAIL.lower():
            flash("This email is reserved.", "error")
            return render_template("register.html")

        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (name, email, password, role, disease) VALUES (?, ?, ?, 'user', ?)",
                (name, email, generate_password_hash(password), disease),
            )
            db.commit()
        except sqlite3.IntegrityError:
            flash("Email already exists. Please log in.", "error")
            return render_template("register.html")

        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    user = get_user()
    disease = user["disease"] if user else "none"
    return render_template(
        "index.html",
        disease_label=DISEASES.get(disease, "None"),
        input_image_file=None,
        segmented_image_file=None,
        result=None,
        confidence=None,
        prediction_timestamp=None,
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict_route():
    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please choose an MRI image.", "error")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        flash("Unsupported file type. Upload an image.", "error")
        return redirect(url_for("home"))

    ext = os.path.splitext(filename)[1].lower()
    unique_in = f"input_{uuid.uuid4().hex}{ext}"
    input_abs = os.path.join(UPLOAD_DIR, unique_in)
    file.save(input_abs)

    # Run prediction
    try:
        result, confidence, seg_unique = predict_and_save(input_abs)
    except Exception as e:
        flash(str(e), "error")
        return redirect(url_for("home"))

    # Save prediction history (input image + segmentation image).
    _save_prediction_history(unique_in, seg_unique, result, confidence)
    latest = _latest_history_row(session["user_id"])
    prediction_timestamp = latest["timestamp"] if latest else utc_now_iso()

    user = get_user()
    return render_template(
        "index.html",
        disease_label=DISEASES.get(user["disease"], "None") if user else "None",
        input_image_file=unique_in,
        segmented_image_file=seg_unique,
        result=result,
        confidence=confidence,
        prediction_timestamp=prediction_timestamp,
    )


@app.route("/history")
@login_required
def history():
    db = get_db()
    rows = db.execute(
        """
        SELECT id, image, mask, result, confidence, timestamp
        FROM history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 300
        """,
        (session["user_id"],),
    ).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()

    counts = db.execute(
        """
        SELECT
          SUM(CASE WHEN result = 'Tumor Detected' THEN 1 ELSE 0 END) AS tumor_count,
          SUM(CASE WHEN result = 'No Tumor' THEN 1 ELSE 0 END) AS no_tumor_count
        FROM history
        WHERE user_id = ?
        """,
        (session["user_id"],),
    ).fetchone()

    # Bar chart: number of scans per day (last 14 days)
    days = [datetime.now(timezone.utc).date() - timedelta(days=i) for i in range(13, -1, -1)]
    day_labels = [d.strftime("%Y-%m-%d") for d in days]
    day_counts_map = {d.strftime("%Y-%m-%d"): 0 for d in days}
    day_rows = db.execute(
        """
        SELECT substr(timestamp, 1, 10) AS day, COUNT(1) AS cnt
        FROM history
        WHERE user_id = ? AND substr(timestamp, 1, 10) BETWEEN ? AND ?
        GROUP BY day
        """,
        (session["user_id"], day_labels[0], day_labels[-1]),
    ).fetchall()
    for r in day_rows:
        day_counts_map[r["day"]] = int(r["cnt"])
    day_counts = [day_counts_map[l] for l in day_labels]

    # Line chart: usage over time (weekly, last 8 weeks)
    now = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=now.weekday())).date()
    week_dates = [week_start - timedelta(weeks=i) for i in range(7, -1, -1)]
    week_labels = []
    week_counts = []
    for wd in week_dates:
        start = datetime(wd.year, wd.month, wd.day, tzinfo=timezone.utc)
        end = start + timedelta(days=6)
        label = start.strftime("W%Y-%m-%d")
        c = db.execute(
            """
            SELECT COUNT(1) AS cnt
            FROM history
            WHERE user_id = ?
              AND timestamp >= ?
              AND timestamp <= ?
            """,
            (session["user_id"], start.isoformat(), end.isoformat()),
        ).fetchone()["cnt"]
        week_labels.append(label)
        week_counts.append(int(c or 0))

    return render_template(
        "dashboard.html",
        tumor_count=int(counts["tumor_count"] or 0),
        no_tumor_count=int(counts["no_tumor_count"] or 0),
        day_labels=day_labels,
        day_counts=day_counts,
        week_labels=week_labels,
        week_counts=week_counts,
    )


def _latest_prediction_result(user_id: int) -> str | None:
    db = get_db()
    row = db.execute(
        "SELECT result FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    return row["result"] if row else None


def _latest_history_row(user_id: int) -> sqlite3.Row | None:
    db = get_db()
    return db.execute(
        """
        SELECT id, image, mask, result, confidence, timestamp
        FROM history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()


def build_prescription_sections() -> dict:
    return {
        "next_steps": [
            "Consult neurologist immediately",
            "MRI/CT scan confirmation",
            "Biopsy if required",
        ],
        "medications": [
            "Dexamethasone (reduce swelling)",
            "Levetiracetam (prevent seizures)",
            "Temozolomide (if malignant — may be considered by oncologist)",
        ],
        "food_healthy": [
            "Fruits (apple, berries)",
            "Vegetables (broccoli, spinach)",
            "Whole grains",
            "Protein (eggs, fish)",
        ],
        "food_avoid": [
            "Junk food",
            "High sugar",
            "Alcohol",
        ],
        "exercise_do": [
            "Light walking",
            "Breathing exercises",
            "Yoga (basic)",
        ],
        "exercise_avoid": [
            "Heavy workouts",
        ],
        "surgery_info": [
            "Surgery required if tumor size increases",
            "Radiotherapy / chemotherapy may be needed",
        ],
        "precautions": [
            "Regular checkups",
            "Avoid stress",
            "Proper sleep",
            "Follow medication strictly",
        ],
        "prescription_disclaimer": "This is not a final prescription. Consult a doctor.",
        "system_disclaimer": MEDICAL_SYSTEM_DISCLAIMER,
    }


def generate_prescription_pdf(user: sqlite3.Row, result: str, confidence: float, prescription_text: str) -> str:
    """
    Generate a prescription PDF file and return its absolute path.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    safe_name = "".join([c for c in (user["name"] or "user") if c.isalnum() or c in ("_", "-")]).strip() or "user"
    filename = f"prescription_{safe_name}.pdf"
    pdf_path = os.path.join(PRESCRIPTIONS_DIR, filename)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    x = 2 * cm
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Medical Prescription (AI‑Assisted Summary)")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(x, y, f"Patient name: {user['name']}")
    y -= 14
    c.drawString(x, y, f"Date: {utc_now_iso()}")
    y -= 14
    c.drawString(x, y, f"Result: {result}")
    y -= 14
    c.drawString(x, y, f"Confidence: {confidence:.2f}%")
    y -= 18

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(x, y, MEDICAL_SYSTEM_DISCLAIMER)
    y -= 18

    c.setFont("Helvetica", 11)
    text = c.beginText(x, y)
    text.setLeading(14)

    # Simple wrapping
    max_chars = 105
    for raw_line in prescription_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            text.textLine("")
            continue
        while len(line) > max_chars:
            text.textLine(line[:max_chars])
            line = line[max_chars:]
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()

    return pdf_path


@app.route("/food")
@login_required
def food():
    user = get_user()
    disease = user["disease"] if user else "none"
    disease_label = DISEASES.get(disease, "None")
    foods = food_list_for_disease(disease)
    return render_template(
        "food.html",
        disease_label=disease_label,
        foods=foods,
    )


@app.route("/articles")
@login_required
def articles():
    return render_template("articles.html")


@app.route("/admin", methods=["GET"])
@admin_required
def admin():
    db = get_db()
    users = db.execute(
        "SELECT id, name, email, role, disease FROM users ORDER BY id DESC"
    ).fetchall()

    total_users = db.execute("SELECT COUNT(1) AS cnt FROM users").fetchone()["cnt"]
    total_predictions = db.execute("SELECT COUNT(1) AS cnt FROM history").fetchone()["cnt"]

    tumor_vs = db.execute(
        """
        SELECT
          SUM(CASE WHEN result='Tumor Detected' THEN 1 ELSE 0 END) AS tumor_count,
          SUM(CASE WHEN result='No Tumor' THEN 1 ELSE 0 END) AS no_tumor_count
        FROM history
        """
    ).fetchone()

    # Bar chart global: scans per day (last 14 days)
    days = [datetime.now(timezone.utc).date() - timedelta(days=i) for i in range(13, -1, -1)]
    day_labels = [d.strftime("%Y-%m-%d") for d in days]
    day_counts_map = {d.strftime("%Y-%m-%d"): 0 for d in days}
    day_rows = db.execute(
        """
        SELECT substr(timestamp, 1, 10) AS day, COUNT(1) AS cnt
        FROM history
        WHERE substr(timestamp, 1, 10) BETWEEN ? AND ?
        GROUP BY day
        """,
        (day_labels[0], day_labels[-1]),
    ).fetchall()
    for r in day_rows:
        day_counts_map[r["day"]] = int(r["cnt"])
    day_counts = [day_counts_map[l] for l in day_labels]

    history_rows = db.execute(
        """
        SELECT h.id, u.email, h.image, h.mask, h.result, h.confidence, h.timestamp
        FROM history h
        JOIN users u ON u.id = h.user_id
        ORDER BY h.timestamp DESC
        LIMIT 400
        """
    ).fetchall()

    return render_template(
        "admin.html",
        users=users,
        history_rows=history_rows,
        total_users=int(total_users or 0),
        total_predictions=int(total_predictions or 0),
        tumor_count=int(tumor_vs["tumor_count"] or 0),
        no_tumor_count=int(tumor_vs["no_tumor_count"] or 0),
        day_labels=day_labels,
        day_counts=day_counts,
    )


@app.post("/admin/delete_user/<int:user_id>")
@admin_required
def admin_delete_user(user_id: int):
    # Never delete yourself/admin.
    db = get_db()
    target = db.execute("SELECT id, role FROM users WHERE id = ?", (user_id,)).fetchone()
    if target is None:
        flash("User not found.", "error")
        return redirect(url_for("admin"))
    if target["role"] == "admin":
        flash("Admin accounts cannot be deleted.", "error")
        return redirect(url_for("admin"))

    db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    flash("User and their history deleted.", "success")
    return redirect(url_for("admin"))


@app.post("/admin/delete_history/<int:history_id>")
@admin_required
def admin_delete_history(history_id: int):
    db = get_db()
    db.execute("DELETE FROM history WHERE id = ?", (history_id,))
    db.commit()
    flash("History record deleted.", "success")
    return redirect(url_for("admin"))


@app.post("/admin/user/<int:user_id>/edit")
@admin_required
def admin_edit_user(user_id: int):
    db = get_db()
    target = db.execute("SELECT id, role, email FROM users WHERE id = ?", (user_id,)).fetchone()
    if target is None:
        flash("User not found.", "error")
        return redirect(url_for("admin"))
    if target["role"] == "admin":
        flash("The admin account cannot be modified here.", "error")
        return redirect(url_for("admin"))

    name = (request.form.get("name") or "").strip()
    disease = (request.form.get("disease") or "none").strip().lower()
    if disease not in DISEASES:
        disease = "none"
    if not name:
        flash("Name is required.", "error")
        return redirect(url_for("admin"))

    db.execute(
        "UPDATE users SET name = ?, disease = ? WHERE id = ? AND role = 'user'",
        (name, disease, user_id),
    )
    db.commit()
    flash("User updated.", "success")
    return redirect(url_for("admin"))


def _save_prediction_history(input_unique: str, seg_unique: str, result: str, confidence: float) -> None:
    db = get_db()
    db.execute(
        """
        INSERT INTO history (user_id, image, mask, result, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session["user_id"], f"{input_unique}", f"{seg_unique}", result, float(confidence), utc_now_iso()),
    )
    db.commit()


@app.get("/prescription/download")
@login_required
def download_prescription():
    user = get_user()
    if user is None:
        return redirect(url_for("login"))

    row = _latest_history_row(user["id"])
    if row is None:
        flash("No prediction found to generate prescription.", "error")
        return redirect(url_for("home"))

    if row["result"] != "Tumor Detected":
        flash("Prescription is available only when tumor is detected.", "error")
        return redirect(url_for("home"))

    sections = build_prescription_sections()

    # Build plain-text prescription for PDF
    prescription_text = "\n".join(
        [
            "A. Next Steps",
            *[f"• {x}" for x in sections["next_steps"]],
            "",
            "B. Medications (General Guidance Only)",
            *[f"• {x}" for x in sections["medications"]],
            f"NOTE: {sections['prescription_disclaimer']}",
            "",
            "C. Food Recommendations",
            "Healthy diet:",
            *[f"• {x}" for x in sections["food_healthy"]],
            "",
            "Avoid:",
            *[f"• {x}" for x in sections["food_avoid"]],
            "",
            "D. Exercises",
            "Recommended:",
            *[f"• {x}" for x in sections["exercise_do"]],
            "",
            "Avoid:",
            *[f"• {x}" for x in sections["exercise_avoid"]],
            "",
            "E. Surgery Info",
            *[f"• {x}" for x in sections["surgery_info"]],
            "",
            "F. Precautions",
            *[f"• {x}" for x in sections["precautions"]],
            "",
            f"DISCLAIMER: {sections['system_disclaimer']}",
        ]
    )

    try:
        pdf_path = generate_prescription_pdf(
            user=user,
            result=row["result"],
            confidence=float(row["confidence"]),
            prescription_text=prescription_text,
        )
        return send_file(pdf_path, as_attachment=True)
    except Exception:
        # Fallback: show prescription in HTML
        return render_template(
            "prescription_fallback.html",
            patient_name=user["name"],
            prediction_timestamp=row["timestamp"],
            result=row["result"],
            confidence=float(row["confidence"]),
            sections=sections,
        )


if __name__ == "__main__":
    # Start server
    app.run(debug=True)

