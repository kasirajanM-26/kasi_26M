# NeuraScan — Brain Tumor Detection & Segmentation

## Abstract

NeuraScan is a Flask web application for MRI-based **brain tumor classification** and **segmentation**. Users upload scans, receive a tumor / no-tumor prediction with confidence, and optional red overlay segmentation focused on the central brain region to reduce false highlights at skull edges. The app includes per-user history, a dashboard, disease-aware food recommendations (general wellness only), and an **admin** area for managing users and viewing all prediction records. Data is stored in **SQLite** with hashed passwords.

---

## Features

| Area | Description |
|------|-------------|
| **Detection** | Binary classification (tumor vs. no tumor) with confidence score. |
| **Segmentation** | Post-processed mask: threshold, morphology, **center 50%** crop, largest connected component, red overlay — reduces edge/skull artifacts. |
| **Web app** | Upload, results, navigation, responsive UI. |
| **Dashboard** | Charts for the logged-in user’s scan activity. |
| **Food recommendation** | Lists vary by profile disease: Diabetes, BP, Thyroid, Sinus, or None — not medical advice. |

---

## Technologies

- **Flask** — Web framework, sessions, routing  
- **TensorFlow / Keras** — Classification and U-Net-style segmentation models  
- **OpenCV** — Image I/O, morphology, connected components, overlay  
- **SQLite** — Users and prediction history  
- **HTML / CSS** — Templates and styling  

---

## Dataset

Training data is **not bundled** in this repository. Models are expected under:

- `models/binary_classifier.h5` — Binary classifier  
- `models/brats_unet_final.keras` — Segmentation network  

Typical sources for brain MRI work include public datasets such as **BraTS** (multi-institutional brain tumor segmentation). Replace paths in `app.py` if your filenames differ.

---

## How to run

### 1. Create a virtual environment

```bash
python -m venv venv
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
python app.py
```

Open the URL shown in the terminal (often `http://127.0.0.1:5000`).

> **Database:** On first run, `database.db` is created in the project root. If you change the schema, set environment variable `AUTO_RESET_ON_SCHEMA_MISMATCH=1` once (see `app.py`) or delete `database.db` manually to recreate tables.

---

## Folder structure

```
bartset/
├── app.py                 # Flask app, models, DB, routes
├── requirements.txt       # Python dependencies
├── database.db            # SQLite (created at runtime)
├── models/                # Place .h5 / .keras models here
├── static/
│   ├── css/
│   └── uploads/           # Saved input + segmentation images
└── templates/             # Jinja2 HTML pages
```

---

## Admin login (documentation only)

The admin account is **created internally** and is **not shown** in the UI. For initial access, use the credentials documented **only here**:

| Field | Value |
|-------|--------|
| **Email** | `admin@neurascan.com` |
| **Password** | `admin123` |

Change the password in production (e.g., update the DB or extend the app with a password-change flow). Only **one** admin email is enforced in code.

---

## Future enhancements

- Email verification and password reset  
- Export history (CSV/PDF)  
- GPU-optimized inference and model versioning  
- Role-based audit log for admin actions  
- Stronger deployment config (HTTPS, secret key from env, production WSGI server)  

---

## Disclaimer

Food recommendations are **general wellness suggestions** only and **not** medical advice. Always follow your clinician’s guidance for diagnosis, treatment, and diet.
