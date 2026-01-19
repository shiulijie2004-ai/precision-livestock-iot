To make your README look like the **second image**, you mainly need these Markdown elements:

* Headings: `##`, `###`
* Numbered lists: `1. 2. 3.`
* Code blocks: triple backticks ```
* Callouts (simple): emoji + **bold** text (GitHub Markdown doesn’t have “official” callout boxes by default, but emojis work well)

Below is a **copy-paste ready README.md section** (you can paste into your repo). It follows the same style as your second image (Installation → create installer file → run installer → interactive setup → launch dashboard).

````md
---

## Installation

> ⚠️ **Important:** Do NOT run the installer using `sudo`. Run as a normal user.

### 1) Create the installer file

Create a file named `install_fyp.sh` in the project root:

```bash
nano install_fyp.sh
````

Paste your installer script content inside `install_fyp.sh`, then save:

* Save: `CTRL + O` → Enter
* Exit: `CTRL + X`

Make it executable:

```bash
chmod +x install_fyp.sh
```

---

### 2) Run the installer (no sudo)

Run:

```bash
bash install_fyp.sh
```

---

## Interactive Setup

During installation, you will be asked:

1. **Project Title**

   * Example: `IoT Application for Tracking Farm Animals`

2. **Custom Algorithms (optional, comma-separated)**

   * Example: `GRU, BiLSTM, XGBoost-Tiny`
   * Leave blank if you do not have custom algorithms.

After that, the installer will:

* Generate a project folder (example: `Student_FYP/`)
* Create a Python environment (example: `venv/`)
* Install all required packages

---

## Launch the Dashboard

After you see **✅ INSTALLATION COMPLETE**, go into the generated folder:

```bash
cd Student_FYP
```

Activate the environment:

```bash
source venv/bin/activate
```

Run the dashboard:

```bash
streamlit run app.py
```

Open in your browser:

* `http://localhost:8501`

---



