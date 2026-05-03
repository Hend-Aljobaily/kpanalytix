"""
KPAnalytix — Construction Intelligence Dashboard
==================================================
3-Phase Streamlit application: Design → Development → Maintenance.

Phase 1  DESIGN         IFC upload, Saudi Building Code (SBC) compliance checker
Phase 2  DEVELOPMENT    BIM viewer, dashboards, delay analysis, what-if, recommendations
Phase 3  MAINTENANCE    Monitoring (coming soon)

Brand:  Deep Plum #240F3E  |  Royal Purple #442270  |  Amethyst #69479E  |  Lilac Haze #F9F4FF

Prerequisites:
  python preprocess_ifc.py                  (one-time, ~4 min)
  python generate_weekly_drone_images.py    (one-time, ~3 min)

Run:
  conda activate construction-kpanalytix
  streamlit run dashboard.py
"""

import base64
import math
import pickle
import re
from datetime import date, timedelta
from pathlib import Path

import ifcopenshell
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "kpanalytix_data"
IFC_DIR = DATA_DIR / "bim"
DRONE_DIR = DATA_DIR / "drone_images"
ASSETS_DIR = BASE_DIR / "assets"
LOGO_FULL = ASSETS_DIR / "KPAnalytix_White.png"   # horizontal wordmark on dark
LOGO_MARK = ASSETS_DIR / "KPAnalytix_White.png"   # also wordmark in sidebar (looks cleaner)
LOGO_FAVICON = ASSETS_DIR / "KPAnalytix_Icon.png"

BUILDINGS = ["BasicHouse", "TallBuilding", "SampleHouse", "Duplex"]
NUM_WEEKS = 4
PROJECT_START = date(2025, 2, 6)  # Week 1 reference date
WEEK_PROGRESS = {1: 18.0, 2: 45.0, 3: 75.0, 4: 96.0}  # planned % per week

# ── KPAnalytix Brand Palette ────────────────────────────────────────────────
DEEP_PLUM     = "#240F3E"   # Primary — darkest, headers / sidebar / text
ROYAL_PURPLE  = "#442270"   # Mid-tone
AMETHYST      = "#69479E"   # Accent — charts / KPI borders / highlights
LILAC_HAZE    = "#F9F4FF"   # Pale background tint
AMETHYST_TINT = "#b5a4d6"   # Lightened Amethyst for legible text on dark plum
PLUM_HOVER    = "rgba(105,71,158,0.12)"

# Convenience aliases
DEEP_INDIGO   = DEEP_PLUM
LAVENDER      = AMETHYST
LAVENDER_PALE = LILAC_HAZE
LAVENDER_TINT = AMETHYST_TINT
INDIGO_MID    = ROYAL_PURPLE
INDIGO_HOVER  = PLUM_HOVER
INDIGO_LIGHT  = ROYAL_PURPLE

# IFC type → display colour for 3D viewer (brand-aware, richer palette)
TYPE_COLORS = {
    "IfcWall": "#f1ece2", "IfcWallStandardCase": "#f1ece2",
    "IfcSlab": "#b8b3c8", "IfcColumn": "#7a7397", "IfcBeam": "#9a93b5",
    "IfcRoof": "#9c4a2a", "IfcDoor": "#6b4a2b", "IfcWindow": "#4ea3d6",
    "IfcStair": "#caa97a", "IfcRailing": "#4a4a55",
    "IfcCovering": "#d8c39a", "IfcPlate": "#a39db0",
    "IfcFurnishingElement": "#b8a78c",
}

# Construction phase order (lower = built first)
TYPE_PHASE_ORDER = {
    "IfcSlab": 1, "IfcColumn": 2, "IfcBeam": 3,
    "IfcWall": 4, "IfcWallStandardCase": 4,
    "IfcStair": 5, "IfcDoor": 6, "IfcWindow": 6,
    "IfcRoof": 7, "IfcRailing": 8, "IfcCovering": 9,
    "IfcPlate": 9, "IfcFurnishingElement": 10,
}

TYPE_LABELS = {
    "IfcWall": "Walls", "IfcWallStandardCase": "Walls (Std)",
    "IfcSlab": "Slabs", "IfcColumn": "Columns", "IfcBeam": "Beams",
    "IfcDoor": "Doors", "IfcWindow": "Windows", "IfcStair": "Stairs",
    "IfcRoof": "Roofs", "IfcRailing": "Railings",
    "IfcFurnishingElement": "Furnishings", "IfcCovering": "Coverings",
    "IfcPlate": "Plates",
}

# ═══════════════════════════════════════════════════════════════════════════
# COMMERCIAL DATA MODEL — budget, dependencies, market, design, compliance
# ═══════════════════════════════════════════════════════════════════════════

# Per-building total contract value (SAR)
BUDGETS = {
    "BasicHouse":    1_900_000,
    "TallBuilding": 14_500_000,
    "SampleHouse":   2_700_000,
    "Duplex":        3_500_000,
}

# Per-building daily standing-overhead cost (SAR/day) used for delay $ math
DAILY_OVERHEAD = {
    "BasicHouse":    4_300,
    "TallBuilding": 29_000,
    "SampleHouse":   6_200,
    "Duplex":        8_600,
}

# Construction phases — drives Sankey, dependency propagation, design tab
PHASES = ["Design", "Foundation", "Structure", "Envelope", "MEP", "Interior", "Finishes", "Handover"]
PHASE_DEPS = [
    ("Design",     "Foundation"),
    ("Foundation", "Structure"),
    ("Structure",  "Envelope"),
    ("Structure",  "MEP"),
    ("Envelope",   "Interior"),
    ("MEP",        "Interior"),
    ("Interior",   "Finishes"),
    ("Finishes",   "Handover"),
]
# Each phase consumes a fraction of the total schedule
PHASE_WEIGHT = {
    "Design": 0.05, "Foundation": 0.12, "Structure": 0.22, "Envelope": 0.15,
    "MEP": 0.15, "Interior": 0.13, "Finishes": 0.13, "Handover": 0.05,
}
# What % of the total budget is allocated to each phase (sums to 1.0)
PHASE_BUDGET_SHARE = {
    "Design": 0.04, "Foundation": 0.10, "Structure": 0.24, "Envelope": 0.16,
    "MEP": 0.18, "Interior": 0.14, "Finishes": 0.10, "Handover": 0.04,
}

# Mock market materials — structured exactly like a real LME / supplier feed
# so swapping in a real API later is a one-function change.
MARKET_MATERIALS = [
    {"material": "Steel rebar (12 mm)",  "unit": "SAR / tonne", "price": 3_010, "change_30d_pct":  6.2,  "supply": "Tight",   "source": "Hadeed (SABIC)"},
    {"material": "OPC cement",           "unit": "SAR / bag",   "price": 16.80, "change_30d_pct":  1.1,  "supply": "Normal",  "source": "Yamama Cement"},
    {"material": "Copper LV cable",      "unit": "SAR / m",     "price": 39.00, "change_30d_pct":  9.8,  "supply": "Tight",   "source": "Riyadh Cables (LME-linked)"},
    {"material": "Aggregate 20 mm",      "unit": "SAR / m\u00b3","price": 94.0, "change_30d_pct": -0.7,  "supply": "Normal",  "source": "Eastern Province Quarries"},
    {"material": "Diesel",               "unit": "SAR / L",     "price": 1.15,  "change_30d_pct":  3.4,  "supply": "Normal",  "source": "Saudi Aramco"},
    {"material": "Skilled labor",        "unit": "SAR / day",   "price": 290.0, "change_30d_pct":  4.6,  "supply": "Tight",   "source": "Al Bawani / Nesma & Partners"},
    {"material": "Float glass 8 mm",     "unit": "SAR / m\u00b2","price": 168.0, "change_30d_pct":  2.0,  "supply": "Normal",  "source": "Obeikan Glass"},
    {"material": "HVAC ducting (galv.)", "unit": "SAR / m\u00b2","price": 225.0, "change_30d_pct": -1.4,  "supply": "Normal",  "source": "Zamil Industrial"},
]

# Compliance checklist items shown on the Design & Compliance tab.
# Status is computed deterministically per building from the building name hash
# so each project tells a slightly different story.
COMPLIANCE_ITEMS = [
    {"item": "Structural sign-off",            "authority": "Engineer of Record"},
    {"item": "Fire & life safety review",      "authority": "Saudi Civil Defense (GDCD)"},
    {"item": "MEP coordination clash check",   "authority": "MEP Consultant"},
    {"item": "Energy code (Mostadam)",         "authority": "MOMRAH"},
    {"item": "Accessibility audit",            "authority": "MOMRAH"},
    {"item": "Civil Defense NOC",              "authority": "Saudi Civil Defense (GDCD)"},
    {"item": "SEC / utilities approval",       "authority": "Saudi Electricity Co. (SEC)"},
    {"item": "Municipality permit",            "authority": "Riyadh Municipality (Amanah)"},
]

# CAD tool open-instructions — used by the BIM viewer "Open in CAD" panel
CAD_TOOLS = [
    {"name": "Revit",      "blurb": "File → Open → IFC. Pick Auto-join Walls in the import options."},
    {"name": "AutoCAD",    "blurb": "Insert tab → Import → IFC. Requires the AutoCAD Architecture toolset."},
    {"name": "BIM Vision", "blurb": "Free viewer. File → Open → select the .ifc you just downloaded."},
    {"name": "Solibri",    "blurb": "File → Open Model → IFC. Solibri Anywhere (free) reads schema 4 natively."},
]

# ═══════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="KPAnalytix — Construction Intelligence",
    page_icon=str(LOGO_FAVICON) if LOGO_FAVICON.exists() else ":building_construction:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  /* ── Global ── */
  .stApp {{
      background:
        radial-gradient(circle at 12% 8%, rgba(105,71,158,0.13) 0%, transparent 45%),
        radial-gradient(circle at 92% 0%, rgba(36,15,62,0.08) 0%, transparent 55%),
        {LILAC_HAZE};
  }}
  #MainMenu, footer {{ visibility: hidden; }}
  /* Keep the header in the DOM so the sidebar toggle stays clickable,
     but make it transparent so it doesn't show a white bar. */
  header[data-testid="stHeader"] {{
      background: transparent !important;
      height: 0;
  }}
  /* Ensure the sidebar collapse / re-open control is always visible */
  [data-testid="collapsedControl"] {{
      display: block !important;
      visibility: visible !important;
      z-index: 999999;
      color: {DEEP_PLUM} !important;
  }}
  [data-testid="collapsedControl"] svg {{
      fill: {DEEP_PLUM} !important;
      color: {DEEP_PLUM} !important;
  }}
  .block-container {{ padding-top: 1.6rem; }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
      background: linear-gradient(175deg, {DEEP_PLUM} 0%, #2e1648 55%, {ROYAL_PURPLE} 100%);
      box-shadow: 4px 0 18px rgba(36,15,62,0.18);
  }}
  section[data-testid="stSidebar"] * {{ color: #ece4f6 !important; }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label {{
      color: {AMETHYST_TINT} !important;
      font-weight: 600;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
  }}
  section[data-testid="stSidebar"] hr {{
      border-color: rgba(255,255,255,0.12) !important;
  }}
  /* Slider track */
  section[data-testid="stSidebar"] [data-baseweb="slider"] div[role="progressbar"] {{
      background: {AMETHYST_TINT} !important;
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 0;
      border-bottom: 2px solid {LAVENDER};
      background: rgba(255,255,255,0.55);
      border-radius: 0.5rem 0.5rem 0 0;
      padding: 0.15rem 0.4rem 0 0.4rem;
  }}
  .stTabs [data-baseweb="tab"] {{
      padding: 0.7rem 1.7rem;
      font-weight: 600;
      color: {DEEP_INDIGO};
      border-bottom: 3px solid transparent;
      transition: all 0.18s ease;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
      background: {PLUM_HOVER};
  }}
  .stTabs [aria-selected="true"] {{
      border-bottom: 3px solid {DEEP_INDIGO} !important;
      color: {DEEP_INDIGO} !important;
      background: linear-gradient(180deg, {PLUM_HOVER}, transparent);
  }}

  /* ── KPI cards ── */
  .kpi {{
      background: white;
      border-radius: 0.7rem;
      padding: 1.05rem 1.25rem;
      box-shadow: 0 4px 14px rgba(36,15,62,0.08), 0 1px 3px rgba(36,15,62,0.06);
      border-left: 4px solid {LAVENDER};
      height: 100%;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
  }}
  .kpi:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(36,15,62,0.13);
  }}
  .kpi .val {{
      font-size: 1.6rem;
      font-weight: 700;
      color: {DEEP_INDIGO};
      line-height: 1.2;
      letter-spacing: -0.01em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
  }}
  .kpi .lbl {{
      font-size: 0.72rem;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-top: 0.3rem;
      font-weight: 600;
  }}

  /* ── Section dividers ── */
  .sec {{
      color: {DEEP_INDIGO};
      font-size: 1.05rem;
      font-weight: 700;
      border-bottom: 2px solid {LAVENDER};
      padding-bottom: 0.35rem;
      margin: 1.7rem 0 0.85rem 0;
      letter-spacing: -0.005em;
      display: flex;
      align-items: center;
      gap: 0.45rem;
  }}
  .sec::before {{
      content: '';
      width: 6px; height: 18px;
      background: linear-gradient(180deg, {DEEP_INDIGO}, {LAVENDER});
      border-radius: 2px;
  }}

  /* ── Image frame ── */
  .img-frame {{
      background: white;
      border-radius: 0.6rem;
      padding: 0.45rem;
      box-shadow: 0 4px 14px rgba(36,15,62,0.10);
      border: 1px solid rgba(105,71,158,0.18);
  }}

  /* ── Risk card ── */
  .risk-card {{
      background: white;
      border-radius: 0.55rem;
      padding: 0.95rem 1.2rem;
      box-shadow: 0 2px 8px rgba(36,15,62,0.07);
      margin-bottom: 0.55rem;
      border-left: 3px solid {LAVENDER};
      transition: transform 0.15s ease;
  }}
  .risk-card:hover {{ transform: translateX(3px); }}

  /* ── Rec card ── */
  .rec-card {{
      background: white;
      border-radius: 0.55rem;
      padding: 1rem 1.2rem;
      border-left: 4px solid {LAVENDER};
      box-shadow: 0 2px 8px rgba(36,15,62,0.07);
      margin-bottom: 0.65rem;
      transition: transform 0.15s ease;
  }}
  .rec-card:hover {{ transform: translateX(3px); }}

  /* ── Plotly chart wrapper polish ── */
  .stPlotlyChart {{
      background: white;
      border-radius: 0.6rem;
      padding: 0.4rem 0.4rem 0.1rem 0.4rem;
      box-shadow: 0 3px 12px rgba(36,15,62,0.07);
      border: 1px solid rgba(105,71,158,0.14);
  }}

  /* ── Dataframe polish ── */
  [data-testid="stDataFrame"] {{
      border-radius: 0.55rem;
      overflow: hidden;
      box-shadow: 0 2px 10px rgba(36,15,62,0.07);
      border: 1px solid rgba(105,71,158,0.18);
  }}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_ifc_analysis(name: str) -> dict:
    path = IFC_DIR / f"{name}.ifc"
    model = ifcopenshell.open(str(path))
    elements = {}
    for ifc_type, label in TYPE_LABELS.items():
        elems = model.by_type(ifc_type)
        if elems:
            elements[label] = len(elems)
    storeys = model.by_type("IfcBuildingStorey")
    projects = model.by_type("IfcProject")
    return {
        "name": name,
        "project_name": projects[0].Name if projects else name,
        "schema": model.schema,
        "storeys": len(storeys) if storeys else 0,
        "elements": elements,
        "structural": sum(v for k, v in elements.items()
                          if k in ("Walls", "Walls (Std)", "Slabs", "Columns", "Beams")),
        "total_products": len(model.by_type("IfcProduct")),
    }


@st.cache_data(show_spinner=False)
def load_geometry(name: str):
    pkl = IFC_DIR / f"{name}_geometry.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def _ifc_bytes(name: str) -> bytes:
    """Cached raw bytes of an IFC file for the BIM viewer download button.

    Without this cache, every rerun would re-read the ~50 MB IFC file from
    disk just to feed Streamlit's download button.
    """
    p = IFC_DIR / f"{name}.ifc"
    try:
        return p.read_bytes()
    except FileNotFoundError:
        return b""


@st.cache_data(show_spinner="Extracting 3D geometry…")
def _extract_geometry_from_bytes(ifc_bytes: bytes) -> list:
    """Extract 3D meshes from raw IFC bytes for the BIM viewer."""
    import tempfile as _tmpmod
    import os as _os
    import ifcopenshell.geom

    tmp = _tmpmod.NamedTemporaryFile(delete=False, suffix=".ifc")
    tmp.write(ifc_bytes)
    tmp.close()
    try:
        model = ifcopenshell.open(tmp.name)
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        structural = {
            "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcColumn", "IfcBeam",
            "IfcRoof", "IfcStair", "IfcDoor", "IfcWindow",
            "IfcRailing", "IfcCovering", "IfcPlate", "IfcFurnishingElement",
        }
        products = [p for p in model.by_type("IfcProduct") if p.is_a() in structural]

        elements = []
        for product in products:
            try:
                shape = ifcopenshell.geom.create_shape(settings, product)
                verts_flat = np.array(shape.geometry.verts, dtype=np.float32)
                faces_flat = np.array(shape.geometry.faces, dtype=np.int32)
                if len(verts_flat) == 0:
                    continue
                verts = verts_flat.reshape(-1, 3)
                faces = faces_flat.reshape(-1, 3)
                if len(verts) > 4000:
                    # simple decimation
                    step = max(1, len(verts) // 4000)
                    keep = np.zeros(len(verts), dtype=bool)
                    keep[::step] = True
                    idx_map = np.full(len(verts), -1, dtype=np.int64)
                    idx_map[np.where(keep)[0]] = np.arange(keep.sum())
                    verts = verts[keep]
                    valid = np.all(idx_map[faces] >= 0, axis=1)
                    faces = idx_map[faces[valid]]
                if len(faces) == 0:
                    continue
                z_min = float(verts[:, 2].min())
                z_max = float(verts[:, 2].max())
                elements.append({
                    "type": product.is_a(),
                    "name": product.Name or product.is_a(),
                    "verts": verts, "faces": faces,
                    "z_min": z_min, "z_max": z_max,
                    "z_mid": (z_min + z_max) / 2,
                })
            except Exception:
                continue
        return elements
    finally:
        _os.unlink(tmp.name)


def _planned_pct(week: float) -> float:
    """Smooth interpolation of WEEK_PROGRESS values for any (possibly fractional) week."""
    if week <= 1:
        return WEEK_PROGRESS[1] * max(0.0, week)
    if week >= NUM_WEEKS:
        # Continue trend slightly past week 4 toward 100%
        return min(100.0, WEEK_PROGRESS[NUM_WEEKS] + (week - NUM_WEEKS) * 4.0)
    lo = int(math.floor(week))
    hi = lo + 1
    t = week - lo
    return WEEK_PROGRESS[lo] * (1 - t) + WEEK_PROGRESS[hi] * t


@st.cache_data(show_spinner=False)
def build_progress_df() -> pd.DataFrame:
    """4-week planned vs actual progress for all buildings."""
    rng = np.random.RandomState(42)
    rows = []
    for b in BUILDINGS:
        # per-building delay profile
        delay_week = int(rng.choice([2, 3]))
        delay_mag = rng.uniform(2.5, 6.0)
        catch_up = rng.uniform(0.8, 2.0)

        last_actual = 0.0
        for w in range(1, NUM_WEEKS + 1):
            planned = _planned_pct(w)
            noise = rng.normal(0, 1.4)
            event = 0.0
            if w == delay_week:
                event = -delay_mag
            elif w == delay_week + 1:
                event = catch_up
            actual = max(0.0, min(100.0, planned + noise + event))
            actual = max(last_actual, actual)  # cumulative
            last_actual = actual

            week_label = (PROJECT_START + timedelta(days=(w - 1) * 7)).strftime("%b %d")
            rows.append({
                "Building": b, "Week": w, "Week Label": week_label,
                "Planned %": round(planned, 1),
                "Actual %": round(actual, 1),
            })
    return pd.DataFrame(rows)


def forecast_completion(df_bld: pd.DataFrame) -> dict:
    """Weighted linear regression on the visible weeks to project completion (in weeks)."""
    recent = df_bld.tail(NUM_WEEKS)
    if len(recent) < 2:
        return {"week": NUM_WEEKS, "conf_lo": NUM_WEEKS, "conf_hi": NUM_WEEKS, "slope": 0.0}

    weeks = recent["Week"].values.astype(float)
    actuals = recent["Actual %"].values

    weights = np.arange(1, len(weeks) + 1, dtype=float)
    w_sum = weights.sum()
    w_bar = (weights * weeks).sum() / w_sum
    a_bar = (weights * actuals).sum() / w_sum
    slope = ((weights * (weeks - w_bar) * (actuals - a_bar)).sum() /
             max(1e-6, (weights * (weeks - w_bar) ** 2).sum()))

    if slope <= 0:
        return {"week": NUM_WEEKS + 2, "conf_lo": NUM_WEEKS + 1,
                "conf_hi": NUM_WEEKS + 3, "slope": 0.0}

    last_actual = actuals[-1]
    last_week = weeks[-1]
    weeks_remaining = (100 - last_actual) / slope
    proj_week = last_week + weeks_remaining

    residuals = actuals - (slope * (weeks - last_week) + last_actual)
    std = max(0.4, residuals.std())
    conf_w = std / max(0.1, slope) * 1.2

    return {
        "week": round(proj_week, 1),
        "conf_lo": round(proj_week - conf_w, 1),
        "conf_hi": round(proj_week + conf_w, 1),
        "slope": round(slope, 2),
    }


def week_to_label(w) -> str:
    """Match the "Week Label" format used in build_progress_df so chart x-axes align."""
    days = (float(w) - 1) * 7
    return (PROJECT_START + timedelta(days=days)).strftime("%b %d")


def display_name(b: str) -> str:
    """Convert a CamelCase building identifier to a spaced display name.

    The internal BUILDINGS identifiers (used for file paths, IFC loading, drone
    image folders, etc.) are kept unchanged — this helper is purely for UI.

    'BasicHouse'    -> 'Basic House'
    'TallBuilding'  -> 'Tall Building'
    'SampleHouse'   -> 'Sample House'
    'Duplex'        -> 'Duplex'
    """
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', b)


# ── Delay incident catalogue (deterministic per building) ───────────────────
DELAY_VENDOR_POOL = {
    "Material Supply": [
        ("Hadeed (SABIC)",              "Rebar #6 (12mm) — 8 tonnes"),
        ("Saudi Aluminium Co.",         "Curtain wall mullions — 240 lm"),
        ("Saudi Ceramics",              "Floor tiles 600x600 — 1,200 m²"),
        ("Yamama Cement Co.",           "OPC bagged cement — 600 bags"),
        ("Riyadh Cables Group",         "LV power cable 4x95mm — 800 m"),
    ],
    "Weather": [
        ("Site Operations",             "Sandstorm stop-work (crane lift)"),
        ("Site Operations",             "Extreme heat — concrete pour halted"),
    ],
    "Labor Shortage": [
        ("Al Bawani Manpower",          "Steel fixers crew — 14 absent"),
        ("Nesma & Partners",            "MEP technicians — 9 unavailable"),
        ("Saudi Binladin Group",        "Carpenters — formwork crew short"),
    ],
    "Permit / Inspection": [
        ("Riyadh Municipality (Amanah)","Structural inspection deferred"),
        ("Saudi Civil Defense (GDCD)",  "Fire-fighting layout review pending"),
        ("SEC Inspections",             "LV substation handover delayed"),
    ],
    "Design Changes": [
        ("Client — Design RFI",         "Lobby ceiling redesign issued"),
        ("Architect of Record",         "Window schedule revision (Rev C)"),
        ("MEP Consultant",              "AHU layout change — Floor 03"),
    ],
}


@st.cache_data(show_spinner=False)
def delay_incidents_for(building: str) -> pd.DataFrame:
    """Return a deterministic table of delay incidents for a building.

    Each incident has: cause, vendor/contractor, item delayed, expected vs actual
    delivery date, weeks lost, and current status. Aggregating "Weeks Lost" by
    cause yields the breakdown displayed in the bar chart, so the two views are
    always consistent.
    """
    rng = np.random.RandomState(abs(hash(building)) % (2**31))
    rows = []
    statuses = ["Resolved", "Mitigating", "Open"]
    for cause, pool in DELAY_VENDOR_POOL.items():
        n = rng.randint(1, min(3, len(pool)) + 1)
        idxs = rng.choice(len(pool), size=n, replace=False)
        for i in idxs:
            vendor, item = pool[int(i)]
            week_idx = int(rng.randint(1, NUM_WEEKS + 1))
            expected = PROJECT_START + timedelta(days=(week_idx - 1) * 7)
            slip_days = int(rng.randint(2, 14))
            actual = expected + timedelta(days=slip_days)
            weeks_lost = round(slip_days / 7.0, 1)
            status = statuses[int(rng.randint(0, 3))]
            rows.append({
                "Cause": cause,
                "Vendor / Contractor": vendor,
                "Item / Issue": item,
                "Expected": expected.strftime("%b %d"),
                "Actual": actual.strftime("%b %d"),
                "Weeks Lost": weeks_lost,
                "Status": status,
            })
    return pd.DataFrame(rows)


progress_df = build_progress_df()


# ═══════════════════════════════════════════════════════════════════════════
# COMMERCIAL HELPERS — budget, dependencies, market, what-if, recommendations
# ═══════════════════════════════════════════════════════════════════════════

# Two project lifecycle phases — Design comes before Development. Every tab
# is associated with one phase so the UI can render a clear phase banner.
PHASE_DESIGN      = "Design Phase"
PHASE_DEVELOPMENT = "Development Phase"

TAB_PHASE = {
    "Design & Compliance":  PHASE_DESIGN,
    "All Projects":         PHASE_DEVELOPMENT,
    "BIM Model Viewer":     PHASE_DEVELOPMENT,
    "Project Dashboard":    PHASE_DEVELOPMENT,
    "Budget & Procurement": PHASE_DEVELOPMENT,
    "Delay & Dependencies": PHASE_DEVELOPMENT,
    "What-If Planner":      PHASE_DEVELOPMENT,
    "Recommendations":      PHASE_DEVELOPMENT,
}

# PM is assigned a subset of buildings (3 of 4).
PM_BUILDINGS = ["BasicHouse", "TallBuilding", "SampleHouse"]

# Three roles with different tab visibility. Design-phase tabs are listed
# first so the project lifecycle reads left-to-right (Design → Development).
ROLE_TABS = {
    "Executive": [
        "Portfolio Overview", "Budget & Procurement",
        "Delay & Dependencies", "Recommendations",
    ],
    "Project Manager": [
        "My Projects", "BIM Model Viewer", "Project Dashboard",
        "Budget & Procurement", "Delay & Dependencies",
        "What-If Planner", "Recommendations",
    ],
    "Engineer": [
        "BIM Model Viewer", "Project Dashboard", "Delay & Dependencies",
    ],
}


def tabs_for_role(role: str) -> list:
    """Ordered list of tab labels visible to the given role."""
    return ROLE_TABS.get(role, ROLE_TABS["Project Manager"])


@st.cache_data(show_spinner=False)
def compute_budget_view(building: str, week: int) -> dict:
    """Budget snapshot for a building at a given week.

    Reuses _planned_pct() and the actuals already in progress_df so the budget
    view is always consistent with the schedule view (no parallel data model).
    """
    budget = BUDGETS[building]
    bld = progress_df[progress_df["Building"] == building]
    row = bld[bld["Week"] == week].iloc[0]
    actual_pct  = float(row["Actual %"])
    planned_pct = float(row["Planned %"])

    planned_spend = budget * planned_pct / 100.0
    spent         = budget * actual_pct  / 100.0
    # Committed = spent + outstanding POs ≈ 8% of remaining work locked in
    committed     = min(budget, spent + budget * 0.08)

    # Burn rate from the last 2 weeks of actuals
    burn = 0.0
    if week >= 2:
        prev = bld[bld["Week"] == week - 1].iloc[0]
        burn = budget * (actual_pct - float(prev["Actual %"])) / 100.0
    burn = max(burn, budget * 0.04)  # floor so it never reads as zero

    # Delay $ — variance × daily overhead. A 1% schedule slip ≈ 0.7 days at the
    # current burn pace; this is a coarse but defensible model.
    variance_pct = actual_pct - planned_pct
    delay_days = max(0.0, -variance_pct) * 0.7
    delay_cost_sar = delay_days * DAILY_OVERHEAD[building]

    # Projected overrun: linear extrapolation of variance to project end
    remaining_pct = max(0.0, 100.0 - actual_pct)
    overrun_factor = (-variance_pct / 100.0) * 1.2 if variance_pct < 0 else 0.0
    projected_overrun_sar = budget * remaining_pct / 100.0 * overrun_factor + delay_cost_sar

    return {
        "budget":               budget,
        "spent":                spent,
        "committed":            committed,
        "planned_spend":        planned_spend,
        "burn_rate_sar_per_week": burn,
        "delay_cost_sar":       delay_cost_sar,
        "projected_overrun_sar": projected_overrun_sar,
        "variance_pct":         variance_pct,
    }


@st.cache_data(show_spinner=False)
def build_phase_progress(building: str, week: int) -> pd.DataFrame:
    """Per-phase progress + downstream-impact propagation.

    A phase's "delay_weeks" is derived deterministically from a building-seeded
    RNG; downstream phases inherit a fraction of upstream slip via PHASE_DEPS.
    """
    rng = np.random.RandomState(abs(hash(building)) % (2**31))
    actual_pct = float(progress_df[(progress_df["Building"] == building) &
                                   (progress_df["Week"] == week)]["Actual %"].iloc[0])

    # Cumulative weight of phases preceding the current "completion frontier"
    cum = 0.0
    phase_stats = {}
    for ph in PHASES:
        ph_start = cum
        ph_end   = cum + PHASE_WEIGHT[ph] * 100.0
        cum = ph_end
        if actual_pct >= ph_end:
            local = 100.0
        elif actual_pct <= ph_start:
            local = 0.0
        else:
            local = (actual_pct - ph_start) / (ph_end - ph_start) * 100.0
        own_delay = round(rng.uniform(-0.4, 1.2), 2) if local > 0 else 0.0
        phase_stats[ph] = {"actual_pct": local, "own_delay": max(0.0, own_delay)}

    # Propagate delays through PHASE_DEPS — each upstream slip adds 60% downstream
    impact = {ph: 0.0 for ph in PHASES}
    for upstream, downstream in PHASE_DEPS:
        impact[downstream] = max(impact[downstream],
                                 impact[upstream] + phase_stats[upstream]["own_delay"] * 0.6)

    rows = []
    for ph in PHASES:
        s = phase_stats[ph]
        planned = min(100.0, max(0.0,
            (actual_pct - sum(PHASE_WEIGHT[p] * 100.0 for p in PHASES[:PHASES.index(ph)]))
            / max(0.01, PHASE_WEIGHT[ph])
        ))
        if s["actual_pct"] >= 100.0:
            status = "Complete"
        elif s["actual_pct"] <= 0.0:
            status = "Not started"
        elif s["own_delay"] > 0.5:
            status = "Delayed"
        else:
            status = "On track"
        rows.append({
            "Phase":            ph,
            "Status":           status,
            "Planned %":        round(planned, 1),
            "Actual %":         round(s["actual_pct"], 1),
            "Delay (weeks)":    round(s["own_delay"], 1),
            "Downstream impact (weeks)": round(impact[ph], 1),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_dependency_sankey(building: str, week: int) -> go.Figure:
    """Sankey of phase dependencies, colored by delay propagation."""
    phases_df = build_phase_progress(building, week)
    delay_by_phase = dict(zip(phases_df["Phase"], phases_df["Delay (weeks)"]))
    impact_by_phase = dict(zip(phases_df["Phase"], phases_df["Downstream impact (weeks)"]))

    def node_color(ph):
        d = delay_by_phase.get(ph, 0.0) + impact_by_phase.get(ph, 0.0)
        if d >= 1.0:   return "#ef4444"   # red — significant delay
        if d >= 0.4:   return "#f59e0b"   # amber — at risk
        return AMETHYST                    # brand purple — on track

    def link_color(src, dst):
        d = delay_by_phase.get(src, 0.0)
        if d >= 0.5:   return "rgba(239,68,68,0.45)"
        if d >= 0.2:   return "rgba(245,158,11,0.45)"
        return "rgba(105,71,158,0.30)"

    labels = [f"{p} ({impact_by_phase.get(p, 0):.1f}w)" for p in PHASES]
    node_idx = {p: i for i, p in enumerate(PHASES)}
    src = [node_idx[u] for u, _ in PHASE_DEPS]
    dst = [node_idx[v] for _, v in PHASE_DEPS]
    vals = [max(1.0, PHASE_WEIGHT[u] * 100.0) for u, _ in PHASE_DEPS]
    link_cols = [link_color(u, v) for u, v in PHASE_DEPS]

    fig = go.Figure(go.Sankey(
        node=dict(pad=18, thickness=18, line=dict(color=DEEP_INDIGO, width=0.5),
                  label=labels, color=[node_color(p) for p in PHASES]),
        link=dict(source=src, target=dst, value=vals, color=link_cols),
    ))
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=DEEP_PLUM),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        paper_bgcolor="white",
    )
    return fig


@st.cache_data(show_spinner=False, ttl=900)
def fetch_market_prices() -> pd.DataFrame:
    """Live-feel market prices. ttl=900 makes the 'updated N min ago' caption
    rotate naturally on each browser session.

    Swap this function out for a real LME / supplier API later — the schema is
    intentionally identical to a typical commodity feed.
    """
    rows = []
    rng = np.random.RandomState(int(date.today().toordinal()))
    for m in MARKET_MATERIALS:
        jitter = rng.uniform(-0.012, 0.012)
        rows.append({
            "Material":     m["material"],
            "Price":        round(m["price"] * (1 + jitter), 2),
            "Unit":         m["unit"],
            "30-day change %": m["change_30d_pct"],
            "Supply":       m["supply"],
            "Source":       m["source"],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_procurement_plan(building: str, week: int) -> pd.DataFrame:
    """Upcoming material orders for the next 4 weeks for a given building.

    Quantities are scaled by the building's budget so TallBuilding orders ~7×
    what BasicHouse does — internally consistent with the BUDGETS table.
    """
    market = fetch_market_prices().set_index("Material")
    scale = BUDGETS[building] / BUDGETS["BasicHouse"]
    rng = np.random.RandomState(abs(hash(building)) % (2**31))

    schedule = [
        # (material name, base qty per week, primary phase)
        ("Steel rebar (12 mm)", 2.5, "Structure"),
        ("OPC cement",          120,  "Foundation"),
        ("Aggregate 20 mm",     35,   "Foundation"),
        ("Copper LV cable",     180,  "MEP"),
        ("Diesel",              420,  "Site logistics"),
        ("Float glass 8 mm",    24,   "Envelope"),
        ("HVAC ducting (galv.)",18,   "MEP"),
        ("Skilled labor",       6,    "All trades"),
    ]
    rows = []
    for offset in range(4):
        target_w = week + offset
        target_label = week_to_label(target_w) if target_w <= NUM_WEEKS + 4 else f"W+{offset}"
        for name, base_qty, phase in schedule:
            qty = round(base_qty * scale * rng.uniform(0.85, 1.15), 1)
            unit_price = float(market.loc[name, "Price"])
            unit       = market.loc[name, "Unit"]
            cost = round(qty * unit_price, 0)
            urgency = "Urgent" if offset <= 1 and market.loc[name, "Supply"] == "Tight" else "Plan"
            rows.append({
                "Need by":  target_label,
                "Material": name,
                "Phase":    phase,
                "Quantity": qty,
                "Unit":     unit.replace("SAR / ", ""),
                "Est. cost (SAR)": cost,
                "Urgency":  urgency,
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def apply_what_if(building: str, week: int,
                  crews: float = 3.0, hours: float = 50.0,
                  lead_time_reduction: float = 0.0) -> dict:
    """Apply scenario adjustments and return both the adjusted DataFrame and
    a fresh forecast computed via the same forecast_completion() helper.

    Scalar args (instead of a dict) so the result can be cached and the
    What-If sliders don't recompute on every keystroke.
    """
    # Productivity multiplier vs baseline (3 crews × 50 hours, no lead-time gain)
    prod = (float(crews) / 3.0) * (float(hours) / 50.0) * (1.0 + float(lead_time_reduction) * 0.6)
    prod = max(0.4, min(2.0, prod))

    bld = progress_df[progress_df["Building"] == building].copy().sort_values("Week")
    base_actual = float(bld[bld["Week"] == week]["Actual %"].iloc[0])

    # Re-baseline future weeks: keep history identical, scale future delta by prod
    weeks_arr = bld["Week"].to_numpy()
    actual_arr = bld["Actual %"].to_numpy(dtype=float)
    future_mask = weeks_arr > week
    new_actual = actual_arr.copy()
    if future_mask.any():
        scaled = base_actual + (actual_arr[future_mask] - base_actual) * prod
        new_actual[future_mask] = np.minimum(100.0, scaled)
        # Enforce monotonicity (cumulative progress can never decrease)
        new_actual = np.maximum.accumulate(new_actual)
    adj_df = bld.copy()
    adj_df["Actual %"] = np.round(new_actual, 1)

    fc_adj = forecast_completion(adj_df[adj_df["Week"] <= NUM_WEEKS])
    fc_base = forecast_completion(bld[bld["Week"] <= NUM_WEEKS])

    bv_base = compute_budget_view(building, week)
    delay_delta_weeks = fc_base["week"] - fc_adj["week"]
    cost_delta_sar   = delay_delta_weeks * 7 * DAILY_OVERHEAD[building]

    return {
        "adjusted_df":  adj_df,
        "forecast":     fc_adj,
        "baseline_fc":  fc_base,
        "productivity": prod,
        "delay_delta_weeks": round(delay_delta_weeks, 1),
        "cost_delta_sar":    round(cost_delta_sar, 0),
        "baseline_overrun":  bv_base["projected_overrun_sar"],
    }


@st.cache_data(show_spinner=False)
def design_phase_tasks(building: str) -> dict:
    """Design-phase fixtures: compliance checklist, RFI register, drawing revs.

    All deterministic per building so the same project always tells the same
    story across reruns and roles.
    """
    rng = np.random.RandomState(abs(hash("design:" + building)) % (2**31))
    statuses = ["Approved", "In review", "Pending", "Approved", "Approved"]
    today = date.today()

    compliance = []
    for c in COMPLIANCE_ITEMS:
        s = statuses[rng.randint(0, len(statuses))]
        due = today + timedelta(days=int(rng.randint(-14, 28)))
        compliance.append({
            "Item":      c["item"],
            "Authority": c["authority"],
            "Status":    s,
            "Due":       due.strftime("%b %d"),
            "Owner":     ["Eng. R.", "Arch. D.", "MEP S.", "Site PM"][rng.randint(0, 4)],
        })

    rfi_topics = [
        "Door schedule clarification — Type D-04",
        "Slab edge detail at curtain wall",
        "MEP riser routing — core 2",
        "Fire-rated ceiling spec confirmation",
        "Stair stringer reinforcement detail",
        "Glazing IGU build-up sign-off",
    ]
    n_rfi = int(rng.randint(3, 6))
    rfis = []
    for i in range(n_rfi):
        rfis.append({
            "RFI #":    f"RFI-{abs(hash(building)) % 900 + 100:03d}-{i+1:02d}",
            "Subject":  rfi_topics[rng.randint(0, len(rfi_topics))],
            "Raised":   (today - timedelta(days=int(rng.randint(2, 30)))).strftime("%b %d"),
            "Status":   ["Open", "Answered", "Closed"][rng.randint(0, 3)],
            "Owner":    ["Architect", "Structural", "MEP"][rng.randint(0, 3)],
        })

    drawing_revs = []
    for code, name in [("A", "Architectural"), ("S", "Structural"),
                       ("M", "Mechanical"),    ("E", "Electrical")]:
        rev_letter = chr(ord("A") + int(rng.randint(1, 5)))
        drawing_revs.append({
            "Discipline": name,
            "Sheet":      f"{code}-101",
            "Revision":   f"Rev {rev_letter}",
            "Issued":     (today - timedelta(days=int(rng.randint(1, 21)))).strftime("%b %d"),
            "Status":     ["For construction", "For review", "Superseded"][rng.randint(0, 3)],
        })

    return {
        "compliance":   pd.DataFrame(compliance),
        "rfis":         pd.DataFrame(rfis),
        "drawing_revs": pd.DataFrame(drawing_revs),
    }


@st.cache_data(show_spinner="Analyzing IFC compliance…")
def analyze_sbc_compliance(ifc_bytes: bytes) -> dict:
    """Parse an uploaded IFC file and test it against Saudi Building Code (SBC).

    Checks 5 SBC codes: Structural (301), Fire/Life Safety (801),
    Accessibility (1001), MEP (501), Energy/Mostadam (601).
    Returns model_info, per-code check results, and an overall score.
    """
    import tempfile as _tmpmod
    import os as _os
    tmp = _tmpmod.NamedTemporaryFile(delete=False, suffix=".ifc")
    tmp.write(ifc_bytes)
    tmp.close()
    try:
        model = ifcopenshell.open(tmp.name)
    finally:
        _os.unlink(tmp.name)

    # ── Gather element counts ────────────────────────────────────────
    _types = [
        "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcColumn", "IfcBeam",
        "IfcDoor", "IfcWindow", "IfcStair", "IfcRoof", "IfcRailing",
        "IfcCovering", "IfcPlate", "IfcFurnishingElement",
        "IfcBuildingStorey", "IfcSpace",
        "IfcDistributionElement", "IfcFlowTerminal",
        "IfcPipeSegment", "IfcDuctSegment",
        "IfcFlowSegment", "IfcEnergyConversionDevice",
    ]

    def _safe_count(t):
        try:
            return len(model.by_type(t) or [])
        except RuntimeError:          # entity not in schema (e.g. IFC2X3)
            return 0

    counts = {t: _safe_count(t) for t in _types}

    storeys  = counts["IfcBuildingStorey"]
    walls    = counts["IfcWall"] + counts["IfcWallStandardCase"]
    columns  = counts["IfcColumn"]
    beams    = counts["IfcBeam"]
    slabs    = counts["IfcSlab"]
    doors    = counts["IfcDoor"]
    windows  = counts["IfcWindow"]
    stairs   = counts["IfcStair"]
    roofs    = counts["IfcRoof"]
    coverings = counts["IfcCovering"]
    railings = counts["IfcRailing"]
    spaces   = counts["IfcSpace"]
    mep_total = sum(counts[t] for t in [
        "IfcDistributionElement", "IfcFlowTerminal",
        "IfcPipeSegment", "IfcDuctSegment",
        "IfcFlowSegment", "IfcEnergyConversionDevice",
    ])
    total_products = len(model.by_type("IfcProduct"))

    projects = model.by_type("IfcProject")
    model_info = {
        "schema": model.schema,
        "project_name": projects[0].Name if projects else "Unknown",
        "storeys": storeys,
        "total_products": total_products,
    }

    checks = []

    # ── SBC 301: Structural ──────────────────────────────────────────
    v, score = [], 100.0
    if columns == 0:
        v.append("No columns (IfcColumn) found — structural frame undefined")
        score -= 30
    elif storeys > 0 and columns / max(1, storeys) < 2:
        v.append(f"Column-to-storey ratio {columns / max(1, storeys):.1f} is below the minimum 2.0")
        score -= 15
    if slabs == 0:
        v.append("No slabs (IfcSlab) found — floor/roof diaphragms missing")
        score -= 25
    if beams == 0 and storeys > 1:
        v.append("No beams (IfcBeam) in a multi-storey building")
        score -= 20
    if walls == 0:
        v.append("No walls defined — lateral load path incomplete")
        score -= 25
    checks.append({
        "code": "SBC 301", "name": "Structural",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{columns} columns · {beams} beams · {slabs} slabs · {walls} walls · {storeys} storey(s)",
    })

    # ── SBC 801: Fire Protection / Life Safety ───────────────────────
    v, score = [], 100.0
    if storeys >= 2 and stairs == 0:
        v.append(f"No stairs (IfcStair) in a {storeys}-storey building — no vertical egress")
        score -= 40
    if doors == 0:
        v.append("No doors found — egress routes undefined")
        score -= 30
    elif storeys > 0 and doors / max(1, storeys) < 1:
        v.append(f"Only {doors} door(s) for {storeys} storey(s) — insufficient egress capacity")
        score -= 20
    if storeys >= 3 and stairs < 2:
        v.append(f"Only {stairs} stairway(s) for a {storeys}-storey building — SBC requires 2+ above 2 storeys")
        score -= 20
    checks.append({
        "code": "SBC 801", "name": "Fire Protection / Life Safety",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{doors} doors · {stairs} stairs · {storeys} storey(s)",
    })

    # ── SBC 1001: Accessibility ──────────────────────────────────────
    v, score = [], 100.0
    if doors == 0:
        v.append("No doors modelled — cannot verify accessible entrances")
        score -= 40
    if storeys >= 2 and stairs == 0:
        v.append("No vertical circulation elements — wheelchair access unverifiable")
        score -= 30
    if spaces == 0:
        v.append("No IfcSpace definitions — cannot verify accessible routes/clearances")
        score -= 20
    if railings == 0 and storeys >= 2:
        v.append("No railings defined for a multi-storey building")
        score -= 10
    checks.append({
        "code": "SBC 1001", "name": "Accessibility",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{doors} doors · {spaces} spaces · {railings} railings",
    })

    # ── SBC 501: MEP Systems ─────────────────────────────────────────
    v, score = [], 100.0
    if mep_total == 0:
        v.append("No MEP elements found (distribution, terminals, pipes, ducts)")
        score = 20
    elif mep_total < max(1, storeys) * 3:
        v.append(f"Only {mep_total} MEP element(s) for {storeys} storey(s) — likely incomplete")
        score -= 25
    if counts["IfcDuctSegment"] == 0 and counts["IfcFlowSegment"] == 0:
        v.append("No HVAC ductwork modelled")
        score -= 15
    if counts["IfcPipeSegment"] == 0:
        v.append("No plumbing piping modelled")
        score -= 15
    checks.append({
        "code": "SBC 501", "name": "MEP Systems",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{mep_total} MEP elements total",
    })

    # ── SBC 601: Energy Efficiency / Mostadam ────────────────────────
    v, score = [], 100.0
    if windows > 0 and walls > 0:
        wwr = windows / (windows + walls)
        if wwr > 0.40:
            v.append(f"Window-to-wall ratio {wwr:.0%} exceeds the 40% maximum")
            score -= 25
        elif wwr < 0.10:
            v.append(f"Window-to-wall ratio {wwr:.0%} is below 10% — natural daylighting insufficient")
            score -= 15
    elif windows == 0:
        v.append("No windows modelled — daylighting and ventilation unverifiable")
        score -= 20
    if roofs > 0 and coverings == 0:
        v.append("Roof present but no IfcCovering — insulation layer not modelled")
        score -= 20
    if roofs == 0:
        v.append("No roof elements — building envelope incomplete for energy analysis")
        score -= 15
    checks.append({
        "code": "SBC 601", "name": "Energy Efficiency / Mostadam",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{windows} windows · {walls} walls · {roofs} roofs · {coverings} coverings",
    })

    overall_score = sum(c["score"] for c in checks) / len(checks)
    return {
        "model_info": model_info,
        "checks": checks,
        "overall_score": overall_score,
        "overall_pass": all(c["passed"] for c in checks),
    }


@st.cache_data(show_spinner=False)
def actionable_recommendations(building: str, week: int) -> list:
    """Concrete, SAR-quantified actions a PM can take this week.

    Cached so the Project Dashboard, Recommendations tab, and any other
    consumer share the same compute. Internally pulls the budget view and
    forecast it needs (both already cached).

    Each rec carries an apply_payload that the What-If tab consumes via
    st.session_state to pre-fill its sliders.
    """
    budget_view = compute_budget_view(building, week)
    bld = progress_df[progress_df["Building"] == building]
    fc = forecast_completion(bld[bld["Week"] <= week])
    market = fetch_market_prices()
    recs = []

    # 1) Pre-buy any material that moved >5% AND is in the next-2-week plan
    proc = build_procurement_plan(building, week)
    next_2 = proc[proc["Need by"].isin(proc["Need by"].unique()[:2])]
    for _, m in market.iterrows():
        if m["30-day change %"] >= 5.0 and m["Material"] in next_2["Material"].values:
            qty_total = float(next_2[next_2["Material"] == m["Material"]]["Quantity"].sum())
            savings = round(qty_total * float(m["Price"]) * (m["30-day change %"] / 100.0) * 0.6, 0)
            recs.append({
                "title":    f"Pre-buy {m['Material']}",
                "category": "Procurement",
                "why":      f"{m['Material']} is up {m['30-day change %']:+.1f}% in 30 days; "
                            f"{qty_total:.1f} {m['Unit'].replace('SAR / ', '')} scheduled in the next 2 weeks.",
                "action":   f"Lock in {qty_total:.1f} {m['Unit'].replace('SAR / ', '')} today at "
                            f"SAR {m['Price']:,.2f} from {m['Source']}.",
                "est_savings_sar":    savings,
                "est_recovery_weeks": 0.0,
                "apply_payload":      {"lead_time_reduction": 0.15},
            })

    # 2) Schedule recovery if we're behind
    if budget_view["variance_pct"] < -2.0:
        recovery_weeks = round(abs(budget_view["variance_pct"]) / 8.0, 1)
        recs.append({
            "title":    "Add a second crew for the next 2 weeks",
            "category": "Schedule",
            "why":      f"Schedule variance is {budget_view['variance_pct']:+.1f}% — "
                        f"projected overrun SAR {budget_view['projected_overrun_sar']:,.0f}.",
            "action":   "Mobilize a second day-shift crew on the critical path. "
                        "Trade-off: ~SAR 38k extra labor for ~2 weeks recovered.",
            "est_savings_sar":    round(budget_view["projected_overrun_sar"] * 0.4, 0),
            "est_recovery_weeks": recovery_weeks,
            "apply_payload":      {"crews": 5, "hours": 60},
        })

    # 3) Forecast slipping past the planned end date
    if fc["week"] > NUM_WEEKS + 0.5:
        recs.append({
            "title":    "Compress finishes phase via parallel trades",
            "category": "Sequencing",
            "why":      f"Forecast completion is week {fc['week']:.1f} (vs target week {NUM_WEEKS}).",
            "action":   "Run flooring and ceiling crews in parallel zones. "
                        "Requires 2-hour daily clash-detection huddle.",
            "est_savings_sar":    round(DAILY_OVERHEAD[building] * 7 * 1.2, 0),
            "est_recovery_weeks": 1.2,
            "apply_payload":      {"hours": 65, "lead_time_reduction": 0.10},
        })

    # 4) Compliance auto-flag — any item with status "Pending" near due date
    comp = design_phase_tasks(building)["compliance"]
    pending = comp[comp["Status"] == "Pending"]
    if not pending.empty:
        recs.append({
            "title":    f"Resolve {len(pending)} pending compliance items",
            "category": "Compliance",
            "why":      f"{len(pending)} approval(s) blocking downstream phases: "
                        f"{', '.join(pending['Item'].head(2).tolist())}.",
            "action":   "Escalate to authority and assign owner; budget half-day each.",
            "est_savings_sar":    round(DAILY_OVERHEAD[building] * 3, 0),
            "est_recovery_weeks": 0.5,
            "apply_payload":      {"lead_time_reduction": 0.05},
        })

    # Sort by est savings descending so the highest-impact card is first
    recs.sort(key=lambda r: r["est_savings_sar"], reverse=True)
    return recs


# ═══════════════════════════════════════════════════════════════════════════
# BRAND ASSETS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _logo_data_uri(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""
    ext = p.suffix.lower().lstrip(".") or "png"
    mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


LOGO_FULL_URI = _logo_data_uri(str(LOGO_FULL))
LOGO_MARK_URI = _logo_data_uri(str(LOGO_MARK)) or LOGO_FULL_URI

# ═══════════════════════════════════════════════════════════════════════════
# 3D BIM VIEWER HELPERS
# ═══════════════════════════════════════════════════════════════════════════

PLANNED_COLOR     = "#d6cee3"   # muted lilac-gray for "planned but not yet built"
HIGHLIGHT_COLOR   = "#f59e0b"   # amber highlight for elements built THIS week
PLANNED_OPACITY   = 1.0         # fully solid — the BIM never changes shape
BUILT_OPACITY     = 1.0
HIGHLIGHT_OPACITY = 1.0


def annotate_elements(elements):
    """Classify each element by the week it first becomes 'built'.

    Returns a list of dicts (one per element, in phase-build order):
        idx         — original index in the input list
        elem        — the element dict
        pos         — position in the build sequence (0-based)
        build_week  — int 1..NUM_WEEKS or None if it stays planned beyond Week 4
    """
    sorted_elems = sorted(
        enumerate(elements),
        key=lambda ie: (TYPE_PHASE_ORDER.get(ie[1]["type"], 10), ie[1]["z_mid"]),
    )
    n_total = len(sorted_elems)
    out = []
    for pos, (idx, elem) in enumerate(sorted_elems):
        bw = None
        for w in range(1, NUM_WEEKS + 1):
            n = max(1, int(round(n_total * _planned_pct(w) / 100)))
            if pos < n:
                bw = w
                break
        out.append({"idx": idx, "elem": elem, "pos": pos, "build_week": bw})
    return out


def build_3d_figure(annotated, week):
    """Render the FULL BIM every week, recolouring built vs planned portions.

    Three colour states:
      - Built earlier weeks → real material colour (TYPE_COLORS)
      - Built THIS week     → amber HIGHLIGHT_COLOR (the visible delta)
      - Planned (future)    → muted lilac-gray PLANNED_COLOR

    Geometry is always identical week-over-week; only the colouring shifts.
    Caller passes the already-annotated list (from annotate_elements) so we
    don't pay the sort/loop twice per render.
    """
    if not annotated:
        return None

    fig = go.Figure()

    for item in annotated:
        elem = item["elem"]
        bw = item["build_week"]
        verts = elem["verts"]
        faces = elem["faces"]

        if bw is None or bw > week:
            color, opacity = PLANNED_COLOR, PLANNED_OPACITY
            status = "Planned (future work)"
        elif bw == week:
            color, opacity = HIGHLIGHT_COLOR, HIGHLIGHT_OPACITY
            status = f"BUILT THIS WEEK (W{week})"
        else:
            color = TYPE_COLORS.get(elem["type"], "#D3D3D3")
            opacity = BUILT_OPACITY
            status = f"Built earlier — Week {bw}"

        hover = (
            f"<b>{elem['name']}</b><br>"
            f"Type: {elem['type'].replace('Ifc', '')}<br>"
            f"Status: {status}<br>"
            f"Element #{item['pos'] + 1} / {len(annotated)}"
            "<extra></extra>"
        )

        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color,
            opacity=opacity,
            flatshading=True,
            lighting=dict(ambient=0.58, diffuse=0.82, specular=0.16,
                          roughness=0.7, fresnel=0.15),
            lightposition=dict(x=120, y=200, z=300),
            hovertemplate=hover,
            showlegend=False,
        ))

    # Ground / pad
    all_v = np.vstack([item["elem"]["verts"] for item in annotated])
    xmin, xmax = all_v[:, 0].min() - 4, all_v[:, 0].max() + 4
    ymin, ymax = all_v[:, 1].min() - 4, all_v[:, 1].max() + 4
    z_ground = all_v[:, 2].min() - 0.05
    fig.add_trace(go.Mesh3d(
        x=[xmin, xmax, xmax, xmin], y=[ymin, ymin, ymax, ymax],
        z=[z_ground] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
        color=LILAC_HAZE, opacity=0.6, hoverinfo="skip", showlegend=False,
        flatshading=True,
        lighting=dict(ambient=0.92, diffuse=0.3),
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.45, y=-1.55, z=1.05),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="perspective"),
            ),
            bgcolor=LILAC_HAZE,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=580,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE — project selection (drives the sidebar selectbox so other
# widgets like "View Dashboard →" buttons can switch the active project
# programmatically via on_click callbacks).
# ═══════════════════════════════════════════════════════════════════════════
if "selected_building" not in st.session_state:
    st.session_state.selected_building = BUILDINGS[0]
if "view_role" not in st.session_state:
    st.session_state.view_role = "Project Manager"
if "what_if_payload" not in st.session_state:
    st.session_state.what_if_payload = {}


def _switch_project(name: str) -> None:
    """Callback used by per-project buttons in the All Projects tab.

    Mutating session_state in an on_click callback is the supported pattern for
    changing a widget's bound value before its next render — Streamlit reruns the
    script automatically after the callback returns.
    """
    if name in BUILDINGS:
        st.session_state.selected_building = name


def _apply_what_if(payload: dict) -> None:
    """Callback used by Recommendation cards to pre-fill the What-If sliders.

    Same pattern as _switch_project: mutate session_state, Streamlit reruns and
    the What-If tab picks up the new values when the user navigates to it.
    """
    st.session_state.what_if_payload = dict(payload or {})


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if LOGO_MARK_URI:
        sb_logo_html = (
            f"<img src='{LOGO_MARK_URI}' alt='KPAnalytix' "
            f"style='max-height:62px; max-width:88%; margin:0 auto 0.4rem auto; display:block;'/>"
        )
    else:
        sb_logo_html = (
            f"<div style='font-size:1.8rem; font-weight:800; color:white; "
            f"letter-spacing:-0.02em;'>KPAnalytix</div>"
        )
    st.markdown(f"""
    <div style='text-align:center; padding:0.8rem 0 0.6rem 0;'>
        {sb_logo_html}
        <div style='font-size:0.78rem; color:{AMETHYST_TINT}; margin-top:0.35rem;
                    letter-spacing:0.04em;'>Construction Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)
    if st.session_state.get("active_phase", "Development") == "Development":
        st.markdown("---")
        view_role = st.radio(
            "View as",
            list(ROLE_TABS.keys()),
            key="view_role",
            horizontal=False,
            help="Switches the visible tabs. Executives see portfolio overview; PMs see assigned projects; Engineers see single-project BIM and schedule.",
        )

        if view_role == "Executive":
            # Executive sees all projects — no project selector
            selected_building = BUILDINGS[0]  # default for data computations
            selected_week = st.slider("Week", 1, NUM_WEEKS, NUM_WEEKS, key="week_slider")
        elif view_role == "Project Manager":
            selected_building = st.selectbox(
                "Project", PM_BUILDINGS, key="selected_building", format_func=display_name
            )
            selected_week = st.slider("Week", 1, NUM_WEEKS, NUM_WEEKS, key="week_slider")
        else:  # Engineer
            selected_building = st.selectbox(
                "Project", BUILDINGS, key="selected_building", format_func=display_name
            )
            selected_week = st.slider("Week", 1, NUM_WEEKS, NUM_WEEKS, key="week_slider")
    else:
        view_role = st.session_state.get("view_role", "Project Manager")
        selected_building = st.session_state.get("selected_building", BUILDINGS[0])
        selected_week = st.session_state.get("week_slider", NUM_WEEKS)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER  (rendered inside each phase tab so the tab bar appears above it)
# ═══════════════════════════════════════════════════════════════════════════
_header_logo = (
    f"<img src='{LOGO_FULL_URI}' alt='KPAnalytix' "
    f"style='height:46px; margin-right:1.1rem; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.25));'/>"
    if LOGO_FULL_URI else ""
)

def _render_header():
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{DEEP_PLUM} 0%,{ROYAL_PURPLE} 55%,{AMETHYST} 100%);
         color:white; padding:1.05rem 2rem; border-radius:0.7rem;
         margin-bottom:0.9rem; box-shadow:0 6px 22px rgba(36,15,62,0.28);
         border:1px solid rgba(255,255,255,0.10);'>
      <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div style='display:flex; align-items:center;'>
          {_header_logo}
          <div>
            <div style='font-size:1.55rem; font-weight:700; letter-spacing:-0.01em;'>
              Construction Intelligence Dashboard
            </div>
            <div style='opacity:0.82; font-size:0.86rem; margin-top:0.1rem;'>
              Real-time BIM analysis &middot; Drone progress tracking &middot; Delay forecasting
            </div>
          </div>
        </div>
        <div style='text-align:right; font-size:0.8rem; opacity:0.85;
                    background:rgba(255,255,255,0.08); padding:0.45rem 0.85rem;
                    border-radius:0.45rem; border:1px solid rgba(255,255,255,0.12);'>
          <div style='font-weight:600; letter-spacing:0.02em;'>{"Portfolio View \u2014 All Projects" if view_role == "Executive" else display_name(selected_building)}</div>
          <div style='opacity:0.85;'>Week {selected_week}/{NUM_WEEKS} &middot; {week_to_label(selected_week)}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════
info = load_ifc_analysis(selected_building)
bld_df = progress_df[progress_df["Building"] == selected_building].copy()
current_row = bld_df[bld_df["Week"] == selected_week].iloc[0]
actual = current_row["Actual %"]
planned = current_row["Planned %"]
variance = round(actual - planned, 1)
fc = forecast_completion(bld_df[bld_df["Week"] <= selected_week])

# ═══════════════════════════════════════════════════════════════════════════
# TAB RENDERERS — each function below renders one tab. The dispatch loop at
# the bottom of the file picks which functions to call based on view_role.
# Function bodies use 4-space indent (same as the original `with tab_X:` blocks)
# so no body re-indent was needed during refactor.
# ═══════════════════════════════════════════════════════════════════════════

def kpi(val, lbl, color=None):
    """Single KPI card — used by every tab's KPI strip.

    Pass `color` (any CSS color) to tint the value text — e.g. green for
    on-track KPIs, red for overruns. When omitted the brand default is used.
    """
    style = f' style="color:{color}"' if color else ""
    return f'<div class="kpi"><div class="val"{style}>{val}</div><div class="lbl">{lbl}</div></div>'


def variance_status(variance: float) -> tuple:
    """Map a schedule variance % to a (label, color) pair.

    Single source of truth for the on-track / minor-delay / behind-schedule
    thresholds — used by every KPI strip and project card so the colors and
    labels never drift out of sync.
    """
    if variance >= -2:
        return "On Track", "#22c55e"
    if variance >= -5:
        return "Minor Delay", "#f59e0b"
    return "Behind Schedule", "#ef4444"


def _phase_banner(tab_label: str) -> None:
    """Render a thin phase banner at the top of a tab body so users always know
    which lifecycle phase the current tab belongs to (Design or Development).
    """
    phase = TAB_PHASE.get(tab_label, PHASE_DEVELOPMENT)
    if phase == PHASE_DESIGN:
        bg, fg, border = "#fff7e6", "#b45309", "#f59e0b"
        icon = "✎"
    else:
        bg, fg, border = "#f0e9ff", DEEP_PLUM, AMETHYST
        icon = "⚙"
    st.html(
        f"<div style='display:flex; align-items:center; gap:0.6rem;"
        f" background:{bg}; border-left:4px solid {border};"
        f" padding:0.45rem 0.9rem; border-radius:0.4rem; margin-bottom:0.7rem;"
        f" font-size:0.78rem; color:{fg}; font-weight:600;"
        f" letter-spacing:0.04em; text-transform:uppercase;'>"
        f"<span style='font-size:1rem'>{icon}</span>"
        f"<span>{phase}</span>"
        f"<span style='opacity:0.55; font-weight:500; text-transform:none;"
        f" letter-spacing:0;'>· {tab_label}</span>"
        f"</div>"
    )


def _sar(value: float) -> str:
    """Format an SAR amount with thousands separator and a sensible suffix.

    Handles negative values by prefixing the sign before the SAR label so the
    Cost-Δ KPI on the What-If tab can pass raw deltas without an abs() wrapper.
    """
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1_000_000:
        return f"{sign}SAR {v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{sign}SAR {v/1_000:.0f}k"
    return f"{sign}SAR {v:.0f}"


# ─────────────────────────────────────────────────────────────────────────
# TAB — BIM MODEL VIEWER
# ─────────────────────────────────────────────────────────────────────────
def _render_bim_tab():
    geom = load_geometry(selected_building)

    col3d, col_img = st.columns([3, 2])

    with col3d:
        st.markdown('<div class="sec">Interactive 3D Model</div>', unsafe_allow_html=True)
        if geom:
            annotated = annotate_elements(geom)
            fig3d = build_3d_figure(annotated, selected_week)
            if fig3d:
                st.plotly_chart(
                    fig3d,
                    use_container_width=True,
                    key=f"bim3d_{selected_building}_{selected_week}",
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["toImage", "resetCameraLastSave3d"],
                    },
                )

            n_total = len(annotated)
            n_built_total = sum(1 for a in annotated if a["build_week"] and a["build_week"] <= selected_week)
            n_new_week = sum(1 for a in annotated if a["build_week"] == selected_week)
            pct = _planned_pct(selected_week)

            st.markdown(
                f"<div style='font-size:0.82rem; color:#555; margin-top:0.4rem; line-height:1.7;'>"
                f"<b>{n_built_total} / {n_total}</b> elements built ({pct:.0f}%) at Week {selected_week} "
                f"&middot; <b style='color:{HIGHLIGHT_COLOR};'>{n_new_week} new this week</b><br>"
                f"<span style='color:{HIGHLIGHT_COLOR}; font-size:1.1rem;'>■</span> built this week "
                f"&nbsp;·&nbsp; <span style='color:{AMETHYST}; font-size:1.1rem;'>■</span> built earlier "
                f"&nbsp;·&nbsp; <span style='color:{PLANNED_COLOR}; font-size:1.1rem;'>■</span> planned "
                f"— full BIM shown every week"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Element inspector ───────────────────────────────────────
            with st.expander("Inspect element details", expanded=False):
                def _opt_label(a):
                    bw = a["build_week"]
                    if bw is None:
                        badge = "Planned"
                    elif bw == selected_week:
                        badge = f"NEW W{bw}"
                    elif bw < selected_week:
                        badge = f"Built W{bw}"
                    else:
                        badge = f"Planned W{bw}"
                    t = a["elem"]["type"].replace("Ifc", "")
                    return f"[{badge}]  {a['elem']['name']}  ({t})"

                # Default to the first "new this week" element if any
                default_idx = 0
                for i, a in enumerate(annotated):
                    if a["build_week"] == selected_week:
                        default_idx = i
                        break

                picked = st.selectbox(
                    "Choose any element to inspect",
                    options=list(range(len(annotated))),
                    index=default_idx,
                    format_func=lambda i: _opt_label(annotated[i]),
                    key=f"elem_picker_{selected_building}_{selected_week}",
                )

                if picked is not None:
                    a = annotated[picked]
                    e = a["elem"]
                    bw = a["build_week"]
                    verts = e["verts"]
                    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
                    y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())
                    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
                    width, depth, height = x_max - x_min, y_max - y_min, z_max - z_min

                    if bw is None:
                        status_html = f"<span style='color:#888;'>Planned beyond Week {NUM_WEEKS}</span>"
                        accent = AMETHYST
                    elif bw == selected_week:
                        status_html = (
                            f"<span style='color:{HIGHLIGHT_COLOR}; font-weight:700;'>"
                            f"BUILT THIS WEEK &middot; {week_to_label(bw)}</span>"
                        )
                        accent = HIGHLIGHT_COLOR
                    elif bw < selected_week:
                        status_html = (
                            f"<span style='color:{AMETHYST}; font-weight:600;'>"
                            f"Built earlier &middot; Week {bw} ({week_to_label(bw)})</span>"
                        )
                        accent = AMETHYST
                    else:
                        status_html = (
                            f"<span style='color:#888;'>Planned for Week {bw} "
                            f"({week_to_label(bw)})</span>"
                        )
                        accent = PLANNED_COLOR

                    st.markdown(
                        f"""<div style='background:white; padding:1rem 1.2rem; border-radius:0.55rem;
                                      border-left:4px solid {accent};
                                      box-shadow:0 2px 10px rgba(36,15,62,0.08); margin-top:0.4rem;'>
                            <div style='font-weight:700; color:{DEEP_PLUM}; font-size:1.05rem;'>
                                {e['name']}
                            </div>
                            <div style='font-size:0.78rem; color:#777; margin-top:0.2rem;'>
                                Type: <b>{e['type'].replace('Ifc', '')}</b>
                                &nbsp;·&nbsp; Element #{a['pos'] + 1} of {len(annotated)}
                            </div>
                            <div style='font-size:0.88rem; margin-top:0.55rem;'>{status_html}</div>
                            <div style='font-size:0.78rem; color:#666; margin-top:0.7rem; line-height:1.65;'>
                                Bounding box: <b>{width:.2f} × {depth:.2f} × {height:.2f} m</b><br>
                                Z-range: {z_min:.2f} m → {z_max:.2f} m<br>
                                Mesh: {len(e['verts'])} vertices &middot; {len(e['faces'])} faces
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            st.warning(
                f"Geometry cache not found for {display_name(selected_building)}. "
                "Run `python preprocess_ifc.py` first."
            )

    with col_img:
        st.markdown('<div class="sec">Drone Capture</div>', unsafe_allow_html=True)
        img_path = DRONE_DIR / selected_building / f"Week{selected_week:02d}_aerial.jpg"
        if img_path.exists():
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(str(img_path), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Run `python generate_weekly_drone_images.py` to generate images.")

        # Element stats
        st.markdown('<div class="sec">Element Summary</div>', unsafe_allow_html=True)
        if info["elements"]:
            stat_df = pd.DataFrame(
                list(info["elements"].items()), columns=["Type", "Count"]
            ).sort_values("Count", ascending=False)
            st.dataframe(stat_df, use_container_width=True, hide_index=True, height=200)

    # ── Weekly Progress Timeline (4 thumbnails of the BIM) ───────────
    st.markdown('<div class="sec">Weekly BIM Progress Timeline</div>', unsafe_allow_html=True)
    st.caption("Hover over a thumbnail to see what was built that week. The current week is highlighted.")
    strip_cols = st.columns(NUM_WEEKS)
    for idx, w in enumerate(range(1, NUM_WEEKS + 1)):
        with strip_cols[idx]:
            is_current = (w == selected_week)
            border = HIGHLIGHT_COLOR if is_current else "transparent"
            week_pct = _planned_pct(w)
            st.html(
                f"<div style='border:2px solid {border}; border-radius:0.5rem;"
                f" padding:0.4rem; margin-bottom:0.4rem; background:white;"
                f" box-shadow:0 2px 8px rgba(36,15,62,0.08);'>"
                f"<div style='font-size:0.75rem; font-weight:700; color:{DEEP_PLUM};'>"
                f"Week {w} &middot; {week_to_label(w)}</div>"
                f"<div style='font-size:0.7rem; color:#666;'>{week_pct:.0f}% planned</div>"
                f"</div>"
            )
            thumb_path = DRONE_DIR / selected_building / f"Week{w:02d}_aerial.jpg"
            if thumb_path.exists():
                st.image(str(thumb_path), use_container_width=True)

    # ── Open in CAD panel ────────────────────────────────────────────
    st.markdown('<div class="sec">Open in CAD / BIM Software</div>', unsafe_allow_html=True)
    st.caption("Download the IFC file and open it in your preferred CAD/BIM tool to make edits.")

    cad_left, cad_right = st.columns([1, 2])
    with cad_left:
        ifc_data = _ifc_bytes(selected_building)
        if ifc_data:
            st.download_button(
                "⤓  Download IFC file",
                data=ifc_data,
                file_name=f"{selected_building}.ifc",
                mime="application/octet-stream",
                key=f"ifc_dl_{selected_building}_{view_role}",
                use_container_width=True,
            )
            st.caption(f"IFC schema: {info['schema']}  ·  {info['total_products']:,} products")
        else:
            st.warning("IFC file not found.")
    with cad_right:
        for tool in CAD_TOOLS:
            st.html(
                f"<div style='background:white; border-left:4px solid {AMETHYST};"
                f" padding:0.55rem 0.9rem; border-radius:0.4rem; margin-bottom:0.4rem;"
                f" box-shadow:0 2px 6px rgba(36,15,62,0.07);'>"
                f"<b style='color:{DEEP_PLUM}'>{tool['name']}</b> &nbsp; "
                f"<span style='font-size:0.78rem; color:#555'>{tool['blurb']}</span>"
                f"</div>"
            )


# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — PROJECT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────
def _render_dash_tab(show_financials=True):
    # KPI row
    status_label, status_color = variance_status(variance)
    proj_date = week_to_label(fc["week"])
    bv = compute_budget_view(selected_building, selected_week)
    overrun_color = "#ef4444" if bv["projected_overrun_sar"] > 0 else "#22c55e"

    if show_financials:
        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        with k1: st.markdown(kpi(info["total_products"], "IFC Products"), unsafe_allow_html=True)
        with k2: st.markdown(kpi(info["storeys"], "Storeys"), unsafe_allow_html=True)
        with k3: st.markdown(kpi(f"{actual:.0f}%", f"Week {selected_week} Progress"), unsafe_allow_html=True)
        with k4: st.markdown(kpi(status_label, f"Variance {variance:+.1f}%", color=status_color),
                             unsafe_allow_html=True)
        with k5: st.markdown(kpi(proj_date, "Projected Completion"), unsafe_allow_html=True)
        with k6: st.markdown(kpi(_sar(bv["budget"]), "Budget"), unsafe_allow_html=True)
        with k7: st.markdown(kpi(_sar(bv["projected_overrun_sar"]), "Projected Overrun",
                                 color=overrun_color), unsafe_allow_html=True)
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: st.markdown(kpi(info["total_products"], "IFC Products"), unsafe_allow_html=True)
        with k2: st.markdown(kpi(info["storeys"], "Storeys"), unsafe_allow_html=True)
        with k3: st.markdown(kpi(f"{actual:.0f}%", f"Week {selected_week} Progress"), unsafe_allow_html=True)
        with k4: st.markdown(kpi(status_label, f"Variance {variance:+.1f}%", color=status_color),
                             unsafe_allow_html=True)
        with k5: st.markdown(kpi(proj_date, "Projected Completion"), unsafe_allow_html=True)

    # S-curve progress + Cost vs schedule (side-by-side, scoped to this project).
    if show_financials:
        sc_col, cost_col = st.columns([3, 2])
    else:
        sc_col = st.container()
    with sc_col:
        st.markdown(f'<div class="sec">S-Curve Progress — {display_name(selected_building)}</div>',
                    unsafe_allow_html=True)

        # Project Dashboard's S-curve is intentionally Planned vs Actual only.
        # The forward-looking forecast (with confidence band) lives exclusively in
        # the Delay Analysis & Forecast tab to avoid duplicate charts.
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=bld_df["Week Label"], y=bld_df["Planned %"],
            mode="lines+markers", name="Planned",
            line=dict(color=DEEP_INDIGO, width=3, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        ))
        actual_so_far = bld_df[bld_df["Week"] <= selected_week]
        fig_sc.add_trace(go.Scatter(
            x=actual_so_far["Week Label"], y=actual_so_far["Actual %"],
            mode="lines+markers", name="Actual",
            line=dict(color=LAVENDER, width=3),
            marker=dict(size=8),
            fill="tonexty", fillcolor="rgba(105,71,158,0.10)",
        ))
        fig_sc.update_layout(
            yaxis=dict(title="Completion %", range=[0, 108]),
            xaxis=dict(title="", tickangle=-45),
            template="plotly_white", height=380,
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
            margin=dict(l=40, r=20, t=30, b=60),
        )
        st.plotly_chart(fig_sc, use_container_width=True, key=f"sc_{selected_building}_{selected_week}")

    if show_financials:
        with cost_col:
            st.markdown('<div class="sec">Cost vs Schedule</div>', unsafe_allow_html=True)
            fig_cost = go.Figure()
            budget = BUDGETS[selected_building]
            sorted_bld = bld_df.sort_values("Week")
            labels = sorted_bld["Week Label"].tolist()
            planned_sar = (budget * sorted_bld["Planned %"] / 100.0).tolist()
            actual_sar  = (budget * sorted_bld["Actual %"]  / 100.0).tolist()
            fig_cost.add_trace(go.Bar(x=labels, y=planned_sar, name="Planned",
                                      marker_color=AMETHYST, opacity=0.6))
            fig_cost.add_trace(go.Bar(x=labels, y=actual_sar,  name="Actual",
                                      marker_color=DEEP_PLUM))
            fig_cost.update_layout(
                barmode="overlay", template="plotly_white", height=380,
                yaxis_title="SAR", xaxis_title=None,
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                margin=dict(l=40, r=10, t=30, b=60),
            )
            st.plotly_chart(fig_cost, use_container_width=True,
                            key=f"cost_sched_{selected_building}_{selected_week}")

    # Weekly drone timeline
    st.markdown('<div class="sec">Weekly Drone Timeline</div>', unsafe_allow_html=True)
    cols = st.columns(NUM_WEEKS)
    for i, col in enumerate(cols):
        w = i + 1
        img_path = DRONE_DIR / selected_building / f"Week{w:02d}_aerial.jpg"
        with col:
            if img_path.exists():
                border = f"3px solid {DEEP_INDIGO}" if w == selected_week else "2px solid #ddd"
                st.markdown(f'<div style="border:{border}; border-radius:0.4rem; overflow:hidden;">', unsafe_allow_html=True)
                st.image(str(img_path), caption=f"Week {w} — {WEEK_PROGRESS[w]:.0f}%",
                         use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.caption(f"Week {w}")

    # Element breakdown
    st.markdown('<div class="sec">IFC Element Breakdown</div>', unsafe_allow_html=True)
    el1, el2 = st.columns(2)
    with el1:
        if info["elements"]:
            edf = pd.DataFrame(list(info["elements"].items()), columns=["Type", "Count"]).sort_values("Count", ascending=True)
            fig_bar = px.bar(edf, x="Count", y="Type", orientation="h",
                             color="Count", color_continuous_scale=[[0, LAVENDER], [1, DEEP_INDIGO]])
            fig_bar.update_layout(template="plotly_white", height=340, showlegend=False,
                                  coloraxis_showscale=False, margin=dict(l=10, r=20, t=30, b=20),
                                  title=dict(text="Count by Type", font=dict(color=DEEP_INDIGO, size=13)))
            st.plotly_chart(fig_bar, use_container_width=True)

    with el2:
        if info["elements"]:
            pie_df = pd.DataFrame(list(info["elements"].items()), columns=["Type", "Count"])
            palette = [DEEP_PLUM, ROYAL_PURPLE, AMETHYST, "#8867b8", "#a888d0",
                       "#c4abe0", "#d8c3ec", "#5a3590", "#7654a8", "#9577c0",
                       "#b39bd2", "#3a1a5f", "#523080"][:len(pie_df)]
            fig_pie = px.pie(pie_df, values="Count", names="Type",
                             color_discrete_sequence=palette, hole=0.45)
            fig_pie.update_layout(template="plotly_white", height=340,
                                  margin=dict(l=10, r=10, t=30, b=20),
                                  title=dict(text="Distribution", font=dict(color=DEEP_INDIGO, size=13)))
            fig_pie.update_traces(textinfo="percent+label", textposition="outside")
            st.plotly_chart(fig_pie, use_container_width=True)

    # Top recommendations summary (3 highest-impact)
    recs = actionable_recommendations(selected_building, selected_week)
    if recs:
        st.markdown('<div class="sec">Top Recommendations</div>', unsafe_allow_html=True)
        for rec in recs[:3]:
            st.html(
                f"<div style='background:white; border-left:4px solid {AMETHYST};"
                f" padding:0.6rem 1rem; border-radius:0.45rem; margin-bottom:0.4rem;"
                f" box-shadow:0 2px 6px rgba(36,15,62,0.07);'>"
                f"<b style='color:{DEEP_PLUM}'>{rec['title']}</b> &middot; "
                f"<span style='color:#22c55e; font-weight:600'>{_sar(rec['est_savings_sar'])} potential savings</span> &middot; "
                f"<span style='color:#666; font-size:0.83rem'>{rec['action']}</span>"
                f"</div>"
            )
        st.caption("See the Recommendations tab for the full list and one-click actions.")


# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — DELAY ANALYSIS & FORECAST
# ─────────────────────────────────────────────────────────────────────────
def _render_delay_tab(show_financials=True):
    # Forecast KPIs — short "Mon DD" format keeps every box the same size
    proj_completion = week_to_label(fc["week"])
    conf_lo = week_to_label(fc["conf_lo"])
    conf_hi = week_to_label(fc["conf_hi"])
    planned_end = week_to_label(NUM_WEEKS)
    delay_weeks = max(0.0, fc["week"] - NUM_WEEKS)
    delay_cost_sar = delay_weeks * 7 * DAILY_OVERHEAD[selected_building]

    delay_col = "#ef4444" if delay_weeks > 0.5 else ("#f59e0b" if delay_weeks > 0 else "#22c55e")
    cost_col_d  = "#ef4444" if delay_cost_sar > 0 else "#22c55e"

    if show_financials:
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1: st.markdown(kpi(proj_completion, "Projected Completion"), unsafe_allow_html=True)
        with d2: st.markdown(kpi(planned_end, "Planned Completion"), unsafe_allow_html=True)
        with d3: st.markdown(kpi(f"{delay_weeks:.1f} wk", "Expected Delay", color=delay_col),
                             unsafe_allow_html=True)
        with d4: st.markdown(kpi(_sar(delay_cost_sar), "Delay Cost (overhead)", color=cost_col_d),
                             unsafe_allow_html=True)
        with d5: st.markdown(kpi(f"{conf_lo} — {conf_hi}", "90% Confidence"), unsafe_allow_html=True)
    else:
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.markdown(kpi(proj_completion, "Projected Completion"), unsafe_allow_html=True)
        with d2: st.markdown(kpi(planned_end, "Planned Completion"), unsafe_allow_html=True)
        with d3: st.markdown(kpi(f"{delay_weeks:.1f} wk", "Expected Delay", color=delay_col),
                             unsafe_allow_html=True)
        with d4: st.markdown(kpi(f"{conf_lo} — {conf_hi}", "90% Confidence"), unsafe_allow_html=True)

    # Forecast S-curve
    st.markdown('<div class="sec">Progress Forecast with Confidence Band</div>', unsafe_allow_html=True)

    fig_fc = go.Figure()
    # Planned
    fig_fc.add_trace(go.Scatter(
        x=bld_df["Week Label"], y=bld_df["Planned %"],
        mode="lines", name="Planned",
        line=dict(color=DEEP_INDIGO, width=2, dash="dash"),
    ))
    # Actual (up to selected week)
    actual_df = bld_df[bld_df["Week"] <= selected_week]
    fig_fc.add_trace(go.Scatter(
        x=actual_df["Week Label"], y=actual_df["Actual %"],
        mode="lines+markers", name="Actual",
        line=dict(color=LAVENDER, width=3), marker=dict(size=8),
    ))
    # Forecast band — starts at the selected week so it overlaps the actual
    # series with no visual break.
    if fc["week"] > selected_week and fc.get("slope", 0) > 0:
        fc_weeks = list(range(selected_week, min(int(math.ceil(fc["conf_hi"])) + 2, NUM_WEEKS + 6)))
        slope = fc.get("slope", 25)
        base = actual_df["Actual %"].iloc[-1]
        fc_mid = [min(100, base + slope * (w - selected_week)) for w in fc_weeks]
        conf_spread = (fc["conf_hi"] - fc["conf_lo"]) / max(1, fc["week"] - selected_week) * 3
        fc_lo = [max(0, v - conf_spread * (i + 1) * 0.3) for i, v in enumerate(fc_mid)]
        fc_hi = [min(105, v + conf_spread * (i + 1) * 0.3) for i, v in enumerate(fc_mid)]
        fc_labels = [week_to_label(w) for w in fc_weeks]

        fig_fc.add_trace(go.Scatter(
            x=fc_labels, y=fc_hi, mode="lines", showlegend=False,
            line=dict(width=0), hoverinfo="skip",
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_labels, y=fc_lo, mode="lines", name="90% Confidence",
            fill="tonexty", fillcolor="rgba(245,158,11,0.15)",
            line=dict(width=0),
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_labels, y=fc_mid, mode="lines", name="Forecast",
            line=dict(color="#f59e0b", width=2, dash="dot"),
        ))

    fig_fc.update_layout(
        yaxis=dict(title="Completion %", range=[0, 108]),
        xaxis=dict(title="", tickangle=-45),
        template="plotly_white", height=380,
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=30, b=60),
    )
    st.plotly_chart(fig_fc, use_container_width=True, key=f"forecast_{selected_building}_{selected_week}")

    # ── Phase dependency Sankey — shows how a delay in one phase ripples
    #    forward through the construction sequence ──────────────────────
    st.markdown('<div class="sec">Phase Dependency Map</div>', unsafe_allow_html=True)
    st.caption("Each node is a construction phase; arrows are dependencies. "
               "Red = delay propagating downstream, amber = at risk, purple = on track. "
               "Numbers show downstream impact in weeks.")
    sankey_fig = build_dependency_sankey(selected_building, selected_week)
    st.plotly_chart(sankey_fig, use_container_width=True,
                    key=f"sankey_{selected_building}_{selected_week}")

    # Phase progress table — quick numerical view of the same data
    phase_df = build_phase_progress(selected_building, selected_week)
    st.dataframe(
        phase_df.style.format({
            "Planned %": "{:.1f}", "Actual %": "{:.1f}",
            "Delay (weeks)": "{:.1f}", "Downstream impact (weeks)": "{:.1f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # Lower row: velocity + delay causes + risk + recommendations
    vc1, vc2 = st.columns(2)

    with vc1:
        # Weekly velocity
        st.markdown('<div class="sec">Weekly Progress Velocity</div>', unsafe_allow_html=True)
        vel_rows = []
        for _, row in bld_df.iterrows():
            w = row["Week"]
            if w == 1:
                prev_a, prev_p = 0, 0
            else:
                prev = bld_df[bld_df["Week"] == w - 1].iloc[0]
                prev_a = prev["Actual %"]
                prev_p = prev["Planned %"]
            vel_rows.append({
                "Week": row["Week Label"],
                "Planned Rate": round(row["Planned %"] - prev_p, 1),
                "Actual Rate": round(row["Actual %"] - prev_a, 1),
            })
        vel_df = pd.DataFrame(vel_rows)
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Bar(x=vel_df["Week"], y=vel_df["Planned Rate"],
                                  name="Planned", marker_color=DEEP_INDIGO, opacity=0.5))
        fig_vel.add_trace(go.Bar(x=vel_df["Week"], y=vel_df["Actual Rate"],
                                  name="Actual", marker_color=LAVENDER))
        fig_vel.update_layout(template="plotly_white", height=310, barmode="group",
                              yaxis=dict(title="% / week"), xaxis=dict(tickangle=-45),
                              legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                              margin=dict(l=40, r=20, t=30, b=60))
        st.plotly_chart(fig_vel, use_container_width=True)

    with vc2:
        # Delay causes — aggregated bar chart driven by the incident catalogue
        st.markdown('<div class="sec">Delay Cause Breakdown</div>', unsafe_allow_html=True)
        incidents_df = delay_incidents_for(selected_building)
        causes_series = incidents_df.groupby("Cause")["Weeks Lost"].sum().round(2)
        causes = causes_series.to_dict()
        cause_df = (causes_series.reset_index()
                                  .sort_values("Weeks Lost", ascending=True))
        fig_cause = px.bar(cause_df, x="Weeks Lost", y="Cause", orientation="h",
                           color="Weeks Lost",
                           color_continuous_scale=[[0, LAVENDER], [1, "#ef4444"]])
        fig_cause.update_layout(template="plotly_white", height=310,
                                showlegend=False, coloraxis_showscale=False,
                                margin=dict(l=10, r=20, t=20, b=20))
        st.plotly_chart(
            fig_cause,
            use_container_width=True,
            key=f"cause_chart_{selected_building}",
        )

    # Reliable cause filter — pills are always visible, click immediately filters
    # the supplier table below. (Plotly bar-chart click events are unreliable
    # across Streamlit versions, so we drive the filter from this control.)
    # The per-building key suffix resets the pill when the user switches projects.
    selected_cause = st.pills(
        "Filter incidents by cause",
        options=list(causes.keys()),
        selection_mode="single",
        default=None,
        key=f"cause_pills_{selected_building}",
    )

    # Full-width incident table — supplier / contractor accountability
    if selected_cause:
        st.markdown(
            f'<div class="sec">Delay Incidents — '
            f'Filtered: <span style="color:{HIGHLIGHT_COLOR};">{selected_cause}</span></div>',
            unsafe_allow_html=True,
        )
        detail_df = incidents_df[incidents_df["Cause"] == selected_cause]
        st.caption(f"Showing {len(detail_df)} incident(s) for **{selected_cause}**. "
                   "Click the pill again to clear, or pick another cause.")
    else:
        st.markdown(
            '<div class="sec">Delay Incidents — Supplier &amp; Contractor Detail</div>',
            unsafe_allow_html=True,
        )
        st.caption("Pick a cause above to filter the supplier table.")
        detail_df = incidents_df.copy()
    detail_df = detail_df.sort_values("Weeks Lost", ascending=False).reset_index(drop=True)

    def _status_emoji(s):
        return {"Open": "🔴 Open", "Mitigating": "🟠 Mitigating", "Resolved": "🟢 Resolved"}.get(s, s)

    detail_display = detail_df.copy()
    detail_display["Status"] = detail_display["Status"].map(_status_emoji)

    st.dataframe(
        detail_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cause": st.column_config.TextColumn("Cause", width="small"),
            "Vendor / Contractor": st.column_config.TextColumn("Vendor / Contractor", width="medium"),
            "Item / Issue": st.column_config.TextColumn("Item / Issue", width="large"),
            "Expected": st.column_config.TextColumn("Expected", width="small"),
            "Actual": st.column_config.TextColumn("Actual", width="small"),
            "Weeks Lost": st.column_config.NumberColumn("Weeks Lost", format="%.1f"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
    )

    top_offender = (
        detail_df.groupby("Vendor / Contractor")["Weeks Lost"].sum().sort_values(ascending=False)
    )
    if len(top_offender):
        leader, leader_weeks = top_offender.index[0], top_offender.iloc[0]
        st.markdown(
            f"<div style='font-size:0.82rem; color:#666; margin-top:0.4rem;'>"
            f"Top contributor: <b style='color:{DEEP_INDIGO};'>{leader}</b> "
            f"&mdash; {leader_weeks:.1f} weeks lost across {int((detail_df['Vendor / Contractor'] == leader).sum())} incident(s)."
            f"</div>",
            unsafe_allow_html=True,
        )

    # Risk matrix + recommendations
    r1, r2 = st.columns(2)

    with r1:
        st.markdown('<div class="sec">Risk Matrix</div>', unsafe_allow_html=True)
        risks = [
            {"Risk": "Extended rain season", "Likelihood": "High", "Impact": "Medium",
             "score": 6, "color": "#f59e0b"},
            {"Risk": "Steel delivery delay", "Likelihood": "Medium", "Impact": "High",
             "score": 6, "color": "#f59e0b"},
            {"Risk": "Labour strike", "Likelihood": "Low", "Impact": "High",
             "score": 3, "color": "#22c55e"},
            {"Risk": "Design revision (client)", "Likelihood": "Medium", "Impact": "Medium",
             "score": 4, "color": "#f59e0b"},
            {"Risk": "Equipment breakdown", "Likelihood": "Low", "Impact": "Low",
             "score": 1, "color": "#22c55e"},
        ]
        for risk in risks:
            st.markdown(f"""<div class="risk-card">
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <div style='font-weight:600; color:{DEEP_INDIGO};'>{risk["Risk"]}</div>
                        <div style='font-size:0.78rem; color:#888;'>
                            Likelihood: {risk["Likelihood"]} &middot; Impact: {risk["Impact"]}
                        </div>
                    </div>
                    <div style='background:{risk["color"]}; color:white; padding:0.2rem 0.7rem;
                                border-radius:1rem; font-size:0.75rem; font-weight:600;'>
                        {risk["Likelihood"]}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="sec">Recommendations</div>', unsafe_allow_html=True)
        recs = []
        if variance < -5:
            recs.append(("Schedule Recovery", "Consider adding weekend shifts or a second crew to recover the "
                         f"{abs(variance):.0f}% gap. Current velocity is below plan.", "#ef4444"))
        elif variance < -2:
            recs.append(("Monitor Closely", "Minor variance detected. Review next week's "
                         "milestones and confirm material deliveries are on schedule.", "#f59e0b"))
        else:
            recs.append(("On Track", "Project is progressing as planned. Maintain current "
                         "pace and continue regular inspections.", "#22c55e"))

        if delay_weeks > 0.3:
            recs.append(("Delay Mitigation", f"Forecast shows ~{delay_weeks:.1f} week delay. "
                         "Recommend fast-tracking MEP rough-in and pre-ordering finishing materials.", "#f59e0b"))

        top_cause = max(causes, key=causes.get)
        recs.append(("Risk Focus", f"Largest delay contributor: {top_cause} "
                     f"({causes[top_cause]:.1f} weeks). Develop contingency plan.", LAVENDER))

        next_w = min(selected_week + 1, NUM_WEEKS)
        recs.append(("Next Milestone", f"Week {next_w}: "
                     f"Target {_planned_pct(next_w):.0f}% completion. "
                     "Ensure subcontractor availability.", DEEP_INDIGO))

        for title, desc, color in recs:
            st.markdown(f"""<div class="rec-card" style="border-left-color:{color};">
                <div style='font-weight:600; color:{DEEP_INDIGO}; margin-bottom:0.3rem;'>{title}</div>
                <div style='font-size:0.85rem; color:#555; line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — ALL PROJECTS (portfolio overview & navigation)
# ─────────────────────────────────────────────────────────────────────────
def _render_all_tab(buildings=None):
    _buildings = buildings or BUILDINGS
    st.markdown('<div class="sec">Portfolio Overview</div>', unsafe_allow_html=True)

    # Pre-load info for all buildings (cached)
    all_info = {b: load_ifc_analysis(b) for b in _buildings}
    wk_df = progress_df[progress_df["Week"] == selected_week].copy()
    wk_df = wk_df[wk_df["Building"].isin(_buildings)].copy()

    total_projects = len(_buildings)
    total_elements = sum(d["total_products"] for d in all_info.values())
    total_storeys = sum(d["storeys"] for d in all_info.values())
    avg_progress = wk_df["Actual %"].mean()

    # Aggregate budget exposure across scoped projects
    portfolio_budget = sum(BUDGETS[b] for b in _buildings)
    portfolio_overrun = 0.0
    portfolio_delay_cost = 0.0
    for b in _buildings:
        bv_b = compute_budget_view(b, selected_week)
        portfolio_overrun    += bv_b["projected_overrun_sar"]
        portfolio_delay_cost += bv_b["delay_cost_sar"]

    exposure_col = "#ef4444" if portfolio_overrun > 0 else "#22c55e"

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    with p1: st.markdown(kpi(total_projects, "Active Projects"), unsafe_allow_html=True)
    with p2: st.markdown(kpi(f"{total_elements:,}", "IFC Elements (all)"), unsafe_allow_html=True)
    with p3: st.markdown(kpi(total_storeys, "Storeys (all)"), unsafe_allow_html=True)
    with p4: st.markdown(kpi(f"{avg_progress:.0f}%", f"Avg Progress · Week {selected_week}"),
                         unsafe_allow_html=True)
    with p5: st.markdown(kpi(_sar(portfolio_budget), "Portfolio Budget"), unsafe_allow_html=True)
    with p6: st.markdown(kpi(_sar(portfolio_overrun), "Total Exposure", color=exposure_col),
                         unsafe_allow_html=True)

    # Cross-building comparison chart
    st.markdown(
        f'<div class="sec">Progress by Project — Week {selected_week}</div>',
        unsafe_allow_html=True,
    )
    wk_df["Project"] = wk_df["Building"].map(display_name)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=wk_df["Project"], y=wk_df["Planned %"],
        name="Planned", marker_color=DEEP_INDIGO, opacity=0.55,
    ))
    fig_comp.add_trace(go.Bar(
        x=wk_df["Project"], y=wk_df["Actual %"],
        name="Actual", marker_color=LAVENDER,
    ))
    fig_comp.update_layout(
        yaxis=dict(title="Completion %", range=[0, 108]),
        xaxis=dict(title=""),
        barmode="group", template="plotly_white", height=340,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_comp, use_container_width=True, key="all_projects_comp_chart")

    # Per-project cards (2x2 grid). Each card has a "View Dashboard →" button
    # that switches the active project via the _switch_project on_click callback.
    st.markdown('<div class="sec">Projects</div>', unsafe_allow_html=True)
    st.caption("Click **View Dashboard →** to switch the active project, then open any other tab.")

    n = len(_buildings)
    grid_rows = [st.columns(2) for _ in range((n + 1) // 2)]
    for i, b in enumerate(_buildings):
        cell = grid_rows[i // 2][i % 2]
        info_b = all_info[b]
        bld_b = progress_df[progress_df["Building"] == b]
        cur_b = bld_b[bld_b["Week"] == selected_week].iloc[0]
        act_b = float(cur_b["Actual %"])
        pln_b = float(cur_b["Planned %"])
        var_b = round(act_b - pln_b, 1)
        is_active = (b == selected_building)

        status_label, status_col = variance_status(var_b)

        border_col = HIGHLIGHT_COLOR if is_active else AMETHYST
        active_badge = (
            f"<span style='background:{HIGHLIGHT_COLOR}; color:white; "
            f"padding:0.15rem 0.55rem; border-radius:1rem; font-size:0.65rem; "
            f"font-weight:700; letter-spacing:0.04em;'>● ACTIVE</span>"
            if is_active else ""
        )

        # Use st.html (NOT st.markdown) so the multi-line template is rendered
        # as raw HTML without markdown parsing. With st.markdown, when the
        # active_badge interpolation is empty for inactive cards, the resulting
        # whitespace-only line closes the HTML block early and the rest of the
        # card gets rendered as a markdown code block.
        card_html = (
            f"<div style=\"background:white; border-radius:0.65rem;"
            f" padding:1.05rem 1.25rem; margin-bottom:0.6rem;"
            f" border-left:5px solid {border_col};"
            f" box-shadow:0 4px 14px rgba(36,15,62,0.09);\">"
            f"<div style=\"display:flex; justify-content:space-between; align-items:center;\">"
            f"<div style=\"font-weight:700; color:{DEEP_PLUM}; font-size:1.1rem;\">{display_name(b)}</div>"
            f"{active_badge}"
            f"</div>"
            f"<div style=\"font-size:0.75rem; color:#888; margin-top:0.15rem;\">"
            f"{info_b['schema']} &middot; {info_b['storeys']} storeys"
            f" &middot; {info_b['total_products']:,} products"
            f"</div>"
            f"<div style=\"margin-top:0.7rem; display:flex; gap:1.4rem;\">"
            f"<div>"
            f"<div style=\"font-size:0.68rem; color:#888; text-transform:uppercase;"
            f" letter-spacing:0.05em; font-weight:600;\">Actual</div>"
            f"<div style=\"font-size:1.35rem; font-weight:700; color:{DEEP_PLUM};\">{act_b:.0f}%</div>"
            f"</div>"
            f"<div>"
            f"<div style=\"font-size:0.68rem; color:#888; text-transform:uppercase;"
            f" letter-spacing:0.05em; font-weight:600;\">Variance</div>"
            f"<div style=\"font-size:1.35rem; font-weight:700; color:{status_col};\">{var_b:+.1f}%</div>"
            f"</div>"
            f"<div>"
            f"<div style=\"font-size:0.68rem; color:#888; text-transform:uppercase;"
            f" letter-spacing:0.05em; font-weight:600;\">Status</div>"
            f"<div style=\"font-size:0.92rem; font-weight:700; color:{status_col};"
            f" margin-top:0.35rem;\">{status_label}</div>"
            f"</div>"
            f"</div>"
            f"</div>"
        )
        with cell:
            st.html(card_html)
            if is_active:
                st.caption(f"Currently active — open **BIM Viewer**, **Project Dashboard**, "
                           f"or **Delay Analysis** to explore {display_name(b)}.")
            else:
                st.button(
                    f"View {display_name(b)} Dashboard →",
                    key=f"goto_{b}",
                    on_click=_switch_project,
                    args=(b,),
                    use_container_width=True,
                )

    # Portfolio summary table
    st.markdown('<div class="sec">Portfolio Summary Table</div>', unsafe_allow_html=True)
    summary = []
    for b in _buildings:
        a = all_info[b]
        row_w = progress_df[(progress_df["Building"] == b) & (progress_df["Week"] == selected_week)]
        act = row_w["Actual %"].values[0] if len(row_w) else 0
        pln = row_w["Planned %"].values[0] if len(row_w) else 0
        v = round(act - pln, 1)
        bv_b = compute_budget_view(b, selected_week)
        summary.append({
            "Project": display_name(b),
            "Schema": a["schema"],
            "Storeys": a["storeys"],
            "Products": a["total_products"],
            f"Week {selected_week} %": act,
            "Variance": f"{v:+.1f}%",
            "Budget":   _sar(bv_b["budget"]),
            "Spent":    _sar(bv_b["spent"]),
            "Overrun":  _sar(bv_b["projected_overrun_sar"]),
            "Active":   "● ACTIVE" if b == selected_building else "",
        })
    st.dataframe(
        pd.DataFrame(summary),
        use_container_width=True,
        hide_index=True,
        column_config={
            f"Week {selected_week} %": st.column_config.ProgressColumn(
                f"Week {selected_week}", min_value=0, max_value=100, format="%.0f%%"
            ),
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# EXECUTIVE REPORT PDF
# ─────────────────────────────────────────────────────────────────────────
def _build_executive_report_pdf():
    from fpdf import FPDF

    def _latin1(txt):
        return txt.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Header bar
    pdf.set_fill_color(36, 15, 62)
    pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(8)
    pdf.cell(0, 10, _latin1("KPAnalytix Executive Report"), align="C",
             new_x="LMARGIN", new_y="NEXT")

    # Date / week
    pdf.set_font("Helvetica", "", 10)
    pdf.set_y(22)
    pdf.set_text_color(220, 220, 220)
    pdf.cell(0, 5, _latin1(f"Week {selected_week} Summary"), align="C",
             new_x="LMARGIN", new_y="NEXT")

    # Portfolio KPIs
    pdf.set_y(35)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Portfolio Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    total_budget = sum(BUDGETS.values())
    total_overrun = 0.0
    wk_all = progress_df[progress_df["Week"] == selected_week]
    avg_pct = wk_all["Actual %"].mean()
    for b in BUILDINGS:
        bv_b = compute_budget_view(b, selected_week)
        total_overrun += bv_b["projected_overrun_sar"]

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _latin1(f"Active Projects: {len(BUILDINGS)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _latin1(f"Average Progress: {avg_pct:.1f}%"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _latin1(f"Portfolio Budget: {_sar(total_budget)}"), new_x="LMARGIN", new_y="NEXT")
    overrun_label = _sar(total_overrun) if total_overrun > 0 else "None"
    pdf.cell(0, 6, _latin1(f"Total Exposure: {overrun_label}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Per-project summary table header
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Project Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 240, 245)
    col_w = [42, 25, 25, 25, 30, 30, 30]
    headers = ["Project", "Progress", "Planned", "Variance", "Budget", "Spent", "Overrun"]
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 9)
    pdf.set_fill_color(255, 255, 255)
    for b in BUILDINGS:
        row_w = wk_all[wk_all["Building"] == b]
        act = float(row_w["Actual %"].values[0]) if len(row_w) else 0
        pln = float(row_w["Planned %"].values[0]) if len(row_w) else 0
        v = round(act - pln, 1)
        bv_b = compute_budget_view(b, selected_week)
        vals = [
            display_name(b), f"{act:.0f}%", f"{pln:.0f}%", f"{v:+.1f}%",
            _sar(bv_b["budget"]), _sar(bv_b["spent"]),
            _sar(bv_b["projected_overrun_sar"]),
        ]
        for w, val in zip(col_w, vals):
            pdf.cell(w, 7, _latin1(val), border=1)
        pdf.ln()

    # Per-project detail pages
    for b in BUILDINGS:
        pdf.add_page()
        pdf.set_fill_color(36, 15, 62)
        pdf.rect(0, 0, 210, 20, style="F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_y(5)
        pdf.cell(0, 10, _latin1(display_name(b)), align="C",
                 new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(28)
        pdf.set_text_color(0, 0, 0)
        info_b = load_ifc_analysis(b)
        bv_b = compute_budget_view(b, selected_week)
        row_w = wk_all[wk_all["Building"] == b]
        act = float(row_w["Actual %"].values[0]) if len(row_w) else 0
        pln = float(row_w["Planned %"].values[0]) if len(row_w) else 0
        v = round(act - pln, 1)

        bld_b = progress_df[progress_df["Building"] == b]
        fc_b = forecast_completion(bld_b[bld_b["Week"] <= selected_week])
        delay_wk = max(0.0, fc_b["week"] - NUM_WEEKS)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Key Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _latin1(f"Schema: {info_b['schema']}  |  Storeys: {info_b['storeys']}  |  Products: {info_b['total_products']:,}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, _latin1(f"Progress: {act:.0f}%  |  Planned: {pln:.0f}%  |  Variance: {v:+.1f}%"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, _latin1(f"Budget: {_sar(bv_b['budget'])}  |  Spent: {_sar(bv_b['spent'])}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, _latin1(f"Projected Overrun: {_sar(bv_b['projected_overrun_sar'])}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, _latin1(f"Expected Delay: {delay_wk:.1f} weeks"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, _latin1(f"Projected Completion: {week_to_label(fc_b['week'])}"), new_x="LMARGIN", new_y="NEXT")

        sl, sc = variance_status(v)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _latin1(f"Status: {sl}"), new_x="LMARGIN", new_y="NEXT")

    # Footer on last page
    pdf.set_y(-25)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "Generated by KPAnalytix Construction Intelligence Platform", align="C")

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────
# TAB — PORTFOLIO OVERVIEW (Executive) & MY PROJECTS (PM)
# ─────────────────────────────────────────────────────────────────────────
def _render_portfolio_tab():
    dl_col, _ = st.columns([1, 4])
    with dl_col:
        pdf_bytes = _build_executive_report_pdf()
        st.download_button(
            "Download Executive Report", pdf_bytes,
            file_name="executive_report.pdf", mime="application/pdf",
            key="exec_report_dl",
        )
    _render_all_tab(BUILDINGS)


def _render_my_projects_tab():
    _render_all_tab(PM_BUILDINGS)


# ─────────────────────────────────────────────────────────────────────────
# TAB — BUDGET & PROCUREMENT
# ─────────────────────────────────────────────────────────────────────────
def _render_budget_tab():
    bv = compute_budget_view(selected_building, selected_week)

    st.markdown(f'<div class="sec">Budget Snapshot — {display_name(selected_building)}</div>',
                unsafe_allow_html=True)

    # KPI strip
    overrun_color = "#ef4444" if bv["projected_overrun_sar"] > 0 else "#22c55e"
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1: st.markdown(kpi(_sar(bv["budget"]),    "Total Budget"),   unsafe_allow_html=True)
    with b2: st.markdown(kpi(_sar(bv["spent"]),     "Spent to Date"),  unsafe_allow_html=True)
    with b3: st.markdown(kpi(_sar(bv["committed"]), "Committed (PO)"), unsafe_allow_html=True)
    with b4: st.markdown(kpi(_sar(bv["burn_rate_sar_per_week"]), "Burn / Week"),
                         unsafe_allow_html=True)
    with b5: st.markdown(kpi(_sar(bv["projected_overrun_sar"]), "Projected Overrun",
                             color=overrun_color), unsafe_allow_html=True)

    # Planned vs Actual S-curve in SAR
    st.markdown('<div class="sec">Planned Spend vs Actual</div>', unsafe_allow_html=True)
    budget = BUDGETS[selected_building]
    bld_full = progress_df[progress_df["Building"] == selected_building].sort_values("Week")
    week_labels   = bld_full["Week Label"].tolist()
    planned_spend = (budget * bld_full["Planned %"] / 100.0).tolist()
    actual_spend  = (budget * bld_full["Actual %"]  / 100.0).tolist()

    fig_spend = go.Figure()
    fig_spend.add_trace(go.Scatter(
        x=week_labels, y=planned_spend,
        name="Planned spend", mode="lines+markers",
        line=dict(color=AMETHYST, width=3),
    ))
    fig_spend.add_trace(go.Scatter(
        x=week_labels, y=actual_spend,
        name="Actual spend", mode="lines+markers",
        line=dict(color=DEEP_PLUM, width=3, dash="dot"),
    ))
    fig_spend.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="SAR", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_spend, use_container_width=True,
                    key=f"budget_spend_{selected_building}_{selected_week}")

    # Live market table
    st.markdown('<div class="sec">Live Market Materials</div>', unsafe_allow_html=True)
    market = fetch_market_prices()
    refresh_ago = (selected_week * 3 + abs(hash(selected_building)) % 7) % 15 + 1
    st.caption(f"Sources: Hadeed (SABIC) · Riyadh Cables · LME · Saudi Aramco · Yamama Cement   ·   "
               f"Updated {refresh_ago} min ago")

    def _color_change(v):
        if v >= 5:    return "color:#ef4444; font-weight:600"
        if v >= 2:    return "color:#f59e0b; font-weight:600"
        if v <= -2:   return "color:#22c55e; font-weight:600"
        return "color:#666"

    st.dataframe(
        market.style.format({"Price": "{:,.2f}", "30-day change %": "{:+.1f}%"})
                    .map(lambda v: _color_change(v) if isinstance(v, (int, float)) else "",
                         subset=["30-day change %"]),
        use_container_width=True, hide_index=True,
    )

    # Procurement plan
    st.markdown('<div class="sec">Procurement Schedule — Next 4 Weeks</div>',
                unsafe_allow_html=True)
    proc = build_procurement_plan(selected_building, selected_week)
    st.dataframe(
        proc.style.format({"Quantity": "{:,.1f}", "Est. cost (SAR)": "{:,.0f}"}),
        use_container_width=True, hide_index=True,
    )

    csv = proc.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⤓  Download procurement order (CSV)",
        data=csv,
        file_name=f"procurement_{selected_building}_W{selected_week:02d}.csv",
        mime="text/csv",
        key=f"proc_dl_{selected_building}_{selected_week}",
    )

    # Top exposures — materials moving >5% AND in next 2 weeks
    next_2 = proc[proc["Need by"].isin(proc["Need by"].unique()[:2])]
    movers = market[market["30-day change %"] >= 5.0]
    exposed = movers[movers["Material"].isin(next_2["Material"].unique())]
    if not exposed.empty:
        st.markdown('<div class="sec">⚠ Top Price Exposures</div>', unsafe_allow_html=True)
        for _, row in exposed.iterrows():
            qty_total = float(next_2[next_2["Material"] == row["Material"]]["Quantity"].sum())
            est_cost  = qty_total * float(row["Price"])
            st.html(
                f"<div style='background:white; border-left:4px solid #ef4444;"
                f" padding:0.75rem 1rem; border-radius:0.45rem; margin-bottom:0.5rem;"
                f" box-shadow:0 2px 8px rgba(36,15,62,0.07);'>"
                f"<b style='color:{DEEP_PLUM}'>{row['Material']}</b> &middot; "
                f"<span style='color:#ef4444; font-weight:600'>{row['30-day change %']:+.1f}%</span> "
                f"in 30 days &middot; {qty_total:,.1f} {row['Unit'].replace('SAR / ','')} "
                f"needed in the next 2 weeks &middot; estimated exposure "
                f"<b>{_sar(est_cost)}</b>"
                f"</div>"
            )


# ─────────────────────────────────────────────────────────────────────────
# TAB — WHAT-IF PLANNER
# ─────────────────────────────────────────────────────────────────────────
def _render_whatif_tab():
    st.markdown(f'<div class="sec">What-If Planner — {display_name(selected_building)}</div>',
                unsafe_allow_html=True)
    st.caption("Adjust the levers below to see the impact on completion date and budget. "
               "Click an Apply button on a Recommendation card to pre-fill these sliders.")

    # Pre-fill sliders from a recommendation that wrote to session_state
    payload = st.session_state.get("what_if_payload") or {}

    sld1, sld2, sld3 = st.columns(3)
    with sld1:
        crews = st.slider(
            "Crews on site", 1, 6, int(payload.get("crews", 3)),
            key=f"whatif_crews_{view_role}_{selected_building}",
            help="Baseline = 3 crews",
        )
    with sld2:
        hours = st.slider(
            "Weekly hours / crew", 40, 80, int(payload.get("hours", 50)), step=5,
            key=f"whatif_hours_{view_role}_{selected_building}",
            help="Baseline = 50 hours/week",
        )
    with sld3:
        lead = st.slider(
            "Material lead-time reduction (%)", 0, 30,
            int(payload.get("lead_time_reduction", 0.0) * 100),
            key=f"whatif_lead_{view_role}_{selected_building}",
            help="Pre-buy / expediting cuts lead time. Baseline = 0%.",
        )

    scenario = apply_what_if(
        selected_building, selected_week,
        crews=crews, hours=hours, lead_time_reduction=lead / 100.0,
    )

    # KPI delta strip
    base_fc = scenario["baseline_fc"]
    adj_fc  = scenario["forecast"]
    base_end = week_to_label(base_fc["week"])
    adj_end  = week_to_label(adj_fc["week"])
    delta_w  = scenario["delay_delta_weeks"]
    delta_sar = scenario["cost_delta_sar"]

    st.markdown('<div class="sec">Scenario Impact</div>', unsafe_allow_html=True)
    week_col = "#22c55e" if delta_w  >= 0 else "#ef4444"
    sar_col  = "#22c55e" if delta_sar >= 0 else "#ef4444"
    sar_sign = "+" if delta_sar >= 0 else ""

    d1, d2, d3, d4 = st.columns(4)
    with d1: st.markdown(kpi(f"{scenario['productivity']:.2f}×", "Productivity Multiplier"),
                         unsafe_allow_html=True)
    with d2: st.markdown(kpi(f"{base_end} → {adj_end}", "Completion Shift"),
                         unsafe_allow_html=True)
    with d3: st.markdown(kpi(f"{delta_w:+.1f} wk", "Schedule Δ", color=week_col),
                         unsafe_allow_html=True)
    with d4: st.markdown(kpi(f"{sar_sign}{_sar(delta_sar)}", "Cost Δ (saved)", color=sar_col),
                         unsafe_allow_html=True)

    # Side-by-side S-curve comparison
    st.markdown('<div class="sec">Plan vs Adjusted Plan</div>', unsafe_allow_html=True)
    base_df = progress_df[progress_df["Building"] == selected_building]
    adj_df  = scenario["adjusted_df"]

    fig_wi = go.Figure()
    fig_wi.add_trace(go.Scatter(
        x=base_df["Week Label"], y=base_df["Planned %"],
        name="Planned", mode="lines",
        line=dict(color=AMETHYST, width=2, dash="dot"),
    ))
    fig_wi.add_trace(go.Scatter(
        x=base_df["Week Label"], y=base_df["Actual %"],
        name="Current actual", mode="lines+markers",
        line=dict(color=DEEP_PLUM, width=3),
    ))
    fig_wi.add_trace(go.Scatter(
        x=adj_df["Week Label"], y=adj_df["Actual %"],
        name="Adjusted scenario", mode="lines+markers",
        line=dict(color="#22c55e", width=3),
    ))
    fig_wi.update_layout(
        height=360, margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="% complete", xaxis_title=None, yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_wi, use_container_width=True,
                    key=f"whatif_chart_{view_role}_{selected_building}_{selected_week}")

    # Save scenario (session-state only — capped at the most recent 20)
    save_col, _ = st.columns([1, 4])
    with save_col:
        if st.button("💾 Save scenario",
                     key=f"whatif_save_{view_role}_{selected_building}",
                     use_container_width=True):
            scenarios = st.session_state.setdefault("saved_scenarios", [])
            scenarios.append({
                "building": selected_building, "week": selected_week,
                "crews": crews, "hours": hours, "lead": lead,
                "delta_weeks": delta_w, "delta_sar": delta_sar,
            })
            del scenarios[:-20]
    saved = st.session_state.get("saved_scenarios", [])
    if saved:
        st.caption(f"{len(saved)} scenario(s) saved this session.")


# ─────────────────────────────────────────────────────────────────────────
# TAB — DESIGN & COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────
def _render_design_tab():
    st.markdown(f'<div class="sec">Design & Compliance — {display_name(selected_building)}</div>',
                unsafe_allow_html=True)
    st.caption("Track design approvals, RFIs, and drawing revisions before construction starts. "
               "Items here belong to the Design Phase that precedes the Development Phase.")

    # Phase 0 timeline strip — Design → Foundation → … → Handover
    actual_pct = float(progress_df[(progress_df["Building"] == selected_building) &
                                   (progress_df["Week"] == selected_week)]["Actual %"].iloc[0])
    cum = 0.0
    current_phase = PHASES[0]
    for ph in PHASES:
        cum += PHASE_WEIGHT[ph] * 100.0
        if actual_pct <= cum:
            current_phase = ph
            break

    pills = []
    cum = 0.0
    for ph in PHASES:
        ph_end = cum + PHASE_WEIGHT[ph] * 100.0
        if actual_pct >= ph_end:
            color = AMETHYST  # done
            badge = "✓"
        elif ph == current_phase:
            color = "#f59e0b"  # in progress
            badge = "●"
        else:
            color = "#cbd5e1"  # future
            badge = "○"
        cum = ph_end
        pills.append(
            f"<div style='flex:1; text-align:center; padding:0.5rem 0.3rem;"
            f" background:white; border-radius:0.4rem; border-top:3px solid {color};"
            f" box-shadow:0 2px 6px rgba(36,15,62,0.08);'>"
            f"<div style='font-size:1.1rem; color:{color}; font-weight:700'>{badge}</div>"
            f"<div style='font-size:0.72rem; color:{DEEP_PLUM}; font-weight:600;"
            f" margin-top:0.15rem'>{ph}</div>"
            f"</div>"
        )
    st.html(
        "<div style='display:flex; gap:0.5rem; margin:0.5rem 0 1rem 0;'>"
        + "".join(pills) + "</div>"
    )

    data = design_phase_tasks(selected_building)

    # Compliance checklist
    st.markdown('<div class="sec">Compliance Checklist</div>', unsafe_allow_html=True)
    comp = data["compliance"]

    def _badge_compliance(s):
        if s == "Approved":  return "background:#dcfce7; color:#166534"
        if s == "In review": return "background:#fef3c7; color:#92400e"
        return "background:#fee2e2; color:#991b1b"

    st.dataframe(
        comp.style.map(lambda s: _badge_compliance(s) if isinstance(s, str) else "",
                       subset=["Status"]),
        use_container_width=True, hide_index=True,
    )

    csv_comp = comp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⤓  Download compliance audit (CSV)",
        data=csv_comp,
        file_name=f"compliance_{selected_building}.csv",
        mime="text/csv",
        key=f"comp_dl_{selected_building}_{view_role}",
    )

    # RFIs and drawing revisions
    rfi_col, dwg_col = st.columns(2)
    with rfi_col:
        st.markdown('<div class="sec">RFI Register</div>', unsafe_allow_html=True)
        st.dataframe(data["rfis"], use_container_width=True, hide_index=True)
    with dwg_col:
        st.markdown('<div class="sec">Drawing Revisions</div>', unsafe_allow_html=True)
        st.dataframe(data["drawing_revs"], use_container_width=True, hide_index=True)

    # Auto-flag pending items
    pending = comp[comp["Status"] != "Approved"]
    if not pending.empty:
        st.markdown('<div class="sec">⚠ Items Blocking Development</div>', unsafe_allow_html=True)
        for _, row in pending.iterrows():
            st.html(
                f"<div style='background:white; border-left:4px solid #f59e0b;"
                f" padding:0.7rem 1rem; border-radius:0.45rem; margin-bottom:0.45rem;"
                f" box-shadow:0 2px 8px rgba(36,15,62,0.07);'>"
                f"<b style='color:{DEEP_PLUM}'>{row['Item']}</b> &middot; "
                f"<span style='color:#92400e; font-weight:600'>{row['Status']}</span> &middot; "
                f"due {row['Due']} &middot; owner {row['Owner']} &middot; "
                f"authority {row['Authority']}"
                f"</div>"
            )


# ─────────────────────────────────────────────────────────────────────────
# TAB — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────
def _render_recs_tab():
    st.markdown(f'<div class="sec">Actionable Recommendations — {display_name(selected_building)}</div>',
                unsafe_allow_html=True)
    st.caption("Each recommendation is a concrete action with an estimated SAR impact. "
               "Click 'Simulate' to pre-fill the What-If Planner with the suggested adjustment.")

    recs = actionable_recommendations(selected_building, selected_week)

    if not recs:
        st.success("No outstanding recommendations — project is on plan and on budget.")
        return

    # Summary KPI strip — total potential savings & recovery
    total_savings = sum(r["est_savings_sar"] for r in recs)
    total_recovery = sum(r["est_recovery_weeks"] for r in recs)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(kpi(len(recs), "Open Recommendations"), unsafe_allow_html=True)
    with s2:
        st.markdown(kpi(_sar(total_savings), "Total Potential Savings"),
                    unsafe_allow_html=True)
    with s3:
        st.markdown(kpi(f"{total_recovery:.1f} wk", "Total Schedule Recovery"),
                    unsafe_allow_html=True)

    # Render each recommendation as a card with two action buttons
    cat_color = {
        "Procurement": "#0ea5e9",
        "Schedule":    "#f59e0b",
        "Sequencing":  "#8b5cf6",
        "Compliance":  "#ef4444",
    }
    for i, rec in enumerate(recs):
        col_color = cat_color.get(rec["category"], AMETHYST)
        st.html(
            f"<div style='background:white; border-left:5px solid {col_color};"
            f" padding:1rem 1.25rem; border-radius:0.55rem; margin-bottom:0.7rem;"
            f" box-shadow:0 4px 14px rgba(36,15,62,0.08);'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<div style='font-weight:700; color:{DEEP_PLUM}; font-size:1.05rem;'>{rec['title']}</div>"
            f"<div style='background:{col_color}; color:white; padding:0.18rem 0.6rem;"
            f" border-radius:0.4rem; font-size:0.7rem; font-weight:700; letter-spacing:0.04em;"
            f" text-transform:uppercase;'>{rec['category']}</div>"
            f"</div>"
            f"<div style='font-size:0.83rem; color:#555; margin-top:0.45rem;'><b>Why:</b> {rec['why']}</div>"
            f"<div style='font-size:0.83rem; color:#555; margin-top:0.25rem;'><b>Action:</b> {rec['action']}</div>"
            f"<div style='display:flex; gap:1.4rem; margin-top:0.55rem; font-size:0.78rem;'>"
            f"<div><span style='color:#888'>Savings</span> &nbsp;"
            f"<b style='color:#22c55e'>{_sar(rec['est_savings_sar'])}</b></div>"
            f"<div><span style='color:#888'>Recovery</span> &nbsp;"
            f"<b style='color:{DEEP_PLUM}'>{rec['est_recovery_weeks']:.1f} wk</b></div>"
            f"</div>"
            f"</div>"
        )
        b1, b2, _ = st.columns([1, 1, 3])
        with b1:
            order_csv = pd.DataFrame([{
                "Project":  display_name(selected_building),
                "Action":   rec["title"],
                "Category": rec["category"],
                "Why":      rec["why"],
                "Detail":   rec["action"],
                "Est savings (SAR)": rec["est_savings_sar"],
                "Est recovery (weeks)": rec["est_recovery_weeks"],
            }]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⤓ Generate order",
                data=order_csv,
                file_name=f"order_{selected_building}_{i}.csv",
                mime="text/csv",
                key=f"rec_dl_{view_role}_{selected_building}_{i}",
                use_container_width=True,
            )
        with b2:
            st.button(
                "→ Simulate in What-If",
                key=f"rec_sim_{view_role}_{selected_building}_{i}",
                on_click=_apply_what_if,
                args=(rec["apply_payload"],),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────
# PHASE — DESIGN (IFC upload + SBC compliance checker)
# ─────────────────────────────────────────────────────────────────────────
def _render_design_phase():
    st.markdown(
        f'<div class="sec">Saudi Building Code (SBC) Compliance Checker</div>',
        unsafe_allow_html=True,
    )

    # ── Upload (default) with pilot toggle on the side ──────────────
    upload_col, toggle_col = st.columns([3, 1])
    with toggle_col:
        pilot_mode = st.toggle("Pilot / Testing", key="design_pilot_mode",
                               help="Use a built-in sample BIM model instead of uploading")
    ifc_bytes = None
    ifc_filename = ""
    geom = None  # pre-processed geometry (available for sample models)

    if pilot_mode:
        with upload_col:
            sample_bldg = st.selectbox(
                "Select sample project",
                BUILDINGS,
                format_func=display_name,
                key="design_sample_building",
            )
        ifc_bytes = _ifc_bytes(sample_bldg)
        ifc_filename = f"{sample_bldg}.ifc"
        geom = load_geometry(sample_bldg)
        if not ifc_bytes:
            st.warning(f"IFC file not found for {display_name(sample_bldg)}.")
            return
    else:
        with upload_col:
            uploaded = st.file_uploader(
                "Upload IFC file",
                type=["ifc"],
                key="design_ifc_upload",
                help="Drag and drop or browse for an .ifc file to analyze.",
            )
        if uploaded is None:
            st.info("Upload an IFC file to begin compliance analysis, or enable **Pilot / Testing** to use a sample model.")
            codes_data = [
                ("SBC 301", "Structural", "Verifies columns, beams, slabs, and walls for structural completeness"),
                ("SBC 801", "Fire Protection / Life Safety", "Checks egress stairs, doors, and fire safety elements"),
                ("SBC 1001", "Accessibility", "Audits accessible entrances, vertical circulation, and space definitions"),
                ("SBC 501", "MEP Systems", "Validates mechanical, electrical, and plumbing model presence"),
                ("SBC 601", "Energy Efficiency (Mostadam)", "Analyzes window-to-wall ratio, insulation, and envelope integrity"),
            ]
            for code, name, desc in codes_data:
                st.html(
                    f"<div style='background:white; border-left:4px solid {AMETHYST};"
                    f" padding:0.65rem 1rem; border-radius:0.45rem; margin-bottom:0.4rem;"
                    f" box-shadow:0 2px 6px rgba(36,15,62,0.07);'>"
                    f"<b style='color:{DEEP_PLUM}'>{code} — {name}</b><br>"
                    f"<span style='font-size:0.83rem; color:#555'>{desc}</span>"
                    f"</div>"
                )
            return
        ifc_bytes = uploaded.getvalue()
        ifc_filename = uploaded.name

    # ── Parse and analyze compliance ───────────────────────────────
    result = analyze_sbc_compliance(ifc_bytes)
    mi = result["model_info"]
    overall = result["overall_score"]
    pass_count = sum(1 for c in result["checks"] if c["passed"])
    total_checks = len(result["checks"])
    overall_color = "#22c55e" if result["overall_pass"] else ("#f59e0b" if overall >= 60 else "#ef4444")
    overall_label = "COMPLIANT" if result["overall_pass"] else "NON-COMPLIANT"

    # ── Minimal report download (top-right) ────────────────────────
    def _build_compliance_pdf():
        from fpdf import FPDF

        def _latin1(txt):
            return txt.encode("latin-1", errors="replace").decode("latin-1")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()
        # Header bar
        pdf.set_fill_color(36, 15, 62)
        pdf.rect(0, 0, 210, 28, style="F")
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(255, 255, 255)
        pdf.set_y(8)
        pdf.cell(0, 10, "SBC Compliance Report", align="C", new_x="LMARGIN", new_y="NEXT")
        # Meta
        pdf.set_y(34)
        pdf.set_text_color(60, 60, 60)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _latin1(f"Project: {mi['project_name']}  |  File: {ifc_filename}  |  {date.today().isoformat()}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, _latin1(f"Schema: {mi['schema']}  |  Storeys: {mi['storeys']}  |  Products: {mi['total_products']:,}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        # Overall score
        pdf.set_font("Helvetica", "B", 14)
        sc = (34, 197, 94) if result["overall_pass"] else (239, 68, 68)
        pdf.set_text_color(*sc)
        pdf.cell(0, 10, _latin1(f"Overall: {overall:.0f}% - {overall_label}  ({pass_count}/{total_checks} passed)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        # Per-code
        for chk in result["checks"]:
            pdf.set_draw_color(*(34, 197, 94) if chk["passed"] else (239, 68, 68))
            pdf.set_line_width(0.8)
            pdf.line(10, pdf.get_y(), 10, pdf.get_y() + 6)
            pdf.set_x(14)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(36, 15, 62)
            lbl = "PASS" if chk["passed"] else "FAIL"
            pdf.cell(0, 6, _latin1(f"[{lbl}]  {chk['code']} - {chk['name']}  ({chk['score']:.0f}%)"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(14)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, _latin1(chk["detail"]), new_x="LMARGIN", new_y="NEXT")
            for v in chk["violations"]:
                pdf.set_x(18)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(180, 50, 50)
                pdf.cell(0, 5, _latin1(f"- {v}"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
        # Footer
        pdf.set_y(-25)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 5, "Generated by KPAnalytix Design Phase", align="C")
        return bytes(pdf.output())

    _r_spacer, _r_btn = st.columns([4, 1])
    with _r_btn:
        st.download_button(
            "Export PDF",
            data=_build_compliance_pdf(),
            file_name=f"SBC_{mi['project_name']}_{date.today().isoformat()}.pdf",
            mime="application/pdf",
            key="design_compliance_report_dl",
        )

    # ── 3D BIM Model + Actions side-by-side ────────────────────────
    col_3d, col_actions = st.columns([3, 2])

    with col_3d:
        st.markdown('<div class="sec">3D BIM Model</div>', unsafe_allow_html=True)
        if geom:
            annotated = annotate_elements(geom)
            fig3d = build_3d_figure(annotated, NUM_WEEKS)
        else:
            with st.spinner("Extracting 3D geometry from IFC…"):
                elements = _extract_geometry_from_bytes(ifc_bytes)
            if elements:
                annotated = annotate_elements(elements)
                fig3d = build_3d_figure(annotated, NUM_WEEKS)
            else:
                annotated, fig3d = [], None

        if fig3d:
            st.plotly_chart(
                fig3d,
                use_container_width=True,
                key=f"design_3d_{ifc_filename}",
                config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["toImage", "resetCameraLastSave3d"],
                },
            )
            st.caption(f"{len(annotated)} structural elements rendered")
        else:
            st.info("No renderable geometry found in this IFC file.")

    with col_actions:
        # Model info
        st.markdown('<div class="sec">Model Info</div>', unsafe_allow_html=True)
        st.markdown(kpi(mi["project_name"], "Project"), unsafe_allow_html=True)
        info_a, info_b = st.columns(2)
        with info_a:
            st.markdown(kpi(mi["schema"], "Schema"), unsafe_allow_html=True)
            st.markdown(kpi(mi["storeys"], "Storeys"), unsafe_allow_html=True)
        with info_b:
            st.markdown(kpi(f"{mi['total_products']:,}", "Products"), unsafe_allow_html=True)
            overall = result["overall_score"]
            overall_color = "#22c55e" if result["overall_pass"] else ("#f59e0b" if overall >= 60 else "#ef4444")
            st.markdown(kpi(f"{overall:.0f}%", "SBC Score", overall_color), unsafe_allow_html=True)

        # CAD tools
        st.markdown('<div class="sec">Open in CAD</div>', unsafe_allow_html=True)
        for tool in CAD_TOOLS:
            st.html(
                f"<div style='background:white; border-left:3px solid {AMETHYST};"
                f" padding:0.4rem 0.7rem; border-radius:0.35rem; margin-bottom:0.35rem;"
                f" font-size:0.82rem; box-shadow:0 1px 4px rgba(36,15,62,0.06);'>"
                f"<b style='color:{DEEP_PLUM}'>{tool['name']}</b> — "
                f"<span style='color:#555'>{tool['blurb']}</span>"
                f"</div>"
            )
        dl_a, dl_b = st.columns(2)
        with dl_a:
            st.download_button(
                "Download IFC",
                data=ifc_bytes,
                file_name=ifc_filename,
                mime="application/octet-stream",
                key="design_phase_ifc_download",
                use_container_width=True,
            )
        with dl_b:
            st.markdown(
                f"<a href='autocad://open' target='_blank' style='display:block;"
                f" text-align:center; background:{DEEP_PLUM}; color:white;"
                f" padding:0.45rem 0.8rem; border-radius:0.4rem; font-weight:600;"
                f" text-decoration:none; font-size:0.88rem;"
                f" box-shadow:0 2px 6px rgba(36,15,62,0.12);'>"
                f"Open AutoCAD</a>",
                unsafe_allow_html=True,
            )

    # ── Overall compliance score ───────────────────────────────────
    st.html(
        f"<div style='background:white; border-radius:0.7rem; padding:1.2rem 1.5rem;"
        f" text-align:center; margin:1rem 0; border:2px solid {overall_color};"
        f" box-shadow:0 4px 14px rgba(36,15,62,0.10);'>"
        f"<div style='font-size:2rem; font-weight:800; color:{overall_color};'>"
        f"{overall:.0f}%</div>"
        f"<div style='font-size:1.1rem; font-weight:700; color:{overall_color};"
        f" letter-spacing:0.05em; margin-top:0.3rem;'>{overall_label}</div>"
        f"<div style='font-size:0.82rem; color:#888; margin-top:0.3rem;'>"
        f"{pass_count}/{total_checks} checks passed</div>"
        f"</div>"
    )

    # Per-code results
    st.markdown('<div class="sec">Compliance Report by SBC Code</div>', unsafe_allow_html=True)
    for check in result["checks"]:
        border = "#22c55e" if check["passed"] else "#ef4444"
        status_lbl = "PASS" if check["passed"] else "FAIL"
        status_bg  = "#dcfce7" if check["passed"] else "#fee2e2"
        status_fg  = "#166534" if check["passed"] else "#991b1b"

        violations_html = ""
        if check["violations"]:
            items = "".join(
                f"<li style='margin-bottom:0.3rem;'>{v}</li>" for v in check["violations"]
            )
            violations_html = (
                f"<div style='margin-top:0.5rem; font-size:0.82rem; color:#555;'>"
                f"<b>Violations:</b><ul style='margin:0.3rem 0 0 1.2rem; padding:0;'>"
                f"{items}</ul></div>"
            )

        st.html(
            f"<div style='background:white; border-left:5px solid {border};"
            f" padding:1rem 1.25rem; border-radius:0.55rem; margin-bottom:0.7rem;"
            f" box-shadow:0 4px 14px rgba(36,15,62,0.08);'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<div>"
            f"<span style='font-weight:700; color:{DEEP_PLUM}; font-size:1.05rem;'>"
            f"{check['code']}</span>"
            f"<span style='color:#666; margin-left:0.5rem;'>{check['name']}</span>"
            f"</div>"
            f"<div style='display:flex; align-items:center; gap:0.7rem;'>"
            f"<span style='font-weight:700; color:{DEEP_PLUM}; font-size:1.1rem;'>"
            f"{check['score']:.0f}%</span>"
            f"<span style='background:{status_bg}; color:{status_fg}; padding:0.2rem 0.7rem;"
            f" border-radius:1rem; font-size:0.72rem; font-weight:700;"
            f" letter-spacing:0.04em;'>{status_lbl}</span>"
            f"</div>"
            f"</div>"
            f"<div style='font-size:0.78rem; color:#888; margin-top:0.3rem;'>{check['detail']}</div>"
            f"{violations_html}"
            f"</div>"
        )



# ─────────────────────────────────────────────────────────────────────────
# PHASE — MAINTENANCE (placeholder)
# ─────────────────────────────────────────────────────────────────────────
def _render_maintenance_phase():
    st.html(
        f"<div style='text-align:center; padding:4rem 2rem;'>"
        f"<div style='font-size:3rem; opacity:0.3;'>&#128295;</div>"
        f"<div style='font-size:1.3rem; font-weight:700; color:{DEEP_INDIGO};"
        f" margin-top:1rem;'>Maintenance Phase</div>"
        f"<div style='font-size:0.95rem; color:#888; margin-top:0.5rem;"
        f" max-width:420px; margin-left:auto; margin-right:auto;'>"
        f"Asset management, preventive maintenance scheduling, and facility "
        f"operations monitoring are coming soon.</div>"
        f"</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE NAVIGATION + TAB DISPATCH
# ═══════════════════════════════════════════════════════════════════════════
TAB_RENDERERS = {
    "Portfolio Overview":    _render_portfolio_tab,
    "My Projects":           _render_my_projects_tab,
    "All Projects":          _render_all_tab,            # kept for safety
    "BIM Model Viewer":      _render_bim_tab,
    "Project Dashboard":     lambda: _render_dash_tab(show_financials=(view_role != "Engineer")),
    "Budget & Procurement":  _render_budget_tab,
    "Delay & Dependencies":  lambda: _render_delay_tab(show_financials=(view_role != "Engineer")),
    "What-If Planner":       _render_whatif_tab,
    "Design & Compliance":   _render_design_tab,
    "Recommendations":       _render_recs_tab,
}

# Phase selector — rendered in main area above the header
active_phase = st.pills(
    "Phase",
    ["Design", "Development", "Maintenance"],
    default="Development",
    key="active_phase",
)

_render_header()

# Phase dispatch
if active_phase == "Design":
    _render_design_phase()
elif active_phase == "Development":
    _visible_tab_labels = tabs_for_role(view_role)
    _tab_objs = st.tabs([f"  {lbl}  " for lbl in _visible_tab_labels])
    for _label, _tab_obj in zip(_visible_tab_labels, _tab_objs):
        with _tab_obj:
            _phase_banner(_label)
            TAB_RENDERERS[_label]()
else:
    _render_maintenance_phase()


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:2.5rem 0 1rem 0; color:#bbb; font-size:0.72rem;'>
    KPAnalytix Construction Intelligence Platform
</div>
""", unsafe_allow_html=True)
