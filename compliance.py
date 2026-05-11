"""
Construction Compliance Checker -- Saudi Building Code (SBC)

Standalone Streamlit application for analyzing IFC building models
against Saudi Building Code requirements. Features interactive 3D BIM
viewer with MEP system visualization, layer controls, automated
compliance analysis, and PDF reporting.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Construction Compliance Checker",
    page_icon=":building_construction:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
DEEP_PLUM     = "#1a0f2e"
ROYAL_PURPLE  = "#362060"
AMETHYST      = "#69479E"
LILAC_HAZE    = "#f7f5fa"
AMETHYST_TINT = "#b5a4d6"
LAVENDER      = "#c2b0e2"
DEEP_INDIGO   = "#2d1754"
SURFACE       = "#ffffff"
BORDER        = "#e9e5f0"
TEXT_PRIMARY   = "#1a1a2e"
TEXT_SECONDARY = "#6b7280"
TEXT_MUTED     = "#9ca3af"

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_MODELS = {
    "Duplex Residence": "Duplex_MEP.ifc",
    "Sample House (IFC4)": "SampleHouse.ifc",
    "Tall Building": "TallBuilding.ifc",
}

CODE_RECOMMENDATIONS = {
    "SBC 301": "Add missing structural elements (IfcColumn, IfcBeam, IfcSlab) to define the structural frame.",
    "SBC 801": "Model egress routes with IfcDoor and IfcStair elements per SBC fire-safety requirements.",
    "SBC 1001": "Define IfcSpace zones and ensure accessible entrances and vertical circulation are modelled.",
    "SBC 501": "Add MEP distribution elements (IfcPipeSegment, IfcDuctSegment, IfcFlowTerminal) to the model.",
    "SBC 601": "Review window-to-wall ratio and add IfcCovering for insulation to meet Mostadam standards.",
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.html(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  .stApp {{
      background: {LILAC_HAZE};
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  #MainMenu, footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{
      background: transparent !important;
      height: 0;
  }}
  [data-testid="collapsedControl"] {{
      display: block !important;
      visibility: visible !important;
      z-index: 999999;
      color: {TEXT_PRIMARY} !important;
  }}
  .block-container {{
      padding-top: 1rem;
      max-width: 1280px;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background: linear-gradient(175deg, {DEEP_PLUM} 0%, #1e1240 60%, {ROYAL_PURPLE} 100%);
      border-right: 1px solid rgba(255,255,255,0.06);
  }}
  section[data-testid="stSidebar"] * {{ color: #e0dae8 !important; }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stRadio label,
  section[data-testid="stSidebar"] .stFileUploader label,
  section[data-testid="stSidebar"] .stCheckbox label {{
      color: {AMETHYST_TINT} !important;
      font-weight: 600;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
  }}
  section[data-testid="stSidebar"] hr {{
      border-color: rgba(255,255,255,0.08) !important;
  }}

  /* Section headers */
  .sec {{
      font-size: 0.75rem;
      font-weight: 700;
      color: {TEXT_SECONDARY};
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin: 1.8rem 0 0.7rem 0;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid {BORDER};
  }}

  /* Cards */
  .card {{
      background: {SURFACE};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      transition: box-shadow 0.2s ease;
  }}
  .card:hover {{
      box-shadow: 0 4px 20px rgba(26,15,46,0.08);
  }}

  /* Download buttons — compact */
  .stDownloadButton > button {{
      background: {DEEP_PLUM} !important;
      color: white !important;
      border: none !important;
      font-weight: 600 !important;
      border-radius: 8px !important;
      font-size: 0.82rem !important;
      padding: 0.5rem 1.2rem !important;
  }}
  .stDownloadButton > button:hover {{
      background: {ROYAL_PURPLE} !important;
  }}

  /* Hide Streamlit elements */
  div[data-testid="stDecoration"] {{ display: none; }}
</style>
""")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def kpi(val, lbl, accent=None):
    c = accent or TEXT_PRIMARY
    return (
        f'<div style="text-align:center; padding:0.8rem 0.6rem;">'
        f'<div style="font-size:1.5rem; font-weight:800; color:{c}; letter-spacing:-0.02em;">{val}</div>'
        f'<div style="font-size:0.68rem; color:{TEXT_MUTED}; text-transform:uppercase;'
        f' letter-spacing:0.08em; margin-top:0.15rem; font-weight:600;">{lbl}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# COMPLIANCE ENGINE
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Analyzing compliance...")
def analyze_sbc_compliance(ifc_bytes: bytes) -> dict:
    """Parse an IFC file and check against Saudi Building Code (SBC)."""
    import ifcopenshell

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
    tmp.write(ifc_bytes)
    tmp.close()
    try:
        model = ifcopenshell.open(tmp.name)
    finally:
        os.unlink(tmp.name)

    _types = [
        "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcColumn", "IfcBeam",
        "IfcDoor", "IfcWindow", "IfcStair", "IfcRoof", "IfcRailing",
        "IfcCovering", "IfcPlate", "IfcFurnishingElement",
        "IfcBuildingStorey", "IfcSpace",
        "IfcDistributionElement", "IfcFlowTerminal",
        "IfcPipeSegment", "IfcDuctSegment",
        "IfcFlowSegment", "IfcEnergyConversionDevice",
        "IfcLightFixture", "IfcOutlet", "IfcSwitchingDevice",
    ]

    def _safe_count(t):
        try:
            return len(model.by_type(t) or [])
        except RuntimeError:
            return 0

    counts = {t: _safe_count(t) for t in _types}

    storeys   = counts["IfcBuildingStorey"]
    walls     = counts["IfcWall"] + counts["IfcWallStandardCase"]
    columns   = counts["IfcColumn"]
    beams     = counts["IfcBeam"]
    slabs     = counts["IfcSlab"]
    doors     = counts["IfcDoor"]
    windows   = counts["IfcWindow"]
    stairs    = counts["IfcStair"]
    roofs     = counts["IfcRoof"]
    coverings = counts["IfcCovering"]
    railings  = counts["IfcRailing"]
    spaces    = counts["IfcSpace"]
    mep_total = sum(counts[t] for t in [
        "IfcDistributionElement", "IfcFlowTerminal",
        "IfcPipeSegment", "IfcDuctSegment",
        "IfcFlowSegment", "IfcEnergyConversionDevice",
    ])
    electrical = sum(counts[t] for t in [
        "IfcLightFixture", "IfcOutlet", "IfcSwitchingDevice",
    ])
    total_products = len(model.by_type("IfcProduct"))

    projects = model.by_type("IfcProject")
    model_info = {
        "schema": model.schema,
        "project_name": projects[0].Name if projects else "Unknown",
        "storeys": storeys,
        "total_products": total_products,
        "mep_total": mep_total,
        "electrical": electrical,
    }

    checks = []

    # SBC 301: Structural
    v, score = [], 100.0
    if columns == 0:
        v.append("No columns (IfcColumn) found")
        score -= 30
    elif storeys > 0 and columns / max(1, storeys) < 2:
        v.append(f"Column-to-storey ratio {columns / max(1, storeys):.1f} below minimum 2.0")
        score -= 15
    if slabs == 0:
        v.append("No slabs (IfcSlab) found")
        score -= 25
    if beams == 0 and storeys > 1:
        v.append("No beams in multi-storey building")
        score -= 20
    if walls == 0:
        v.append("No walls defined")
        score -= 25
    checks.append({
        "code": "SBC 301", "name": "Structural",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{columns} columns, {beams} beams, {slabs} slabs, {walls} walls",
    })

    # SBC 801: Fire Protection / Life Safety
    v, score = [], 100.0
    if storeys >= 2 and stairs == 0:
        v.append(f"No stairs in {storeys}-storey building")
        score -= 40
    if doors == 0:
        v.append("No doors found")
        score -= 30
    elif storeys > 0 and doors / max(1, storeys) < 1:
        v.append(f"Only {doors} door(s) for {storeys} storey(s)")
        score -= 20
    if storeys >= 3 and stairs < 2:
        v.append(f"Only {stairs} stairway(s) for {storeys}-storey building")
        score -= 20
    checks.append({
        "code": "SBC 801", "name": "Fire / Life Safety",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{doors} doors, {stairs} stairs, {storeys} storey(s)",
    })

    # SBC 1001: Accessibility
    v, score = [], 100.0
    if doors == 0:
        v.append("No doors modelled")
        score -= 40
    if storeys >= 2 and stairs == 0:
        v.append("No vertical circulation elements")
        score -= 30
    if spaces == 0:
        v.append("No IfcSpace definitions")
        score -= 20
    if railings == 0 and storeys >= 2:
        v.append("No railings in multi-storey building")
        score -= 10
    checks.append({
        "code": "SBC 1001", "name": "Accessibility",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{doors} doors, {spaces} spaces, {railings} railings",
    })

    # SBC 501: MEP Systems
    v, score = [], 100.0
    if mep_total == 0:
        v.append("No MEP elements found")
        score = 20
    elif mep_total < max(1, storeys) * 3:
        v.append(f"Only {mep_total} MEP element(s) for {storeys} storey(s)")
        score -= 25
    if counts["IfcDuctSegment"] == 0 and counts["IfcFlowSegment"] == 0:
        v.append("No HVAC ductwork modelled")
        score -= 15
    if counts["IfcPipeSegment"] == 0:
        v.append("No plumbing piping modelled")
        score -= 15
    if counts["IfcFlowTerminal"] == 0:
        v.append("No flow terminals modelled")
        score -= 10
    if electrical == 0:
        v.append("No electrical elements detected")
        score -= 10
    checks.append({
        "code": "SBC 501", "name": "MEP Systems",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{mep_total} MEP, {electrical} electrical, "
                  f"{counts['IfcPipeSegment']} pipes, {counts['IfcDuctSegment']} ducts",
    })

    # SBC 601: Energy Efficiency / Mostadam
    v, score = [], 100.0
    if windows > 0 and walls > 0:
        wwr = windows / (windows + walls)
        if wwr > 0.40:
            v.append(f"WWR {wwr:.0%} exceeds 40% maximum")
            score -= 25
        elif wwr < 0.10:
            v.append(f"WWR {wwr:.0%} below 10%")
            score -= 15
    elif windows == 0:
        v.append("No windows modelled")
        score -= 20
    if roofs > 0 and coverings == 0:
        v.append("No insulation layer modelled")
        score -= 20
    if roofs == 0:
        v.append("No roof elements")
        score -= 15
    checks.append({
        "code": "SBC 601", "name": "Energy / Mostadam",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{windows} windows, {walls} walls, {roofs} roofs, {coverings} coverings",
    })

    overall_score = sum(c["score"] for c in checks) / len(checks)
    return {
        "model_info": model_info,
        "checks": checks,
        "overall_score": overall_score,
        "overall_pass": all(c["passed"] for c in checks),
    }


# ---------------------------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------------------------
def build_compliance_pdf(result, filename):
    from fpdf import FPDF

    mi = result["model_info"]
    overall = result["overall_score"]
    pass_count = sum(1 for c in result["checks"] if c["passed"])
    total_checks = len(result["checks"])
    overall_label = "COMPLIANT" if result["overall_pass"] else "NON-COMPLIANT"

    def _latin1(txt):
        return txt.encode("latin-1", errors="replace").decode("latin-1")

    class CompliancePDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 5, f"Construction Compliance Checker  |  Page {self.page_no()}", align="C")

    pdf = CompliancePDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover page
    pdf.add_page()
    pdf.set_fill_color(26, 15, 46)
    pdf.rect(0, 0, 210, 55, style="F")
    pdf.set_fill_color(54, 32, 96)
    pdf.rect(0, 50, 210, 5, style="F")

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(16)
    pdf.cell(0, 10, "SBC Compliance Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Construction Compliance Checker", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(65)
    pdf.set_text_color(60, 60, 60)
    pdf.set_font("Helvetica", "", 10)
    for line in [
        f"Project: {mi['project_name']}",
        f"File: {filename}",
        f"Date: {date.today().strftime('%B %d, %Y')}",
        f"Schema: {mi['schema']}  |  Storeys: {mi['storeys']}  |  Products: {mi['total_products']:,}",
    ]:
        pdf.cell(0, 6, _latin1(line), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 18)
    sc = (34, 197, 94) if result["overall_pass"] else (239, 68, 68)
    pdf.set_text_color(*sc)
    pdf.cell(0, 12, _latin1(f"{overall:.0f}% -- {overall_label}  ({pass_count}/{total_checks} passed)"),
             align="C", new_x="LMARGIN", new_y="NEXT")

    # Detail pages
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(26, 15, 46)
    pdf.cell(0, 10, "Detailed Analysis by SBC Code", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for chk in result["checks"]:
        pdf.set_draw_color(*(34, 197, 94) if chk["passed"] else (239, 68, 68))
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 10, pdf.get_y() + 6)
        pdf.set_x(14)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(26, 15, 46)
        lbl = "PASS" if chk["passed"] else "FAIL"
        pdf.cell(0, 6, _latin1(f"[{lbl}]  {chk['code']} -- {chk['name']}  ({chk['score']:.0f}%)"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(14)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, _latin1(chk["detail"]), new_x="LMARGIN", new_y="NEXT")
        for v in chk["violations"]:
            pdf.set_x(18)
            pdf.set_text_color(180, 50, 50)
            pdf.cell(0, 5, _latin1(f"- {v}"), new_x="LMARGIN", new_y="NEXT")
        if not chk["passed"]:
            rec = CODE_RECOMMENDATIONS.get(chk["code"], "")
            if rec:
                pdf.set_x(14)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(146, 64, 14)
                pdf.cell(0, 5, _latin1(f"Recommendation: {rec}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# 3D GEOMETRY EXTRACTION
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    "IfcWall": "#e8e3d8", "IfcWallStandardCase": "#e8e3d8",
    "IfcSlab": "#c4bdd6", "IfcColumn": "#8a82a0", "IfcBeam": "#a09ab8",
    "IfcRoof": "#a85d3a", "IfcDoor": "#7a5a3a", "IfcWindow": "#6db8e0",
    "IfcStair": "#c4a878", "IfcRailing": "#555560",
    "IfcCovering": "#d4c090", "IfcPlate": "#a89db0",
    "IfcFurnishingElement": "#b8a78c",
    "IfcPipeSegment": "#2196F3", "IfcPipeFitting": "#1976D2",
    "IfcDuctSegment": "#78909C", "IfcDuctFitting": "#607D8B",
    "IfcFlowSegment": "#26A69A", "IfcFlowTerminal": "#00897B",
    "IfcFlowFitting": "#00796B", "IfcDistributionElement": "#546E7A",
    "IfcEnergyConversionDevice": "#FF7043",
    "IfcLightFixture": "#FFD54F", "IfcOutlet": "#FFCA28",
    "IfcSwitchingDevice": "#FFC107",
}

_STRUCTURAL_TYPES = {
    "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcColumn", "IfcBeam",
    "IfcRoof", "IfcStair", "IfcDoor", "IfcWindow",
    "IfcRailing", "IfcCovering", "IfcPlate", "IfcFurnishingElement",
}

_VIEWER_TYPES = set(TYPE_COLORS.keys())


@st.cache_data(show_spinner="Extracting 3D geometry...")
def extract_geometry(ifc_bytes: bytes) -> list:
    """Extract 3D meshes from raw IFC bytes using ifcopenshell.geom."""
    import ifcopenshell
    import ifcopenshell.geom

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
    tmp.write(ifc_bytes)
    tmp.close()
    try:
        model = ifcopenshell.open(tmp.name)
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        products = [p for p in model.by_type("IfcProduct") if p.is_a() in _VIEWER_TYPES]

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
                elements.append({
                    "type": product.is_a(),
                    "name": product.Name or product.is_a(),
                    "verts": verts,
                    "faces": faces,
                })
            except Exception:
                continue
        return elements
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# SYNTHETIC MEP GENERATION
# ---------------------------------------------------------------------------
def _generate_synthetic_mep(elements: list) -> dict:
    """Generate realistic MEP system traces based on building geometry.

    Returns dict with keys: pipes, ducts, electrical.
    Each is a list of (x_list, y_list, z_list) line segments.
    """
    if not elements:
        return {"pipes": [], "ducts": [], "electrical": []}

    all_v = np.vstack([e["verts"] for e in elements])
    xmin, xmax = float(all_v[:, 0].min()), float(all_v[:, 0].max())
    ymin, ymax = float(all_v[:, 1].min()), float(all_v[:, 1].max())
    zmin, zmax = float(all_v[:, 2].min()), float(all_v[:, 2].max())

    dx, dy = xmax - xmin, ymax - ymin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    # Estimate floor levels
    floor_h = 3.0
    n_floors = max(1, round((zmax - zmin) / floor_h))
    floors = [zmin + i * (zmax - zmin) / n_floors for i in range(n_floors + 1)]

    pipes, ducts, electrical = [], [], []

    for fi in range(len(floors) - 1):
        fz = floors[fi]
        pipe_z = fz + 0.35
        duct_z = fz + (floors[fi + 1] - fz) - 0.45
        elec_z = fz + (floors[fi + 1] - fz) - 0.25

        # --- Plumbing ---
        # Main supply (east-west)
        pipes.append(([xmin + dx * 0.06, cx - dx * 0.05, cx - dx * 0.05],
                      [cy - dy * 0.2, cy - dy * 0.2, cy - dy * 0.05],
                      [pipe_z, pipe_z, pipe_z]))
        pipes.append(([cx + dx * 0.05, xmax - dx * 0.06],
                      [cy - dy * 0.2, cy - dy * 0.2],
                      [pipe_z, pipe_z]))
        # Main drain (parallel offset)
        pipes.append(([xmin + dx * 0.06, xmax - dx * 0.06],
                      [cy + dy * 0.18, cy + dy * 0.18],
                      [pipe_z - 0.12, pipe_z - 0.12]))
        # Branch pipes (N-S)
        for frac in [0.22, 0.5, 0.78]:
            bx = xmin + dx * frac
            pipes.append(([bx, bx], [ymin + dy * 0.08, ymax - dy * 0.08],
                          [pipe_z, pipe_z]))
        # Fixture connections
        for frac_x in [0.3, 0.7]:
            for frac_y in [0.25, 0.75]:
                px, py = xmin + dx * frac_x, ymin + dy * frac_y
                pipes.append(([px, px, px + dx * 0.08],
                              [py, py + dy * 0.06, py + dy * 0.06],
                              [pipe_z, pipe_z, pipe_z + 0.8]))

        # --- HVAC Ducts ---
        # Main trunk (E-W center)
        ducts.append(([xmin + dx * 0.05, xmax - dx * 0.05],
                      [cy + dy * 0.02, cy + dy * 0.02],
                      [duct_z, duct_z]))
        # Branch ducts (N-S from trunk)
        for frac in [0.2, 0.4, 0.6, 0.8]:
            bx = xmin + dx * frac
            ducts.append(([bx, bx], [ymin + dy * 0.12, cy], [duct_z, duct_z]))
            ducts.append(([bx, bx], [cy, ymax - dy * 0.12], [duct_z, duct_z]))
        # Return duct (parallel)
        ducts.append(([xmin + dx * 0.08, xmax - dx * 0.08],
                      [cy - dy * 0.25, cy - dy * 0.25],
                      [duct_z - 0.15, duct_z - 0.15]))

        # --- Electrical ---
        # Main conduit run
        electrical.append(([xmin + dx * 0.04, xmax - dx * 0.04],
                           [cy + dy * 0.35, cy + dy * 0.35],
                           [elec_z, elec_z]))
        # Secondary run
        electrical.append(([xmin + dx * 0.04, xmax - dx * 0.04],
                           [cy - dy * 0.35, cy - dy * 0.35],
                           [elec_z, elec_z]))
        # Branch circuits (N-S)
        for frac in [0.12, 0.28, 0.44, 0.6, 0.76, 0.92]:
            bx = xmin + dx * frac
            electrical.append(([bx, bx], [ymin + dy * 0.04, ymax - dy * 0.04],
                               [elec_z, elec_z]))
        # Wall drops (vertical segments to outlets/switches)
        for fx in [0.15, 0.35, 0.55, 0.75, 0.9]:
            for fy in [0.15, 0.5, 0.85]:
                px, py = xmin + dx * fx, ymin + dy * fy
                electrical.append(([px, px], [py, py],
                                   [fz + 0.4, fz + 1.2]))

    # Vertical risers between floors
    if n_floors > 1:
        z_bot, z_top = floors[0], floors[-1]
        # Pipe risers
        for (rx, ry) in [(xmin + dx * 0.08, cy - dy * 0.2),
                         (xmax - dx * 0.08, cy + dy * 0.18)]:
            pipes.append(([rx, rx], [ry, ry], [z_bot + 0.35, z_top + 0.35]))
        # Duct shaft
        ducts.append(([xmax - dx * 0.12, xmax - dx * 0.12],
                      [cy, cy],
                      [z_bot + floor_h - 0.45, z_top + floor_h - 0.45]))
        # Electrical riser
        electrical.append(([xmin + dx * 0.08, xmin + dx * 0.08],
                           [cy + dy * 0.35, cy + dy * 0.35],
                           [z_bot + floor_h - 0.25, z_top + floor_h - 0.25]))

    return {"pipes": pipes, "ducts": ducts, "electrical": electrical}


# ---------------------------------------------------------------------------
# 3D FIGURE BUILDER
# ---------------------------------------------------------------------------
_MEP_STYLES = {
    "pipes":      {"color": "#2196F3", "width": 5, "label": "Plumbing"},
    "ducts":      {"color": "#78909C", "width": 7, "label": "HVAC Ducts"},
    "electrical": {"color": "#FFC107", "width": 3, "label": "Electrical"},
}


def build_3d_figure(
    elements: list,
    mep: dict,
    show_structure: bool = True,
    show_plumbing: bool = True,
    show_hvac: bool = True,
    show_electrical: bool = True,
    height: int = 600,
):
    """Build interactive Plotly 3D figure with structural + MEP layers."""
    fig = go.Figure()
    has_data = False

    # Structural mesh elements
    if show_structure and elements:
        for elem in elements:
            if elem["type"] not in _STRUCTURAL_TYPES:
                continue
            v, f = elem["verts"], elem["faces"]
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=TYPE_COLORS.get(elem["type"], "#D3D3D3"),
                opacity=0.88,
                flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.80, specular=0.14,
                              roughness=0.65, fresnel=0.12),
                lightposition=dict(x=120, y=200, z=300),
                hovertemplate=(
                    f"<b>{elem['name']}</b><br>"
                    f"Type: {elem['type'].replace('Ifc', '')}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))
            has_data = True

    # MEP line traces
    layer_map = {
        "pipes": show_plumbing,
        "ducts": show_hvac,
        "electrical": show_electrical,
    }
    for cat, visible in layer_map.items():
        if not visible or not mep.get(cat):
            continue
        style = _MEP_STYLES[cat]
        all_x, all_y, all_z = [], [], []
        for (xs, ys, zs) in mep[cat]:
            all_x.extend(list(xs) + [None])
            all_y.extend(list(ys) + [None])
            all_z.extend(list(zs) + [None])
        fig.add_trace(go.Scatter3d(
            x=all_x, y=all_y, z=all_z,
            mode="lines",
            line=dict(color=style["color"], width=style["width"]),
            name=style["label"],
            hovertemplate=f"<b>{style['label']}</b><extra></extra>",
            showlegend=True,
        ))
        has_data = True

    if not has_data:
        return None

    # Ground plane
    ref_verts = np.vstack([e["verts"] for e in elements]) if elements else np.array([[0, 0, 0]])
    xmin, xmax = ref_verts[:, 0].min() - 3, ref_verts[:, 0].max() + 3
    ymin, ymax = ref_verts[:, 1].min() - 3, ref_verts[:, 1].max() + 3
    z_ground = ref_verts[:, 2].min() - 0.05
    fig.add_trace(go.Mesh3d(
        x=[xmin, xmax, xmax, xmin], y=[ymin, ymin, ymax, ymax],
        z=[z_ground] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
        color="#f0edf5", opacity=0.5, hoverinfo="skip", showlegend=False,
        flatshading=True, lighting=dict(ambient=0.92, diffuse=0.3),
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.45, y=-1.55, z=1.05),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="perspective"),
            ),
            bgcolor="#f7f5fa",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            yanchor="top", y=0.98, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e0dae8", borderwidth=1,
            font=dict(size=11, color="#1a1a2e"),
        ),
    )
    return fig


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.html(
        f"<div style='text-align:center; padding:1.2rem 0 0.6rem;'>"
        f"<div style='font-size:1.15rem; font-weight:800; color:white;"
        f" letter-spacing:-0.01em;'>Construction Compliance</div>"
        f"<div style='font-size:0.68rem; color:{AMETHYST_TINT};"
        f" margin-top:0.15rem; font-weight:500; letter-spacing:0.04em;'>CHECKER</div>"
        f"</div>"
    )
    st.divider()

    source = st.radio("Model Source", ["Upload IFC", "Sample Model"],
                       key="model_source", label_visibility="collapsed")

    ifc_bytes = None
    ifc_filename = ""

    if source == "Sample Model":
        sample_name = st.selectbox("Select sample", list(SAMPLE_MODELS.keys()),
                                    key="sample_select")
        sample_path = DATA_DIR / SAMPLE_MODELS[sample_name]
        if sample_path.exists():
            ifc_bytes = sample_path.read_bytes()
            ifc_filename = SAMPLE_MODELS[sample_name]
        else:
            st.warning(f"Sample not found: {sample_path.name}")
    else:
        uploaded = st.file_uploader("Upload IFC file", type=["ifc"], key="ifc_upload")
        if uploaded is not None:
            ifc_bytes = uploaded.getvalue()
            ifc_filename = uploaded.name

    # Layer toggles
    st.divider()
    st.html(
        f"<div style='font-size:0.72rem; color:{AMETHYST_TINT};"
        f" text-transform:uppercase; font-weight:600;"
        f" letter-spacing:0.08em; margin-bottom:0.3rem;'>Model Layers</div>"
    )
    show_structure  = st.checkbox("Structure", value=True, key="layer_structure")
    show_plumbing   = st.checkbox("Plumbing", value=True, key="layer_plumbing")
    show_hvac       = st.checkbox("HVAC Ducts", value=True, key="layer_hvac")
    show_electrical = st.checkbox("Electrical", value=True, key="layer_electrical")

    # Re-upload for verification
    if "original_result" in st.session_state and st.session_state.original_result is not None:
        st.divider()
        st.html(
            f"<div style='font-size:0.72rem; color:{AMETHYST_TINT};"
            f" text-transform:uppercase; font-weight:600;"
            f" letter-spacing:0.08em; margin-bottom:0.3rem;'>Verify Corrections</div>"
        )
        corrected = st.file_uploader("Upload corrected IFC", type=["ifc"], key="reupload_ifc")
        if corrected is not None:
            ifc_bytes = corrected.getvalue()
            ifc_filename = corrected.name

    st.divider()
    st.html(
        "<div style='font-size:0.65rem; color:rgba(255,255,255,0.3); text-align:center;"
        " padding:0.3rem 0;'>ifcopenshell + Plotly</div>"
    )


# ===========================================================================
# MAIN CONTENT
# ===========================================================================

# Header
st.html(
    f"<div style='margin-bottom:1.2rem;'>"
    f"<h1 style='font-size:1.6rem; font-weight:800; color:{TEXT_PRIMARY};"
    f" margin:0; letter-spacing:-0.02em;'>Construction Compliance Checker</h1>"
    f"<p style='font-size:0.85rem; color:{TEXT_SECONDARY}; margin:0.2rem 0 0;'>"
    f"Saudi Building Code compliance analysis</p>"
    f"</div>"
)

# -- Empty state --
if ifc_bytes is None:
    st.html(
        f"<div style='text-align:center; padding:4rem 1rem 3rem;'>"
        f"<div style='font-size:1.6rem; font-weight:700; color:{TEXT_PRIMARY};"
        f" margin-bottom:0.5rem;'>Upload an IFC model to begin</div>"
        f"<div style='font-size:0.9rem; color:{TEXT_SECONDARY}; max-width:480px;"
        f" margin:0 auto 2rem;'>Select a sample model from the sidebar or upload "
        f"your own .ifc file to run automated SBC compliance checks.</div>"
        f"</div>"
    )
    codes_data = [
        ("SBC 301", "Structural", "Columns, beams, slabs, walls"),
        ("SBC 801", "Fire / Life Safety", "Egress, stairs, doors"),
        ("SBC 1001", "Accessibility", "Entrances, circulation"),
        ("SBC 501", "MEP Systems", "Plumbing, HVAC, electrical"),
        ("SBC 601", "Energy / Mostadam", "WWR, insulation, envelope"),
    ]
    cards_html = "".join(
        f"<div style='background:{SURFACE}; border:1px solid {BORDER};"
        f" padding:0.85rem 1.1rem; border-radius:10px;'>"
        f"<div style='font-weight:700; color:{TEXT_PRIMARY}; font-size:0.88rem;'>"
        f"{code}</div>"
        f"<div style='font-size:0.78rem; color:{AMETHYST}; font-weight:600;"
        f" margin-top:0.1rem;'>{name}</div>"
        f"<div style='font-size:0.75rem; color:{TEXT_MUTED}; margin-top:0.2rem;'>"
        f"{desc}</div>"
        f"</div>"
        for code, name, desc in codes_data
    )
    st.html(
        f"<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));"
        f" gap:0.6rem; max-width:800px; margin:0 auto 2rem;'>{cards_html}</div>"
    )
    st.stop()

# -- Analyze --
result = analyze_sbc_compliance(ifc_bytes)
mi = result["model_info"]
overall = result["overall_score"]
pass_count = sum(1 for c in result["checks"] if c["passed"])
total_checks = len(result["checks"])
overall_color = "#22c55e" if result["overall_pass"] else ("#f59e0b" if overall >= 60 else "#ef4444")
overall_label = "COMPLIANT" if result["overall_pass"] else "NON-COMPLIANT"

# Store original result for comparison
if "original_result" not in st.session_state:
    st.session_state.original_result = None
if st.session_state.original_result is None:
    st.session_state.original_result = result

# Comparison banner
orig = st.session_state.original_result
if orig is not None and orig is not result and orig["overall_score"] != result["overall_score"]:
    delta = result["overall_score"] - orig["overall_score"]
    delta_sign = "+" if delta > 0 else ""
    delta_color = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#888")
    st.html(
        f"<div style='background:{SURFACE}; border:2px solid {delta_color};"
        f" border-radius:10px; padding:0.9rem 1.4rem; margin-bottom:1rem;'>"
        f"<span style='font-weight:700; color:{TEXT_PRIMARY}; font-size:0.9rem;'>"
        f"Verification</span>"
        f"<span style='margin-left:1.5rem; color:{TEXT_MUTED}; font-size:0.85rem;'>"
        f"Original: {orig['overall_score']:.0f}%</span>"
        f"<span style='margin-left:1rem; color:{TEXT_PRIMARY}; font-weight:600;"
        f" font-size:0.85rem;'>Corrected: {overall:.0f}%</span>"
        f"<span style='margin-left:1rem; color:{delta_color}; font-weight:700;"
        f" font-size:0.85rem;'>{delta_sign}{delta:.0f}%</span>"
        f"</div>"
    )

# -- KPI + Score strip --
st.html(
    f"<div style='display:flex; align-items:center; gap:0; background:{SURFACE};"
    f" border:1px solid {BORDER}; border-radius:12px; overflow:hidden;"
    f" margin-bottom:1rem;'>"
    # Score badge (left)
    f"<div style='background:linear-gradient(135deg, {DEEP_PLUM}, {ROYAL_PURPLE});"
    f" padding:1rem 1.6rem; display:flex; flex-direction:column;"
    f" align-items:center; min-width:140px;'>"
    f"<div style='font-size:2rem; font-weight:800; color:white;'>{overall:.0f}%</div>"
    f"<div style='display:inline-block; background:{"rgba(255,255,255,0.2)" if result["overall_pass"] else "rgba(239,68,68,0.3)"};"
    f" color:white; padding:0.15rem 0.6rem; border-radius:1rem; font-size:0.65rem;"
    f" font-weight:700; letter-spacing:0.06em; margin-top:0.2rem;'>{overall_label}</div>"
    f"</div>"
    # KPIs (right)
    f"<div style='display:flex; flex:1; justify-content:space-around; padding:0.6rem 1rem;'>"
    + kpi(str(mi["storeys"]), "Storeys")
    + kpi(f'{mi["total_products"]:,}', "Products")
    + kpi(str(mi["mep_total"]), "MEP")
    + kpi(f"{pass_count}/{total_checks}", "Passed")
    + f"</div></div>"
)

# -- 3D BIM Viewer --
st.markdown('<div class="sec">3D Model</div>', unsafe_allow_html=True)

geo = extract_geometry(ifc_bytes)
mep_data = _generate_synthetic_mep(geo) if geo else {"pipes": [], "ducts": [], "electrical": []}

fig_3d = build_3d_figure(
    geo, mep_data,
    show_structure=show_structure,
    show_plumbing=show_plumbing,
    show_hvac=show_hvac,
    show_electrical=show_electrical,
    height=600,
)
if fig_3d:
    st.plotly_chart(fig_3d, use_container_width=True, config={
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
        "displaylogo": False,
    })
else:
    st.info("No renderable geometry found in this IFC model.")

# -- Action bar (report + download) --
_act_left, _act_right = st.columns([3, 1])
with _act_left:
    st.download_button(
        "Download Compliance Report (PDF)",
        data=build_compliance_pdf(result, ifc_filename),
        file_name=f"SBC_Compliance_{mi['project_name']}_{date.today().isoformat()}.pdf",
        mime="application/pdf",
        key="export_pdf",
    )
with _act_right:
    st.download_button(
        "Download IFC",
        data=ifc_bytes,
        file_name=ifc_filename,
        mime="application/octet-stream",
        key="download_ifc_main",
    )

# -- Compliance cards --
st.markdown('<div class="sec">Compliance Results</div>', unsafe_allow_html=True)

for check in result["checks"]:
    passed = check["passed"]
    sc = check["score"]
    bar_color = "#22c55e" if sc >= 80 else ("#f59e0b" if sc >= 60 else "#ef4444")
    border_c = "#22c55e" if passed else "#ef4444"
    badge_bg = "#ecfdf5" if passed else "#fef2f2"
    badge_fg = "#065f46" if passed else "#991b1b"
    badge_lbl = "PASS" if passed else "FAIL"

    violations_html = ""
    if check["violations"]:
        items = "".join(
            f"<li style='margin-bottom:0.2rem;'>{v}</li>" for v in check["violations"]
        )
        violations_html = (
            f"<ul style='margin:0.5rem 0 0 1.2rem; padding:0;"
            f" font-size:0.8rem; color:#555;'>{items}</ul>"
        )

    rec_html = ""
    if not passed:
        rec = CODE_RECOMMENDATIONS.get(check["code"], "")
        if rec:
            rec_html = (
                f"<div style='margin-top:0.6rem; padding:0.45rem 0.7rem;"
                f" background:#fffbeb; border-radius:6px; font-size:0.78rem;"
                f" color:#92400e; border-left:3px solid #f59e0b;'>{rec}</div>"
            )

    st.html(
        f"<div style='background:{SURFACE}; border:1px solid {BORDER};"
        f" border-left:4px solid {border_c}; padding:1rem 1.3rem;"
        f" border-radius:10px; margin-bottom:0.6rem;'>"
        # Header row
        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<div>"
        f"<span style='font-weight:700; color:{TEXT_PRIMARY}; font-size:0.95rem;'>"
        f"{check['code']}</span>"
        f"<span style='color:{TEXT_SECONDARY}; margin-left:0.5rem; font-size:0.88rem;'>"
        f"{check['name']}</span>"
        f"</div>"
        f"<div style='display:flex; align-items:center; gap:0.6rem;'>"
        f"<span style='font-weight:700; color:{TEXT_PRIMARY}; font-size:1rem;'>"
        f"{sc:.0f}%</span>"
        f"<span style='background:{badge_bg}; color:{badge_fg}; padding:0.15rem 0.55rem;"
        f" border-radius:6px; font-size:0.68rem; font-weight:700;"
        f" letter-spacing:0.04em;'>{badge_lbl}</span>"
        f"</div></div>"
        # Detail
        f"<div style='font-size:0.78rem; color:{TEXT_MUTED}; margin-top:0.35rem;'>"
        f"{check['detail']}</div>"
        # Progress bar
        f"<div style='height:4px; background:#f0edf5; border-radius:2px; margin-top:0.5rem;'>"
        f"<div style='height:100%; width:{sc:.0f}%; background:{bar_color};"
        f" border-radius:2px;'></div></div>"
        f"{violations_html}"
        f"{rec_html}"
        f"</div>"
    )

# Footer
st.html(
    f"<div style='text-align:center; padding:2rem 0 1rem;"
    f" font-size:0.72rem; color:{TEXT_MUTED};'>"
    f"Construction Compliance Checker  &#183;  {date.today().strftime('%B %Y')}</div>"
)
