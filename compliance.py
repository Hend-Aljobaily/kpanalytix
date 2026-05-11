"""
Construction Compliance -- Saudi Building Code (SBC) Compliance Checker

Standalone Streamlit application for analyzing IFC building models
against Saudi Building Code requirements. Features an embedded 3D BIM
viewer, automated compliance analysis, PDF reporting, and a
download-edit-reverify workflow for CAD integration.
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
    page_title="Construction Compliance",
    page_icon=":building_construction:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
DEEP_PLUM      = "#240F3E"
ROYAL_PURPLE   = "#442270"
AMETHYST       = "#69479E"
LILAC_HAZE     = "#F9F4FF"
AMETHYST_TINT  = "#b5a4d6"
LAVENDER       = "#c2b0e2"
DEEP_INDIGO    = "#2d1754"

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

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.html(f"""
<style>
  .stApp {{
      background:
        radial-gradient(circle at 12% 8%, rgba(105,71,158,0.13) 0%, transparent 45%),
        radial-gradient(circle at 92% 0%, rgba(36,15,62,0.08) 0%, transparent 55%),
        {LILAC_HAZE};
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
      color: {DEEP_PLUM} !important;
  }}
  .block-container {{ padding-top: 1.6rem; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background: linear-gradient(175deg, {DEEP_PLUM} 0%, #2e1648 55%, {ROYAL_PURPLE} 100%);
      box-shadow: 4px 0 18px rgba(36,15,62,0.18);
  }}
  section[data-testid="stSidebar"] * {{ color: #ece4f6 !important; }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stRadio label,
  section[data-testid="stSidebar"] .stFileUploader label {{
      color: {AMETHYST_TINT} !important;
      font-weight: 600;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
  }}
  section[data-testid="stSidebar"] hr {{
      border-color: rgba(255,255,255,0.12) !important;
  }}

  /* KPI cards */
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

  /* Section dividers */
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

  /* Download buttons */
  .stDownloadButton > button {{
      background: {DEEP_PLUM} !important;
      color: white !important;
      border: none !important;
      font-weight: 600 !important;
  }}
  .stDownloadButton > button:hover {{
      background: {ROYAL_PURPLE} !important;
  }}
</style>
""")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def kpi(val, lbl, color=None):
    style = f' style="color:{color}"' if color else ""
    return f'<div class="kpi"><div class="val"{style}>{val}</div><div class="lbl">{lbl}</div></div>'


# ═══════════════════════════════════════════════════════════════════════════
# COMPLIANCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Analyzing IFC compliance...")
def analyze_sbc_compliance(ifc_bytes: bytes) -> dict:
    """Parse an IFC file and check against Saudi Building Code (SBC).

    Checks 5 codes: Structural (301), Fire/Life Safety (801),
    Accessibility (1001), MEP (501), Energy/Mostadam (601).
    """
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
        v.append("No columns (IfcColumn) found -- structural frame undefined")
        score -= 30
    elif storeys > 0 and columns / max(1, storeys) < 2:
        v.append(f"Column-to-storey ratio {columns / max(1, storeys):.1f} is below the minimum 2.0")
        score -= 15
    if slabs == 0:
        v.append("No slabs (IfcSlab) found -- floor/roof diaphragms missing")
        score -= 25
    if beams == 0 and storeys > 1:
        v.append("No beams (IfcBeam) in a multi-storey building")
        score -= 20
    if walls == 0:
        v.append("No walls defined -- lateral load path incomplete")
        score -= 25
    checks.append({
        "code": "SBC 301", "name": "Structural",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{columns} columns, {beams} beams, {slabs} slabs, {walls} walls, {storeys} storey(s)",
    })

    # SBC 801: Fire Protection / Life Safety
    v, score = [], 100.0
    if storeys >= 2 and stairs == 0:
        v.append(f"No stairs (IfcStair) in a {storeys}-storey building -- no vertical egress")
        score -= 40
    if doors == 0:
        v.append("No doors found -- egress routes undefined")
        score -= 30
    elif storeys > 0 and doors / max(1, storeys) < 1:
        v.append(f"Only {doors} door(s) for {storeys} storey(s) -- insufficient egress capacity")
        score -= 20
    if storeys >= 3 and stairs < 2:
        v.append(f"Only {stairs} stairway(s) for a {storeys}-storey building -- SBC requires 2+ above 2 storeys")
        score -= 20
    checks.append({
        "code": "SBC 801", "name": "Fire Protection / Life Safety",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{doors} doors, {stairs} stairs, {storeys} storey(s)",
    })

    # SBC 1001: Accessibility
    v, score = [], 100.0
    if doors == 0:
        v.append("No doors modelled -- cannot verify accessible entrances")
        score -= 40
    if storeys >= 2 and stairs == 0:
        v.append("No vertical circulation elements -- wheelchair access unverifiable")
        score -= 30
    if spaces == 0:
        v.append("No IfcSpace definitions -- cannot verify accessible routes/clearances")
        score -= 20
    if railings == 0 and storeys >= 2:
        v.append("No railings defined for a multi-storey building")
        score -= 10
    checks.append({
        "code": "SBC 1001", "name": "Accessibility",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{doors} doors, {spaces} spaces, {railings} railings",
    })

    # SBC 501: MEP Systems
    v, score = [], 100.0
    if mep_total == 0:
        v.append("No MEP elements found (distribution, terminals, pipes, ducts)")
        score = 20
    elif mep_total < max(1, storeys) * 3:
        v.append(f"Only {mep_total} MEP element(s) for {storeys} storey(s) -- likely incomplete")
        score -= 25
    if counts["IfcDuctSegment"] == 0 and counts["IfcFlowSegment"] == 0:
        v.append("No HVAC ductwork modelled")
        score -= 15
    if counts["IfcPipeSegment"] == 0:
        v.append("No plumbing piping modelled")
        score -= 15
    if counts["IfcFlowTerminal"] == 0:
        v.append("No flow terminals (fixtures, outlets) modelled")
        score -= 10
    if electrical == 0:
        v.append("No electrical elements (lights, outlets, switches) detected")
        score -= 10
    checks.append({
        "code": "SBC 501", "name": "MEP Systems",
        "passed": len(v) == 0, "score": max(0, score), "violations": v,
        "detail": f"{mep_total} MEP elements, {electrical} electrical, "
                  f"{counts['IfcPipeSegment']} pipes, {counts['IfcDuctSegment']} ducts",
    })

    # SBC 601: Energy Efficiency / Mostadam
    v, score = [], 100.0
    if windows > 0 and walls > 0:
        wwr = windows / (windows + walls)
        if wwr > 0.40:
            v.append(f"Window-to-wall ratio {wwr:.0%} exceeds the 40% maximum")
            score -= 25
        elif wwr < 0.10:
            v.append(f"Window-to-wall ratio {wwr:.0%} is below 10% -- natural daylighting insufficient")
            score -= 15
    elif windows == 0:
        v.append("No windows modelled -- daylighting and ventilation unverifiable")
        score -= 20
    if roofs > 0 and coverings == 0:
        v.append("Roof present but no IfcCovering -- insulation layer not modelled")
        score -= 20
    if roofs == 0:
        v.append("No roof elements -- building envelope incomplete for energy analysis")
        score -= 15
    checks.append({
        "code": "SBC 601", "name": "Energy Efficiency / Mostadam",
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


# ═══════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ═══════════════════════════════════════════════════════════════════════════
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
            self.cell(0, 5, f"Construction Compliance Report  |  Page {self.page_no()}", align="C")

    pdf = CompliancePDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover page
    pdf.add_page()
    pdf.set_fill_color(36, 15, 62)
    pdf.rect(0, 0, 210, 55, style="F")
    pdf.set_fill_color(68, 34, 112)
    pdf.rect(0, 50, 210, 5, style="F")

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(16)
    pdf.cell(0, 10, "SBC Compliance Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Saudi Building Code Compliance Analysis", align="C", new_x="LMARGIN", new_y="NEXT")

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
    pdf.set_text_color(36, 15, 62)
    pdf.cell(0, 10, "Detailed Analysis by SBC Code", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for chk in result["checks"]:
        pdf.set_draw_color(*(34, 197, 94) if chk["passed"] else (239, 68, 68))
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 10, pdf.get_y() + 6)
        pdf.set_x(14)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(36, 15, 62)
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


# ═══════════════════════════════════════════════════════════════════════════
# 3D VIEWER (Plotly Mesh3d)
# ═══════════════════════════════════════════════════════════════════════════
TYPE_COLORS = {
    # Structural
    "IfcWall": "#f1ece2", "IfcWallStandardCase": "#f1ece2",
    "IfcSlab": "#b8b3c8", "IfcColumn": "#7a7397", "IfcBeam": "#9a93b5",
    "IfcRoof": "#9c4a2a", "IfcDoor": "#6b4a2b", "IfcWindow": "#4ea3d6",
    "IfcStair": "#caa97a", "IfcRailing": "#4a4a55",
    "IfcCovering": "#d8c39a", "IfcPlate": "#a39db0",
    "IfcFurnishingElement": "#b8a78c",
    # MEP — Pipes (blue tones)
    "IfcPipeSegment": "#2196F3", "IfcPipeFitting": "#1976D2",
    # MEP — Ducts (gray tones)
    "IfcDuctSegment": "#78909C", "IfcDuctFitting": "#607D8B",
    # MEP — Flow / Distribution
    "IfcFlowSegment": "#26A69A", "IfcFlowTerminal": "#00897B",
    "IfcFlowFitting": "#00796B",
    "IfcDistributionElement": "#546E7A",
    "IfcEnergyConversionDevice": "#FF7043",
    # MEP — Electrical (yellow/amber tones)
    "IfcLightFixture": "#FFD54F", "IfcOutlet": "#FFCA28",
    "IfcSwitchingDevice": "#FFC107",
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


def build_3d_figure(elements: list, height: int = 580):
    """Build a Plotly Mesh3d figure from extracted geometry elements."""
    if not elements:
        return None

    fig = go.Figure()

    for elem in elements:
        verts = elem["verts"]
        faces = elem["faces"]
        color = TYPE_COLORS.get(elem["type"], "#D3D3D3")
        hover = (
            f"<b>{elem['name']}</b><br>"
            f"Type: {elem['type'].replace('Ifc', '')}"
            "<extra></extra>"
        )
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=0.92, flatshading=True,
            lighting=dict(ambient=0.58, diffuse=0.82, specular=0.16,
                          roughness=0.7, fresnel=0.15),
            lightposition=dict(x=120, y=200, z=300),
            hovertemplate=hover, showlegend=False,
        ))

    # Ground plane
    all_v = np.vstack([e["verts"] for e in elements])
    xmin, xmax = all_v[:, 0].min() - 4, all_v[:, 0].max() + 4
    ymin, ymax = all_v[:, 1].min() - 4, all_v[:, 1].max() + 4
    z_ground = all_v[:, 2].min() - 0.05
    fig.add_trace(go.Mesh3d(
        x=[xmin, xmax, xmax, xmin], y=[ymin, ymin, ymax, ymax],
        z=[z_ground] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
        color=LILAC_HAZE, opacity=0.6, hoverinfo="skip", showlegend=False,
        flatshading=True, lighting=dict(ambient=0.92, diffuse=0.3),
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
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
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.html(
        f"<div style='text-align:center; padding:1rem 0 0.5rem;'>"
        f"<div style='font-size:1.3rem; font-weight:800; color:white;"
        f" letter-spacing:-0.01em;'>Construction Compliance</div>"
        f"<div style='font-size:0.75rem; color:{AMETHYST_TINT};"
        f" margin-top:0.2rem;'>Saudi Building Code Checker</div>"
        f"</div>"
    )
    st.divider()

    source = st.radio("Model Source", ["Upload IFC", "Sample Model"], key="model_source",
                       label_visibility="collapsed")

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
            st.warning(f"Sample file not found: {sample_path.name}")
    else:
        uploaded = st.file_uploader("Upload IFC file", type=["ifc"], key="ifc_upload")
        if uploaded is not None:
            ifc_bytes = uploaded.getvalue()
            ifc_filename = uploaded.name

    # Re-upload for verification
    if "original_result" in st.session_state and st.session_state.original_result is not None:
        st.divider()
        st.html(
            f"<div style='font-size:0.78rem; color:{AMETHYST_TINT};"
            f" text-transform:uppercase; font-weight:600;"
            f" letter-spacing:0.07em; margin-bottom:0.4rem;'>Verify Corrections</div>"
        )
        corrected = st.file_uploader("Upload corrected IFC", type=["ifc"], key="reupload_ifc")
        if corrected is not None:
            ifc_bytes = corrected.getvalue()
            ifc_filename = corrected.name

    st.divider()
    st.html(
        f"<div style='font-size:0.7rem; color:rgba(255,255,255,0.4); text-align:center;"
        f" padding:0.5rem 0;'>Powered by ifcopenshell + Plotly</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

# Page header
st.html(
    f"<div style='background:linear-gradient(135deg, {DEEP_PLUM} 0%, {ROYAL_PURPLE} 100%);"
    f" border-radius:0.7rem; padding:1.2rem 1.8rem; margin-bottom:1.2rem;"
    f" box-shadow:0 4px 18px rgba(36,15,62,0.18);'>"
    f"<div style='font-size:1.4rem; font-weight:800; color:white;"
    f" letter-spacing:-0.01em;'>Saudi Building Code (SBC) Compliance Checker</div>"
    f"<div style='font-size:0.85rem; color:{AMETHYST_TINT}; margin-top:0.2rem;'>"
    f"Automated compliance analysis across structural, fire safety, accessibility, MEP, and energy codes</div>"
    f"</div>"
)

# ── Empty state ────────────────────────────────────────────────────────
if ifc_bytes is None:
    st.html(
        f"<div style='text-align:center; padding:3rem 1rem 2rem;'>"
        f"<h2 style='color:{DEEP_PLUM}; margin:0 0 0.5rem;'>Upload an IFC Model to Begin</h2>"
        f"<p style='color:#666; font-size:0.95rem; max-width:520px; margin:0 auto 1.5rem;'>"
        f"Upload a Building Information Model (.ifc) or select a sample model "
        f"from the sidebar to run automated Saudi Building Code compliance checks.</p>"
        f"</div>"
    )
    codes_data = [
        ("SBC 301", "Structural", "Columns, beams, slabs, walls"),
        ("SBC 801", "Fire / Life Safety", "Egress stairs, doors, fire elements"),
        ("SBC 1001", "Accessibility", "Entrances, circulation, clearances"),
        ("SBC 501", "MEP Systems", "Mechanical, electrical, plumbing"),
        ("SBC 601", "Energy / Mostadam", "WWR, insulation, envelope"),
    ]
    cards_html = "".join(
        f"<div style='background:white; border-left:4px solid {AMETHYST};"
        f" padding:0.75rem 1rem; border-radius:0.5rem;"
        f" box-shadow:0 2px 8px rgba(36,15,62,0.07);'>"
        f"<b style='color:{DEEP_PLUM}; font-size:0.92rem;'>{code} -- {name}</b><br>"
        f"<span style='font-size:0.82rem; color:#555;'>{desc}</span>"
        f"</div>"
        for code, name, desc in codes_data
    )
    st.html(
        f"<div style='display:grid; grid-template-columns:1fr 1fr; gap:0.6rem;"
        f" max-width:720px; margin:0 auto 2rem;'>{cards_html}</div>"
    )
    st.stop()

# ── Analyze ────────────────────────────────────────────────────────────
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

# ── Comparison banner (re-upload) ──────────────────────────────────────
orig = st.session_state.original_result
if orig is not None and orig is not result and orig["overall_score"] != result["overall_score"]:
    delta = result["overall_score"] - orig["overall_score"]
    delta_sign = "+" if delta > 0 else ""
    delta_color = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#888")
    st.html(
        f"<div style='background:white; border:2px solid {delta_color};"
        f" border-radius:0.7rem; padding:1rem 1.5rem; margin-bottom:1rem;'>"
        f"<div style='font-weight:700; color:{DEEP_PLUM}; font-size:1.05rem;'>"
        f"Verification Result</div>"
        f"<div style='display:flex; gap:2rem; margin-top:0.5rem; font-size:0.9rem;'>"
        f"<div style='color:#888;'>Original: {orig['overall_score']:.0f}%</div>"
        f"<div style='color:{DEEP_PLUM}; font-weight:600;'>Corrected: {overall:.0f}%</div>"
        f"<div style='color:{delta_color}; font-weight:700;'>{delta_sign}{delta:.0f}%</div>"
        f"</div></div>"
    )

# ── KPI Banner ─────────────────────────────────────────────────────────
_kpi_items = [
    (str(mi["storeys"]), "Storeys"),
    (f"{mi['total_products']:,}", "Products"),
    (str(mi["mep_total"]), "MEP Elements"),
    (f"{pass_count}/{total_checks}", "Checks Passed"),
]
_kpi_cells = "".join(
    f"<div style='text-align:center; min-width:100px;'>"
    f"<div style='font-size:1.4rem; font-weight:700; color:{DEEP_PLUM};'>{val}</div>"
    f"<div style='font-size:0.78rem; color:#888; text-transform:uppercase;"
    f" letter-spacing:0.04em;'>{lbl}</div>"
    f"</div>"
    for val, lbl in _kpi_items
)
_score_badge = (
    f"<div style='text-align:center; min-width:120px;'>"
    f"<div style='font-size:1.8rem; font-weight:800; color:{overall_color};'>{overall:.0f}%</div>"
    f"<div style='display:inline-block; background:{overall_color}; color:white;"
    f" padding:0.2rem 0.7rem; border-radius:1rem; font-size:0.75rem;"
    f" font-weight:700; letter-spacing:0.04em;'>{overall_label}</div>"
    f"</div>"
)
st.html(
    f"<div style='display:flex; justify-content:space-between; align-items:center;"
    f" flex-wrap:wrap; gap:1rem; background:white; border-radius:0.7rem;"
    f" padding:1.1rem 1.8rem; margin-bottom:1rem;"
    f" box-shadow:0 2px 10px rgba(36,15,62,0.08); border:1px solid #eee;'>"
    f"{_kpi_cells}{_score_badge}"
    f"</div>"
)

# ── 3D BIM Viewer + Edit Panel ─────────────────────────────────────────
st.markdown('<div class="sec">3D BIM Model</div>', unsafe_allow_html=True)

col_viewer, col_edit = st.columns([3, 1])

with col_viewer:
    geo = extract_geometry(ifc_bytes)
    if geo:
        fig_3d = build_3d_figure(geo, height=580)
        if fig_3d:
            st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No renderable geometry found in this IFC model.")

with col_edit:
    st.markdown('<div class="sec">Edit Model</div>', unsafe_allow_html=True)
    st.download_button(
        "Download IFC",
        data=ifc_bytes,
        file_name=ifc_filename,
        mime="application/octet-stream",
        key="download_ifc_main",
        use_container_width=True,
    )
    _apps = ["Revit", "AutoCAD", "Navisworks", "Solibri"]
    _pills = " ".join(
        f"<span style='display:inline-block; background:{DEEP_PLUM}; color:white;"
        f" padding:0.3rem 0.7rem; border-radius:1rem; font-size:0.75rem;"
        f" font-weight:600; margin:0.15rem 0.1rem;'>{n}</span>"
        for n in _apps
    )
    st.html(
        f"<div style='margin-top:0.6rem;'>"
        f"<div style='font-size:0.78rem; color:#888; margin-bottom:0.35rem;'>Compatible with</div>"
        f"<div>{_pills}</div>"
        f"</div>"
    )
    st.html(
        f"<div style='margin-top:1.2rem; padding:0.8rem; background:white;"
        f" border-radius:0.5rem; box-shadow:0 2px 8px rgba(36,15,62,0.06);'>"
        f"<div style='font-size:0.82rem; font-weight:600; color:{DEEP_PLUM};"
        f" margin-bottom:0.3rem;'>Workflow</div>"
        f"<div style='font-size:0.78rem; color:#666; line-height:1.6;'>"
        f"1. Review compliance issues below<br>"
        f"2. Download the IFC file<br>"
        f"3. Edit in your CAD application<br>"
        f"4. Re-upload via sidebar to verify</div>"
        f"</div>"
    )
    # Model info
    st.html(
        f"<div style='margin-top:1rem; font-size:0.78rem; color:#888;'>"
        f"<div><b style='color:{DEEP_PLUM};'>Project:</b> {mi['project_name']}</div>"
        f"<div><b style='color:{DEEP_PLUM};'>Schema:</b> {mi['schema']}</div>"
        f"<div><b style='color:{DEEP_PLUM};'>File:</b> {ifc_filename}</div>"
        f"</div>"
    )

# ── Compliance Results ─────────────────────────────────────────────────
st.markdown('<div class="sec">Compliance Analysis Results</div>', unsafe_allow_html=True)

# Overall score card + PDF
_score_col, _pdf_col = st.columns([3, 1])
with _score_col:
    st.html(
        f"<div style='background:linear-gradient(135deg, #fafafa 0%, #f3f0f7 100%);"
        f" border-radius:0.7rem; padding:1.4rem 1.8rem;"
        f" text-align:center; border:2px solid {overall_color};"
        f" box-shadow:0 4px 14px rgba(36,15,62,0.10);'>"
        f"<div style='font-size:2.2rem; font-weight:800; color:{overall_color};'>"
        f"{overall:.0f}%</div>"
        f"<div style='font-size:1.1rem; font-weight:700; color:{overall_color};"
        f" letter-spacing:0.05em; margin-top:0.3rem;'>{overall_label}</div>"
        f"<div style='font-size:0.82rem; color:#888; margin-top:0.3rem;'>"
        f"{pass_count}/{total_checks} checks passed</div>"
        f"<div style='font-size:0.72rem; color:#aaa; margin-top:0.5rem;'>"
        f"Generated on {date.today().strftime('%B %d, %Y')}</div>"
        f"</div>"
    )
with _pdf_col:
    st.download_button(
        "Export PDF Report",
        data=build_compliance_pdf(result, ifc_filename),
        file_name=f"SBC_Compliance_{mi['project_name']}_{date.today().isoformat()}.pdf",
        mime="application/pdf",
        key="export_pdf",
        use_container_width=True,
    )

# Per-code compliance cards
st.markdown('<div class="sec">Compliance Report by SBC Code</div>', unsafe_allow_html=True)

for check in result["checks"]:
    border = "#22c55e" if check["passed"] else "#ef4444"
    bar_color = "#22c55e" if check["score"] >= 80 else ("#f59e0b" if check["score"] >= 60 else "#ef4444")
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

    rec_html = ""
    if not check["passed"]:
        rec = CODE_RECOMMENDATIONS.get(check["code"], "")
        if rec:
            rec_html = (
                f"<div style='margin-top:0.6rem; padding:0.5rem 0.7rem; background:#fffbeb;"
                f" border-radius:0.35rem; font-size:0.8rem; color:#92400e;"
                f" border-left:3px solid #f59e0b;'>"
                f"<b>Recommended action:</b> {rec}</div>"
            )

    progress_bar = (
        f"<div style='height:6px; background:#eee; border-radius:3px; margin-top:0.6rem;'>"
        f"<div style='height:100%; width:{check['score']:.0f}%; background:{bar_color};"
        f" border-radius:3px; transition:width 0.4s ease;'></div></div>"
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
        f"<div style='font-size:0.82rem; color:#888; margin-top:0.4rem;'>{check['detail']}</div>"
        f"{progress_bar}"
        f"{violations_html}"
        f"{rec_html}"
        f"</div>"
    )
