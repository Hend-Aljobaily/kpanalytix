"""
KPAnalytix — IFC Geometry Preprocessor
=======================================
Extracts 3D meshes from IFC files and caches them as pickle files
for fast loading in the Streamlit dashboard.

Run once:  python preprocess_ifc.py
Output:    kpanalytix_data/bim/{name}_geometry.pkl
"""

import os
import pickle
import time
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import numpy as np

IFC_DIR = Path("C:/Users/henda/kpanalytix/kpanalytix_data/bim")
BUILDINGS = ["BasicHouse", "TallBuilding", "SampleHouse", "Duplex"]

# IFC types we care about for the 3D viewer (skip space/site/etc.)
STRUCTURAL_TYPES = {
    "IfcWall", "IfcWallStandardCase",
    "IfcSlab", "IfcColumn", "IfcBeam",
    "IfcRoof", "IfcStair", "IfcDoor", "IfcWindow",
    "IfcRailing", "IfcCovering", "IfcPlate",
    "IfcFurnishingElement",
}

# Maximum vertex count per element (decimate if larger)
MAX_VERTS_PER_ELEMENT = 4000
# Maximum total verts per building
MAX_TOTAL_VERTS = 80000


def _decimate_mesh(verts, faces, target_verts):
    """Simple mesh decimation by uniform vertex subsampling."""
    n = len(verts)
    if n <= target_verts:
        return verts, faces

    # Keep every Nth vertex
    step = max(1, n // target_verts)
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[::step] = True
    kept_indices = np.where(keep_mask)[0]

    # Build old→new index map
    index_map = np.full(n, -1, dtype=np.int64)
    index_map[kept_indices] = np.arange(len(kept_indices))

    new_verts = verts[kept_indices]

    # Keep only faces where ALL 3 vertices survived
    valid = np.all(index_map[faces] >= 0, axis=1)
    new_faces = index_map[faces[valid]]

    return new_verts, new_faces


def extract_geometry(name):
    """Extract meshes from one IFC file, return list of element dicts."""
    ifc_path = IFC_DIR / f"{name}.ifc"
    print(f"\n  [{name}] Loading {ifc_path.name} ...")

    model = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    products = [p for p in model.by_type("IfcProduct") if p.is_a() in STRUCTURAL_TYPES]
    print(f"    {len(products)} structural products to process")

    elements = []
    total_verts = 0
    t0 = time.time()

    for i, product in enumerate(products):
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts_flat = np.array(shape.geometry.verts, dtype=np.float32)
            faces_flat = np.array(shape.geometry.faces, dtype=np.int32)

            if len(verts_flat) == 0:
                continue

            verts = verts_flat.reshape(-1, 3)
            faces = faces_flat.reshape(-1, 3)

            # Decimate if too large
            if len(verts) > MAX_VERTS_PER_ELEMENT:
                verts, faces = _decimate_mesh(verts, faces, MAX_VERTS_PER_ELEMENT)

            if len(faces) == 0:
                continue

            z_min = float(verts[:, 2].min())
            z_max = float(verts[:, 2].max())

            elements.append({
                "type": product.is_a(),
                "name": product.Name or product.is_a(),
                "verts": verts,
                "faces": faces,
                "z_min": z_min,
                "z_max": z_max,
                "z_mid": (z_min + z_max) / 2,
            })
            total_verts += len(verts)

        except Exception:
            continue

        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(products)} processed")

    elapsed = time.time() - t0
    print(f"    {len(elements)} elements extracted, {total_verts:,} total verts, {elapsed:.1f}s")

    # If total verts exceed budget, decimate proportionally
    if total_verts > MAX_TOTAL_VERTS:
        ratio = MAX_TOTAL_VERTS / total_verts
        print(f"    Decimating all meshes to {ratio:.0%} ...")
        for elem in elements:
            target = max(12, int(len(elem["verts"]) * ratio))
            elem["verts"], elem["faces"] = _decimate_mesh(
                elem["verts"], elem["faces"], target
            )
        new_total = sum(len(e["verts"]) for e in elements)
        print(f"    After decimation: {new_total:,} verts")

    return elements


def main():
    print("=" * 55)
    print("  KPAnalytix — IFC Geometry Preprocessor")
    print("=" * 55)

    for name in BUILDINGS:
        elements = extract_geometry(name)

        out_path = IFC_DIR / f"{name}_geometry.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(elements, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"    Saved: {out_path.name} ({size_mb:.1f} MB)")

    print("\n" + "=" * 55)
    print("  Done — geometry cached for dashboard")
    print("=" * 55)


if __name__ == "__main__":
    main()
