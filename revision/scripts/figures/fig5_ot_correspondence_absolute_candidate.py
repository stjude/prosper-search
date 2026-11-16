"""Recolor retained Figure 5 patch costs on one numeric scale; no new OT analysis.

Run from the repository root with PROSPER_OUT="$PWD/prosper_revision_output".
The previous candidate and the artwork of record are read-only inputs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import fig5_ot_correspondence_option2_candidate as layout
from _common import prosper_out, sha256


FIGURE = "fig5_ot_correspondence/candidates_20261114_absolute_cost"
STEM = "Figure_5_candidate_absolute_cost"
OLD_MOUNT = "/PROSPER_WORKSPACE"


def fixed_display_values(cost):
    values = np.asarray(cost, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Retained costs must fit the shared 0-to-1 scale without clipping")
    return 1.0 - values


def resolve_source(recorded):
    path = Path(recorded)
    if not path.exists() and str(path).startswith(OLD_MOUNT + "/"):
        path = Path(__file__).resolve().parents[3] / path.relative_to(OLD_MOUNT)
    return path


def main():
    root = prosper_out()
    source = root / "figures_20261021" / layout.FIGURE
    destination = root / "figures_20261021" / FIGURE
    destination.mkdir(parents=True, exist_ok=False)
    checksums = {
        name: digest
        for digest, name in (
            line.split(maxsplit=1)
            for line in (source / "SHA256SUMS.txt").read_text().splitlines()
        )
    }
    inputs = [source / "SHA256SUMS.txt", source / "PROVENANCE.md"]

    def bound_file(name):
        path = source / name
        assert sha256(path) == checksums[name], name
        inputs.append(path)
        return path

    summary = json.loads(bound_file("matched_cost_summary.json").read_text())
    manifest = pd.read_csv(bound_file("panel_manifest.csv"), dtype={"slide_scan": str})
    recipe_path = root / layout.base.V2_DIR_NAME / "recipe.json"
    assert sha256(recipe_path) == "99610bf53cf0cb6f42bfccbc70eaf487f6999f7be90d2344d1f481527112762e"
    recipe = json.loads(recipe_path.read_text())
    inputs.append(recipe_path)
    rows, readback = [], {"display_scale": [0, 1], "clipped_values": 0, "sides": []}

    for row_index, (panel, index) in enumerate(layout.base.ROWS):
        pair = recipe["panels"][panel]["pairs"][index]
        tag = f"{panel}{index}_{pair['query']}_vs_{pair['neighbour']}"
        saved = summary[tag]
        assert saved["fidelity"]["pass"]
        assert saved["plan_sha256"] == pair["plan_sha256"]
        assert saved["rank"] == pair["rank"]
        assert abs(saved["distance"] - pair["distance"]) < 1e-12
        saved["prior_pair_range"] = [saved.pop("joint_norm_min"), saved.pop("joint_norm_max")]
        saved["display_cost_range"] = [0, 1]
        cells = []
        for role in ("query", "neighbour"):
            side = saved["sides"][role]
            sid = side["slide"]
            group = manifest[(manifest["row"] == row_index) & (manifest["role"] == role)]
            assert len(group) == 3 and set(group["slide_scan"]) == {sid}
            images = {}
            for kind in ("he", "overlay"):
                entry = group[group["kind"] == kind].iloc[0]
                path = resolve_source(entry["source"])
                assert sha256(path) == entry["sha256"], path
                inputs.append(path)
                image = Image.open(path).convert("RGB")
                assert image.size == tuple(pair["slides"][sid]["level_dims"])
                images[kind] = image
                cells.append(dict(im=image, kind=kind, slide=sid, role=role,
                                  src=str(path), sha=entry["sha256"],
                                  um_px=float(entry["um_per_px_level"]), label=entry["label"]))

            csv_path = bound_file(Path(side["patch_values_csv"]).name)
            patches = pd.read_csv(csv_path)
            assert len(patches) == side["n_patches"]
            assert not patches.duplicated(["wsi_loc_x", "wsi_loc_y"]).any()
            grouped = patches.groupby("owner_tile")["matched_cost"]
            assert (grouped.nunique() == 1).all()
            tile_costs = grouped.first()
            assert list(tile_costs.index) == list(range(side["n_tiles"]))
            assert abs(tile_costs.mean() - saved["distance"]) < 1e-7
            assert abs(tile_costs.min() - side["tile_cost_min"]) < 1e-12
            assert abs(tile_costs.max() - side["tile_cost_max"]) < 1e-12
            patches["display_value"] = fixed_display_values(patches["matched_cost"])
            image = layout.paint(
                images["he"], patches[["wsi_loc_x", "wsi_loc_y"]].to_numpy(),
                patches["matched_cost"].to_numpy(), float(pair["slides"][sid]["downsample"]),
                cmap=f"{layout.HEAT_CMAP}_r",
            )
            overlay_path = destination / Path(side["overlay_png"]).name
            image.save(overlay_path)
            values_path = destination / csv_path.name
            patches.to_csv(values_path, index=False)
            before = group[group["kind"] == "matched_cost"].iloc[0]
            cells.append(dict(im=image, kind="matched_cost", slide=sid, role=role,
                              src=str(overlay_path), sha=sha256(overlay_path),
                              um_px=float(before["um_per_px_level"]), label=before["label"]))
            side["retained_patch_values_csv"] = str(csv_path)
            side["patch_values_csv"] = str(values_path)
            side["overlay_png"] = str(overlay_path)
            side["overlay_sha256"] = sha256(overlay_path)
            readback["sides"].append(dict(
                pair=tag, role=role, source_patch_sha256=sha256(csv_path),
                n_tiles=len(tile_costs), n_patches=len(patches),
                cost_min=float(tile_costs.min()), cost_max=float(tile_costs.max()),
                mean_tile_cost=float(tile_costs.mean()), ot_distance=saved["distance"],
            ))
        rows.append(dict(panel=panel, idx=index, pair=pair, cells=cells))

    retained = destination / "render_sources"
    retained.mkdir()
    for path in (Path(__file__), Path(layout.__file__), Path(layout.base.__file__),
                 Path(__file__).with_name("_common.py")):
        shutil.copy2(path, retained / path.name)
        inputs.append(path)
    readback["inputs"] = {str(path): sha256(path) for path in inputs}
    (destination / "display_readback.json").write_text(json.dumps(readback, indent=2) + "\n")
    inputs.append(destination / "display_readback.json")
    layout.render(rows, summary, inputs, destination, figure=FIGURE, stem=STEM, absolute_cost=True)


if __name__ == "__main__":
    main()
