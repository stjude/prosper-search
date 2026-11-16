"""Per-group hit@5 by recorded metastasis status (Metastatic vs Primary), main configuration.

Descriptive point counts only. Reproduces the S1b stratum membership rule (first metadata row per
slide_id) and checks the S1b macro/micro hit@5 points before writing. No bootstrap.
"""
import hashlib, json, os, sys
import pandas as pd

OUT = os.path.join(os.environ["PROSPER_OUT"], "r2_3_s1b_pergroup_metastasis_20261005")
FRAME = os.path.join(os.environ["PROSPER_OUT"], "r2_3_internal_failure_20261003", "all_query_frame.csv")
S1B = os.path.join(os.environ["PROSPER_OUT"], "s1b_stratified", "stratified_uni2_ot_cosine.json")

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

s1b = json.load(open(S1B))
meta_path = s1b["config"]["rich_metadata"]
meta = pd.read_csv(meta_path, low_memory=False)
meta = meta.drop_duplicates("slide_id", keep="first")[["slide_id", "Met"]]
fr = pd.read_csv(FRAME)
fr["slide_id"] = fr["query"].astype(str)
meta["slide_id"] = meta["slide_id"].astype(str)
df = fr.merge(meta, on="slide_id", how="left")
df = df[df["Met"].isin(["Metastatic", "Primary"])]

checks = {}
for s in ["Metastatic", "Primary"]:
    d = df[df["Met"] == s]
    ref = s1b["strata"]["metastasis"][s]
    micro = d["hit5"].mean()
    macro = d.groupby("true_label")["hit5"].mean().mean()
    checks[s] = {"n_queries": int(len(d)), "n_queries_s1b": ref["n_queries"],
                 "micro_hit5": micro, "micro_hit5_s1b": ref["point"]["micro_hit_at_5"],
                 "macro_hit5": macro, "macro_hit5_s1b": ref["point"]["macro_hit_at_5"]}
    assert len(d) == ref["n_queries"], checks
    assert abs(micro - ref["point"]["micro_hit_at_5"]) < 1e-9 and abs(macro - ref["point"]["macro_hit_at_5"]) < 1e-9, checks

g = df.groupby(["true_label", "Met"])["hit5"].agg(correct="sum", n="count").reset_index()
g["correct"] = g["correct"].astype(int)
wide = g.pivot(index="true_label", columns="Met", values=["correct", "n"])
wide.columns = [f"{a}_{b.lower()}" for a, b in wide.columns]
wide = wide.reset_index().rename(columns={"true_label": "group"})
wide["hit5_metastatic"] = wide["correct_metastatic"] / wide["n_metastatic"]
wide["hit5_primary"] = wide["correct_primary"] / wide["n_primary"]
wide["metastatic_minus_primary"] = wide["hit5_metastatic"] - wide["hit5_primary"]
wide = wide.sort_values("group")
wide.to_csv(os.path.join(OUT, "per_group_hit5_by_metastasis.csv"), index=False)
summary = {
    "n_groups": int(len(wide)),
    "groups_lower_on_metastatic": int((wide["metastatic_minus_primary"] < 0).sum()),
    "groups_higher_on_metastatic": sorted(wide.loc[wide["metastatic_minus_primary"] > 0, "group"].tolist()),
    "macro_contrast": float(wide["metastatic_minus_primary"].mean()),
    "s1b_point_checks": checks,
    "inputs": {"frame": FRAME, "frame_sha256": sha(FRAME), "s1b_json": S1B, "s1b_sha256": sha(S1B),
               "rich_metadata": meta_path, "rich_metadata_sha256": sha(meta_path)},
    "membership_rule": "first metadata row per slide_id (matches S1b stratify_cli); Met in {Metastatic, Primary}",
    "note": "descriptive point counts; no intervals; not an effect of metastasis",
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
open(os.path.join(OUT, "COMPLETE"), "w").write(sha(os.path.join(OUT, "per_group_hit5_by_metastasis.csv")) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "inputs"}, indent=1))
