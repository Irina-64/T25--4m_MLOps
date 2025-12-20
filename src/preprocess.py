import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

ELO_INIT = 1500
ELO_K = 20
LAST_N = 3


def norm_name(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def safe_div(a, b):
    return a / (b + 1e-9)


fighters_base_path = os.path.join(RAW, "Fighters.csv")
fstats_path = os.path.join(RAW, "Fstats.csv")
fights_path = os.path.join(RAW, "Fights.csv")

fighters_base = (
    pd.read_csv(fighters_base_path)
    if os.path.exists(fighters_base_path)
    else pd.DataFrame()
)
fstats = pd.read_csv(fstats_path) if os.path.exists(fstats_path) else pd.DataFrame()
fights = pd.read_csv(fights_path).reset_index(drop=True)
fights["fight_index"] = fights.index


fights["_R_key"] = fights["Fighter_1"].astype(str).apply(norm_name)
fights["_B_key"] = fights["Fighter_2"].astype(str).apply(norm_name)


def normalize_result(s):
    s = s.astype(str).str.lower().str.strip()
    return s.replace(
        {
            "w": "win",
            "l": "loss",
            "won": "win",
            "lost": "loss",
            "no contest": "nc",
            "no-contest": "nc",
            "draw": "draw",
        }
    )

def pick_cols(prefix: str, df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c != key_col]
    out = df[cols].copy()
    out.columns = [f"{prefix}_{c}" for c in out.columns]
    return out
r1 = normalize_result(fights["Result_1"])
r2 = normalize_result(fights["Result_2"])
mask_bad = r1.isin(["draw", "nc"]) | r2.isin(["draw", "nc"])

target = np.where(r1.eq("win"), 1, np.where(r2.eq("win"), 0, np.nan))

fights["target"] = target
fights.loc[mask_bad, "target"] = np.nan
fights = fights.dropna(subset=["target"]).copy()
fights["target"] = fights["target"].astype(int)


wanted = [
    "Full Name",
    "Ht.",
    "Wt.",
    "Reach",
    "Stance",
    "W",
    "L",
    "D",
    "Belt",
    "KD",
    "STR",
    "TD",
    "SUB",
    "Ctrl",
    "Sig. Str. %",
    "Head_%",
    "Body_%",
    "Leg_%",
    "Distance_%",
    "Clinch_%",
    "Ground_%",
    "Sub. Att",
    "Weight_Class",
    "Fighting Style",
]

available = [c for c in wanted if c in fstats.columns]
fighter_profile = (
    fstats[available].copy()
    if (not fstats.empty and available)
    else pd.DataFrame(columns=wanted)
)

if not fighters_base.empty:
    for col in [
        "Ht.",
        "Wt.",
        "Reach",
        "Stance",
        "W",
        "L",
        "D",
        "Weight_Class",
        "Fighting Style",
    ]:
        if col in fighters_base.columns:
            tmp = fighters_base[["Full Name", col]].copy()
            tmp["Full Name"] = tmp["Full Name"].apply(norm_name)
            if fighter_profile.empty:
                fighter_profile = tmp
            else:
                fighter_profile = fighter_profile.merge(
                    tmp, on="Full Name", how="outer"
                )

fighter_profile["Full Name"] = fighter_profile["Full Name"].fillna("").apply(norm_name)
fighter_profile = fighter_profile.drop_duplicates(subset=["Full Name"]).set_index(
    "Full Name", drop=False
)

for c in fighter_profile.columns:
    if c not in ["Full Name", "Weight_Class", "Stance", "Fighting Style"]:
        fighter_profile[c] = pd.to_numeric(fighter_profile[c], errors="coerce")


def compute_finish_stats(fdf):
    """
    INPUT: fights df with columns:
       Fighter_1, Fighter_2, Method, Round
    OUTPUT: per-fighter finish stats
    """
    df = fdf.copy()
    df["f1"] = df["Fighter_1"].astype(str).apply(norm_name)
    df["f2"] = df["Fighter_2"].astype(str).apply(norm_name)

    df["Method"] = df["Method"].astype(str).str.lower()
    df["is_finish"] = df["Method"].apply(
        lambda m: int(
            ("ko" in m) or ("tko" in m) or ("sub" in m) or ("submission" in m)
        )
    )

    df["Round_num"] = pd.to_numeric(df["Round"], errors="coerce")

    all_names = pd.unique(pd.concat([df["f1"], df["f2"]]))
    rows = []

    for name in all_names:
        p = df[(df["f1"] == name) | (df["f2"] == name)]
        total = len(p)
        if total == 0:
            continue

        finishes = p[p["is_finish"] == 1]
        fn = len(finishes)

        finish_rate = fn / total

        r1 = len(finishes[finishes["Round_num"] == 1]) / total
        r2 = len(finishes[finishes["Round_num"] == 2]) / total
        r3p = len(finishes[finishes["Round_num"] >= 3]) / total

        avg_fr = finishes["Round_num"].mean() if fn > 0 else 0.0

        rows.append(
            {
                "name": name,
                "finish_rate": finish_rate,
                "pct_finish_r1": r1,
                "pct_finish_r2": r2,
                "pct_finish_r3p": r3p,
                "avg_finish_round": avg_fr,
            }
        )

    return pd.DataFrame(rows).set_index("name")


finish_stats = compute_finish_stats(fights)


fighter_profile = fighter_profile.join(finish_stats, how="left")
for col in [
    "finish_rate",
    "pct_finish_r1",
    "pct_finish_r2",
    "pct_finish_r3p",
    "avg_finish_round",
]:
    if col in fighter_profile:
        fighter_profile[col] = fighter_profile[col].fillna(0.0)

static_profile = {
    name: {**row.to_dict()}
    for name, row in fighter_profile.set_index("Full Name").iterrows()
}


running = defaultdict(
    lambda: {
        "n": 0,
        "wins": 0,
        "losses": 0,
        "kd_sum": 0.0,
        "str_sum": 0.0,
        "td_sum": 0.0,
        "sub_sum": 0.0,
        "ctrl_sum": 0.0,
        "sig_sum": 0.0,
        "elo": ELO_INIT,
        "last_results": deque(maxlen=LAST_N),
        "last_kd": deque(maxlen=LAST_N),
        "last_str": deque(maxlen=LAST_N),
        "last_td": deque(maxlen=LAST_N),
        "last_ctrl": deque(maxlen=LAST_N),
    }
)


def snapshot(name):
    s = running[name]
    n = s["n"]
    return {
        "n": n,
        "winrate": safe_div(s["wins"], n),
        "avg_kd": safe_div(s["kd_sum"], n),
        "avg_str": safe_div(s["str_sum"], n),
        "avg_td": safe_div(s["td_sum"], n),
        "avg_sub": safe_div(s["sub_sum"], n),
        "avg_ctrl": safe_div(s["ctrl_sum"], n),
        "avg_sig": safe_div(s["sig_sum"], n),
        "last_winrate": safe_div(sum(s["last_results"]), len(s["last_results"]))
        if s["last_results"]
        else 0,
        "last_avg_str": safe_div(sum(s["last_str"]), len(s["last_str"]))
        if s["last_str"]
        else 0,
        "elo": s["elo"],
    }


out_rows = []
f_sorted = fights.sort_values("fight_index")

for _, f in f_sorted.iterrows():
    r = f["_R_key"]
    b = f["_B_key"]

    snap_r = snapshot(r)
    snap_b = snapshot(b)

    r_static = static_profile.get(r, {})
    b_static = static_profile.get(b, {})

    row = {
        "fight_index": f["fight_index"],
        "fight_id": f.get("fight_id", np.nan),
        "target": int(f["target"]),
        "R_Ht.": r_static.get("Ht.", np.nan),
        "R_Wt.": r_static.get("Wt.", np.nan),
        "R_Reach": r_static.get("Reach", np.nan),
        "R_Stance": r_static.get("Stance", "Unknown"),
        "R_Weight_Class": r_static.get("Weight_Class", "Unknown"),
        "R_Fighting Style": r_static.get("Fighting Style", "Unknown"),
        "B_Ht.": b_static.get("Ht.", np.nan),
        "B_Wt.": b_static.get("Wt.", np.nan),
        "B_Reach": b_static.get("Reach", np.nan),
        "B_Stance": b_static.get("Stance", "Unknown"),
        "B_Weight_Class": b_static.get("Weight_Class", "Unknown"),
        "B_Fighting Style": b_static.get("Fighting Style", "Unknown"),
        "R_FightCount": snap_r["n"],
        "R_WinRate": snap_r["winrate"],
        "R_avg_KD": snap_r["avg_kd"],
        "R_avg_STR": snap_r["avg_str"],
        "R_avg_TD": snap_r["avg_td"],
        "R_avg_SUB": snap_r["avg_sub"],
        "R_avg_Ctrl": snap_r["avg_ctrl"],
        "R_avg_Sig": snap_r["avg_sig"],
        "R_last_winrate": snap_r["last_winrate"],
        "R_last_avg_STR": snap_r["last_avg_str"],
        "R_ELO": snap_r["elo"],
        "B_FightCount": snap_b["n"],
        "B_WinRate": snap_b["winrate"],
        "B_avg_KD": snap_b["avg_kd"],
        "B_avg_STR": snap_b["avg_str"],
        "B_avg_TD": snap_b["avg_td"],
        "B_avg_SUB": snap_b["avg_sub"],
        "B_avg_Ctrl": snap_b["avg_ctrl"],
        "B_avg_Sig": snap_b["avg_sig"],
        "B_last_winrate": snap_b["last_winrate"],
        "B_last_avg_STR": snap_b["last_avg_str"],
        "B_ELO": snap_b["elo"],
        "Weight_Class": f.get("Weight_Class", "Unknown"),
        "_R_key": r,
        "_B_key": b,
    }

    for col in [
        "finish_rate",
        "pct_finish_r1",
        "pct_finish_r2",
        "pct_finish_r3p",
        "avg_finish_round",
    ]:
        row[f"R_{col}"] = r_static.get(col, 0.0)
        row[f"B_{col}"] = b_static.get(col, 0.0)
        row[f"DIFF_{col}"] = row[f"R_{col}"] - row[f"B_{col}"]
    diff_keys = [
        "FightCount",
        "WinRate",
        "avg_KD",
        "avg_STR",
        "avg_TD",
        "avg_SUB",
        "avg_Ctrl",
        "avg_Sig",
        "last_winrate",
        "last_avg_STR",
        "ELO",
    ]
    for k in diff_keys:
        row["DIFF_" + k] = row["R_" + k] - row["B_" + k]

    out_rows.append(row)

    def num(col, default=0.0):
        return float(pd.to_numeric(f.get(col, default), errors="coerce") or 0.0)

    obs_r_kd = num("KD_1")
    obs_b_kd = num("KD_2")
    obs_r_str = num("STR_1")
    obs_b_str = num("STR_2")
    obs_r_td = num("TD_1")
    obs_b_td = num("TD_2")
    obs_r_sub = num("SUB_1")
    obs_b_sub = num("SUB_2")
    obs_r_ctrl = num("Ctrl_1")
    obs_b_ctrl = num("Ctrl_2")
    obs_r_sig = num("Sig. Str. %_1")
    obs_b_sig = num("Sig. Str. %_2")

    result_r = int(f["target"])
    result_b = 1 - result_r

    rr = running[r]
    rr["n"] += 1
    rr["wins"] += result_r
    rr["losses"] += 1 - result_r
    rr["kd_sum"] += obs_r_kd
    rr["str_sum"] += obs_r_str
    rr["td_sum"] += obs_r_td
    rr["sub_sum"] += obs_r_sub
    rr["ctrl_sum"] += obs_r_ctrl
    rr["sig_sum"] += obs_r_sig
    rr["last_results"].append(result_r)
    rr["last_kd"].append(obs_r_kd)
    rr["last_str"].append(obs_r_str)
    rr["last_td"].append(obs_r_td)
    rr["last_ctrl"].append(obs_r_ctrl)

    bb = running[b]
    bb["n"] += 1
    bb["wins"] += result_b
    bb["losses"] += 1 - result_b
    bb["kd_sum"] += obs_b_kd
    bb["str_sum"] += obs_b_str
    bb["td_sum"] += obs_b_td
    bb["sub_sum"] += obs_b_sub
    bb["ctrl_sum"] += obs_b_ctrl
    bb["sig_sum"] += obs_b_sig
    bb["last_results"].append(result_b)
    bb["last_kd"].append(obs_b_kd)
    bb["last_str"].append(obs_b_str)
    bb["last_td"].append(obs_b_td)
    bb["last_ctrl"].append(obs_b_ctrl)

    exp_r = 1.0 / (1.0 + 10 ** ((bb["elo"] - rr["elo"]) / 400))
    score_r = result_r
    margin = 1.0 + 0.1 * abs(obs_r_kd - obs_b_kd) + 0.001 * abs(obs_r_str - obs_b_str)
    rr["elo"] += ELO_K * margin * (score_r - exp_r)
    bb["elo"] += ELO_K * margin * ((1 - score_r) - (1 - exp_r))

df = pd.DataFrame(out_rows).set_index("fight_index").sort_index()

df = df[~((df["R_FightCount"] == 0) & (df["B_FightCount"] == 0))].copy()

num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for c in [
    "R_Stance",
    "B_Stance",
    "Weight_Class",
    "R_Weight_Class",
    "B_Weight_Class",
    "R_Fighting Style",
    "B_Fighting Style",
]:
    if c in df:
        df[c] = df[c].fillna("Unknown")


ohe_cols = [
    "Weight_Class",
    "R_Stance",
    "B_Stance",
    "R_Fighting Style",
    "B_Fighting Style",
]
df = pd.get_dummies(
    df, columns=[c for c in ohe_cols if c in df.columns], drop_first=True
)

for c in df.columns:
    if pd.api.types.is_float_dtype(df[c]):
        if not df[c].isna().any() and (df[c] % 1 == 0).all():
            df[c] = df[c].astype("int64")


processed_path = os.path.join(OUT, "processed.csv")
df.to_csv(processed_path, index=False)

profiles = []
for name, snap in running.items():
    s = snapshot(name)
    base = static_profile.get(name, {})
    profiles.append(
        {
            "name": name,
            "elo": s["elo"],
            "fight_count": s["n"],
            "winrate": s["winrate"],
            "avg_kd": s["avg_kd"],
            "avg_str": s["avg_str"],
            "avg_td": s["avg_td"],
            "avg_ctrl": s["avg_ctrl"],
            "last_winrate": s["last_winrate"],
            "last_avg_str": s["last_avg_str"],
            "Ht.": base.get("Ht.", np.nan),
            "Wt.": base.get("Wt.", np.nan),
            "Reach": base.get("Reach", np.nan),
            "Stance": base.get("Stance", "Unknown"),
            "Weight_Class": base.get("Weight_Class", "Unknown"),
            "Fighting Style": base.get("Fighting Style", "Unknown"),
            "finish_rate": base.get("finish_rate", 0.0),
            "pct_finish_r1": base.get("pct_finish_r1", 0.0),
            "pct_finish_r2": base.get("pct_finish_r2", 0.0),
            "pct_finish_r3p": base.get("pct_finish_r3p", 0.0),
            "avg_finish_round": base.get("avg_finish_round", 0.0),
        }
    )

profiles_df = pd.DataFrame(profiles).set_index("name")
profiles_path = os.path.join(OUT, "fighters_profiles.parquet")
profiles_df.to_parquet(profiles_path)

print("✅ processed.csv saved:", processed_path)
print("✅ fighters_profiles.parquet saved:", profiles_path)
print("📦 processed shape:", df.shape)
