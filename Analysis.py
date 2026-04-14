"""
PatientSim Undertriage Analysis
================================
Mixed-effects linear probability models examining the effect of simulated
patient persona characteristics on LLM triage undertriage risk.

Inputs:
    - Single-turn control CSV
    - Single-turn personas CSV
    - Multi-turn personas CSV

Outputs:
    - Descriptive undertriage rates by factor
    - Single-turn mixed-effects LPM
    - Multi-turn mixed-effects LPM
    - Pooled interaction model (single-turn x multi-turn)
    - Forest plot (PDF, SVG, PNG)
    - Publication table (PDF, SVG, PNG)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats as spstats
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Configuration

SINGLE_TURN_CONTROL = "results/Final Runs/patientsim_single_turn_control_gpt-5-mini_20260404_123907.csv"
SINGLE_TURN_PERSONAS = "results/Final Runs/patientsim_single_turn_gpt-5-mini_20260404_124102.csv"
MULTI_TURN_PERSONAS = "results/Final Runs/patientsim_multi_turn_gpt-5-mini_20260404_140955.csv"

OUTPUT_STEM = "Analysis_Forest_Plots"

# Reference levels
REF_CEFR = "B"
REF_PERSONALITY = "plain"
REF_RECALL = "high"
REF_DAZED = "normal"

# Significance threshold for inclusion in forest plot
P_THRESHOLD = 0.10

# Helper functions


def classify_case(gold_standard):
    g = str(gold_standard).strip()
    if g == "D":
        return "emergent"
    elif g in ("C", "C/D"):
        return "urgent"
    else:
        return "other"


def is_undertriage(row):
    cat = row["case_type"]
    resp = str(row["llm_response"]).strip()
    if cat == "emergent":
        return int(resp in ("A", "B", "C"))
    elif cat == "urgent":
        return int(resp in ("A", "B"))
    else:
        return np.nan


def prepare_data(df):
    df = df.copy()
    df["case_type"] = df["gold_standard"].apply(classify_case)
    df["undertriage"] = df.apply(is_undertriage, axis=1)
    df["cefr_type"] = pd.Categorical(df["cefr_type"], categories=[REF_CEFR, "A", "C"])
    df["personality_type"] = pd.Categorical(
        df["personality_type"],
        categories=[REF_PERSONALITY, "pleasing", "verbose", "impatient", "distrust", "overanxious"],
    )
    df["recall_level_option"] = pd.Categorical(df["recall_level_option"], categories=[REF_RECALL, "low"])
    df["dazed_level_option"] = pd.Categorical(df["dazed_level_option"], categories=[REF_DAZED, "high"])
    return df


def filter_triageable(df):
    return df[df["case_type"].isin(["emergent", "urgent"])].dropna(subset=["undertriage"]).copy()


def fit_lpm(df, formula):
    model = smf.mixedlm(formula, df, groups=df["case"])
    return model.fit(reml=True)


def get_direct(result, param_key):
    coef = result.fe_params[param_key] * 100
    ci = result.conf_int().loc[param_key] * 100
    p = result.pvalues[param_key]
    return coef, ci[0], ci[1], p


def get_summed(result, pmain, pinter):
    params = result.fe_params
    vcov = result.cov_params()
    c = (params[pmain] + params[pinter]) * 100
    se = np.sqrt(vcov.loc[pmain, pmain] + vcov.loc[pinter, pinter] + 2 * vcov.loc[pmain, pinter]) * 100
    lo, hi = c - 1.96 * se, c + 1.96 * se
    z = c / se if se > 0 else 0
    p = 2 * (1 - spstats.norm.cdf(abs(z)))
    return c, lo, hi, p


def wilson_ci(series):
    n = len(series)
    p = series.mean()
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, centre - margin, centre + margin, n


def case_ut_counts(sub):
    if len(sub) == 0:
        return 0, 0, 0.0, 0, 0.0
    case_sums = sub.groupby("case")["undertriage"].sum()
    n_cases = len(case_sums)
    n_ge1 = int((case_sums >= 1).sum())
    n_ge3 = int((case_sums >= 3).sum())
    pct_ge1 = n_ge1 / n_cases * 100 if n_cases > 0 else 0.0
    pct_ge3 = n_ge3 / n_cases * 100 if n_cases > 0 else 0.0
    return n_cases, n_ge1, pct_ge1, n_ge3, pct_ge3


def print_model_table(title, result, param_labels):
    print(f"\n{'=' * 75}")
    print(title)
    print(f"{'=' * 75}")
    hdr = f"{'Factor':<55} {'pp':>7} {'95% CI':>18} {'P':>8}"
    print(hdr)
    print("-" * 75)
    for pk, label in param_labels.items():
        if pk not in result.fe_params.index:
            continue
        coef, lo, hi, p = get_direct(result, pk)
        pstr = "<0.001" if p < 0.001 else f"{p:.3f}"
        print(f"{label:<55} {coef:>+6.1f}  ({lo:>+5.1f}, {hi:>+5.1f})  {pstr:>8}")


# Load and prepare data

print("Loading data...")
ctrl = prepare_data(pd.read_csv(SINGLE_TURN_CONTROL))
st = prepare_data(pd.read_csv(SINGLE_TURN_PERSONAS))
mt = prepare_data(pd.read_csv(MULTI_TURN_PERSONAS))

print(f"  Control:     {len(ctrl):>6} rows, {ctrl['case'].nunique()} cases")
print(f"  Single-turn: {len(st):>6} rows, {st['case'].nunique()} cases")
print(f"  Multi-turn:  {len(mt):>6} rows, {mt['case'].nunique()} cases")

# Descriptive undertriage rates (verbose)

endash = "\u2013"
dagger = "\u2020"
ddagger = "\u2021"
geq = "\u2265"

print(f"\n\nDESCRIPTIVE UNDERTRIAGE RATES (Wilson 95% CI)")
print("=" * 115)
print(
    f"{'Condition':<16} {'Subgroup':<16} {'n':>5} "
    f"{'Undertriage':>14} {'95% CI':>20} "
    f"{'Cases '+geq+'1 UT':>14} {'Cases '+geq+'3 UT':>14}"
)
print("-" * 115)

table_rows = []

for label, df in [("Control", ctrl), ("Single-turn", st), ("Multi-turn", mt)]:
    rel = filter_triageable(df)

    if label in ("Single-turn", "Multi-turn"):
        base = rel[
            (rel["personality_type"] == REF_PERSONALITY)
            & (rel["cefr_type"] == REF_CEFR)
            & (rel["recall_level_option"] == REF_RECALL)
            & (rel["dazed_level_option"] == REF_DAZED)
        ]
        subgroups = [
            ("Overall", rel),
            ("Urgent", rel[rel["case_type"] == "urgent"]),
            ("Emergent", rel[rel["case_type"] == "emergent"]),
            ("Base case", base),
            ("Base " + endash + " Urgent", base[base["case_type"] == "urgent"]),
            ("Base " + endash + " Emergent", base[base["case_type"] == "emergent"]),
        ]
        if label == "Multi-turn":
            persona_rates = rel.groupby("persona_id")["undertriage"].mean().sort_values()
            best_id = persona_rates.index[0]
            worst_id = persona_rates.index[-1]
            best_sub = rel[rel["persona_id"] == best_id]
            worst_sub = rel[rel["persona_id"] == worst_id]
            subgroups += [
                ("Best" + dagger, best_sub),
                ("Best " + endash + " Urgent", best_sub[best_sub["case_type"] == "urgent"]),
                ("Best " + endash + " Emergent", best_sub[best_sub["case_type"] == "emergent"]),
                ("Worst" + ddagger, worst_sub),
                ("Worst " + endash + " Urgent", worst_sub[worst_sub["case_type"] == "urgent"]),
                ("Worst " + endash + " Emergent", worst_sub[worst_sub["case_type"] == "emergent"]),
            ]
    else:
        subgroups = [
            ("Overall", rel),
            ("Urgent", rel[rel["case_type"] == "urgent"]),
            ("Emergent", rel[rel["case_type"] == "emergent"]),
        ]

    for subgroup, sub in subgroups:
        p, lo, hi, n = wilson_ci(sub["undertriage"])
        n_ut = int(sub["undertriage"].sum())
        n_cases, n_ge1, pct_ge1, n_ge3, pct_ge3 = case_ut_counts(sub)
        row_label = label if subgroup == "Overall" else ""
        if subgroup == "Base case":
            row_label = "  Base case*"
        elif subgroup.startswith("Best"):
            row_label = "  Best" + dagger if subgroup == "Best" + dagger else ""
        elif subgroup.startswith("Worst"):
            row_label = "  Worst" + ddagger if subgroup == "Worst" + ddagger else ""
        ge1_str = f"{n_ge1}/{n_cases} ({pct_ge1:.1f}%)"
        ge3_str = f"{n_ge3}/{n_cases} ({pct_ge3:.1f}%)"
        ut_str = f"{n_ut} ({p * 100:.1f}%)"
        print(
            f"{row_label:<16} {subgroup:<16} {n:>5} "
            f"{ut_str:>14} "
            f"{lo * 100:>8.1f}% {endash} {hi * 100:.1f}% "
            f"{ge1_str:>14} {ge3_str:>14}"
        )
        table_rows.append({
            "Condition": label, "Subgroup": subgroup, "n": n,
            "Undertriage events": n_ut, "Undertriage (%)": round(p * 100, 1),
            "95% CI lower (%)": round(lo * 100, 1), "95% CI upper (%)": round(hi * 100, 1),
            "Cases total": n_cases, "Cases with >=1 UT": n_ge1,
            "Cases >=1 UT (%)": round(pct_ge1, 1),
            "Cases with >=3 UT": n_ge3, "Cases >=3 UT (%)": round(pct_ge3, 1),
        })
    print()

table_df = pd.DataFrame(table_rows)
table_df.to_csv(f"Analysis_Raw_Results_Table.csv", index=False)
print("Saved: Analysis_Raw_Results_Table.csv")
print(f"*Base case: {REF_PERSONALITY} personality, CEFR {REF_CEFR}, {REF_RECALL} recall, {REF_DAZED} cognition")
if "best_id" in dir():
    print(f"{dagger}Best persona: {best_id}")
    print(f"{ddagger}Worst persona: {worst_id}")

# Personality breakdown
for label, df in [("Single-turn", st), ("Multi-turn", mt)]:
    rel = filter_triageable(df)
    print(f"\n{label} undertriage by personality:")
    tbl = rel.groupby("personality_type")["undertriage"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    for idx, row in tbl.iterrows():
        print(f"  {idx:<15} {row['mean']:.3f}  (n={int(row['count'])})")

# Publication table (condensed) - no n column, denominators in footnote

print("\n\nTABLE FOR PUBLICATION (condensed)")
print("=" * 100)

pub_rows = []

for label, df in [("Control", ctrl), ("Single-turn", st), ("Multi-turn", mt)]:
    rel = filter_triageable(df)
    p, lo, hi, n = wilson_ci(rel["undertriage"])
    n_ut = int(rel["undertriage"].sum())
    nc, n1, p1, n3, p3 = case_ut_counts(rel)
    pub_rows.append({
        "row_label": f"{label} (all personas)" if label != "Control" else "Control",
        "n_obs": n, "n_ut": n_ut, "ut": p, "ci_lo": lo, "ci_hi": hi,
        "n_cases": nc, "n_ge1": n1, "pct_ge1": p1, "n_ge3": n3, "pct_ge3": p3,
    })

mt_rel = filter_triageable(mt)
mt_base = mt_rel[
    (mt_rel["personality_type"] == REF_PERSONALITY)
    & (mt_rel["cefr_type"] == REF_CEFR)
    & (mt_rel["recall_level_option"] == REF_RECALL)
    & (mt_rel["dazed_level_option"] == REF_DAZED)
]
mt_persona_rates = mt_rel.groupby("persona_id")["undertriage"].mean().sort_values()
mt_best_id = mt_persona_rates.index[0]
mt_worst_id = mt_persona_rates.index[-1]
mt_best = mt_rel[mt_rel["persona_id"] == mt_best_id]
mt_worst = mt_rel[mt_rel["persona_id"] == mt_worst_id]

for sub_label, sub in [
    ("  Base case*", mt_base),
    ("  Best persona" + dagger, mt_best),
    ("  Worst persona" + ddagger, mt_worst),
]:
    p, lo, hi, n = wilson_ci(sub["undertriage"])
    n_ut = int(sub["undertriage"].sum())
    nc, n1, p1, n3, p3 = case_ut_counts(sub)
    pub_rows.append({
        "row_label": sub_label,
        "n_obs": n, "n_ut": n_ut, "ut": p, "ci_lo": lo, "ci_hi": hi,
        "n_cases": nc, "n_ge1": n1, "pct_ge1": p1, "n_ge3": n3, "pct_ge3": p3,
    })

print(
    f"{'':40} {'Undertriage':>14}  {'95% CI':>18}  "
    f"{'Cases '+geq+'1 UT':>14}  {'Cases '+geq+'3 UT':>14}"
)
print("-" * 105)
for r in pub_rows:
    ci_str = f"{r['ci_lo']*100:.1f}{endash}{r['ci_hi']*100:.1f}%"
    ut_str = f"{r['n_ut']} ({r['ut']*100:.1f}%)"
    ge1 = f"{r['n_ge1']} ({r['pct_ge1']:.0f}%)"
    ge3 = f"{r['n_ge3']} ({r['pct_ge3']:.0f}%)"
    print(
        f"{r['row_label']:40} "
        f"{ut_str:>14}  {ci_str:>18}  "
        f"{ge1:>14}  {ge3:>14}"
    )
print("-" * 105)
print(
    f"Each all-personas row: 28 cases x 37 configurations x 5 trials (n=5,180). "
    f"Control, base, best, worst: 28 cases x 5 trials (n=140)."
)
print(f"Case counts out of 28 total (22 urgent, 6 emergent).")
print(
    f"*{REF_PERSONALITY}, CEFR {REF_CEFR}, {REF_RECALL} recall, {REF_DAZED} cognition. "
    f"{dagger}Overanxious, CEFR B, high recall, normal cognition. "
    f"{ddagger}Pleasing, CEFR C, low recall, normal cognition."
)

pub_fig, pub_ax = plt.subplots(figsize=(11, 3.5))
pub_ax.axis("off")

col_labels = [
    "", "Undertriage,\nn (%)", "95% CI",
    "Cases with\n" + geq + "1 undertriage",
    "Cases with\n" + geq + "3 undertriage",
]

cell_data = []
for r in pub_rows:
    ci_str = f"{r['ci_lo']*100:.1f}{endash}{r['ci_hi']*100:.1f}"
    ut_str = f"{r['n_ut']} ({r['ut']*100:.1f}%)"
    ge1 = f"{r['n_ge1']} ({r['pct_ge1']:.0f}%)"
    ge3 = f"{r['n_ge3']} ({r['pct_ge3']:.0f}%)"
    cell_data.append([r["row_label"], ut_str, ci_str, ge1, ge3])

tbl = pub_ax.table(
    cellText=cell_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colWidths=[0.32, 0.15, 0.17, 0.18, 0.18],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    cell.set_linewidth(0.5)
    if row == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_height(0.12)
    else:
        cell.set_facecolor("white")
        cell.set_height(0.09)
    if col == 0:
        cell.set_text_props(ha="left")
        cell._loc = "left"

worst_row_idx = len(pub_rows)
for col in range(len(col_labels)):
    tbl[(worst_row_idx, col)].set_text_props(fontweight="bold")

pub_ax.set_title(
    "Undertriage Rates by Interaction Mode and Persona Configuration (GPT-5-mini)",
    fontsize=10, fontweight="bold", pad=15,
)

footnote = (
    f"All-personas rows: 28 cases \u00d7 37 persona configurations \u00d7 5 trials (n = 5,180). "
    f"Control, base case, best, and worst rows: 28 cases \u00d7 5 trials (n = 140). "
    f"Case counts out of 28 total (22 urgent, 6 emergent).\n"
    f"*{REF_PERSONALITY}, CEFR {REF_CEFR}, {REF_RECALL} recall, {REF_DAZED} cognition.  "
    f"{dagger}Overanxious, CEFR B, high recall, normal cognition.  "
    f"{ddagger}Pleasing, CEFR C, low recall, normal cognition."
)
pub_fig.text(0.05, 0.01, footnote, fontsize=6.5, color="#444444", va="bottom")

for fmt in ["pdf", "svg", "png"]:
    outpath = f"Analysis_Table_1.{fmt}"
    kwargs = {"dpi": 300} if fmt == "png" else {}
    pub_fig.savefig(outpath, format=fmt, bbox_inches="tight", facecolor="white", **kwargs)
print(f"\nSaved publication table: Analysis_Table_1.pdf / .svg / .png")

# Model formulas

BASE_FORMULA = (
    "undertriage ~ C(personality_type) + C(cefr_type) "
    "+ C(recall_level_option) + C(dazed_level_option)"
)

INTERACTION_FORMULA = (
    "undertriage ~ C(turn_type) * (C(personality_type) + C(cefr_type) "
    "+ C(recall_level_option) + C(dazed_level_option))"
)

BASE_PARAM_LABELS = {
    "Intercept": f"Baseline ({REF_PERSONALITY}, CEFR {REF_CEFR}, {REF_RECALL} recall, {REF_DAZED} cognition)",
    "C(personality_type)[T.pleasing]": "Pleasing",
    "C(personality_type)[T.verbose]": "Verbose",
    "C(personality_type)[T.impatient]": "Impatient",
    "C(personality_type)[T.distrust]": "Distrustful",
    "C(personality_type)[T.overanxious]": "Overanxious",
    "C(cefr_type)[T.A]": "CEFR A (low proficiency)",
    "C(cefr_type)[T.C]": "CEFR C (high proficiency)",
    "C(recall_level_option)[T.low]": "Low recall",
    "C(dazed_level_option)[T.high]": "Cognitive impairment",
}

# Model 1: Single-turn

st_rel = filter_triageable(st)
st_res = fit_lpm(st_rel, BASE_FORMULA)
print_model_table(f"MODEL 1: SINGLE-TURN (N={len(st_rel)}, {st_rel['case'].nunique()} cases)", st_res, BASE_PARAM_LABELS)

# Model 2: Multi-turn

mt_rel2 = filter_triageable(mt)
mt_res = fit_lpm(mt_rel2, BASE_FORMULA)
print_model_table(f"MODEL 2: MULTI-TURN (N={len(mt_rel2)}, {mt_rel2['case'].nunique()} cases)", mt_res, BASE_PARAM_LABELS)

# Model 3: Pooled interaction model

st["turn_type"] = "single"
mt["turn_type"] = "multi"
combined = pd.concat([st, mt], ignore_index=True)
crel = filter_triageable(combined)
crel["turn_type"] = pd.Categorical(crel["turn_type"], categories=["single", "multi"])

int_res = fit_lpm(crel, INTERACTION_FORMULA)

INT_PARAM_LABELS = {
    "Intercept": "Baseline (single-turn, plain, CEFR B, high recall, normal)",
    "C(turn_type)[T.multi]": "Multi-turn (main effect)",
    **{k: v for k, v in BASE_PARAM_LABELS.items() if k != "Intercept"},
    "C(turn_type)[T.multi]:C(personality_type)[T.pleasing]": "Multi-turn \u00d7 Pleasing",
    "C(turn_type)[T.multi]:C(personality_type)[T.verbose]": "Multi-turn \u00d7 Verbose",
    "C(turn_type)[T.multi]:C(personality_type)[T.impatient]": "Multi-turn \u00d7 Impatient",
    "C(turn_type)[T.multi]:C(personality_type)[T.distrust]": "Multi-turn \u00d7 Distrustful",
    "C(turn_type)[T.multi]:C(personality_type)[T.overanxious]": "Multi-turn \u00d7 Overanxious",
    "C(turn_type)[T.multi]:C(cefr_type)[T.A]": "Multi-turn \u00d7 CEFR A",
    "C(turn_type)[T.multi]:C(cefr_type)[T.C]": "Multi-turn \u00d7 CEFR C",
    "C(turn_type)[T.multi]:C(recall_level_option)[T.low]": "Multi-turn \u00d7 Low recall",
    "C(turn_type)[T.multi]:C(dazed_level_option)[T.high]": "Multi-turn \u00d7 Cognitive impairment",
}

print_model_table(f"MODEL 3: POOLED INTERACTION (N={len(crel)}, {crel['case'].nunique()} cases)", int_res, INT_PARAM_LABELS)

# Net effects in multi-turn
PARAM_KEYS = [
    ("C(personality_type)[T.pleasing]", "Pleasing"),
    ("C(personality_type)[T.verbose]", "Verbose"),
    ("C(personality_type)[T.impatient]", "Impatient"),
    ("C(personality_type)[T.distrust]", "Distrustful"),
    ("C(personality_type)[T.overanxious]", "Overanxious"),
    ("C(cefr_type)[T.A]", "CEFR A (low proficiency)"),
    ("C(cefr_type)[T.C]", "CEFR C (high proficiency)"),
    ("C(recall_level_option)[T.low]", "Low recall"),
    ("C(dazed_level_option)[T.high]", "Cognitive impairment"),
]

times = "\u00d7"
print(f"\n{'=' * 75}")
print("NET EFFECTS IN MULTI-TURN (main + interaction, from pooled model)")
print(f"{'=' * 75}")
print(f"{'Factor':<55} {'pp':>7} {'95% CI':>18} {'P':>8}")
print("-" * 75)
for pk, label in PARAM_KEYS:
    ik = f"C(turn_type)[T.multi]:{pk}"
    c, lo, hi, p = get_summed(int_res, pk, ik)
    pstr = "<0.001" if p < 0.001 else f"{p:.3f}"
    print(f"Multi-turn {times} {label:<43} {c:>+6.1f}  ({lo:>+5.1f}, {hi:>+5.1f})  {pstr:>8}")

# Forest plot

print("\nGenerating forest plot...")

st_rows_fp = [(label, *get_direct(st_res, pk)) for pk, label in PARAM_KEYS]

int_rows_fp = []
for pk, label in PARAM_KEYS:
    ik = f"C(turn_type)[T.multi]:{pk}"
    int_rows_fp.append((f"Multi-turn {times} {label}", *get_summed(int_res, pk, ik)))

keep_base = set()
for i, (pk, label) in enumerate(PARAM_KEYS):
    p_st = st_rows_fp[i][4]
    p_int = int_rows_fp[i][4]
    if p_st < P_THRESHOLD or p_int < P_THRESHOLD:
        keep_base.add(label)

st_rows_fp = [r for r in st_rows_fp if r[0] in keep_base]
int_rows_fp = [r for r in int_rows_fp if r[0].replace(f"Multi-turn {times} ", "") in keep_base]

print(f"Parameters included (P<{P_THRESHOLD} in {geq}1 model): {sorted(keep_base)}")

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(16, 4.5), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1.15]}
)


def draw_panel(ax, rows, title, panel_letter, xlim, annot_x):
    ax.set_xlim(xlim)
    for i, (label, coef, lo, hi, p) in enumerate(rows):
        ypos = len(rows) - 1 - i
        ax.plot([lo, hi], [ypos, ypos], color="black", linewidth=1.4, solid_capstyle="butt")
        ax.plot(coef, ypos, "s", color="black", markersize=5, markeredgecolor="black", markeredgewidth=0.6)
        pstr = "P<.001" if p < 0.001 else f"P={p:.2f}"
        annot = f"{coef:+.1f} ({lo:+.1f} {endash} {hi:+.1f})  {pstr}"
        ax.text(annot_x, ypos, annot, fontsize=7.5, va="center", color="black")
    ax.axvline(0, color="black", linewidth=0.6, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8.5)
    delta = "\u0394"
    ax.set_xlabel(f"{delta} Undertriage Risk, Percentage Points", fontsize=8)
    ax.tick_params(left=False, labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.text(-0.02, 1.10, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.text(0.05, 1.10, title, transform=ax.transAxes, fontsize=9.5, va="top")


draw_panel(ax_a, st_rows_fp, "Single-Turn Model", "A", xlim=(-12, 12), annot_x=12.5)
draw_panel(ax_b, int_rows_fp, "Net Effect in Multi-Turn (Interaction Model)", "B", xlim=(-12, 16), annot_x=16.5)

fig.text(
    0.04, -0.06,
    f"Reference: {REF_PERSONALITY} personality, CEFR {REF_CEFR}, {REF_RECALL} recall, {REF_DAZED} cognition. "
    f"Parameters shown if P<{P_THRESHOLD:.2f} in {geq}1 model.\n"
    f"Panel A: mixed-effects linear probability model, single-turn data (N\u2009=\u2009{len(st_rel):,}). "
    f"Panel B: net effect (main + interaction) from pooled model (N\u2009=\u2009{len(crel):,}). "
    f"Random intercept by clinical case.",
    fontsize=7, color="#444444", va="top",
)

for fmt in ["pdf", "svg", "png"]:
    outpath = f"{OUTPUT_STEM}.{fmt}"
    kwargs = {"dpi": 300} if fmt == "png" else {}
    plt.savefig(outpath, format=fmt, bbox_inches="tight", facecolor="white", **kwargs)
    print(f"  Saved: {outpath}")

print("\nDone.")
