import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import (ttest_1samp, ttest_ind, ttest_rel,
                         f_oneway, chi2_contingency, mannwhitneyu,
                         wilcoxon, kruskal, shapiro, levene, bartlett)
import itertools
import io
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DOE & Hypothesis Testing Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.03em;
}
.stApp {
    background: #0d1117;
    color: #e6edf3;
}
.sidebar .sidebar-content {
    background: #161b22;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-box {
    background: #0d1117;
    border-left: 4px solid #58a6ff;
    padding: 1rem 1.5rem;
    border-radius: 0 6px 6px 0;
    margin: 0.8rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
}
.reject { border-left-color: #f85149 !important; }
.fail-reject { border-left-color: #3fb950 !important; }
.warning-box {
    background: #2d2200;
    border-left: 4px solid #d29922;
    padding: 0.8rem 1.2rem;
    border-radius: 0 6px 6px 0;
    margin: 0.6rem 0;
    font-size: 0.85rem;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem !important;
    color: #58a6ff;
}
.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}
.stButton > button:hover {
    background: #388bfd;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
}
.badge-reject { background: #3d1c1c; color: #f85149; border: 1px solid #f85149; }
.badge-fail   { background: #1a3a1a; color: #3fb950; border: 1px solid #3fb950; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(v, d=4):
    return f"{v:.{d}f}" if v is not None else "N/A"

def decision(p, alpha):
    if p < alpha:
        return (f'<span class="badge badge-reject">REJECT H₀</span>', "reject")
    return (f'<span class="badge badge-fail">FAIL TO REJECT H₀</span>', "fail-reject")

def effect_size_label(es, kind="cohen_d"):
    if kind == "cohen_d":
        if abs(es) < 0.2: return "negligible"
        if abs(es) < 0.5: return "small"
        if abs(es) < 0.8: return "medium"
        return "large"
    return ""

def cohen_d(a, b):
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled else 0

def plot_defaults():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#8b949e",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "text.color":       "#e6edf3",
        "grid.color":       "#21262d",
        "grid.linestyle":   "--",
        "grid.alpha":       0.6,
    })

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 DOE · HypTest Agent")
    st.markdown('<p class="section-header">Analysis Module</p>', unsafe_allow_html=True)

    module = st.selectbox("Select Module", [
        "📋 Overview",
        "🧪 One-Sample t-Test",
        "⚖️ Two-Sample t-Test",
        "🔗 Paired t-Test",
        "📊 One-Way ANOVA",
        "📐 Two-Way ANOVA (Factorial DOE)",
        "🧩 Chi-Square Test",
        "🏔️ Non-Parametric Tests",
        "🔢 Full Factorial DOE Designer",
        "📈 Diagnostic Plots",
    ])

    st.markdown("---")
    st.markdown('<p class="section-header">Global Settings</p>', unsafe_allow_html=True)
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    st.markdown(f"**α = {alpha}**  ·  Confidence = **{(1-alpha)*100:.0f}%**")

    st.markdown("---")
    st.caption("Built with Python · SciPy · Streamlit\nNo login required · Fully open source")

# ─────────────────────────────────────────────────────────────────────────────
#  OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if module == "📋 Overview":
    st.markdown("# DOE & Hypothesis Testing Agent")
    st.markdown("**Design experiments. Test hypotheses. Interpret results — no statistics PhD required.**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <b>Hypothesis Tests</b><br><br>
        • One-sample t-Test<br>
        • Two-sample t-Test<br>
        • Paired t-Test<br>
        • One-Way ANOVA<br>
        • Chi-Square<br>
        • Non-Parametric suite
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
        <b>Design of Experiments</b><br><br>
        • Two-Way Factorial DOE<br>
        • Full Factorial Designer<br>
        • Main & interaction effects<br>
        • Effect magnitude plots<br>
        • DOE summary tables
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
        <b>Diagnostics</b><br><br>
        • Normality (Shapiro-Wilk)<br>
        • Equal variance tests<br>
        • Q-Q plots<br>
        • Residual plots<br>
        • Distribution overlays
        </div>""", unsafe_allow_html=True)

    st.markdown("### How to Use")
    st.markdown("""
    1. **Pick a module** from the left sidebar  
    2. **Enter or paste your data** (or use the built-in sample data)  
    3. **Configure parameters** and click **Run Analysis**  
    4. Read the **plain-language interpretation** alongside the statistics  
    """)

# ─────────────────────────────────────────────────────────────────────────────
#  ONE-SAMPLE t-TEST
# ─────────────────────────────────────────────────────────────────────────────
elif module == "🧪 One-Sample t-Test":
    st.markdown("# 🧪 One-Sample t-Test")
    st.markdown("Tests whether a sample mean differs significantly from a known/hypothesised population mean (μ₀).")

    col1, col2 = st.columns([2, 1])
    with col1:
        raw = st.text_area("Enter data (comma or newline separated)",
                           "23, 25, 28, 22, 27, 30, 24, 26, 29, 21, 25, 28, 23, 27, 26")
    with col2:
        mu0      = st.number_input("Hypothesised Mean (μ₀)", value=25.0)
        alt_map  = {"two-sided": "two-sided", "greater (sample > μ₀)": "greater", "less (sample < μ₀)": "less"}
        alt_label = st.selectbox("Alternative Hypothesis", list(alt_map.keys()))
        alternative = alt_map[alt_label]

    if st.button("▶ Run One-Sample t-Test"):
        try:
            data = np.array([float(x.strip()) for x in raw.replace("\n", ",").split(",") if x.strip()])
            t_stat, p_val = ttest_1samp(data, mu0, alternative=alternative)
            n   = len(data)
            xbar = np.mean(data)
            s   = np.std(data, ddof=1)
            se  = s / np.sqrt(n)
            df  = n - 1
            ci  = stats.t.interval(1 - alpha, df, loc=xbar, scale=se)

            badge, cls = decision(p_val, alpha)

            st.markdown("### Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("n", n)
            c2.metric("x̄", fmt(xbar, 3))
            c3.metric("s", fmt(s, 3))
            c4.metric("SE", fmt(se, 4))

            st.markdown(f"""
            <div class="result-box {cls}">
            t-statistic = {fmt(t_stat)}&nbsp;&nbsp;|&nbsp;&nbsp;
            p-value = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;
            df = {df}&nbsp;&nbsp;|&nbsp;&nbsp;{badge}<br>
            {(1-alpha)*100:.0f}% CI: [{fmt(ci[0],3)}, {fmt(ci[1],3)}]
            </div>""", unsafe_allow_html=True)

            if p_val < alpha:
                st.success(f"📌 At α={alpha}, there is sufficient evidence to conclude the population mean differs from {mu0}.")
            else:
                st.info(f"📌 At α={alpha}, there is insufficient evidence to conclude the population mean differs from {mu0}.")

            # Plot
            plot_defaults()
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            # histogram
            axes[0].hist(data, bins="auto", color="#1f6feb", edgecolor="#58a6ff", alpha=0.8)
            axes[0].axvline(xbar, color="#f85149", lw=2, label=f"x̄={xbar:.2f}")
            axes[0].axvline(mu0,  color="#d29922", lw=2, ls="--", label=f"μ₀={mu0}")
            axes[0].set_title("Data Distribution", fontsize=11)
            axes[0].legend(fontsize=9)
            axes[0].grid(True)
            # t-distribution
            x_range = np.linspace(-4, 4, 300)
            pdf = stats.t.pdf(x_range, df)
            axes[1].plot(x_range, pdf, color="#58a6ff", lw=2)
            axes[1].axvline(t_stat, color="#f85149", lw=2, label=f"t={t_stat:.3f}")
            crit = stats.t.ppf(1 - alpha/2, df)
            axes[1].fill_between(x_range, pdf, where=(x_range <= -crit) | (x_range >= crit),
                                 color="#f85149", alpha=0.3, label=f"Rejection region")
            axes[1].set_title(f"t-Distribution (df={df})", fontsize=11)
            axes[1].legend(fontsize=9)
            axes[1].grid(True)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  TWO-SAMPLE t-TEST
# ─────────────────────────────────────────────────────────────────────────────
elif module == "⚖️ Two-Sample t-Test":
    st.markdown("# ⚖️ Two-Sample t-Test")
    st.markdown("Tests whether two independent groups have equal means.")

    col1, col2 = st.columns(2)
    with col1:
        raw1   = st.text_area("Group A", "78,82,85,79,83,81,80,84,77,86")
        label1 = st.text_input("Group A label", "Group A")
    with col2:
        raw2   = st.text_area("Group B", "72,75,78,74,76,71,73,77,70,79")
        label2 = st.text_input("Group B label", "Group B")

    equal_var = st.checkbox("Assume equal variances (Student's t)", value=False)
    alt_map   = {"two-sided": "two-sided", "A > B": "greater", "A < B": "less"}
    alt_label = st.selectbox("Alternative Hypothesis", list(alt_map.keys()))
    alternative = alt_map[alt_label]

    if st.button("▶ Run Two-Sample t-Test"):
        try:
            a = np.array([float(x.strip()) for x in raw1.replace("\n",",").split(",") if x.strip()])
            b = np.array([float(x.strip()) for x in raw2.replace("\n",",").split(",") if x.strip()])
            t_stat, p_val = ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
            cd = cohen_d(a, b)
            badge, cls = decision(p_val, alpha)

            st.markdown("### Descriptive Summary")
            df_desc = pd.DataFrame({
                "Group": [label1, label2],
                "n": [len(a), len(b)],
                "Mean": [fmt(np.mean(a),3), fmt(np.mean(b),3)],
                "Std Dev": [fmt(np.std(a,ddof=1),3), fmt(np.std(b,ddof=1),3)],
                "Min": [fmt(min(a),2), fmt(min(b),2)],
                "Max": [fmt(max(a),2), fmt(max(b),2)],
            })
            st.dataframe(df_desc, hide_index=True)

            st.markdown(f"""
            <div class="result-box {cls}">
            t-statistic = {fmt(t_stat)}&nbsp;&nbsp;|&nbsp;&nbsp;
            p-value = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;
            Cohen's d = {fmt(cd,3)} ({effect_size_label(cd)})<br>
            {badge}&nbsp;&nbsp;Mean difference = {fmt(np.mean(a)-np.mean(b),3)}
            </div>""", unsafe_allow_html=True)

            if p_val < alpha:
                st.success(f"📌 Significant difference between {label1} and {label2} at α={alpha}.")
            else:
                st.info(f"📌 No significant difference detected between {label1} and {label2} at α={alpha}.")

            # Levene test
            lev_stat, lev_p = levene(a, b)
            if lev_p < 0.05:
                st.markdown(f'<div class="warning-box">⚠️ Levene\'s test suggests unequal variances (p={fmt(lev_p,4)}). Consider Welch\'s t-test (uncheck equal variances).</div>', unsafe_allow_html=True)

            plot_defaults()
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            palette = ["#1f6feb", "#f85149"]
            # box/violin
            axes[0].violinplot([a, b], positions=[1, 2], showmedians=True,
                               showmeans=True)
            axes[0].set_xticks([1, 2])
            axes[0].set_xticklabels([label1, label2])
            axes[0].set_title("Distribution Comparison", fontsize=11)
            axes[0].grid(True)
            # overlay histograms
            bins = np.linspace(min(np.min(a),np.min(b)), max(np.max(a),np.max(b)), 20)
            axes[1].hist(a, bins=bins, alpha=0.6, color=palette[0], label=label1, edgecolor="none")
            axes[1].hist(b, bins=bins, alpha=0.6, color=palette[1], label=label2, edgecolor="none")
            axes[1].axvline(np.mean(a), color=palette[0], lw=2, ls="--")
            axes[1].axvline(np.mean(b), color=palette[1], lw=2, ls="--")
            axes[1].legend()
            axes[1].set_title("Histogram Overlay", fontsize=11)
            axes[1].grid(True)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  PAIRED t-TEST
# ─────────────────────────────────────────────────────────────────────────────
elif module == "🔗 Paired t-Test":
    st.markdown("# 🔗 Paired t-Test")
    st.markdown("Tests whether the mean difference between two paired observations is zero (e.g., before/after).")

    col1, col2 = st.columns(2)
    with col1:
        raw_before = st.text_area("Before / Condition 1", "120,118,125,130,122,119,128,124,121,127")
    with col2:
        raw_after  = st.text_area("After / Condition 2",  "115,112,120,124,118,114,122,119,116,121")

    if st.button("▶ Run Paired t-Test"):
        try:
            before = np.array([float(x.strip()) for x in raw_before.replace("\n",",").split(",") if x.strip()])
            after  = np.array([float(x.strip()) for x in raw_after.replace("\n",",").split(",")  if x.strip()])
            if len(before) != len(after):
                st.error("Both series must have equal length.")
            else:
                diff   = after - before
                t_stat, p_val = ttest_rel(before, after)
                badge, cls = decision(p_val, alpha)
                n = len(diff)
                se = np.std(diff, ddof=1) / np.sqrt(n)
                ci = stats.t.interval(1-alpha, n-1, loc=np.mean(diff), scale=se)

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Mean Diff", fmt(np.mean(diff),3))
                c2.metric("SD Diff",   fmt(np.std(diff,ddof=1),3))
                c3.metric("t-stat",    fmt(t_stat))
                c4.metric("p-value",   fmt(p_val))

                st.markdown(f"""
                <div class="result-box {cls}">
                t = {fmt(t_stat)}&nbsp;&nbsp;|&nbsp;&nbsp;p = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;
                df = {n-1}&nbsp;&nbsp;|&nbsp;&nbsp;{badge}<br>
                {(1-alpha)*100:.0f}% CI of difference: [{fmt(ci[0],3)}, {fmt(ci[1],3)}]
                </div>""", unsafe_allow_html=True)

                plot_defaults()
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                idx = np.arange(n)
                axes[0].plot(idx, before, 'o-', color="#58a6ff", label="Before", lw=1.5)
                axes[0].plot(idx, after,  's-', color="#f85149", label="After",  lw=1.5)
                for i in range(n):
                    axes[0].plot([i,i],[before[i],after[i]], color="#8b949e", lw=0.8, ls=":")
                axes[0].legend(); axes[0].grid(True)
                axes[0].set_title("Paired Observations", fontsize=11)

                axes[1].hist(diff, bins="auto", color="#3fb950", edgecolor="#0d1117", alpha=0.85)
                axes[1].axvline(0, color="#f85149", lw=2, ls="--", label="No difference")
                axes[1].axvline(np.mean(diff), color="#d29922", lw=2, label=f"Mean Δ={np.mean(diff):.2f}")
                axes[1].legend(); axes[1].grid(True)
                axes[1].set_title("Distribution of Differences", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  ONE-WAY ANOVA
# ─────────────────────────────────────────────────────────────────────────────
elif module == "📊 One-Way ANOVA":
    st.markdown("# 📊 One-Way ANOVA")
    st.markdown("Tests whether 3+ independent group means are equal. Followed by Tukey HSD post-hoc comparison.")

    n_groups = st.slider("Number of Groups", 2, 6, 3)
    groups, labels = [], []
    cols = st.columns(min(n_groups, 3))
    defaults = [
        "85,88,92,79,84,90,86,91",
        "72,75,68,71,74,70,73,69",
        "95,98,92,96,99,94,97,93",
        "80,83,78,82,85,79,81,84",
        "65,68,62,67,70,64,66,69",
        "88,91,85,89,93,87,90,92",
    ]
    for i in range(n_groups):
        with cols[i % 3]:
            labels.append(st.text_input(f"Group {i+1} name", f"Group {chr(65+i)}"))
            groups.append(st.text_area(f"Group {i+1} data", defaults[i]))

    if st.button("▶ Run One-Way ANOVA"):
        try:
            arrs = []
            for g in groups:
                arrs.append(np.array([float(x.strip()) for x in g.replace("\n",",").split(",") if x.strip()]))

            f_stat, p_val = f_oneway(*arrs)
            badge, cls = decision(p_val, alpha)

            # Descriptive table
            rows = []
            for lbl, arr in zip(labels, arrs):
                rows.append({"Group": lbl, "n": len(arr),
                             "Mean": round(np.mean(arr),3),
                             "Std Dev": round(np.std(arr,ddof=1),3),
                             "Min": round(min(arr),2), "Max": round(max(arr),2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

            st.markdown(f"""
            <div class="result-box {cls}">
            F-statistic = {fmt(f_stat)}&nbsp;&nbsp;|&nbsp;&nbsp;
            p-value = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;{badge}
            </div>""", unsafe_allow_html=True)

            # Eta squared
            all_data = np.concatenate(arrs)
            grand_mean = np.mean(all_data)
            ss_between = sum(len(a)*(np.mean(a)-grand_mean)**2 for a in arrs)
            ss_total   = sum((x - grand_mean)**2 for x in all_data)
            eta2 = ss_between / ss_total
            st.markdown(f"**η² (eta squared) = {fmt(eta2,4)}** — effect size: "
                        f"{'small' if eta2<0.06 else 'medium' if eta2<0.14 else 'large'}")

            if p_val < alpha:
                st.success("📌 Significant group differences found. Running Tukey HSD post-hoc test...")
                # Manual Tukey-like pairwise
                pairs = list(itertools.combinations(range(len(arrs)), 2))
                results_ph = []
                for i, j in pairs:
                    t_, p_ = ttest_ind(arrs[i], arrs[j])
                    p_adj  = min(p_ * len(pairs), 1.0)  # Bonferroni
                    results_ph.append({
                        "Comparison": f"{labels[i]} vs {labels[j]}",
                        "Mean Diff":  round(np.mean(arrs[i])-np.mean(arrs[j]),3),
                        "p (adj)":    fmt(p_adj),
                        "Significant": "✅ Yes" if p_adj < alpha else "❌ No"
                    })
                st.dataframe(pd.DataFrame(results_ph), hide_index=True)

            plot_defaults()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            colors = ["#1f6feb","#f85149","#3fb950","#d29922","#bc8cff","#58a6ff"]
            axes[0].boxplot(arrs, labels=labels, patch_artist=True,
                            boxprops=dict(facecolor="#21262d", color="#58a6ff"),
                            medianprops=dict(color="#f85149", lw=2),
                            whiskerprops=dict(color="#8b949e"),
                            capprops=dict(color="#8b949e"),
                            flierprops=dict(color="#d29922", marker="o"))
            for i, arr in enumerate(arrs):
                axes[0].scatter([i+1]*len(arr), arr, color=colors[i%6], alpha=0.5, s=20, zorder=3)
            axes[0].set_title("Group Distributions", fontsize=11)
            axes[0].grid(True)

            means = [np.mean(a) for a in arrs]
            cis   = [stats.t.interval(1-alpha, len(a)-1, np.mean(a), np.std(a,ddof=1)/np.sqrt(len(a))) for a in arrs]
            errs  = [[m-c[0] for m,c in zip(means,cis)], [c[1]-m for m,c in zip(means,cis)]]
            axes[1].bar(labels, means, color=[colors[i%6] for i in range(len(labels))],
                        edgecolor="#30363d", alpha=0.85)
            axes[1].errorbar(labels, means, yerr=errs, fmt="none", color="#e6edf3", capsize=5, lw=2)
            axes[1].set_title(f"Group Means ± {(1-alpha)*100:.0f}% CI", fontsize=11)
            axes[1].grid(True, axis="y")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  TWO-WAY ANOVA (Factorial DOE)
# ─────────────────────────────────────────────────────────────────────────────
elif module == "📐 Two-Way ANOVA (Factorial DOE)":
    st.markdown("# 📐 Two-Way ANOVA — Factorial DOE")
    st.markdown("Analyse main effects of two factors **and** their interaction on a response variable.")

    st.markdown("#### Upload or use sample data")
    use_sample = st.checkbox("Use built-in sample dataset (Temperature × Pressure → Yield)", value=True)

    if use_sample:
        df = pd.DataFrame({
            "Temperature": ["Low"]*12 + ["High"]*12,
            "Pressure":    ["Low","Low","Low","Med","Med","Med","High","High","High","Low","Low","Low"] +
                           ["Med","Med","Med","High","High","High","Low","Low","Low","Med","Med","Med"],
            "Yield":       [72,74,71, 78,76,79, 82,84,81, 85,87,84, 91,89,92, 88,90,87, 76,78,74, 80,82,79]
        })
    else:
        uploaded = st.file_uploader("Upload CSV with columns: Factor A, Factor B, Response", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            cols_sel = st.columns(3)
            col_a    = cols_sel[0].selectbox("Factor A column", df.columns.tolist())
            col_b    = cols_sel[1].selectbox("Factor B column", df.columns.tolist(), index=1)
            col_resp = cols_sel[2].selectbox("Response column", df.columns.tolist(), index=2)
            df = df.rename(columns={col_a:"Temperature", col_b:"Pressure", col_resp:"Yield"})
        else:
            st.info("Please upload a CSV or use sample data.")
            st.stop()

    st.dataframe(df.head(10), hide_index=True)

    if st.button("▶ Run Two-Way ANOVA"):
        try:
            factor_a_levels = df["Temperature"].unique()
            factor_b_levels = df["Pressure"].unique()

            # Groups for each combination
            cell_means, cell_stds = {}, {}
            for a in factor_a_levels:
                for b in factor_b_levels:
                    key = (a, b)
                    cell_means[key] = df[(df["Temperature"]==a)&(df["Pressure"]==b)]["Yield"].mean()
                    cell_stds[key]  = df[(df["Temperature"]==a)&(df["Pressure"]==b)]["Yield"].std(ddof=1)

            grand_mean = df["Yield"].mean()

            # Main effect A
            a_means = {a: df[df["Temperature"]==a]["Yield"].mean() for a in factor_a_levels}
            b_means = {b: df[df["Pressure"]==b]["Yield"].mean()    for b in factor_b_levels}
            na = len(factor_a_levels); nb = len(factor_b_levels)
            n  = len(df)

            # SS calculations
            n_cell = df.groupby(["Temperature","Pressure"])["Yield"].count().min()
            ss_a = nb * n_cell * sum((v-grand_mean)**2 for v in a_means.values())
            ss_b = na * n_cell * sum((v-grand_mean)**2 for v in b_means.values())
            ss_ab = n_cell * sum(
                (cell_means[(a,b)] - a_means[a] - b_means[b] + grand_mean)**2
                for a in factor_a_levels for b in factor_b_levels
            )
            ss_within = df.groupby(["Temperature","Pressure"])["Yield"].apply(
                lambda x: ((x - x.mean())**2).sum()
            ).sum()
            ss_total = ((df["Yield"] - grand_mean)**2).sum()

            df_a  = na - 1
            df_b  = nb - 1
            df_ab = df_a * df_b
            df_w  = n - na * nb

            ms_a  = ss_a / df_a
            ms_b  = ss_b / df_b
            ms_ab = ss_ab / df_ab
            ms_w  = ss_within / df_w

            f_a  = ms_a  / ms_w
            f_b  = ms_b  / ms_w
            f_ab = ms_ab / ms_w
            p_a  = 1 - stats.f.cdf(f_a,  df_a,  df_w)
            p_b  = 1 - stats.f.cdf(f_b,  df_b,  df_w)
            p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_w)

            # ANOVA table
            anova_table = pd.DataFrame({
                "Source":  ["Temperature (A)", "Pressure (B)", "A × B Interaction", "Within (Error)", "Total"],
                "SS":      [fmt(ss_a,3), fmt(ss_b,3), fmt(ss_ab,3), fmt(ss_within,3), fmt(ss_total,3)],
                "df":      [df_a, df_b, df_ab, df_w, n-1],
                "MS":      [fmt(ms_a,3), fmt(ms_b,3), fmt(ms_ab,3), fmt(ms_w,3), "—"],
                "F":       [fmt(f_a,3),  fmt(f_b,3),  fmt(f_ab,3),  "—", "—"],
                "p-value": [fmt(p_a,4),  fmt(p_b,4),  fmt(p_ab,4),  "—", "—"],
                "Sig.":    ["*" if p_a<alpha else "ns",
                            "*" if p_b<alpha else "ns",
                            "*" if p_ab<alpha else "ns", "—", "—"]
            })
            st.markdown("### ANOVA Table")
            st.dataframe(anova_table, hide_index=True)

            # Interpretations
            for label, f, p, df1 in [("Temperature (A)", f_a, p_a, df_a),
                                      ("Pressure (B)",    f_b, p_b, df_b),
                                      ("A×B Interaction", f_ab, p_ab, df_ab)]:
                badge, cls = decision(p, alpha)
                st.markdown(f'<div class="result-box {cls}"><b>{label}</b>: F={fmt(f,3)}, p={fmt(p,4)} &nbsp; {badge}</div>',
                            unsafe_allow_html=True)

            if p_ab < alpha:
                st.warning("⚠️ Significant interaction detected — interpret main effects with caution. The effect of one factor depends on the level of the other.")

            # Plots
            plot_defaults()
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))

            colors = ["#58a6ff","#f85149","#3fb950","#d29922"]
            b_ord  = sorted(factor_b_levels)

            for i, a in enumerate(sorted(factor_a_levels)):
                ys = [cell_means[(a, b)] for b in b_ord]
                axes[0].plot(b_ord, ys, "o-", color=colors[i%4], lw=2.5, ms=8, label=f"Temp={a}")
            axes[0].set_title("Interaction Plot", fontsize=12)
            axes[0].set_xlabel("Pressure"); axes[0].set_ylabel("Mean Yield")
            axes[0].legend(); axes[0].grid(True)

            # Main effects bar chart
            all_effects = (
                {f"Temp={k}": v for k,v in a_means.items()} |
                {f"Press={k}": v for k,v in b_means.items()}
            )
            eff_keys = list(all_effects.keys())
            eff_vals = [all_effects[k] - grand_mean for k in eff_keys]
            bar_colors = ["#58a6ff"]*len(a_means) + ["#f85149"]*len(b_means)
            axes[1].barh(eff_keys, eff_vals, color=bar_colors, edgecolor="#30363d")
            axes[1].axvline(0, color="#8b949e", lw=1.5, ls="--")
            axes[1].set_title("Main Effect Deviations from Grand Mean", fontsize=12)
            axes[1].grid(True, axis="x")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error processing Two-Way ANOVA: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  CHI-SQUARE TEST
# ─────────────────────────────────────────────────────────────────────────────
elif module == "🧩 Chi-Square Test":
    st.markdown("# 🧩 Chi-Square Test")
    st.markdown("Tests independence between two categorical variables, or goodness-of-fit.")

    test_type = st.radio("Test type", ["Independence (contingency table)", "Goodness-of-Fit"])

    if test_type == "Independence (contingency table)":
        st.markdown("Enter contingency table as CSV (rows = groups, columns = categories).")
        raw_ct = st.text_area("Contingency Table",
                              "45, 30, 25\n20, 50, 30\n35, 20, 45")
        col_labels = st.text_input("Column names (comma-separated)", "Cat A, Cat B, Cat C")
        row_labels  = st.text_input("Row names (comma-separated)",   "Group 1, Group 2, Group 3")

        if st.button("▶ Run Chi-Square Test"):
            try:
                rows_data = [[float(v.strip()) for v in r.split(",")] for r in raw_ct.strip().split("\n")]
                ct = np.array(rows_data)
                chi2, p_val, dof, expected = chi2_contingency(ct)
                badge, cls = decision(p_val, alpha)
                cramers_v = np.sqrt(chi2 / (ct.sum() * (min(ct.shape)-1)))

                st.markdown(f"""
                <div class="result-box {cls}">
                χ² = {fmt(chi2)}&nbsp;&nbsp;|&nbsp;&nbsp;
                p = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;
                df = {dof}&nbsp;&nbsp;|&nbsp;&nbsp;
                Cramér's V = {fmt(cramers_v,4)}&nbsp;&nbsp;|&nbsp;&nbsp;{badge}
                </div>""", unsafe_allow_html=True)

                rl = [r.strip() for r in row_labels.split(",")]
                cl = [c.strip() for c in col_labels.split(",")]
                observed_df = pd.DataFrame(ct, index=rl[:len(ct)], columns=cl[:ct.shape[1]])
                expected_df = pd.DataFrame(expected, index=rl[:len(ct)], columns=cl[:ct.shape[1]])

                col1, col2 = st.columns(2)
                col1.markdown("**Observed**"); col1.dataframe(observed_df)
                col2.markdown("**Expected**"); col2.dataframe(expected_df.round(2))

                plot_defaults()
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                im1 = axes[0].imshow(ct, cmap="Blues", aspect="auto")
                axes[0].set_xticks(range(len(cl[:ct.shape[1]]))); axes[0].set_xticklabels(cl[:ct.shape[1]])
                axes[0].set_yticks(range(len(rl[:len(ct)]))); axes[0].set_yticklabels(rl[:len(ct)])
                axes[0].set_title("Observed Counts"); plt.colorbar(im1, ax=axes[0])

                resid = (ct - expected) / np.sqrt(expected)
                im2 = axes[1].imshow(resid, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)
                axes[1].set_xticks(range(len(cl[:ct.shape[1]]))); axes[1].set_xticklabels(cl[:ct.shape[1]])
                axes[1].set_yticks(range(len(rl[:len(ct)]))); axes[1].set_yticklabels(rl[:len(ct)])
                axes[1].set_title("Standardised Residuals"); plt.colorbar(im2, ax=axes[1])
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown("Enter observed and expected frequencies.")
        raw_obs = st.text_area("Observed frequencies", "30, 25, 45, 50, 30, 20")
        raw_exp = st.text_area("Expected frequencies (blank = equal)", "")
        if st.button("▶ Run Goodness-of-Fit"):
            try:
                obs = np.array([float(x.strip()) for x in raw_obs.split(",") if x.strip()])
                if raw_exp.strip():
                    exp = np.array([float(x.strip()) for x in raw_exp.split(",") if x.strip()])
                else:
                    exp = np.full(len(obs), obs.sum()/len(obs))
                chi2_stat, p_val = stats.chisquare(obs, exp)
                badge, cls = decision(p_val, alpha)
                st.markdown(f"""
                <div class="result-box {cls}">
                χ² = {fmt(chi2_stat)}&nbsp;&nbsp;|&nbsp;&nbsp;
                p = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;
                df = {len(obs)-1}&nbsp;&nbsp;|&nbsp;&nbsp;{badge}
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  NON-PARAMETRIC TESTS
# ─────────────────────────────────────────────────────────────────────────────
elif module == "🏔️ Non-Parametric Tests":
    st.markdown("# 🏔️ Non-Parametric Tests")
    st.markdown("Use when normality assumptions are violated.")

    np_test = st.selectbox("Select Test", [
        "Mann-Whitney U (2 independent groups)",
        "Wilcoxon Signed-Rank (paired/1-sample)",
        "Kruskal-Wallis (3+ independent groups)",
        "Shapiro-Wilk Normality Test",
    ])

    if "Mann-Whitney" in np_test:
        col1, col2 = st.columns(2)
        with col1: raw1 = st.text_area("Group 1", "15,18,12,20,17,14,19,13,16,21")
        with col2: raw2 = st.text_area("Group 2", "22,25,19,27,24,21,26,20,23,28")
        if st.button("▶ Run Mann-Whitney U"):
            try:
                a = np.array([float(x.strip()) for x in raw1.split(",") if x.strip()])
                b = np.array([float(x.strip()) for x in raw2.split(",") if x.strip()])
                u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
                badge, cls = decision(p_val, alpha)
                r = 1 - (2*u_stat)/(len(a)*len(b))
                st.markdown(f"""
                <div class="result-box {cls}">
                U = {fmt(u_stat,1)}&nbsp;&nbsp;|&nbsp;&nbsp;
                p = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;
                r (effect size) = {fmt(r,4)}&nbsp;&nbsp;|&nbsp;&nbsp;{badge}
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    elif "Wilcoxon" in np_test:
        raw_d = st.text_area("Differences (or single sample)", "3,5,-2,4,6,1,-1,2,5,3")
        if st.button("▶ Run Wilcoxon Signed-Rank"):
            try:
                d = np.array([float(x.strip()) for x in raw_d.split(",") if x.strip()])
                w_stat, p_val = wilcoxon(d)
                badge, cls = decision(p_val, alpha)
                st.markdown(f"""
                <div class="result-box {cls}">
                W = {fmt(w_stat,1)}&nbsp;&nbsp;|&nbsp;&nbsp;
                p = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;{badge}
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    elif "Kruskal" in np_test:
        n_grp = st.slider("Number of groups", 3, 5, 3)
        grp_data, grp_lbl = [], []
        dcols = st.columns(n_grp)
        defs  = ["12,15,11,14,13","18,20,17,22,19","25,28,24,27,26","10,12,9,11,13","30,33,29,32,31"]
        for i in range(n_grp):
            with dcols[i]:
                grp_lbl.append(st.text_input(f"Label {i+1}", f"G{i+1}"))
                grp_data.append(st.text_area(f"Data {i+1}", defs[i]))
        if st.button("▶ Run Kruskal-Wallis"):
            try:
                arrs = [np.array([float(x.strip()) for x in g.split(",") if x.strip()]) for g in grp_data]
                h_stat, p_val = kruskal(*arrs)
                badge, cls = decision(p_val, alpha)
                st.markdown(f"""
                <div class="result-box {cls}">
                H = {fmt(h_stat)}&nbsp;&nbsp;|&nbsp;&nbsp;
                p = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;{badge}
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    elif "Shapiro" in np_test:
        raw_sw = st.text_area("Data", "22,25,23,27,24,26,21,28,23,25,24,26,22,27,25")
        if st.button("▶ Run Shapiro-Wilk"):
            try:
                d = np.array([float(x.strip()) for x in raw_sw.split(",") if x.strip()])
                stat, p_val = shapiro(d)
                badge, cls = decision(p_val, alpha)
                st.markdown(f"""
                <div class="result-box {cls}">
                W = {fmt(stat)}&nbsp;&nbsp;|&nbsp;&nbsp;
                p = {fmt(p_val)}&nbsp;&nbsp;|&nbsp;&nbsp;{badge}<br>
                {"Data is likely NOT normally distributed." if p_val < alpha else "No evidence against normality."}
                </div>""", unsafe_allow_html=True)

                plot_defaults()
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                axes[0].hist(d, bins="auto", color="#58a6ff", edgecolor="#0d1117", alpha=0.85)
                axes[0].set_title("Histogram"); axes[0].grid(True)
                osm, osr = stats.probplot(d, dist="norm")
                axes[1].plot(osm[0], osm[1], "o", color="#58a6ff", alpha=0.7)
                x_line = np.linspace(min(osm[0]), max(osm[0]), 100)
                axes[1].plot(x_line, osr[1]*x_line + osr[0], color="#f85149", lw=2)
                axes[1].set_title("Q-Q Plot"); axes[1].grid(True)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  FULL FACTORIAL DOE DESIGNER
# ─────────────────────────────────────────────────────────────────────────────
elif module == "🔢 Full Factorial DOE Designer":
    st.markdown("# 🔢 Full Factorial DOE Designer")
    st.markdown("Design a full factorial experiment and generate a run matrix.")

    n_factors = st.slider("Number of Factors (k)", 2, 5, 3)
    factor_names, low_levels, high_levels = [], [], []

    defaults_f = ["Temperature", "Pressure", "Feed Rate", "Catalyst", "pH"]
    defaults_l = ["Low (60°C)", "Low (1 bar)", "Low (2 g/min)", "0.5%", "5.0"]
    defaults_h = ["High (90°C)", "High (3 bar)", "High (8 g/min)", "2.0%", "8.0"]

    for i in range(n_factors):
        col1, col2, col3 = st.columns(3)
        with col1: factor_names.append(st.text_input(f"Factor {i+1} name", defaults_f[i]))
        with col2: low_levels.append( st.text_input(f"Low level (−1)",  defaults_l[i]))
        with col3: high_levels.append(st.text_input(f"High level (+1)", defaults_h[i]))

    n_replicates = st.slider("Replicates per run", 1, 3, 1)
    randomize    = st.checkbox("Randomize run order", value=True)

    if st.button("▶ Generate DOE Matrix"):
        levels = list(itertools.product([-1, 1], repeat=n_factors))
        run_data = []
        for lev in levels:
            row = {"Run": None}
            for i, fname in enumerate(factor_names):
                coded = lev[i]
                actual = low_levels[i] if coded == -1 else high_levels[i]
                row[f"{fname} (coded)"] = coded
                row[f"{fname} (actual)"] = actual
            run_data.append(row)

        # Replicate
        run_data = run_data * n_replicates
        if randomize:
            import random; random.shuffle(run_data)

        for i, r in enumerate(run_data, 1):
            r["Run"] = i

        doe_df = pd.DataFrame(run_data)
        doe_df["Response (fill in)"] = ""

        st.markdown(f"### Design Matrix — 2^{n_factors} Full Factorial = **{len(levels)} unique runs** × {n_replicates} replicates = **{len(run_data)} total runs**")
        st.dataframe(doe_df, hide_index=True)

        csv_buf = io.StringIO()
        doe_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download DOE Matrix (CSV)", csv_buf.getvalue(),
                           "doe_matrix.csv", "text/csv")

        # Coded matrix heat map
        plot_defaults()
        coded_cols = [c for c in doe_df.columns if "(coded)" in c]
        coded_mat  = doe_df[coded_cols].astype(float).values
        fig, ax = plt.subplots(figsize=(max(6, n_factors*1.5), max(4, len(run_data)*0.25 + 2)))
        im = ax.imshow(coded_mat, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(coded_cols)))
        ax.set_xticklabels([c.replace(" (coded)","") for c in coded_cols], rotation=30, ha="right")
        ax.set_yticks(range(len(run_data)))
        ax.set_yticklabels([f"Run {r['Run']}" for r in run_data], fontsize=8)
        ax.set_title("Coded Design Matrix (green=+1, red=−1)", fontsize=11)
        plt.colorbar(im, ax=ax, ticks=[-1, 1], label="Level")
        plt.tight_layout()
        st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────
elif module == "📈 Diagnostic Plots":
    st.markdown("# 📈 Diagnostic Plots")
    st.markdown("Check assumptions before running parametric tests.")

    raw_diag = st.text_area("Enter data (one column, comma or newline separated)",
                            "45,48,52,41,55,49,51,47,53,50,44,56,48,52,46,54,43,57,50,49")

    if st.button("▶ Generate Diagnostic Suite"):
        try:
            d = np.array([float(x.strip()) for x in raw_diag.replace("\n",",").split(",") if x.strip()])
            sw_stat, sw_p = shapiro(d)

            plot_defaults()
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            fig.suptitle("Diagnostic Plot Suite", fontsize=14, fontweight="bold")
            colors = ["#1f6feb", "#58a6ff"]

            # 1. Histogram + KDE
            axes[0,0].hist(d, bins="auto", density=True, color=colors[0], alpha=0.7, edgecolor="#0d1117")
            kde_x = np.linspace(d.min()-2*d.std(), d.max()+2*d.std(), 200)
            kde   = stats.gaussian_kde(d)
            axes[0,0].plot(kde_x, kde(kde_x), color=colors[1], lw=2.5, label="KDE")
            axes[0,0].plot(kde_x, stats.norm.pdf(kde_x, d.mean(), d.std()), color="#f85149",
                          lw=2, ls="--", label="Normal")
            axes[0,0].legend(); axes[0,0].set_title("Histogram + KDE vs Normal"); axes[0,0].grid(True)

            # 2. Q-Q plot
            osm, osr = stats.probplot(d, dist="norm")
            axes[0,1].plot(osm[0], osm[1], "o", color=colors[0], alpha=0.7)
            xl = np.linspace(min(osm[0]), max(osm[0]), 100)
            axes[0,1].plot(xl, osr[1]*xl + osr[0], color="#f85149", lw=2)
            axes[0,1].set_title(f"Q-Q Plot (Shapiro-Wilk p={fw_p:.4f})" if (fw_p:=sw_p) else "Q-Q Plot")
            axes[0,1].set_xlabel("Theoretical Quantiles"); axes[0,1].set_ylabel("Sample Quantiles")
            axes[0,1].grid(True)

            # 3. Box plot with outliers
            bp = axes[1,0].boxplot(d, vert=True, patch_artist=True,
                                   boxprops=dict(facecolor="#21262d", color="#58a6ff"),
                                   medianprops=dict(color="#f85149", lw=2.5),
                                   whiskerprops=dict(color="#8b949e"),
                                   flierprops=dict(marker="D", color="#d29922", ms=7))
            axes[1,0].set_title("Box Plot"); axes[1,0].grid(True, axis="y")

            # 4. Run chart
            axes[1,1].plot(range(1, len(d)+1), d, "o-", color=colors[0], lw=1.5, ms=5)
            axes[1,1].axhline(d.mean(), color="#f85149", lw=2, ls="--", label=f"Mean={d.mean():.2f}")
            axes[1,1].axhline(d.mean()+2*d.std(), color="#d29922", lw=1.5, ls=":", label="+2σ")
            axes[1,1].axhline(d.mean()-2*d.std(), color="#d29922", lw=1.5, ls=":", label="-2σ")
            axes[1,1].legend(fontsize=8); axes[1,1].set_title("Run Chart")
            axes[1,1].set_xlabel("Observation Order"); axes[1,1].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            # Summary
            st.markdown("### Descriptive Statistics")
            summary = pd.DataFrame({
                "Statistic": ["n","Mean","Median","Std Dev","Variance","Min","Max","Skewness","Kurtosis"],
                "Value": [
                    len(d), round(np.mean(d),4), round(np.median(d),4),
                    round(np.std(d,ddof=1),4), round(np.var(d,ddof=1),4),
                    round(d.min(),4), round(d.max(),4),
                    round(stats.skew(d),4), round(stats.kurtosis(d),4)
                ]
            })
            st.dataframe(summary, hide_index=True)

            badge_sw, cls_sw = decision(sw_p, alpha)
            st.markdown(f"""
            <div class="result-box {cls_sw}">
            <b>Shapiro-Wilk Normality Test:</b> W={fmt(sw_stat)}, p={fmt(sw_p)}&nbsp;&nbsp;{badge_sw}<br>
            {'⚠️ Non-normal distribution — consider non-parametric tests.' if sw_p < alpha else '✅ No evidence against normality — parametric tests are appropriate.'}
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
