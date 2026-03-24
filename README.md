# 🔬 DOE & Hypothesis Testing Agent

A fully open-source, no-login-required Streamlit app for **Design of Experiments** and **Inferential Statistics**.

---

## Features

| Module | What it does |
|--------|-------------|
| One-Sample t-Test | Test a sample mean against a known value |
| Two-Sample t-Test | Compare two independent group means |
| Paired t-Test | Before/after or matched-pair comparisons |
| One-Way ANOVA | Compare 3+ group means + Tukey post-hoc |
| **Two-Way ANOVA (Factorial DOE)** | Main effects + interaction analysis |
| Chi-Square Test | Categorical independence & goodness-of-fit |
| Non-Parametric Suite | Mann-Whitney, Wilcoxon, Kruskal-Wallis, Shapiro-Wilk |
| **Full Factorial DOE Designer** | Generate 2^k run matrices + download CSV |
| Diagnostic Plots | Histogram, Q-Q, Boxplot, Run chart, Shapiro-Wilk |

---

## 🚀 Run Locally (3 commands)

```bash
# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/doe-agent.git
cd doe-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/doe-agent.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub (free)
   - Click **New app**
   - Select your repo → `main` branch → `app.py`
   - Click **Deploy** — done! ✅

Your app gets a public URL like `https://your-app.streamlit.app` — **no payment required**.

---

## 📁 Project Structure

```
doe-agent/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## Statistical Methods Used

- **t-tests**: `scipy.stats.ttest_1samp`, `ttest_ind`, `ttest_rel`
- **ANOVA**: `scipy.stats.f_oneway` + manual SS/MS calculations for two-way
- **Chi-Square**: `scipy.stats.chi2_contingency`, `chisquare`
- **Non-parametric**: `mannwhitneyu`, `wilcoxon`, `kruskal`
- **Normality**: `scipy.stats.shapiro`
- **DOE Design**: `itertools.product` for full factorial generation
- **Effect sizes**: Cohen's d, eta squared (η²), Cramér's V, rank-biserial r

---

## License

MIT — free to use, modify, and share.
