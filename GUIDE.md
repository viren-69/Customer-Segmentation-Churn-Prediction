# Project Guide: Customer Segmentation + Churn Prediction

This guide is for **you**, to learn the project well enough to explain any part of it in an interview. It assumes you've read Project 1's guide (the Olist data) but no prior machine-learning knowledge.

1. [The project in 60 seconds](#1-the-project-in-60-seconds)
2. [The two business problems](#2-the-two-business-problems)
3. [The data and its traps](#3-the-data-and-its-traps)
4. [Machine-learning ideas you need](#4-machine-learning-ideas-you-need)
5. [Decision log: the thought process](#5-decision-log-the-thought-process)
6. [Code walkthrough, file by file](#6-code-walkthrough-file-by-file)
7. [Notebook walkthrough](#7-notebook-walkthrough)
8. [Results](#8-results)
9. [Limitations and what I would do next](#9-limitations-and-what-i-would-do-next)
10. [Cheat sheet](#10-cheat-sheet)
11. [Interview questions with answers](#11-interview-questions-with-answers)
12. [How to run](#12-how-to-run)

**Suggested study order:** sections 1–5 first (the *why*), then section 4 again slowly. Then open each notebook next to section 7 and each `src/` file next to section 6. Finish by practising section 11 out loud.

---

## 1. The project in 60 seconds

> "This project covers the full customer-analytics loop in two parts. First, I segmented about 95,000 real e-commerce customers from my Olist database using RFM (recency, frequency and monetary value) and K-Means. That found five segments, including 'lapsed big spenders': 19% of customers but 36% of revenue, the obvious win-back target.
>
> Second, I built a churn model on IBM's Telco data. I first removed leakage columns, like a churn reason that only exists for churners. Logistic regression and random forest were tied at about 0.86 cross-validated ROC-AUC, so I chose the simpler, explainable logistic regression. Instead of the default 0.5 cut-off, I chose who to contact by expected campaign profit. That threshold, 0.21, catches 86% of churners and earns about 20% more than the default on unseen customers. I tested 27 different cost assumptions to make sure the conclusion holds.
>
> SHAP showed tenure, contract type and fiber internet drive churn. My main recommendation: move month-to-month customers, especially on fiber, to one-year contracts, validated with an A/B test first."

Memorise: **95K customers / 5 segments / 19% → 36%; 7,043 subscribers / 26.5% churn; ROC-AUC 0.85; threshold 0.21 → 86% recall; $52.6K vs $44.0K vs $27.5K; 42.7% vs 2.8% (contracts).**

---

## 2. The two business problems

| | Part A: Segmentation | Part B: Churn prediction |
|---|---|---|
| Question | Which *groups* of customers do we have? | *Which* customers will leave, and *why*? |
| Data | Olist (e-commerce, 95K customers) | IBM Telco (subscriptions, 7,043 customers) |
| ML type | **Unsupervised**: no right answer to learn from | **Supervised**: we know who churned |
| Method | RFM + K-Means | Logistic Regression vs Random Forest + SHAP |
| Output | 5 named segments, each with an action | A risk score per customer, the drivers, and who to contact |

**Why two datasets?** Olist has no churn label. It isn't a subscription business, and 97% of customers buy only once, so "churn" isn't well defined there. Telco has a clean, labelled churn outcome. Each dataset is used for the question it can answer. Say this openly if asked; it shows you thought about fit.

---

## 3. The data and its traps

### Telco (7,043 rows × 33 columns)
Each row is a customer: demographics (gender, senior, partner, dependents), services (phone, internet type, add-ons like tech support), account (contract, billing, payment method, tenure, charges) and the outcome (churned in the last quarter: 26.5% did).

| Trap | Why it matters | What we did |
|---|---|---|
| **Leakage: `Churn Reason`** | Only filled in for customers who already left. A model would learn "has reason → churns" | Dropped |
| **Leakage: `Churn Score`** | IBM's own model output: churners score 65–100, others 5–80 | Dropped |
| **`CLTV`** | IBM-derived customer value; may be computed using churn | Kept for reference only, never a feature |
| **`Churn Label` + `Churn Value`** | Two copies of the target | Kept one as the target (`churn`) |
| Constant columns (Count, Country, State) | Same for every row, so they carry no information | Dropped |
| IDs and location (ID, city, zip, lat/long) | The model could memorise individuals instead of learning patterns | Dropped |
| 11 blank `Total Charges` | All are brand-new customers (tenure 0), not yet billed | Set to 0, not "missing" |
| "No internet service" values in add-on columns | Same meaning as "No" for that add-on; `internet_service` already records it | Merged into "No" |
| **Class imbalance**: 26.5% churn | "Nobody churns" is 73.5% accurate but useless | Judge with ROC-AUC, PR-AUC, recall, profit; never accuracy alone |

> **Data leakage** is when a feature contains information that wouldn't be available at prediction time. It's the #1 reason models look great in testing and fail in production. Always ask: "Would I know this value *before* the customer churns?"

### Olist (from Project 1)
- One row per **person** (`customer_unique_id`, not the per-order `customer_id`).
- Only orders before **1 Sep 2018** (the last complete month) and not canceled/unavailable, with the same revenue definition as Project 1 (item prices, no freight).
- **97% of customers bought once**, so frequency barely varies. This shapes the segmentation results (section 7.1).
- Spend is very skewed (median R$ 90, max R$ 13,440).

---

## 4. Machine-learning ideas you need

**Supervised vs unsupervised.** Supervised learning learns from examples with known answers (who churned). Unsupervised learning finds structure without answers (groups of similar customers).

**Features and target.** Features (X) are the inputs (contract, tenure…); the target (y) is what we predict (churn 0/1).

**Train / test split.** We hide 20% of customers (the **test set**) and never look at them until the very end. Their score tells us how the model will do on *new* customers. `stratify=y` keeps 26.5% churners in both parts.

**Overfitting.** A model that memorises the training data scores great on it but poorly on new data. Evidence against it here: cross-validation (0.859) and test (0.849) scores are close.

**Cross-validation (5-fold).** The training set is split into 5 parts. Train on 4, validate on the 5th, rotate, and average. This gives a more reliable score than one split, plus a **standard deviation** that shows how much the score varies just by chance. We use it to choose model settings and to compare models, without touching the test set.

**Hyperparameters.** Settings chosen *before* training (e.g. `C` for logistic regression, `max_depth` for a forest). `GridSearchCV` tries every combination with cross-validation and keeps the best.

**Preprocessing.**
- *One-hot encoding:* turns a category like contract = {month-to-month, one year, two year} into 0/1 columns, because models need numbers. `drop="if_binary"` keeps one column for yes/no features.
- *Standard scaling:* rescales numbers to mean 0 and standard deviation 1, so tenure (0–72) and charges (18–119) are comparable. Logistic regression needs this; trees don't, but it doesn't hurt them.
- *Pipeline:* preprocessing and model joined into one object. Inside cross-validation, the scaler is fitted on each training fold only. Fitting it on all the data first would leak information from the validation fold.

**Logistic regression.** Predicts the probability of churn as a weighted sum of features passed through an S-shaped curve. Each weight (coefficient) has a meaning: **exp(coefficient) = odds ratio**, how much that feature multiplies the odds of churning. `C` controls regularisation (smaller C = coefficients pushed towards 0 = simpler model).

**Random forest.** Hundreds of decision trees, each trained on a random sample of customers and features. Their votes are averaged. It captures non-linear effects and interactions automatically. `max_depth` and `min_samples_leaf` stop trees memorising individuals.

**Metrics for imbalanced yes/no problems:**
| Metric | Question it answers | Here |
|---|---|---|
| Accuracy | % of all predictions right | Misleading: "nobody churns" gets 73.5% |
| **ROC-AUC** | If I pick a random churner and a random non-churner, how often does the model rank the churner higher? 0.5 = coin flip, 1 = perfect | 0.85 |
| **PR-AUC** | Precision vs recall across thresholds; baseline = churn rate (0.27) | 0.65 |
| **Precision** | Of the customers we flag, how many really churn? | 49% at threshold 0.21 |
| **Recall** | Of the customers who churn, how many do we flag? | 86% at threshold 0.21 |
| Confusion matrix | Counts of true/false positives/negatives | in notebook 3 |

**Threshold.** Models output a probability; a business needs yes/no. The cut-off trades precision against recall. 0.5 is a default, not a business decision.

**SHAP.** A method from game theory that splits each prediction into one contribution per feature (how much each pushed this customer's risk up or down from the average). Averaged over customers, it ranks features *and* shows the direction of their effect. It works for any model, which lets us compare logistic regression and random forest on equal terms.

**K-Means.** Picks k centre points and assigns each customer to the nearest one. Each centre moves to the average of its customers, and this repeats until stable. It needs features on similar scales (hence log + standardise) and a chosen k.
- *Inertia:* total distance of customers to their centre; always decreases with k → look for the **elbow**.
- *Silhouette:* −1 to 1; how much closer each customer is to their own cluster than to the next nearest one.

---

## 5. Decision log: the thought process

1. **Remove leakage before anything else.** Checked every column with "would I know this before they churn?" `Churn Reason` and `Churn Score` fail; `CLTV` is suspicious, so it's kept only for reporting.
2. **Lock the test set first.** Every choice (model, settings, threshold) uses the training set only. The test set is used once, to report.
3. **Include a dumb baseline.** "Nobody churns" proves accuracy is useless here and gives a floor every model must beat.
4. **Compare a simple and a complex model.** Logistic regression (interpretable) vs random forest (flexible). Their CV scores differ by 0.002 with a fold-to-fold std of about 0.01: a tie.
5. **When tied, choose the simpler model.** Rule: the complex model must win by more than one standard deviation. Logistic regression gives odds ratios, calibrated probabilities (needed for profit maths), and is easier to explain, monitor and retrain.
6. **Don't rebalance classes; tune the threshold instead.** Methods like `class_weight="balanced"` or oversampling distort the probabilities. Keeping honest probabilities and choosing the threshold by business value achieves the same goal (catch more churners) transparently.
7. **Choose the threshold by profit, on out-of-fold predictions.** Each training customer is scored by a model that never saw them (`cross_val_predict`). This avoids tuning on the test set.
8. **Stress-test the assumptions.** The profit numbers are illustrative, so the analysis is repeated for 27 combinations of cost, save rate and customer value (decision 11 below).
9. **Explain with two methods and two models.** Odds ratios for direct meaning; SHAP to show direction and to check that random forest agrees. Agreement means the drivers come from the data, not the algorithm.
10. **Separate association from causation.** The "what-if one-year contract" simulation is a model estimate, so the recommendation says "A/B test first".
11. **Segment count: metrics narrow, business decides.** k = 4 has the highest silhouette, but k = 5 gives clean, actionable segments with a score almost as good.
12. **Logic in `src/`, story in notebooks.** Notebooks read like a report; the reusable, tested code lives in modules, so a notebook cell is usually one function call.

---

## 6. Code walkthrough, file by file

### 6.1 `src/config.py`
Paths (`RAW_DIR`, `FIGURES_DIR`…), `DATABASE_URL` from `.env`, and `RANDOM_STATE = 42`. One seed everywhere makes splits, K-Means and random forests give identical numbers on every run (**reproducibility**). Also `RFM_SNAPSHOT_DATE` and the chart colours.

### 6.2 `src/telco_data.py`
- Column lists: `LEAKAGE_COLUMNS`, `CONSTANT_COLUMNS`, `ID_COLUMNS`, and the 19 `FEATURES` split into `NUMERIC_FEATURES` (3) and `CATEGORICAL_FEATURES` (16). Naming them once means every other file uses the same lists.
- `clean(raw)`:
  1. Saves `customer_id` and `cltv` aside as *reference* columns.
  2. Drops leakage, constant and ID columns; renames columns to snake_case; `churn_value` → `churn`.
  3. `pd.to_numeric(..., errors="coerce")` turns the blank text in `Total Charges` into NaN. Rows with tenure 0 get 0.0.
  4. Replaces "No internet service" / "No phone service" with "No" in the 7 add-on columns.
  5. **Fails loudly** if any feature is still missing. A test proves this: a blank total for a non-new customer raises an error instead of being silently filled.

### 6.3 `src/churn_model.py`
- `split()`: `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
- `build_preprocessor()`: a `ColumnTransformer` that applies `StandardScaler` to numeric columns and `OneHotEncoder(drop="if_binary", handle_unknown="ignore")` to categorical ones. `handle_unknown="ignore"` means an unseen category in the future becomes all zeros instead of crashing.
- `build_pipelines()`: returns `{name: (pipeline, grid)}` for the baseline (`DummyClassifier(strategy="prior")`), logistic regression (grid over `C`) and random forest (grid over `max_depth`, `min_samples_leaf`). `model__C` means "the `C` parameter of the step named `model`".
- `tune()`: `GridSearchCV(..., scoring="roc_auc", cv=CV)` with `StratifiedKFold(5, shuffle=True)`. It returns a `TunedModel` dataclass with the refitted best pipeline, its settings, and CV mean ± std.
- `evaluate()`: all the metrics at a given threshold. `predict_proba(...)[:, 1]` is the probability of class 1 (churn).
- `out_of_fold_proba()`: `cross_val_predict(method="predict_proba")`, a prediction for every training customer from a model that didn't see them.
- `campaign_profit(y, proba, monthly, threshold, offer_cost, save_rate, months_of_value)`:
  `profit = Σ(contacted churners × save_rate × months × monthly charge) − offer_cost × number contacted`.
  Contacting a loyal customer only costs money. The assumptions are parameters with defaults ($50, 30%, 12 months).
- `THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)`: the `np.round` fixes a **floating-point bug a unit test caught**. `np.arange` produced 0.30000000000000004, so the threshold evaluated wasn't exactly the one reported (0.3).
- `profit_curve()` / `best_threshold()`: profit at every candidate threshold, then the best one.
- `sensitivity()`: repeats the threshold choice for every combination of assumptions, then compares contact-everyone, default 0.5 and tuned threshold on the test set.
- `logistic_odds_ratios()`: `exp(coef_)` per feature name (`get_feature_names_out()` gives the one-hot column names like `contract_Two year`), sorted by size.

### 6.4 `src/explain.py`
- `shap_explanation(pipeline, X_explain, X_background)` transforms both sets with the pipeline's own preprocessor, then:
  - logistic regression → `shap.LinearExplainer` (exact and fast; values are in **log-odds**),
  - random forest → `shap.TreeExplainer` (exact for trees; values are in **probability**). `[:, :, 1]` keeps the churn class.
- The background (200 training customers) is the "average customer" that contributions are measured from.
- `mean_abs_shap()` gives global importance (average absolute contribution per feature).

### 6.5 `src/rfm.py`
- `RFM_SQL` aggregates per `customer_unique_id`: last order date, `COUNT(DISTINCT order_id)`, `SUM(price)`, with the snapshot date as a **bind parameter** (`:snapshot`).
- `load_rfm()`: recency = snapshot − last order, in days.
- `prepare_features()`: `log1p` on frequency and monetary (log(1+x) handles skew and works with 0), then `StandardScaler`.
- `choose_k()`: inertia and silhouette for k = 2…8. Silhouette is computed on a 10K sample because it compares every pair of points, which is slow on 95K.
- `fit_segments()`: `KMeans(n_clusters=k, n_init=10, random_state=42)`. `n_init=10` runs 10 random starts and keeps the best, because K-Means can get stuck in a poor local solution.
- `profile()`: per-cluster size, average R/F/M, share of customers and revenue.
- `name_segments()`: names clusters by comparing their averages with the customer-base medians (repeat buyers if avg frequency ≥ 1.5; otherwise recent/lapsed × big/small). Names never depend on the **cluster number**, because K-Means numbering is arbitrary and can change between runs; a unit test checks this.

### 6.6 `src/plots.py`
`setup_style()` gives every chart the same look. `save()` writes to `reports/figures/`. `churn_rate_bars()` draws sorted churn-rate bars labelled with the rate and group size (n), plus a dashed overall-rate line. Showing n matters: a 60% rate in a group of 10 isn't evidence.

### 6.7 `tests/test_project.py` (9 tests)
Leakage columns removed; blank total → 0 for new customers; unexpected blanks fail loudly; "No internet service" merged; profit maths checked by hand; best threshold excludes unprofitable customers (this caught the float bug); RFM features standardised; segment names follow the profile, not the cluster number; sensitivity covers every scenario.

---

## 7. Notebook walkthrough

### 7.1 `01_rfm_segmentation.ipynb`
1. Load RFM from Postgres: 94,983 customers.
2. `describe()` + frequency counts: 92,097 bought once, so frequency is nearly constant.
3. Histograms: monetary is very skewed until `log1p`.
4. `choose_k`: elbow flattens at 4–5; silhouette 0.36 (k=4) vs 0.34 (k=5).
5. Choose k = 5 (reasoning in the notebook) → profile table → share-of-customers vs share-of-revenue chart → scatter of recency × spend coloured by segment.
6. K-Means found the cut-offs itself: about 9 months for recent vs lapsed, about R$ 80 for big vs small spend.
7. Saves segment assignments to `data/processed/` (not in git).

### 7.2 `02_churn_eda.ipynb`
1. Leakage check with evidence: `Churn Reason` is filled for 100% of churners and 0% of others; `Churn Score` ranges 65–100 for churners vs 5–80 for others.
2. Churn-rate bars: contract, payment method, paperless billing; internet type, tech support, online security; senior, dependents, partner.
3. Tenure buckets (52.9% churn in the first 6 months → 9.5% after 4 years) and monthly-charge histograms.
4. Contract × internet table: month-to-month + fiber = 54.6% churn across 2,128 customers.

### 7.3 `03_churn_model.ipynb`
1. Split: 5,634 train / 1,409 test, both 26.5% churn.
2. `tune()`: comparison table (baseline 0.500, LR 0.859 ± 0.013, RF 0.861 ± 0.011) → choose LR.
3. Profit curve on out-of-fold predictions → threshold 0.21.
4. Test-set table, ROC and PR curves, confusion matrix at 0.21 (320 churners caught, 54 missed, 334 loyal customers contacted).
5. Campaign comparison: $0 / $27.5K / $44.0K / **$52.6K**.
6. Sensitivity: 27 scenarios. Model beats contact-everyone in all 27; the tuned threshold ≥ default in 25 (worst shortfall $558); best threshold ranges 0.05–0.83.
7. Odds ratios (e.g. dependents 0.20×, fiber 2.0×, two-year contract 0.43×; tenure 0.28× per ~24.6 months).
8. SHAP beeswarm; LR vs RF rank agreement; electronic-check lesson; tenure/total-charges correlation caveat (0.83).
9. SHAP waterfall for the riskiest customer (88%): new (1 month), fiber, month-to-month, no security or support add-ons, electronic check.
10. What-if: month-to-month → one-year lowers average predicted churn from 42.9% to 29.3% (model estimate, not causal).
11. Saves `reports/model_metrics.json`.

---

## 8. Results

**Segmentation (Olist, k = 5):**
| Segment | Customers | Revenue | Avg recency | Avg spend |
|---|---|---|---|---|
| Recent big spenders | 28.9% | 42.1% | 122 days | R$ 207 |
| Lapsed big spenders | 19.2% | 35.9% | 364 days | R$ 266 |
| Recent small spenders | 29.8% | 9.0% | 155 days | R$ 43 |
| Lapsed small spenders | 19.0% | 7.4% | 432 days | R$ 55 |
| Repeat buyers | 3.0% | 5.6% | 224 days | R$ 260 (2.1 orders) |

**Churn model (test set):** ROC-AUC 0.849, PR-AUC 0.646. At 0.5: precision 65%, recall 57%. At 0.21: precision 49%, **recall 86%**.

**Business value (test set):** $52.6K tuned vs $44.0K default vs $27.5K contact-everyone (assumptions: $50 offer, 30% save rate, 12 months of value).

**Drivers:** tenure (longer = safer), contract (month-to-month risky), fiber internet (risky), dependents (safer), add-ons like tech support and online security (safer).

---

## 9. Limitations and what I would do next

**Limitations**
- The Telco data is a single snapshot (one quarter), so we can't see *when* customers churn or how behaviour changes before they leave.
- The profit assumptions are placeholders; the best threshold depends heavily on them.
- The what-if contract analysis shows association, not causation.
- The Telco data is from one US state and is a sample IBM published; it may not generalise.
- Olist segmentation is driven by recency and spend because 97% of customers bought once.
- K-Means assumes roughly round, similar-sized clusters; customer behaviour isn't always like that.

**Next steps**
- **A/B test** the contract offer to measure its real effect.
- **Uplift modelling:** predict who will *respond* to an offer, not just who will churn. Some customers churn regardless; some stay regardless.
- **Survival analysis:** model *time until* churn (useful with dated, longitudinal data).
- Try gradient boosting (XGBoost/LightGBM) and probability calibration checks.
- **Deploy:** score customers monthly, monitor ROC-AUC and data drift, and retrain quarterly.
- Link segments to channels and campaign costs to measure segment-level ROI.

---

## 10. Cheat sheet

| Term | One-line meaning |
|---|---|
| Data leakage | A feature that wouldn't be known at prediction time (e.g. churn reason) |
| Stratified split | Train and test keep the same class proportions |
| Cross-validation | Rotate train/validation across k folds and average the scores |
| Hyperparameter | Setting chosen before training (C, max_depth, k) |
| Regularisation | Penalty that keeps a model simple to avoid overfitting |
| One-hot encoding | One 0/1 column per category value |
| Standardisation | (x − mean) / std |
| Pipeline | Preprocessing + model fitted together; prevents leakage in CV |
| Odds ratio | exp(coefficient); multiplier on the odds of churning |
| ROC-AUC | Probability the model ranks a random churner above a random non-churner |
| PR-AUC | Area under precision-recall; baseline = positive rate |
| Precision / recall | Correct among flagged / flagged among all churners |
| Threshold | Probability cut-off that turns scores into decisions |
| Out-of-fold prediction | Prediction for a row from a model that didn't train on it |
| SHAP value | One feature's contribution to one prediction |
| Multicollinearity | Correlated features share an effect; their individual coefficients are unstable |
| RFM | Recency, Frequency, Monetary |
| K-Means | Clustering by distance to k centres |
| Inertia / elbow | Within-cluster distance; pick k where it stops falling fast |
| Silhouette | Cluster separation score from −1 to 1 |
| log1p | log(1 + x); tames skew and handles zeros |

---

## 11. Interview questions with answers

**Q: Walk me through the project.** Section 1; offer to go deeper on leakage, model choice or the threshold.

**Q: How did you handle class imbalance?**
"I didn't resample. I kept the model's probabilities honest and dealt with the imbalance in two places: evaluation and decision. For evaluation, I used ROC-AUC, PR-AUC and recall instead of accuracy, because predicting 'nobody churns' already gives 73.5% accuracy. For the decision, I chose the threshold by expected profit, which lowered it from 0.5 to 0.21 and raised recall from 57% to 86%."

**Q: What is data leakage? Did you find any?**
"A feature that wouldn't be available when you make the prediction. This dataset has two obvious ones: Churn Reason, which only exists for churners, and IBM's Churn Score, which ranges 65–100 for churners vs 5–80 for others. Leaving them in would give a near-perfect but useless model. I also kept CLTV out of the features because it's IBM-derived and may use churn. Leakage can also happen in preprocessing, which is why scaling sits inside the pipeline so it's fitted per CV fold."

**Q: Why logistic regression over random forest?**
"They tied: 0.859 vs 0.861 cross-validated ROC-AUC, with a fold-to-fold standard deviation of about 0.01, so the gap is noise. When models tie, I choose the simpler one: it has interpretable odds ratios, well-calibrated probabilities for the profit maths, and it's easier to explain and maintain. I kept the forest to check that SHAP gives the same top drivers, and it does."

**Q: Explain ROC-AUC to a manager.**
"Pick one customer who left and one who stayed. 85% of the time, our model gives the leaver the higher risk score. A coin flip would be 50%."

**Q: How did you choose the threshold?**
"By expected profit. Each contacted customer costs $50; a real churner we contact stays with 30% probability, and that's worth a year of their bill. I computed profit at every threshold using out-of-fold predictions on the training set, so the test set stayed clean. The best was 0.21. On the test set that earned $52.6K vs $44.0K at 0.5 and $27.5K for contacting everyone."

**Q: Those assumptions are made up. How do you know it holds?**
"I tested 27 combinations: offer cost $25–100, save rate 20–50%, customer value 6–24 months. Model targeting beat contacting everyone in all 27, and the tuned threshold matched or beat 0.5 in 25 (within $558 in the other two). But the best threshold moved from 0.05 to 0.83, so the first thing I'd do in a real company is get the true offer cost and customer value from finance."

**Q: What drives churn?**
"Short tenure, month-to-month contracts and fiber internet raise risk. Having dependents, add-ons like tech support and online security, and longer contracts lower it. Both models and both explanation methods (odds ratios and SHAP) agree."

**Q: What's SHAP and why use it?**
"SHAP splits each prediction into contributions from each feature, relative to an average customer. Globally it ranks features and shows the direction of each effect; for one customer, it explains their score, which is what a retention agent needs. It works for any model, so I could compare logistic regression and random forest fairly."

**Q: Electronic-check payers churn at 45%. Why isn't it a top feature?**
"Because e-check payers are mostly month-to-month (78% vs 43%), newer (median 18 vs 35 months) and on fiber (67% vs 32%). Once those are in the model, little independent effect remains. It's a marker of risk more than a cause, so a push to auto-pay alone probably wouldn't help much."

**Q: Can you say one-year contracts reduce churn by X%?**
"No, not causally. The model estimates average risk would fall from 43% to 29% if month-to-month customers were on one-year contracts. But customers choose their contracts, and the ones who commit may be different in ways we can't see. I'd recommend an A/B test: offer a discounted one-year contract to a random half and compare churn after 90 days."

**Q: Why log-transform and scale before K-Means?**
"K-Means uses distances. Spend runs from R$ 1 to over R$ 13,000 with a long tail, so without the log a few extreme spenders would dominate and get clusters of their own. Without scaling, monetary in the hundreds would outweigh frequency (1–2) purely because of units."

**Q: How did you choose k?**
"The elbow and silhouette: the elbow flattens at 4–5, and silhouette is 0.36 at k=4 and 0.34 at k=5. I chose 5 because its segments are clean and actionable, a 2×2 of recent/lapsed × big/small spenders plus repeat buyers. At k=4 one segment mixed small and big spenders. Metrics narrow the options; business usefulness decides."

**Q: What would you do differently with more time or data?**
Section 9: A/B test, uplift modelling, survival analysis, gradient boosting, deployment with monitoring.

**Q: How would you deploy this?**
"Save the fitted pipeline (preprocessing and model together), score all customers monthly from the database, and send the top-risk list with each customer's top SHAP reasons to the retention team's CRM. Monitor ROC-AUC on each month's actual churners and watch for input drift, like a new contract type. Retrain quarterly, and re-check the threshold whenever offer costs change."

---

## 12. How to run

```bash
source .venv/bin/activate
pytest                                  # 9 unit tests
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_rfm_segmentation.ipynb 02_churn_eda.ipynb 03_churn_model.ipynb
jupyter notebook                        # open and read them interactively
```
Or use VS Code: open a notebook and pick the `.venv` kernel, or **Run and Debug → Run tests / Re-run all notebooks**.

Notebook 1 needs the Olist Postgres database from Project 1 (`brew services start postgresql@16`). Notebooks 2–3 only need `data/raw/Telco_customer_churn.xlsx`.
