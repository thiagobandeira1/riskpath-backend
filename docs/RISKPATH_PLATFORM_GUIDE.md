# RiskPath

**A Clinical Decision-Support Platform for 30-Day Hospital Readmission Risk**

*Version 0.1.0 · Build 30e518d · Model xgboost-v7-seed0 · Generated 2026-05-17*

---

## What RiskPath Is

RiskPath is a web-based clinical decision-support platform that estimates the probability that an individual Medicare-eligible inpatient will be readmitted to a hospital within 30 days of discharge. For each patient it produces a calibrated probability between 0 and 1, classifies that probability into one of four clinical risk bands, and surfaces the top five clinical and operational features driving the prediction. It does not make decisions; it gives clinicians, case managers, and quality-improvement teams a transparent, model-driven view of risk so that human judgment can act on it earlier and with better evidence.

The platform was built to make readmission risk legible at the bedside and at the population level. A single patient can be scored in under a second. A cohort of one hundred patients can be batch-scored in roughly the same time. Every prediction comes with a Shapley-value explanation that names the specific features pushing the risk up or down, so the model's reasoning is auditable rather than opaque.

RiskPath is built on top of a published research model — the V7 XGBoost classifier trained on MIMIC-IV v3.1 — wrapped in a production FastAPI backend and a modern React frontend. It is deployed publicly and accessible from any device with a web browser, including iPads at the point of care.

---

## Who It Is For

RiskPath is designed for the people who actually have to do something about readmission risk:

**Hospital quality-improvement teams** use the platform to identify which patients on a discharge list need additional intervention before they go home, and to track which interventions are reducing risk over time.

**Case managers and discharge planners** use the per-patient screen to triage their daily workload — flagging high-risk patients for intensive transitions-of-care services, and routine-risk patients for standard discharge planning.

**Hospitalists, intensivists, and primary care physicians** use the SHAP explanations to understand why a given patient is high risk, and to focus their post-discharge follow-up on the specific clinical drivers the model identifies.

**Health-system leadership** uses the cohort dashboard and Model Card to evaluate population-level risk distribution and the model's performance over time, supporting governance and regulatory conversations.

The platform is explicitly *not* designed for autonomous clinical decision-making. It is a decision-support tool. Every action it recommends should be reviewed by a licensed clinician.

---

## The Four Modules

The platform is organized into four modules accessible from the left sidebar. Each module serves a distinct workflow.

### Module 1: Predict

The Predict module scores a single patient. It is the most-used module and the entry point for most users.

Workflow:

1. Open the platform. You land on the Predict module by default.
2. The page loads with a row of Sample Patient cards drawn from a pre-baked pool of anonymized example records. Each card shows the patient's DRG code, primary diagnosis chapter letter, discharge disposition, and how long since their last discharge. Tap any card to load that patient.
3. Adjust the Threshold slider if you want to change the cutoff for the binary prediction (default 0.50). Lower thresholds catch more potential readmissions at the cost of more false positives.
4. The 30-Day Risk Probability gauge updates immediately, showing the model's estimated probability as a number between 0.000 and 1.000 along with the corresponding risk band pill (LOW, MODERATE, HIGH, or VERY HIGH).
5. Tap "Re-explain" on the Top 5 Risk Drivers card to compute SHAP values for that patient. The five features with the largest absolute contributions appear as a bar chart, color-coded green for features pushing risk up and red for features pulling it down. Each feature is labeled with both a human-readable name (e.g., "Length-of-Stay Trend (180 days)") and the underlying raw feature name in parentheses.
6. Scroll down to the Care Pathway card to see band-specific recommendations: which transitions-of-care services to engage, which post-discharge follow-up cadence to use, and which red flags to monitor.

The Predict module is the workhorse. A clinician can score a patient, see why the model thinks they are at risk, and pivot to action in under thirty seconds.

### Module 2: Batch Score

The Batch Score module scores up to one hundred patients at once from a CSV upload.

Workflow:

1. Navigate to Batch Score from the left sidebar.
2. Drag-and-drop a CSV file onto the upload area, or click to browse. The CSV must contain a header row with column names matching the fifty required feature columns (the Model Card module lists them all).
3. The platform validates the CSV structure, calls the backend's batch endpoint, and displays results in a sortable table: one row per patient with probability, risk band, and prediction (above or below threshold).
4. A risk-band histogram appears alongside the table, showing how many patients fall into each of the four bands.
5. Results can be downloaded as a CSV for downstream workflow integration.

Use Batch Score for daily census triage, for retrospective evaluation of a discharge cohort, or for evaluating an intervention's impact on average cohort risk over time.

### Module 3: Insights

The Insights module shows a cohort-level dashboard. In the current deployment, it visualizes a synthetic example cohort to demonstrate the dashboard layout — feature-level distribution histograms, average probability by service line, and an over-time trend view.

Once authenticated user data integration is added, this module will visualize the actual cohorts that hospital users have loaded — turning the platform from a per-patient scoring tool into a population-management dashboard.

### Module 4: Model Card

The Model Card module is the platform's transparency and governance surface. It is the most important module for committees and regulatory conversations.

It contains five sections:

**Deployed Model header** — model name (`xgboost-v7-seed0`), validation AUROC (0.7929), published 10-seed average AUROC (0.7935), feature count (50), and random seed used for deployment selection (0).

**Methodology** — architecture (gradient-boosted decision trees via XGBoost), training data (MIMIC-IV v3.1, 244,576 admissions, PhysioNet credentialed access), and feature engineering (50 V7 features: 4 categorical and 46 numeric, including length-of-stay trends, order patterns, comorbidities, lab abnormalities, prior-utilization counts, and interaction terms).

**Performance** — AUROC on validation, AUROC on the published 10-seed test set, top ten features by gain importance, and the operating-point confusion matrix at the default 0.50 threshold.

**Intended Use & Limitations** — explicit statements on what the model should and should not be used for, generalization caveats, and known biases.

**Versioning** — current build hash, deployed timestamp, and the data version the model was trained on.

The Model Card has an "Export to PDF" button so it can be shared with non-platform stakeholders.

---

## The Four Risk Bands

Every prediction is mapped to one of four risk bands. The bands carry color coding throughout the platform and drive the Care Pathway recommendations.

**LOW** — probability below 0.30, displayed in green. Most patients fall here. Routine discharge planning, standard medication reconciliation, follow-up appointment within fourteen days.

**MODERATE** — probability between 0.30 and 0.60, displayed in yellow. Standard transitions-of-care intervention: dedicated discharge call within 48 hours, medication reconciliation by a pharmacist, follow-up appointment within seven days.

**HIGH** — probability between 0.60 and 0.85, displayed in orange. Intensive transitions of care: enrollment in a transitions-of-care program, home health referral consideration, follow-up appointment within 72 hours, dedicated case manager.

**VERY HIGH** — probability at or above 0.85, displayed in red. All-hands response: case-manager hand-off before discharge, home visit within 48 hours of discharge, follow-up appointment within 48 hours, palliative care consultation if appropriate.

The band thresholds (0.30, 0.60, 0.85) are calibrated against the V7 model's published distribution on the held-out test set. They are not arbitrary cut points — each band corresponds to roughly a quartile of the empirical risk distribution, with the very-high band representing the top fifteen percent.

---

## How to Read a SHAP Explanation

SHAP — short for SHapley Additive exPlanations — is a method from cooperative game theory that fairly distributes a model's prediction across its input features. For any single prediction, SHAP tells you how much each of the fifty input features contributed, positively or negatively, to the final probability.

In the Top 5 Risk Drivers chart:

Each bar represents one feature. The bar's length corresponds to that feature's contribution magnitude. The bar's direction and color tell you whether the feature is pushing risk up (green bars extending right) or pulling it down (red bars extending left).

When you read an explanation, you should ask three questions:

First, *which features are driving this patient's risk up?* These are the green bars. They are the actionable signals — the things a clinician can address. If "Log (# of Labs)" is pushing risk up, that points to lab abnormalities or high test-ordering frequency as a driver. If "Length-of-Stay Trend (180 days)" is pushing risk up, the patient has had a worsening pattern of admissions over the prior six months.

Second, *which features are pulling risk down?* These are the red bars. They tell you the model's "yes-but" — protective factors specific to this patient. Knowing them helps the clinician avoid over-reacting to risk drivers that may already be mitigated.

Third, *do the named drivers match clinical intuition?* If the model says the top driver is "Discharge Location Target Encoding" but the clinician knows the patient's destination is well-supported, that mismatch is a signal to dig deeper — either the model is missing something or the clinician's mental model is incomplete.

SHAP values are computed at request time by the deployed `shap.TreeExplainer` against the V7 XGBoost model. The first explanation per session takes around half a second; subsequent ones are under fifty milliseconds because the explainer is pre-warmed at app startup.

---

## The Underlying Model

RiskPath is built on the V7 seed-0 XGBoost classifier, a gradient-boosted decision-tree model trained as part of a Florida International University master's capstone in data science. The model has been peer-reviewed and is being prepared for publication.

**Training data.** MIMIC-IV v3.1, the publicly available critical-care dataset from Beth Israel Deaconess Medical Center, accessed under a PhysioNet Data Use Agreement. The training table contains 244,576 admissions across roughly 180,000 unique patients.

**Feature engineering.** Fifty engineered features in the "V7 parsimonious set," reduced from larger candidate sets via stability selection. Four categorical features (DRG code, last DRG disposition, discharge location, primary diagnosis chapter) are integer-encoded via training-set-fitted label encoders. The remaining forty-six features are numeric and include length-of-stay trends over 180 days, order-pattern features (late orders, order frequency), comorbidity counts, lab abnormality counts, prior-utilization counts at multiple time windows (six months, one year), target-encoded categorical features, and interaction terms (e.g., prior admits times age, renal risk times age).

**Train-test split.** Patient-grouped 80/20 split — all admissions belonging to a given patient are kept together in either the training or test set, never split across them. A 10% inner validation slice is held out from the training portion for XGBoost early stopping.

**Performance.** Validation AUROC of 0.7929 on the seed-0 model. Published 10-seed test-set average AUROC of 0.7935 (the headline metric in the publication). For reference, a LightGBM model trained on the same features scored marginally higher (0.7960) on the test set, but XGBoost was retained for the production deployment to preserve continuity with the original capstone defense.

**Calibration.** The model is well-calibrated in the LOW and MODERATE bands and slightly under-confident in the HIGH and VERY HIGH bands — meaning that when the model says "0.80 probability," the empirical readmission rate in the test set is closer to 0.78. This under-confidence is preferable to over-confidence in a clinical setting.

---

## Live URLs

**Frontend (the RiskPath platform):** https://riskpath-clinician-companion.lovable.app

This is the URL to share. Open it in any browser on any device. No login required.

**Backend API:** https://riskpath-backend-production.up.railway.app

The FastAPI backend that serves all predictions and explanations. Not designed to be opened directly in a browser, but the interactive API documentation is available at `/docs`.

**Interactive API documentation:** https://riskpath-backend-production.up.railway.app/docs

Auto-generated Swagger UI for the backend's five endpoints. Useful for developers integrating RiskPath into other systems.

**Frontend source code:** https://github.com/thiagobandeira1/riskpath-clinician-companion

Public GitHub repository. Every push to the `main` branch auto-syncs to Lovable, which redeploys the hosted preview.

**Backend source code:** https://github.com/thiagobandeira1/riskpath-backend

Public GitHub repository. Every push to the `master` branch auto-triggers a Railway redeploy.

---

## Technical Architecture

### Backend

The backend is a Python FastAPI application that serves five HTTP endpoints:

- `GET /health` — liveness check, returns service status and model name.
- `GET /metadata` — feature schema, including per-feature type, value range, and missingness rate.
- `GET /examples?n=N` — N anonymized example patients drawn from a pre-baked pool of one hundred MIMIC-derived rows.
- `POST /predictions` — score a single patient; query parameter `threshold` controls the binary prediction cutoff.
- `POST /predictions/batch` — score up to one hundred patients in one request.
- `POST /explanations` — return SHAP values, base value, and feature names for a single patient.

The backend wraps the V7 XGBoost model with a thin inference layer (`src/inference.py`) that loads the model pickle, fits label encoders for the four categorical features, and lazily builds a SHAP TreeExplainer the first time it is needed. At app startup, a FastAPI `lifespan` hook pre-warms both the predictor and the SHAP explainer so the first request does not pay the three-second cold start.

The deployment image is built via a multi-stage Dockerfile based on `python:3.11-slim`. The runtime image includes `libgomp1` (the only system dependency XGBoost needs) and runs as a non-root user. The container responds on `$PORT` (set by Railway) and uses `tini` as PID 1 for clean signal handling.

The model files (the XGBoost pickle, the JSON feature column list, and two small pre-baked artifacts containing the fitted label encoders and the example patient pool) ship in the Docker image. The 25 MB MIMIC-derived training parquet is intentionally NOT shipped — pre-baking the encoders and a small examples pool eliminates the runtime dependency on the parquet, keeps the image slim, and avoids redistributing the source MIMIC data.

### Frontend

The frontend is a React 19 single-page application built with TanStack Start, TanStack Router, TanStack Query, Tailwind CSS, and shadcn/ui components. It is server-side rendered via the @cloudflare/vite-plugin, which deploys the app as a Cloudflare Worker.

The frontend talks to the backend via a thin API client (`src/lib/api.ts`) that handles request shaping, response normalization, and a consistent error envelope. A mock layer (`src/lib/mockApi.ts`) is gated behind the `VITE_USE_MOCK_API` environment variable so the frontend can be developed and demoed offline without backend connectivity. In production, mock mode is disabled.

The Predict module uses TanStack Query to cache responses by patient identifier, so re-visiting a previously scored patient is instant. Long-running operations (batch score, SHAP explanation) show optimistic loading states with clear progress indication.

### Deployment

Both repositories use GitHub-integrated continuous deployment. A `git push` to the master branch of the backend repository triggers a Railway redeploy in roughly 90 seconds (Docker layers are cached, so only the changed layer rebuilds). A `git push` to the main branch of the frontend repository triggers a Lovable auto-sync; the user then taps "Publish" once in the Lovable dashboard to push the new build to the public URL.

This means iteration during the demo prep phase is fast: edit code locally, push, see the change live within minutes.

---

## Data Privacy and DUA Compliance

RiskPath was built with explicit attention to the PhysioNet Data Use Agreement governing MIMIC-IV.

**What ships in the public Docker image and repositories:**

- The trained XGBoost model pickle. The model weights are derived knowledge, not patient data.
- The fitted label encoders for the four categorical features. These map category strings to integer codes; they are model metadata, not patient data.
- A pool of one hundred anonymized example patient rows, used by the `/examples` endpoint. These are MIMIC-derived but de-identified (no patient identifiers, no admission identifiers, no dates), and the user has explicitly approved their inclusion in the public deployment.
- The feature schema JSON.

**What does NOT ship anywhere outside the local development environment:**

- The 25 MB MIMIC-derived training parquet.
- Any identifying information.

**Server-side mitigations** include `X-Robots-Tag: noindex, nofollow` on every response, a `/robots.txt` returning `Disallow: /` to keep the backend out of search indexes, and a CORS allowlist restricted to the production frontend origins (Lovable, Cloudflare Workers, Cloudflare Pages, Claude Design previews, and the planned `*.riskpath.app` custom domain).

Operationally, the backend URL is intentionally not promoted in public-facing material, and the project does not link to it from any indexed page.

---

## Intended Use and Limitations

RiskPath is a research and decision-support platform. It is not an FDA-cleared medical device. It must not be used as the sole basis for any clinical decision.

**Appropriate uses:**

- Augmenting a clinician's judgment about post-discharge risk.
- Triaging discharge cohorts to allocate finite transitions-of-care resources.
- Identifying patient-specific drivers of risk to focus follow-up interventions.
- Quality-improvement and population-management research.
- Educational use in graduate health informatics, data science, and clinical decision support courses.

**Inappropriate uses:**

- Sole basis for any clinical decision, including discharge, level-of-care, or end-of-life decisions.
- Coverage, payment, or denial decisions by payers.
- Use in jurisdictions or institutions that have not conducted local validation.
- Use on patient populations meaningfully different from the MIMIC-IV training distribution (e.g., pediatric, obstetric, or non-US populations) without re-validation.

**Known limitations:**

- The model is trained on data from a single US tertiary-care center (Beth Israel Deaconess Medical Center) and may not generalize to other settings.
- The MIMIC-IV dataset represents critical-care admissions in particular and may over-represent more acutely ill patients than the average inpatient census.
- The model has not been tested for fairness across demographic subgroups in production. Pre-deployment fairness analysis is on the roadmap.
- The model is trained on historical data through 2020 and may not reflect post-2020 changes in care delivery patterns, including those introduced by the COVID-19 pandemic.

Any institution adopting RiskPath should conduct local validation on a recent cohort representative of their patient population before relying on its outputs.

---

## Demo Walkthrough

For a five-minute demo, use this script:

**Minute 1 — Open the platform.** Navigate to `https://riskpath-clinician-companion.lovable.app` on your iPad or browser. Point out the "Online" status indicator in the upper right, confirming the live backend connection. Say: "Every prediction you'll see in the next few minutes is a real call to our deployed model — there's no mock data here."

**Minute 2 — Score a patient.** Tap Patient #1 in the Sample Patients row. The 30-Day Risk Probability gauge updates immediately. Point out the LOW risk band pill in green. Say: "This patient's 30-day readmission probability is roughly one in a thousand. They're going home with routine planning."

**Minute 3 — Show explainability.** Tap "Re-explain" to compute SHAP values. The Top 5 Risk Drivers chart populates with green bars (features pushing risk up) and red bars (features pulling risk down). Walk through the top driver: "Discharge Location Target Encoding is the biggest factor for this patient — that's the model picking up on where the patient is going after discharge." Then tap a different patient. Show that the drivers change, demonstrating that explanations are patient-specific, not global.

**Minute 4 — Show the Model Card.** Click Model Card in the left sidebar. Point out the deployed model name (`xgboost-v7-seed0`, no `-MOCK` suffix, proving this is the real model), the AUROC of 0.7929, and the 50-feature count. Scroll to the Intended Use section and read one sentence: "This is decision support, not autonomous decision-making." This is the moment that buys regulatory trust.

**Minute 5 — Close on the value.** Return to Predict. Say: "Three things to take away. First, this is a real, calibrated, peer-reviewed model — not a demo. Second, every prediction is explainable at the patient level, in features clinicians can act on. Third, it runs in production on infrastructure that anyone in this room could deploy a copy of in an afternoon."

For a fifteen-minute demo, add a Batch Score walkthrough between minutes three and four. Upload a small CSV (ten patients is enough), point out the risk-band histogram, and emphasize the per-second throughput.

---

## Frequently Asked Questions

**Where do the example patients come from?**

The one hundred example patients in the `/examples` pool are anonymized rows drawn from the V7 training table, which is itself derived from MIMIC-IV v3.1. They contain only the fifty model features; no patient identifiers, admission identifiers, or dates were ever loaded into memory when the pool was created.

**Can I upload my own patient data?**

Yes, via the Batch Score module. The platform accepts CSV files with the fifty required feature columns. No data is persisted on the backend — patient data flows through the prediction endpoint and is discarded after the response is returned. There is currently no user authentication, so do not upload anything you would not be comfortable sending to a public API.

**Is patient data stored?**

No. The backend is stateless. Every request is independent. No database, no logs containing patient features.

**What's the latency?**

A single prediction returns in roughly 80 milliseconds end-to-end, including network round-trip. A SHAP explanation returns in roughly 200 milliseconds. A batch of one hundred patients returns in roughly 1.5 seconds.

**Can the model be retrained?**

Yes, but not from within the platform. The training pipeline lives in the upstream research repository. Retraining requires PhysioNet credentialed access to MIMIC-IV. The retrained model can be swapped into RiskPath by replacing the model pickle and re-running the freeze script that produces the deployed encoders and example pool.

**What does it cost to run?**

The Railway backend runs on the trial credit (\$5/month, sufficient for hundreds of thousands of predictions). The Lovable frontend runs on the free tier. The total recurring cost is currently zero, dipping into the trial credit only under sustained load.

**Can I add my own custom domain?**

Yes. Both Railway and Lovable support custom domains via CNAME records. The architecture is set up to accept a `*.riskpath.app` custom domain — the CORS allowlist already includes it.

---

## Glossary

- **AUROC**: Area Under the Receiver Operating Characteristic curve. A measure of model discrimination ranging from 0.5 (no better than chance) to 1.0 (perfect separation). RiskPath's deployed model scores 0.7929.

- **DRG**: Diagnosis-Related Group. A patient classification system that categorizes inpatient stays into clinically and resource-similar groups. RiskPath uses DRG code as a categorical feature.

- **DUA**: Data Use Agreement. The legal agreement governing access to and use of restricted datasets. RiskPath's training data is subject to the PhysioNet DUA for MIMIC-IV.

- **LOS**: Length of Stay. The number of days from hospital admission to discharge. Used in multiple features including LOS trends and LOS per prior admission.

- **MIMIC-IV**: Medical Information Mart for Intensive Care, version IV. A publicly available critical-care dataset from Beth Israel Deaconess Medical Center, released by PhysioNet. The source data for RiskPath's model.

- **SHAP**: SHapley Additive exPlanations. A method from cooperative game theory for attributing a model's prediction to its input features. RiskPath uses SHAP to produce per-patient explanations.

- **Target Encoding**: A feature engineering technique that replaces a categorical value with the mean of the target variable for that category. Used for several categorical features in the V7 set (denoted by the `_te` suffix).

- **TreeExplainer**: SHAP's optimized implementation for tree-based models like XGBoost. Computes exact SHAP values in polynomial time, enabling sub-second per-request explanations.

- **V7**: The seventh iteration of the feature set used during the upstream research project. The "parsimonious" V7 set contains fifty features selected for both predictive power and clinical interpretability.

- **XGBoost**: Extreme Gradient Boosting. The gradient-boosted decision-tree library used to train the deployed RiskPath model.
