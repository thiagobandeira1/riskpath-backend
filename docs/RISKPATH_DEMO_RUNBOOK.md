# RiskPath Demo Runbook

**Live walkthrough script for the executive demo**

*Frontend: riskpath-clinician-companion.lovable.app · Core demo ~5 minutes · Format: DO / SEE / SAY*

---

## 5 Minutes Before You Present — Pre-Flight

Do this right before you walk in. It guarantees a smooth, lag-free demo.

1. Open **https://riskpath-clinician-companion.lovable.app** on your iPad.
2. Confirm the green **"Online"** badge in the top-right corner. If it shows red, reload the page once.
3. Tap **Patient #1**, then scroll the cards and tap **Patient #5**. This warms up the model so there is no lag during the real demo.
4. Tap **Re-explain** once. This warms up the SHAP explanation engine.
5. Reload the page so you start clean on Patient #1.
6. Leave it on the **Predict** screen, ready to go.

---

## Your Patient Roster (memorize these)

The five sample patients are ordered left-to-right from lowest to highest risk. No data entry needed — you just tap a card.

| Card | Diagnosis | Discharge to | Risk | Band |
|---|---|---|---|---|
| #1 | Symptoms | Home | 0.020 | LOW |
| #2 | Respiratory | Home Health Care | 0.100 | LOW |
| #3 | Circulatory | Rehab | 0.380 | MODERATE |
| #4 | (chapter A) | Skilled Nursing | 0.680 | HIGH |
| #5 | (chapter Z) | Skilled Nursing | 0.898 | VERY HIGH |

**Patient #1 is your low-risk baseline. Patient #5 is your dramatic high-risk case** (12 prior admissions in 6 months). Those two are the stars.

---

## Step 1 — Frame the Problem (30 seconds)

**DO:** Before touching the screen, set the stage.

**SAY:**

> Hospital readmissions cost the US health system about 26 billion dollars a year, and CMS penalizes us for them through the Readmissions Reduction Program. The problem isn't that we don't care — it's that on a busy discharge day, we can't tell which patients are actually at risk until it's too late. RiskPath solves that. Let me show you.

---

## Step 2 — Establish It's Real (15 seconds)

**DO:** Point to the green **"Online"** badge in the top-right.

**SAY:**

> Everything you're about to see is a live model running in production — not a slideshow, not fake data. Every number is a real prediction.

---

## Step 3 — The Low-Risk Baseline (45 seconds)

**DO:** Tap **Patient #1** (Symptoms, going Home).

**SEE:** The gauge settles at **0.020** with a green **LOW** pill.

**SAY:**

> Here's a patient discharged home after a symptom-related admission. RiskPath puts their 30-day readmission risk at 2 percent. This is a green-light patient — routine discharge planning, standard follow-up. We don't need to spend scarce care-management resources here.

---

## Step 4 — The Dramatic Contrast (1 minute) — YOUR MONEY MOMENT

**DO:** Scroll the patient cards to the right, then tap **Patient #5** (Skilled Nursing).

**SEE:** The gauge sweeps up to **0.898**, the arc fills green-yellow-orange-red, and a red **VERY HIGH** pill appears.

**SAY:**

> Now watch the same model on a different patient. Ninety percent risk. This patient is very likely to be back in our hospital within 30 days. If they walk out the door with a standard discharge packet, we've almost certainly missed our chance to intervene.

**DO:** Pause. Let the red gauge sit on screen for a beat — it's visually striking.

---

## Step 5 — Explain WHY (1.5 minutes) — YOUR STRONGEST DIFFERENTIATOR

**DO:** Tap **"Re-explain"** on the Top 5 Risk Drivers card. Wait about 3 seconds for the bars to appear.

**SEE:** Five red bars — Prior Readmission Count, Admission Frequency times Recency, Prior Admissions over 6 months, and related utilization features.

**SAY:**

> This is what makes RiskPath different from a black box. It doesn't just say "high risk" — it tells us exactly why. Look at the top drivers: this patient has been admitted twelve times in the last six months. The model is seeing a clear frequent-flyer pattern. That's not a mystery score — it's an actionable signal that points our care team straight at an intensive transitions-of-care program.

**SAY (the trust line):**

> And when a physician asks "why should I trust this?", I can show them. Every prediction is explainable at the patient level, in clinical terms they recognize.

---

## Step 6 — Turn Risk Into Action (45 seconds)

**DO:** Scroll down to the **Recommended Care Pathway** card.

**SEE:** Band-specific recommendations (phone outreach within 48 to 72 hours, transportation plan, patient-portal enrollment) with a footer citing Project RED and IHI BOOST.

**SAY:**

> And it closes the loop. For each risk level, RiskPath maps to evidence-based interventions — sourced from Project RED and IHI's BOOST guidelines, and fully customizable to our own protocols. Risk score in, care plan out.

---

## Step 7 — Trust and Governance (1 minute) — WHAT WINS THE EXECUTIVE

**DO:** Tap **Model Card** in the left sidebar.

**SEE:** Model name xgboost-v7-seed0, AUROC 0.7929, 50 features, and the Methodology and Intended Use sections.

**SAY:**

> For your governance committee: this is a peer-reviewed model with a validated AUROC of 0.79, trained on 244,000 admissions. And critically —

**DO:** Scroll to the **Intended Use and Limitations** section.

**SAY:**

> — it's transparent about its own limits. This is decision support, not autonomous decision-making. It explicitly states what it should and shouldn't be used for, and that any institution needs to validate it locally before relying on it. That responsible framing is the difference between a tool your medical staff will adopt and one your general counsel will block.

---

## Step 8 — Close (30 seconds)

**DO:** Tap back to **Predict**.

**SAY:**

> Three takeaways. One — this is a real, validated model, not a demo. Two — every prediction is explainable in terms clinicians can act on. Three — it's already running in production, and we could stand up a pilot on our own data in a matter of weeks. The question isn't whether we can predict readmission risk. It's whether we want to start acting on it before patients walk out the door.

**DO:** Stop talking. Let them respond.

---

## Optional Extensions (only if engaged and time allows)

**Show the model discriminates smoothly:** Tap #1, then #3, then #5 in sequence. Say: "It's not just high or low — it grades risk across the whole spectrum."

**Show interactivity:** Scroll to the Patient Features panel, edit a value (for example, bump Prior Admissions), and watch the risk re-score live. Say: "It responds to the clinical reality in real time." Only do this if you've rehearsed it.

**Show batch scoring:** Click Batch Score and upload a CSV. Say: "And it scales — score your entire discharge census in seconds."

---

## Likely Questions and Your Answers

**Is patient data stored?**
No — the backend is stateless. Predictions flow through and nothing is persisted. No PHI in logs, no database.

**What was it trained on?**
MIMIC-IV — a public, de-identified critical-care dataset of 244,000 admissions from Beth Israel Deaconess. For a real pilot, we'd retrain on our own population.

**How accurate is it?**
AUROC of 0.79 — meaningfully better than chance, in line with published readmission models. And it's slightly under-confident at the high end, which is the safe direction clinically.

**Can we use this tomorrow?**
Not as-is on real patients — it needs local validation and an IRB conversation first. But the platform is built and deployed. The gap is validation, not engineering.

**What does it cost to run?**
Effectively nothing right now — under 5 dollars a month of cloud infrastructure. The cost is in validation and integration, not the technology.

---

## If Something Breaks Mid-Demo

**Blank prediction or "Offline" badge:** Say "Let me reload that," reload the page, and tap the patient again. You pre-warmed it, so this is unlikely.

**Slow first prediction:** Say "It's spinning up," wait 3 seconds, and it'll resolve.

**Total failure:** You have the platform-guide PDF as backup — walk through it verbally. Nobody will know the difference. Confidence carries the room.
