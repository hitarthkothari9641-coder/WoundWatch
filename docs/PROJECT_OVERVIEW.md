# Wound Trace: post-surgical wound monitoring from a phone photo

> A little care. A clearer picture.

Wound Trace helps people photograph a surgical wound at home, tracks how its colour changes over time, and gives their care team a plain-language reading of what an AI model saw. It is a **research prototype and screening aid. It is not a diagnosis and does not replace a clinician.**

Live demo: https://wound-trace.dvpatel.workers.dev (demo accounts only, synthetic data)

---

## The problem

After surgery, most patients are seen in clinic once, days after discharge. Subtle changes (spreading redness, colour drift, texture change) happen in between, and people in Tier 2 and Tier 3 cities often live hours from the hospital. Early signs get noticed late.

## What Wound Trace does

| For the patient | For the care team |
|---|---|
| Guided photo capture with quality checks (blur, light, framing) | Screening band: Low, Needs review, Elevated, with a calibrated probability |
| Built-in help in **English, Hindi and Gujarati** with read-aloud | Grad-CAM map: where the model looked |
| Timeline of every photo for the same wound | CIELAB colour heat map: measured red-green appearance versus surrounding skin |
| One-tap branded PDF report to show the doctor | "How to read this scan" guidance: what to check, and the limits |

Every screen says what the tool cannot do. A "Low" result never claims infection is absent.

## How it works

```
 Phone / browser ──► Cloudflare Worker ──► FastAPI backend ──► PyTorch models
   (Expo app,        (static UI at edge,     (auth, CSRF, SQLite,   quality gate
    web dashboard)    security headers,       reports, guidance)     DeepLabV3+ segmenter
                      rate limit, shared                              11-backbone ensemble
                      secret to origin)                               Grad-CAM + CIELAB map
```

- **Web dashboard**: plain JavaScript, served from Cloudflare's edge, branded with the Wound Trace kit (Outfit, Plus Jakarta Sans, Space Mono; Sindoor, Kesari, Haldi, Peacock, Samudra palette).
- **Mobile app**: React Native with Expo and TypeScript. One codebase for iOS, Android and the browser. Scan the QR code in the dashboard Help panel (or open `/app/`) to run the same app on any phone with no install and no account.
- **Backend**: FastAPI with cookie sessions, CSRF checks and per-patient access control. A shared secret means only the Cloudflare Worker can reach it.
- **Model**: a frozen-feature ensemble of 11 ImageNet backbones with logistic-regression heads, calibrated with Platt scaling, plus a segmenter to locate the wound and ring of surrounding skin.

## The model, honestly

Only **686** public surgical-wound images carry infection labels (573 Low, 90 Medium, 23 High; 113 count as "Elevated"). That is a small, imbalanced dataset, so the evaluation is cautious.

- Protocol: 549 development images, 5-fold cross-validation with 3 seeds, model chosen by a rule fixed before looking, then the held-out test split (137 images, 20 Elevated) read once.
- Fine-tuning a ResNet-18 did worse than frozen features. The ensemble of frozen features did better.

| Held-out test (137 images) | Baseline | Wound Trace v2 |
|---|---|---|
| Accuracy | 76.6% | 76.6% |
| Precision | 34.2% | 34.2% |
| Recall | 65.0% | 65.0% |
| F1 | 44.8% | 44.8% |
| AUROC | 0.812 | **0.838** (95% CI 0.72 to 0.92) |
| Average precision | 0.542 | **0.578** |

**What this means:** at the chosen operating point, accuracy, precision and recall did not change. Ranking quality (AUROC, average precision) improved on development cross-validation and on test, within the noise expected from only 20 positive test images. We report it that way on purpose.

Known limits:
- The wound segmenter was trained on foot ulcers and finds no wound in about 74% of surgical photos, so the quality gate is advisory in the demo and the guidance says when it could not locate the wound.
- Scores on training images are not held-out; the model card marks which are.
- No prospective clinical validation.

## Built for Tier 2 and Tier 3 cities

- **Language**: English, Hindi, Gujarati, with read-aloud for low-literacy users. (Hindi and Gujarati text should be reviewed by a native speaker and a clinician before public launch.)
- **Big targets, plain words**: 48 px minimum touch targets, one action per screen, no jargon.
- **Low bandwidth**: one photo per upload, edge-served UI, works in any phone browser.
- **Safety first**: "when to contact your doctor" is always one tap away.

## Security

- HttpOnly session cookies, CSRF header on every write, per-patient authorisation.
- Strict Content-Security-Policy, HSTS, no framing, no referrer; self-hosted fonts.
- Rate limiting at the edge; origin gated by a shared secret.
- Model checkpoints load with `weights_only=True`.
- Demo uses synthetic accounts only. No real patient photographs.

## What I built

1. Reproduced the baseline and designed a cross-validated, leakage-safe experiment protocol.
2. Trained and compared frozen-feature and fine-tuned models, and shipped the best (v2 ensemble).
3. Added per-image inference with Grad-CAM, colour heat map and rule-based guidance for 686 images.
4. Rebranded the dashboard to the Wound Trace kit and added a Model page that shows results and limits.
5. Added the trilingual Help assistant and a branded PDF report.
6. Deployed to Cloudflare with edge security headers and an origin gate.
7. Built the React Native app and tested it in the iOS Simulator.

## Stack

Python, PyTorch, scikit-learn, FastAPI, SQLite, JavaScript, Cloudflare Workers, React Native, Expo, TypeScript, Matplotlib, Playwright.

## Honest status and next steps

- The live backend currently runs on a laptop behind a free Cloudflare tunnel. A permanent host (a small VM or container service) is the next step.
- Native-speaker and clinician review of the Hindi and Gujarati help text.
- Retrain the segmenter on surgical wounds; collect more infection-labelled data with consent.
- Prospective study with clinicians before any real-world use.
- Dataset and weight redistribution rights need confirming before the weights are made public.

## Disclaimer

Wound Trace is a research prototype. It is not a medical device and gives no diagnosis. If a wound looks worse, or there is fever, spreading redness, pus or increasing pain, contact a doctor or the nearest hospital the same day.
