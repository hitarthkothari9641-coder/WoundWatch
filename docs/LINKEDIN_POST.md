# LinkedIn post (copy, edit names, attach the demo video)

---

After surgery, most patients are seen once, days after discharge. What happens in between often goes unseen, especially for people living hours from a hospital.

So we built **Wound Trace**: take a phone photo of a surgical wound, and get a plain-language reading for your care team.

What it does
- Checks photo quality, then tracks colour change across photos of the same wound
- Shows where the AI model looked (Grad-CAM) and a measured redness map
- Writes a one-page branded PDF report for the doctor
- Help built in, in English, Hindi and Gujarati, with read-aloud, for Tier 2 and Tier 3 cities

What I want to be straight about
- It is a research screening aid, not a diagnosis
- Only 686 public images had infection labels. On the held-out test, accuracy, precision and recall matched our baseline (76.6%, 34.2%, 65.0%). Ranking quality improved (AUROC 0.81 to 0.84), but with only 20 positive test images that is within noise
- Fine-tuning made the model worse than frozen features, so we shipped an 11-model ensemble of frozen features
- The segmenter misses the wound in many surgical photos, so the app says so instead of pretending

Built with PyTorch, FastAPI, Cloudflare Workers, React Native (Expo), one codebase for iOS and Android.

Live demo: https://wound-trace.dvpatel.workers.dev
Write-up: [link to portfolio page]

Next: a permanent backend, clinician and native-speaker review of the help text, and a segmenter trained on surgical wounds.

If you work in surgery, wound care or digital health in India, I would value your feedback.

#HealthTech #ComputerVision #DigitalHealth #MedTech #ReactNative #Cloudflare #MachineLearning #India

---

## Shorter version (for X / Threads)

Built Wound Trace: phone photo of a surgical wound -> quality check -> colour-drift tracking -> Grad-CAM + redness map -> branded PDF for the doctor. Help in English, Hindi, Gujarati with read-aloud. Honest result: ranking improved (AUROC 0.81 to 0.84), accuracy and recall unchanged on a 137-image test. Research prototype, not a diagnosis. Demo: https://wound-trace.dvpatel.workers.dev

## Video notes

- `dist/demo/dashboard.mp4`: dashboard on an iPhone 18 Pro Max sized screen (440 x 956), with captions.
- `dist/demo/app_demo.mp4`: the React Native app in the iOS Simulator.
- Add captions or a voice-over before posting. Check you have the right to show the dataset photographs used in the demo.

## Portfolio page blurb (2 lines)

Wound Trace is a post-surgical wound monitoring prototype: phone photos, colour-drift tracking, explainable AI maps and plain-language guidance in English, Hindi and Gujarati. React Native app, Cloudflare-hosted dashboard, PyTorch model evaluated honestly on a held-out set.
