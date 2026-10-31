"""Plain-language reading guide for a scan's Grad-CAM and colour heatmap.

Rule-based and deliberately conservative: it describes where the model looked and what
the measured colours are, and says what a clinician could inspect. It never states
that infection is absent and never replaces examination.
"""
import numpy as np

DISCLAIMER = ("Screening signal for clinician review, not a diagnosis. Grad-CAM shows where the model "
              "looked, not infected tissue; the colour map shows measured red-green appearance only. "
              "A Low result does not rule out infection; act on symptoms and clinical assessment.")


def attention_regions(cam, wound, ring):
    """Share of Grad-CAM mass inside the wound, the surrounding ring and elsewhere."""
    total = float(cam.sum())
    if total <= 0: return None
    w, r = float((cam * wound).sum()) / total, float((cam * ring).sum()) / total
    return {"wound": w, "ring": r, "outside": max(0.0, 1.0 - w - r)}


def build_guidance(result, shares=None, colour=None, threshold=None, outline_found=True):
    """result: predict_image dict. shares: attention_regions output or None.
    colour: feature_row dict or None. Returns {summary, observations, check, disclaimer}."""
    status = result.get("status")
    obs, check = [], []
    if status == "retake":
        return {"summary": "Photograph quality is too low for a reliable reading.",
                "observations": list(result.get("reasons") or []),
                "check": ["Retake in even lighting, in focus, with the whole wound in frame."],
                "disclaimer": DISCLAIMER}
    p = result.get("probability")
    if status == "defer":
        summary = "Score sits close to the decision threshold; the model cannot separate Low from Elevated here."
        check.append("Prioritise clinical examination; do not rely on the score.")
    elif result.get("risk_band") == "elevated":
        summary = "Model flags this photograph as Elevated for clinician review."
        check.append("Examine for erythema, swelling, warmth, pain, discharge or odour; consider culture/escalation per protocol.")
    else:
        summary = "Model scores this photograph as Low; this does not exclude infection."
        check.append("Continue routine follow-up; escalate if symptoms or other signs are present.")
    if p is not None:
        obs.append(f"Calibrated Elevated probability {p:.0%}" + (f" (decision threshold {threshold:.0%})." if threshold is not None else "."))
    if not outline_found:
        obs.append("The wound outline could not be located automatically, so the region reading and colour measurements are unavailable. The attention map still shows where the model looked.")
        check.append("Locate and examine the wound and surrounding skin directly.")
    elif shares is None:
        obs.append("Grad-CAM unavailable for this photograph; no attention reading.")
    else:
        w, r, o = shares["wound"], shares["ring"], shares["outside"]
        obs.append(f"Model attention: {w:.0%} on segmented wound, {r:.0%} on surrounding skin ring, {o:.0%} elsewhere.")
        if o >= 0.5:
            obs.append("Most attention is outside the segmented wound: possible background, dressing, lighting or segmentation error. Treat the explanation as unreliable.")
            check.append("Verify the wound outline on the photograph before trusting the heat map.")
        elif r >= 0.35:
            obs.append("Substantial attention on the skin around the wound.")
            check.append("Inspect periwound skin for redness, warmth, swelling or induration spreading from the edge.")
        if w >= 0.5:
            obs.append("Attention is concentrated on the wound bed.")
            check.append("Inspect the wound bed for slough, exudate, tissue colour change or necrosis.")
    if not outline_found:
        pass
    elif colour:
        d = colour.get("erythema_a_mean_difference")
        if d is not None:
            direction = "redder" if d > 0 else "less red"
            obs.append(f"Wound bed is {direction} than surrounding skin (a* difference {d:+.1f}); raw colour, lighting-dependent.")
        if colour.get("a_ring_mean") is not None:
            obs.append(f"Surrounding skin a* {colour['a_ring_mean']:.1f}, wound a* {colour['a_wound_mean']:.1f}. Compare with the same wound's earlier photographs, not with other patients.")
    else:
        obs.append("Colour measurements unavailable (wound or ring could not be measured).")
    return {"summary": summary, "observations": obs, "check": check, "disclaimer": DISCLAIMER}
