"""Measured colour overlays and classifier Grad-CAM (not diagnostic masks)."""
import numpy as np
import torch
from PIL import Image
from .color_features import rgb_to_cielab, feature_row
from .common import image_tensor, device
from .inference import segment_wound

def colour_analysis(image, segmenter):
    # Bound CIELAB allocation, and use a resolution-consistent ring width.
    view=image.copy();view.thumbnail((512,512),Image.Resampling.BICUBIC)
    mask=segment_wound(view,segmenter)
    ring=max(1,round(12*max(view.size)/256))
    try:
        features=feature_row(view,mask,ring)
        error=None
    except ValueError as exc:
        features=None;error=str(exc)
    return dict(features=features,feature_error=error,colour_method='cielab-v2-512-scaled-ring',
                colour_size=list(view.size),ring_pixels=ring,
                colour_scope='Predicted wound and surrounding ring; raw appearance, not diagnosis')

def palette(values):
    stops=np.array([[36,85,161],[68,174,190],[245,218,107],[204,47,62]],dtype=float)
    x=np.clip(values,0,1)*3
    low=np.minimum(x.astype(int),2);mix=(x-low)[...,None]
    return (stops[low]*(1-mix)+stops[low+1]*mix).astype(np.uint8)

def overlay(image, values):
    rgb=np.asarray(image.convert('RGB'),dtype=float)
    colours=palette(values)
    return Image.fromarray(np.clip(.4*rgb+.6*colours,0,255).astype(np.uint8))

def colour_heatmap(image):
    view=image.copy();view.thumbnail((768,768),Image.Resampling.BICUBIC)
    lab=rgb_to_cielab(view)
    # Fixed scale across photos, not per-image min/max normalization.
    return overlay(view,(lab[...,1]+20)/80)

def gradcam(image,classifier,checkpoint):
    """Elevated raw logit, layer4, gradient spatial mean + weighted ReLU.

    Hook is removed even on failure. autograd.grad does not mutate parameter
    gradients. The caller serializes access to the shared classifier.
    """
    activation=[]
    hook=classifier.layer4.register_forward_hook(lambda _m,_i,out:activation.append(out))
    try:
        with torch.enable_grad():
            x=image_tensor(image,checkpoint['image_size']).unsqueeze(0).to(device()).requires_grad_(True)
            logits=classifier(x)
            gradients=torch.autograd.grad(logits[0,1],activation[0])[0]
            cam=torch.relu((gradients.mean((2,3),keepdim=True)*activation[0]).sum(1))[0]
            maximum=float(cam.max().detach())
            if not np.isfinite(maximum) or maximum<=1e-12:
                raise ValueError('No positive Elevated-class Grad-CAM activation for this photograph')
            values=(cam/maximum).detach().cpu().numpy()
        view=image.copy();view.thumbnail((768,768),Image.Resampling.BICUBIC)
        values=np.asarray(Image.fromarray(values).resize(view.size,Image.Resampling.BILINEAR))
        return overlay(view,values)
    finally:
        hook.remove()

def compare_features(current,previous):
    if not current.get('features') or not previous.get('features'):
        return None,'Both scans need valid wound and ring measurements.'
    if current.get('colour_method','legacy')!=previous.get('colour_method','legacy'):
        return None,'Colour methods differ. Recalculate colour for both scans before comparing.'
    if current.get('colour_segmenter_sha256')!=previous.get('colour_segmenter_sha256'):
        return None,'Segmenter versions differ. Recalculate colour for both scans with the same model.'
    return {k:current['features'][k]-previous['features'][k] for k in current['features']},None
