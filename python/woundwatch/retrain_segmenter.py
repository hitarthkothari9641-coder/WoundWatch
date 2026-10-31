"""Fine-tune a separate segmenter candidate; never overwrite deployed weights.

Uses only FUSeg training masks, selects on FUSeg validation Dice, and evaluates
the selected candidate on the external set once. No surgical-wound accuracy
claim can be made from these foot-ulcer datasets.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from .common import seed_all, device
from .config import CFG
from .inference import load_segmenter
from .train_segmentation import Pairs, loss_fn, metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--output',type=Path,default=CFG.root/'experiments'/'segmenter-retrain')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1: parser.error('epochs and batch size must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'candidate.pt').exists(): parser.error('Choose a fresh output directory to preserve previous runs')
    seed_all(CFG.seed); dev=device(); net,ck=load_segmenter()
    tr=Pairs(CFG.fuseg/'train/images',CFG.fuseg/'train/labels',True)
    va=Pairs(CFG.fuseg/'validation/images',CFG.fuseg/'validation/labels')
    ex=Pairs(CFG.fuseg_external/'images',CFG.fuseg_external/'labels')
    # Match inference bicubic preprocessing in this candidate experiment.
    for ds in (tr,va,ex): ds.resize_resampling=Image.Resampling.BICUBIC
    train=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=0)
    valid=DataLoader(va,batch_size=args.batch_size); external=DataLoader(ex,batch_size=args.batch_size)
    base_dice,base_iou=metrics(net,valid,dev)
    base_external=metrics(net,external,dev)
    best=copy.deepcopy(net.state_dict()); best_dice=base_dice; selected_epoch=0
    # Preserve encoder representation; adjust decoder with conservative LR.
    for name,p in net.named_parameters(): p.requires_grad_(name.startswith(('aspp.','low.','decoder.','head.')))
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=1e-5,weight_decay=1e-4)
    history=[]
    print(f'Baseline validation Dice={base_dice:.5f}; external Dice={base_external[0]:.5f}; train={len(tr)}, val={len(va)}',flush=True)
    for epoch in range(1,args.epochs+1):
        net.train()
        for module in net.modules():
            if isinstance(module,nn.BatchNorm2d): module.eval()
        total=0
        for x,y in train:
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(net(x.to(dev)),y.to(dev));loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1)
            opt.step();total+=float(loss.detach())*len(y)
        dice,iou=metrics(net,valid,dev)
        history.append(dict(epoch=epoch,loss=total/len(tr),validation_dice=dice,validation_iou=iou))
        print(history[-1],flush=True)
        if dice>best_dice:
            best_dice=dice;best=copy.deepcopy(net.state_dict());selected_epoch=epoch
            torch.save({**ck,'state_dict':{k:v.cpu() for k,v in best.items()},'selected_epoch':epoch},args.output/'best-so-far.pt')
        (args.output/'progress.json').write_text(json.dumps(history,indent=2))
    net.load_state_dict(best)
    ext_dice,ext_iou=metrics(net,external,dev)
    report=dict(protocol='Frozen encoder; decoder fine-tuning on FUSeg train; selection only on validation Dice',
        baseline_validation_dice=base_dice,baseline_validation_iou=base_iou,
        baseline_external_dice=base_external[0],baseline_external_iou=base_external[1],
        selected_epoch=selected_epoch,candidate_validation_dice=best_dice,
        candidate_external_dice=ext_dice,candidate_external_iou=ext_iou,
        training_images=len(tr),validation_images=len(va),external_images=len(ex),history=history,
        base_sha256=hashlib.sha256((CFG.models/'segmenter.pt').read_bytes()).hexdigest(),
        promoted=False,limitation='Foot-ulcer data; no surgical-wound masks or patient-disjoint prospective validation')
    ck.update(state_dict={k:v.cpu() for k,v in best.items()},retraining=report)
    torch.save(ck,args.output/'candidate.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
