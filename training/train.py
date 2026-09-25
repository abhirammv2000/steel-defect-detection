"""Training script. No GPU on this machine (checked with nvidia-smi before
starting), so this is scoped for a reasonable CPU training time: a
lightweight U-Net (ResNet18 encoder, ImageNet-pretrained), images resized
down to 128x800, and a bounded subset of the full ~12.5k training images.
The subset size, epoch count, and measured Dice scores for the committed
model are in models/training_run.json and README.md's Results section.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp

from app.dataset import SteelDefectDataset, load_annotations, NUM_CLASSES, TRAIN_RESIZE


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float=0.5, eps: float=1e-7) -> float:
    pred=(torch.sigmoid(pred)>threshold).float()
    intersection=(pred*target).sum(dim=(2,3))
    union=pred.sum(dim=(2,3))+target.sum(dim=(2,3))
    dice=(2*intersection+eps)/(union+eps)
    # only average over (image, class) pairs where the class actually appears or was
    # predicted, otherwise every correctly-empty prediction scores a trivial dice=1
    # and inflates the mean for classes that are genuinely rare
    present=(target.sum(dim=(2,3))>0)|(pred.sum(dim=(2,3))>0)
    if present.sum()==0:
        return 1.0
    return dice[present].mean().item()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv", default="severstal-steel-defect-detection/train.csv")
    parser.add_argument("--images-dir", default="severstal-steel-defect-detection/train_images")
    parser.add_argument("--subset-size", type=int, default=1500, help="cap on training images, see README")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/steel_defect_unet.pt")
    args=parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading annotations...")
    annotations=load_annotations(args.csv)
    if args.subset_size and args.subset_size<len(annotations):
        annotations=annotations.sample(n=args.subset_size, random_state=args.seed).reset_index(drop=True)
    print(f"Using {len(annotations)} images (of {len(load_annotations(args.csv))} total in train.csv)")

    dataset=SteelDefectDataset(annotations, args.images_dir, resize=TRAIN_RESIZE)
    val_size=int(len(dataset)*args.val_fraction)
    train_size=len(dataset)-val_size
    train_ds,val_ds=random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    print(f"Train: {train_size}  Val: {val_size}")

    train_loader=DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader=DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model=smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES)
    device=torch.device("cpu")
    model.to(device)

    # Dice + BCE is a standard combination for imbalanced binary segmentation, since
    # defects only cover a small fraction of most images. BCE alone would let the
    # model collapse to predicting "no defect" everywhere and still score well.
    bce_loss=nn.BCEWithLogitsLoss()
    def dice_loss(pred, target, eps=1e-7):
        pred=torch.sigmoid(pred)
        intersection=(pred*target).sum(dim=(2,3))
        union=pred.sum(dim=(2,3))+target.sum(dim=(2,3))
        return 1-((2*intersection+eps)/(union+eps)).mean()

    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)

    history=[]
    training_start=time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_start=time.time()
        train_loss=0.0
        for images,masks in train_loader:
            images,masks=images.to(device),masks.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=bce_loss(outputs,masks)+dice_loss(outputs,masks)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*images.size(0)
        train_loss/=train_size

        model.eval()
        val_dice_scores=[]
        with torch.no_grad():
            for images,masks in val_loader:
                images,masks=images.to(device),masks.to(device)
                outputs=model(images)
                val_dice_scores.append(dice_coefficient(outputs,masks))
        val_dice=float(np.mean(val_dice_scores)) if val_dice_scores else 0.0

        epoch_time=time.time()-epoch_start
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_dice={val_dice:.4f}  ({epoch_time:.1f}s)")
        history.append({"epoch":epoch+1,"train_loss":train_loss,"val_dice":val_dice,"epoch_seconds":epoch_time})

    total_time=time.time()-training_start
    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")

    run_info={
        "subset_size":len(annotations),"train_size":train_size,"val_size":val_size,
        "epochs":args.epochs,"batch_size":args.batch_size,"lr":args.lr,
        "resize":list(TRAIN_RESIZE),"encoder":"resnet18","total_training_seconds":total_time,
        "final_val_dice":history[-1]["val_dice"] if history else None,"history":history,
    }
    with open("models/training_run.json","w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Wrote models/training_run.json, final val_dice={run_info['final_val_dice']:.4f}, total_training_seconds={total_time:.1f}")


if __name__=="__main__":
    main()
