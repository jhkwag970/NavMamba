import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import MultiKITTIOxtsDataset
from models import NavLSTM, NavTR, NavMamba

import os
from time import time
from tqdm import tqdm
from tensorboardX import SummaryWriter


# ------------------------------------------------------------
# 1. Argument Parser
# ------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train GPS-INS Transformer/LSTM on KITTI OXTS")

    # data
    parser.add_argument("--root", type=str, required=True,
                        help="Path to KITTI raw root")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # model
    parser.add_argument("--model", type=str, choices=["NavTR", "NavMamba", "NavLSTM"], default="NavTR")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--num_layers", type=int, default=4)

    # training
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--val_interval", type=int, default=1)
    return parser.parse_args()


# ------------------------------------------------------------
# 2. Training / Validation Epochs
# ------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None, device="cuda", train=True):
    model.train() if train else model.eval()
    epoch_loss, n_samples = 0.0, 0
    total_sq_err, total_pts = 0.0, 0

    start_time = time()
    bar = tqdm(loader, ncols=100, leave=False)
    bar.set_description("Train" if train else "Val")

    for batch in bar:
        imu_seq, gnss_seq, target, scale = batch
        imu_seq, gnss_seq, target, scale = imu_seq.to(device), gnss_seq.to(device), target.to(device), scale.to(device)

        with torch.set_grad_enabled(train):
            pred = model(imu_seq, gnss_seq)
            pred_m = pred * scale.view(-1, 1, 1)
            tgt_m = target * scale.view(-1, 1, 1)

            loss = criterion(pred_m, tgt_m)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = imu_seq.size(0)
        epoch_loss += loss.item() * bs
        n_samples += bs

        if not train:
            total_sq_err += torch.sum((pred_m - tgt_m) ** 2).item()
            total_pts += pred.numel() / 3

        avg_loss = epoch_loss / n_samples
        elapsed = time() - start_time
        speed = n_samples / max(elapsed, 1e-6)
        rem = (len(loader.dataset) - n_samples) / max(speed, 1e-6)
        bar.set_postfix(loss=f"{avg_loss:.4f}", eta=f"{rem/60:.1f}m")

    bar.close()
    if not train:
        rmse_m = (total_sq_err / total_pts) ** 0.5
        return epoch_loss / len(loader.dataset), rmse_m
    return epoch_loss / len(loader.dataset), None


# ------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.save_dir)
    print(f"TensorBoard logs: {args.save_dir}")

    # Datasets
    train_set = MultiKITTIOxtsDataset(args.root, seq_len=args.seq_len, stride=args.stride, split="train")
    val_set = MultiKITTIOxtsDataset(args.root, seq_len=args.seq_len, stride=args.stride, split="val")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    # Model
    if args.model == "NavTR":
        model = NavTR(hidden_size=args.hidden_dim, nhead=args.nhead, mlp_ratio=args.mlp_ratio, num_layers=args.num_layers, max_seq_len=args.seq_len).to(args.device)
    elif args.model == "NavMamba":
        model = NavMamba(hidden_size=args.hidden_dim, nhead=args.nhead, mlp_ratio=args.mlp_ratio, num_layers=args.num_layers, max_seq_len=args.seq_len).to(args.device)
    elif args.model == "NavLSTM":
        model = NavLSTM(hidden_size=args.hidden_dim, num_layers=args.num_layers).to(args.device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_rmse = float("inf")
    best_epoch = -1
    last_checkpoint = None  
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, args.device, train=True)
        val_loss, val_rmse = None, None

        if epoch % args.val_interval == 0:
            val_loss, val_rmse = run_epoch(model, val_loader, criterion, optimizer=None, device=args.device, train=False)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("RMSE/val_m", val_rmse, epoch)

        writer.add_scalar("Loss/train", train_loss, epoch)

        if val_loss:
            print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f} | RMSE {val_rmse:.3f} m")
        else:
            print(f"Epoch {epoch:03d} | Train {train_loss:.6f}")

        if val_rmse is not None and val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            save_path = os.path.join(args.save_dir, f"best_{args.model}_epoch{epoch:03d}_rmse{val_rmse:.3f}.pth")

            if last_checkpoint is not None and os.path.exists(last_checkpoint):
                try:
                    os.remove(last_checkpoint)
                except Exception as e:
                    print(f"Warning: could not delete old checkpoint ({e})")

            torch.save(model.state_dict(), save_path)
            last_checkpoint = save_path

    writer.close()
    print(f"Training finished. Best RMSE = {best_rmse:.3f} m @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
