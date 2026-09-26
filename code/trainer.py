"""Training loop for SAM-CRN."""

import torch

from config import DEVICE, CKPT_PATH


def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(DEVICE)
        out = model(batch["ehr"], batch["iot"], batch["physio"], batch["lab"])
        loss, _ = criterion(out, batch)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += loss.item() * batch["x"].size(0)
        n += batch["x"].size(0)
    return total / n


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    best_val = float("inf")
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, train=True)
        va = run_epoch(model, val_loader,   criterion, None,      train=False)
        scheduler.step()
        flag = ""
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), CKPT_PATH)
            flag = "  *saved*"
        print(f"Epoch {ep:02d} | train_loss={tr:.4f} | val_loss={va:.4f}{flag}")

    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    return model