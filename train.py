import torch
import torch.nn as nn
import os
import sys

import argparse

from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed,
)
from models import ClipCaptionModel,ClipCaptionPrefix
from datasets_clip import ClipDataset






def training_img_caption(config):
    os.makedirs("./checkpoints", exist_ok=True)
    set_seed(config["seed"])
    accelerator = Accelerator()
    accelerator.print(config)

    epochs = config["epochs"]
    output_prefix = "nmt"
    output_dir = "checkpoints"
    prefix_length = config["prefix_length"]

    train_dataset = ConcatDataset(
        (
            ClipDataset("./text_b16.pt", prefix_length),
            ClipDataset("./train_img_b16.pt", prefix_length),
            ClipDataset("./train_img_b16.pt", prefix_length),
            ClipDataset("./train_img_b16.pt", prefix_length),
        )
    )

    val_dataset = ClipDataset("./val_img_b16.pt", prefix_length)

    accelerator.print(len(train_dataset), len(val_dataset))
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )

    accelerator.print("Loading models")
    model = ClipCaptionModel(prefix_length)
    model = model.to(accelerator.device)
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5000,
        num_training_steps=epochs * len(train_dataloader),
    )

    for epoch in range(epochs):
        accelerator.print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(
            total=len(train_dataloader),
            desc=output_prefix,
            disable=not accelerator.is_local_main_process,
        )
        # Train phase
        model.train()
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, prefix_length - 1 : -1]
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0
            )
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress.set_postfix({"loss": loss.item()})
            progress.update()

        progress.close()
        if epoch % config["save_every"] == 0 or epoch == epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(
                unwrapped_model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

        val_loss = []

        if epoch % config["val_every"] == 0 or epoch == epochs - 1:
            accelerator.print("Running evaluate")
            model.eval()
            for step, (tokens, mask, prefix) in enumerate(val_dataloader):
                with torch.no_grad():
                    outputs = model(tokens, prefix, mask)
                    logits = outputs.logits[:, prefix_length - 1 : -1]
                    loss = nnf.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        tokens.flatten(),
                        ignore_index=0,
                    )
                    val_loss.append(accelerator.gather(loss))
            accelerator.print(epoch, ">>>>>>>>", torch.cat(val_loss).mean())
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64, help="Batchsize, 64 for V38, 32 for V28, and 16 for P100")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=1)
    args = parser.parse_args()


    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "prefix_length": 10,
        "save_every": args.save_every,
        "val_every": args.val_every,
    }

    training_img_caption(config)


if __name__ == "__main__":
    main()
