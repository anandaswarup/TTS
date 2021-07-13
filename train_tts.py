"""Tacotron training"""

import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.sampler as samplers
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import config.config as cfg
from tts.dataset import BucketBatchSampler, TTSDataset, collate
from tts.model import Tacotron

if cfg.text_processor == "en":
    from text.en.processor import _symbol_to_id
else:
    raise NotImplementedError


def save_checkpoint(checkpoint_dir, model, optimizer, scaler, scheduler, step):
    """Write checkpoint to disk
    """
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"model_step{step:09d}.pth")

    torch.save(checkpoint_state, checkpoint_path)
    print(f"Written checkpoint: {checkpoint_path} to disk")


def load_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler):
    """Load the checkpoint from the disk
    """
    print(f"Loading checkpoint: {checkpoint_path} from disk")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["step"]


def log_alignment(alignment, path):
    """Save the alignment to disk
    """
    _ = plt.figure(figsize=(10, 6))
    plt.imshow(alignment, vmin=0, vmax=0.6, origin="lower")
    plt.xlabel("Decoder steps")
    plt.ylabel("Encoder steps")

    plt.savefig(path)


def train_model(data_dir, checkpoint_dir, alignments_dir,
                resume_checkpoint_path):
    """Train the model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(alignments_dir, exist_ok=True)

    # Specify the device to train on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    num_chars = len(_symbol_to_id)

    model = Tacotron(num_chars=num_chars)
    model = model.to(device)
    model.train()

    # Instantiate the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.vocoder_training["learning_rate"])
    scaler = amp.GradScaler()
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=cfg.tts_training["lr_scheduler_milestones"],
        gamma=cfg.tts_training["lr_scheduler_gamma"])

    if resume_checkpoint_path is not None:
        global_step = load_checkpoint(resume_checkpoint_path, model, optimizer,
                                      scaler, scheduler)
    else:
        global_step = 0

    # Instantiate the dataloader
    dataset = TTSDataset(data_dir)
    sampler = samplers.RandomSampler(dataset)
    batch_sampler = BucketBatchSampler(
        sampler=sampler,
        batch_size=cfg.tts_training["batch_size"],
        drop_last=True,
        sort_key=dataset.sort_key,
        bucket_size_multiplier=cfg.tts_training["bucket_size_multiplier"])
    collate_fn = partial(
        collate, reduction_factor=cfg.tts_model["decoder"]["reduction_factor"])
    loader = DataLoader(dataset,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn,
                        num_workers=cfg.tts_training["num_workers"],
                        pin_memory=True)

    num_epochs = cfg.tts_training["num_steps"] // len(loader) + 1
    start_epoch = global_step // len(loader) + 1

    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss = 0

        for idx, (texts, text_lengths, mels,
                  mel_lengths) in enumerate(loader, 1):

            texts, mels = texts.to(device), mels.to(device)
            optimizer.zero_grad()

            with amp.autocast():
                ys, alignments = model(texts, mels)
                loss = F.l1_loss(ys[:, :, :mels.size(-1)], mels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(),
                            cfg.tts_training["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            avg_loss += (loss.item() - avg_loss) / idx

            if global_step % cfg.tts_training["checkpoint_interval"] == 0:
                # Save checkpoint
                save_checkpoint(checkpoint_dir, model, optimizer, scaler,
                                scheduler, global_step)

                # Save alignment state
                index = 0
                alignment = alignments[
                    index, :text_lengths[index], :mel_lengths[index] //
                    cfg.tts_model["decoder"]["reduction_factor"]]

                alignment = alignment.detach().cpu().numpy()

                alignment_path = os.path.join(
                    alignments_dir, f"model_step{global_step:09d}.png")

                log_alignment(alignment, alignment_path)

        print(
            f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Current lr: {scheduler.get_last_lr()}",
            flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TTS model")

    parser.add_argument(
        "--data_dir",
        help="Path to processed dataset to be used to train the model",
        required=True)

    parser.add_argument(
        "--checkpoint_dir",
        help="Path to location where training checkpoints will be saved",
        required=True)

    parser.add_argument(
        "--alignments_dir",
        help="Path to location where training alignments will be saved",
        required=True)

    parser.add_argument(
        "--resume_checkpoint_path",
        help="If specified load checkpoint and resume training from that point"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    alignments_dir = args.alignments_dir
    resume_checkpoint_path = args.resume_checkpoint_path

    train_model(data_dir, checkpoint_dir, alignments_dir,
                resume_checkpoint_path)
