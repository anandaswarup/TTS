"""Train the WaveRNN model"""

import argparse
import os

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from wavernn.data_utils import WaveRNNDataset
from wavernn.model import WaveRNNModel


def save_checkpoint(checkpoint_dir, model, optimizer, scaler, scheduler, step):
    """Save checkpoint to disk
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
    """Load checkpoint from the specified path
    """
    print(f"Loading checkpoint: {checkpoint_path} from disk")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["step"]


def prepare_dataloaders(data_dir):
    """Prepare the dataloaders
    """
    train_dataset = WaveRNNDataset(os.path.join(data_dir, "train"))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.vocoder_training["batch_size"],
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = WaveRNNDataset(os.path.join(data_dir, "val"))
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=8, shuffle=False, num_workers=1, pin_memory=False, drop_last=True
    )

    return train_dataloader, val_dataloader


def validate(model, device, val_dataloader):
    """Validate the model
    """
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for idx, (mels, qwavs) in enumerate(val_dataloader, 1):
            mels, qwavs = mels.to(device), qwavs.to(device)

            wav_hat = model(qwavs[:, :-1], mels)
            loss = F.cross_entropy(wav_hat.transpose(1, 2).contiguous(), qwavs[:, 1:])

            val_loss += loss.item()
        val_loss = val_loss / (idx + 1)
    model.train()

    print(f"Val Loss: {val_loss:.6f}", flush=True)


def train_model(data_dir, checkpoint_dir, resume_checkpoint_path=None):
    """Train the model
    """
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Specify the device on which to perform the training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = WaveRNNModel()
    model = model.to(device)

    # Instantiate the optimizer, scaler (for mixed precision training) and scheduler (for learning rate decay)
    optimizer = optim.Adam(model.parameters(), lr=cfg.vocoder_training["lr"])
    scaler = amp.GradScaler()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=cfg.vocoder_training["lr_scheduler_step_size"],
        gamma=cfg.vocoder_training["lr_scheduler_gamma"],
    )

    # Instantiate the dataloader
    train_dataloader, val_dataloader = prepare_dataloaders(data_dir)

    # Load checkpoint and resume training from that point (if specified)
    if resume_checkpoint_path is not None:
        global_step = load_checkpoint(resume_checkpoint_path, model, optimizer, scaler, scheduler)
    else:
        global_step = 0

    n_epochs = cfg.vocoder_training["n_steps"] // len(train_dataloader) + 1
    start_epoch = global_step // len(train_dataloader) + 1

    model.train()

    # Main training loop
    for epoch in range(start_epoch, n_epochs + 1):
        print(f"Epoch: {epoch}", flush=True)
        for _, (mels, qwavs) in enumerate(train_dataloader, 1):
            mels, qwavs = mels.to(device), qwavs.to(device)

            model.zero_grad()

            # Forward pass and loss computation
            with amp.autocast():
                wav_hat = model(qwavs[:, :-1], mels)
                loss = F.cross_entropy(wav_hat.transpose(1, 2).contiguous(), qwavs[:, 1:])

            # Gradient computation
            scaler.scale(loss).backward()

            # Weights update
            scaler.step(optimizer)

            # Scaler state update
            scaler.update()

            # lr scheduler state update
            scheduler.step()

            global_step += 1

            print(
                f"Step: {global_step}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()}", flush=True,
            )

            if global_step % cfg.vocoder_training["checkpoint_interval"] == 0:
                validate(model, device, val_dataloader)
                save_checkpoint(checkpoint_dir, model, optimizer, scaler, scheduler, global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the vocoder model")

    parser.add_argument("--data_dir", help="Path to processed dataset to be used to train the model", required=True)

    parser.add_argument(
        "--checkpoint_dir", help="Path to location where training checkpoints will be saved", required=True
    )

    parser.add_argument(
        "--resume_checkpoint_path", help="If specified load checkpoint and resume training from that point"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    resume_checkpoint_path = args.resume_checkpoint_path

    train_model(data_dir, checkpoint_dir, resume_checkpoint_path)
