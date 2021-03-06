"""Dataset preprocessing"""
import argparse
import os

import librosa
import numpy as np

import config.config as cfg


def _split_dataset(items):
    """Split the dataset items into train/eval splits
    """
    eval_split_size = min(500, int(len(items) * 0.01))

    np.random.seed(0)
    np.random.shuffle(items)

    return items[eval_split_size:], items[:eval_split_size]


def _compute_melspectrogram(wav):
    """Compute the mel-spectrogram
    """
    # Apply pre-emphasis
    wav = librosa.effects.preemphasis(wav, coef=0.97)

    # Compute the mel spectrogram
    mel = librosa.feature.melspectrogram(y=wav,
                                         sr=cfg.audio["sampling_rate"],
                                         hop_length=cfg.audio["hop_length"],
                                         win_length=cfg.audio["win_length"],
                                         n_fft=cfg.audio["n_fft"],
                                         n_mels=cfg.audio["n_mels"],
                                         fmin=cfg.audio["fmin"],
                                         norm=1,
                                         power=1)

    # Convert to log scale
    mel = librosa.core.amplitude_to_db(mel, top_db=None) - cfg.audio["ref_db"]

    # Normalize
    mel = np.maximum(mel, -cfg.audio["max_db"])
    mel = mel / cfg.audio["max_db"]

    return mel


def _mulaw_compression(wav):
    """Compress the waveform using mu-law compression
    """
    wav = np.pad(wav, (cfg.audio["win_length"] // 2, ), mode="reflect")
    wav = wav[:(
        (wav.shape[0] - cfg.audio["win_length"]) // cfg.audio["hop_length"] +
        1) * cfg.audio["hop_length"]]

    wav = 2**(cfg.audio["n_bits"] - 1) + librosa.mu_compress(
        wav, mu=2**cfg.audio["n_bits"] - 1)

    return wav


def _process_utterance(wav, filename, mel_dir, qwav_dir):
    """Process a single wav file
    This writes the mel spectrogram as well as the quantized wav to disk and returns a tuple to write to the
    metadata.csv file
    """
    peak = np.abs(wav).max()
    if peak >= 1:
        wav = wav / peak * 0.999

    # Compute mel spectrogram
    mel = _compute_melspectrogram(wav)

    # Quantize the wavform
    qwav = _mulaw_compression(wav)

    # Save to disk
    mel_path = os.path.join(mel_dir, filename + ".npy")
    qwav_path = os.path.join(qwav_dir, filename + ".npy")
    np.save(mel_path, mel)
    np.save(qwav_path, qwav)

    return mel.shape[-1]


def write_metadata(metadata, out_file):
    """Write the metadata to train.txt file
    """
    with open(out_file, "w") as file_writer:
        for m in metadata:
            file_writer.write("|".join([str(x) for x in m]) + "\n")


def preprocess_dataset(root_dir, out_dir):
    """Process dataset and write processed dataset to disk
    """
    #  Load dataset items from disk
    items = []
    with open(os.path.join(root_dir, "metadata.csv"), "r") as file_reader:
        for line in file_reader:
            parts = line.strip().split("|")
            text = parts[1]
            wav_path = os.path.join(root_dir, "wavs", f"{parts[0]}.wav")
            items.append([text, wav_path])

    # Split into train and eval sets
    train_items, eval_items = _split_dataset(items)

    # Process the train split
    print("Processing train split")

    mel_dir = os.path.join(out_dir, "train", "mel")
    qwav_dir = os.path.join(out_dir, "train", "qwav")

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(qwav_dir, exist_ok=True)

    train_metadata = []
    for text, wav_path in train_items:
        # Get filename of file being processed
        filename = os.path.splitext(os.path.basename(wav_path))[0]

        # Load wav file from disk
        wav, _ = librosa.load(wav_path, sr=cfg.audio["sampling_rate"])

        # Get length of wav file in seconds
        wav_length = len(wav) / cfg.audio["sampling_rate"]

        # Process only those wav files whose length is less than 10 seconds (this is to reduce the computational
        # requirements of seq2seq processing using RNNs)
        if wav_length <= 10:
            num_frames = _process_utterance(wav, filename, mel_dir, qwav_dir)
            train_metadata.append((filename, text, num_frames))
    write_metadata(train_metadata, os.path.join(out_dir, "train/train.csv"))

    # Process the eval split
    print("Processing eval split")

    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    eval_metadata = []
    for text, wav_path in eval_items:
        filename = os.path.splitext(os.path.basename(wav_path))[0]
        eval_metadata.append((filename, text))
    write_metadata(eval_metadata, os.path.join(out_dir, "eval/heldout.csv"))


def preprocess(root_dir, out_dir):
    """Preprocess the dataset
    """
    os.makedirs(out_dir, exist_ok=True)

    if cfg.dataset == "LJSpeech":
        preprocess_dataset(root_dir, out_dir)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset and split into train and eval splits")

    parser.add_argument("--dataset_dir",
                        help="Path to the root of the downloaded dataset",
                        required=True)

    parser.add_argument("--out_dir",
                        help="Output path to write the processed dataset",
                        required=True)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    out_dir = args.out_dir

    preprocess(dataset_dir, out_dir)
