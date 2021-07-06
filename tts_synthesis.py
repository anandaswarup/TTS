"""TTS synthesis"""

import argparse
import os

import soundfile as sf
import torch

import config.config as cfg
from tts.model import Tacotron
from vocoder.model import WaveRNN

if cfg.text_processor == "en":
    from text.en.processor import load_cmudict, symbol_to_id, text_to_sequence
else:
    raise NotImplementedError


def _load_synthesis_instances(filename):
    """Load the synthesis instances from file
    """
    with open(filename, "r") as file_reader:
        synthesis_instances = file_reader.readlines()

    synthesis_instances = [
        instance.strip("\n") for instance in synthesis_instances
    ]

    synthesis_instances = [
        instance.split("|") for instance in synthesis_instances
    ]

    return synthesis_instances


def synthesize_all(synthesis_instances, seq2seq_checkpoint_path,
                   vocoder_checkpoint_path, out_dir):
    """Synthesize all utterances present in the synthesis file
    """
    os.makedirs(out_dir, exist_ok=True)

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate TTS (Tacotron) model
    num_chars = len(symbol_to_id)

    tts_model = Tacotron(num_chars=num_chars)
    tts_model = tts_model.to(device)
    tts_model.eval()

    # Restore the TTS model to specified training checkpoint
    tts_checkpoint = torch.load(seq2seq_checkpoint_path,
                                map_location=lambda storage, loc: storage)
    tts_model.load_state_dict(tts_checkpoint["model"])
    tts_step = tts_checkpoint["step"]

    # Instantiate vocoder model
    vocoder_model = WaveRNN(
        n_mels=cfg.audio["n_mels"],
        hop_length=cfg.audio["hop_length"],
        num_bits=cfg.audio["n_bits"],
        audio_embedding_dim=cfg.vocoder_model["audio_embedding_dim"],
        conditioning_rnn_size=cfg.vocoder_model["conditioning_rnn_size"],
        rnn_size=cfg.vocoder_model["rnn_size"],
        fc_size=cfg.vocoder_model["fc_size"])
    vocoder_model = vocoder_model.to(device)
    vocoder_model.eval()

    # Restore the vocoder model to specified training checkpoint
    vocoder_checkpoint = torch.load(vocoder_checkpoint_path,
                                    map_location=lambda storage, loc: storage)
    vocoder_model.load_state_dict(vocoder_checkpoint["model"])
    vocoder_step = vocoder_checkpoint["step"]

    cmudict = load_cmudict()

    # Generate waveforms for all synthesis instances
    for fileid, text in synthesis_instances:
        print(f"Synthesizing text for: {fileid}", flush=True)

        text = torch.LongTensor(text_to_sequence(text, cmudict)).unsqueeze(0)
        text = text.to(device)

        # Synthesize audio
        with torch.no_grad():
            mel_hat, _ = tts_model.generate(text)
            mel_hat = mel_hat.transpose(1, 2).contiguous()
            wav_hat = vocoder_model.generate(mel_hat)

        # Write the generated wavform to disk
        out_path = os.path.join(
            out_dir,
            f"tts_model_step{tts_step:09d}_vocoder_model_step{vocoder_step:09d}_{fileid}.wav"
        )
        sf.write(out_path, wav_hat, cfg.audio["sampling_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text to Speech Synthesis")

    parser.add_argument(
        "--synthesis_file",
        help="Path to the file containing text to be synthesized",
        required=True)

    parser.add_argument(
        "--seq2seq_checkpoint",
        help="Path to the trained seq2seq model to use for synthesis",
        required=True)

    parser.add_argument(
        "--vocoder_checkpoint",
        help="Path to the trained vocoder model to use for synthesis",
        required=True)

    parser.add_argument(
        "--out_dir",
        help="Path to where synthesized waveforms will be saved to disk",
        required=True)

    args = parser.parse_args()

    synthesis_file = args.synthesis_file
    seq2seq_checkpoint_path = args.seq2seq_checkpoint
    vocoder_checkpoint_path = args.vocoder_checkpoint
    out_dir = args.out_dir

    synthesis_instances = _load_synthesis_instances(synthesis_file)

    synthesize_all(synthesis_instances, seq2seq_checkpoint_path,
                   vocoder_checkpoint_path, out_dir)
