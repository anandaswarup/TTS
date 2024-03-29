# Recurrent Neural Network based Text-to-Speech systems

This repository contains code to train a End-to-End Speech Synthesis system, based on the Tacotron2 model with modifications as described in [Location Relative Attention Mechanisms for Robust Long-Form Speech Synthesis](https://arxiv.org/pdf/1910.10288.pdf).
 
The system consists of two parts:

1. A Tacotron2 model with Dynamic Convolutional Attention which modifies the hybrid location sensitive attention mechanism to be purely location based, resulting in better generalization on long utterances. This model takes text (in the form of character sequence) as input and predicts a sequence of mel-spectrogram frames as output (the seq2seq model).

2. A WaveRNN based vocoder; which takes the mel-spectrogram predicted in the previous step as input and generates a waveform as output (the vocoder model).

All audio processing parameters, model hyperparameters, training configuration etc are specified in `config.py`. 

# Quick start
## Train TTS from scratch
1. Download and extract the dataset (The assumption made is that the dataset to be used for training is in the same format as the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/))

2. Preprocess the downloaded dataset; perform feature extraction on the wav files and create train/val/eval splits

    ```python
    python preprocess.py \
            --dataset_dir <Path to the root of the downloaded dataset> \
            --out_dir <Output path to write the processed dataset>
    ```

3. Train the Tacotron2 model

    ```python
    python train_Tacotron2.py \
            --data_dir <Path to the processed dataset to be used to train the model> \
            --checkpoint_dir <Path to location where training checkpoints will be saved> \
            --alignments_dir <Path to the location where training alignments will be saved> \
            --resume_checkpoint_path <If specified load checkpoint and resume training>
    ```

4. Train the WaveRNN model

    ```python
    python train_WaveRNN.py \
            --data_dir <Path to the processed dataset to be used to train the model> \
            --checkpoint_dir <Path to location where training checkpoints will be saved> \
            --resume_checkpoint_path <If specified load checkpoint and resume training>
    ```

## Synthesize using a trained TTS model
1. Prepare the text to be synthesized
 
    The text to be synthesized should be placed in the `synthesis.txt` file which has the following format

    ```
    <TEXT_ID_1> TEXT_1
    <TEXT_ID_2> TEXT_2
    .
    .
    .
    ```

2. Text to speech synthesis
 
    ```python
    python tts_Synthesis.py \
            --synthesis_file <Path to the synthesis.txt file (created in Step 1)> \
            --Tacotron2_checkpoint <Path to the trained Tacotron2 model to use for synthesis> \
            --WaveRNN_checkpoint <Path to the trained WaveRNN model to use for synthesis> \
            --out_dir <Path to where the synthesized waveforms will be written to disk>
    ```
## Acknowledgements

This code is based on the code in the following repositories
1. [bshall/Tacotron](https://github.com/bshall/Tacotron)
2. [mozilla/TTS](https://github.com/mozilla/TTS)
3. [NVIDIA/Tacotron2](https://github.com/NVIDIA/tacotron2)

## References

1. [Location Relative Attention Mechanisms for Robust Long-Form Speech Synthesis](https://arxiv.org/pdf/1910.10288.pdf)
2. [Tacotron: Towards End-To-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf)
3. [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf)
