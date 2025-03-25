import streamlit as st
import os
import io
import librosa
import torch
import torchaudio
import numpy as np
import plotly.express as px
from torchaudio.functional import resample
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import torch.nn.functional as F


def load_audio(audiopath, sampling_rate=22000):
    """
    Loads an audio file, resamples it, and ensures proper format for processing.

    Args:
        audiopath (str or io.BytesIO): Path to the audio file or an in-memory file object.
        sampling_rate (int): Target sampling rate.

    Returns:
        torch.Tensor: Processed audio tensor.
    """
    try:
        if isinstance(audiopath, str):  # File path input
            if audiopath.endswith('.mp3'):
                audio, lsr = librosa.load(audiopath, sr=sampling_rate)
                audio = torch.FloatTensor(audio)  # Convert to tensor
            else:
                raise ValueError(f"Unsupported audio format provided: {audiopath[-4:]}")
        elif isinstance(audiopath, io.BytesIO):  # If in-memory file
            audio, lsr = torchaudio.load(audiopath)
            audio = audio[0]  # Remove channel data if stereo
        else:
            raise TypeError("Invalid input type. Expected file path or BytesIO object.")

        # Ensure correct sampling rate
        if lsr != sampling_rate:
            audio = resample(audio, lsr, sampling_rate)

        # Normalize audio
        if torch.any(audio > 2) or torch.any(audio < -1):
            print(f"Warning: Audio data out of range. Max={audio.max()} min={audio.min()}")
            audio = torch.clamp(audio, -1, 1)

        return audio.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None


# CLASSIFY AUDIO CLIP FUNCTION
def classify_audio_clip(clip):
    """
    Determines whether an audio clip is AI-generated.

    :param clip: torch tensor containing the audio waveform data.
    :return: The probability of the audio clip being AI-generated.
    """

    # Initialize classifier model
    classifier = AudioMiniEncoderWithClassifierHead(
        num_classes=2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
        resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
        dropout=0, kernel_size=5, distribute_zero_label=False
    )

    # Load pretrained model weights
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)

    # Process the audio clip
    clip = clip.cpu().unsqueeze(0)  # Move to CPU and add batch dimension

    # Perform classification and return AI-generated probability
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0].item()  # Convert tensor to float


st.set_page_config(layout="wide")


def main():
    st.title("DeepReson AI Detection Tool")
    st.write("This tool can help you determine whether an audio clip is AI-generated.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "flac"])

    if uploaded_file is not None:
        if st.button("Analyse Audio"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("Your Results are below")

                # Load audio file
                audio_clip = load_audio(uploaded_file)
                if audio_clip is None:
                    st.error("Error processing audio file. Please upload a valid file.")
                    return

                result = classify_audio_clip(audio_clip)
                st.info(f"AI-Generated Probability: {result:.2f}")
                st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI-generated.")

            with col2:
                st.info("Uploaded audio is below")
                st.audio(uploaded_file)

                # Display audio waveform
                fig = px.line()
                fig.add_scatter(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
                fig.update_layout(
                    title="Audio Waveform",
                    xaxis_title="Time",
                    yaxis_title="Amplitude"
                )

                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.info("Disclaimer")
                st.warning("This tool is for educational purposes only. The results are not 100% accurate.")


if __name__ == "__main__":
    main()
