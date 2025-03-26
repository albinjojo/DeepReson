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
            audio, lsr = torchaudio.load(audiopath)
            audio = audio.mean(dim=0)  # Convert stereo to mono
        elif isinstance(audiopath, io.BytesIO):  # If in-memory file
            audiopath.seek(0)  # Reset pointer
            audio, lsr = torchaudio.load(audiopath)
            audio = audio.mean(dim=0)  # Convert stereo to mono
        else:
            raise TypeError("Invalid input type. Expected file path or BytesIO object.")

        # Ensure correct sampling rate
        if lsr != sampling_rate:
            audio = resample(audio, lsr, sampling_rate)

        # Normalize audio
        audio = audio / torch.max(torch.abs(audio))

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
    try:
        # Initialize classifier model
        classifier = AudioMiniEncoderWithClassifierHead(
            classes=2,  # Added the required classes argument
            spec_dim=1, 
            embedding_dim=512, 
            depth=5, 
            downsample_factor=4,
            resnet_blocks=2, 
            attn_blocks=4, 
            num_attn_heads=4, 
            base_channels=32,
            dropout=0, 
            kernel_size=5, 
            distribute_zero_label=False
        )

        # Load pretrained model weights
        if not os.path.exists('classifier.pth'):
            raise FileNotFoundError("Pretrained model file 'classifier.pth' not found.")
        state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
        classifier.load_state_dict(state_dict)

        # Process the audio clip
        clip = clip.cpu().unsqueeze(0)  # Move to CPU and add batch dimension

        # Perform classification and return AI-generated probability
        results = F.softmax(classifier(clip), dim=-1)
        return results[0][0].item()  # Convert tensor to float
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")  # Display error in Streamlit
        print(f"Error during classification: {str(e)}")  # Print error in console
        return None


st.set_page_config(
    layout="wide",
    page_title="DeepReson AI Detection",
    page_icon="🎵",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    .stButton>button {
        width: 100%;
        background: rgba(76, 175, 80, 0.8);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 500;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background: rgba(76, 175, 80, 1);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stInfo {
        background: rgba(232, 245, 233, 0.7);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid rgba(76, 175, 80, 0.8);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stWarning {
        background: rgba(255, 243, 224, 0.7);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid rgba(255, 152, 0, 0.8);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stError {
        background: rgba(255, 235, 238, 0.7);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid rgba(244, 67, 54, 0.8);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stSuccess {
        background: rgba(232, 245, 233, 0.7);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid rgba(76, 175, 80, 0.8);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    h1 {
        color: #1a237e;
        text-align: center;
        padding: 20px;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stMarkdown {
        text-align: center;
        color: #424242;
    }
    .stFileUploader {
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stAudio {
        background: rgba(255, 255, 255, 0.7);
        padding: 15px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .plotly-graph-div {
        background: rgba(255, 255, 255, 0.7) !important;
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .tips-container {
        background: rgba(227, 242, 253, 0.7);
        padding: 20px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .tips-container h4 {
        color: #1a237e;
        margin-bottom: 15px;
    }
    .tips-container ul {
        list-style-type: none;
        padding-left: 0;
    }
    .tips-container li {
        margin: 10px 0;
        padding-left: 25px;
        position: relative;
    }
    .tips-container li:before {
        content: "•";
        color: #4CAF50;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🎵 DeepReson AI Detection Tool")
    st.markdown("""
        <div style='text-align: center; color: #424242; margin-bottom: 30px;'>
            <h3>Detect AI-Generated Audio with Confidence</h3>
            <p>Upload your audio file to analyze whether it was generated by AI</p>
        </div>
    """, unsafe_allow_html=True)

    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "📁 Upload an audio file",
        type=["mp3", "wav", "flac"],
        help="Supported formats: MP3, WAV, FLAC"
    )

    if uploaded_file is not None:
        if st.button("🔍 Analyze Audio"):
            col1, col2, col3 = st.columns([1, 1.5, 1])

            with col1:
                st.markdown("### 📊 Analysis Results")
                st.info("Processing your audio file...")

                # Load audio file
                audio_clip = load_audio(uploaded_file)
                if audio_clip is None:
                    st.error("❌ Error processing audio file. Please upload a valid file.")
                    return

                result = classify_audio_clip(audio_clip)
                if result is not None:
                    st.markdown("### 📈 Results")
                    st.info(f"AI-Generated Probability: {result:.2f}")
                    st.success(f"🎯 The uploaded audio is {result * 100:.2f}% likely to be AI-generated.")
                else:
                    st.error("❌ An error occurred during audio classification.")

            with col2:
                st.markdown("### 🎵 Audio Preview")
                st.info("Listen to your uploaded audio")
                st.audio(uploaded_file)

                # Display audio waveform with enhanced styling
                normalized_audio = audio_clip / torch.max(torch.abs(audio_clip))
                fig = px.line()
                fig.add_scatter(x=list(range(len(normalized_audio.squeeze()))), 
                              y=normalized_audio.squeeze().numpy(),
                              line=dict(color='#4CAF50'))
                fig.update_layout(
                    title="Audio Waveform",
                    xaxis_title="Time",
                    yaxis_title="Amplitude",
                    template="plotly_white",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.markdown("### ℹ️ Information")
                st.info("About the Analysis")
                st.warning("""
                    ⚠️ **Disclaimer**  
                    This tool is for educational purposes only.  
                    The results are not 100% accurate and should not be used as the sole basis for any decisions.
                """)
                st.markdown("""
                    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 5px;'>
                        <h4>💡 Tips</h4>
                        <ul>
                            <li>Use clear, high-quality audio</li>
                            <li>Ensure minimal background noise</li>
                            <li>Results may vary based on audio quality</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
