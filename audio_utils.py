"""Audio processing utilities for music genre classification."""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


def load_audio(
    file_path: str,
    sr: int = 22050,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with optional duration limit.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate
        duration: Duration to load in seconds
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract MFCC features from audio."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)


def extract_chroma(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract chroma features from audio."""
    return librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)


def extract_spectral_centroid(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract spectral centroid from audio."""
    return librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)


def extract_spectral_rolloff(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    roll_percent: float = 0.85,
) -> np.ndarray:
    """Extract spectral rolloff from audio."""
    return librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=hop_length, roll_percent=roll_percent
    )


def extract_zero_crossing_rate(
    y: np.ndarray,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract zero crossing rate from audio."""
    return librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)


def extract_rmse(
    y: np.ndarray,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract RMSE (energy) from audio."""
    return librosa.feature.rms(y=y, hop_length=hop_length)


def plot_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "Waveform",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot audio waveform."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    fmax: int = 8000,
    title: str = "Mel Spectrogram",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot mel spectrogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    img = librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        fmax=fmax
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    return ax


def plot_audio_features(
    y: np.ndarray,
    sr: int,
    title: str = "Audio Features",
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    """
    Plot multiple audio features.
    
    Args:
        y: Audio data
        sr: Sample rate
        title: Figure title
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Waveform
    ax1 = plt.subplot(3, 2, 1)
    plot_waveform(y, sr, title="Waveform", ax=ax1)
    
    # Mel Spectrogram
    ax2 = plt.subplot(3, 2, 2)
    plot_mel_spectrogram(y, sr, title="Mel Spectrogram", ax=ax2)
    
    # Spectral Centroid
    ax3 = plt.subplot(3, 2, 3)
    spectral_centroid = extract_spectral_centroid(y, sr)
    times = librosa.times_like(spectral_centroid, sr=sr)
    ax3.plot(times, spectral_centroid[0], color='#2B4F72', linewidth=2)
    ax3.set_title('Spectral Centroid', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (seconds)', fontsize=10)
    ax3.set_ylabel('Frequency (Hz)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Spectral Rolloff
    ax4 = plt.subplot(3, 2, 4)
    spectral_rolloff = extract_spectral_rolloff(y, sr)
    ax4.plot(times, spectral_rolloff[0], color='#A23B72', linewidth=2)
    ax4.set_title('Spectral Rolloff', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (seconds)', fontsize=10)
    ax4.set_ylabel('Frequency (Hz)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Zero Crossing Rate
    ax5 = plt.subplot(3, 2, 5)
    zcr = extract_zero_crossing_rate(y)
    ax5.plot(times, zcr[0], color='#F18F01', linewidth=2)
    ax5.set_title('Zero Crossing Rate', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time (seconds)', fontsize=10)
    ax5.set_ylabel('Rate', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # RMSE
    ax6 = plt.subplot(3, 2, 6)
    rmse = extract_rmse(y)
    ax6.plot(times, rmse[0], color='#73AB84', linewidth=2)
    ax6.set_title('RMSE (Energy)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Time (seconds)', fontsize=10)
    ax6.set_ylabel('Energy', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()