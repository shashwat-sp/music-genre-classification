"""Data preprocessing utilities for audio files."""

import numpy as np
import librosa
import torch
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def generate_mel_spectrogram(
    audio_path: str,
    sr: int = 22050,
    duration: float = 30.0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_len: Optional[int] = None,
    normalize: bool = True
) -> Optional[torch.Tensor]:
    """
    Generate normalized mel-spectrogram from audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        duration: Duration to load (seconds)
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
        max_len: Maximum time steps (padding/truncation)
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Mel-spectrogram tensor of shape (1, n_mels, time)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if normalize:
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        if max_len:
            if mel_spec_db.shape[1] < max_len:
                pad_width = max_len - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :max_len]
        
        mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
        return mel_tensor
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def generate_mel_segments(
    audio_path: str,
    sr: int = 22050,
    segment_duration: float = 3.0,
    hop_duration: float = 1.5,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> List[torch.Tensor]:
    """
    Split audio file into overlapping segments.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        segment_duration: Duration of each segment (seconds)
        hop_duration: Hop between segments (seconds)
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
    
    Returns:
        List of mel-spectrogram tensors
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"  [skip] {audio_path}: {e}")
        return []
    
    segment_len = int(segment_duration * sr)
    hop_len_samples = int(hop_duration * sr)
    
    segments = []
    start = 0
    
    while start + segment_len <= len(y):
        chunk = y[start : start + segment_len]
        
        mel = librosa.feature.melspectrogram(
            y=chunk, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        segments.append(tensor)
        start += hop_len_samples
    
    return segments


def compute_audio_features(
    audio_path: str,
    sr: int = 22050,
    duration: float = 30.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract audio features for traditional ML models.
    
    Returns:
        Tuple of (spectral_centroid, mfccs, chroma)
    """
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    return spectral_centroid, mfccs, chroma