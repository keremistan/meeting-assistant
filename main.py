import whisper
import os
import torch
import torchaudio

# Monkeypatch torchaudio.list_audio_backends if it's missing (removed in torchaudio 2.1+)
# This is required because speechbrain 1.0.3 depends on this deprecated function.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def transcribe_audio():
    # Define paths
    data_folder = "data"
    audio_file = "sample.mp3"
    audio_path = os.path.join(data_folder, audio_file)
    output_file = "transcription.txt"
    output_path = os.path.join(data_folder, output_file)

    # Configuration
    num_speakers = None  # Set to a specific integer (e.g. 3) to enforce speaker count, or None for auto-detection

    # Device configuration
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    # 1. Transcribe with Whisper
    print(f"Loading Whisper model...")
    model = whisper.load_model("turbo", device=device)

    print(f"Transcribing {audio_file}...")
    result = model.transcribe(audio_path)
    segments = result["segments"]

    # 2. diarization using SpeechBrain + sklearn
    print("Loading Speaker Recognition model (SpeechBrain)...")
    # Use a model that doesn't strictly need a HF token (spkrec-ecapa-voxceleb is open)
    # Note: run_opts={"device": "cpu"} or "cuda" if available
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_model",
        run_opts={"device": device},
    )

    print("Computing speaker embeddings...")
    embeddings = []

    # Load audio for embedding extraction (SpeechBrain expects 16kHz)
    signal, fs = torchaudio.load(audio_path)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        signal = resampler(signal)
        fs = 16000

    # Ensure mono
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    for seg in segments:
        start_time = seg["start"]
        end_time = seg["end"]

        # Convert time to samples
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)

        # Extract segment
        segment_signal = signal[:, start_sample:end_sample]

        # Compute embedding (needs to be at least a certain length, handle short segments)
        if segment_signal.shape[1] < 1600:  # < 0.1s
            # Pad or skip?? Skip might desync. Pad with zeros.
            pad_len = 1600 - segment_signal.shape[1]
            segment_signal = torch.nn.functional.pad(segment_signal, (0, pad_len))

        with torch.no_grad():
            full_emb = spk_model.encode_batch(segment_signal)
            embeddings.append(full_emb[0, 0, :].cpu().numpy())

    embeddings = np.array(embeddings)

    print("Clustering speakers...")
    # Using Agglomerative Clustering with a threshold to auto-determine # of speakers
    # Cosine distance is standard for embeddings, but sklearn uses euclidean by default.
    # We can normalize embeddings so euclidean == cosine (roughly)
    # normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    if num_speakers is not None:
        clustering = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric="euclidean",
            linkage="ward",
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=2.0,  # Tune this threshold. 1.0-2.0 is usually decent for normalized vectors
            metric="euclidean",
            linkage="ward",
        )
    labels = clustering.fit_predict(embeddings)

    num_speakers = len(set(labels))
    print(f"Detected {num_speakers} speakers.")

    # 3. Save formatted output
    with open(output_path, "w") as f:
        for i, seg in enumerate(segments):
            start_fmt = format_timestamp(seg["start"])
            end_fmt = format_timestamp(seg["end"])
            speaker = f"Speaker {labels[i]}"
            text = seg["text"].strip()

            line = f"[{start_fmt} --> {end_fmt}] {speaker}: {text}\n"
            f.write(line)

    print(f"Transcription saved to {output_path}")


def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02}:{secs:02}"


if __name__ == "__main__":
    transcribe_audio()
