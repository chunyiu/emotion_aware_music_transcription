"""
Shared pitch detection methods.
Each returns (times, f0, confidence_or_voiced) arrays.
"""

import numpy as np
import librosa


def detect_pitch_simple_pyin(audio_path, sr=16000, hop_length=512,
                              fmin=65.4, fmax=2093.0):
    """
    Simple pYIN pitch detection (librosa.pyin).

    Returns:
        times: array of frame times
        f0: array of F0 values (Hz), NaN for unvoiced
        voiced_flag: boolean array
        voiced_probs: array of voicing probabilities
        y: audio signal
        sr_out: sample rate used
    """
    y, sr_out = librosa.load(audio_path, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr_out,
        hop_length=hop_length,
        fill_na=None
    )
    times = librosa.times_like(f0, sr=sr_out, hop_length=hop_length)
    return times, f0, voiced_flag, voiced_probs, y, sr_out


def detect_pitch_pyin_hmm(audio_path, config=None):
    """
    pYIN extraction followed by Viterbi HMM smoothing.

    Args:
        audio_path: Path to audio file.
        config: ExperimentConfig-like object with pyin_fmin, pyin_fmax,
                pyin_frame_length, pyin_hop_length, median_kernel_size,
                and Viterbi HMM parameters.

    Returns:
        times: array of frame times
        smoothed_midi: array of smoothed MIDI pitch values (0 = unvoiced)
        f0: raw F0 array
        voiced_probs: voicing probability array
        y: audio signal
        sr: sample rate
    """
    import scipy.signal

    if config is None:
        config = _default_hmm_config()

    y, sr = librosa.load(audio_path, sr=None)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=config.pyin_fmin,
        fmax=config.pyin_fmax,
        sr=sr,
        frame_length=config.pyin_frame_length,
        hop_length=config.pyin_hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=config.pyin_hop_length)

    # Viterbi HMM decoding
    converter = _ViterbiHMMDecoder(config)
    smoothed_midi = converter.viterbi_decode(f0, voiced_probs)

    # Median filter
    smoothed_midi = scipy.signal.medfilt(
        smoothed_midi, kernel_size=config.median_kernel_size
    )

    return times, smoothed_midi, f0, voiced_probs, y, sr


def detect_pitch_crepe(audio_path, model='full', step_size=10):
    """
    CREPE (TensorFlow) pitch detection.

    Returns:
        times: array of frame times
        frequency: array of F0 values (Hz)
        confidence: array of confidence values
    """
    import crepe

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    time, frequency, confidence, _ = crepe.predict(
        y, sr,
        model_capacity=model,
        step_size=step_size,
        viterbi=True
    )
    return time, frequency, confidence


def detect_pitch_torchcrepe(audio_path, frame_rate=100, fmin=50.0, fmax=1100.0,
                             chunk_seconds=10, model='full'):
    """
    TorchCrepe (PyTorch) pitch detection with chunk-based processing.

    Returns:
        times: array of frame times
        f0: array of F0 values (Hz)
        conf: array of confidence/periodicity values
    """
    import torch
    import torchcrepe

    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    if y.size == 0:
        raise ValueError(f"Empty audio signal for {audio_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
# After
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    hop_length = int(sr / frame_rate)
    max_chunk = int(sr * chunk_seconds)

    all_f0, all_conf = [], []

    for start in range(0, len(y), max_chunk):
        end = min(start + max_chunk, len(y))
        chunk = y[start:end]
        if chunk.size == 0:
            continue

        try:
            out = torchcrepe.predict(
                torch.tensor(chunk[None, :], device=device),
                sr,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                model=model,
                batch_size=1024,
                device=device,
                return_periodicity=True
            )
        except Exception as e:
            print(f"  [TorchCrepe] Skipping chunk {start}:{end}: {e}")
            continue

        if isinstance(out, (list, tuple)) and len(out) >= 2:
            pitch, confidence = out[0], out[1]
        else:
            pitch = out[0] if isinstance(out, (list, tuple)) else out
            confidence = torch.ones_like(pitch)

        f0_chunk = np.atleast_1d(pitch.squeeze(0).cpu().numpy())
        conf_chunk = np.atleast_1d(confidence.squeeze(0).cpu().numpy())

        if f0_chunk.size > 0:
            all_f0.append(f0_chunk)
            all_conf.append(conf_chunk)

    if not all_f0:
        raise RuntimeError(f"TorchCrepe produced no frames for {audio_path}")

    f0 = np.concatenate(all_f0)
    conf = np.concatenate(all_conf)
    times = np.arange(len(f0)) / frame_rate

    return times, f0, conf


# ---------------------------------------------------------------------------
# Internal: Viterbi HMM decoder for pYIN+HMM pipeline
# ---------------------------------------------------------------------------

class _DefaultHMMConfig:
    pyin_fmin = 65.4
    pyin_fmax = 1047.0
    pyin_frame_length = 2048
    pyin_hop_length = 256
    median_kernel_size = 7
    self_trans_prob = 0.85
    neighbor_prob = 0.06
    unvoiced_stay = 0.5
    min_note_duration = 0.3
    pitch_threshold = 0.5
    vibrato_tolerance = 0.3
    merge_threshold = 0.5
    min_gap = 0.08


def _default_hmm_config():
    return _DefaultHMMConfig()


class _ViterbiHMMDecoder:
    """Viterbi HMM for pitch smoothing (from Pipeline C/D/1/2)."""

    def __init__(self, config):
        self.config = config
        self.fmin = config.pyin_fmin
        self.fmax = config.pyin_fmax
        self.midi_min = int(librosa.hz_to_midi(self.fmin))
        self.midi_max = int(librosa.hz_to_midi(self.fmax))
        self.midi_notes = np.arange(self.midi_min, self.midi_max + 1)
        self.n_states = len(self.midi_notes)
        self.transition_matrix = self._build_transition_matrix()
        self.start_prob = np.ones(self.n_states + 1) / (self.n_states + 1)

    def _build_transition_matrix(self):
        n = self.n_states
        A = np.zeros((n + 1, n + 1))
        cfg = self.config
        for i in range(n):
            A[i, i] = cfg.self_trans_prob
            if i > 0:
                A[i, i - 1] = cfg.neighbor_prob
            if i < n - 1:
                A[i, i + 1] = cfg.neighbor_prob
            remaining = 1 - A[i].sum()
            A[i, -1] = max(0, remaining)
        A[-1, -1] = cfg.unvoiced_stay
        remaining = 1 - cfg.unvoiced_stay
        A[-1, :-1] = remaining / n
        return A

    def _emission_prob(self, observed_midi, voiced_prob, sigma=2.0):
        emissions = np.zeros(self.n_states + 1)
        if observed_midi > 0 and voiced_prob > 0.1:
            for i, midi_note in enumerate(self.midi_notes):
                diff = abs(observed_midi - midi_note)
                emissions[i] = np.exp(-0.5 * (diff / sigma) ** 2) * voiced_prob
            emissions[-1] = (1 - voiced_prob) * 0.2
        else:
            emissions[:-1] = 0.1
            emissions[-1] = 0.8
        emissions /= (emissions.sum() + 1e-12)
        return emissions

    def viterbi_decode(self, f0, voiced_probs):
        n_frames = len(f0)
        n_states = self.n_states + 1

        log_A = np.log(self.transition_matrix + 1e-12)
        log_pi = np.log(self.start_prob + 1e-12)
        log_delta = np.zeros((n_frames, n_states))
        psi = np.zeros((n_frames, n_states), dtype=int)

        obs_midi = np.zeros_like(f0)
        valid_mask = (f0 > 0) & np.isfinite(f0)
        obs_midi[valid_mask] = librosa.hz_to_midi(f0[valid_mask])

        log_delta[0] = log_pi + np.log(
            self._emission_prob(obs_midi[0], voiced_probs[0]) + 1e-12
        )

        for t in range(1, n_frames):
            emission = np.log(
                self._emission_prob(obs_midi[t], voiced_probs[t]) + 1e-12
            )
            for j in range(n_states):
                seq_probs = log_delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(seq_probs)
                log_delta[t, j] = np.max(seq_probs) + emission[j]

        states = np.zeros(n_frames, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(n_frames - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        smoothed_midi = np.array([
            self.midi_notes[s] if s < self.n_states else 0
            for s in states
        ])
        return smoothed_midi
