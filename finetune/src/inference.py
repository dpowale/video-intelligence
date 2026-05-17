import torchaudio
import numpy as np
from faster_whisper import WhisperModel

class StreamingVADTranscriber:
    """
    Streaming Transcription using Faster-Whisper, with VAD (Voice Activity Detection) preprocessing
    utilizing Silero VAD.
    """
    def __init__(self, model_size="large-v3", device="cuda"):
        compute_type = "float16" if device == "cuda" else "int8"
        # Faster-whisper for 4x speedup vs typical HF Whisper
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        # Load external VAD (e.g. Silero VAD)
        import torch
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
        self.get_speech_timestamps = utils[0]

    def _get_vad_segments(self, audio_array, sr=16000):
        # Silero expects float tensor
        import torch
        tensor_audio = torch.tensor(audio_array, dtype=torch.float32)
        timestamps = self.get_speech_timestamps(tensor_audio, self.vad_model, sampling_rate=sr)
        return timestamps

    def transcribe_file(self, audio_path):
        import soundfile as sf
        audio_array, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
        print("Running VAD...")
        segments = self._get_vad_segments(audio_array)
        if not segments:
            return "No voice active."

        # Compile valid chunks
        active_audio = np.concatenate([audio_array[s['start']:s['end']] for s in segments])

        print("Transcribing (Faster-Whisper)...")
        segments, info = self.model.transcribe(active_audio, beam_size=5)
        
        res = []
        for s in segments:
            res.append(s.text)
        return " ".join(res)

    def stream_transcribe(self, chunk_generator):
        """Processes rolling chunks of audio array in a simulated streaming fashion."""
        for chunk in chunk_generator:
            # Check VAD per chunk
            if len(self._get_vad_segments(chunk)) > 0:
                segments, _ = self.model.transcribe(chunk, beam_size=5)
                for s in segments:
                    yield s.text

if __name__ == "__main__":
    t = StreamingVADTranscriber()
    # print(t.transcribe_file("sample.wav"))
