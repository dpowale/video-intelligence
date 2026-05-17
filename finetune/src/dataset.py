import torch
import torchaudio
import numpy as np
from datasets import load_dataset, Audio

class NoiseAugmentation:
    """Applies Gaussian noise to audio samples to improve model robustness."""
    def __init__(self, snr_min=5.0, snr_max=15.0):
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __call__(self, waveform_data):
        snr_db = np.random.uniform(self.snr_min, self.snr_max)
        snr = 10 ** (snr_db / 10)
        
        waveform = torch.tensor(waveform_data) 
        signal_power = waveform.norm(p=2) ** 2 / waveform.numel()
        noise_power = signal_power / snr
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        
        return (waveform + noise).numpy()

def prepare_dataset(config, processor):
    """Loads and preprocesses audio dataset for Whisper fine-tuning."""
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["language_abbr"],
        split=config["data"]["train_split"]
    )

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    noise_aug = NoiseAugmentation(*config["augmentation"].get("snr_db", [5.0, 15.0])) if config["augmentation"].get("apply_noise") else None

    def prepare_item(batch):
        audio = batch["audio"]
        audio_array = audio["array"]
        if noise_aug:
            audio_array = noise_aug(audio_array)

        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_features[0]

        text_content = batch.get("sentence", batch.get("text", batch.get("transcription", "")))
        batch["labels"] = processor.tokenizer(text_content).input_ids
        return batch

    processed_dataset = dataset.map(prepare_item, remove_columns=dataset.column_names, num_proc=4)
    return processed_dataset
