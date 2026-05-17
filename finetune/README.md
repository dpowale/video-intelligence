# Whisper Fine-Tuning Framework

A production-grade minimal framework for fine-tuning Whisper-family Audio Speech Recognition (ASR) models using PEFT (Parameter-Efficient Fine-Tuning) and Faster-Whisper.

## Features Included:
- **LoRA / PEFT fine-tuning**: Train large models efficiently on single consumer GPUs.
- **Faster-Whisper inference**: Use `faster-whisper` for optimized CTranslate2 integration.
- **Noise Custom Augmentation**: Add on-the-fly random Gaussian Signal-to-Noise injection.
- **VAD Preprocessing**: Pre-filtering non-speech chunks via Silero VAD before generating tokens.
- **Streaming transcription**: Generator-based streaming layout in `inference.py`.
- **Evaluation Metrics**: WER and CER generation using `jiwer`.
- **Docker Support**: Containerized training environment.

## Usage

**Running Locally:**
```bash
cd finetune
pip install -r requirements.txt
python src/train.py --config configs/config.yaml
```

**Evaluate:**
```bash
python src/evaluate.py
```

**Docker Engine (GPU):**
```bash
docker-compose up --build
```
