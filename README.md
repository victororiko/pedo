# Elumina — Fine-tuned Gemma 4 27B for Kenya

**Elumina** is a multilingual AI assistant built by **Intevia Ltd**, a technology company founded in Nairobi, Kenya, building intelligent systems grounded in African realities.

Fine-tuned from Google's Gemma 4 27B using QLoRA on RunPod.

## Languages
- English
- Swahili (Kiswahili)
- Sheng
- Other Kenyan languages (Kikuyu, Luo, Kalenjin, Luhya, etc.)

## Project Structure

```
Finetune/
├── config/
│   └── training_config.yaml      # Hyperparameters & paths
├── data/
│   ├── raw/                      # Source material (add your data here)
│   ├── identity/                 # Identity training data
│   │   └── identity_dataset.jsonl
│   ├── cultural/                 # Kenyan cultural knowledge
│   └── processed/                # Final merged training data
├── scripts/
│   ├── prepare_data.py           # Merge & format datasets
│   ├── train.py                  # QLoRA fine-tuning
│   ├── merge_adapter.py          # Merge LoRA into base model
│   └── inference.py              # Test the model
├── setup_runpod.sh               # RunPod environment setup
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Set up RunPod
- Create a RunPod pod with **A100 80GB** or **H100**
- Use the `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` template
- SSH in and clone this repo

```bash
bash setup_runpod.sh
```

### 2. Prepare Data
Add your raw training data to `data/raw/`, then:
```bash
python scripts/prepare_data.py
```

### 3. Train
```bash
python scripts/train.py
```

### 4. Merge & Export
```bash
python scripts/merge_adapter.py
```

### 5. Test
```bash
python scripts/inference.py
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | A100 80GB | H100 80GB |
| RAM | 64GB | 128GB |
| Disk | 200GB | 500GB |
| RunPod Cost | ~$1.50/hr | ~$3.00/hr |

## Dataset Guidelines

### Identity Data
Conversations where Elumina identifies itself correctly. Already templated in `data/identity/`.

### Cultural Data
- Kenyan history, geography, politics
- Swahili/Sheng language pairs and usage
- Local customs, food, music, technology landscape
- East African current affairs
- Kenyan business, finance, M-Pesa, etc.

### Format
All data should be in JSONL format with chat-style messages:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## License
Model weights are subject to Google's Gemma license terms. Fine-tuned adapter weights are owned by Intevia Ltd.
