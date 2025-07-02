# InfoGate: Information-Theoretic Gating for Continual Learning

This repository contains the official implementation of our AAAI 2026 submission:

> **InfoGate: Information-Theoretic Gating for Continual Learning in Memory-Augmented Transformers**  
> *AAAI 2026 (Under Review)*

## What is InfoGate?

**InfoGate** is a simple yet effective gating mechanism for memory selection in continual learning. It uses only **forward-pass signals** — predictive entropy, model confidence, and attention sharpness — to decide whether a given input should be written to external memory.

The core idea:  
> Not all samples are worth remembering. InfoGate selects those that are **confident**, **informative**, and **semantically focused**, improving both memory quality and long-term generalization.

## Features

- **Entropy  and confidence-based gating** for sparse memory updates  
- **Attention sharpness filtering** to avoid off-focus tokens  
- **KL-regularized memory replay** for semantic stability  
- Supports **vision (Split MNIST, CIFAR-10, TinyImageNet)** and **NLP (WikiText-30K)** continual learning settings  
- Modular implementation with PyTorch and plug-and-play memory policies

## Getting Started

### Run on Split MNIST

```bash
python experiments/run_split_mnist.py --config configs/split_mnist.yaml
```

### Run on WikiText-30K

```bash
python experiments/run_wikitext.py --buffer_size 400 --gating entropy_confidence_attention
```

All logs and final memory buffers are saved under `./results/`.

## Reproducing Plots

```bash
python plots/plot_gate_distribution.py
python plots/plot_imbalance_accuracy.py
```

Figures will be saved in `plots/figures/`.

---

## Project Structure

```
InfoGate-Continual-Learning/
├── models/               # Transformer and memory modules
├── memory/               # Gating, replay, and consolidation
├── experiments/          # Training scripts per dataset
├── configs/              # YAML configs for reproducibility
├── plots/                # Visualization code
├── results/              # Raw .json logs and buffers
├── train.py              # Main training loop
├── README.md
├── requirements.txt
```

---

## Key Results

| Task           | InfoGate Accuracy | RS Accuracy | Forgetting ↓ |
| -------------- | ----------------- | ----------- | ------------ |
| Split CIFAR-10 | 0.52              | 0.49        | 0.046        |
| WikiText-30K   | 0.90              | 0.72        | 0.03         |
| TinyImageNet   | 0.22              | 0.18        | 0.05         |

---

## License

This project is licensed under the [MIT License](LICENSE).
If you find this useful, please consider citing our paper.

---
