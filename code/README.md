# SAM-CRN: Secure Adaptive Multimodal Clinical Reasoning Network

A single-folder PyTorch implementation of SAM-CRN featuring:

- 70 / 15 / 15 train–val–test split with per-class name reporting
- Multimodal encoders (EHR, IoT, Physiological, Lab)
- Graph Attention over a clinical modality graph
- Mamba-style State-Space temporal module
- Adaptive Graph-Temporal Fusion (learnable gate `alpha`)
- Knowledge-Guided Clinical Attention
- Uncertainty-Aware Multi-task heads (classification + heteroscedastic regression)
- SHAP + attention explainability
- Isolation Forest anomaly detection

## Quickstart

```bash
pip install -r requirements.txt
# edit CSV_PATH in config.py, then:
python main.py
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Paths, hyperparameters, column groups |
| `preprocessing.py` | Cleaning, encoding, scaling |
| `split.py` | 70/15/15 stratified split + per-class report |
| `dataset.py` | PyTorch Dataset returning per-modality tensors |
| `encoders.py` | Per-modality MLP encoder |
| `graph_attention.py` | GAT layer over 4-modality graph |
| `ssm.py` | Mamba-style state-space block |
| `fusion.py` | Adaptive graph-temporal gate |
| `knowledge_attention.py` | Concept-level clinical attention |
| `heads.py` | Multi-task heads with uncertainty |
| `model.py` | Full `SAM_CRN` model |
| `losses.py` | Multi-task loss |
| `trainer.py` | Train/val loop |
| `metrics.py` | Metrics + confusion-matrix reporting |
| `anomaly.py` | Isolation Forest on embeddings |
| `explainability.py` | SHAP + attention inspection |
| `main.py` | End-to-end pipeline |
