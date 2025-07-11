# Fluxformer: Flow-Guided Duplex Attention Transformer via Spatio-Temporal Clustering for Action Recognition

**Authors**: Younggi Hong, Min Ju Kim, Isack Lee, Seok Bong Yoo  
**Published in**: *IEEE Robotics and Automation Letters (RA-L)*, Vol. 8, No. 10, 2023  
**DOI**: [10.1109/LRA.2023.3307285](https://doi.org/10.1109/LRA.2023.3307285)  
**Code**: [github.com/YGspace/Fluxformer](https://github.com/YGspace/Fluxformer)

---

## 🔍 Abstract

Fluxformer is a novel action recognition framework designed to address the limitations of existing transformer-based models. While vision transformers offer powerful representation learning, they often struggle with computational inefficiency, loss of temporal precision due to frame sampling, and spatial degradation due to progressive downsampling. Fluxformer mitigates these issues via:

- **Duplex Attention**: Combines RGB and optical flow in a joint self- and cross-attention mechanism.
- **MAEE (Meaningful Action Event Extractor)**: Identifies keyframes using flow-based scene detection and temporal clustering.
- **SAC (Spatial Attention Clustering)**: Preserves spatial structure via clustering-based support token generation.

These components allow Fluxformer to retain fine-grained spatio-temporal information while remaining efficient and scalable.

---

## 🧠 Key Contributions

- **Flow-Guided Duplex Attention**: Introduces a dual-stream attention mechanism integrating RGB and flow modalities through both self- and cross-attention.
- **Meaningful Frame Extraction (MAEE)**: Employs optical flow and K-means clustering to dynamically extract salient temporal segments.
- **Spatial Attention Clustering (SAC)**: Enhances spatial fidelity using tubelet embeddings and spatial clustering to generate auxiliary spatial tokens.
- **Efficient Computation**: Achieves a strong balance between accuracy and efficiency, supporting real-time inference (31 ms per frame).

---

## 🧪 Benchmarks

| Dataset          | Accuracy (Top-1 / Top-5) | Model Compared | Performance Highlight |
|------------------|--------------------------|----------------|------------------------|
| HMDB-51          | **83.6 / 88.6**           | I3D, ViViT, MViT | SOTA accuracy with fewer clips |
| Kinetics-400 (5%)| **75.9 / 83.3**           | R(2+1)D, ViViT | Efficient despite lower sampling |
| Kinetics-600 (5%)| **59.7 / 77.7**           | ViViT, MViT     | Best accuracy under constrained FLOPs |
| SSV2 (Lightweight) | **82.9 / 93.1**         | ViViT, MViT     | Large gain on low frame-rate/high-motion data |

### 🧮 Computation

| Model         | FLOPs (G) | Params (M) |
|---------------|-----------|------------|
| ViViT         | 3992      | 310.8      |
| MViT          | 225       | 51.2       |
| **Fluxformer**| **243**   | **86.4**   |

> Note: Optical flow estimation uses ~20 GFLOPs. Core action recognition uses ~223 GFLOPs.

---

## ⚙️ Architecture

Fluxformer is composed of the following modules:

1. **MAEE**: Extracts informative frames by detecting scene changes via optical flow, followed by K-means temporal clustering.
2. **Duplex Attention**: RGB and Flow embeddings are passed through both self-attention and cross-attention layers to enhance temporal sensitivity.
3. **SAC**: Converts spatio-temporal volumes into compact support tokens that are concatenated to final layer outputs to retain spatial awareness.
4. **Classifier**: The final classification token is refined through MLP after global self-attention.

> See Fig. 2–5 in the paper for detailed diagrams of each module.

---

## 📦 Implementation

- **Backbone**: MViT
- **Embedding**: 96-D tubelet embeddings from 16-frame clips
- **Optical Flow**: Lightweight version of GMFlow (~7ms/frame)
- **Training**: Cosine learning rate schedule, soft cross-entropy, batch size = 2
- **Hardware**: Intel i7-10700K, RTX 3090

---

## 📊 Ablation Summary

| Components Added      | Accuracy (Kinetics-400 5%) |
|------------------------|----------------------------|
| Base (MViT)            | 54.2%                      |
| + Duplex Attention     | 58.9%                      |
| + SAC                  | 66.1%                      |
| + MAEE (final model)   | **75.9%**                  |

---

## 📌 Citation

```bibtex
@article{hong2023fluxformer,
  title={Fluxformer: Flow-Guided Duplex Attention Transformer via Spatio-Temporal Clustering for Action Recognition},
  author={Hong, Younggi and Kim, Min Ju and Lee, Isack and Yoo, Seok Bong},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={10},
  pages={6411--6418},
  year={2023},
  publisher={IEEE},
  doi={10.1109/LRA.2023.3307285}
}
