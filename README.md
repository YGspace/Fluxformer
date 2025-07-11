# Fluxformer: Flow-Guided Duplex Attention Transformer via Spatio-Temporal Clustering for Action Recognition

**Authors:** Younggi Hong, Min Ju Kim, Isack Lee, Seok Bong Yoo\*  
**Published in:** IEEE Robotics and Automation Letters (RA-L), Vol. 8, No. 10, pp. 6411–6418, Oct. 2023  
**DOI:** [10.1109/LRA.2023.3307285](https://doi.org/10.1109/LRA.2023.3307285)  
**Code & Models:** [GitHub - YGspace/Fluxformer](https://github.com/YGspace/Fluxformer)

---

## 📌 Overview

**Fluxformer** is a robust action recognition framework designed to overcome the computational and representational limitations of standard vision transformers. It introduces a novel *duplex attention mechanism* that jointly leverages RGB appearance and optical flow, enhanced by spatial-temporal clustering to retain fine-grained motion and structure information.

---

## 🔍 Motivation

While Vision Transformers (ViTs) have shown success in action recognition and classification automation tasks, they suffer from:
- **Quadratic complexity** with increasing input resolution
- **Heavy data requirements** for effective training
- **Temporal sparsity**, due to reliance on a few sampled frames
- **Spatial degradation**, from iterative token downsampling

Fluxformer directly addresses these issues with a *multi-modal*, *multi-scale*, and *flow-aware* transformer architecture.

---

## 🧠 Key Contributions

1. **Duplex Attention**:  
   A bi-directional attention mechanism that integrates both *RGB* and *optical flow* modalities, enabling fine-grained motion-aware feature modeling.

2. **Flow-Guided Frame Selection**:  
   Temporal saliency is enhanced by analyzing flow dynamics to prioritize meaningful frames, reducing redundancy and preserving critical motion cues.

3. **Spatio-Temporal Clustering Tokens**:  
   Frames are clustered spatially based on semantic saliency, converting inputs into compact tokens while retaining high spatial resolution.

4. **Efficiency with Accuracy**:  
   The architecture maintains high recognition accuracy with reduced computation, outperforming existing SOTA transformer-based action recognition models.

---

## 📊 Results Summary

- **Benchmark Performance**: Achieved top-tier accuracy on multiple standard datasets (e.g., Something-Something V1/V2, Kinetics).
- **Efficiency**: Demonstrated competitive FLOPs and latency with significant gains in accuracy.

See full results and comparisons in the [paper](https://doi.org/10.1109/LRA.2023.3307285).

---

## 📁 Citation

If you use this work in your research, please cite:

```bibtex
@article{hong2023fluxformer,
  title     = {Fluxformer: Flow-Guided Duplex Attention Transformer via Spatio-Temporal Clustering for Action Recognition},
  author    = {Hong, Younggi and Kim, Min Ju and Lee, Isack and Yoo, Seok Bong},
  journal   = {IEEE Robotics and Automation Letters},
  volume    = {8},
  number    = {10},
  pages     = {6411--6418},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/LRA.2023.3307285}
}
