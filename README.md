---
license: mit
tags:
- binary-neural-network
- perceptron
- logic-gate
- educational
- 1-bit
---
# OneBit-NOT

An educational **single-neuron, one-trainable-bit binary perceptron** trained to implement the NOT truth table.

## Architecture

The model uses a step activation and one trainable binary weight:

```text
y = step(w * x + 0.5)
w = 1 - 2 * weight_bit
weight_bit in {0, 1}
step(z) = 1 if z >= 0 else 0
```

The bias `0.5` is a **fixed constant**, not a trainable parameter. Initially `weight_bit=0` (w=+1). Discrete learning flips this bit only if doing so decreases the number of errors on the two training examples. The optimum is `weight_bit=1` (w=-1).

This is a threshold neuron (perceptron), not a differentiable neural network. The training algorithm does not use backpropagation.

## Model size

- Trainable parameters: **1**
- Trainable parameter information: **1 bit**
- Serialized `model.bin`: **1 byte** (file storage is byte-addressed)
- Fixed constants, code and metadata are not included in the parameter count.

## Training data and evaluation

The complete NOT truth table is both the training set and the evaluation set:

| Input | Target | Prediction |
|---|---|---|
| 0 | 1 | 1 |
| 1 | 0 | 0 |

Training accuracy: 100% (2/2). This is **training-set accuracy**, not a held-out generalization claim.

## Usage

No runtime dependencies beyond Python 3.10+.

```bash
python3 python/train.py
python3 python/inference.py
```

or:

```python
from python.model import OneBitNOT
model = OneBitNOT.load("model.bin")
assert model.predict(0) == 1
assert model.predict(1) == 0
```

## Limitations

This is a toy educational experiment designed specifically for a binary NOT operation. It does not support real-valued inputs, general logical operations, or learned representations. The hypothesis class and fixed bias were chosen so the NOT mapping is representable. It does not integrate with the standard Transformers `pipeline` API; load using the included Python module.

## ggml CPU inference (C++17)

This repository includes `python/` for training and Python inference and `ggml/` for C++ inference. Both share the root `model.bin`.

This repository also includes a standalone ggml-based CPU inference program. It reads
**the same `model.bin`** produced by `train.py`. It does not use llama.cpp or its CLI.

Requirements: CMake 3.20+, C++17 compiler, Git, internet access on first CMake configure.

```bash
python3 python/train.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/onebit-not-ggml model.bin
./build/onebit-not-ggml model.bin 0
./build/onebit-not-ggml model.bin 1
```

Expected output for the first inference command:

```text
0 -> 1
1 -> 0
```

The program constructs the computation graph `step(x*w + 0.5)` via ggml tensor
operations and executes it on ggml's CPU backend. The model stores **one trainable bit**
(0 for +1, 1 for -1); ggml expands that bit to a temporary float32 tensor at runtime.
The 0.5 bias is a fixed, non-trainable constant. The executable and transient
computation tensors are much larger than 1 bit. This is a demonstration of a
one-bit *parameter*, not of one-bit memory usage for the entire runtime.

The CMake build fetches the latest ggml source from the `master` branch; for
reproducible releases, pin `GIT_TAG` to a specific ggml commit. This integration
has not been compiled against a local ggml installation in the packaging environment.

## Repository layout

```text
OneBit-NOT/
├── README.md
├── LICENSE
├── model.bin
├── requirements.txt
├── CMakeLists.txt
├── python/
│   ├── model.py
│   ├── train.py
│   └── inference.py
└── ggml/
    └── main.cpp
```

The one-bit trainable value is saved in one byte; runtime memory and source code are larger. ggml support is a standalone example, not a `llama-cli` compatible architecture.
