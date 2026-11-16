"""One trainable binary-weight perceptron for NOT."""
from pathlib import Path


class OneBitNOT:
    """y = step(w*x + 0.5), where w is encoded by one bit."""

    def __init__(self, weight_bit: int = 0):
        if weight_bit not in (0, 1):
            raise ValueError("weight_bit must be 0 or 1")
        self.weight_bit = weight_bit

    @property
    def weight(self) -> int:
        return 1 - 2 * self.weight_bit

    def predict(self, x: int) -> int:
        if x not in (0, 1):
            raise ValueError("Input must be 0 or 1")
        return int(self.weight * x + 0.5 >= 0)

    def save(self, path: str | Path = "model.bin") -> None:
        # One byte on disk, but only the least significant bit stores the weight.
        Path(path).write_bytes(bytes([self.weight_bit]))

    @classmethod
    def load(cls, path: str | Path = "model.bin") -> "OneBitNOT":
        data = Path(path).read_bytes()
        if len(data) != 1 or data[0] not in (0, 1):
            raise ValueError("Invalid model: expected one byte equal to 0 or 1")
        return cls(data[0])
