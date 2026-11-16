"""Run the trained model on input 0 and input 1."""
from pathlib import Path
from model import OneBitNOT

model = OneBitNOT.load(Path(__file__).resolve().parent.parent / "model.bin")
for x in (0, 1):
    print(f"{x} -> {model.predict(x)}")
