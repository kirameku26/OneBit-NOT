"""Train a binary perceptron using a discrete weight-flip update."""
from pathlib import Path
from model import OneBitNOT

DATA = [(0, 1), (1, 0)]


def errors(model: OneBitNOT) -> int:
    return sum(model.predict(x) != target for x, target in DATA)


def main() -> None:
    model = OneBitNOT(weight_bit=0)  # Start with w=+1
    print(f"Before: bit={model.weight_bit}, weight={model.weight:+d}, errors={errors(model)}")
    # Flip the only trainable bit if it improves the objective.
    candidate = OneBitNOT(weight_bit=model.weight_bit ^ 1)
    if errors(candidate) < errors(model):
        model = candidate
    print(f"After:  bit={model.weight_bit}, weight={model.weight:+d}, errors={errors(model)}")
    assert errors(model) == 0
    model.save(Path(__file__).resolve().parent.parent / "model.bin")
    for x, target in DATA:
        print(f"input={x}, prediction={model.predict(x)}, target={target}")


if __name__ == "__main__":
    main()
