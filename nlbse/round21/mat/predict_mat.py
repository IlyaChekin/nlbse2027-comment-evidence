"""Predict both NLBSE labels with the frozen compact encoder and MAT marker recovery."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "round20"))
import predict_factor
from marker import recover


def decisions(text, config, model, tok, heads):
    result = predict_factor.decisions(text, config, model, tok, heads)
    result[:, 1] = recover(text, result[:, 1])
    return result


if __name__ == "__main__":
    predict_factor.main(decisions)
