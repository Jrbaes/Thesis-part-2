"""Subprocess worker for prediction.

Invoked as:
    python _predict_worker.py <input_pickle> <output_pickle>

Reads a payload dict, calls predict_with_venn_abers, writes PredictionResult.
Any crash here is contained and does NOT affect the Streamlit main process.
"""
from __future__ import annotations

import pickle
import sys


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(1)

    tmp_in = sys.argv[1]
    tmp_out = sys.argv[2]

    with open(tmp_in, "rb") as fh:
        payload = pickle.load(fh)

    from backend import predict_with_venn_abers  # noqa: PLC0415

    result = predict_with_venn_abers(
        model=payload["model"],
        input_frame=payload["input_frame"],
        calibrator=payload["calibrator"],
    )

    with open(tmp_out, "wb") as fh:
        pickle.dump(result, fh)


if __name__ == "__main__":
    main()
