"""Subprocess worker for counterfactual computation.

Invoked as:
    python _cf_worker.py <input_pickle> <output_pickle>

Reads a payload dict from <input_pickle>, runs compute_counterfactuals,
and writes the resulting DataFrame to <output_pickle>.
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

    from counterfactuals import compute_counterfactuals  # noqa: PLC0415

    cf_df = compute_counterfactuals(
        model=payload["model"],
        preprocessor=payload["preprocessor"],
        calibrator=payload["calibrator"],
        all_widget_values=payload["all_widget_values"],
        input_feature_names=payload["input_feature_names"],
        dictionary_labels=payload["dictionary_labels"],
        current_probability=payload["current_probability"],
    )

    with open(tmp_out, "wb") as fh:
        pickle.dump(cf_df, fh)


if __name__ == "__main__":
    main()
