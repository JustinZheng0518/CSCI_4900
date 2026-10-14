"""
Experiment 06: ELMo Alignment Sanity Check for Gao et al. (2018)

Purpose:
    Verify that the VUA verb-classification CSV data aligns correctly
    with the precomputed ELMo representations released for Gao et al. (2018).

Checks:
    1. Every CSV sentence can be found in the corresponding HDF5 file.
    2. sentence.split() token count matches the ELMo sequence length.
    3. verb_idx is within sentence bounds.
    4. Report lemma vs. surface-form differences for inspection.

Important:
    The CSV "verb" column is usually a lemma, while the token at verb_idx
    is the actual inflected surface form. For example:

        verb column: go
        sentence token: going

    Therefore lemma/surface-form differences are expected and are NOT
    considered alignment failures.

    Also, HDF5 treats "/" as a hierarchy separator. Therefore checking
    only f.keys() is NOT sufficient for sentences containing "/" such as
    "ARC/INFO". We directly test sentence membership with:

        sentence in f

This script does NOT train a model.
"""

from pathlib import Path

import h5py
import pandas as pd


# ============================================================
# Paths
# ============================================================

DATASETS = {
    "train": {
        "csv": Path("data/VUA/VUA_formatted_train.csv"),
        "elmo": Path("embeddings/ELMo/elmo/VUA_train.hdf5"),
    },
    "val": {
        "csv": Path("data/VUA/VUA_formatted_val.csv"),
        "elmo": Path("embeddings/ELMo/elmo/VUA_val.hdf5"),
    },
    "test": {
        "csv": Path("data/VUA/VUA_formatted_test.csv"),
        "elmo": Path("embeddings/ELMo/elmo/VUA_test.hdf5"),
    },
}


# ============================================================
# Split checker
# ============================================================

def check_split(split_name, csv_path, elmo_path):

    print(f"\n===== {split_name.upper()} =====")

    df = pd.read_csv(
        csv_path,
        encoding="latin1",
    )

    missing_sentences = []
    length_mismatches = []
    invalid_targets = []
    lemma_surface_differences = []

    unique_sentences = df["sentence"].nunique()

    with h5py.File(elmo_path, "r") as f:

        for row_idx, row in df.iterrows():

            sentence = str(row["sentence"])
            verb = str(row["verb"])
            verb_idx = int(row["verb_idx"])

            # ------------------------------------------------
            # 1. Check sentence exists in HDF5
            #
            # Do NOT rely on f.keys().
            # "/" inside sentence text is interpreted by HDF5
            # as a hierarchy separator.
            # ------------------------------------------------

            if sentence not in f:

                missing_sentences.append(
                    (
                        row_idx,
                        sentence,
                    )
                )

                continue

            tokens = sentence.split()

            elmo_dataset = f[sentence]

            elmo_shape = elmo_dataset.shape
            elmo_length = elmo_shape[0]

            # ------------------------------------------------
            # 2. Check sentence token count vs ELMo length
            # ------------------------------------------------

            if len(tokens) != elmo_length:

                length_mismatches.append(
                    (
                        row_idx,
                        len(tokens),
                        elmo_length,
                        elmo_shape,
                        sentence,
                    )
                )

            # ------------------------------------------------
            # 3. Check target index
            # ------------------------------------------------

            if not (0 <= verb_idx < len(tokens)):

                invalid_targets.append(
                    (
                        row_idx,
                        verb_idx,
                        len(tokens),
                        sentence,
                    )
                )

                continue

            # ------------------------------------------------
            # 4. Compare lemma column with sentence surface form
            #
            # This is informational only.
            # Differences are expected:
            # go -> going
            # win -> won
            # show -> shown
            # ------------------------------------------------

            target_token = tokens[verb_idx]

            if target_token.lower() != verb.lower():

                lemma_surface_differences.append(
                    (
                        row_idx,
                        verb,
                        target_token,
                        verb_idx,
                        sentence,
                    )
                )

        # ====================================================
        # Summary
        # ====================================================

        print("CSV rows:                     ", len(df))
        print("Unique CSV sentences:         ", unique_sentences)
        print("Top-level HDF5 keys:          ", len(f.keys()))
        print("Missing sentence rows:        ", len(missing_sentences))
        print("Token length mismatches:      ", len(length_mismatches))
        print("Invalid target indices:       ", len(invalid_targets))
        print(
            "Lemma/surface differences:    ",
            len(lemma_surface_differences),
        )

        # ====================================================
        # Detailed diagnostics
        # ====================================================

        if missing_sentences:

            print("\nFirst 3 missing sentences:")

            for item in missing_sentences[:3]:
                print(item)

        if length_mismatches:

            print("\nFirst 3 token length mismatches:")

            for item in length_mismatches[:3]:
                print(item)

        if invalid_targets:

            print("\nFirst 3 invalid target indices:")

            for item in invalid_targets[:3]:
                print(item)

        if lemma_surface_differences:

            print("\nFirst 5 lemma/surface-form differences:")

            for item in lemma_surface_differences[:5]:
                print(item)

        # ====================================================
        # Final split status
        # ====================================================

        alignment_ok = (
            len(missing_sentences) == 0
            and len(length_mismatches) == 0
            and len(invalid_targets) == 0
        )

        if alignment_ok:

            print(
                f"\n{split_name.upper()} alignment: PASS"
            )

        else:

            print(
                f"\n{split_name.upper()} alignment: FAIL"
            )

    return {
        "split": split_name,
        "rows": len(df),
        "unique_sentences": unique_sentences,
        "missing_sentences": len(missing_sentences),
        "length_mismatches": len(length_mismatches),
        "invalid_targets": len(invalid_targets),
        "lemma_surface_differences":
            len(lemma_surface_differences),
        "alignment_ok": alignment_ok,
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "===== Gao 2018 ELMo Alignment Check ====="
    )

    results = []

    for split_name, paths in DATASETS.items():

        result = check_split(
            split_name,
            paths["csv"],
            paths["elmo"],
        )

        results.append(result)

    # ========================================================
    # Overall summary
    # ========================================================

    print("\n===== Overall Summary =====")

    all_passed = True

    for result in results:

        status = (
            "PASS"
            if result["alignment_ok"]
            else "FAIL"
        )

        print(
            f"{result['split']:>5}: "
            f"{status} | "
            f"missing={result['missing_sentences']} | "
            f"length_mismatch={result['length_mismatches']} | "
            f"invalid_target={result['invalid_targets']}"
        )

        if not result["alignment_ok"]:
            all_passed = False

    if all_passed:

        print(
            "\nAll VUA splits are aligned with "
            "the Gao 2018 ELMo representations."
        )

    else:

        print(
            "\nOne or more splits have alignment problems."
        )


if __name__ == "__main__":
    main()