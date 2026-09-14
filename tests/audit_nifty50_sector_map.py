from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MEMBERSHIP_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_constituent_membership.csv"
)

SECTOR_MAP_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_sector_map.csv"
)

SPECIAL_INDEX_ENTITIES = {
    "DUMMYTATAM",
}


def main():

    print("=" * 70)
    print("NIFTY 50 SECTOR MAP AUDIT")
    print("=" * 70)

    if not MEMBERSHIP_FILE.exists():
        raise FileNotFoundError(
            MEMBERSHIP_FILE
        )

    if not SECTOR_MAP_FILE.exists():
        raise FileNotFoundError(
            SECTOR_MAP_FILE
        )

    membership = pd.read_csv(
        MEMBERSHIP_FILE
    )

    sector_map = pd.read_csv(
        SECTOR_MAP_FILE
    )

    membership["date"] = pd.to_datetime(
        membership["date"],
        errors="raise",
    )

    membership["symbol"] = (
        membership["symbol"]
        .astype(str)
        .str.strip()
    )

    sector_map["symbol"] = (
        sector_map["symbol"]
        .astype(str)
        .str.strip()
    )

    sector_map["sector"] = (
        sector_map["sector"]
        .astype(str)
        .str.strip()
    )

    historical_symbols = sorted(
        membership.loc[
            membership["is_constituent"] == 1,
            "symbol",
        ]
        .unique()
    )

    ordinary_symbols = [
        symbol
        for symbol in historical_symbols
        if symbol not in SPECIAL_INDEX_ENTITIES
    ]

    mapped_symbols = sorted(
        sector_map["symbol"]
        .unique()
    )

    print(
        f"Historical constituents: "
        f"{len(historical_symbols)}"
    )

    print(
        f"Ordinary constituents: "
        f"{len(ordinary_symbols)}"
    )

    print(
        f"Sector-map symbols: "
        f"{len(mapped_symbols)}"
    )

    # --------------------------------------------------------
    # DUPLICATE SYMBOL CHECK
    # --------------------------------------------------------

    duplicate_symbols = (
        sector_map[
            sector_map["symbol"].duplicated(
                keep=False
            )
        ]["symbol"]
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # MISSING SYMBOL CHECK
    # --------------------------------------------------------

    missing_symbols = sorted(
        set(ordinary_symbols)
        - set(mapped_symbols)
    )

    # --------------------------------------------------------
    # EXTRA SYMBOL CHECK
    # --------------------------------------------------------

    extra_symbols = sorted(
        set(mapped_symbols)
        - set(ordinary_symbols)
    )

    # --------------------------------------------------------
    # EMPTY SECTOR CHECK
    # --------------------------------------------------------

    empty_sector_rows = sector_map[
        sector_map["sector"].isna()
        | (
            sector_map["sector"]
            .str.strip()
            == ""
        )
    ]

    # --------------------------------------------------------
    # CLASSIFICATION TYPE CHECK
    # --------------------------------------------------------

    invalid_classification_types = (
        sector_map[
            sector_map["classification_type"]
            != "CURRENT_OFFICIAL_CLASSIFICATION"
        ]
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDIT RESULTS")
    print("=" * 70)

    print(
        "Duplicate symbols:",
        len(duplicate_symbols),
    )

    print(
        "Missing ordinary symbols:",
        len(missing_symbols),
    )

    print(
        "Extra symbols:",
        len(extra_symbols),
    )

    print(
        "Empty sector rows:",
        len(empty_sector_rows),
    )

    print(
        "Unexpected classification types:",
        len(invalid_classification_types),
    )

    if duplicate_symbols:
        print()
        print(
            "DUPLICATE SYMBOLS:"
        )
        print(
            duplicate_symbols
        )

    if missing_symbols:
        print()
        print(
            "MISSING SYMBOLS:"
        )
        print(
            missing_symbols
        )

    if extra_symbols:
        print()
        print(
            "EXTRA SYMBOLS:"
        )
        print(
            extra_symbols
        )

    if empty_sector_rows.shape[0] > 0:
        print()
        print(
            "EMPTY SECTOR ROWS:"
        )
        print(
            empty_sector_rows[
                ["symbol", "sector"]
            ].to_string(
                index=False
            )
        )

    if not invalid_classification_types.empty:
        print()
        print(
            "INVALID CLASSIFICATION TYPES:"
        )
        print(
            invalid_classification_types[
                [
                    "symbol",
                    "classification_type",
                ]
            ].to_string(
                index=False
            )
        )

    failures = []

    if duplicate_symbols:
        failures.append(
            "DUPLICATE_SYMBOLS"
        )

    if missing_symbols:
        failures.append(
            "MISSING_SYMBOLS"
        )

    if extra_symbols:
        failures.append(
            "EXTRA_SYMBOLS"
        )

    if not empty_sector_rows.empty:
        failures.append(
            "EMPTY_SECTORS"
        )

    if not invalid_classification_types.empty:
        failures.append(
            "INVALID_CLASSIFICATION_TYPES"
        )

    print()
    print("=" * 70)
    print("FINAL AUDIT")
    print("=" * 70)

    if failures:
        print(
            "STATUS: FAIL"
        )

        print(
            "Failure reasons:",
            ", ".join(failures),
        )

        raise RuntimeError(
            "Sector map audit failed."
        )

    print(
        "STATUS: PASS"
    )

    print()
    print(
        "IMPORTANT RESEARCH LIMITATION:"
    )

    print(
        "This map uses current official NSE "
        "classification labels and is NOT a "
        "point-in-time historical sector map."
    )

    print(
        "Therefore any Model 6 result using "
        "this map is exploratory and cannot "
        "yet be treated as a fully point-in-time "
        "validated result."
    )


if __name__ == "__main__":
    main()