import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

L3_DIR = Path(__file__).resolve().parent
ROOT = L3_DIR.parent
DATA_PATH = ROOT / "AmesHousing.csv"

sys.path.append(str(ROOT / "L1"))
from preprocessing import load_raw, clean  # noqa: E402


def clean_for_clustering(housing):
    """L1 cleaning + filling the last NaNs, because PCA and k-means can't handle missing values."""
    df = clean(housing)
    df["Garage Yr Blt"] = df["Garage Yr Blt"].fillna(df["Year Built"])
    return df


def load_data():
    """Cleaned data without the price + the price separately (only to describe clusters later)."""
    df = clean_for_clustering(load_raw(DATA_PATH))
    price = df["SalePrice"].copy()
    return df.drop(columns=["SalePrice"]), price


QUAL = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {
    "Lot Shape":      {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Utilities":      {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
    "Land Slope":     {"Sev": 0, "Mod": 1, "Gtl": 2},
    "Exter Qual": QUAL, "Exter Cond": QUAL, "Bsmt Qual": QUAL, "Bsmt Cond": QUAL,
    "Heating QC": QUAL, "Kitchen Qual": QUAL, "Fireplace Qu": QUAL,
    "Garage Qual": QUAL, "Garage Cond": QUAL, "Pool QC": QUAL,
    "Bsmt Exposure":  {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFin Type 1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFin Type 2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Electrical":     {"Mix": 0, "FuseP": 1, "FuseF": 2, "FuseA": 3, "SBrkr": 4},
    "Functional":     {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Garage Finish":  {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Paved Drive":    {"N": 0, "P": 1, "Y": 2},
    "Fence":          {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}


def encode(df, columns=None):
    """Ordinal encoding for ordered columns, one-hot for the rest.
    columns: the column list from training. Use it for new houses, so they get exactly the same columns."""
    X = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        X[col] = X[col].map(mapping)
    nominal = X.select_dtypes(exclude="number").columns
    X = pd.get_dummies(X, columns=nominal, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)   # same columns, same order as in training
    return X


# continuous area / money columns with a skewed distribution (skewness > 0.75)
# only these are logged: scores, counts, years and 0/1 columns are left as they are
LOG_COLUMNS = [
    "Lot Frontage", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Wood Deck SF", "Open Porch SF",
    "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area", "Misc Val",
]


def apply_log(X):
    """log(1 + x) on the skewed area / money columns (log1p because many of them are 0)."""
    X = X.copy()
    X[LOG_COLUMNS] = np.log1p(X[LOG_COLUMNS])
    return X


def make_scaler(name):
    """'minmax' -> every column in [0, 1]; 'standard' -> every column with mean 0 and std 1."""
    if name == "minmax":
        return MinMaxScaler()
    if name == "standard":
        return StandardScaler()
    raise ValueError(f"unknown scaler: {name}")