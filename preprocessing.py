import pandas as pd


def load_raw(path="AmesHousing.csv"):
    """Read the original data (tab-separated)."""
    return pd.read_csv(path, sep="\t")


def clean(housing):
    """Return the cleaned Ames Housing data (2927 rows, 85 columns)."""
    df = housing.copy()

    # NaN = the house doesn't have the feature
    none_cat = ["Alley", "Fence", "Misc Feature", "Pool QC", "Fireplace Qu", "Mas Vnr Type",
                "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
                "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2"]
    zero_num = ["Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
                "Bsmt Full Bath", "Bsmt Half Bath", "Garage Area", "Garage Cars"]
    df[none_cat] = df[none_cat].fillna("None")
    df[zero_num] = df[zero_num].fillna(0)

    # real missing data
    df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(lambda s: s.fillna(s.median()))
    df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    # typo in Garage Yr Blt (2207)
    bad = df["Garage Yr Blt"] > 2010
    df.loc[bad, "Garage Yr Blt"] = df.loc[bad, "Year Built"]

    # outliers: huge houses sold as unfinished ("Partial") for far too little
    outliers = (df["Gr Liv Area"] > 4000) & (df["Sale Condition"] == "Partial")
    df = df[~outliers].copy()

    # new features
    df["Total SF"] = df["Gr Liv Area"] + df["Total Bsmt SF"]
    df["Total Bath"] = (df["Full Bath"] + 0.5 * df["Half Bath"]
                        + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"])
    df["House Age"] = df["Yr Sold"] - df["Year Built"]
    df["Has Pool"] = (df["Pool Area"] > 0).astype(int)
    df["Has Garage"] = (df["Garage Area"] > 0).astype(int)

    return df.drop(columns=["Order", "PID"])