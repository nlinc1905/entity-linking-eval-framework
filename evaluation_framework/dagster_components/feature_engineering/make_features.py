import typing as t
import polars as pl
import pandas as pd
import recordlinkage as rl


def make_features(
    df: t.Union[pl.DataFrame, str],
    train_test_ratio: float,
    output_file_path_train: str,
    output_file_path_test: str,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Makes features for pairwise comparison of node attributes."""
    if isinstance(df, str):
        df = pl.read_parquet(df)

    # TODO: remove dependency on Pandas - recordlinkage requires it unfortunately
    df = df.to_pandas().set_index('index')

    # create record pairs
    # TODO: could do blocking here
    indexer = rl.Index()
    indexer.full()
    # indexer.block("first_name")
    pairs = indexer.index(df)

    # compare records on features generated on attributes
    # TODO: determine which features to build - these are just starters ripped out of the package docs
    compare_cl = rl.Compare()
    compare_cl.exact("id", "id", label="true_label")
    compare_cl.exact("first_name", "first_name", label="first_name")
    compare_cl.string("last_name", "last_name", method="jarowinkler", threshold=0.85, label="last_name")
    compare_cl.exact("birth_date", "birth_date", label="birth_date")
    compare_cl.string("email", "email", method="jarowinkler", threshold=0.85, label="email")
    features = compare_cl.compute(pairs, df, df)

    # split into train and test sets
    train = features.sample(frac=train_test_ratio, random_state=14)
    test = features[~features.index.isin(train.index)]

    # dump features
    train.to_csv(output_file_path_train, index=True)
    test.to_csv(output_file_path_test, index=True)

    return train, test
