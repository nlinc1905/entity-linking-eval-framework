import typing as t
import polars as pl
import recordlinkage as rl


def make_features(
    df: t.Union[pl.DataFrame, str],
    train_test_ratio: float,
    output_file_path_train: str,
    output_file_path_test: str,
) -> None:
    """Makes features for pairwise comparison of node attributes."""
    if isinstance(df, str):
        df = pl.read_parquet(df)

    # split into train and test sets
    df = df.sample(n=len(df), with_replacement=False, shuffle=True, seed=14)
    train_test_split_index = int(len(df) * train_test_ratio)
    # TODO: remove dependency on Pandas - recordlinkage requires it unfortunately
    train = df[:train_test_split_index].to_pandas().set_index('index')
    test = df[train_test_split_index:].to_pandas().set_index('index')

    # create record pairs
    # TODO: could do blocking here
    indexer = rl.Index()
    indexer.full()
    # indexer.block("first_name")
    train_pairs = indexer.index(train)
    test_pairs = indexer.index(test)

    # compare records on features generated on attributes
    # TODO: determine which features to build - these are just starters ripped out of the package docs
    compare_cl = rl.Compare()
    compare_cl.exact("id", "id", label="true_label")
    compare_cl.exact("first_name", "first_name", label="first_name")
    compare_cl.string("last_name", "last_name", method="jarowinkler", threshold=0.85, label="last_name")
    compare_cl.exact("birth_date", "birth_date", label="birth_date")
    compare_cl.string("email", "email", method="jarowinkler", threshold=0.85, label="email")
    train_features = compare_cl.compute(train_pairs, train, train)
    test_features = compare_cl.compute(test_pairs, test, test)

    # dump features
    train_features.to_csv(output_file_path_train, index=True)
    test_features.to_csv(output_file_path_test, index=True)


if __name__ == "__main__":
    INPUT_FILE_PATH = "eval_data/raw-1000-1_0-0_5.parquet"
    TRAIN_TEST_RATIO = 0.8
    OUTPUT_FILE_PATH_TRAIN = INPUT_FILE_PATH.replace("raw", "train").replace("parquet", "csv")
    OUTPUT_FILE_PATH_TEST = INPUT_FILE_PATH.replace("raw", "test").replace("parquet", "csv")

    make_features(
        df=INPUT_FILE_PATH,
        train_test_ratio=TRAIN_TEST_RATIO,
        output_file_path_train=OUTPUT_FILE_PATH_TRAIN,
        output_file_path_test=OUTPUT_FILE_PATH_TEST,
    )
