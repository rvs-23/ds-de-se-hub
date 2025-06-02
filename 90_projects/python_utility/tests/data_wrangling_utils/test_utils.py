import numpy as np
import pandas as pd
import pytest

# from python_utility.data_wrangling_utils.utils import find_problematic_cols_df
# from data_wrangling_utils.utils import find_problematic_cols_df~


# If running pytest from the root of 'ds-de-se-repo' and 'projects' is a directory,
# you might need to add 'projects' to sys.path or install 'utility_scripts' as a package.
# For simplicity, assuming pytest can find it or 'utility_scripts' is installed.


# Helper to compare results, as order of columns within lists matters
# and should match the original DataFrame's column order.
def assert_results_equal(result, expected_constant, expected_null):
    assert sorted(result.get("constant_cols", [])) == sorted(expected_constant)
    assert sorted(result.get("all_null_cols", [])) == sorted(expected_null)


def test_basic_scenario():
    """Test with a mix of constant, null, and normal columns."""
    data = {
        "A": [1, 1, 1, 1],
        "B": [np.nan, np.nan, np.nan, np.nan],
        "C": [1, 2, 3, 4],
        "D": ["x", "x", np.nan, "x"],  # Constant with a NaN
        "E": [True, True, True, True],
    }
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=["A", "D", "E"], expected_null=["B"])


def test_only_constant_columns():
    """Test when all columns are constant."""
    data = {"A": [5, 5, 5], "B": ["z", "z", "z"]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=["A", "B"], expected_null=[])


def test_only_null_columns():
    """Test when all columns are null."""
    data = {"A": [np.nan, np.nan], "B": [None, None]}  # None also treated as NaN by pandas
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=[], expected_null=["A", "B"])


def test_no_problematic_columns():
    """Test with a DataFrame having no constant or all-null columns."""
    data = {"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [1.1, 2.2, np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=[], expected_null=[])


def test_empty_dataframe():
    """Test with an empty DataFrame."""
    df = pd.DataFrame()
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=[], expected_null=[])


def test_ignore_cols_functional():
    """Test the ignore_cols parameter functionality."""
    data = {"A": [1, 1], "B": [np.nan, np.nan], "C": [1, 2]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, ignore_cols=["A", "B"])
    assert_results_equal(result, expected_constant=[], expected_null=[])

    result_ignore_one_const = find_problematic_cols_df(df, ignore_cols=["A"])
    assert_results_equal(result_ignore_one_const, expected_constant=[], expected_null=["B"])

    result_ignore_normal = find_problematic_cols_df(df, ignore_cols=["C"])
    assert_results_equal(result_ignore_normal, expected_constant=["A"], expected_null=["B"])


def test_ignore_cols_all_columns_ignored():
    """Test when all columns are in ignore_cols."""
    data = {"A": [1, 1], "B": [np.nan, np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, ignore_cols=["A", "B"])
    assert_results_equal(result, expected_constant=[], expected_null=[])


def test_ignore_cols_raises_value_error():
    """Test that ignore_cols raises ValueError for non-existent columns."""
    data = {"A": [1, 1]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError) as excinfo:
        find_problematic_cols_df(df, ignore_cols=["Z"])
    assert "Columns in 'ignore_cols' not found in DataFrame: ['Z']" in str(excinfo.value)


def test_check_constant_false():
    """Test when check_constant is False."""
    data = {"A": [1, 1], "B": [np.nan, np.nan], "C": [1, 2]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, check_constant=False)
    assert_results_equal(result, expected_constant=[], expected_null=["B"])


def test_check_all_null_false():
    """Test when check_all_null is False."""
    data = {"A": [1, 1], "B": [np.nan, np.nan], "C": [1, 2]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, check_all_null=False)
    assert_results_equal(result, expected_constant=["A"], expected_null=[])


def test_both_checks_false():
    """Test when both check_constant and check_all_null are False."""
    data = {"A": [1, 1], "B": [np.nan, np.nan], "C": [1, 2]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, check_constant=False, check_all_null=False)
    assert_results_equal(result, expected_constant=[], expected_null=[])


def test_constant_col_with_nans():
    """Test a column that is constant among non-NaN values but also contains NaNs."""
    data = {"A": [10, np.nan, 10, 10, np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    # Such a column IS considered constant by the nunique(dropna=True) == 1 logic
    assert_results_equal(result, expected_constant=["A"], expected_null=[])


def test_dataframe_with_single_row():
    """Test DataFrame with a single row (all columns will appear constant)."""
    data = {"A": [1], "B": ["hello"], "C": [np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=["A", "B"], expected_null=["C"])


def test_dataframe_with_only_one_column_constant():
    data = {"A": [1, 1, 1]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=["A"], expected_null=[])


def test_dataframe_with_only_one_column_null():
    data = {"A": [np.nan, np.nan, np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=[], expected_null=["A"])
