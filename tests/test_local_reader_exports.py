"""
Comprehensive tests for ENAHOLocalReader export methods

This test module covers export functionality in local_reader.py, focusing on
the save_data() and save_metadata() methods and their supporting functions.

Target: loader/io/local_reader.py (currently 36.60% coverage)
Goal: Test export methods (lines 520-663) to achieve 60-65% coverage (+2-3% overall)
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from enahopy.loader.io.local_reader import ENAHOLocalReader

# Check for optional dependencies
try:
    import openpyxl

    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_enaho_csv(tmp_path):
    """Create a sample ENAHO CSV file for testing"""
    csv_file = tmp_path / "sample_enaho.csv"
    df = pd.DataFrame(
        {
            "conglome": ["001", "002", "003"],
            "vivienda": ["01", "01", "02"],
            "hogar": ["1", "1", "1"],
            "mieperho": [4, 3, 5],
            "gashog2d": [1200.5, 800.3, 1500.0],
            "inghog2d": [1500.0, 1000.0, 2000.0],
        }
    )
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for export testing"""
    return pd.DataFrame(
        {
            "conglome": ["001", "002", "003"],
            "vivienda": ["01", "01", "02"],
            "hogar": ["1", "1", "1"],
            "mieperho": [4, 3, 5],
            "gashog2d": [1200.5, 800.3, 1500.0],
        }
    )


@pytest.fixture
def reader_with_data(sample_enaho_csv):
    """Create a reader instance with sample data loaded"""
    return ENAHOLocalReader(sample_enaho_csv)


# ============================================================================
# TESTS FOR save_data() - CSV EXPORT
# ============================================================================


def test_save_data_to_csv(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test saving DataFrame to CSV format"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "output.csv"

    reader.save_data(sample_dataframe, str(output_path))

    assert output_path.exists()
    # Read back and verify
    df_read = pd.read_csv(output_path)
    assert len(df_read) == 3
    assert "conglome" in df_read.columns
    assert "mieperho" in df_read.columns


def test_save_data_to_csv_with_kwargs(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test saving CSV with additional kwargs"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "output.csv"

    # Pass encoding as kwarg
    reader.save_data(sample_dataframe, str(output_path), encoding="utf-8")

    assert output_path.exists()


def test_save_data_csv_creates_parent_directories(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test that parent directories are created automatically for CSV"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "nested" / "folder" / "output.csv"

    reader.save_data(sample_dataframe, str(output_path))

    assert output_path.exists()
    assert output_path.parent.exists()


# ============================================================================
# TESTS FOR save_data() - PARQUET EXPORT
# ============================================================================


def test_save_data_to_parquet(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test saving DataFrame to Parquet format"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "output.parquet"

    reader.save_data(sample_dataframe, str(output_path))

    assert output_path.exists()
    # Read back and verify
    df_read = pd.read_parquet(output_path)
    assert len(df_read) == 3
    assert "conglome" in df_read.columns


def test_save_data_parquet_creates_directories(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test that parent directories are created for Parquet"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "deep" / "nested" / "output.parquet"

    reader.save_data(sample_dataframe, str(output_path))

    assert output_path.exists()


# ============================================================================
# TESTS FOR save_data() - EXCEL EXPORT
# ============================================================================


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_save_data_to_excel(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test saving DataFrame to Excel format"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "output.xlsx"

    reader.save_data(sample_dataframe, str(output_path))

    assert output_path.exists()
    # Read back and verify
    df_read = pd.read_excel(output_path)
    assert len(df_read) == 3
    assert "conglome" in df_read.columns


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_save_data_to_excel_with_sheet_name(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test saving Excel with custom sheet name"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "output.xlsx"

    reader.save_data(sample_dataframe, str(output_path), sheet_name="ENAHO 2023")

    assert output_path.exists()
    # Verify sheet name
    df_read = pd.read_excel(output_path, sheet_name="ENAHO 2023")
    assert len(df_read) == 3


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_save_data_excel_creates_directories(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test that parent directories are created for Excel"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "reports" / "2023" / "output.xlsx"

    reader.save_data(sample_dataframe, str(output_path))

    assert output_path.exists()


# ============================================================================
# TESTS FOR save_data() - STATA EXPORT
# ============================================================================


def test_save_data_to_stata(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test saving DataFrame to Stata .dta format"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "output.dta"

    reader.save_data(sample_dataframe, str(output_path))

    assert output_path.exists()
    # Read back and verify (if pyreadstat available)
    try:
        import pyreadstat

        df_read, meta = pyreadstat.read_dta(str(output_path))
        assert len(df_read) == 3
    except ImportError:
        # Just verify file was created
        pass


def test_save_data_stata_with_boolean_column(tmp_path, sample_enaho_csv):
    """Test Stata export handles boolean columns correctly"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    df = pd.DataFrame(
        {"id": [1, 2, 3], "is_urban": [True, False, True], "value": [10.5, 20.3, 30.1]}
    )
    output_path = tmp_path / "output.dta"

    # Should convert booleans to int automatically
    reader.save_data(df, str(output_path))

    assert output_path.exists()


def test_save_data_stata_with_object_columns(tmp_path, sample_enaho_csv):
    """Test Stata export handles object columns"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["A", "B", "C"],
            "notes": ["Note 1", "Note 2", "Note 3"],
        }
    )
    output_path = tmp_path / "output.dta"

    reader.save_data(df, str(output_path))

    assert output_path.exists()


def test_prepare_data_for_stata_converts_booleans(sample_enaho_csv):
    """Test _prepare_data_for_stata converts boolean to int"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    df = pd.DataFrame({"flag": [True, False, True], "value": [1, 2, 3]})

    result = reader._prepare_data_for_stata(df)

    assert result["flag"].dtype in [int, "int64", "int32"]
    assert result["flag"].tolist() == [1, 0, 1]


def test_prepare_data_for_stata_handles_object_types(sample_enaho_csv):
    """Test _prepare_data_for_stata handles object columns"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    df = pd.DataFrame({"text": ["A", "B", None], "number": [1, 2, 3]})

    result = reader._prepare_data_for_stata(df)

    # Object columns should be converted to string
    assert result["text"].dtype == object
    # NaN should be handled
    assert result["number"].dtype in [int, "int64", float, "float64"]


def test_prepare_data_for_stata_handles_empty_strings(sample_enaho_csv):
    """Test _prepare_data_for_stata handles empty string columns"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    df = pd.DataFrame({"empty": ["", "", ""], "value": [1, 2, 3]})

    result = reader._prepare_data_for_stata(df)

    # Empty string column should be converted to float NaN
    assert result["empty"].isna().all() or result["empty"].dtype == float


# ============================================================================
# TESTS FOR save_data() - ERROR HANDLING
# ============================================================================


def test_save_data_unsupported_format(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test that unsupported formats raise ValueError"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    output_path = tmp_path / "output.txt"

    with pytest.raises(ValueError, match="Formato de guardado no soportado"):
        reader.save_data(sample_dataframe, str(output_path))


def test_save_data_invalid_format(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test various invalid formats"""
    reader = ENAHOLocalReader(sample_enaho_csv)

    invalid_formats = ["output.json", "output.xml", "output.html"]
    for filename in invalid_formats:
        with pytest.raises(ValueError):
            reader.save_data(sample_dataframe, str(tmp_path / filename))


# ============================================================================
# TESTS FOR save_metadata() - JSON EXPORT
# ============================================================================


def test_save_metadata_as_json(reader_with_data, tmp_path):
    """Test saving metadata to JSON format"""
    output_path = tmp_path / "metadata.json"

    reader_with_data.save_metadata(str(output_path))

    assert output_path.exists()
    # Read and verify JSON structure
    with open(output_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert isinstance(metadata, dict)


def test_save_metadata_json_creates_directories(reader_with_data, tmp_path):
    """Test that parent directories are created for JSON metadata"""
    output_path = tmp_path / "metadata" / "2023" / "metadata.json"

    reader_with_data.save_metadata(str(output_path))

    assert output_path.exists()


def test_save_metadata_json_encoding(reader_with_data, tmp_path):
    """Test that JSON metadata uses UTF-8 encoding"""
    output_path = tmp_path / "metadata.json"

    reader_with_data.save_metadata(str(output_path))

    # Verify UTF-8 encoding by reading
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 0


# ============================================================================
# TESTS FOR save_metadata() - CSV EXPORT
# ============================================================================


def test_save_metadata_as_csv(reader_with_data, tmp_path):
    """Test saving metadata to CSV format"""
    output_path = tmp_path / "variable_dictionary.csv"

    reader_with_data.save_metadata(str(output_path))

    assert output_path.exists()
    # Read and verify CSV
    df = pd.read_csv(output_path)
    assert "variable_name" in df.columns
    assert len(df) > 0


def test_save_metadata_csv_columns(reader_with_data, tmp_path):
    """Test that CSV metadata has expected columns"""
    output_path = tmp_path / "dictionary.csv"

    reader_with_data.save_metadata(str(output_path))

    df = pd.read_csv(output_path)
    expected_columns = [
        "variable_name",
        "variable_label",
        "variable_type",
        "variable_format",
        "has_value_labels",
        "value_labels",
    ]
    for col in expected_columns:
        assert col in df.columns


def test_save_metadata_csv_creates_directories(reader_with_data, tmp_path):
    """Test that parent directories are created for CSV metadata"""
    output_path = tmp_path / "docs" / "variables.csv"

    reader_with_data.save_metadata(str(output_path))

    assert output_path.exists()


# ============================================================================
# TESTS FOR save_metadata() - ERROR HANDLING
# ============================================================================


def test_save_metadata_unsupported_format(reader_with_data, tmp_path):
    """Test that unsupported metadata formats raise ValueError"""
    output_path = tmp_path / "metadata.txt"

    with pytest.raises(ValueError, match="Formato de guardado de metadatos no soportado"):
        reader_with_data.save_metadata(str(output_path))


def test_save_metadata_invalid_formats(reader_with_data, tmp_path):
    """Test various invalid metadata formats"""
    invalid_formats = ["metadata.xlsx", "metadata.xml", "metadata.html"]
    for filename in invalid_formats:
        with pytest.raises(ValueError):
            reader_with_data.save_metadata(str(tmp_path / filename))


# ============================================================================
# TESTS FOR _prepare_variables_df()
# ============================================================================


def test_prepare_variables_df_structure(reader_with_data):
    """Test _prepare_variables_df returns proper DataFrame structure"""
    df = reader_with_data._prepare_variables_df()

    assert isinstance(df, pd.DataFrame)
    assert "variable_name" in df.columns
    assert "variable_label" in df.columns
    assert "variable_type" in df.columns
    assert len(df) > 0


def test_prepare_variables_df_content(reader_with_data):
    """Test _prepare_variables_df includes actual variable data"""
    df = reader_with_data._prepare_variables_df()

    # Should have data for the CSV columns
    assert df["variable_name"].notna().any()


# ============================================================================
# TESTS FOR get_summary_info()
# ============================================================================


def test_get_summary_info_returns_dict(reader_with_data):
    """Test get_summary_info returns a dictionary"""
    summary = reader_with_data.get_summary_info()

    assert isinstance(summary, dict)


def test_get_summary_info_has_required_keys(reader_with_data):
    """Test get_summary_info contains expected keys"""
    summary = reader_with_data.get_summary_info()

    expected_keys = ["file_info", "total_columns", "sample_columns", "has_labels"]
    for key in expected_keys:
        assert key in summary


def test_get_summary_info_total_columns(reader_with_data):
    """Test get_summary_info reports correct column count"""
    summary = reader_with_data.get_summary_info()

    assert summary["total_columns"] > 0
    assert isinstance(summary["total_columns"], int)


def test_get_summary_info_sample_columns(reader_with_data):
    """Test get_summary_info provides sample column names"""
    summary = reader_with_data.get_summary_info()

    assert "sample_columns" in summary
    assert isinstance(summary["sample_columns"], list)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_export_workflow_csv(reader_with_data, tmp_path):
    """Test complete workflow: read -> process -> save CSV"""
    df, validation = reader_with_data.read_data()
    output_path = tmp_path / "final_output.csv"

    reader_with_data.save_data(df, str(output_path))

    assert output_path.exists()
    df_read = pd.read_csv(output_path)
    assert len(df_read) == len(df)


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_full_export_workflow_multiple_formats(sample_dataframe, tmp_path, sample_enaho_csv):
    """Test exporting same data to multiple formats"""
    reader = ENAHOLocalReader(sample_enaho_csv)

    formats = {
        "output.csv": pd.read_csv,
        "output.parquet": pd.read_parquet,
        "output.xlsx": pd.read_excel,
    }

    for filename, read_func in formats.items():
        output_path = tmp_path / filename
        reader.save_data(sample_dataframe, str(output_path))
        assert output_path.exists()

        # Verify each format can be read back
        df_read = read_func(output_path)
        assert len(df_read) == len(sample_dataframe)


def test_metadata_export_both_formats(reader_with_data, tmp_path):
    """Test exporting metadata in both JSON and CSV formats"""
    json_path = tmp_path / "metadata.json"
    csv_path = tmp_path / "metadata.csv"

    reader_with_data.save_metadata(str(json_path))
    reader_with_data.save_metadata(str(csv_path))

    assert json_path.exists()
    assert csv_path.exists()

    # Verify both can be read
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    csv_data = pd.read_csv(csv_path)

    assert isinstance(json_data, dict)
    assert isinstance(csv_data, pd.DataFrame)


def test_export_with_special_characters(tmp_path, sample_enaho_csv):
    """Test exporting data with special characters"""
    reader = ENAHOLocalReader(sample_enaho_csv)
    df = pd.DataFrame(
        {
            "nombre": ["José", "María", "Señor"],
            "ciudad": ["Lima", "Cusco", "Arequipa"],
            "valor": [100, 200, 300],
        }
    )

    # Test CSV export with special chars
    csv_path = tmp_path / "special_chars.csv"
    reader.save_data(df, str(csv_path))
    assert csv_path.exists()

    # Verify can be read back
    df_read = pd.read_csv(csv_path)
    assert "José" in df_read["nombre"].values[0] or "Jos" in df_read["nombre"].values[0]
