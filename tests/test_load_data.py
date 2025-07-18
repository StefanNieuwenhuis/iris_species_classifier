import pytest
import pandas as pd
import sqlite3

from pathlib import Path
from data.load_data import get_iris_species_data


@pytest.fixture
def mock_iris_db(tmp_path: Path) -> Path:
    """Create in-memory SQLite DB and populate with dummy Iris data"""

    db_path = tmp_path / "mock_iris.sqlite"
    conn = sqlite3.connect(db_path)

    df = pd.DataFrame(
        {
            "Id": [1, 2],
            "SepalLengthCm": [5.1, 4.9],
            "SepalWidthCm": [3.5, 3.0],
            "PetalLengthCm": [1.4, 1.4],
            "PetalWidthCm": [0.2, 0.2],
            "Species": ["Iris-setosa", "Iris-setosa"],
        }
    )

    df.to_sql("Iris", conn, index=False, if_exists="replace")
    conn.close()

    return db_path


@pytest.mark.usefixtures("mock_iris_db")
def test_loading_data(monkeypatch: pytest.MonkeyPatch, mock_iris_db: Path) -> None:
    # Patch the RAW_DATA_PATH config path
    monkeypatch.setenv("IRIS_DB_PATH", str(mock_iris_db))

    df = get_iris_species_data()

    assert not df.empty
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    assert list(df.columns) == [
        "Id",
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
        "Species",
    ]
