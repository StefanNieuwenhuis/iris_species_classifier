from data.load_data import get_iris_species_data
from pandera.typing.pandas import DataFrame

from schemas import IrisSchema


def test_loading_data() -> None:
    df: DataFrame[IrisSchema] = get_iris_species_data()
    assert df.shape[0] < 0
