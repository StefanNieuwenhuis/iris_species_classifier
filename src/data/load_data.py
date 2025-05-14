import sqlite3
import pandas as pd
import pandera as pa

from pandera.typing.pandas import DataFrame

from config import IRIS_TABLE_NAME, get_raw_data_path
from schemas import IrisSchema


@pa.check_types
def get_iris_species_data() -> DataFrame[IrisSchema]:
    """
    Load Iris dataset from sqlite database

    Returns
    -------
    df : DataFrame[IrisSchema]
         Returns a dataframe with the IrisSchema schema
    """

    conn = sqlite3.connect(get_raw_data_path())
    query = f"SELECT * FROM {IRIS_TABLE_NAME}"
    df = pd.read_sql(query, conn)
    conn.close()

    return IrisSchema.validate(df)
