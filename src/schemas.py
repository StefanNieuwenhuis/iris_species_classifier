from pandera import DataFrameModel, Field
from pandera.typing.pandas import Series


class IrisSchema(DataFrameModel):
    """
    DataFrame Schema for the Iris dataset
    """

    Id: Series[int] = Field(gt=0)
    SepalLengthCm: Series[float] = Field(gt=0)
    SepalWidthCm: Series[float] = Field(gt=0)
    PetalLengthCm: Series[float] = Field(gt=0)
    PetalWidthCm: Series[float] = Field(gt=0)
    Species: Series[str]
