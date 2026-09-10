import pandera.pandas as pa
from pandera.typing import Series


class RawChurnSchema(pa.DataFrameModel):
    customerID: Series[str] = pa.Field(unique=True, nullable=False)
    gender: Series[str] = pa.Field(nullable=False, isin=["Female", "Male"])
    SeniorCitizen: Series[int] = pa.Field(nullable=False, isin=[0, 1])
    Partner: Series[str] = pa.Field(nullable=False, isin=["Yes", "No"])
    Dependents: Series[str] = pa.Field(nullable=False, isin=["Yes", "No"])
    tenure: Series[int] = pa.Field(nullable=False, ge=0)
    PhoneService: Series[str] = pa.Field(nullable=False, isin=["Yes", "No"])
    MultipleLines: Series[str] = pa.Field(
        nullable=False, isin=["Yes", "No", "No phone service"]
    )
    InternetService: Series[str] = pa.Field(
        nullable=False, isin=["DSL", "Fiber optic", "No"]
    )
    OnlineSecurity: Series[str] = pa.Field(
        nullable=False, isin=["Yes", "No", "No internet service"]
    )
    OnlineBackup: Series[str] = pa.Field(
        nullable=False, isin=["Yes", "No", "No internet service"]
    )
    DeviceProtection: Series[str] = pa.Field(
        nullable=False, isin=["Yes", "No", "No internet service"]
    )
    TechSupport: Series[str] = pa.Field(
        nullable=False, isin=["Yes", "No", "No internet service"]
    )
    StreamingTV: Series[str] = pa.Field(
        nullable=False, isin=["Yes", "No", "No internet service"]
    )
    StreamingMovies: Series[str] = pa.Field(
        nullable=False, isin=["Yes", "No", "No internet service"]
    )
    Contract: Series[str] = pa.Field(
        nullable=False, isin=["Month-to-month", "One year", "Two year"]
    )
    PaperlessBilling: Series[str] = pa.Field(nullable=False, isin=["Yes", "No"])
    PaymentMethod: Series[str] = pa.Field(
        nullable=False,
        isin=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    MonthlyCharges: Series[float] = pa.Field(nullable=False, ge=0)
    TotalCharges: Series[float] = pa.Field(nullable=False, ge=0)
    Churn: Series[str] = pa.Field(nullable=False, isin=["Yes", "No"])

    class Config:
        coerce = True  # Automatically convert data types to match the schema
        strict = True  # Raise an error if unexpected columns are present
