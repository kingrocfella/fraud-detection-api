from typing import Literal

from pydantic import BaseModel


class DefaultResponse(BaseModel):
    response: str
    status: str = "success"


class FraudDetectionRequest(BaseModel):
    transaction_type: Literal["deposit", "withdrawal", "payment", "transfer"]
    location: str
    device_used: Literal["mobile", "web", "atm", "pos"]
    payment_channel: Literal["Bank Transfer", "Mobile App", "USSD", "Card"]
    amount_ngn: float
    bvn_linked: bool
    new_device_transaction: bool
    sender_persona: Literal["Student", "Salary Earner", "Trader"]
    is_weekend: Literal[0, 1]
    is_salary_week: Literal[0, 1]
    is_night_txn: Literal[0, 1]
    is_device_shared: Literal[0, 1]
    user_txn_count_total: int
    user_avg_txn_amt: float
    user_txn_frequency_24h: int
    txn_count_last_24h: int
    txn_count_last_1h: int
    total_amount_last_1h: float
    avg_gap_between_txns: float
    ip_geo_region: Literal[
        "South East",
        "North West",
        "North East",
        "South West",
        "North Central",
        "South South",
    ]
