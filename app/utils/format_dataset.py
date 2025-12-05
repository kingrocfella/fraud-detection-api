import json

from datasets import Dataset, load_dataset

from app.config import DATA_TRAIN_END
from app.schemas import FraudDetectionRequest


def generate_prompts_from_dataset():
    # Load the dataset
    ds = load_dataset(
        "electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset"
    )
    train = ds["train"]

    # Truncate dataset in dev environment
    if DATA_TRAIN_END:
        new_train = []
        for i in range(0, int(DATA_TRAIN_END)):
            new_train.append(train[i])
        train = new_train

    # Generate prompt from datasets
    formatted_ds = Dataset.from_list([generate_prompt(record) for record in train])

    return formatted_ds


def generate_prompt(record):
    """
    Generate a prompt for fraud detection.

    Args:
        record: Either a dict (from dataset) or FraudDetectionRequest (from API)

    Returns:
        dict with 'instruction' and optionally 'output' keys
    """

    # Helper function to get value from either dict or object
    def get_value(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    prompt = f"""
    A financial transaction occurred with the following details:
    
    - Transaction Type: {get_value(record, 'transaction_type')}
    - Location: {get_value(record, 'location')}
    - Device Used: {get_value(record, 'device_used')}
    - Payment Channel: {get_value(record, 'payment_channel')}
    - Transaction Amount: {get_value(record, 'amount_ngn')}
    - BVN Linked: {get_value(record, 'bvn_linked')}
    - New Device Transaction: {get_value(record, 'new_device_transaction')}
    - Sender Persona: {get_value(record, 'sender_persona')}
    - Is Device Shared: {get_value(record, 'is_device_shared')}
    - Is Weekend: {get_value(record, 'is_weekend')}
    - Is Salary Week: {get_value(record, 'is_salary_week')}
    - Is Night Transaction: {get_value(record, 'is_night_txn')}
    - User Transaction Count Total: {get_value(record, 'user_txn_count_total')}
    - User Transaction Frequency Last 24 Hours: {get_value(record, 'user_txn_frequency_24h')}
    - User Average Transaction Amount: {get_value(record, 'user_avg_txn_amt')}
    - User Average Gap Between Transactions: {get_value(record, 'avg_gap_between_txns')}
    - User Transaction Count Last 24 Hours: {get_value(record, 'txn_count_last_24h')}
    - User Transaction Count Last 1 Hour: {get_value(record, 'txn_count_last_1h')}
    - Total Amount Last 1 Hour: {get_value(record, 'total_amount_last_1h')}
    - IP Geo Region: {get_value(record, 'ip_geo_region')}
  

  Is this transaction fraudulent? Answer with a simple yes or no    
  """

    # For training data (dict), include the output
    if isinstance(record, dict) and "is_fraud" in record:
        return {"instruction": prompt, "output": "yes" if record["is_fraud"] else "no"}

    # For API requests, only return instruction
    return {"instruction": prompt}
