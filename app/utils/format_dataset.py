from datasets import Dataset, load_dataset

from app.config import DATA_TRAIN_END, logger


def generate_prompts_from_dataset():
    """Generate prompts from dataset."""
    logger.info("Loading dataset...")
    ds = load_dataset(
        "electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset"
    )
    logger.info("Dataset loaded successfully")
    train = ds["train"]

    # Truncate dataset in dev environment
    if DATA_TRAIN_END:
        limit = min(int(DATA_TRAIN_END), len(train))
        # Hugging Face datasets support efficient slicing via select
        train = train.select(range(limit))

    logger.info("Truncating dataset to %d samples", len(train))

    # Generate prompt from datasets
    logger.info("Generating prompts from dataset...")
    formatted_ds = Dataset.from_list(
        [
            generate_prompt(record, record_number=i, total_records=len(train))
            for i, record in enumerate(train)
        ]
    )
    logger.info("Prompts generated successfully")
    return formatted_ds


def generate_prompt(record, record_number=None, total_records=1, state_reasoning=None):
    """
    Generate a prompt for fraud detection.

    Args:
        record: Either a dict (from dataset) or FraudDetectionRequest (from API)
        record_number: Optional record number/index for logging purposes
        total_records: Total number of records for progress calculation
        state_reasoning: Whether to include state reasoning in the prompt
    Returns:
        dict with 'instruction' and optionally 'output' keys
    """

    if record_number is not None and total_records > 0:
        if record_number % 5000 == 0:
            progress = int(((record_number + 1) / total_records) * 100)
            logger.info(
                "Progress: %d%% (%d/%d)", progress, record_number + 1, total_records
            )
    else:
        logger.info("Generating prompt for API request")

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
  

  Is this transaction fraudulent? Answer with a simple yes or no.    
  {state_reasoning if state_reasoning else ""}
  """

    # For training data (dict), include the output
    if isinstance(record, dict) and "is_fraud" in record:
        if record_number is not None:
            if record_number % 5000 == 0:
                logger.info(
                    "Prompt generated successfully for record number: %d", record_number
                )
        else:
            logger.info("Prompt generated successfully")
        return {"instruction": prompt, "output": "yes" if record["is_fraud"] else "no"}

    # For API requests, only return instruction
    logger.info("Prompt generated successfully")
    return {"instruction": prompt}
