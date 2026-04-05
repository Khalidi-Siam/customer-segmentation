"""
Why this file exists:
- This module is the validation step between data ingestion and data transformation.
- In a full ML pipeline, it checks dataset schema, missing columns, null values,
	duplicate rows, unexpected data types, and out-of-range values before training.

How it works in the pipeline:
1. Data ingestion saves raw/train/test CSV files.
2. Data validation verifies those files are safe and consistent.
3. Data transformation runs only on validated data.
4. Model training uses transformed, trustworthy data.

Current project note:
- For the current Mall_Customers dataset, validation is not required right now
	because the dataset is already clean and stable.
- This file is kept as a placeholder so validation logic can be added later
	without changing the overall pipeline design.
"""

