import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def calculate_luhn_check_digit(npi_9digits: str) -> str:
    digits = [int(d) for d in npi_9digits]
    sum_ = 0
    parity = (len(digits) + 1) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            doubled = digit * 2
            sum_ += doubled - 9 if doubled > 9 else doubled
        else:
            sum_ += digit
    check_digit = (10 - (sum_ % 10)) % 10
    return str(check_digit)

def generate_valid_npi() -> str:
    first_digit = random.choice(['1', '2'])
    middle_digits = ''.join(str(random.randint(0, 9)) for _ in range(8))
    npi_9 = first_digit + middle_digits
    check_digit = calculate_luhn_check_digit(npi_9)
    return npi_9 + check_digit

def generate_invalid_npi() -> str:
    choice_type = random.choice(['wrong_first', 'wrong_check', 'non_numeric'])
    if choice_type == 'wrong_first':
        first_digit = random.choice([str(x) for x in range(3, 10)])
        middle_digits = ''.join(str(random.randint(0, 9)) for _ in range(8))
        npi_9 = first_digit + middle_digits
        check_digit = calculate_luhn_check_digit(npi_9)
        return npi_9 + check_digit
    elif choice_type == 'wrong_check':
        first_digit = random.choice(['1', '2'])
        middle_digits = ''.join(str(random.randint(0, 9)) for _ in range(8))
        npi_9 = first_digit + middle_digits
        correct_check = calculate_luhn_check_digit(npi_9)
        wrong_check_digit = str((int(correct_check) + random.randint(1, 9)) % 10)
        return npi_9 + wrong_check_digit
    else:
        invalid = ''.join(random.choice('ABCDEF0123456789') for _ in range(10))
        return invalid

def generate_date(start_date, end_date):
    delta = (end_date - start_date).days
    random_day = random.randint(0, delta)
    dt = start_date + timedelta(days=random_day)
    return f"{dt.month}/{dt.day}/{dt.year}"

def create_dataset(num_rows, max_invalid_missing, start_date, end_date):
    call_types = [
        "Detail Only", "Detail with Sample", "Call Only", "Group Detail", "Sample Only"
    ]
    call_record_types = [
        "Non-Sampled Call", "Sampled Call", "POC Call Report", "Marketing Event", "HCP Call", 
        "Group Training", "P2P Call", "SAM HCO Call Report", "Event Call Report"
    ]
    account_target_types = [
        "DSA NEXT BEST", "DSS TOP 30", "NON-TARGET", "DSA TOP 30", "DSA NT AFFILIATE"
    ]
    territories = [
        "3101A1", "3102A1", "3103A1", "3201S1", "3202S1",
        "3203S1", "3301A1", "3302A1", "3303A1", "3401S1",
        "3402S1", "3501A1", "3502S1", "3601A1", "3602S1"
    ]

    rows = []
    valid_npis_pool = [generate_valid_npi() for _ in range(1000)]

    invalid_missing_count = 0

    for _ in range(num_rows):
        # Decide pattern of missingness:
        # Ensure total invalid+missing <= max_invalid_missing
        if invalid_missing_count < max_invalid_missing:
            pattern = random.choices(
                population=[1, 2, 3, 4],
                weights=[0.7, 0.1, 0.1, 0.1],
                k=1
            )[0]
        else:
            pattern = 1  # Force valid if limit reached

        if pattern == 2 or pattern == 4:
            account_npi = None
            invalid_missing_count += 1
        else:
            npi_type = random.choices(['valid', 'invalid'], weights=[0.85, 0.15], k=1)[0]
            if npi_type == 'valid':
                account_npi = random.choice(valid_npis_pool)
            else:
                if invalid_missing_count < max_invalid_missing:
                    account_npi = generate_invalid_npi()
                    invalid_missing_count += 1
                else:
                    account_npi = random.choice(valid_npis_pool)

        if pattern == 3 or pattern == 4:
            date_val = None
            invalid_missing_count += 1
        else:
            date_val = generate_date(start_date, end_date)

        rows.append({
            "account_npi": account_npi,
            "call_type": random.choice(call_types),
            "call_record_type": random.choice(call_record_types),
            "date": date_val,
            "account_target_type": random.choice(account_target_types),
            "territory": random.choice(territories),
        })

    df = pd.DataFrame(rows)
    return df

def main():
    num_rows_per_file = 60000
    start_date = datetime.strptime("1/1/2024", "%m/%d/%Y")
    end_date = datetime.strptime("12/31/2024", "%m/%d/%Y")

    for i in range(1, 4):
        max_invalid_missing = random.choice([1000, 2000, 3000, 4000])
        print(f"Generating file {i} with max {max_invalid_missing} invalid/missing rows...")
        df = create_dataset(num_rows_per_file, max_invalid_missing, start_date, end_date)
        filename = f"synthetic_calls_data_part{i}.csv"
        df.to_csv(filename, index=False)
        print(f"Created file {filename} with {num_rows_per_file} rows, max {max_invalid_missing} invalid/missing data.")

if __name__ == "__main__":
    main()


