#!/usr/bin/env python3
"""Test the GenAI patches for robust FY/CQ classification."""

import sys
import pandas as pd

# Set RG_TESTS to avoid Streamlit initialization
import os
os.environ["RG_TESTS"] = "1"

# Import the function we're testing
from app import extract_issuer_financial_data

print("=" * 60)
print("Testing GenAI Patches - FY/CQ Classification")
print("=" * 60)

# Test 1: Non-December FY dates should be classified as FY (not CQ)
print("\nTest 1: Non-December FY (June 30th) classification")
df_june = pd.DataFrame([{
    "Company Name": "TestCo",
    "Period Ended": "30/06/2024",
    "Period Ended.1": "30/06/2023",
    "EBITDA Margin": 10.0,
    "EBITDA Margin.1": 9.0
}])

try:
    data = extract_issuer_financial_data(df_june, "TestCo")

    # Check that the data was extracted
    assert "EBITDA Margin" in data["financial_data"], "EBITDA Margin not found in financial_data"

    # Check that June 30th dates are classified as FY (not CQ)
    june_2024_type = data["period_types"].get("2024-06-30")
    june_2023_type = data["period_types"].get("2023-06-30")

    print(f"  2024-06-30: {june_2024_type}")
    print(f"  2023-06-30: {june_2023_type}")

    assert june_2024_type == "FY", f"Expected FY for 2024-06-30, got {june_2024_type}"
    assert june_2023_type == "FY", f"Expected FY for 2023-06-30, got {june_2023_type}"

    print("  PASS: Non-December FY correctly classified")
except Exception as e:
    print(f"  FAIL: {str(e)}")
    sys.exit(1)

# Test 2: December dates should still be classified as FY
print("\nTest 2: December FY classification")
df_dec = pd.DataFrame([{
    "Company Name": "TestCo2",
    "Period Ended": "31/12/2024",
    "Period Ended.1": "31/12/2023",
    "EBITDA Margin": 15.0,
    "EBITDA Margin.1": 14.0
}])

try:
    data = extract_issuer_financial_data(df_dec, "TestCo2")

    dec_2024_type = data["period_types"].get("2024-12-31")
    dec_2023_type = data["period_types"].get("2023-12-31")

    print(f"  2024-12-31: {dec_2024_type}")
    print(f"  2023-12-31: {dec_2023_type}")

    assert dec_2024_type == "FY", f"Expected FY for 2024-12-31, got {dec_2024_type}"
    assert dec_2023_type == "FY", f"Expected FY for 2023-12-31, got {dec_2023_type}"

    print("  PASS: December FY correctly classified")
except Exception as e:
    print(f"  FAIL: {str(e)}")
    sys.exit(1)

# Test 3: 1900 sentinel dates should be ignored
print("\nTest 3: 1900 sentinel date handling")
df_sentinel = pd.DataFrame([{
    "Company Name": "TestCo3",
    "Period Ended": "31/12/2024",
    "Period Ended.1": "01/01/1900",  # Sentinel date - should be skipped
    "Period Ended.2": "31/12/2023",
    "EBITDA Margin": 12.0,
    "EBITDA Margin.1": 11.0,  # This should be skipped
    "EBITDA Margin.2": 10.0
}])

try:
    data = extract_issuer_financial_data(df_sentinel, "TestCo3")

    # Check that 1900 date was NOT included
    assert "1900-01-01" not in data["period_types"], "1900 sentinel date should not be included"

    # Check that valid dates were included
    assert "2024-12-31" in data["period_types"], "2024-12-31 should be included"
    assert "2023-12-31" in data["period_types"], "2023-12-31 should be included"

    print(f"  Valid periods found: {len(data['period_types'])}")
    print(f"  Period types: {list(data['period_types'].keys())}")
    print("  PASS: 1900 sentinel dates correctly ignored")
except Exception as e:
    print(f"  FAIL: {str(e)}")
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS: All GenAI patch tests passed!")
print("=" * 60)
