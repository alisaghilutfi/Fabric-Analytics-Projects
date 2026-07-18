# Data Profile — ws_Finance_Analysis

Generated during project setup. Reference for Silver notebook logic and CLAUDE.md rules.

---

## customers.csv

| Property | Value |
|---|---|
| Rows | 5,000 |
| Columns | 11 |
| Nulls | 0 in all columns |
| Duplicate customer_id | 0 |

### Columns

| Column | Type | Notes |
|---|---|---|
| customer_id | string | Format C00001–C05000. PK. |
| fisrt_name | string | **Typo in source** — corrected to `first_name` in Silver |
| second_name | string | Clean |
| gender | string | Male (2,456), Female (2,544) |
| date_of_birth | string | Format: dd-MM-yyyy |
| city | string | Clean |
| state | string | 13 Indian states |
| occupation | string | 6 values: Retired, Student, Salaried, Self Employed, Freelancer, Business Owner |
| customer_segment | string | 5 values: Retail (54%), Premium (18%), SME (16%), Corporate (7%), Wealth (5%) |
| annual_income | float | Range: ₹180,000 – ₹5,026,136. Mean: ₹724,095 |
| join_date | string | Format: dd-MM-yyyy |

---

## finance_transactions.csv

| Property | Value |
|---|---|
| Rows | 50,069 |
| Columns | 15 |
| Date range | 2023-01-01 to 2026-04-30 |
| Years covered | 2023, 2024, 2025, 2026 (partial) |

### Columns

| Column | Type | Notes |
|---|---|---|
| transaction_id | string | Format T00000001–T00050069 |
| transaction_date | string | Format: dd-MM-yyyy |
| account_id | string | 7,987 distinct accounts (~2 per customer) |
| customer_id | string | 3,983 distinct customers (1,017 customers have no transactions) |
| transaction_type | string | 10 values — see below |
| channel | string | **DIRTY** — see issues |
| merchant_category | string | 14 values |
| amount | float | Range: -₹21,449 to ₹312,586. Mean: ₹9,110 |
| fee_amount | float | **24 nulls** — fill with 0.0 |
| tax_amount | float | Clean |
| currency | string | **DIRTY** — see issues |
| transaction_status | string | Success (85.7%), Failed (10.2%), Pending (4.1%) |
| is_fraud | string | Yes (632, 1.26%), No (49,437) |
| risk_score | int | Range: 1–100. Mean: 36.1 |
| reference_no | string | Unique reference. Not used in model. |

### Transaction type values
Loan EMI (9,140), Transfer (8,479), Card Payment (5,329), Deposit (4,970),
Bill Payment (4,310), Withdrawal (4,279), Interest Credit (4,017),
Fee Charge (3,640), Investment (3,568), Refund (2,337)

### Merchant category values (14)
Bank Charges, Education, Entertainment, Food Delivery, Fuel, Groceries,
Healthcare, Insurance, Mutual Fund, Rent, Salary, Shopping, Travel, Utilities

---

## Data Quality Issues & Remediations

| # | Issue | Severity | Remediation in Silver |
|---|---|---|---|
| 1 | **Duplicate transaction_ids**: 69 exact full-row duplicates (138 rows total share IDs) | High | `dropDuplicates()` before write |
| 2 | **Dirty `channel`**: `M@bile App` (765 rows), plus whitespace variants (`  Net Banking`, ` Mobile App `, etc.) | Medium | `trim()` + `when("M@bile App" → "Mobile App")` |
| 3 | **Dirty `currency`**: `inr` (604 rows), `inR` (1 row) alongside correct `INR` | Medium | `upper(trim(col("currency")))` |
| 4 | **`fee_amount` nulls**: 24 rows | Low | `coalesce(fee_amount, 0.0)` |
| 5 | **Negative `amount` values**: 9 rows across 5 transaction types | Low | Flag as `is_reversal = True`, keep in fact |
| 6 | **Column name typo `fisrt_name`** in customers | Low | `withColumnRenamed("fisrt_name", "first_name")` |
| 7 | **`is_fraud` as string** ("Yes"/"No") | Low | Cast to boolean `is_fraud_bool` |
| 8 | **1,017 customers with no transactions** | Info | Valid — keep in dim_customer |

---

## Referential Integrity

- All `customer_id` values in transactions exist in customers ✓
- 1,017 customers have no transactions — correct (not orphans in wrong direction)
- No orphaned account_ids (account is not a separate dimension — it's an attribute on transactions)

---

## Star Schema Design

```
fact_transactions (50,000 rows after dedup)
    ├── dim_customer  (5,000 rows)  via customer_id
    ├── dim_date      (1,461 rows)  via transaction_date → Date
    ├── dim_channel   (7 rows)      via channel_key
    └── dim_merchant  (14 rows)     via merchant_key
```

Account is intentionally NOT a separate dimension — it has no attributes beyond the ID
and would add unnecessary complexity. account_id is kept as a column in fact_transactions
for filtering/counting if needed.
