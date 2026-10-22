# Evaluation report

## Summary

| Metric | Value |
|---|---|
| documents | 10 |
| classification_accuracy | 1 |
| field_accuracy | 0.983 |
| mean_field_score | 0.993 |
| non_null_field_accuracy | 0.982 |
| hallucinated_fields | 0 |
| truth_null_fields | 1 |
| missed_fields | 0 |
| mean_confidence_correct | 0.843 |
| mean_confidence_incorrect | 0.84 |
| confidence_spread | 0.93 |
| distinct_confidence_values | 12 |
| validation_expected | 2 |
| validation_caught | 2 |
| validation_extra | 0 |

## Classification

| Document | Truth | Predicted | Confidence |
|---|---|---|---|
| 01_receipt_en.pdf | receipt | ✅ receipt | 0.953 |
| 02_receipt_vi.pdf | receipt | ✅ receipt | 0.955 |
| 03_receipt_th.pdf | receipt | ✅ receipt | 0.953 |
| 04_discharge_summary_en.pdf | discharge_summary | ✅ discharge_summary | 0.963 |
| 05_discharge_summary_vi.pdf | discharge_summary | ✅ discharge_summary | 0.963 |
| 06_discharge_summary_th.png | discharge_summary | ✅ discharge_summary | 0.823 |
| 07_lab_report_en.pdf | lab_report | ✅ lab_report | 0.92 |
| 08_lab_report_vi.jpg | lab_report | ✅ lab_report | 0.745 |
| 09_prescription_en.pdf | prescription | ✅ prescription | 0.92 |
| 10_prescription_vi.png | prescription | ✅ prescription | 0.745 |

## Fields

| Document | Field | Outcome | Score | Confidence |
|---|---|---|---|---|
| 01_receipt_en.pdf | hospital_name | correct | 1.0 | 0.98 |
| 01_receipt_en.pdf | patient_name | correct | 1.0 | 0.97 |
| 01_receipt_en.pdf | date | correct | 1.0 | 0.86 |
| 01_receipt_en.pdf | items | correct | 1.0 | 0.84 |
| 01_receipt_en.pdf | grand_total | correct | 1.0 | 0.97 |
| 01_receipt_en.pdf | payment_method | correct | 1.0 | 0.96 |
| 02_receipt_vi.pdf | hospital_name | correct | 1.0 | 0.98 |
| 02_receipt_vi.pdf | patient_name | correct | 1.0 | 0.97 |
| 02_receipt_vi.pdf | date | correct | 1.0 | 0.97 |
| 02_receipt_vi.pdf | items | correct | 1.0 | 0.84 |
| 02_receipt_vi.pdf | grand_total | correct | 1.0 | 0.98 |
| 02_receipt_vi.pdf | payment_method | correct | 1.0 | 0.97 |
| 03_receipt_th.pdf | hospital_name | correct | 1.0 | 0.98 |
| 03_receipt_th.pdf | patient_name | correct | 1.0 | 0.97 |
| 03_receipt_th.pdf | date | correct | 1.0 | 0.72 |
| 03_receipt_th.pdf | items | correct | 1.0 | 0.84 |
| 03_receipt_th.pdf | grand_total | correct | 1.0 | 0.97 |
| 03_receipt_th.pdf | payment_method | correct | 1.0 | 0.96 |
| 04_discharge_summary_en.pdf | hospital_name | correct | 1.0 | 0.98 |
| 04_discharge_summary_en.pdf | patient_name | correct | 1.0 | 0.97 |
| 04_discharge_summary_en.pdf | admission_date | correct | 1.0 | 0.97 |
| 04_discharge_summary_en.pdf | discharge_date | correct | 1.0 | 0.97 |
| 04_discharge_summary_en.pdf | diagnosis | correct | 1.0 | 0.84 |
| 04_discharge_summary_en.pdf | procedures_performed | correct | 1.0 | 0.84 |
| 04_discharge_summary_en.pdf | attending_physician | correct | 1.0 | 0.97 |
| 04_discharge_summary_en.pdf | discharge_instructions | correct | 1.0 | 0.97 |
| 05_discharge_summary_vi.pdf | hospital_name | correct | 1.0 | 0.98 |
| 05_discharge_summary_vi.pdf | patient_name | correct | 1.0 | 0.97 |
| 05_discharge_summary_vi.pdf | admission_date | correct | 1.0 | 0.97 |
| 05_discharge_summary_vi.pdf | discharge_date | correct | 1.0 | 0.97 |
| 05_discharge_summary_vi.pdf | diagnosis | correct | 1.0 | 0.84 |
| 05_discharge_summary_vi.pdf | procedures_performed | partial | 0.667 | 0.84 |
| 05_discharge_summary_vi.pdf | attending_physician | correct_null | 1.0 | 0.05 |
| 05_discharge_summary_vi.pdf | discharge_instructions | correct | 1.0 | 0.96 |
| 06_discharge_summary_th.png | hospital_name | correct | 1.0 | 0.73 |
| 06_discharge_summary_th.png | patient_name | correct | 1.0 | 0.68 |
| 06_discharge_summary_th.png | admission_date | correct | 1.0 | 0.73 |
| 06_discharge_summary_th.png | discharge_date | correct | 1.0 | 0.72 |
| 06_discharge_summary_th.png | diagnosis | correct | 1.0 | 0.49 |
| 06_discharge_summary_th.png | procedures_performed | correct | 1.0 | 0.49 |
| 06_discharge_summary_th.png | attending_physician | correct | 1.0 | 0.71 |
| 06_discharge_summary_th.png | discharge_instructions | correct | 1.0 | 0.7 |
| 07_lab_report_en.pdf | lab_name | correct | 1.0 | 0.98 |
| 07_lab_report_en.pdf | patient_name | correct | 1.0 | 0.97 |
| 07_lab_report_en.pdf | date | correct | 1.0 | 0.97 |
| 07_lab_report_en.pdf | tests | correct | 1.0 | 0.84 |
| 08_lab_report_vi.jpg | lab_name | correct | 1.0 | 0.73 |
| 08_lab_report_vi.jpg | patient_name | correct | 1.0 | 0.73 |
| 08_lab_report_vi.jpg | date | correct | 1.0 | 0.71 |
| 08_lab_report_vi.jpg | tests | correct | 1.0 | 0.49 |
| 09_prescription_en.pdf | doctor_name | correct | 1.0 | 0.98 |
| 09_prescription_en.pdf | patient_name | correct | 1.0 | 0.98 |
| 09_prescription_en.pdf | date | correct | 1.0 | 0.98 |
| 09_prescription_en.pdf | medications | correct | 1.0 | 0.84 |
| 10_prescription_vi.png | doctor_name | correct | 1.0 | 0.71 |
| 10_prescription_vi.png | patient_name | correct | 1.0 | 0.71 |
| 10_prescription_vi.png | date | correct | 1.0 | 0.71 |
| 10_prescription_vi.png | medications | correct | 0.924 | 0.49 |

## Validation

| Document | Expected errors | Raised errors | Missed | Extra |
|---|---|---|---|---|
| 01_receipt_en.pdf | - | - | - | - |
| 02_receipt_vi.pdf | total_mismatch | total_mismatch | - | - |
| 03_receipt_th.pdf | - | - | - | - |
| 04_discharge_summary_en.pdf | - | - | - | - |
| 05_discharge_summary_vi.pdf | - | - | - | - |
| 06_discharge_summary_th.png | date_order | date_order | - | - |
| 07_lab_report_en.pdf | - | - | - | - |
| 08_lab_report_vi.jpg | - | - | - | - |
| 09_prescription_en.pdf | - | - | - | - |
| 10_prescription_vi.png | - | - | - | - |

## Field problems (truth vs predicted)

- **05_discharge_summary_vi.pdf / procedures_performed** (partial, conf 0.84)
  - truth: `["Phẫu thuật nội soi cắt ruột thừa ngày 11/03/2024"]`
  - predicted: `["Phẫu thuật nội soi cắt ruột thừa ngày 11/03/2024", "kháng sinh, giảm đau, truyền dịch"]`
