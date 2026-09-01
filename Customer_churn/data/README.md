# Dataset

This project uses the **Customer Churn Dataset — business customers of a
leading Bulgarian telecom operator**.

- **Source:** https://data.mendeley.com/datasets/nrb55gr66h/1
- **DOI:** 10.17632/nrb55gr66h.1
- **Author:** Dimitar Tokmakov, Plovdivski universitet Paisij Hilendarski
- **Published:** 11 November 2024
- **Licence:** CC BY 4.0

## How to obtain

1. Open the Mendeley link above and download the archive.
2. Place `Baza_customer_Telecom_v2.csv` in this `data/` folder.
3. Update `DATA_PATH` in Cell 2 of the notebook if your filename differs.

## Note on the published metadata

The Mendeley page states 8,454 instances and 14 attributes plus 1 target.
The actual file contains **8,453 rows and 14 columns** (13 features + target).
The description also lists "CRM PID" and "Value Segment" as separate
attributes, but the file merges them into one column, and contains an
undocumented column `EffectiveSegment`. The notebook reports what is in the
file and flags the discrepancy.