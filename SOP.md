# Standard Operating Procedure (SOP)
## UZIO vs ADP – Payment & Emergency Contact Comparison Tool

**Tool Link:** [Access Tool Here](https://uzio-adp-payment-and-emergency-data-review.streamlit.app/)

### 1. Purpose
This tool automates the comparison of employee payment and emergency contact data between UZIO and ADP systems. It identifies discrepancies to ensure data consistency across both platforms.

### 2. Prerequisites
You must use the provided **Payment and Emergency Tool Input Template.xlsx** file. Do not create a new Excel file from scratch.

### 3. Preparing the Input Template
Open the `Payment and Emergency Tool Input Template.xlsx` file and populate the sheets as follows:

1.  **Uzio Data**
    *   Copy employee data from the UZIO export.
    *   **Required Columns**: Ensure `Employee ID` is populated. Other key columns include `Paycheck Distribution`, `Paycheck Percentage`, `Paycheck Amount`, `Routing Number`, `Account Number`, and Emergency Contact details (`Name`, `Relationship`, `Phone`).

2.  **ADP Payment Data**
    *   Copy payment data from the ADP export.
    *   **Required Columns**: `ASSOCIATE ID`, `DEPOSIT TYPE`, `DEPOSIT AMOUNT`, `DEPOSIT PERCENT`, `ROUTING NUMBER`, `ACCOUNT NUMBER`, `PRIORITY #`.

3.  **ADP Emergency Contact Data**
    *   Copy emergency contact data from the ADP export.
    *   **Required Columns**: `Associate ID`, `Contact Name`, `Mobile Phone`, `Relationship Description`.

4.  **Payment_Mapping & Emergency_Mapping**
    *   These sheets map the columns between UZIO and ADP.
    *   **Action**: Verify that the column names listed under `UZIO Column` match the headers in your "Uzio Data" sheet, and `ADP Column` match the headers in your ADP sheets.
    *   *Example*: If UZIO has "Mobile Phone" and ADP has "Cell Number", map them in the Emergency_Mapping sheet.

### 4. Step-by-Step Guide

#### Step 1: Access the Tool
Open the [Tool Link](https://uzio-adp-payment-and-emergency-data-review.streamlit.app/) in your web browser.

#### Step 2: Upload Data
1.  Locate the section labeled **"Upload Excel workbook"**.
2.  Click **"Browse files"** and select your prepared **Payment and Emergency Tool Input Template.xlsx**.
3.  Wait for the file to finish uploading.

#### Step 3: Run the Audit
1.  Once the file is uploaded, the red **"Run Audit"** button will become active.
2.  Click **"Run Audit"**.
3.  Wait a moment for the system to process the data. You will see a "Running audit..." indicator.

#### Step 4: Download Reports
After processing is complete, two download buttons will appear:
1.  **Download Payment Report (.xlsx)**: Contains comparison results for payment details.
2.  **Download Emergency Contact Report (.xlsx)**: Contains comparison results for emergency contacts.

Click both buttons to save the reports to your computer.

### 5. Understanding the Reports
The downloaded reports contain detailed comparisons. Look for the **"Status"** column to identify issues:

| Status | Meaning | Action Required |
| :--- | :--- | :--- |
| **Data Match** | Information matches in both systems. | No action needed. |
| **Data Mismatch** | The values in UZIO and ADP are different. | Investigate and correct the data in the source system. |
| **Value missing in Uzio (ADP has value)** | ADP has data, but UZIO is blank. | Update UZIO with the missing information. |
| **Value missing in ADP (Uzio has value)** | UZIO has data, but ADP is blank. | Update ADP with the missing information. |
| **Employee ID Not Found in ADP** | The employee exists in UZIO but not in the ADP file. | Verify if the employee should be in ADP. |
| **Employee ID Not Found in Uzio** | The employee exists in ADP but not in the UZIO file. | Verify if the employee should be in UZIO. |

### 6. Troubleshooting
*   **File Upload Error**: Ensure your file is in `.xlsx` format and not `.csv`.
*   **"Key Error" or "Column Not Found"**: Check that your input Excel file has the exact sheet names required ("Uzio Data", "ADP Payment Data", "ADP Emergency Contact Data") and that the headers in the Mapping sheets match your data sheets exactly.
