# ARTI308-Lab4-Seeds

# Lab 4 - Data Quality Assessment & Preprocessing

## Student Information
- Name: Zakiyah Al-Thunayyan
- ID: 2240005958
- Course: ARTI308

---

## Tasks Completed

### Task 1: Data Quality Assessment
- Checked dataset shape (210 rows, 8 columns).
- Verified data types.
- Checked for missing values.
- Checked for duplicate records.
- Visualized outliers using boxplot.

---

### Task 2: Missing Value Handling
- Missing values were intentionally introduced for demonstration.
- Applied Mean Imputation to handle missing values.
- Verified that no missing values remained.

---

### Task 3: Outlier Detection & Handling (IQR)
- Detected outliers using the IQR method.
- Removed rows containing extreme values.
- Compared boxplots before and after removal.

---

### Task 4: Data Normalization
Applied two scaling techniques:

1. **Z-Score Standardization**
   - Mean ≈ 0
   - Standard Deviation ≈ 1

2. **Min-Max Normalization**
   - Scaled features between 0 and 1.

---

### Task 5: PCA (Dimensionality Reduction)
- Checked feature correlation using a heatmap.
- Applied PCA when strong correlation was detected.
- Reduced dataset to 2 principal components.
- First two components explained approximately 89% of variance.

---


## Conclusion
The dataset was cleaned, standardized, normalized, and reduced using PCA.  
The preprocessing pipeline ensures that the data is ready for machine learning models.
