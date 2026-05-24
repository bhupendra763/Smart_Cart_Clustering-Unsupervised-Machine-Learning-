# 🛒 SmartCart Customer Segmentation using Clustering

> Unsupervised machine learning pipeline to segment retail customers based on demographics, spending behavior, and purchase patterns — enabling targeted marketing strategies.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

---

## Overview

SmartCart is a customer segmentation project that applies unsupervised clustering techniques to a retail marketing dataset. The goal is to group customers into meaningful segments based on their demographic profile and purchasing behavior, providing actionable insights for personalized marketing.

The project walks through a complete ML pipeline: data cleaning, feature engineering, outlier removal, encoding, scaling, dimensionality reduction with PCA, and clustering via **K-Means** and **Agglomerative Clustering**.

---

## Dataset

**File:** `SmartCart.csv`  
**Records:** 2,240 customers | **Features:** 22 columns

| Feature | Description |
|---|---|
| `ID` | Unique customer identifier |
| `Year_Birth` | Customer's birth year |
| `Education` | Education level (Basic, 2n Cycle, Graduation, Master, PhD) |
| `Marital_Status` | Marital status |
| `Income` | Annual household income |
| `Kidhome` / `Teenhome` | Number of kids / teens at home |
| `Dt_Customer` | Date of enrollment |
| `Recency` | Days since last purchase |
| `MntWines` / `MntFruits` / ... | Spend on wines, fruits, meat, fish, sweets, gold |
| `NumWebPurchases` / `NumStorePurchases` / ... | Purchase channel counts |
| `Response` | Whether customer accepted a campaign offer |
| `Complain` | Whether customer filed a complaint |

**Missing Data:** 24 null values in `Income` → imputed with median.

---

## Project Pipeline

```
Raw Data
   │
   ▼
Missing Value Imputation (Income → Median)
   │
   ▼
Feature Engineering
  ├── Year_Birth → Age
  ├── Dt_Customer → Customer_Tenure_Days
  ├── MntXxx columns → Total_Spending
  ├── Kidhome + Teenhome → Total_Child
  ├── Education → 3-level consolidation
  └── Marital_Status → Living_Status (Partner / Alone)
   │
   ▼
Outlier Removal (Age < 90, Income < 600,000)
   │
   ▼
Correlation Heatmap Analysis
   │
   ▼
One-Hot Encoding (Education, Living_Status)
   │
   ▼
Standard Scaling
   │
   ▼
PCA (3 Components)
   │
   ▼
Optimal K Selection (Elbow + Silhouette Score)
   │
   ▼
Clustering
  ├── K-Means (k=4)
  └── Agglomerative Clustering (k=4, linkage=ward)
   │
   ▼
Cluster Characterization & Visualization
```

---

## Feature Engineering

| New Feature | Source | Description |
|---|---|---|
| `Age` | `Year_Birth` | `2026 - Year_Birth` |
| `Customer_Tenure_Days` | `Dt_Customer` | Days since joining, relative to latest date |
| `Total_Spending` | `MntWines` + `MntFruits` + `MntMeatProducts` + `MntFishProducts` + `MntSweetProducts` + `MntGoldProds` | Total spend across all categories |
| `Total_Child` | `Kidhome` + `Teenhome` | Total children in household |
| `Education` | Original `Education` | Consolidated to: `UnderGraduate`, `Graduation`, `PostGraduation` |
| `Living_Status` | `Marital_Status` | Simplified to: `Partner` or `Alone` |

---

## Modeling

### Optimal K Selection

Two methods were used together to pick the best number of clusters:

- **Elbow Method** — plots WCSS vs K; the "knee" point is detected automatically using `KneeLocator`
- **Silhouette Score** — measures cluster cohesion and separation; higher is better

Both metrics were combined in a dual-axis plot for visual comparison. **K = 4** was selected as optimal.

### Algorithms

| Algorithm | Parameters | Notes |
|---|---|---|
| K-Means | `n_clusters=4`, `random_state=42` | Fast, centroid-based |
| Agglomerative Clustering | `n_clusters=4`, `linkage='ward'` | Hierarchical, no centroid assumption |

Clustering was performed on the first **3 PCA components** to reduce noise and enable 3D visualization.

---

## Results

After clustering, each segment was profiled using group means across key features:

- **Income vs. Total Spending** scatter plots revealed distinct spending tiers
- **Cluster size distribution** was analyzed via count plots
- **Cluster summary table** (`.groupby("cluster").mean()`) provided the full demographic and behavioral profile per segment

The 4 customer segments capture meaningful differences in purchasing power, household composition, and engagement level — directly actionable for targeted campaigns.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, PCA, KMeans, Agglomerative Clustering, Silhouette Score |
| `kneed` | Automated elbow/knee detection |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bhupendra763/smartcart-clustering.git
cd smartcart-clustering
```

### 2. Install dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn kneed
```

### 3. Add the dataset

Place `SmartCart.csv` in the project root directory.

### 4. Run the notebook

```bash
jupyter notebook Smart_Cart_Clustering.ipynb
```

---

## Project Structure

```
smartcart-clustering/
│
├── SmartCart.csv                   # Raw dataset
├── Smart_Cart_Clustering.ipynb     # Main analysis notebook
└── README.md                       # Project documentation
```

---

## 📊 Key Takeaways

- Feature engineering significantly improved cluster separability — especially `Total_Spending`, `Age`, and `Living_Status`
- PCA to 3 components retained sufficient variance for meaningful 3D cluster visualization
- Agglomerative Clustering with Ward linkage produced tight, well-separated clusters comparable to K-Means
- The 4-cluster solution balances interpretability with granularity for practical marketing use

---

*Built with 🐍 Python · scikit-learn · pandas*
