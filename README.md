### 1. Download the Dataset
Download the full dataset from Kaggle:

**[2023 US Civil Flights: Delay, Meteo, and Aircraft](https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft/data)**

---

### 2. Organize Files
After downloading, place all CSV files into the following directory:

```
data/raw/
```

---

### 3. Run Initial EDA
Execute the notebook:

```
eda_base.ipynb
```

This will give you a first look into the core dataset (`US_flights_2023.csv`) including:
- Basic distributions
- Delay patterns
- Airline comparisons

---

### 4. Join Datasets into Master File
Run the following notebook to combine all raw sources:

```
join_tables.ipynb
```

This will merge:
- Flights data
- Aircraft metadata
- Airport geolocation
- Weather and delay information

The resulting file will be saved as:
```
data/processed/master_file.csv
```

---

### 5. Perform Full Feature EDA
Launch:

```
eda_master.ipynb
```

This performs an in-depth exploration of the full dataset, including:
- Correlation heatmaps
- Weather vs delay analysis
- Aircraft age, departure time, and cancellation patterns
- Time series of delays and disruptions

It will also output a cleaned version of the dataset:
```
data/processed/df_cleaned_ready_for_modeling.csv
```

---

### 6. Prepare for Modeling
Before modeling, review and adjust the `columns_to_drop` list inside `eda_master.ipynb` depending on your use case.

---
