# Data Directory

Add the CDC Diabetes Health Indicators dataset here:

```text
data/raw/cdc_diabetes_health_indicators.csv
```

Accepted alternate filenames:

```text
data/raw/pima_indians_diabetes.csv
data/raw/pima-indians-diabetes.csv
data/raw/heart.csv
data/raw/kidney_disease.csv
data/raw/liver.csv
data/raw/cdc_diabetes_health_indicators.csv
data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
data/raw/diabetes_012_health_indicators_BRFSS2015.csv
```

If the filename is different, the loader uses the first `.csv` file it finds in this folder. In the dashboard, `Saved CDC results` trains once with fixed settings and reloads the saved artifacts on later runs. `Simulate hospitals` retrains only for the selected hospital simulation settings.
