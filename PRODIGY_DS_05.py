import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Sample Traffic Accident Data Simulation
# (Aap iski jagah US Accidents CSV load kar sakte hain: df = pd.read_csv('US_Accidents.csv'))
data = {
    "Start_Time": pd.date_range(
        start="2023-01-01", periods=500, freq="h"
    ),
    "Weather_Condition": np.random.choice(
        ["Clear", "Rain", "Snow", "Fog", "Overcast"], size=500
    ),
    "Road_Condition": np.random.choice(
        ["Dry", "Wet", "Icy", "Slush"], size=500
    ),
    "Severity": np.random.choice([1, 2, 3, 4], size=500, p=[0.1, 0.5, 0.3, 0.1]),
    "Latitude": np.random.uniform(37.0, 38.0, size=500),
    "Longitude": np.random.uniform(-122.5, -121.5, size=500),
}

df = pd.DataFrame(data)

# Time Features Extract Karein
df["Hour"] = df["Start_Time"].dt.hour

# 2. Visualizations
plt.figure(figsize=(14, 10))

# Plot 1: Accidents by Time of Day (Hour)
plt.subplot(2, 2, 1)
sns.countplot(data=df, x="Hour", palette="viridis")
plt.title("Accidents Frequency by Hour of Day")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Accident Count")

# Plot 2: Accidents by Weather Condition
plt.subplot(2, 2, 2)
sns.countplot(data=df, x="Weather_Condition", palette="magma")
plt.title("Accidents by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Accident Count")

# Plot 3: Severity by Road Condition
plt.subplot(2, 2, 3)
sns.countplot(data=df, x="Road_Condition", hue="Severity", palette="Set2")
plt.title("Accident Severity by Road Condition")
plt.xlabel("Road Condition")
plt.ylabel("Count")

# Plot 4: Accident Hotspots (Latitude vs Longitude)
plt.subplot(2, 2, 4)
sns.scatterplot(
    data=df,
    x="Longitude",
    y="Latitude",
    hue="Severity",
    palette="coolwarm",
    alpha=0.7,
)
plt.title("Accident Hotspots Map (Coordinates)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.show()
