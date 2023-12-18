import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from plot import plot

df = pd.read_csv("Automobile.csv")

# Drop rows that have missing values.
# We could try to fill these values in, but instead we'll
# simply remove them from the dataset.
df = df.dropna()

# Grab the categorical features.
# Also use an sklearn LabelEncoder to transform values like
# ['usa', 'europe', 'japan'] into values like [0, 1, 2].
names = df["name"]
origins = df["origin"]
label_encoder = LabelEncoder()
origins_encoded = label_encoder.fit_transform(origins)

# Grab the numeric features.
# Store these as `data` to be visualized below.
numeric_features = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
]
data = df[numeric_features].values

# Set up an sklearn pipeline that has two steps.
# 1) Normalize the data with StandardScaler
# Subtracting the mean and dividing by the variance. We do this first step to
# avoid some features with larger values being weighted higher in future calculations.
# 2) Reduce the dimensionality of the data to 2-dimensions with PCA
# PCA finds the axes with the highest variance.
# Then it selects the top-k components and projects the data onto those k axes.
# This is called a linear transformation.
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2, random_state=42),
)
data_embed = pipeline.fit_transform(data)

# Set up chart data to be displayed nicely
chart_data = pd.DataFrame(
    {
        "Principal Component 1": data_embed[:, 0],
        "Principal Component 2": data_embed[:, 1],
        "Name": names,
        "Origin": origins,
        "Horsepower": df["horsepower"],
        "Miles/Gallon": df["mpg"],
        "Displacement": df["displacement"],
        "Weight": df["weight"],
        "Acceleration": df["acceleration"],
        "Model Year": df["model_year"],
        "# Cylinders": df["cylinders"],
    }
)

# Make some pretty plots!
# Try hovering over the points in the scatter plot to explore.
# Which features do the principal components seem to capture?
plot(
    chart_data,
    x="Principal Component 1",
    y="Principal Component 2",
    colors=["Horsepower", "Origin", "Model Year"],
    tooltip=[
        "Name",
        "Origin",
        "Horsepower",
        "Miles/Gallon",
        "Displacement",
        "Weight",
        "Acceleration",
        "Model Year",
        "# Cylinders",
    ],
)
