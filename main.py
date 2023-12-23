import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plot import plot

# Load in automobile data from .csv file
df = pd.read_csv("Automobile.csv")

# Drop rows that have missing values.
# We could try to fill these values in, but instead we'll
# simply remove them from the dataset.
df = df.dropna()

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

# Set up an sklearn pipeline that has two steps:
# 1) Use StandardScaler to normalize the data (subtract the mean and divide by the variance)
# 2) Use PCA to reduce the dimensionality of the data to 2-dimensions
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2, random_state=42),
)
data_embed = pipeline.fit_transform(data)

# Collect all of our data into a DataFrame to be displayed nicely
chart_data = pd.DataFrame(
    {
        "Principal Component 1": data_embed[:, 0],
        "Principal Component 2": data_embed[:, 1],
        "Name": df["name"],
        "Origin": df["origin"],
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
# NOTE: Update these 3 lines to change which features are used for colors:
color_feature_1 = "Horsepower"
color_feature_2 = "Origin"
color_feature_3 = "Model Year"
plot(
    chart_data,
    x="Principal Component 1",
    y="Principal Component 2",
    colors=[color_feature_1, color_feature_2, color_feature_3],
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
