# Assignment

Understanding high dimensional data is difficult. Visualization tools can help uncover trends, relationships, and outliers, but line charts and scatter plots can only compare one or two features of your data at a time.

A statistical method called "dimensionality reduction" can help us take complex data and simplify it enough to be visualized. One form of dimensionality reduction is called PCA (principal component analysis). We'll use PCA in this assignment to find trends in a dataset of automobiles.

PCA works by finding axes in the data that have the highest variation or variance. The axis with the highest variance is called the "first principal component", the axis with the second highest variance is the "second principal component", etc. PCA sorts the axes by variance and then selects a small subset of those axes. To visualize high dimensional data in just two dimensions we can use the first two principal components.

Clone or download the source code from this repo. Follow the instructions in `README.md` to install the Python requirements and run `main.py`. Altair visualizations should open in your browser. Read the comments in `main.py` to understand what these plots are showing. Hover over points in the plots to view information about each vehicle. Update `color_feature_1` in `main.py` to explore all features in the dataset. Use what you've learned to answer the questions below.

## Questions

1. Open cloud.png. Which arrow is the first principal component? Why?

2. What is the dimensionality of the data before we use PCA? What is the dimensionality after?

3. What attributes of automobiles does Principal Component 1 capture?

4. What attributes of automobiles does Principal Component 2 capture?

5. If similar plots are generated when using two different features for the color scale, what does that tell you about the relationship between those two features?

## Extra Credit

1. Normalization: Remove the StandardScaler from the pipeline. What effect did this have on the results? Why?

2. Linearity: Update the code to use sklearn.manifold.TSNE instead of sklearn.decomposition.PCA. How have the plots changed? What new perspectives do you gain from using t-SNE over PCA? Why?

3. Compression: Draw a line chart by hand. The x-axis should be "Number of principal components". The y-axis should be "information loss". Draw a line that shows how information might be lost from the original dataset as you reduce dimensionality and project data into lower dimensions with PCA. Explain your thinking.
