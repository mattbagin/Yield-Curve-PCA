import numpy as np
import pandas as pd
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# read Bank of Canada yield curve history into dataframe
yields_df = pd.read_csv("yield_curves.csv", index_col=0)

# re-name dataframe columns
col_list = [
    (x.replace(" ZC", "")).replace("YR", "") for x in yields_df.columns.to_list()
]
col_list = [x[0 : len(x) - 2] + "." + x[len(x) - 2 :] for x in col_list]
yields_df.columns = col_list

# clean-up data by removing na, strippined whitespace, and converting to numeric
yields_df = yields_df.replace(" na", np.nan)
yields_df = yields_df.replace(" ", "")
yields_df = yields_df.apply(lambda x: x.str.strip())
yields_df = yields_df.iloc[:, 0:100]
yields_df = yields_df.dropna(how="any")
# yields_df = yields_df.drop(columns=". ")
yields_df = yields_df.apply(pd.to_numeric)
yields_df.index = pd.to_datetime(yields_df.index)
# yields_df.to_csv("cleaned_data.csv", index=True)

"""# plot the history of a specific term point
years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter("%Y")

fig, ax = plt.subplots()
ax.plot(yields_df["25.00"])
plt.title("25Y Yield History")

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(yields_df.index[0], "Y")
datemax = np.datetime64(yields_df.index[-1], "Y") + np.timedelta64(1, "Y")
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter("%Y-%m-%d")
ax.format_ydata = lambda x: f"{x*100}%"  # format the price.
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

plt.show()"""

# Principal Component Analysis on the yield curve using sci kit
scaled_data = preprocessing.scale(yields_df)

pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]
# print(per_var)

"""plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Principal Component")
plt.title("Scree Plot")
plt.show()"""


pca_df = pd.DataFrame(pca_data, index=yields_df.index, columns=labels)

"""plt.scatter(pca_df["PC1"], pca_df["PC2"])
plt.title("My PCA Graph")
plt.xlabel(f"PC1 - {per_var[0]}%")
plt.ylabel(f"PC2 - {per_var[1]}%")

for sample in pca_df.index[0:100]:
    plt.annotate(sample, (pca_df["PC1"].loc[sample], pca_df["PC2"].loc[sample]))

plt.show()"""

plt.plot(yields_df.columns, pca.components_[0], label="Level")
plt.plot(yields_df.columns, pca.components_[1], label="Slope")
plt.plot(yields_df.columns, pca.components_[2], label="Curvature")
plt.title("First Three Principal Components")
plt.legend()
plt.show()

"""loading_scores = pd.Series(pca.components_[0], index=yields_df.columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

top_10_tenors = sorted_loading_scores[0:10].index.values

print(loading_scores[top_10_tenors])"""
