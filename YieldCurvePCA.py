import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read Bank of Canada yield curve history into dataframe
yields_df = pd.read_csv("yield_curves.csv", index_col=0)

# re-name dataframe columns
col_list = [
    (x.replace(" ZC", "")).replace("YR", "") for x in yields_df.columns.to_list()
]
col_list = [x[0 : len(x) - 2] + "." + x[len(x) - 2 :] for x in col_list]
yields_df.columns = col_list
yields_df = yields_df.replace(" na", np.nan)
yields_df = yields_df.replace(" ", "")

yields_df = yields_df.apply(lambda x: x.str.strip())
yields_df = yields_df.dropna(how="any")
yields_df = yields_df.drop(columns=". ")

# yields_df.to_csv("cleaned_data.csv", index=True)

yields_df = yields_df.apply(pd.to_numeric)

# print(yields_df.columns)
# print(yields_df.head())
# print(yields_df.dtypes)
# print(yields_df.std())

# standardize the dataframe

yields_std = (yields_df - yields_df.mean()) / yields_df.std()

# print(yields_std.head())

cov_matrix = np.array(np.cov(yields_std, rowvar=False))

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# idx = eigenvalues.argsort()[::-1]
# eigenvalues = eigenvalues[idx]
# eigenvectors = eigenvectors[:, idx]
# print(eigenvalues)
# print(eigenvectors)

eigenval_df = pd.DataFrame({"Eigenvalues": eigenvalues})
eigenvec_df = pd.DataFrame(eigenvectors)

eigenval_df["Explained Variance"] = eigenval_df["Eigenvalues"] / np.sum(
    eigenval_df["Eigenvalues"]
)
# eigenval_df.style.format({"Explained Variance": "{:.2%"})
# print(eigenval_df)
# print(eigenvec_df)

# print(yields_std.shape)
# print(eigenvec_df.shape)

principal_components = np.dot(yields_std, eigenvec_df)
pc_df = pd.DataFrame(
    data=principal_components, index=yields_std.index, columns=yields_std.columns
)
# print(pc_df.head())
# pc_df.to_csv("PCAs.csv")

plt.plot(pc_df.iloc[0])
plt.title("First Principal Component")

plt.show()
