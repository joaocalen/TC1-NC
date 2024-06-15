import os
import requests
import pandas as pd


def load_data_energy():
    csv_filename = "Renewable_Energy_Consumption_in_the_US.csv"
    csv_url = "https://storage.googleapis.com/kagglesdsdata/datasets/4962496/8352264/dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240531T131003Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=67056505172956b7c17a6afbde9d66ab941769f53d09951ab578d832886b88c9e9d8cc2eda441b3eb97f0ef863c474c6b742b040e39d242f9410580de0e007f12598eb6fda4ca1d3b01347657507877a0e62d658b42ee2b13147f9fe643d445120d56cc1f8741d9d9e750add7d06a351b811c602f282b7ae632c4cf372138addbad3bb50a6b6a6f9d0596bfd41779c4b901c8265669d00c23e494f669dee5f0c81ebd9ba80360147b480ebe164b3de9679b1ff62490833de99112e01354c3286b3f0118d89625e75092c6711ff6119571a3d0c3f683d3c1cf13a4a648242e435f7a26756e2e0ece1dc62dd47adee56b370a6c78f21dca7b1ab7418e2a4e6cf0d"

    # Download the dataset if not available locally
    if not os.path.exists(csv_filename):
        print("Downloading the dataset...")
        response = requests.get(csv_url)
        with open(csv_filename, "wb") as f:
            f.write(response.content)

    # Load the dataset
    data = pd.read_csv(csv_filename)

    # Preprocess the data
    data = data.dropna(subset=["Total Renewable Energy"])

    # Convert date to datetime
    data["dt"] = pd.to_datetime(data[["Year", "Month"]].assign(DAY=1))

    ts = data[['dt', 'Total Renewable Energy']].dropna().copy()
    ts = ts.groupby('dt', as_index=False).sum()
    return ts

def load_data_climate_change():
    csv_filename = "GlobalTemperatures.csv"
    # Check if the CSV file exists
    if not os.path.exists(csv_filename):
        print("Downloading the dataset...")
        url = "https://storage.googleapis.com/kagglesdsdata/datasets/29/2150/GlobalTemperatures.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240528%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240528T001256Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3acad3f962d03803957eaa3e067f56bcb7e659807d1d15cfd995a41d2bc75180f134d37fbacce461ea12e0ff136115c9ced67ed9413f99fa1fc2ef5402d511a6b35345bca2b0b3ff26e1922e2ed137d895e42a3c3c8f1322f1927df1d11f07287b8fa83e6a85182c4f553e85d34ebc7a2a912daf2e19849216cd05493a1dbc8915306fe5aec86197d11f7782559498b0bdd63b53c9c349954c91393c99001329ab52cbde12665747abada39257beba000812d77f90f488c2d0095630d78ac7c3795522e606144a9a9f7ff07672911f16105a546e882b2aa5df6cdeab0fd38aea791753c8504e10659297823cfc2f63a4c6d3843638759212682326fef0ef413b"
        response = requests.get(url)
        with open(csv_filename, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("CSV file found locally.")

    # Load the dataset
    print("Loading CSV file...")
    data = pd.read_csv(csv_filename)
    print("Dataset loaded")

    # Preprocess the data
    print("Preprocessing data...")
    data = data.dropna(
        subset=["LandAverageTemperature"]
    )

    # Convert date to datetime
    data["dt"] = pd.to_datetime(data["dt"])

    ts = data[["dt", "LandAverageTemperature"]].dropna().copy()
    ts = ts.groupby("dt", as_index=False).mean()
    return ts


def create_lag_features(data, lags=1, targets=["Total Renewable Energy"]):
    df = pd.DataFrame(data)
    columns = [df[targets].shift(i) for i in range(lags, 0, -1)]
    columns = [df] + columns
    df = pd.concat(columns, axis=1)
    df.dropna(inplace=True)
    # df.columns = ['dt', target] + [f'{target}_lag_{i}' for i in range(1, lags + 1)]
    df.columns = (
        ["dt"]
        + targets
        + [f"{target}_lag_{i}" for target in targets for i in range(1, lags + 1)]
    )
    return df
