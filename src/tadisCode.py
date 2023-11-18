from tardis_dev import datasets

datasets.download(
    exchange="binance",
    data_types=[
        "book_snapshot_25"
    ],
    from_date="2023-04-14",  # Earliest date as per your API key access
    to_date="2023-09-09",  # Latest date as per your API key access
    symbols=["adausdt"],
    api_key="TD.Is8O6MvHHQAxTNeb.nHBwFeFnSwZKuBO.ytzIYxI-wciClZo.ocwIPaWIuXv-1Tx.qHUZVDXLgYjEtqA.a4DC",
)