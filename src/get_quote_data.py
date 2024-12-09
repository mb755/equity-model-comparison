import configparser as cfg
import os
import pandas as pd

from utils.config_parser import default_parser
from utils.alpaca_data import get_active_assets, stockday_request

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

from datetime import date, timedelta
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

###########################################################
# parse command line arguments
###########################################################

parser = default_parser(description="Save some marketdata to disk.")

args = vars(parser.parse_args())

output_suffix = args["output_suffix"]
config_file = args["config_file"]
all_assets = args["all_assets"]
ticker_file = args["ticker_file"]
overwrite = args["overwrite"]
start_date = args["start_date"]
end_date = args["end_date"] or start_date

###########################################################
# grab initial values from config file
###########################################################

config = cfg.ConfigParser()
config.read(config_file)

api_key = config.get("alpaca", "api_key")
secret_key = config.get("alpaca", "secret_key")

###########################################################
# get quote data
###########################################################

trading_client = TradingClient(api_key=api_key, secret_key=secret_key)

if all_assets:
    assets_df = get_active_assets(trading_client)
    tickers = assets_df["symbol"].tolist()
else:
    tickers = pd.read_csv(ticker_file, header=None).iloc[:, 0].tolist()

marketdata_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

start_date = date.fromisoformat(start_date)
end_date = date.fromisoformat(end_date)

date_list = [
    start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
]

dates = pd.date_range(start_date, end_date).to_pydatetime()
dates = list(map(lambda x: x.date(), dates))

for date_itr in (pbar_date := tqdm(dates)):
    date_str = date_itr.strftime("%Y-%m-%d")
    pbar_date.set_description(f"Processing {date_str}")

    output_filename = f"{root_dir}/output/quote_data_{date_str}{output_suffix}.h5"

    # if output already exists, print a warning
    initial_mode = "a"
    if os.path.exists(output_filename):
        if overwrite:
            tqdm.write(f"Output file {output_filename} already exists. Overwriting.")
            initial_mode = "w"
        else:
            tqdm.write(f"Output file {output_filename} already exists. Exiting.")
            continue

    mode = initial_mode

    for ticker in (pbar_ticker := tqdm(tickers, leave=False)):
        pbar_ticker.set_description(f"Processing {ticker}")

        request_params = stockday_request(ticker, date_itr)
        result = marketdata_client.get_stock_quotes(request_params)

        result_df = result.df

        if result_df.empty:
            tqdm.write(f"No data for {ticker} on {date_str}. Skipping.")
            continue

        # flatten conditions, so the result can be serialized
        result_df["conditions"] = result_df["conditions"].apply(lambda x: ",".join(x))

        result_df.to_hdf(
            output_filename,
            key=ticker.replace(
                ".", "_"
            ),  # for easier access to preferred shares that look like LXP.PRC
            mode=mode,
            format="table",
            complevel=9,
            complib="blosc",
        )

        # everything after the first table has to be appended
        mode = "a"
