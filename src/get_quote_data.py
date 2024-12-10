import configparser as cfg
import os
import pandas as pd

from utils.config_parser import default_parser
from utils.alpaca_data import get_quote_summaries_across_dates

from alpaca.data.historical import StockHistoricalDataClient

from datetime import time, date, timedelta

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

###########################################################
# parse command line arguments
###########################################################

parser = default_parser(description="Save some marketdata to disk.")

args = vars(parser.parse_args())

output_suffix = args["output_suffix"]
config_file = args["config_file"]
ticker_file = args["ticker_file"]
overwrite = args["overwrite"]
start_date = args["start_date"]
end_date = args["end_date"] or start_date
start_time = args["start_time"]
end_time = args["end_time"]
grid_span = args["grid_span"]
look_span = args["look_span"]

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

tickers = pd.read_csv(ticker_file, header=None).iloc[:, 0].tolist()

marketdata_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

start_date = date.fromisoformat(start_date)
end_date = date.fromisoformat(end_date)
start_time = time.fromisoformat(start_time)
end_time = time.fromisoformat(end_time)
grid_span = timedelta(seconds=grid_span)
look_span = timedelta(seconds=look_span)

quote_data = get_quote_summaries_across_dates(
    marketdata_client=marketdata_client,
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    start_time=start_time,
    end_time=end_time,
    grid_span=grid_span,
    look_span=look_span,
)

# save quote data to disk as a pickle file
quote_data.to_pickle(os.path.join(root_dir, f"output/quote_data{output_suffix}.pkl"))
