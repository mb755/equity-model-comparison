import pandas as pd
import numpy as np

from alpaca.data.requests import StockQuotesRequest

from datetime import datetime, time
from pytz import timezone
from tqdm import tqdm


def stockday_request(ticker, date):
    """!@brief Create a request for stock quotes for a single day
    @param ticker (str): Ticker symbol to get data for
    @param date (datetime.date): Date to get data for

    @return StockQuotesRequest: Request object for stock quotes
    """

    nyc = timezone("US/Eastern")

    request_params = StockQuotesRequest(
        symbol_or_symbols=[ticker],
        # these times are in UTC, this loads in a full day of ticks
        start=nyc.localize(datetime.combine(date, time(3, 50))),
        end=nyc.localize(datetime.combine(date, time(21, 10))),
    )

    return request_params


def stockstime_request(tickers, date, time, look_span):
    """!@brief Create a request for stock quotes around a particular time
    @param ticker (str): Ticker symbol to get data for
    @param date (datetime.date): Date to get data for
    @param time (datetime.time): Time to get data for
    @param duration (datetime.timedelta): Duration to get data for

    @return StockQuotesRequest: Request object for stock quotes
    """

    nyc = timezone("US/Eastern")

    request_params = StockQuotesRequest(
        symbol_or_symbols=tickers,
        # these times are in UTC, this loads in a full day of ticks
        start=nyc.localize(datetime.combine(date, time) - look_span),
        end=nyc.localize(datetime.combine(date, time) + look_span),
    )

    return request_params


def select_before_after_rows(df, reference_time):
    """
    Efficiently select the latest row before and first row after reference time for each ticker.

    Parameters:
    -----------
    df : pd.DataFrame
        Multi-index DataFrame with levels [ticker, timestamp]
    reference_time : datetime
        The reference time to compare against

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by ticker with rows before and after reference time
    """
    # Create boolean masks for rows before and after reference time
    before_mask = df.index.get_level_values(1) < reference_time
    after_mask = df.index.get_level_values(1) >= reference_time

    # Filter the DataFrame
    before_df = df[before_mask]
    after_df = df[after_mask]

    # Get the last row before reference time for each ticker
    latest_before_rows = before_df.groupby(level=0).last()

    # Get the first row after reference time for each ticker
    first_after_rows = after_df.groupby(level=0).first()

    # Rename columns to indicate before/after
    latest_before_rows.columns = [f"{col}_before" for col in latest_before_rows.columns]
    first_after_rows.columns = [f"{col}_after" for col in first_after_rows.columns]

    # Combine the two DataFrames
    combined_rows = pd.concat([latest_before_rows, first_after_rows], axis=1)

    # Handle tickers missing from either before or after
    all_tickers = set(df.index.get_level_values(0))
    before_tickers = set(latest_before_rows.index)
    after_tickers = set(first_after_rows.index)

    # Add NaN rows for tickers missing from before or after
    missing_before_tickers = all_tickers - before_tickers
    missing_after_tickers = all_tickers - after_tickers

    if missing_before_tickers:
        nan_before_rows = pd.DataFrame(
            np.nan,
            index=pd.Index(missing_before_tickers, name=df.index.names[0]),
            columns=[f"{col}_before" for col in df.columns],
        )
        combined_rows = pd.concat([combined_rows, nan_before_rows])

    if missing_after_tickers:
        nan_after_rows = pd.DataFrame(
            np.nan,
            index=pd.Index(missing_after_tickers, name=df.index.names[0]),
            columns=[f"{col}_after" for col in df.columns],
        )
        combined_rows = pd.concat([combined_rows, nan_after_rows])

    return combined_rows


# CR TODO: consider handling different timezones
def get_stockstime_data(marketdata_client, tickers, date, time, look_span):
    """!@brief Get stock quotes around a particular time for a list of tickers
    @param trading_client (TradingClient): Alpaca TradingClient object
    @param tickers (List[str]): List of ticker symbols to get data for
    @param date (datetime.date): Date to get data for
    @param time (datetime.time): Time to get data for
    @param duration (datetime.timedelta): Duration to get data for

    @return pd.DataFrame: DataFrame containing stock quotes for each ticker
    """
    request_params = stockstime_request(tickers, date, time, look_span)
    quotes = marketdata_client.get_stock_quotes(request_params)

    quotes_df = quotes.df

    # CR TODO: add max_width argument, any market wider than that should be discarded
    quotes_df["mid_price"] = (quotes_df["ask_price"] + quotes_df["bid_price"]) / 2
    quotes_df["spread"] = (quotes_df["ask_price"] - quotes_df["bid_price"]) / quotes_df[
        "mid_price"
    ]

    nyc = timezone("US/Eastern")
    reference_time = nyc.localize(datetime.combine(date, time))

    quote_summary = select_before_after_rows(
        quotes_df[["mid_price", "spread"]], reference_time
    )

    return quote_summary


def get_quote_summaries(
    marketdata_client, tickers, date, start_time, end_time, grid_span, look_span
):
    """
    Collects quote summaries at regular intervals between start_time and end_time.

    Parameters:
    - marketdata_client: Client object to fetch market data.
    - tickers: List of stock symbols to fetch data for.
    - date: Date for which the data is fetched.
    - start_time: Start time (datetime.time object) for the grid.
    - end_time: End time (datetime.time object) for the grid.
    - grid_span: Time delta representing the spacing between grid points.
    - look_span: Time delta representing the duration of data to fetch around each grid point.

    Returns:
    - DataFrame: A combined DataFrame indexed by symbol and time containing quote summaries.
    """
    # Generate the grid of time points
    time_points = []
    current_time = datetime.combine(date, start_time)
    end_datetime = datetime.combine(date, end_time)

    while current_time <= end_datetime:
        time_points.append(current_time.time())
        current_time += grid_span

    # Collect data for each time point
    all_data = []
    for t in time_points:
        df = get_stockstime_data(
            marketdata_client=marketdata_client,
            tickers=tickers,
            date=date,
            time=t,
            look_span=look_span,
        )
        df["time"] = t  # Add time column for tracking
        all_data.append(df)

    # Combine data into a single DataFrame
    combined_data = pd.concat(all_data, keys=time_points, names=["grid_time", "symbol"])
    combined_data.reset_index(level=0, inplace=True)

    return combined_data


def get_quote_summaries_across_dates(
    marketdata_client,
    tickers,
    start_date,
    end_date,
    start_time,
    end_time,
    grid_span,
    look_span,
):
    """
    Collects quote summaries at regular intervals across a range of dates, handling missing data.

    Parameters:
    - marketdata_client: Client object to fetch market data.
    - tickers: List of stock symbols to fetch data for.
    - start_date: Start date (datetime.date object) for the range.
    - end_date: End date (datetime.date object) for the range.
    - start_time: Start time (datetime.time object) for the grid each day.
    - end_time: End time (datetime.time object) for the grid each day.
    - grid_span: Time delta representing the spacing between grid points.
    - look_span: Time delta representing the duration of data to fetch around each grid point.

    Returns:
    - DataFrame: A combined DataFrame indexed by date, symbol, and time containing quote summaries.
    """
    all_data = []

    # Loop through the date range
    dates = pd.date_range(start_date, end_date).to_pydatetime()
    dates = list(map(lambda x: x.date(), dates))

    for date_itr in (pbar_date := tqdm(dates)):
        date_str = date_itr.strftime("%Y-%m-%d")
        pbar_date.set_description(f"Processing {date_str}")
        try:
            # Get quote summaries for the current date
            daily_data = get_quote_summaries(
                marketdata_client=marketdata_client,
                tickers=tickers,
                date=date_itr,
                start_time=start_time,
                end_time=end_time,
                grid_span=grid_span,
                look_span=look_span,
            )

            # Add a date column for tracking
            daily_data["date"] = date_itr
            all_data.append(daily_data)

        except KeyError as e:
            # Handle missing data for closed markets or other issues
            tqdm.write(f"No data for {date_itr}: {e}")

    # Combine all the data into a single DataFrame
    if all_data:
        combined_data = pd.concat(
            all_data, keys=range(len(all_data)), names=["batch_index"]
        )
        combined_data.reset_index(level=0, drop=True, inplace=True)
        return combined_data
    else:
        # Return an empty DataFrame if no data was collected
        print("No data available for the specified date range.")
        return pd.DataFrame()


def compute_returns(
    quote_df: pd.DataFrame, price_window: str = "before_after"
) -> pd.DataFrame:
    """
    Computes returns between consecutive timepoints for each symbol.

    Parameters:
    - quote_df (pd.DataFrame): DataFrame containing quote data with 'symbol' as the index and columns:
        ['grid_time', 'mid_price_before', 'spread_before',
         'mid_price_after', 'spread_after', 'time', 'date']
    - price_window (str): Determines which prices to use for return calculation.
        - 'before_after': Use T0 mid_price_before to T1 mid_price_after
        - 'after_before': Use T0 mid_price_after to T1 mid_price_before

    Returns:
    - pd.DataFrame: DataFrame with columns ['symbol', 'return', 'start_time', 'end_time']
      where 'start_time' and 'end_time' are timezone-aware timestamps in US/Eastern.
    """

    # Validate price_window parameter
    if price_window not in ["before_after", "after_before"]:
        raise ValueError("price_window must be either 'before_after' or 'after_before'")

    # Reset index to have 'symbol' as a column
    quote_df = quote_df.reset_index()

    # Combine 'date' and 'time' into a single 'timestamp' column without converting to string
    quote_df["timestamp"] = pd.to_datetime(quote_df["date"]) + pd.to_timedelta(
        quote_df["time"].astype(str)
    )

    # Localize to US/Eastern timezone
    eastern = timezone("US/Eastern")
    quote_df["timestamp"] = quote_df["timestamp"].dt.tz_localize(eastern)

    # Sort the DataFrame by 'symbol' and 'timestamp'
    quote_df = quote_df.sort_values(by=["symbol", "timestamp"]).reset_index(drop=True)

    # Define price_t0 and price_t1 based on price_window
    if price_window == "before_after":
        quote_df["price_t0"] = quote_df["mid_price_before"]
        quote_df["price_t1"] = quote_df.groupby("symbol")["mid_price_after"].shift(-1)
    elif price_window == "after_before":
        quote_df["price_t0"] = quote_df["mid_price_after"]
        quote_df["price_t1"] = quote_df.groupby("symbol")["mid_price_before"].shift(-1)
    else:
        raise ValueError("price_window must be either 'before_after' or 'after_before")

    # Calculate returns
    quote_df["return"] = (quote_df["price_t1"] / quote_df["price_t0"]) - 1

    # Define start_time and end_time
    quote_df["start_time"] = quote_df["timestamp"]
    quote_df["end_time"] = quote_df.groupby("symbol")["timestamp"].shift(-1)

    # Select required columns and drop rows with NaN in 'end_time'
    returns_df = quote_df[["symbol", "return", "start_time", "end_time"]].dropna()

    return returns_df


def merge_predictors_and_responders(predictors_df, responders_df):
    """
    Merges two dataframes: one containing predictors and one containing responder symbols.
    Both dataframes are expected to have ['symbol', 'return', 'start_time', 'end_time'].

    This function pivots each dataframe so each symbol becomes its own column. The returned
    dataframe includes all responder returns and all predictor returns in a single, flat structure.

    Parameters
    ----------
    predictors_df : pd.DataFrame
        DataFrame with predictor symbols. Columns: ['symbol', 'return', 'start_time', 'end_time'].
    responders_df : pd.DataFrame
        DataFrame with responder (target) symbols. Same columns as above.

    Returns
    -------
    pd.DataFrame
        A single DataFrame with one row per (start_time, end_time).
        Columns will be:
          - start_time
          - end_time
          - resp_<symbol> for each responder symbol
          - pred_<symbol> for each predictor symbol
    """
    import pandas as pd

    # Pivot responders so each symbol is its own column
    responders_wide = responders_df.pivot(
        index=["start_time", "end_time"], columns="symbol", values="return"
    )
    # Prefix responder columns for clarity
    responders_wide.columns = [f"resp_{col}" for col in responders_wide.columns]
    responders_wide = responders_wide.reset_index()

    # Pivot predictors so each symbol is its own column
    predictors_wide = predictors_df.pivot(
        index=["start_time", "end_time"], columns="symbol", values="return"
    )
    # Prefix predictor columns
    predictors_wide.columns = [f"pred_{col}" for col in predictors_wide.columns]
    predictors_wide = predictors_wide.reset_index()

    # Merge on (start_time, end_time)
    merged = pd.merge(
        responders_wide, predictors_wide, on=["start_time", "end_time"], how="inner"
    )

    return merged
