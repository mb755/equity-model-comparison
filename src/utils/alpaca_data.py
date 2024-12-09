import pandas as pd
import numpy as np

from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.requests import StockQuotesRequest

from datetime import datetime, time, timedelta
from pytz import timezone


def get_active_assets(trading_client):
    """!@brief Get all active assets from the Alpaca API
    @param trading_client (TradingClient): Alpaca TradingClient object

    @return DataFrame: DataFrame containing all active assets
    """
    search_params = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE
    )

    active_assets = trading_client.get_all_assets(search_params)
    assets_df = pd.DataFrame(active_assets)

    # extract column names
    assets_df.columns = list(zip(*assets_df.loc[0]))[0]
    assets_df = assets_df.map(lambda x: x[1])

    return assets_df


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


def stockstime_request(tickers, date, time, duration):
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
        start=nyc.localize(datetime.combine(date, time) - duration),
        end=nyc.localize(datetime.combine(date, time) + duration),
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
def get_stockstime_data(marketdata_client, tickers, date, time, duration):
    """!@brief Get stock quotes around a particular time for a list of tickers
    @param trading_client (TradingClient): Alpaca TradingClient object
    @param tickers (List[str]): List of ticker symbols to get data for
    @param date (datetime.date): Date to get data for
    @param time (datetime.time): Time to get data for
    @param duration (datetime.timedelta): Duration to get data for

    @return pd.DataFrame: DataFrame containing stock quotes for each ticker
    """
    request_params = stockstime_request(tickers, date, time, duration)
    quotes = marketdata_client.get_stock_quotes(request_params)

    quotes_df = quotes.df

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
    marketdata_client, tickers, date, start_time, end_time, grid_span
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
            duration=timedelta(seconds=10),  # Assuming a 10-second window
        )
        df["time"] = t  # Add time column for tracking
        all_data.append(df)

    # Combine data into a single DataFrame
    combined_data = pd.concat(all_data, keys=time_points, names=["grid_time", "symbol"])
    combined_data.reset_index(level=0, inplace=True)

    return combined_data


def get_quote_summaries_across_dates(
    marketdata_client, tickers, start_date, end_date, start_time, end_time, grid_span
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

    Returns:
    - DataFrame: A combined DataFrame indexed by date, symbol, and time containing quote summaries.
    """
    all_data = []

    # Loop through the date range
    current_date = start_date
    while current_date <= end_date:
        try:
            # Get quote summaries for the current date
            daily_data = get_quote_summaries(
                marketdata_client=marketdata_client,
                tickers=tickers,
                date=current_date,
                start_time=start_time,
                end_time=end_time,
                grid_span=grid_span,
            )

            # Add a date column for tracking
            daily_data["date"] = current_date
            all_data.append(daily_data)

        except KeyError as e:
            # Handle missing data for closed markets or other issues
            print(f"No data for {current_date}: {e}")

        # Move to the next day
        current_date += timedelta(days=1)

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
