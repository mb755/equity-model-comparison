"""!@package config_parser
@brief Module creating a parser for default command line arguments

@details
The default config_parser object can be further customized as needed.

For function details see the function documentation:
- config_parser.py

@author Mate Balogh
@date 2024-08-06
"""

import argparse
import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def default_parser(description):
    """!@brief creates a parser for default command line arguments
    @details
    The default arguments are: <br>
    -s, --start-date: start date for data collection <br>
    -e, --end-date: end date for data collection <br>
    -o, --output-suffix: string appended to all output filenames <br>
    -c, --config-file: configuration file defining initial hyperparameters <br>
    -a, --all-assets: use all assets instead of a pre-specified list of tickers <br>
    -t, --ticker-file: filename containing list of tickers to get data for <br>
    -ow, --overwrite: overwrite existing files <br>

    @param description (str): text that is displayed in the -h help output

    @return config parser: command line argument parser
    """

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--start-date",
        help="Start date for data collection (YYYY-MM-DD, inclusive)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-e",
        "--end-date",
        help="End date for data collection (YYYY-MM-DD, inclusive)",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-suffix",
        help="This string is appended to all output filenames",
        required=False,
        default="",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--config-file",
        help="Configuration file defining api keys",
        required=False,
        default=root_dir + "/config/api-keys.ini",
    )
    parser.add_argument(
        "-a",
        "--all-assets",
        help="Use all assets instead of a pre-specified list of tickers",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--ticker-file",
        help="Filename containing list of tickers to get data for",
        required=False,
        default=root_dir + "/config/default-tickers.csv",
    )
    parser.add_argument(
        "-ow",
        "--overwrite",
        help="Overwrite existing files",
        required=False,
        action="store_true",
    )
    return parser


def remove_argument(parser, arg):
    """!@brief remove an argument from an already existing command line parser
    @param parser (config parser): command line argument parser to remove from
    @param arg (str): parameter to remove, this is the long -- handle, with _s
    instead of -s (eg. random_seed instead of random-seed)
    """
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return
