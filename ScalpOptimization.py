import time
import random
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Callable
from tabulate import tabulate
# Or another suitable optimizer from scikit-optimize
from skopt import Optimizer, gp_minimize
# For defining parameter search spaces
from skopt.space import Real, Integer, Categorical
# For handling named arguments in the objective function
from skopt.utils import use_named_args
import scipy.stats.mstats  # For winsorization

import matplotlib.pyplot as plt       # For plotting charts
# For displaying HTML content (if using Jupyter Notebook)
from IPython.display import HTML
from tqdm import tqdm

import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from joblib import Parallel, delayed
import hashlib
import json

events_df = pd.read_csv("events.csv", header=None)
events = events_df.values

probabilities = events[:, 0]
probabilities = probabilities / probabilities.sum()

account_details = {
    "Rally": {
        "initial_account_value": 0,
        "min_account_threshold": -1250,
        "payout_profit_target": 1500,
        "min_trading_days": 10,
        "flip_value": 1.25,
        "tick_value": 1.25,
        "payout_amount": 1500,
        "max_contracts": 20
    },
    "Grand Prix": {
        "initial_account_value": 0,
        "min_account_threshold": -7500,
        "payout_profit_target": 10500,
        "min_trading_days": 10,
        "flip_value": 12.5,
        "tick_value": 12.5,
        "payout_amount": 3000,
        "max_contracts": 15
    }
}


def simulateTradingDay(
    trade_outcomes: np.ndarray,  # ndarray of potential trade outcomes
    daily_profit_target: float,
    drawdown_limit: float,
    initial_account_value: float,
    min_account_threshold: float,
    max_trades: int,
    flip_on_payout_met: bool,
    flip_value: float,
    payout_threshold: float
) -> Tuple[float, float, int]:
    """
    Simulates a single trading day for a futures strategy.

    Args:
        trade_outcomes: NumPy ndarray of potential trade outcomes (profit/loss in dollars).
        daily_profit_target: The daily profit target for the day.
        drawdown_limit: Maximum allowable drawdown for the day (percentage or dollars).
        initial_account_value: Starting account value for the day.
        min_account_threshold: Minimum allowable account value during the day.
        max_trades: Maximum number of trades allowed in a day.
        flip_on_payout_met: If True, apply the flip logic when the account value
                            reaches or exceeds the payout_threshold within the day.
        flip_value: The fixed profit/loss amount applied in the flip logic
                    (if flip_on_payout_met is True and flip_value is positive).
                    A positive value results in a 50/50 chance of a profit or loss.
        payout_threshold: The threshold for triggering a payout (used in flip logic).

    Returns:
        A tuple containing:
            - final_account_value: Account value at the end of the day.
            - daily_profit_loss: Total profit or loss for the day.
            - num_trades: Number of trades executed during the day.
                          0 if a flip trade occurred.

    Raises:
        ValueError: If input values are invalid (e.g., negative account values,
                    drawdown_limit out of range).
    """

    # Input validation (ensure trade_outcomes is an ndarray)
    if not isinstance(trade_outcomes, np.ndarray):
        raise ValueError("trade_outcomes must be a NumPy ndarray.")

    # ... (Other input validation checks - similar to previous versions)

    account_value = initial_account_value
    daily_profit_loss = 0
    num_trades = 0

    if flip_value > 0 and flip_on_payout_met and account_value >= payout_threshold:
        # Apply flip logic if conditions are met
        num_trades = 0
        daily_profit_loss += flip_value if np.random.random() < 0.5 else -flip_value
        account_value += daily_profit_loss
    else:
        max_potential_trades = max_trades

        # Pre-generate trade outcomes for potential trades (using the provided ndarrays)
        all_trade_outcomes = np.random.choice(
            trade_outcomes,
            size=max_potential_trades,
            p=probabilities
        )

        # Calculate cumulative profit/loss and account values (using ndarrays)
        cumulative_profit_loss = np.cumsum(all_trade_outcomes)
        cumulative_account_values = initial_account_value + cumulative_profit_loss

        # Find the index where trading should stop
        stop_index = np.where(
            (cumulative_profit_loss >= daily_profit_target) |
            ((initial_account_value - cumulative_account_values) > drawdown_limit) |
            (cumulative_account_values < min_account_threshold) |
            (np.arange(max_potential_trades) >= max_trades - 1) |
            (flip_on_payout_met & (cumulative_account_values >= payout_threshold))
        )[0]

        if stop_index.size > 0:
            stop_index = stop_index[0]
        else:
            stop_index = max_potential_trades

        # Update final values based on the stopping point
        num_trades = stop_index + 1
        daily_profit_loss = cumulative_profit_loss[stop_index]
        account_value = cumulative_account_values[stop_index]

    return account_value, daily_profit_loss, num_trades


def simulatePayoutPeriod(
    trade_outcomes: np.ndarray,  # ndarray of potential trade outcomes
    daily_profit_target: float,
    payout_profit_target: float,
    min_trading_days: int,
    initial_account_value: float,
    min_account_threshold: float,
    max_trades: int,
    daily_drawdown_limit: float,
    is_first_payout_period: bool,
    flip_on_payout_met: bool,
    flip_value: float
) -> Tuple[float, float, bool, int, int, float, float, pd.DataFrame]:
    """
    Simulates a payout period for a futures strategy.

    Args:
        trade_outcomes: NumPy ndarray of potential trade outcomes (profit/loss in dollars).
        daily_profit_target: The daily profit target for each trading day.
        payout_profit_target: The account value target that triggers a payout.
        min_trading_days: Minimum number of trading days required for a payout.
        initial_account_value: Starting account value for the payout period.
        min_account_threshold: Minimum allowable account value.
        max_trades: Maximum number of trades allowed per day.
        daily_drawdown_limit: Maximum allowable drawdown within a single trading day.
        is_first_payout_period: Boolean indicating if this is the first payout period in the simulation.
        flip_on_payout_met: If True, apply the flip logic when the payout threshold is met within a day.
        flip_value: The fixed profit/loss amount applied in the flip logic
                    (if flip_on_payout_met is True and flip_value is positive).
                    A positive value results in a 50/50 chance of a profit or loss.

    Returns:
        A tuple containing:
            - final_account_value: Account value at the end of the payout period.
            - avg_trades_per_day: Average number of trades executed per day
                                  during the payout period (including flip days).
            - payout_achieved: True if the payout_profit_target was reached or exceeded,
                               False otherwise.
            - total_trading_days: Total number of trading days in the payout period
                                  (including flip days).
            - actual_trading_days: The number of trading days where actual trading occurred
                                   (excluding flip days).
            - avg_trades_per_trading_day: The average number of trades per day on
                                          actual trading days (excluding flip days).
            - max_drawdown: The maximum drawdown experienced during the payout period.
            - df: DataFrame containing the daily results of the payout period.

    Raises:
        ValueError: If input values are invalid (e.g., negative account values,
                    drawdown_limit out of range).
    """

    # Input validation
    if not isinstance(trade_outcomes, np.ndarray):
        raise ValueError("trade_outcomes must be a NumPy ndarray.")

    account_value = initial_account_value
    total_trading_days = 0
    payout_complete = False
    total_trades = 0
    highest_account_value = initial_account_value
    freeze_min_account_threshold = False
    payout_achieved = account_value >= payout_profit_target

    actual_trading_days = 0
    num_actual_trades = 0

    # Data tracking
    payout_period_data = []

    max_drawdown = 0
    abs_min_account_threshold = abs(min_account_threshold)  # Calculate once

    if is_first_payout_period:
        peak_account_value = abs_min_account_threshold
    else:
        peak_account_value = initial_account_value

    while not payout_complete:
        daily_profit_loss = 0

        if (flip_on_payout_met and payout_achieved):
            # Apply flip logic for the rest of the payout period
            # since the payout target has been met
            num_trades = 1  # Consider this a trade for avg calculation even if it's a flip

            if np.random.random() < 0.5:
                daily_profit_loss += flip_value
            else:
                daily_profit_loss -= flip_value

            account_value += daily_profit_loss
        else:
            account_value, daily_profit_loss, num_trades = simulateTradingDay(
                trade_outcomes,  # Pass ndarrays directly
                daily_profit_target,
                daily_drawdown_limit,
                account_value,
                min_account_threshold,
                max_trades,
                flip_on_payout_met,
                flip_value,
                payout_profit_target
            )

            # Increment num_actual_trades and actual_trading_days
            num_actual_trades += num_trades
            actual_trading_days += 1

        total_trading_days += 1
        total_trades += num_trades

        # Record data for this trading day
        payout_period_data.append({
            'Day': total_trading_days,
            'Number of Trades': num_trades,
            'Daily Profit/Loss': daily_profit_loss,
            'Account Value': account_value
        })

        if not payout_achieved:
            payout_achieved = account_value >= payout_profit_target

        # Check if payout is complete
        payout_complete = (
            account_value < min_account_threshold
            or (
                total_trading_days >= min_trading_days
                and account_value >= payout_profit_target
            )
        )

        # Update peak_account_value and calculate max_drawdown
        if is_first_payout_period:
            adjusted_account_value = account_value + abs_min_account_threshold
        else:
            adjusted_account_value = account_value

        # Update peak_account_value and calculate max_drawdown
        peak_account_value = max(peak_account_value, adjusted_account_value)
        current_drawdown = max(
            1, (peak_account_value - adjusted_account_value) / peak_account_value)
        max_drawdown = max(max_drawdown, current_drawdown)

        # Update min_account_threshold if this is the first payout period
        if is_first_payout_period and not freeze_min_account_threshold and account_value > highest_account_value:
            delta = account_value - highest_account_value
            highest_account_value = account_value
            min_account_threshold = min(0, min_account_threshold + delta)
            freeze_min_account_threshold = min_account_threshold == 0

    if total_trading_days > 0:
        avg_trades_per_day = total_trades / total_trading_days
    else:
        avg_trades_per_day = 0

    # Calculate average trades per actual trading day
    if actual_trading_days > 0:
        avg_trades_per_trading_day = num_actual_trades / actual_trading_days
    else:
        avg_trades_per_trading_day = 0

    # Create and print DataFrame
    df = pd.DataFrame(payout_period_data)
    # if len(df) <= 20:
    # print("\n" + df.to_string(index=False) + "\n")

    return account_value, avg_trades_per_day, payout_achieved, total_trading_days, actual_trading_days, avg_trades_per_trading_day, max_drawdown, df


def sampleSimulatePayoutPeriodCall():
    # Example parameter values (adjust these to match your strategy and preferences)
    tick_size = 12.5
    q1 = 0
    q2 = 4
    q3 = 2
    win_multiplier = 4
    loss_multiplier = 2.5

    full_win = q1 * 1 * tick_size + q2 * 2 * tick_size + q3 * 2 * tick_size
    partial_1 = q1 * 1 * tick_size + q2 * -5 * tick_size + q3 * -5 * tick_size
    full_loss = q1 * -8 * tick_size + q2 * -8 * tick_size + q3 * -8 * tick_size

    # Convert trade_outcomes and probabilities to ndarrays
    trade_outcomes = np.array([full_win, partial_1, full_loss])

    daily_profit_target = win_multiplier * full_win
    payout_profit_target = 10500  # Aiming for a $10,000 account value
    min_trading_days = 10
    initial_account_value = 7500
    min_account_threshold = 0  # Allow for an initial loss of up to $2000
    max_trades = 6
    daily_drawdown_limit = loss_multiplier * -full_loss
    is_first_payout_period = False
    flip_on_payout_met = True
    flip_value = tick_size

    print(trade_outcomes, daily_profit_target, daily_drawdown_limit)

    # Call the function
    final_account_value, avg_trades_per_day, payout_achieved, total_trading_days, actual_trading_days, avg_trades_per_trading_day, df = simulatePayoutPeriod(
        trade_outcomes,  # Pass as ndarray
        daily_profit_target,
        payout_profit_target,
        min_trading_days,
        initial_account_value,
        min_account_threshold,
        max_trades,
        daily_drawdown_limit,
        is_first_payout_period,
        flip_on_payout_met,
        flip_value
    )

    # Print the results
    print("\n*** Payout Period Results ***")
    print(f"Final Account Value: ${final_account_value:.2f}")
    print(f"Average Trades per Day: {avg_trades_per_day:.2f}")
    print(f"Payout Achieved: {payout_achieved}")
    print(f"Total Trading Days: {total_trading_days:.0f}")
    print(
        f"Actual Trading Days (excluding flip days): {actual_trading_days:.0f}")
    print(
        f"Average Trades per Actual Trading Day: {avg_trades_per_trading_day:.2f}")


# sampleSimulatePayoutPeriodCall()


def simulateEvaluation(
    trade_outcomes: np.ndarray,
    daily_profit_target: float,
    payout_profit_target: float,
    min_trading_days: int,
    initial_account_value: float,
    min_account_threshold: float,
    max_trades: int,
    daily_drawdown_limit: float,
    flip_on_payout_met: bool,
    flip_value: float,
    payouts_to_achieve: int,
    payout_amount: float
) -> Tuple[float, int, bool, float, float, float, float, float, float, float, pd.DataFrame]:
    """
    Simulates multiple payout periods to evaluate a trading strategy.

    Args:
        trade_outcomes: NumPy ndarray of potential trade outcomes (profit/loss in dollars).
        daily_profit_target: The daily profit target for each trading day.
        payout_profit_target: The initial account value target that triggers a payout.
        min_trading_days: Minimum number of trading days required for a payout.
        initial_account_value: The starting account value for the evaluation.
        min_account_threshold: The minimum allowable account value.
                               If the account value falls below this threshold,
                               the evaluation ends.
        max_trades: Maximum number of trades allowed per day.
        daily_drawdown_limit: Maximum allowable drawdown within a single trading day.
        flip_on_payout_met: If True, apply the flip logic when the payout threshold is met within a day.
        flip_value: The fixed profit/loss amount applied in the flip logic
                    (if flip_on_payout_met is True and flip_value is positive).
                    A positive value results in a 50/50 chance of a profit or loss.
        payouts_to_achieve: The target number of successful payouts to achieve
                            during the evaluation.
        payout_amount: The amount to be withdrawn from the account value
                       after each successful payout.

    Returns:
        A tuple containing:
            - final_account_value: The account value at the end of the evaluation.
            - total_payouts: The total number of successful payouts achieved.
            - evaluation_succeeded: True if the target number of payouts was achieved,
                                    False otherwise.
            - avg_trades_per_day: The average number of trades per day across all payout periods
                                  (including flip days).
            - avg_total_trading_days: The average number of total trading days per payout period
                                      (including flip days).
            - avg_actual_trading_days: The average number of actual trading days per payout period
                                       (excluding flip days).
            - avg_avg_trades_per_trading_day: The average of the average number of trades per trading day
                                              across all payout periods (excluding flip days).
            - total_amount_paid_out: The total amount paid out during the evaluation.
            - total_trading_days_all_periods: The total number of trading days across all payout periods
                                                (including flip days, without manipulation).
            - max_drawdown: The maximum drawdown experienced across all payout periods.
            - df_evaluation: DataFrame containing the evaluation results.
    Raises:
        ValueError: If input values are invalid (e.g., negative account values,
                    drawdown_limit out of range).
    """

    # Input validation (ensure trade_outcomes is an ndarray)
    if not isinstance(trade_outcomes, np.ndarray):
        raise ValueError("trade_outcomes must be a NumPy ndarray.")

    account_value = initial_account_value
    total_payouts = 0

    total_avg_trades_per_day = 0
    total_total_trading_days = 0
    total_actual_trading_days = 0
    total_avg_trades_per_trading_day = 0
    total_amount_paid_out = 0

    payout_period_tracking_data = []
    evaluation_data = []
    total_trading_days_all_periods = 0
    evaluation_max_drawdown = 0

    while total_payouts < payouts_to_achieve and account_value >= min_account_threshold:
        payout_period_tracking_data.append({
            'Payout Period': total_payouts,
            'Account Value': account_value,
            'Payout Profit Target': payout_profit_target,
            'Minimum Account Threshold': min_account_threshold
        })

        # Pass all necessary parameters to simulatePayoutPeriod
        (
            account_value,
            pp_avg_trades_per_day,  # Use different variable names for return values
            payout_achieved,
            pp_total_trading_days,
            pp_actual_trading_days,
            pp_avg_trades_per_trading_day,
            pp_max_drawdown,
            _
        ) = simulatePayoutPeriod(
            trade_outcomes,
            daily_profit_target,
            payout_profit_target,
            min_trading_days,
            account_value,
            min_account_threshold,
            max_trades,
            daily_drawdown_limit,
            total_payouts == 0,
            flip_on_payout_met,
            flip_value,
        )

        total_trading_days_all_periods += pp_total_trading_days
        evaluation_max_drawdown = max(evaluation_max_drawdown, pp_max_drawdown)

        if payout_achieved:
            if total_payouts == 0:
                min_account_threshold = 0

            # Accumulate values for averaging (including the first payout)
            if total_payouts == 1:
                total_avg_trades_per_day = pp_avg_trades_per_day
                total_total_trading_days = pp_total_trading_days
                total_actual_trading_days = pp_actual_trading_days
                total_avg_trades_per_trading_day = pp_avg_trades_per_trading_day
            else:
                total_avg_trades_per_day += pp_avg_trades_per_day
                total_total_trading_days += pp_total_trading_days
                total_actual_trading_days += pp_actual_trading_days
                total_avg_trades_per_trading_day += pp_avg_trades_per_trading_day

            # Update payout_profit_target, withdraw payout_amount, and track total paid out
            payout_profit_target = account_value
            account_value -= payout_amount
            total_amount_paid_out += payout_amount

            total_payouts += 1
        elif total_payouts == 0:  # Added condition to break if the first payout fails
            break

        # Record data for this payout period
        evaluation_data.append({
            'Payout Period': total_payouts,  # 1-indexed for user readability
            'Account Value': account_value,
            'Payout Achieved': payout_achieved,
            'Total Trading Days': pp_total_trading_days
        })

    evaluation_succeeded = total_payouts == payouts_to_achieve

    # Calculate averages, excluding the first payout if there were multiple
    avg_trades_per_day = 0
    avg_total_trading_days = 0
    avg_actual_trading_days = 0
    avg_avg_trades_per_trading_day = 0

    # Calculate averages, excluding the first payout if there were multiple
    if total_payouts > 1:
        divisor = (total_payouts - 1)
        avg_trades_per_day = total_avg_trades_per_day / divisor
        avg_total_trading_days = total_total_trading_days / divisor
        avg_actual_trading_days = total_actual_trading_days / divisor
        avg_avg_trades_per_trading_day = total_avg_trades_per_trading_day / divisor
    else:
        # If only one or zero payouts were achieved, use the accumulated totals directly (or 0 if none)
        avg_trades_per_day = total_avg_trades_per_day if total_payouts > 0 else 0
        avg_total_trading_days = total_total_trading_days if total_payouts > 0 else 0
        avg_actual_trading_days = total_actual_trading_days if total_payouts > 0 else 0
        avg_avg_trades_per_trading_day = total_avg_trades_per_trading_day if total_payouts > 0 else 0

    df_evaluation = pd.DataFrame(evaluation_data)
    df_payout_period_tracking = pd.DataFrame(payout_period_tracking_data)

    # # Print the payout_period_tracking DataFrame
    # print(tabulate(df_payout_period_tracking, headers='keys',
    #       tablefmt='psql', showindex=False, floatfmt=".2f"))

    return (
        account_value,
        total_payouts,
        evaluation_succeeded,
        avg_trades_per_day,
        avg_total_trading_days,
        avg_actual_trading_days,
        avg_avg_trades_per_trading_day,
        total_amount_paid_out,
        total_trading_days_all_periods,
        evaluation_max_drawdown,
        df_evaluation
    )


def sampleSimulateEvaluationCall():
    # Example parameter values (reusing from sampleSimulatePayoutPeriodCall)
    tick_size = 12.5
    q1 = 0
    q2 = 0
    q3 = 6
    t1 = 1
    t2 = 2
    t3 = 3
    win_multiplier = 3
    loss_multiplier = 1.5

    full_win = q1 * t1 * tick_size + q2 * t2 * tick_size + q3 * t3 * tick_size
    partial_1 = q1 * 1 * tick_size + q2 * -5 * tick_size + q3 * -5 * tick_size
    full_loss = q1 * -8 * tick_size + q2 * -8 * tick_size + q3 * -8 * tick_size

    trade_outcomes = np.array([full_win, partial_1, full_loss])
    daily_profit_target = win_multiplier * full_win
    payout_profit_target = 10500
    min_trading_days = 10
    initial_account_value = 0
    min_account_threshold = -7500
    max_trades = 5
    daily_drawdown_limit = loss_multiplier * -full_loss
    flip_on_payout_met = True
    flip_value = tick_size

    # Additional parameters specific to simulateEvaluation
    payouts_to_achieve = 3  # Example: Aim to achieve 3 payouts
    payout_amount = 3000     # Example: Withdraw $5000 after each successful payout

    print(trade_outcomes, daily_profit_target, daily_drawdown_limit)

    # Call simulateEvaluation
    (
        final_account_value,
        total_payouts,
        evaluation_succeeded,
        avg_trades_per_day,
        avg_total_trading_days,
        avg_actual_trading_days,
        avg_avg_trades_per_trading_day,
        total_amount_paid_out,
        df_evaluation
    ) = simulateEvaluation(
        trade_outcomes,
        daily_profit_target,
        payout_profit_target,
        min_trading_days,
        initial_account_value,
        min_account_threshold,
        max_trades,
        daily_drawdown_limit,
        flip_on_payout_met,
        flip_value,
        payouts_to_achieve,
        payout_amount
    )

    # Print the results
    print("\n*** Evaluation Results ***")
    print(f"Final Account Value: ${final_account_value:.2f}")
    print(f"Total Payouts Achieved: {total_payouts}")
    print(f"Total Amount Paid Out: ${total_amount_paid_out:.2f}")
    print(f"Evaluation Succeeded: {evaluation_succeeded}")
    print("\n*** Averages Across Payout Periods ***")
    print(f"Average Trades per Day: {avg_trades_per_day:.2f}")
    print(f"Average Total Trading Days: {avg_total_trading_days:.2f}")
    print(f"Average Actual Trading Days: {avg_actual_trading_days:.2f}")
    print(
        f"Average Trades per Actual Trading Day: {avg_avg_trades_per_trading_day:.2f}")
    print(f"Total Amount Paid Out: ${total_amount_paid_out:.2f}")

    print("\n*** Payout Period Data ***")
    print(f"Average Total Trading Days: {avg_total_trading_days:.2f}")
    print(tabulate(df_evaluation, headers='keys',
          tablefmt='psql', showindex=False, floatfmt=".2f"))


# sampleSimulateEvaluationCall()

def calculate_trade_outcomes(q1: int, q2: int, q3: int, tick_value: float) -> np.ndarray:
    """
    Calculates the potential trade outcomes (profit/loss) based on trade quantities and tick value.

    Args:
        q1, q2, q3: Integer quantities representing the trade sizes for different trade types.
        tick_value: The tick value used to calculate profit/loss.

    Returns:
        A NumPy ndarray containing the calculated trade outcomes.
    """

    # Pre-allocate the trade_outcomes array with the correct size
    # Get the number of rows (events) in the events array
    num_events = events.shape[0]
    trade_outcomes = np.empty(num_events)

    for index, event in enumerate(events):
        probability, tick_multiplier_1, tick_multiplier_2, tick_multiplier_3 = event

        # Calculate the trade outcome for this event
        trade_outcome = (q1 * tick_multiplier_1 + q2 *
                         tick_multiplier_2 + q3 * tick_multiplier_3) * tick_value

        # Assign the trade outcome to the corresponding index in the trade_outcomes array
        trade_outcomes[index] = trade_outcome

    return trade_outcomes

# # Sample calls
# space_with_real_steps = [
#     Categorical(get_step_values(0.2, 3.7, 0.3), name='some_param')
# ]

# space_with_integer_steps = [
#     # Directly use range for integers
#     Categorical(range(1, 11, 2), name='num_layers')
# ]


def constraint_func(x, accountType):
    """
    This function acts as a constraint for the optimizer.
    It ensures that the sum of the first three parameters (q1, q2, q3) 
    does not exceed the maximum number of contracts allowed for the given account type.

    Args:
        x: A list of parameter values representing a point in the search space.

    Returns:
        True if the constraint is satisfied (sum of q1, q2, q3 <= max_contracts), False otherwise.
    """
    q1, q2, q3, *_ = x  # Unpack the first three parameters (q1, q2, q3)
    # Get max_contracts from account_details
    max_contracts = account_details[accountType]['max_contracts']
    return q1 + q2 + q3 <= max_contracts

# objective_function(params)
# optimize (partially)


def initialize_param_space(accountType: str):
    """
    Initializes the parameter space for the trading strategy optimization based on the account type.

    Args:
        accountType: The type of account ("Rally" or "Grand Prix") to determine the parameter ranges.

    Returns:
        A dictionary where keys are parameter names and values are tuples of the format:
        (param_type, low, high, [step])
        - param_type: A string indicating the parameter type ("bool", "int", or "float").
        - low, high: The lower and upper bounds of the parameter search space.
        - step (optional): The step size for integer or float parameters.

    Parameter Descriptions:
        - q1, q2, q3: Trade size quantities for different trade types (integers).
        - win_multiplier: Multiplier for calculating the daily profit target (float with step).
        - loss_multiplier: Multiplier for calculating the daily drawdown limit (float).
        - max_trades: The maximum number of trades allowed per day (integer).
        - flip_on_payout_met: Whether to apply the "flip" logic when the payout target is met within a day (boolean).

    Raises:
        ValueError: If `accountType` is not "Rally" or "Grand Prix".
    """

    if accountType not in ["Rally", "Grand Prix"]:
        raise ValueError(
            "Invalid accountType. Must be 'Rally' or 'Grand Prix'")

    max_contracts = account_details[accountType]['max_contracts']

    # Directly create param_space
    param_space = {
        'q1': ("int", 0, max_contracts, 1),
        'q2': ("int", 0, max_contracts, 1),
        'q3': ("int", 0, max_contracts, 1),
        'win_multiplier': ("float", 1, 5.5, 0.5),  # Using np.arange for steps
        'loss_multiplier': ("float", 1, 3.5, 0.5),  # Using np.arange for steps
        'max_trades': ("int", 1, 15, 1),
        'flip_on_payout_met': ("bool", True, False)
    }

    return param_space


def calculate_n_range_simulations(accountType: str):
    """
    Calculates a reasonable number of simulations for dynamic range calculation based on the parameter space,
    taking into account the constraint on the sum of quantity parameters and the number of items in each category.

    Args:
        accountType: The type of account ("Rally" or "Grand Prix") to determine the parameter ranges.

    Returns:
        The calculated number of simulations for range calculation.
    """
    param_space = initialize_param_space(
        accountType)  # Get the parameter space

    # Define dimensions for Bayesian optimization (needed to analyze the parameter space)
    dimensions = []
    for key, param_settings in param_space.items():
        # Unpack the first three elements
        param_type, low, high = param_settings[:3]
        # Get step if available, else None
        step = param_settings[3] if len(param_settings) == 4 else None

        if param_type == "bool":
            dimensions.append(Categorical([True, False], name=key))
        elif param_type == "int":
            if step is not None:  # Check if step is provided
                dimensions.append(Categorical(
                    range(low, high + 1, step), name=key))
            else:
                dimensions.append(Integer(low, high, name=key))
        elif param_type == "float":
            if step is not None:
                dimensions.append(Categorical(
                    np.arange(low, high + step, step), name=key))
            else:
                dimensions.append(Real(low, high, name=key))

    n_categorical_dimensions = sum(1 for dim in dimensions if isinstance(
        dim, Categorical))  # Count Categorical dimensions

    # 2. Estimate the Proportion of Feasible Combinations (using random sampling)
    # Get max_contracts from account details
    max_contracts = account_details[accountType]['max_contracts']
    # Number of samples for estimating feasible proportion (adjust as needed)
    n_samples_for_estimation = 1000
    feasible_count = 0
    for _ in range(n_samples_for_estimation):
        point = [random.choice(dim.categories)
                 for dim in dimensions]  # Generate a random point
        if point[0] + point[1] + point[2] <= max_contracts:  # Check the constraint
            feasible_count += 1
    feasible_proportion = feasible_count / n_samples_for_estimation

    # 3. Adjust n_range_simulations, considering the number of items in each category
    category_sizes = [len(dim.categories)
                      for dim in dimensions if isinstance(dim, Categorical)]
    avg_category_size = sum(
        category_sizes) / len(category_sizes) if category_sizes else 1  # Handle empty list

    # You can adjust these factors based on your desired confidence level and computational constraints
    base_simulations_per_dimension = 10  # Base number of simulations per dimension
    desired_confidence_factor = 1.5  # Increase simulations for higher confidence

    n_range_simulations = int(
        (base_simulations_per_dimension * n_categorical_dimensions * desired_confidence_factor * avg_category_size) /
        feasible_proportion
    )

    return n_range_simulations


def generate_random_parameter_sets(param_space, n_samples, accountType):
    """
    Generates random parameter sets within the defined parameter space, respecting the max_contracts constraint.

    Args:
        param_space: The parameter space dictionary.
        n_samples: The number of random parameter sets to generate
        accountType: The type of account ("Rally" or "Grand Prix") to determine the parameter ranges.

    Returns:
        A list of dictionaries, where each dictionary represents a parameter set.
    """

    max_contracts = account_details[accountType]['max_contracts']

    def generate_valid_point():
        """
        Helper function to generate a single valid parameter set that satisfies the max_contracts constraint.
        """
        while True:
            # Generate random values for each parameter based on its type and step size
            point = [
                # Handle int with step
                random.choice(list(range(low, high + 1, step))) if param_type == "int" and step is not None else
                random.choice(list(np.arange(low, high + step, step))) if param_type == "float" and step is not None else
                random.randint(low, high) if param_type == "int" else
                random.uniform(low, high) if param_type == "float" else
                # Assuming boolean parameters have True/False as low/high
                # Handle boolean parameters correctly
                random.choice([low, high])
                for _, param_settings in param_space.items()
                # Unpack only the first 3 elements
                for param_type, low, high in [param_settings[:3]]
                # Extract step if available
                for step in [param_settings[3] if len(param_settings) == 4 else None]
            ]

            # Check if the sum of q1, q2, and q3 satisfies the max_contracts constraint
            if point[0] + point[1] + point[2] <= max_contracts:
                return point

    parameter_sets = []
    for _ in range(n_samples):
        # Generate a valid parameter set
        valid_point = generate_valid_point()

        # Create a dictionary mapping parameter names to their values
        parameter_set = {param_name: x for param_name,
                         x in zip(param_space.keys(), valid_point)}
        parameter_sets.append(parameter_set)

    return parameter_sets


def calculate_dynamic_ranges(param_space, accountType, n_range_simulations, n_simulations_per_set, objective_weights, n_jobs, payouts_to_achieve):
    """
    Calculates dynamic ranges for objective values based on simulations with random parameter sets.

    Args:
        param_space: The parameter space dictionary.
        accountType: The type of account ("Rally" or "Grand Prix").
        n_range_simulations: The number of random parameter sets to simulate for each dynamic range set.
        n_simulations_per_set: The number of simulations to run for each parameter set.
        objective_weights: Dictionary specifying weights, maximization/minimization, and winsorization for each objective.
        n_jobs: Number of parallel jobs (-1 for all cores).
        payouts_to_achieve: The target number of successful payouts to achieve during the evaluation.

    Returns:
        A dictionary where keys are objective names and values are tuples of the format: 
        (min_value, max_value, mean_value, std_value)
    """

    # Look up static parameters from account_details
    initial_account_value = account_details[accountType]['initial_account_value']
    min_account_threshold = account_details[accountType]['min_account_threshold']
    payout_profit_target = account_details[accountType]['payout_profit_target']
    min_trading_days = account_details[accountType]['min_trading_days']
    flip_value = account_details[accountType]['flip_value']
    tick_value = account_details[accountType]['tick_value']
    payout_amount = account_details[accountType]['payout_amount']

    random_parameter_sets = generate_random_parameter_sets(
        param_space, n_range_simulations, accountType)

    # Lists to store objective values from initial simulations
    initial_final_values = []
    initial_total_trading_days = []
    initial_trades_per_day_values = []
    initial_total_trading_days_per_period_values = []
    initial_actual_trading_days_per_period_values = []
    initial_avg_trades_per_trading_day_values = []
    initial_total_amount_paid_out_values = []
    initial_max_drawdown_values = []

    # Pre-calculate trade_outcomes and other derived values for each parameter set
    simulation_inputs = [
        (
            calculate_trade_outcomes(
                params['q1'], params['q2'], params['q3'], tick_value),
            params['win_multiplier'] * calculate_trade_outcomes(
                params['q1'], params['q2'], params['q3'], tick_value)[0],
            params['loss_multiplier'] * -calculate_trade_outcomes(
                params['q1'], params['q2'], params['q3'], tick_value)[-1],
            params['max_trades'],
            params['flip_on_payout_met']
        )
        for params in random_parameter_sets
    ]

    # Parallelize the initial simulations
    start_time = time.time()
    # with Parallel(n_jobs=n_jobs) as parallel:
    #     results = parallel(
    #         delayed(simulateEvaluation)(
    #             trade_outcomes=trade_outcomes,
    #             daily_profit_target=daily_profit_target,
    #             daily_drawdown_limit=daily_drawdown_limit,
    #             max_trades=max_trades,
    #             flip_on_payout_met=flip_on_payout_met,
    #             payout_profit_target=payout_profit_target,
    #             min_trading_days=min_trading_days,
    #             initial_account_value=initial_account_value,
    #             min_account_threshold=min_account_threshold,
    #             flip_value=flip_value,
    #             payouts_to_achieve=payouts_to_achieve,
    #             payout_amount=payout_amount
    #         )
    #         for (trade_outcomes, daily_profit_target, daily_drawdown_limit, max_trades, flip_on_payout_met) in simulation_inputs
    #     )

    results = []
    for (trade_outcomes, daily_profit_target, daily_drawdown_limit, max_trades, flip_on_payout_met) in tqdm(simulation_inputs, desc="Simulating"):
        results.append(simulateEvaluation(
            trade_outcomes=trade_outcomes,
            daily_profit_target=daily_profit_target,
            daily_drawdown_limit=daily_drawdown_limit,
            max_trades=max_trades,
            flip_on_payout_met=flip_on_payout_met,
            payout_profit_target=payout_profit_target,
            min_trading_days=min_trading_days,
            initial_account_value=initial_account_value,
            min_account_threshold=min_account_threshold,
            flip_value=flip_value,
            payouts_to_achieve=payouts_to_achieve,
            payout_amount=payout_amount
        ))

    end_time = time.time()
    print(
        f"Time taken for initial simulations: {end_time - start_time:.2f} seconds")

    # Extract objective values from the results
    for (
        final_account_value,
        total_trading_days_all_periods,
        avg_trades_per_day,
        avg_total_trading_days,
        avg_actual_trading_days,
        avg_avg_trades_per_trading_day,
        total_amount_paid_out,
        max_drawdown,
        _  # Ignore the DataFrame for now
    ) in results:
        initial_final_values.append(final_account_value)
        initial_total_trading_days.append(total_trading_days_all_periods)
        initial_trades_per_day_values.append(avg_trades_per_day)
        initial_total_trading_days_per_period_values.append(
            avg_total_trading_days)
        initial_actual_trading_days_per_period_values.append(
            avg_actual_trading_days)
        initial_avg_trades_per_trading_day_values.append(
            avg_avg_trades_per_trading_day)
        initial_total_amount_paid_out_values.append(total_amount_paid_out)
        initial_max_drawdown_values.append(max_drawdown)

    # Calculate Dynamic Ranges with Winsorization and Store Mean/Std
    dynamic_ranges = {}
    for obj_name, obj_settings in objective_weights.items():
        if obj_name == "avg_final_value":
            values = initial_final_values
        elif obj_name == "total_trading_days":
            values = initial_total_trading_days
        elif obj_name == "avg_trades_per_day":
            values = initial_trades_per_day_values
        elif obj_name == "avg_total_trading_days":
            values = initial_total_trading_days_per_period_values
        elif obj_name == "avg_actual_trading_days":
            values = initial_actual_trading_days_per_period_values
        elif obj_name == "avg_avg_trades_per_trading_day":
            values = initial_avg_trades_per_trading_day_values
        elif obj_name == "avg_total_amount_paid_out":
            values = initial_total_amount_paid_out_values
        elif obj_name == "max_drawdown":
            values = initial_max_drawdown_values

        winsorized_values = scipy.stats.mstats.winsorize(values, limits=(
            obj_settings["lower_percentile"]/100, obj_settings["upper_percentile"]/100))
        mean_value = np.mean(winsorized_values)
        std_value = np.std(winsorized_values)
        min_value = min(winsorized_values)
        max_value = max(winsorized_values)

        dynamic_ranges[obj_name] = (
            min_value, max_value, mean_value, std_value)

    return dynamic_ranges


def generate_multiple_dynamic_ranges(
    num_dynamic_ranges: int,
    param_space: dict,
    accountType: str,
    n_simulations_per_set: int,
    n_jobs: int,
    payouts_to_achieve: int
) -> List[Dict]:
    """
    Generates multiple sets of dynamic ranges for objective values based on simulations 
    with random parameter sets.

    Args:
        num_dynamic_ranges: The number of dynamic range sets to generate.
        param_space: The parameter space dictionary.
        accountType: The type of account ("Rally" or "Grand Prix").
        n_simulations_per_set: The number of simulations to run for each parameter set.
        n_jobs: Number of parallel jobs (-1 for all cores).
        payouts_to_achieve: The target number of successful payouts to achieve 
                             during the evaluation.

    Returns:
        A list of dictionaries, where each dictionary represents a set of 
        dynamic ranges for the objectives.
    """

    # Create simplified objective_weights with placeholders for percentiles
    objective_weights = {
        "avg_final_value": {"lower_percentile": 5, "upper_percentile": 95},
        "total_trading_days": {"lower_percentile": 25, "upper_percentile": 75},
        "avg_trades_per_day": {"lower_percentile": 25, "upper_percentile": 75},
        "avg_total_trading_days": {"lower_percentile": 25, "upper_percentile": 75},
        "avg_actual_trading_days": {"lower_percentile": 25, "upper_percentile": 75},
        "avg_avg_trades_per_trading_day": {"lower_percentile": 25, "upper_percentile": 75},
        "avg_total_amount_paid_out": {"lower_percentile": 5, "upper_percentile": 95},
        "max_drawdown": {"lower_percentile": 25, "upper_percentile": 75}
    }

    # Calculate n_range_simulations outside the loop
    n_range_simulations = calculate_n_range_simulations(accountType)

    # Parallelize the dynamic range calculations
    # with Parallel(n_jobs=n_jobs) as parallel:
    #     all_dynamic_ranges = parallel(
    #         delayed(calculate_dynamic_ranges)(
    #             param_space, accountType, n_range_simulations, n_simulations_per_set,
    #             objective_weights, n_jobs, payouts_to_achieve
    #         )
    #         for _ in range(num_dynamic_ranges)
    #     )

    all_dynamic_ranges = []
    for _ in tqdm(range(num_dynamic_ranges), desc="Generating Dynamic Ranges"):
        all_dynamic_ranges.append(calculate_dynamic_ranges(
            param_space, accountType, n_range_simulations, n_simulations_per_set,
            objective_weights, n_jobs, payouts_to_achieve
        ))

    # Save each set of dynamic ranges to a JSON file
    # Assuming you want to save in a "candidates" subdirectory
    save_directory = f"dynamic_ranges/{accountType}/candidates"
    os.makedirs(save_directory, exist_ok=True)

    for i, dynamic_ranges in enumerate(all_dynamic_ranges):
        # Sort objectives alphabetically and generate hash for filename
        sorted_objectives = sorted(objective_weights.keys())
        filename = hashlib.md5(
            "_".join(sorted_objectives).encode()).hexdigest() + f"_{i}.json"
        filepath = os.path.join(save_directory, filename)
        with open(filepath, "w") as f:
            json.dump(dynamic_ranges, f)

    return all_dynamic_ranges


def analyze_dynamic_ranges(account_type: str, objectives: List[str], min_max_std_threshold=0.1, mean_std_threshold=0.05):
    """
    Analyzes the stability of dynamic ranges loaded from JSON files,
    creates a DataFrame with stability information, 
    calculates weighted averages for stable ranges, and saves them to a file.

    Args:
        account_type: The type of account ("Rally" or "Grand Prix").
        objectives: A list of objective names to analyze.
        min_max_std_threshold: The threshold for the standard deviation of min and max values (default: 0.1).
        mean_std_threshold: The threshold for the standard deviation of mean and std values (default: 0.05).

    Returns:
        A tuple containing:
            - stable_ranges: A dictionary of stable ranges for each objective (if found), 
                             or None if no stable ranges were identified.
            - range_stats: A dictionary containing statistics (mean, std, range) for each 
                           objective and range type across the multiple sets.
            - df: The DataFrame containing stability information for each dynamic range set.
    """

    load_directory = f"dynamic_ranges/{account_type}/candidates"
    filenames = [f for f in os.listdir(load_directory) if f.endswith('.json')]

    data = []
    for filename in filenames:
        filepath = os.path.join(load_directory, filename)
        with open(filepath, 'r') as f:
            dynamic_ranges = json.load(f)  # Load dynamic ranges from JSON file

        range_stats = {}  # Store statistics for each objective and range type
        is_stable_overall = True

        for obj_name in objectives:
            min_val, max_val, mean_value, std_value = dynamic_ranges[obj_name]

            # Calculate standard deviations for min, max, mean, and std values
            range_stats[obj_name] = {
                'min_std': np.std([min_val]),
                'max_std': np.std([max_val]),
                'mean_std': np.std([mean_value]),
                'std_std': np.std([std_value])
            }

            # Check stability using the provided thresholds
            is_stable = all([
                range_stats[obj_name]['min_std'] < min_max_std_threshold,
                range_stats[obj_name]['max_std'] < min_max_std_threshold,
                range_stats[obj_name]['mean_std'] < mean_std_threshold,
                range_stats[obj_name]['std_std'] < mean_std_threshold
            ])

            is_stable_overall &= is_stable  # Update overall stability

        # Create a row for the DataFrame
        row = {'guid': filename.replace(".json", "")}
        for obj_name in objectives:
            for stat_name, stat_value in range_stats[obj_name].items():
                row[f"{obj_name}_{stat_name}"] = stat_value
        row['stable'] = is_stable_overall

        data.append(row)

    df = pd.DataFrame(data)

    # Calculate weighted averages for stable ranges
    stable_ranges = {}
    if df['stable'].any():  # Check if there are any stable range sets
        for obj_name in objectives:
            stable_df = df[df['stable']]

            # Weights based on inverse of standard deviation (lower std -> higher weight)
            weights = 1 / (stable_df[f"{obj_name}_min_std"] + stable_df[f"{obj_name}_max_std"]
                           + stable_df[f"{obj_name}_mean_std"] + stable_df[f"{obj_name}_std_std"])

            # Calculate weighted averages
            stable_ranges[obj_name] = (
                np.average(stable_df[f"{obj_name}_min"], weights=weights),
                np.average(stable_df[f"{obj_name}_max"], weights=weights),
                np.average(
                    stable_df[f"{obj_name}_mean_mean"], weights=weights),
                np.average(stable_df[f"{obj_name}_std_mean"], weights=weights)
            )

        # Save stable_ranges to a file
        save_directory = f"dynamic_ranges/{account_type}"
        os.makedirs(save_directory, exist_ok=True)

        # Sort objectives alphabetically and generate hash for filename
        sorted_objectives = sorted(objectives)
        filename = hashlib.md5(
            "_".join(sorted_objectives).encode()).hexdigest() + ".json"
        filepath = os.path.join(save_directory, filename)
        with open(filepath, "w") as f:
            json.dump(stable_ranges, f)

    return stable_ranges, range_stats, df


def generate_and_analyze_dynamic_ranges(account_type: str, num_dynamic_ranges: int = 20, n_simulations_per_set: int = 10, n_jobs: int = -1, payouts_to_achieve: int = 3):
    """
    Generates and analyzes multiple sets of dynamic ranges for the specified account type.

    Args:
        account_type: The type of account ("Rally" or "Grand Prix").
        num_dynamic_ranges: The number of dynamic range sets to generate (default: 20).
        n_simulations_per_set: The number of simulations to run for each parameter set (default: 10).
        n_jobs: Number of parallel jobs (-1 for all cores, default: -1).
        payouts_to_achieve: The target number of successful payouts to achieve during the evaluation (default 3).

    Returns:
        A tuple containing:
            - stable_ranges: A dictionary of stable ranges for each objective (if found), or None if no stable ranges were identified.
            - range_stats: A dictionary containing statistics (mean, std, range) for each objective and range type across the multiple sets.
            - df_analysis: The DataFrame containing stability information for each dynamic range set.
    """

    # Get the param_space
    param_space = initialize_param_space(account_type)

    # Generate multiple dynamic ranges
    generate_multiple_dynamic_ranges(
        num_dynamic_ranges,
        param_space,
        account_type,
        n_simulations_per_set,
        n_jobs,
        payouts_to_achieve
    )

    # Analyze the generated dynamic ranges
    stable_ranges, range_stats, df_analysis = analyze_dynamic_ranges(
        account_type, list(objective_weights.keys()))

    # Print or display the results (you can customize this based on your needs)
    print(df_analysis)

    if stable_ranges:
        print("\nStable Ranges Found:")
        for obj_name, ranges in stable_ranges.items():
            print(f"{obj_name}: {ranges}")
    else:
        print("\nNo Stable Ranges Found.")

    return stable_ranges, range_stats, df_analysis


stable_ranges, range_stats, df_analysis = generate_and_analyze_dynamic_ranges(
    account_type="Grand Prix",
    num_dynamic_ranges=50,
    payouts_to_achieve=6)  # Or "Rally"


def optimize(
    accountType: str,
    payouts_to_achieve: int,
    success_probability_threshold: float = 0.8,
    objective_weights: Dict[str,
                            Dict[str, Union[float, bool, int, int]]] = None,
    acq_func: str = "EI",
    n_calls: int = 100,
    n_random_starts: int = 10,
    n_jobs: int = -1,
    num_simulations: int = 100,
    top_n_results: int = 5,
) -> List[Tuple[Dict[str, Any], float, float, np.ndarray, float]]:
    """
    Optimizes trading strategy parameters using nested Monte Carlo simulations and Bayesian optimization.

    Args:
        accountType: Account type to look up static parameters.
        payouts_to_achieve: Target number of successful payouts.
        success_probability_threshold: Minimum probability of success (default: 0.8).
        objective_weights: Dictionary specifying weights, maximization/minimization, and winsorization for each objective.
        acq_func: Acquisition function for Bayesian optimization (default: "EI").
        n_calls: Maximum iterations for Bayesian optimization (default: 100).
        n_random_starts: Number of random initialization points (default: 10).
        n_jobs: Number of parallel jobs (-1 for all cores, default: -1).
        num_simulations: Number of Monte Carlo simulations per parameter set.
        top_n_results: Number of top results to return (default: 5).

    Returns:
        A tuple containing:
            - top_results: List of top parameter sets with their avg_final_value, avg_total_trading_days, trade_outcomes, and combined_score.
            - param_space: The parameter space used for the optimization.
    """

    # Input validation for objective_weights
    if objective_weights is None:
        objective_weights = {
            "avg_final_value": {"weight": 0.75, "maximize": True, "lower_percentile": 1, "upper_percentile": 99},
            "total_trading_days": {"weight": 0.25, "maximize": False, "lower_percentile": 1, "upper_percentile": 99}
        }
    elif sum(obj["weight"] for obj in objective_weights.values()) != 1.0:
        raise ValueError("The sum of weights in objective_weights must be 1.0")

    # 2. Load or Calculate Dynamic Ranges
    save_directory = f"dynamic_ranges/{accountType}"

    # Sort objectives alphabetically and generate hash for filename
    sorted_objectives = sorted(objective_weights.keys())
    filename = hashlib.md5(
        "_".join(sorted_objectives).encode()).hexdigest() + ".json"
    filepath = os.path.join(save_directory, filename)

    if os.path.exists(filepath):
        # Load dynamic ranges from file
        with open(filepath, 'r') as f:
            dynamic_ranges = json.load(f)
        print(f"Loaded dynamic ranges from {filepath}")
    else:
        raise FileNotFoundError(
            f"No pre-calculated dynamic ranges found for objectives: {sorted_objectives} in {save_directory}.")

    # Initialize the parameter space based on the account type (moved inside optimize)
    param_space = initialize_param_space(accountType)

    # Define dimensions for Bayesian optimization
    dimensions = []
    for key, param_settings in param_space.items():
        # Unpack the first three elements
        param_type, low, high = param_settings[:3]
        # Get step if available, else None
        step = param_settings[3] if len(param_settings) == 4 else None

        if param_type == "bool":
            dimensions.append(Categorical([True, False], name=key))
        elif param_type == "int":
            if step is not None:  # Check if step is provided
                dimensions.append(Categorical(
                    range(low, high + 1, step), name=key))
            else:
                dimensions.append(Integer(low, high, name=key))
        elif param_type == "float":
            if step is not None:
                dimensions.append(Categorical(
                    np.arange(low, high + step, step), name=key))
            else:
                dimensions.append(Real(low, high, name=key))

    # Look up static parameters from account_details
    initial_account_value = account_details[accountType]['initial_account_value']
    min_account_threshold = account_details[accountType]['min_account_threshold']
    payout_profit_target = account_details[accountType]['payout_profit_target']
    min_trading_days = account_details[accountType]['min_trading_days']
    flip_value = account_details[accountType]['flip_value']
    tick_value = account_details[accountType]['tick_value']
    payout_amount = account_details[accountType]['payout_amount']

    @use_named_args(dimensions=dimensions)
    def objective_function(params):
        """
        Objective function to be optimized. It calculates various metrics for a given set of parameters.

        Args:
            params: A dictionary of parameters to be evaluated.

        Returns:
            The combined score (a single scalar value) after normalization and weighting.
        """

        # Calculate trade_outcomes based on the optimized parameters and tick_value
        trade_outcomes = calculate_trade_outcomes(
            params['q1'], params['q2'], params['q3'], tick_value
        )

        # Calculate daily_profit_target and daily_drawdown_limit
        full_win = trade_outcomes[0]
        full_loss = trade_outcomes[-1]
        daily_profit_target = params['win_multiplier'] * full_win
        daily_drawdown_limit = params['loss_multiplier'] * -full_loss

        # Extract max_trades and flip_on_payout_met from params
        max_trades = params['max_trades']
        flip_on_payout_met = params['flip_on_payout_met']

        # Run multiple evaluations with the same parameters to get an average
        # Declare and reset lists for each optimization run
        final_values = []
        total_trading_days_values = []
        trades_per_day_values = []
        total_trading_days_per_period_values = []
        actual_trading_days_per_period_values = []
        avg_trades_per_trading_day_values = []
        total_amount_paid_out_values = []
        max_drawdown_values = []

        # Inner loop for Monte Carlo simulations
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
                delayed(simulateEvaluation)(
                    trade_outcomes,
                    daily_profit_target,
                    payout_profit_target,
                    min_trading_days,
                    initial_account_value,
                    min_account_threshold,
                    max_trades,
                    daily_drawdown_limit,
                    flip_on_payout_met,
                    flip_value,
                    payouts_to_achieve,
                    payout_amount
                )
                for _ in range(num_simulations)
            )

        # Extract objective values from the results
        for (
            final_account_value,
            total_trading_days_all_periods,
            result_avg_trades_per_day,  # Renamed
            result_avg_total_trading_days,  # Renamed
            result_avg_actual_trading_days,  # Renamed
            result_avg_avg_trades_per_trading_day,  # Renamed
            total_amount_paid_out,
            max_drawdown,
            _  # Ignore the DataFrame for now
        ) in results:
            final_values.append(final_account_value)
            total_trading_days_values.append(total_trading_days_all_periods)
            trades_per_day_values.append(
                result_avg_trades_per_day)  # Use renamed variable
            total_trading_days_per_period_values.append(
                result_avg_total_trading_days)  # Use renamed variable
            actual_trading_days_per_period_values.append(
                result_avg_actual_trading_days)  # Use renamed variable
            avg_trades_per_trading_day_values.append(
                result_avg_avg_trades_per_trading_day)  # Use renamed variable
            total_amount_paid_out_values.append(total_amount_paid_out)
            max_drawdown_values.append(max_drawdown)

        # Calculate averages
        avg_final_value = np.mean(final_values)
        avg_total_trading_days_all_periods = np.mean(total_trading_days_values)
        avg_trades_per_day = np.mean(trades_per_day_values)
        avg_total_trading_days = np.mean(total_trading_days_per_period_values)
        avg_actual_trading_days = np.mean(
            actual_trading_days_per_period_values)
        avg_avg_trades_per_trading_day = np.mean(
            avg_trades_per_trading_day_values)
        avg_total_amount_paid_out = np.mean(total_amount_paid_out_values)
        avg_max_drawdown = np.mean(max_drawdown_values)

        # Normalize and standardize the calculated average objective values
        normalized_objectives = []
        objective_values = [avg_final_value, avg_total_trading_days_all_periods,
                            avg_trades_per_day, avg_total_trading_days,
                            avg_actual_trading_days, avg_avg_trades_per_trading_day,
                            avg_total_amount_paid_out, avg_max_drawdown]

        for i, obj_name in enumerate(objective_weights):
            value = objective_values[i]  # Use the calculated average value
            min_val, max_val, mean_value, std_value = dynamic_ranges[obj_name]

            # Normalize to 0-1 range using dynamic_ranges (avoid division by zero)
            if max_val - min_val != 0:
                normalized_value = (value - min_val) / (max_val - min_val)
            else:
                normalized_value = 0  # Or some other appropriate handling for zero range

            # Standardize
            normalized_value = (normalized_value - mean_value) / std_value

            normalized_objectives.append(normalized_value)

        # Calculate the combined score based on weights and maximization/minimization settings
        combined_score = 0
        for i, obj_name in enumerate(objective_weights):
            value = normalized_objectives[i]
            weight = objective_weights[obj_name]["weight"]
            maximize = objective_weights[obj_name]["maximize"]
            combined_score += weight * (value if maximize else -value)

        # Return the combined score (for the optimizer)
        return combined_score

    # Create a closure to capture accountType
    def constraint_func_with_account_type(x):
        return constraint_func(x, accountType)

    # Create the optimizer object
    opt = Optimizer(
        dimensions=dimensions,
        acq_func=acq_func,
        n_initial_points=n_random_starts,
        space_constraint=constraint_func_with_account_type
    )

    # Run the optimization loop with outer progress bar
    combined_scores = []
    for _ in tqdm(range(n_calls), desc="Optimization Progress"):
        next_params = opt.ask()
        combined_score = objective_function(
            params={dim.name: val for dim, val in zip(opt.space.dimensions, next_params)})

        # Pass the combined score to opt.tell
        opt.tell(next_params, combined_score)
        combined_scores.append(combined_score)  # Store the combined score

    # Get top N results from the optimizer (using combined_scores)
    top_results = []
    for i, (point, _) in enumerate(zip(opt.Xi, opt.yi)):
        params = {dim.name: x for dim, x in zip(opt.space.dimensions, point)}

        # Denormalize the objective values for reporting
        denormalized_objectives = []
        for j, obj_name in enumerate(objective_weights):
            # Get the combined score for this evaluation
            normalized_value = opt.yi[i]
            min_val, max_val, mean_value, std_value = dynamic_ranges[obj_name]

            # Reverse the standardization
            denormalized_value = normalized_value * std_value + mean_value

            # Reverse the 0-1 normalization (avoid multiplication by zero)
            if max_val - min_val != 0:
                denormalized_value = denormalized_value * \
                    (max_val - min_val) + min_val
            else:
                denormalized_value = min_val  # Or some other appropriate handling for zero range

            denormalized_objectives.append(denormalized_value)

        # Recalculate trade_outcomes using the current params and tick_value
        trade_outcomes = calculate_trade_outcomes(
            params['q1'], params['q2'], params['q3'], tick_value
        )

        # Include all objectives and trade_outcomes in top_results
        top_results.append(
            (params, *denormalized_objectives,
             trade_outcomes, combined_scores[i])
        )

    # Sort by combined_score (descending order)
    top_results.sort(key=lambda x: x[-1], reverse=True)

    # Filter based on success probability
    filtered_results = [
        result for result in top_results
        if sum(simulateEvaluation(*result[0:12])[2] for _ in range(num_simulations)) / num_simulations >= success_probability_threshold
    ]

    # Return top N filtered results
    return filtered_results[:top_n_results], param_space


def run_optimization(
    account_type: str,
    payouts_to_achieve: int,
    success_probability_threshold: float = 0.8,
    objective_weights: Dict[str,
                            Dict[str, Union[float, bool, int, int]]] = None,
    num_simulations: int = 100,
    acq_func: str = "EI",
    n_calls: int = 100,
    n_random_starts: int = 10,
    n_jobs: int = -1,
    top_n_results: int = 5,
):
    """
    Initiates the optimization process and displays the results.

    Args:
        account_type: The type of account ("Rally" or "Grand Prix").
        payouts_to_achieve: The target number of successful payouts.success_probability_threshold: The minimum required probability of success (achieving the target number of payouts) for a parameter set to be considered in the final results (default: 0.8).
        objective_weights: A dictionary of dictionaries specifying weights and settings for each objective.
        num_simulations: The number of Monte Carlo simulations (evaluations) to run for each parameter set (default: 100).
        acq_func: The acquisition function used for Bayesian optimization (default: "EI").
        n_calls: The maximum number of iterations for the Bayesian optimization process (default: 100).
        n_random_starts: The number of random initialization points for Bayesian optimization (default: 10).
        n_jobs: The number of parallel jobs to run for Bayesian optimization. -1 means using all available cores (default: -1).
        top_n_results: The number of top-performing parameter sets to return (default: 5).

    """

    # Call the optimize function
    top_results, param_space = optimize(
        account_type,
        payouts_to_achieve,
        success_probability_threshold,
        objective_weights,
        acq_func,
        n_calls,
        n_random_starts,
        n_jobs,
        num_simulations,
        top_n_results
    )

    # Display the results
    simple_display_results(top_results, objective_weights,
                           param_space=param_space,
                           num_simulations=num_simulations,
                           accountType=account_type,
                           payouts_to_achieve=payouts_to_achieve,
                           acq_func=acq_func,
                           n_calls=n_calls,
                           n_random_starts=n_random_starts,
                           n_jobs=n_jobs,
                           top_n_results=top_n_results,
                           success_probability_threshold=success_probability_threshold)


def simple_display_results(top_results, objective_weights, **kwargs):
    """
    Prints optimization parameters and top results to the console and saves the top results table to a PDF.

    Args:
        top_results: The list of top results from the optimization.
        **kwargs: All other parameters passed to the optimize function.
    """

    # Extract relevant parameters from kwargs (excluding param_space)
    params_to_display = {k: v for k, v in kwargs.items() if k != 'param_space'}

    # Print optimization parameters in a four-column table
    print("\nOptimization Parameters:")
    table_data = []
    for i in range(0, len(params_to_display), 2):  # Iterate in steps of 2
        row = [list(params_to_display.items())[i]]
        if i + 1 < len(params_to_display):
            row.append(list(params_to_display.items())[i + 1])
        table_data.append(row)

    # Flatten the table_data for tabulate
    flattened_data = [item for sublist in table_data for item in sublist]

    print(tabulate(flattened_data, headers=[
          "Parameter", "Value"] * 2, tablefmt="grid"))

    # Extract headers from objective_weights
    headers = list(objective_weights.keys()) + ["Trade Outcomes"]

    # Prepare table data for top results (transpose and round values)
    table_data = []
    for result in top_results:
        params, *objective_values, trade_outcomes = result
        row = [str(params)] + [round(value, 2)
                               for value in objective_values] + [np.round(trade_outcomes, 2).tolist()]
        table_data.append(row)

    # Transpose the table data
    transposed_data = list(map(list, zip(*table_data)))

    # Arrange transposed data into four columns
    four_column_data = []
    for i in range(0, len(transposed_data), 4):
        row = transposed_data[i:i+4]
        four_column_data.append(row)

    # Flatten the four_column_data for tabulate
    flattened_data = [item for sublist in four_column_data for item in sublist]

    # Print top results to console (transposed and in four columns)
    print("\nTop Results:")
    print(tabulate(flattened_data, headers=headers *
          (len(transposed_data) // 4), tablefmt="grid"))

    # Create 'results' folder if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Get account type from kwargs
    account_type = kwargs.get('accountType', 'Unknown Account')

    # Get current date and time and format it
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

    # Create the header for the PDF
    pdf_header = f"{account_type} - {formatted_datetime}"

    # Create the filename for the PDF
    pdf_filename = pdf_header.replace(" ", "_").replace(
        "/", "_").replace(":", "_") + ".pdf"
    pdf_filepath = os.path.join('results', pdf_filename)

    # Create a figure and axes for the table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Add the header as a title to the figure
    fig.suptitle(pdf_header, fontsize=12)

    # Create the table using matplotlib
    table = ax.table(
        cellText=transposed_data,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )

    # Adjust table properties
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Save the figure to a PDF in the 'results' folder
    with PdfPages(pdf_filepath) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)

# Example call to run_optimization with balanced settings


# Define objective_weights
objective_weights = {
    "avg_final_value": {"weight": 0.7, "maximize": True, "lower_percentile": 5, "upper_percentile": 95},
    "total_trading_days": {"weight": 0.3, "maximize": False, "lower_percentile": 5, "upper_percentile": 95}
}

# run_optimization(
#     account_type="Grand Prix",  # Or "Grand Prix" depending on your test case
#     payouts_to_achieve=3,  # Reduced for faster evaluation
#     success_probability_threshold=0.2,  # Lowered for quicker results
#     objective_weights=objective_weights,
#     num_simulations=50,     # Reduced for faster simulations
#     n_calls=20,             # Reduced for fewer optimization iterations
#     n_random_starts=5       # Reduced for fewer initial random points
# )

# region
# def display_results(top_results):
#     """
#     Displays the top optimization results in an HTML table and generates bar charts
#     to visualize the average final value and probability of success for each parameter set.

#     Args:
#         top_results: A list of tuples, each containing:
#             - params: A dictionary where keys are parameter names and values are the optimized values.
#             - avg_final_value: The average final account value achieved with these parameters.
#             - probability_of_success: The probability of achieving the target number of payouts
#                                       with these parameters.

#     Returns:
#         An HTML object containing the formatted table and charts for display.
#     """

#     # Create table data
#     headers = ["Parameters", "Avg. Final Value", "Probability of Success"]
#     table_data = [
#         [str(params), f"${avg_final_value:.2f}",
#          f"{probability_of_success:.2%}"]
#         for params, avg_final_value, probability_of_success in top_results
#     ]

#     # Generate table HTML
#     table_html = tabulate(table_data, headers, tablefmt="html")

#     # Generate charts
#     # Example: bar chart for avg_final_value
#     param_names = [str(params) for params, _, _ in top_results]
#     avg_final_values = [avg_final_value for _,
#                         avg_final_value, _ in top_results]
#     probabilities = [probability_of_success for _,
#                      _, probability_of_success in top_results]

#     plt.figure(figsize=(10, 6))
#     plt.bar(param_names, avg_final_values, color='skyblue')
#     plt.xlabel('Parameter Set')
#     plt.ylabel('Average Final Value ($)')
#     plt.title('Top Parameter Sets by Average Final Value')
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     # Save the chart to a temporary file
#     plt.savefig('avg_final_value_chart.png')
#     plt.close()

#     # Generate chart HTML for avg_final_value
#     chart1_html = f'<img src="avg_final_value_chart.png" alt="Avg Final Value Chart">'

#     # Example: bar chart for probabilities
#     plt.figure(figsize=(10, 6))
#     plt.bar(param_names, probabilities, color='lightgreen')
#     plt.xlabel('Parameter Set')
#     plt.ylabel('Probability of Success')
#     plt.title('Top Parameter Sets by Probability of Success')
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     # Save the chart to a temporary file
#     plt.savefig('probability_chart.png')
#     plt.close()

#     # Generate chart HTML for probabilities
#     chart2_html = f'<img src="probability_chart.png" alt="Probability Chart">'

#     # Combine table and chart HTML
#     html_content = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#     <title>Optimization Results</title>
#     </head>
#     <body>
#     <h1>Optimization Results</h1>
#     {table_html}
#     <h2>Average Final Value</h2>
#     {chart1_html}
#     <h2>Probability of Success</h2>
#     {chart2_html}
#     </body>
#     </html>
#     """

#     # Display the HTML content
#     return HTML(html_content)

# def run_optimization_with_weights(
#     param_space,
#     num_simulations,
#     payouts_to_achieve,
#     initial_account_value,
#     min_account_threshold,
#     success_probability_threshold,
#     weight_combinations,
#     acq_func="EI",
#     n_calls=100,
#     n_random_starts=10,
#     n_jobs=-1,
#     top_n_results=5,
#     maximize_value=True,
# ):
#     """
#     Runs the optimization process with multiple weight combinations for the objective function
#     and displays the results for each combination.

#     Args:
#         # ... (Same arguments as in the `optimize` function)
#         weight_combinations: A list of tuples, where each tuple represents a combination
#                              of weights for the average final value and the probability
#                              of success objectives. Each tuple should have the format:
#                              (value_weight, probability_weight), where both weights are
#                              between 0 and 1, and their sum is 1.

#     """

#     for value_weight, probability_weight in weight_combinations:
#         print(
#             f"\n--- Optimization with value_weight={value_weight}, probability_weight={probability_weight} ---\n")

#         top_results = optimize(
#             param_space,
#             num_simulations,
#             payouts_to_achieve,
#             initial_account_value,
#             min_account_threshold,
#             success_probability_threshold,
#             acq_func,
#             n_calls,
#             n_random_starts,
#             n_jobs,
#             top_n_results,
#             value_weight,
#             probability_weight,
#             maximize_value,
#         )

#         display_results(top_results)
# endregion
