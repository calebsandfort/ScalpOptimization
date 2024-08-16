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
import tqdm

import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate

events_df = pd.read_csv("events.csv")
events = events_df.values

probabilities = events[:, 0]

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
) -> Tuple[float, float, bool, int, int, float, pd.DataFrame]:
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

    return account_value, avg_trades_per_day, payout_achieved, total_trading_days, actual_trading_days, avg_trades_per_trading_day, df


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
) -> Tuple[float, int, bool, float, float, float, float, float, pd.DataFrame]:
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
        - constraint: A custom constraint function ensuring q1 + q2 + q3 <= max_contracts

    Raises:
        ValueError: If `accountType` is not "Rally" or "Grand Prix".
    """

    if accountType not in ["Rally", "Grand Prix"]:
        raise ValueError(
            "Invalid accountType. Must be 'Rally' or 'Grand Prix'")

    max_contracts = account_details[accountType]['max_contracts']

    # Define dimensions with the constraint
    dimensions = [
        Integer(0, max_contracts, name='q1'),
        Integer(0, max_contracts, name='q2'),
        Integer(0, max_contracts, name='q3'),
        # Discretized using np.arange
        Categorical(np.arange(1, 5.5, 0.5), name='win_multiplier'),
        # Discretized using np.arange
        Categorical(np.arange(1, 3.5, 0.5), name='loss_multiplier'),
        Integer(1, 15, name='max_trades'),
        Categorical([True, False], name='flip_on_payout_met'),
    ]

    # Create a custom constraint function
    def constraint_func(x):
        q1, q2, q3, *_ = x
        return q1 + q2 + q3 <= max_contracts

    # Add the constraint to the dimensions
    dimensions.append(Constraint(constraint_func, name='constraint'))

    # Convert dimensions to param_space format (including the constraint)
    param_space = {dim.name: ("int" if isinstance(dim, Integer) else "float" if isinstance(
        dim, Real) else "bool", dim.low, dim.high, dim.step if dim.step is not None else None) for dim in dimensions}

    return param_space


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

    Args:accountType: Account type to look up static parameters.
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

    all_final_values = []
    all_total_trading_days = []
    avg_trades_per_day_values = []
    total_trading_days_per_period_values = []
    actual_trading_days_per_period_values = []
    avg_avg_trades_per_trading_day_values = []
    total_amount_paid_out_values = []
    final_values = []  # Declare final_values in the outer scope

    # Initialize the parameter space based on the account type (moved inside optimize)
    param_space = initialize_param_space(accountType)

    # Define dimensions for Bayesian optimization
    dimensions = []
    for key, (param_type, low, high, step) in param_space.items():
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

    # Objective function to be optimized
    @ use_named_args(dimensions=dimensions)
    def objective_function(**params):
        """
        Objective function to be optimized. It calculates various metrics for a given set of parameters.

        Args:
            **params: A dictionary of parameters to be evaluated.

        Returns:
            A tuple containing the calculated metrics:
                - avg_final_value
                - avg_total_trading_days_all_periods
                - avg_trades_per_day
                - avg_total_trading_days
                - avg_actual_trading_days
                - avg_avg_trades_per_trading_day
                - avg_total_amount_paid_out
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
        nonlocal final_values
        final_values = []
        total_trading_days_values = []
        trades_per_day_values = []
        total_trading_days_per_period_values = []
        actual_trading_days_per_period_values = []
        avg_trades_per_trading_day_values = []
        total_amount_paid_out_values = []

        # Inner loop for Monte Carlo simulations
        for _ in tqdm(range(num_simulations), desc="Running simulations", leave=False):
            # Simulate the evaluation and collect results
            (
                final_account_value,
                total_payouts,
                evaluation_succeeded,
                avg_trades_per_day,
                avg_total_trading_days,
                avg_actual_trading_days,
                avg_avg_trades_per_trading_day,
                total_amount_paid_out,
                total_trading_days_all_periods,
                _  # Ignore the DataFrame for now
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

            # Only consider successful evaluations
            if evaluation_succeeded:
                final_values.append(final_account_value)
                total_trading_days_values.append(
                    total_trading_days_all_periods)
                trades_per_day_values.append(avg_trades_per_day)
                total_trading_days_per_period_values.append(
                    avg_total_trading_days)
                actual_trading_days_per_period_values.append(
                    avg_actual_trading_days)
                avg_trades_per_trading_day_values.append(
                    avg_avg_trades_per_trading_day)
                total_amount_paid_out_values.append(total_amount_paid_out)

        # Calculate averages only if there were successful evaluations
        if final_values:
            avg_final_value = sum(final_values) / len(final_values)
            avg_total_trading_days_all_periods = sum(
                total_trading_days_values) / len(total_trading_days_values)
            avg_trades_per_day = sum(
                trades_per_day_values) / len(trades_per_day_values)
            avg_total_trading_days = sum(
                total_trading_days_per_period_values) / len(total_trading_days_per_period_values)
            avg_actual_trading_days = sum(
                actual_trading_days_per_period_values) / len(actual_trading_days_per_period_values)
            avg_avg_trades_per_trading_day = sum(
                avg_trades_per_trading_day_values) / len(avg_avg_trades_per_trading_day_values)
            avg_total_amount_paid_out = sum(
                total_amount_paid_out_values) / len(total_amount_paid_out_values)
        else:
            # Penalize parameter sets that didn't succeed in any simulations
            avg_final_value = - \
                float('inf') if objective_weights["avg_final_value"]["maximize"] else float(
                    'inf')
            avg_total_trading_days_all_periods = float(
                'inf') if objective_weights["total_trading_days"]["maximize"] else -float('inf')
            avg_trades_per_day = float('inf') if objective_weights.get(
                "avg_trades_per_day", {}).get("maximize", False) else -float('inf')
            avg_total_trading_days = float('inf') if objective_weights.get(
                "avg_total_trading_days", {}).get("maximize", False) else -float('inf')
            avg_actual_trading_days = float('inf') if objective_weights.get(
                "avg_actual_trading_days", {}).get("maximize", False) else -float('inf')
            avg_avg_trades_per_trading_day = float('inf') if objective_weights.get(
                "avg_avg_trades_per_trading_day", {}).get("maximize", False) else -float('inf')
            avg_total_amount_paid_out = -float('inf') if objective_weights.get(
                "avg_total_amount_paid_out", {}).get("maximize", False) else float('inf')

        all_final_values.extend(final_values)
        all_total_trading_days.extend(total_trading_days_values)
        avg_trades_per_day_values.extend(trades_per_day_values)
        total_trading_days_per_period_values.extend(
            total_trading_days_per_period_values)
        actual_trading_days_per_period_values.extend(
            actual_trading_days_per_period_values)
        avg_avg_trades_per_trading_day_values.extend(
            avg_avg_trades_per_trading_day_values)
        total_amount_paid_out_values.extend(total_amount_paid_out_values)

        # Return all the objective values
        return avg_final_value, avg_total_trading_days_all_periods, avg_trades_per_day, avg_total_trading_days, avg_actual_trading_days, avg_avg_trades_per_trading_day, avg_total_amount_paid_out

    # Create the optimizer object
    opt = Optimizer(
        dimensions=dimensions,
        acq_func=acq_func,
        n_initial_points=n_random_starts,
    )

    # Run the optimization loop with outer progress bar
    for _ in tqdm(range(n_calls), desc="Optimization Progress"):
        next_params = opt.ask()
        result = objective_function(
            **{dim.name: val for dim, val in zip(opt.space.dimensions, next_params)})
        opt.tell(next_params, result)

    # Winsorize and standardize objective values across all simulations
    objective_values = {
        "avg_final_value": all_final_values,
        "total_trading_days": all_total_trading_days,
        "avg_trades_per_day": avg_trades_per_day_values,
        "avg_total_trading_days": total_trading_days_per_period_values,
        "avg_actual_trading_days": actual_trading_days_per_period_values,
        "avg_avg_trades_per_trading_day": avg_avg_trades_per_trading_day_values,
        "avg_total_amount_paid_out": total_amount_paid_out_values
    }

    normalized_objective_values = {}

    for obj_name, obj_settings in objective_weights.items():
        values = objective_values[obj_name]

        winsorized_values = scipy.stats.mstats.winsorize(values, limits=(
            obj_settings["lower_percentile"]/100, obj_settings["upper_percentile"]/100))
        mean_value = np.mean(winsorized_values)
        std_value = np.std(winsorized_values)

        normalized_values = [(v - mean_value) / std_value for v in values]
        normalized_objective_values[obj_name] = normalized_values

        # Store the mean and standard deviation for each objective
        if obj_name == "avg_final_value":
            mean_final_value, std_final_value = mean_value, std_value
        elif obj_name == "total_trading_days":
            mean_total_trading_days, std_total_trading_days = mean_value, std_value
        elif obj_name == "avg_trades_per_day":
            mean_avg_trades_per_day, std_avg_trades_per_day = mean_value, std_value
        elif obj_name == "avg_total_trading_days":
            mean_avg_total_trading_days, std_avg_total_trading_days = mean_value, std_value
        elif obj_name == "avg_actual_trading_days":
            mean_avg_actual_trading_days, std_avg_actual_trading_days = mean_value, std_value
        elif obj_name == "avg_avg_trades_per_trading_day":
            mean_avg_avg_trades_per_trading_day, std_avg_avg_trades_per_trading_day = mean_value, std_value
        elif obj_name == "avg_total_amount_paid_out":
            mean_avg_total_amount_paid_out, std_avg_total_amount_paid_out = mean_value, std_value

    # Get top N results from the optimizer
    top_results = []
    for i, (point, value) in enumerate(zip(opt.Xi, opt.yi)):
        params = {dim.name: x for dim, x in zip(opt.space.dimensions, point)}

        normalized_objectives = {
            obj_name: normalized_objective_values[obj_name][i] for obj_name in objective_weights}

        # Recalculate trade_outcomes using the current params and tick_value
        trade_outcomes = calculate_trade_outcomes(
            params['q1'], params['q2'], params['q3'], tick_value
        )

        # Include all objectives and trade_outcomes in top_results
        top_results.append(
            (params, *[normalized_objectives[obj_name]
             for obj_name in objective_weights], trade_outcomes)
        )

    # Sort by weighted objective, considering maximization/minimization settings
    top_results = sorted(
        top_results,
        key=lambda x: sum(
            objective_weights[obj_name]["weight"] *
            (x[i+1] if objective_weights[obj_name]["maximize"] else -x[i+1])
            for i, obj_name in enumerate(objective_weights)
        ),
        reverse=True,
    )[:top_n_results]

    # Denormalize values for the top results
    for i, obj_name in enumerate(objective_weights):
        if obj_name == "avg_final_value":
            mean_value, std_value = mean_final_value, std_final_value
        elif obj_name == "total_trading_days":
            mean_value, std_value = mean_total_trading_days, std_total_trading_days
        elif obj_name == "avg_trades_per_day":
            mean_value, std_value = mean_avg_trades_per_day, std_avg_trades_per_day
        elif obj_name == "avg_total_trading_days":
            mean_value, std_value = mean_avg_total_trading_days, std_avg_total_trading_days
        elif obj_name == "avg_actual_trading_days":
            mean_value, std_value = mean_avg_actual_trading_days, std_avg_actual_trading_days
        elif obj_name == "avg_avg_trades_per_trading_day":
            mean_value, std_value = mean_avg_avg_trades_per_trading_day, std_avg_avg_trades_per_trading_day
        elif obj_name == "avg_total_amount_paid_out":
            mean_value, std_value = mean_avg_total_amount_paid_out, std_avg_total_amount_paid_out
        else:
            # Handle other objectives if you add them in the future
            continue

        for result in top_results:
            result[i+1] = result[i+1] * std_value + mean_value

    return top_results, param_space


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

run_optimization(
    account_type="Grand Prix",  # Or "Grand Prix" depending on your test case
    payouts_to_achieve=3,  # Reduced for faster evaluation
    success_probability_threshold=0.2,  # Lowered for quicker results
    objective_weights=objective_weights,
    num_simulations=50,     # Reduced for faster simulations
    n_calls=20,             # Reduced for fewer optimization iterations
    n_random_starts=5       # Reduced for fewer initial random points
)

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
