import pandas as pd
from pyomo.core.base.set import OrderedScalarSet


def get_period_length(period: pd.Timestamp, index: OrderedScalarSet):
    """Gets duration of models period

    Arguments:
    ----------
        period: pd.Timestamp
            The period to get the duration of
        index: OrderedScalarSet
            The index of the model
    Returns:
    --------
        period_length: pd.Timedelta
            The duration of the period
        period_conversion_factor: float
            The conversion factor of the period to hours
    """
    if period == index.last():
        period_length = 0
        period_conversion_factor = 1
    else:
        period_length = index.next(period) - period
        period_conversion_factor = period_length.total_seconds() / 3600
    return period_length, period_conversion_factor


def resolve_piecewise_schedule(
    schedule: dict,
    index: OrderedScalarSet,
    default,
    name: str = "schedule",
) -> dict:
    """Evaluate a piecewise constant schedule on the model index.

    A value applies from its timestamp (inclusive) until the next timestamp
    of the schedule (exclusive). Periods before the first timestamp of the
    schedule get the default. Timestamps of the schedule before the first or
    from the last index timestamp on do not need to be part of the index.

    Arguments:
    ----------
        schedule: dict
            The schedule with timestamps as keys.
        index: OrderedScalarSet
            The index of the model
        default
            The value before the first timestamp of the schedule.
        name: str
            The name of the schedule for error messages.
    Returns:
    --------
        values: dict
            The value that is valid for each period of the index.
    Raises:
    -------
        ValueError
            If a timestamp of the schedule lies strictly between two index
            timestamps (it can not be represented exactly) or the timezones
            of the schedule and the index do not match.
    """
    first, last = index.first(), index.last()
    changes = sorted((pd.Timestamp(t), v) for t, v in schedule.items())
    try:
        for time, _ in changes:
            if first < time < last and time not in index:
                raise ValueError(
                    f"The timestamp {time} of {name} is not part of the model "
                    "index. Add it to the index, e.g. by using "
                    "generate_common_time_series."
                )
    except TypeError as error:
        raise ValueError(
            f"The timestamps of {name} and the model index must either both "
            "be timezone aware or both be naive."
        ) from error
    values = {}
    current = default
    position = 0
    for period in index:
        while position < len(changes) and changes[position][0] <= period:
            current = changes[position][1]
            position += 1
        values[period] = current
    return values
