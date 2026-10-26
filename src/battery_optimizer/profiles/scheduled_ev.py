"""
Profile of an electric vehicle that follows a known schedule.

A ScheduledEV models one battery whose state is continuous over alternating
periods of driving and connection to a charger. In contrast to the EV class
there is no single charging session: the vehicle can be connected several
times, drives with a known fixed power in between and may have any number of
SoC targets, each of which only applies at its own timestamp.
"""

import datetime
from typing import Annotated, Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from battery_optimizer.profiles.battery import NewBattery

NonNegativeW = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class SoCTarget(BaseModel):
    """
    A state of charge that has to be reached at a point in time.

    The SoC is evaluated at the beginning of the period that starts at
    ``time``, i.e. it is the state the battery has when ``time`` is reached.
    Discharging (driving) in the period starting at ``time`` is therefore not
    affected by the target.
    """

    time: datetime.datetime = Field(
        title="Deadline",
        description="The point in time at which the SoC has to be reached.",
    )
    soc: float = Field(
        ge=0,
        le=1,
        title="Target SoC",
        description=(
            "The minimum SoC (0-1) the battery must have at the deadline."
        ),
    )


def _check_no_duplicate_instants(
    schedule: Any, field_name: str
) -> dict[Any, Any]:
    """
    Reject schedules that contain the same instant more than once.

    Parameters
    ----------
    schedule : Any
        The raw schedule as given by the user.
    field_name : str
        The name of the field for the error message.

    Returns
    -------
    dict[Any, Any]
        The unchanged schedule.

    Raises
    ------
    ValueError
        If the schedule is not a dict or two keys describe the same instant.
    """
    if not isinstance(schedule, dict):
        return schedule
    seen: set[pd.Timestamp] = set()
    for key in schedule:
        instant = pd.Timestamp(key)
        if instant in seen:
            raise ValueError(
                f"{field_name} contains the timestamp {instant} more than "
                "once."
            )
        seen.add(instant)
    return schedule


class ScheduledEV(NewBattery):
    """
    Stores all information about an electric vehicle with a known schedule.

    Power and energy are assumed to be specified in W and Wh respectively.

    All schedules are piecewise constant: a value applies from its timestamp
    (inclusive) until the next timestamp of the same schedule (exclusive).
    Before the first timestamp the vehicle is not connected and does not
    drive. Timestamps must either all be timezone aware or all be naive.

    Driving withdraws ``driving_power * duration`` from the stored energy of
    the battery. It is battery-side energy: efficiencies are not applied and
    it is not supplied to the energy network. Charging and V2G (discharging
    to the energy network, only possible if ``max_discharge_power > 0``) are
    only possible while the vehicle is connected. A vehicle can not drive
    and be connected at the same time.

    Each SoC target is a minimum that has to be reached at its timestamp and
    is not enforced at any other time. The inherited ``start_soc`` is the SoC
    at the first timestamp of the model.
    """

    connected: dict[datetime.datetime, bool] = Field(
        title="Connection to the charger",
        description=(
            "Piecewise constant schedule that states from which timestamp "
            "the vehicle is connected to a charger (True) or not (False). "
            "Charging and V2G are only allowed while connected."
        ),
    )

    driving_power: dict[datetime.datetime, NonNegativeW] = Field(
        default_factory=dict,
        title="Driving power",
        description=(
            "Piecewise constant, non-negative battery-side power in W that is "
            "withdrawn from the battery while driving. Not given means the "
            "vehicle never drives."
        ),
    )

    targets: list[SoCTarget] = Field(
        default_factory=list,
        title="SoC targets",
        description=(
            "SoCs that must be reached at specific timestamps, e.g. 100% at "
            "06:00 every morning. Given as a list of SoCTarget, (time, soc) "
            "tuples or as dict mapping the time to the soc. Targets only "
            "apply at their timestamp so the vehicle can discharge "
            "immediately afterwards."
        ),
    )

    @field_validator("connected", "driving_power", mode="before")
    @classmethod
    def check_unique_schedule_timestamps(
        cls, value: Any, info: Any
    ) -> Any:
        """
        Reject schedules defining the same instant twice.

        Parameters
        ----------
        value : Any
            The raw schedule.
        info : Any
            The validation info containing the field name.

        Returns
        -------
        Any
            The unchanged schedule.
        """
        return _check_no_duplicate_instants(value, info.field_name)

    @field_validator("targets", mode="before")
    @classmethod
    def convert_targets(cls, value: Any) -> Any:
        """
        Convert dicts and tuples to SoCTarget input.

        Parameters
        ----------
        value : Any
            The raw targets.

        Returns
        -------
        Any
            A list that can be validated as list of SoCTarget.
        """
        if isinstance(value, dict):
            value = list(value.items())
        if isinstance(value, (list, tuple)):
            return [
                {"time": item[0], "soc": item[1]}
                if isinstance(item, (tuple, list))
                else item
                for item in value
            ]
        return value

    @model_validator(mode="after")
    def check_schedule(self) -> "ScheduledEV":
        """
        Validate that the schedules and targets are consistent.

        Returns
        -------
        ScheduledEV
            The validated instance.

        Raises
        ------
        ValueError
            If the connection schedule is empty, timestamps mix timezone
            aware and naive values, a deadline is defined twice, a target is
            outside min_soc and max_soc or the vehicle drives while it is
            connected.
        """
        if not self.connected:
            raise ValueError("connected must contain at least one timestamp.")

        raw_times = [
            *self.connected,
            *self.driving_power,
            *(t.time for t in self.targets),
        ]
        if len({pd.Timestamp(t).tzinfo is None for t in raw_times}) > 1:
            raise ValueError(
                "All timestamps must either be timezone aware or naive."
            )

        target_times = [pd.Timestamp(t.time) for t in self.targets]
        if len(set(target_times)) != len(target_times):
            raise ValueError("Each target timestamp may only be used once.")
        for target in self.targets:
            if not self.min_soc <= target.soc <= self.max_soc:
                raise ValueError(
                    f"Target soc ({target.soc}) at {target.time} is outside "
                    f"the range of min_soc ({self.min_soc}) and max_soc "
                    f"({self.max_soc})."
                )

        # Both schedules are piecewise constant so it is sufficient to check
        # the timestamps where one of them changes.
        connected = {pd.Timestamp(t): v for t, v in self.connected.items()}
        driving = {pd.Timestamp(t): v for t, v in self.driving_power.items()}
        is_connected = False
        power = 0.0
        for time in sorted(set(connected) | set(driving)):
            is_connected = connected.get(time, is_connected)
            power = driving.get(time, power)
            if is_connected and power > 0:
                raise ValueError(
                    f"The vehicle is connected and drives at {time}. "
                    "Driving while connected is not supported."
                )
        return self

    def schedule_timestamps(self) -> list[pd.Timestamp]:
        """
        Get all timestamps at which the schedule changes or a target is due.

        Returns
        -------
        list[pd.Timestamp]
            The sorted unique timestamps of connected, driving_power and
            targets.
        """
        timestamps = {pd.Timestamp(t) for t in self.connected}
        timestamps.update(pd.Timestamp(t) for t in self.driving_power)
        timestamps.update(pd.Timestamp(t.time) for t in self.targets)
        return sorted(timestamps)
