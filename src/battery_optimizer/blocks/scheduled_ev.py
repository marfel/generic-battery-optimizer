"""
Battery block of an electric vehicle that follows a known schedule.

ScheduledEVBlock extends the basic battery block with a fixed driving
consumption, gating of charging and V2G by the connection to a charger and
SoC targets that only apply at their own timestamp.
"""

import datetime
import pyomo.environ as pyo
from battery_optimizer.blocks.battery import NewBatteryBlock as BatteryBlock
from battery_optimizer.helpers.blocks import (
    get_period_length,
    resolve_piecewise_schedule,
)
from battery_optimizer.profiles.scheduled_ev import ScheduledEV


class ScheduledEVBlock(BatteryBlock):
    """
    Pyomo block representing an electric vehicle with a known schedule.

    Power and energy are assumed to be specified in W and Wh respectively.
    The state of the battery is continuous over the whole horizon. Driving
    withdraws ``driving_power * period length`` from the stored energy of the
    battery (no efficiency applied) and is not exchanged with the energy
    network. Charging and discharging to the network are only possible while
    the vehicle is connected.

    A SoC target is enforced for the energy that is stored at the beginning
    of the period starting at the target timestamp (the SoC at the end of the
    preceding period). Nothing is enforced after the target.

    Parameters
    ----------
    index : pyo.Set
        The set of time indices for the optimization.
    battery : ScheduledEV
        The scheduled electric vehicle profile.

    Raises
    ------
    ValueError
        If a schedule timestamp can not be represented by the index, a target
        is outside the horizon or a target at the first timestamp is higher
        than start_soc.
    """

    def __init__(self, index: pyo.Set, battery: ScheduledEV):
        """
        Initialize the ScheduledEVBlock.

        Resolve the schedules on the index and validate the targets.

        Parameters
        ----------
        index : pyo.Set
            The set of time indices for the optimization.
        battery : ScheduledEV
            The scheduled electric vehicle profile.
        """
        super().__init__(index, battery)
        self.connected = resolve_piecewise_schedule(
            battery.connected, index, False, "connected"
        )
        self.driving_power = resolve_piecewise_schedule(
            battery.driving_power, index, 0, "driving_power"
        )
        self.targets: dict[datetime.datetime, float] = {}
        for target in battery.targets:
            time = target.time
            try:
                outside = time < index.first() or time > index.last()
            except TypeError as error:
                raise ValueError(
                    "The timestamps of the targets and the model index must "
                    "either both be timezone aware or both be naive."
                ) from error
            if outside:
                raise ValueError(
                    f"The target at {time} is outside of the model horizon."
                )
            if time not in index:
                raise ValueError(
                    f"The target timestamp {time} is not part of the model "
                    "index. Add it to the index, e.g. by using "
                    "generate_common_time_series."
                )
            if time == index.first() and target.soc > battery.start_soc:
                raise ValueError(
                    f"The target soc ({target.soc}) at the first timestamp "
                    f"exceeds start_soc ({battery.start_soc})."
                )
            self.targets[time] = target.soc

    def _internal_withdrawal(self, i: datetime.datetime) -> float:
        """
        Get the energy in Wh the vehicle uses for driving in the period of i.

        Parameters
        ----------
        i : datetime.datetime
            The current timestamp.

        Returns
        -------
        float
            The driving energy. The last timestamp has no duration.
        """
        if i == self.index.last():
            return 0
        return self.driving_power[i] * get_period_length(i, self.index)[1]

    def _populate_block(self, block: pyo.Block) -> pyo.Block:
        """
        Build the scheduled EV block.

        Parameters
        ----------
        block : pyo.Block
            The Pyomo block to populate with the variables and constraints.

        Returns
        -------
        pyo.Block
            The populated block.
        """
        block = super()._populate_block(block)

        # No exchange with the network while the vehicle is not connected
        for i in self.index:
            if not self.connected[i]:
                block.energy_sink[i].setub(0)
                block.energy_source[i].setub(0)

        if self.targets:
            block.target_times = pyo.Set(
                initialize=sorted(self.targets), ordered=True
            )

            def target_reached(_, i: datetime.datetime) -> pyo.Constraint:
                """
                Ensure the target SoC is reached when i begins.

                Parameters
                ----------
                _ : pyo.Block
                    The Pyomo block (not used).
                i : datetime.datetime
                    The deadline.

                Returns
                -------
                pyo.Constraint
                    The constraint enforcing the SoC at the deadline.
                """
                target = self.targets[i] * self.battery.capacity
                if i == self.index.first():
                    # start_soc >= target was checked in __init__
                    return pyo.Constraint.Skip
                return block.soc[self.index.prev(i)] >= target

            block.target_soc = pyo.Constraint(
                block.target_times, rule=target_reached
            )
        return block
