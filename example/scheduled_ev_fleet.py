"""
Fleet of two EVs with a known daily schedule and a shared grid limit.

Each EV drives 08:00-12:00 and 14:00-18:00 with a fixed 2 kW (battery side)
and is connected to a charger in between (12:00-14:00) and overnight. The only
SoC target is 100% at 06:00 the next morning, there are no targets for the
short stop at noon.

The grid can only deliver 1.5 kW to the whole fleet from 18:00 to 06:00.
Overnight the fleet can therefore only charge 18 kWh but needs 32 kWh to be
full again, so the optimizer charges the remainder during the stop at noon.

Run with: python example/scheduled_ev_fleet.py [solver]
(the solver defaults to glpk)
"""

import sys

import pandas as pd

from battery_optimizer import optimize
from battery_optimizer.profiles.profiles import PowerPriceProfile, ProfileStack
from battery_optimizer.profiles.scheduled_ev import ScheduledEV


def day(hour: int) -> pd.Timestamp:
    """Return the given hour of the first day."""
    return pd.Timestamp("2026-10-26", tz="UTC") + pd.Timedelta(hours=hour)


def make_vehicle(name: str) -> ScheduledEV:
    """Create a vehicle with the daily schedule."""
    return ScheduledEV(
        name=name,
        capacity=20000,  # Wh
        max_charge_power=5000,  # W
        start_soc=1,
        # Piecewise constant from the timestamp on, False before the first one
        connected={
            day(6): True,
            day(8): False,  # first trip
            day(12): True,  # stop at noon, no target
            day(14): False,  # second trip
            day(18): True,  # overnight
        },
        # Battery-side withdrawal in W, not delivered to the network
        driving_power={day(8): 2000, day(12): 0, day(14): 2000, day(18): 0},
        # The target applies at 06:00 only, driving may follow directly. More
        # (time, soc) pairs can be added for further days.
        targets=[(day(24 + 6), 1.0)],
    )


def main(solver: str = "glpk"):
    """Optimize the fleet and print the charging power and SoC."""
    # The index needs the schedule timestamps of the vehicles, which is done
    # by optimize() automatically; they all lie on the hourly grid here.
    index = pd.date_range(day(6), day(24 + 6), freq="h")
    night = [not 6 <= t.hour < 18 for t in index]
    grid = PowerPriceProfile(
        index=index,
        # Power in W for the whole fleet, price in ct/kWh
        power=[1500 if n else 10000 for n in night],
        price=[5 if n else 20 for n in night],
        name="grid",
    )
    result = optimize(
        buy_prices=ProfileStack([grid]),
        scheduled_evs=[make_vehicle("ev1"), make_vehicle("ev2")],
        solver=solver,
    )
    ev_power, ev_soc = result[6], result[7]
    print("Charging power in W")
    print(ev_power)
    print("SoC")
    print(ev_soc)
    return ev_power, ev_soc


if __name__ == "__main__":
    main(*sys.argv[1:2])
