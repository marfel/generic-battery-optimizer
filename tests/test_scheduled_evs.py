"""
Tests the ScheduledEV profile and block.

A ScheduledEV has one continuous battery state, drives with a fixed
battery-side power, can only exchange energy with the network while connected
and has SoC targets that only apply at their timestamps.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

try:
    from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
except ImportError:  # older pyomo versions

    class NoFeasibleSolutionError(Exception):
        """Placeholder that is never raised."""

from battery_optimizer import optimize
from battery_optimizer.export import Exporter
from battery_optimizer.helpers.blocks import resolve_piecewise_schedule
from battery_optimizer.helpers.model_setup import generate_common_time_series
from battery_optimizer.model import Model
from battery_optimizer.profile_stack_problem import ProfileStackProblem
from battery_optimizer.profiles.ev import EV
from battery_optimizer.profiles.profiles import PowerPriceProfile, ProfileStack
from battery_optimizer.profiles.scheduled_ev import ScheduledEV
from battery_optimizer.solver import Solver
from tests.helpers import find_solver


def ts(text: str) -> pd.Timestamp:
    """Create a UTC timestamp."""
    return pd.Timestamp(text, tz="UTC")


def hourly(start: str, end: str) -> pd.DatetimeIndex:
    """Create an hourly UTC index."""
    return pd.date_range(start=start, end=end, freq="1h", tz="UTC")


def solve(index, evs, buy=None, sell=None, extra=None):
    """
    Build a model with the given evs, solve it and return the model.

    buy and sell are tuples of (power, price) callables/values or None.
    The solved model is None if the solver found no optimal solution.
    """
    model = Model(list(index))
    buy_power, buy_price = buy if buy is not None else (100000, 10)
    model.add_buy_profile(
        "buy",
        {t: buy_power(t) if callable(buy_power) else buy_power for t in index},
        {t: buy_price(t) if callable(buy_price) else buy_price for t in index},
    )
    if sell is not None:
        model.add_sell_profile(
            "sell", {t: sell[0] for t in index}, {t: sell[1] for t in index}
        )
    for ev in evs:
        model.add_scheduled_ev(ev)
    if extra is not None:
        extra(model)
    model.add_energy_paths()
    model.generate_objective()
    try:
        result = Solver(find_solver()).solve(model.model)
    except NoFeasibleSolutionError:
        # some solver interfaces raise instead of returning None
        return None
    return model if result is not None else None


def soc(model: Model, name: str) -> dict[pd.Timestamp, float]:
    """Get the stored energy in Wh at the end of each period."""
    return {t: v.value for t, v in model.model.scheduled_evs.component(name).soc.items()}


def flow(model: Model, name: str, variable: str) -> dict:
    """Get the energy_sink or energy_source of an ev."""
    block = model.model.scheduled_evs.component(name)
    return {t: v.value for t, v in getattr(block, variable).items()}


def make_ev(**kwargs) -> ScheduledEV:
    """Create a scheduled ev; only overrides are needed."""
    values = {
        "name": "ev",
        "capacity": 10000,
        "max_charge_power": 5000,
        "connected": {ts("2026-10-26T00:00"): True},
    }
    values.update(kwargs)
    return ScheduledEV(**values)


class TestValidation:
    """Validation and defaults of the ScheduledEV profile."""

    def test_defaults(self):
        """No single-session EV fields are required and V2G is off."""
        ev = make_ev()
        assert ev.max_discharge_power == 0
        assert ev.driving_power == {}
        assert ev.targets == []
        assert ev.start_soc == 0

    def test_connected_required(self):
        """The connection schedule must be given and not be empty."""
        with pytest.raises(ValidationError):
            ScheduledEV(capacity=1, max_charge_power=1)
        with pytest.raises(ValidationError, match="at least one"):
            make_ev(connected={})

    def test_targets_formats(self):
        """Targets can be given as models, tuples or dict."""
        t1, t2 = ts("2026-10-26T06:00"), ts("2026-10-27T06:00")
        from_tuples = make_ev(targets=[(t1, 1), (t2, 0.5)])
        from_dict = make_ev(targets={t1: 1, t2: 0.5})
        assert from_tuples.targets == from_dict.targets
        assert [t.soc for t in from_tuples.targets] == [1, 0.5]

    @pytest.mark.parametrize("soc", [-0.1, 1.1])
    def test_target_range(self, soc):
        """Targets must be between 0 and 1."""
        with pytest.raises(ValidationError):
            make_ev(targets=[(ts("2026-10-26T06:00"), soc)])

    def test_target_within_min_max_soc(self):
        """Targets must be inside min_soc and max_soc."""
        with pytest.raises(ValidationError, match="max_soc"):
            make_ev(max_soc=0.8, targets=[(ts("2026-10-26T06:00"), 0.9)])
        with pytest.raises(ValidationError, match="min_soc"):
            make_ev(min_soc=0.2, targets=[(ts("2026-10-26T06:00"), 0.1)])

    def test_duplicate_targets_rejected(self):
        """A deadline may only be defined once, even in other timezones."""
        t = ts("2026-10-26T06:00")
        with pytest.raises(ValidationError, match="only be used once"):
            make_ev(targets=[(t, 1), (t, 0.5)])
        with pytest.raises(ValidationError, match="only be used once"):
            make_ev(
                targets=[(t, 1), (t.tz_convert("Europe/Berlin"), 1)],
            )

    def test_duplicate_schedule_instants_rejected(self):
        """The same instant in different notations is rejected."""
        with pytest.raises(ValidationError, match="more than once"):
            make_ev(
                connected={
                    "2026-10-26T06:00:00+00:00": True,
                    "2026-10-26T07:00:00+01:00": False,
                }
            )

    @pytest.mark.parametrize("power", [-1, float("nan"), float("inf")])
    def test_invalid_driving_power(self, power):
        """Driving power must be a finite non-negative number."""
        with pytest.raises(ValidationError):
            make_ev(driving_power={ts("2026-10-26T08:00"): power})

    def test_missing_values_rejected(self):
        """None is not a valid schedule value."""
        with pytest.raises(ValidationError):
            make_ev(connected={ts("2026-10-26T08:00"): None})
        with pytest.raises(ValidationError):
            make_ev(driving_power={ts("2026-10-26T08:00"): None})

    def test_mixed_timezone_awareness_rejected(self):
        """Naive and aware timestamps must not be mixed."""
        with pytest.raises(ValidationError, match="timezone"):
            make_ev(
                driving_power={pd.Timestamp("2026-10-26T08:00"): 100},
            )

    def test_driving_while_connected_rejected(self):
        """Driving while connected is rejected, also after a transition."""
        with pytest.raises(ValidationError, match="Driving while connected"):
            make_ev(driving_power={ts("2026-10-26T08:00"): 100})
        with pytest.raises(ValidationError, match="Driving while connected"):
            make_ev(
                connected={
                    ts("2026-10-26T00:00"): False,
                    ts("2026-10-26T10:00"): True,
                },
                driving_power={ts("2026-10-26T08:00"): 100},
            )

    def test_driving_while_disconnected_accepted(self):
        """The default before the first connected timestamp is False."""
        ev = make_ev(
            connected={ts("2026-10-26T10:00"): True},
            driving_power={
                ts("2026-10-26T08:00"): 100,
                ts("2026-10-26T10:00"): 0,
            },
        )
        assert ev.schedule_timestamps() == [
            ts("2026-10-26T08:00"),
            ts("2026-10-26T10:00"),
        ]

    def test_start_soc_limits_inherited(self):
        """The battery validation still applies."""
        with pytest.raises(ValidationError, match="start_soc"):
            make_ev(start_soc=0.1, min_soc=0.2)


class TestScheduleResolution:
    """Piecewise constant evaluation of schedules on the index."""

    index = hourly("2026-10-26T00:00", "2026-10-26T05:00")

    def resolve(self, schedule, index=None, default=0):
        import pyomo.environ as pyo

        model = pyo.ConcreteModel()
        model.i = pyo.Set(
            ordered=pyo.Set.SortedOrder,
            initialize=list(self.index if index is None else index),
        )
        return resolve_piecewise_schedule(schedule, model.i, default, "test")

    def test_transitions_are_half_open(self):
        """A value applies from its timestamp until the next one."""
        values = self.resolve(
            {ts("2026-10-26T02:00"): 5, ts("2026-10-26T04:00"): 7}
        )
        assert list(values.values()) == [0, 0, 5, 5, 7, 7]

    def test_events_before_and_after_horizon(self):
        """Earlier events set the initial state, later ones are ignored."""
        values = self.resolve(
            {
                ts("2026-10-25T22:00"): 1,
                ts("2026-10-26T03:00"): 2,
                ts("2026-10-27T00:00"): 3,
            }
        )
        assert list(values.values()) == [1, 1, 1, 2, 2, 2]

    def test_off_index_event_rejected(self):
        """Events between index timestamps are not represented silently."""
        with pytest.raises(ValueError, match="not part of the model index"):
            self.resolve({ts("2026-10-26T02:30"): 1})

    def test_timezone_mismatch_rejected(self):
        """Naive schedules can not be used with an aware index."""
        with pytest.raises(ValueError, match="timezone"):
            self.resolve({pd.Timestamp("2026-10-26T02:00"): 1})


class TestIndexGeneration:
    """The common index contains all schedule transitions and deadlines."""

    ev = make_ev(
        connected={
            ts("2026-10-26T00:00"): True,
            ts("2026-10-26T08:10"): False,
        },
        driving_power={
            ts("2026-10-26T08:10"): 100,
            ts("2026-10-26T10:20"): 0,
        },
        targets=[(ts("2026-10-26T06:15"), 1), (ts("2026-10-29T06:00"), 1)],
    )

    def test_timestamps_added(self):
        """Transitions and deadlines within the range are added."""
        profile = {ts("2026-10-26T00:00"): 1, ts("2026-10-26T12:00"): 2}
        index = generate_common_time_series(
            [profile], scheduled_evs=[self.ev]
        )
        assert index == [
            ts("2026-10-26T00:00"),
            ts("2026-10-26T06:15"),
            ts("2026-10-26T08:10"),
            ts("2026-10-26T10:20"),
            ts("2026-10-26T12:00"),
        ]

    def test_without_scheduled_evs_unchanged(self):
        """Without scheduled evs the index is the same as before."""
        profile = {ts("2026-10-26T00:00"): 1, ts("2026-10-26T12:00"): 2}
        assert generate_common_time_series([profile]) == [
            ts("2026-10-26T00:00"),
            ts("2026-10-26T12:00"),
        ]

    def test_scheduled_ev_not_added_to_empty_index_range(self):
        """Without other timestamps all schedule timestamps are used."""
        assert generate_common_time_series(scheduled_evs=[self.ev]) == (
            self.ev.schedule_timestamps()
        )


class TestEnergyBalance:
    """Driving reduces the stored energy exactly once."""

    index = [
        ts("2026-10-26T00:00"),
        ts("2026-10-26T00:30"),
        ts("2026-10-26T02:00"),
        ts("2026-10-26T03:00"),
    ]

    def ev(self, **kwargs):
        values = dict(
            start_soc=1,
            charge_efficiency=0.5,
            discharge_efficiency=0.5,
            connected={ts("2026-10-26T02:00"): True},
            driving_power={
                ts("2026-10-26T00:00"): 4000,
                ts("2026-10-26T02:00"): 0,
            },
        )
        values.update(kwargs)
        return make_ev(**values)

    def test_driving_irregular_intervals(self):
        """Driving energy is power times the (irregular) period length."""
        model = solve(self.index, [self.ev()], sell=(10000, 100))
        energy = soc(model, "ev")
        assert energy[self.index[0]] == pytest.approx(10000 - 2000)
        assert energy[self.index[1]] == pytest.approx(10000 - 2000 - 6000)
        assert energy[self.index[2]] == pytest.approx(2000)
        assert energy[self.index[3]] == pytest.approx(2000)

    def test_driving_is_not_a_network_flow(self):
        """Driving works without V2G and creates no energy_source flow."""
        model = solve(self.index, [self.ev()], sell=(10000, 100))
        assert model is not None
        assert all(v == 0 for v in flow(model, "ev", "energy_source").values())
        df = Exporter(model).to_df()
        assert (df.to_ev_power()["ev"] == 0).all()

    def test_driving_with_v2g_does_not_apply_discharge_efficiency(self):
        """Network discharge applies the efficiency, driving does not."""
        ev = self.ev(max_discharge_power=10000)
        model = solve(self.index, [ev], sell=(10000, 100))
        energy = soc(model, "ev")
        # Selling 1000 Wh to the network costs 2000 Wh of stored energy while
        # connected from 02:00. Driving before that cost exactly 8000 Wh.
        sold = sum(flow(model, "ev", "energy_source").values())
        assert sold > 0
        assert energy[self.index[3]] == pytest.approx(2000 - 2 * sold)
        assert energy[self.index[1]] == pytest.approx(2000)

    def test_charge_efficiency_after_driving(self):
        """Charging uses the charge efficiency."""
        ev = self.ev(
            targets=[(self.index[3], 0.4)],
        )
        model = solve(self.index, [ev])
        # 2000 Wh are left after driving, charge_efficiency is 0.5
        assert soc(model, "ev")[self.index[2]] == pytest.approx(4000)
        assert sum(flow(model, "ev", "energy_sink").values()) == (
            pytest.approx(4000)
        )

    def test_driving_below_min_soc_is_infeasible(self):
        """The driving can not be skipped, so too little energy fails."""
        assert solve(self.index, [self.ev(start_soc=0.5)]) is None


class TestConnectionGating:
    """Charging and V2G are only possible while connected."""

    connected = {
        ts("2026-10-26T00:00"): False,
        ts("2026-10-26T02:00"): True,
        ts("2026-10-26T05:00"): False,
    }
    index = hourly("2026-10-26T00:00", "2026-10-26T06:00")

    def is_connected(self, time):
        return ts("2026-10-26T02:00") <= time < ts("2026-10-26T05:00")

    def test_charging_only_while_connected(self):
        """Without driving the charging is still zero while disconnected."""
        ev = make_ev(connected=self.connected)
        # The grid pays for consumption so the ev charges whenever it can.
        model = solve(self.index, [ev], buy=(10000, -1))
        sink = flow(model, "ev", "energy_sink")
        assert sum(sink.values()) == pytest.approx(10000)
        assert sum(v for t, v in sink.items() if not self.is_connected(t)) == 0

    def test_v2g_only_while_connected(self):
        """V2G is only possible while connected."""
        ev = make_ev(
            start_soc=1, max_discharge_power=5000, connected=self.connected
        )
        # Grid energy is too expensive to resell, only the ev can sell.
        model = solve(self.index, [ev], buy=(10000, 1000), sell=(10000, 100))
        source = flow(model, "ev", "energy_source")
        assert sum(source.values()) == pytest.approx(10000)
        assert sum(
            v for t, v in source.items() if not self.is_connected(t)
        ) == 0


class TestTargets:
    """Targets only apply at their timestamp."""

    def test_discharge_after_deadline(self):
        """Later driving is possible, the target is not held afterwards."""
        index = hourly("2026-10-26T00:00", "2026-10-26T08:00")
        ev = make_ev(
            capacity=8000,
            start_soc=0.2,
            max_charge_power=4000,
            connected={
                ts("2026-10-26T00:00"): True,
                ts("2026-10-26T05:00"): False,
            },
            driving_power={
                ts("2026-10-26T05:00"): 2000,
                ts("2026-10-26T07:00"): 0,
            },
            targets=[(ts("2026-10-26T04:00"), 1)],
        )
        model = solve(
            index,
            [ev],
            buy=(10000, lambda t: 1 if t.hour == 4 else 10),
        )
        energy = soc(model, "ev")
        assert energy[ts("2026-10-26T03:00")] == pytest.approx(8000)
        assert energy[ts("2026-10-26T07:00")] == pytest.approx(4000)
        assert energy[ts("2026-10-26T08:00")] == pytest.approx(4000)

    def test_deadline_is_beginning_of_period(self):
        """Charging in the period starting at the deadline is too late."""
        index = hourly("2026-10-26T00:00", "2026-10-26T06:00")
        ev = make_ev(
            capacity=6400,
            start_soc=0,
            max_charge_power=1600,
            targets=[(ts("2026-10-26T04:00"), 1)],
        )
        model = solve(index, [ev], buy=(10000, lambda t: 1 if t.hour == 4 else 5))
        sink = flow(model, "ev", "energy_sink")
        assert sink[ts("2026-10-26T04:00")] == pytest.approx(0)
        assert soc(model, "ev")[ts("2026-10-26T03:00")] == pytest.approx(6400)
        # One period less than required is infeasible
        assert (
            solve(index, [ev.model_copy(update={"max_charge_power": 1500})])
            is None
        )

    def test_final_horizon_deadline(self):
        """A target at the last timestamp refers to the last real period."""
        index = hourly("2026-10-26T00:00", "2026-10-26T04:00")
        ev = make_ev(
            capacity=4000,
            max_charge_power=1000,
            targets=[(index[-1], 1)],
        )
        model = solve(index, [ev])
        assert soc(model, "ev")[index[-2]] == pytest.approx(4000)
        slow = ev.model_copy(update={"max_charge_power": 900})
        assert solve(index, [slow]) is None

    def test_first_index_deadline(self):
        """A target at the first timestamp refers to start_soc."""
        index = hourly("2026-10-26T00:00", "2026-10-26T02:00")
        ok = make_ev(start_soc=0.5, targets=[(index[0], 0.5)])
        assert solve(index, [ok]) is not None
        with pytest.raises(ValueError, match="start_soc"):
            solve(index, [make_ev(targets=[(index[0], 0.5)])])

    def test_multiple_daily_targets(self):
        """Each morning target is reached and driving follows at once."""
        index = hourly("2026-10-26T20:00", "2026-10-28T10:00")
        ev = make_ev(
            capacity=20000,
            start_soc=1,
            max_charge_power=5000,
            connected={
                ts("2026-10-26T20:00"): True,
                ts("2026-10-27T07:00"): False,
                ts("2026-10-27T17:00"): True,
                ts("2026-10-28T07:00"): False,
            },
            driving_power={
                ts("2026-10-27T07:00"): 1500,
                ts("2026-10-27T17:00"): 0,
                ts("2026-10-28T07:00"): 1500,
            },
            targets=[
                (ts("2026-10-27T07:00"), 1),
                (ts("2026-10-28T07:00"), 1),
            ],
        )
        model = solve(index, [ev])
        energy = soc(model, "ev")
        assert energy[ts("2026-10-27T06:00")] == pytest.approx(20000)
        assert energy[ts("2026-10-28T06:00")] == pytest.approx(20000)
        assert energy[ts("2026-10-27T16:00")] == pytest.approx(5000)
        assert energy[ts("2026-10-28T10:00")] == pytest.approx(20000 - 4500)

    def test_targets_outside_horizon_rejected(self):
        """Targets that can not be reached by the model are rejected."""
        index = hourly("2026-10-26T00:00", "2026-10-26T04:00")
        for target in ["2026-10-25T23:00", "2026-10-26T05:00"]:
            with pytest.raises(ValueError, match="outside of the model"):
                solve(index, [make_ev(targets=[(ts(target), 0.5)])])

    def test_off_index_event_rejected(self):
        """Deadlines and transitions must be part of the index."""
        index = hourly("2026-10-26T00:00", "2026-10-26T04:00")
        with pytest.raises(ValueError, match="not part of the model index"):
            solve(index, [make_ev(targets=[(ts("2026-10-26T02:30"), 0.5)])])
        with pytest.raises(ValueError, match="not part of the model index"):
            solve(
                index,
                [make_ev(connected={ts("2026-10-26T01:30"): False})],
            )

    def test_timezone_mismatch_rejected(self):
        """A naive model index does not work with aware schedules."""
        index = pd.date_range("2026-10-26T00:00", "2026-10-26T04:00", freq="1h")
        with pytest.raises(ValueError, match="timezone"):
            solve(index, [make_ev()])
        with pytest.raises(ValueError, match="timezone"):
            solve(index, [make_ev(connected={pd.Timestamp("2026-10-26"): True},
                                  targets=[(ts("2026-10-26T01:00"), 0.1)])])


class TestFleet:
    """Several EVs share a grid limit and need daytime charging."""

    index = hourly("2026-10-26T06:00", "2026-10-27T06:00")

    @staticmethod
    def vehicle(name: str, stop_connected: bool = True) -> ScheduledEV:
        return ScheduledEV(
            name=name,
            capacity=20000,
            max_charge_power=5000,
            start_soc=1,
            connected={
                ts("2026-10-26T06:00"): True,
                ts("2026-10-26T08:00"): False,
                ts("2026-10-26T12:00"): stop_connected,
                ts("2026-10-26T14:00"): False,
                ts("2026-10-26T18:00"): True,
            },
            driving_power={
                ts("2026-10-26T08:00"): 2000,
                ts("2026-10-26T12:00"): 0,
                ts("2026-10-26T14:00"): 2000,
                ts("2026-10-26T18:00"): 0,
            },
            targets=[(ts("2026-10-27T06:00"), 1)],
        )

    def fleet(self, stop_connected=True):
        # The grid delivers 1500 W in total during the night, daytime prices
        # are higher so that the optimizer only charges during the day if it
        # has to.
        return solve(
            self.index,
            [self.vehicle("ev1", stop_connected), self.vehicle("ev2", stop_connected)],
            buy=(
                lambda t: 10000 if t.hour in range(6, 18) else 1500,
                lambda t: 20 if t.hour in range(6, 18) else 5,
            ),
        )

    def test_daytime_charging_is_needed(self):
        """Night energy (18000 Wh) is not enough to refill 32000 Wh."""
        model = self.fleet()
        sink = {
            name: flow(model, name, "energy_sink") for name in ("ev1", "ev2")
        }
        total = {
            t: sink["ev1"][t] + sink["ev2"][t] for t in self.index
        }
        day = sum(total[t] for t in self.index if ts("2026-10-26T12:00") <= t < ts("2026-10-26T14:00"))
        night = sum(total[t] for t in self.index if t >= ts("2026-10-26T18:00"))
        assert day == pytest.approx(14000)
        assert night == pytest.approx(18000)
        for t in self.index:
            if t >= ts("2026-10-26T18:00"):
                assert total[t] <= 1500 + 1e-6
        for name in ("ev1", "ev2"):
            energy = soc(model, name)
            assert energy[ts("2026-10-27T05:00")] == pytest.approx(20000)
            # No charging outside the stop and the night
            assert sum(
                sink[name][t]
                for t in self.index
                if ts("2026-10-26T06:00") <= t < ts("2026-10-26T12:00")
            ) == 0

    def test_without_daytime_connection_infeasible(self):
        """Without the daytime stop the targets can not be reached."""
        assert self.fleet(stop_connected=False) is None


class TestIntegration:
    """Model, ProfileStackProblem, optimize and exports."""

    def test_add_ev_dispatches(self):
        """add_ev accepts ScheduledEV and EV side by side."""
        index = hourly("2026-10-26T00:00", "2026-10-26T04:00")
        model = Model(list(index))
        model.add_buy_profile(
            "buy", {t: 1000 for t in index}, {t: 1 for t in index}
        )
        block = model.add_ev(make_ev())
        assert block is model.model.scheduled_evs.component("ev")
        model.add_ev(
            EV(
                name="legacy",
                capacity=1000,
                max_charge_power=1000,
                end_soc=1,
                charge_start_time=index[0],
                charge_end_time=index[2],
            )
        )
        assert "legacy" in model.model.evs.component_map()
        model.add_energy_paths()
        model.generate_objective()
        assert Solver(find_solver()).solve(model.model) is not None

    def test_exports(self):
        """Power and SoC of scheduled evs are exported."""
        index = hourly("2026-10-26T00:00", "2026-10-26T04:00")
        ev = make_ev(
            capacity=4000,
            max_charge_power=2000,
            driving_power={ts("2026-10-26T02:00"): 1000},
            connected={
                ts("2026-10-26T00:00"): True,
                ts("2026-10-26T02:00"): False,
            },
            targets=[(ts("2026-10-26T02:00"), 0.5)],
        )
        model = solve(index, [ev])
        exporter = Exporter(model)
        df = exporter.to_df()
        assert list(df.to_ev_power().columns) == ["ev"]
        assert list(df.to_ev_soc().columns) == ["ev"]
        assert df.to_ev_power()["ev"].sum() == pytest.approx(2000)
        assert df.to_ev_soc()["ev"].loc[ts("2026-10-26T01:00")] == (
            pytest.approx(0.5)
        )
        assert df.to_ev_soc()["ev"].loc[ts("2026-10-26T03:00")] == (
            pytest.approx(0.0)
        )
        assert "ev" in exporter.to_dict()

    def test_optimize_wrapper_adds_schedule_timestamps(self):
        """The wrapper adds transitions and deadlines to the index."""
        index = hourly("2026-10-26T00:00", "2026-10-26T04:00")
        buy = ProfileStack(
            [
                PowerPriceProfile(
                    index=index,
                    price=[1] * len(index),
                    power=[10000] * len(index),
                    name="grid",
                )
            ]
        )
        ev = make_ev(
            capacity=4000,
            connected={
                ts("2026-10-26T00:00"): False,
                ts("2026-10-26T01:30"): True,
            },
            targets=[(ts("2026-10-26T03:00"), 1)],
        )
        result = optimize(
            buy_prices=buy, scheduled_evs=[ev], solver=find_solver()
        )
        ev_soc = result[7]
        assert ts("2026-10-26T01:30") in ev_soc.index
        assert ev_soc["ev"].loc[ts("2026-10-26T02:00")] == pytest.approx(1)
        # charging is not possible before 01:30
        assert result[6]["ev"].loc[ts("2026-10-26T01:00")] == 0

    def test_profile_stack_problem_rejects_target_beyond_horizon(self):
        """Deadlines after the horizon do not extend it, they are rejected."""
        index = hourly("2026-10-26T00:00", "2026-10-26T04:00")
        buy = ProfileStack(
            [
                PowerPriceProfile(
                    index=index,
                    price=[1] * len(index),
                    power=[10000] * len(index),
                    name="grid",
                )
            ]
        )
        ev = make_ev(targets=[(ts("2026-10-27T06:00"), 1)])
        problem = ProfileStackProblem(buy_prices=buy, scheduled_evs=[ev])
        with pytest.raises(ValueError, match="outside of the model"):
            problem.set_up()
