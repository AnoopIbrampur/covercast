"""From a cover forecast to an actual shift schedule, and what it costs when the night happens.

The daily cost model in `staffing.py` treats labor as smooth: forecast 10%
high, pay 10% more. Real schedules don't work that way. People come in whole
units, shifts are blocks of 3 to 8 hours, somebody has to be on the floor
whenever the doors are open, and demand arrives in a dinner rush rather than
evenly. This module builds the schedule a manager would actually post and
prices it by replaying the night that really happened.

Pipeline for one store-day:

1. **When guests arrive.** An hourly arrival profile for the store, from the
   times its bookings were for, shrunk toward its genre's profile (and the
   genre's toward everyone's) when it has few bookings. Only bookings for
   days up to the cutoff are used.
2. **Staff needed per hour.** Each cover takes `labor_minutes_per_cover` of
   staff time, split evenly over its arrival hour and the next.
3. **The schedule.** An integer program picks how many people start at each
   hour for each shift length. It minimises wages plus the expected cost of
   being short, over demand scenarios drawn from the forecast distribution
   (daily total from the distribution, split into hours multinomially),
   subject to a minimum crew in open hours. A point forecast is the same
   program with one scenario and no intraday noise, which is how a
   spreadsheet schedule gets made.
4. **Replay.** The day's real cover count is split into hours (multinomial on
   the profile, the same draw for every policy), and each schedule is charged
   wages for every hour scheduled plus the short-staffing penalty for every
   staff-hour of need it didn't cover. A perfect-foresight schedule, solved on
   the realized hours, is the floor every policy is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity, kron

from .data import Panel

# Service day runs 09:00 to 03:00; hours 24-26 are after midnight. Bookings
# outside that window (a handful at 04:00-08:00) are dropped.
DAY_START, DAY_END = 9, 27
HOURS = np.arange(DAY_START, DAY_END)


@dataclass(frozen=True)
class ScheduleParams:
    wage: float = 18.0
    labor_minutes_per_cover: float = 15.0  # 4 covers per labor hour, as in StaffingCosts
    under_multiplier: float = 2.0
    min_crew: int = 1
    shift_lengths: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    n_scenarios: int = 200
    # Bookings worth of evidence at which a store's own profile counts as much as its genre's.
    profile_shrinkage: float = 50.0
    open_share: float = 0.01  # hours with at least this share of arrivals must be staffed

    @property
    def shifts(self) -> np.ndarray:
        """(n_shifts, n_hours) 0/1 coverage matrix for every start x length that fits the day."""
        rows = []
        for length in self.shift_lengths:
            for start in range(len(HOURS) - length + 1):
                r = np.zeros(len(HOURS))
                r[start : start + length] = 1
                rows.append(r)
        return np.array(rows)


def service_hour(hour: np.ndarray) -> np.ndarray:
    """Clock hour -> index into HOURS, or -1 if outside the service day."""
    h = np.where(hour < DAY_START - 6, hour + 24, hour)
    return np.where((h >= DAY_START) & (h < DAY_END), h - DAY_START, -1)


@dataclass
class Profiles:
    arrival: np.ndarray  # (S, n_hours), rows sum to 1
    n_bookings: np.ndarray  # (S,) bookings the store's own part rests on
    genre: dict[int, np.ndarray] = field(default_factory=dict)


def arrival_profiles(panel: Panel, ci: int, params: ScheduleParams) -> Profiles:
    """Per-store hourly arrival shares, from bookings for days up to the cutoff."""
    H = len(HOURS)
    idx = service_hour(panel.res_visit_hour)
    keep = (panel.res_visit_day <= ci) & (panel.res_booked_day <= ci) & (idx >= 0)
    s, h, w = panel.res_store[keep], idx[keep], panel.res_visitors[keep]

    own = np.zeros((panel.n_stores, H))
    np.add.at(own, (s, h), w)
    n = np.zeros(panel.n_stores)
    np.add.at(n, s, 1.0)

    k = params.profile_shrinkage
    overall = own.sum(axis=0)
    overall = overall / overall.sum() if overall.sum() > 0 else np.full(H, 1 / H)
    genre_prof, genre_n = {}, {}
    for g in np.unique(panel.genre):
        members = panel.genre == g
        tot = own[members].sum(axis=0)
        ng = n[members].sum()
        shares = tot / tot.sum() if tot.sum() > 0 else overall
        genre_prof[g] = (ng * shares + k * overall) / (ng + k)
        genre_n[g] = ng

    own_shares = np.divide(own, own.sum(axis=1, keepdims=True), out=np.zeros_like(own), where=own.sum(1, keepdims=True) > 0)
    prior = np.stack([genre_prof[g] for g in panel.genre])
    wgt = (n / (n + k))[:, None]
    prof = wgt * own_shares + (1 - wgt) * prior
    return Profiles(prof / prof.sum(axis=1, keepdims=True), n, genre_prof)


def staff_need(arrivals: np.ndarray, params: ScheduleParams) -> np.ndarray:
    """Arrivals per hour (..., H) -> staff needed per hour (..., H)."""
    per_cover = params.labor_minutes_per_cover / 60.0
    spill = np.zeros_like(arrivals)
    spill[..., 1:] = arrivals[..., :-1]
    return per_cover * 0.5 * (arrivals + spill)


def open_hours(profile: np.ndarray, params: ScheduleParams) -> np.ndarray:
    """Hours that need a minimum crew: any hour with real arrival share, and the hour after it."""
    o = profile >= params.open_share
    o[1:] |= o[:-1]
    return o


def solve(need: np.ndarray, open_mask: np.ndarray, params: ScheduleParams, A: np.ndarray | None = None) -> np.ndarray:
    """Staff on the floor per hour (H,) for the cheapest schedule; see solve_shifts."""
    A = params.shifts if A is None else A
    return solve_shifts(need, open_mask, params, A) @ A


def solve_shifts(need: np.ndarray, open_mask: np.ndarray, params: ScheduleParams, A: np.ndarray | None = None) -> np.ndarray:
    """Cheapest integer schedule against K equally likely need scenarios. need: (K, H).

    Returns how many people work each shift in `params.shifts` (J,).

        minimise   wage * sum_j len_j x_j  +  under_multiplier * wage * sum_t E[(need_t - cover_t)+]
        subject to cover_t = sum_j A_jt x_j,   cover_t >= min_crew on open hours,   x integer >= 0

    Coverage is a whole number of people, so E[(need_t - c)+] only matters at
    c = 0, 1, 2, ... It is convex and decreasing in c, so it can be written as
    S_t(0) minus a stack of shrinking increments d_t1 >= d_t2 >= ..., each
    unlocked by one more person on the floor (z_tl in [0, 1], sum_l z_tl <= cover_t).
    Because the increments shrink, the LP fills them in order and z comes out
    integral without being declared so. The program's size no longer depends on
    the number of scenarios, so the forecast distribution can be sampled densely.
    """
    A = params.shifts if A is None else A  # (J, H)
    need = np.atleast_2d(need)
    J, H = A.shape
    levels = int(np.ceil(need.max())) + 1 if need.size else 1
    levels = max(levels, params.min_crew)
    c_grid = np.arange(levels + 1)  # 0..L
    S = np.maximum(need[:, :, None] - c_grid[None, None, :], 0).mean(axis=0)  # (H, L+1)
    d = S[:, :-1] - S[:, 1:]  # (H, L), each row non-increasing
    pen = params.under_multiplier * params.wage

    n_z = H * levels
    c = np.concatenate([params.wage * A.sum(axis=1), -pen * d.ravel()])
    # sum_l z_tl - cover_t <= 0
    z_sum = kron(identity(H), np.ones((1, levels)))
    link = hstack([-csr_matrix(A.T), z_sum])
    cons = [LinearConstraint(link, lb=-np.inf, ub=0)]
    if open_mask.any():
        crew = hstack([csr_matrix(A.T)[open_mask], csr_matrix((int(open_mask.sum()), n_z))])
        cons.append(LinearConstraint(crew, lb=params.min_crew, ub=np.inf))

    integrality = np.concatenate([np.ones(J), np.zeros(n_z)])
    ub = np.concatenate([np.full(J, np.inf), np.ones(n_z)])
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(0, ub),
               options={"disp": False, "mip_rel_gap": 1e-6})
    if not res.success:
        raise RuntimeError(f"schedule solve failed: {res.message}")
    return np.round(res.x[:J])


def describe_shifts(x: np.ndarray, params: ScheduleParams) -> list[dict]:
    """[{start, end, people}] in clock hours (end may exceed 24 for after-midnight)."""
    A = params.shifts
    out = []
    for j in np.flatnonzero(x > 0):
        cols = np.flatnonzero(A[j])
        out.append({"start": int(HOURS[cols[0]]), "end": int(HOURS[cols[-1]] + 1), "people": int(x[j])})
    return sorted(out, key=lambda r: (r["start"], r["end"]))


def replay_cost(on_floor: np.ndarray, need: np.ndarray, params: ScheduleParams) -> tuple[float, float, float]:
    """(total $, staff-hours scheduled, staff-hours short) for a schedule against the realized need."""
    hours = float(on_floor.sum())
    short = float(np.maximum(need - on_floor, 0).sum())
    return params.wage * (hours + params.under_multiplier * short), hours, short
