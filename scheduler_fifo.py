from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, List, Dict, Tuple, Optional, Set, Union
from collections import deque

# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class Event:
    """
    One all-day calendar item waiting in the global FIFO.
    - sid:  human-readable subject id (e.g., "UE4/Cardio/FA")
    - phase: "new" | "rev3" | "rev7" | "rev30" ...
    - target: due date (J0/J+N) when this event *becomes eligible*
    - seq: creation order (not used for scheduling here; nice for logs/debug)
    """
    sid: str
    phase: str
    target: date
    seq: int

@dataclass
class DayPlan:
    """What we actually scheduled for a given day (all-day blocks)."""
    date: date
    items: List[Event]

def _as_weekday_caps(cap: Union[int, Tuple[int,int,int,int,int,int,int]]) -> Tuple[int,int,int,int,int,int,int]:
    """
    Normalize the capacity setting.
    - int -> same capacity Monday..Sunday
    - 7-tuple -> specific capacity per weekday (Mon=0 .. Sun=6)
    """
    if isinstance(cap, int):
        return (cap, cap, cap, cap, cap, cap, cap)
    if isinstance(cap, tuple) and len(cap) == 7:
        return cap
    raise ValueError("cap must be int or 7-tuple (Mon..Sun).")

# -----------------------------
# FIFO scheduler (short-first)
# -----------------------------

class SchedulerFIFO:
    """
    Deterministic, single global FIFO planner (no daily sorting).

    Principles
    ----------
    • Priority is encoded by insertion order:
      older subjects are earlier in the FIFO than newer ones.
    • "Short-first" policy is hard-coded:
      when a NEW is placed on day d, append reviews in the order
      rev3 -> rev7 -> rev30 with targets d+3, d+7, d+30.
    • Skips are *date-based*: skips_after["<sid> (phase)"] = not_before_date.
      While current_day < not_before, the event is ignored (it stays in the FIFO).
      On/after not_before, it pops naturally before later items.
    • Days off: cap=0 (from blackouts or weekday caps) => nothing is popped;
      the FIFO simply waits; carry/cascade emerges naturally.
    """

    def __init__(
        self,
        subjects: Iterable[str],
        offsets: Iterable[int] = (0, 3, 7, 30),
        cap: Union[int, Tuple[int,int,int,int,int,int,int]] = 4,
        start_date: Optional[date] = None,
        blackouts: Optional[Set[date]] = None,
        # keys must be in the UI format: "<sid> (revX)" or "<sid> (new)"
        skips_after: Optional[Dict[str, date]] = None,
    ) -> None:
        # Subjects to introduce, in given order (this order drives priority)
        self.subjects = list(subjects)

        # Offsets: include 0 for NEW; store non-zero in ascending order for short-first
        self.offsets = sorted(set(int(x) for x in offsets))
        if 0 not in self.offsets:
            self.offsets = [0] + self.offsets
        self._review_offsets = sorted([o for o in self.offsets if o != 0])  # e.g., [3,7,30]

        # Capacity and timeline settings
        self.weekday_caps = _as_weekday_caps(cap)
        self.start_date = start_date or date.today()
        self.blackouts = set(blackouts or set())

        # Skip rules: {"UE4/Cardio/FA (rev3)": date(YYYY,MM,DD), ...}
        self.skips_after = dict(skips_after or {})

        # Monotonic counter for Event.seq (debugging / trace only)
        self._seq = 0

    # ---- helpers -------------------------------------------------------------

    def _cap_on(self, d: date) -> int:
        """Capacity for a given day (0 if blackout)."""
        if d in self.blackouts:
            return 0
        return self.weekday_caps[d.weekday()]

    # ---- main ---------------------------------------------------------------

    def plan(self) -> List[DayPlan]:
        """
        Build a full deterministic plan:
        - Scan the global FIFO Q once per day.
        - Pop eligible events up to cap; carry the rest by leaving them in Q.
        - If capacity remains, introduce NEW subjects today and enqueue their reviews.
        - Stop when all subjects introduced and Q is empty.
        """
        d = self.start_date
        planned: Dict[date, List[Event]] = {}
        skip_consumed_once: set[str] = set()  # consume exactly once per skipped eid


        Q = deque()      # future events only (NEW of today is not enqueued)
        subj_idx = 0     # next subject index to introduce

        def append_reviews_for(sid: str, base_day: date) -> None:
            """Short-first: enqueue rev3, rev7, rev30 with the right targets."""
            for off in self._review_offsets:
                self._seq += 1
                Q.append(Event(sid, f"rev{off}", base_day + timedelta(days=off), self._seq))

        while True:
            cap_today = self._cap_on(d)
            today: List[Event] = []

            # --- 1) scan FIFO once and pop eligible events up to capacity ----
            Q_next = deque()
            while Q and len(today) < cap_today:
                ev = Q.popleft()

                # Not due yet? Keep it for later.
                if ev.target > d:
                    Q_next.append(ev)
                    continue

                # Skip hold active? Keep it until not_before passes.
                eid = f"{ev.sid} ({ev.phase})"
                nb = self.skips_after.get(eid)
                if nb is not None and d < nb:
                    # First time we hit this skipped event on a due day: consume a slot
                    if eid not in skip_consumed_once and len(today) < cap_today:
                        self._seq += 1
                        # Synthetic placeholder to show the missed slot today
                        today.append(Event(ev.sid, f"skip {ev.phase}", d, self._seq))
                        skip_consumed_once.add(eid)
                    # Keep the real event in the FIFO for later
                    Q_next.append(ev)
                    continue


                
                # Eligible -> place it today.
                today.append(ev)

                # If we just placed a NEW, enqueue its reviews (short-first).
                if ev.phase == "new":
                    append_reviews_for(ev.sid, d)

            # Carry anything we didn't read yet unchanged.
            while Q:
                Q_next.append(Q.popleft())
            Q = Q_next

            # --- 2) If room remains, introduce NEW subjects today -------------
            while len(today) < cap_today and subj_idx < len(self.subjects):
                sid = self.subjects[subj_idx]
                subj_idx += 1

                self._seq += 1
                new_ev = Event(sid, "new", d, self._seq)

                # If this NEW has a skip "not_before", we can't place it today:
                # put the NEW itself into FIFO; it'll pop later automatically.
                eid = f"{sid} (new)"
                nb = self.skips_after.get(eid)
                if nb is not None and d < nb:
                    Q.append(new_ev)
                else:
                    # Place NEW today and enqueue its reviews.
                    today.append(new_ev)
                    append_reviews_for(sid, d)

            # Record the day
            planned[d] = today

            # --- 3) termination: all subjects introduced AND FIFO empty -------
            if subj_idx >= len(self.subjects) and not Q:
                break

            # move to next day
            d = d + timedelta(days=1)

        # return ordered list
        days = sorted(planned.keys())
        return [DayPlan(dt, planned[dt]) for dt in days]
