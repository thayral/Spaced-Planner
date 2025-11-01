
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, List, Dict, Tuple, Optional, Set, Union
from collections import deque

@dataclass(frozen=True)
class Event:
    sid: str
    phase: str       # 'new' | 'rev3' | 'rev7' | 'rev30'
    target: date
    seq: int

@dataclass
class DayPlan:
    date: date
    items: List[Event]

def _as_weekday_caps(cap: Union[int, Tuple[int,int,int,int,int,int,int]]) -> Tuple[int,int,int,int,int,int,int]:
    if isinstance(cap, int):
        return (cap, cap, cap, cap, cap, cap, cap)
    if isinstance(cap, tuple) and len(cap) == 7:
        return cap
    raise ValueError("cap must be int or 7-tuple (Mon..Sun).")

class SchedulerFIFO:
    def __init__(
        self,
        subjects: Iterable[str],
        offsets: Iterable[int] = (0,3,7,30),
        cap: Union[int, Tuple[int,int,int,int,int,int,int]] = 4,
        start_date: Optional[date] = None,
        blackouts: Optional[Set[date]] = None,
        skips_after: Optional[Dict[str, date]] = None,
    ) -> None:
        self.subjects = list(subjects)
        self.offsets = sorted(set(int(x) for x in offsets))
        if 0 not in self.offsets:
            self.offsets = [0] + self.offsets
        self.weekday_caps = _as_weekday_caps(cap)
        self.start_date = start_date or date.today()
        self.blackouts = set(blackouts or set())
        self.skips_after = dict(skips_after or {})
        self._review_offsets = sorted([o for o in self.offsets if o != 0])  # short-first
        self._seq = 0

    def _cap_on(self, d: date) -> int:
        if d in self.blackouts:
            return 0
        return self.weekday_caps[d.weekday()]

    def plan(self) -> List[DayPlan]:
        d = self.start_date
        planned: Dict[date, List[Event]] = {}
        from collections import deque
        Q = deque()
        subj_idx = 0

        def append_reviews_for(sid: str, base_day: date):
            for off in self._review_offsets:  # short-first
                self._seq += 1
                Q.append(Event(sid, f"rev{off}", base_day + timedelta(days=off), self._seq))

        while True:
            cap_today = self._cap_on(d)
            today: List[Event] = []

            Q_next = deque()
            while Q and len(today) < cap_today:
                ev = Q.popleft()
                if ev.target > d:
                    Q_next.append(ev)
                    continue
                eid = f"{ev.sid} ({ev.phase})"
                nb = self.skips_after.get(eid)
                if nb is not None and d < nb:
                    Q_next.append(ev)
                    continue
                today.append(ev)
                if ev.phase == "new":
                    append_reviews_for(ev.sid, d)

            while Q:
                Q_next.append(Q.popleft())
            Q = Q_next

            while len(today) < cap_today and subj_idx < len(self.subjects):
                sid = self.subjects[subj_idx]; subj_idx += 1
                self._seq += 1
                new_ev = Event(sid, "new", d, self._seq)
                eid = f"{sid} (new)"
                nb = self.skips_after.get(eid)
                if nb is not None and d < nb:
                    Q.append(new_ev)
                else:
                    today.append(new_ev)
                    append_reviews_for(sid, d)

            planned[d] = today

            if subj_idx >= len(self.subjects) and not Q:
                break

            d = d + timedelta(days=1)

        days = sorted(planned.keys())
        return [DayPlan(dt, planned[dt]) for dt in days]
