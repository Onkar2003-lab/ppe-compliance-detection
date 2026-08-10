"""S6.2 — dwell and debounce: turning per-frame non-compliance into one alert per incident (O6).

The detector answers a question every frame; a site manager needs an answer per *incident*.
The S2 skeleton already made that point the hard way — a monitor that logs whenever a person
is seen without a helmet produced thousands of rows for one worker walking through a yard. This
module is the layer that fixes it, and it is deliberately pure: it takes timestamps and
per-track compliance states in, and gives alerts out, with no detector, tracker or frame
anywhere near it. That is what lets S6.5 evidence the zone/dwell behaviour *completely* by
test, rather than sampling it on footage nobody has labelled.

**Dwell.** A violation fires only after a worker has been in the zone and missing required PPE
for at least ``dwell_seconds`` continuously. The threshold is a real operating dial: zero fires
the moment someone is seen non-compliant, which suits a hard exclusion zone, while a few
seconds suppresses the person who walks past the edge of the area on their way somewhere else.
Default zero, so the demo's out-of-the-box behaviour is the one that is easiest to check.

**Debounce.** Once an incident has fired, it does not fire again for that person until it has
*ended* — they put the helmet on, or they leave the zone. Re-entering starts a fresh incident.

**Missed frames do not end an incident.** Detection is intermittent: F27 measured that the
weak link in this pipeline is finding the person at all, so a worker will drop out for a frame
or two while still standing exactly where they were. Requiring an unbroken run of detections
would let a flickering track accumulate no dwell and alert on nobody. An incident therefore
survives a gap of up to ``grace_seconds``; beyond that the person is treated as gone and the
timer resets. The gap is bridged, never invented — the alert still reports the dwell that
elapsed, and a track that never comes back never fires.

**Time is supplied, never read.** For a recorded file the caller passes ``frame_index / fps``,
so the same clip produces the same alerts on any machine; for a live camera it passes the wall
clock. Reading the clock inside would make this untestable and the file case irreproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.associate import CLASS_NAMES

# Alarm behaviour follows process-industry alarm-management practice, where the failure mode is
# identical to ours: an operator swamped by fleeting alarms stops reading them. ANSI/ISA-18.2 and
# EEMUA 191 both prescribe **on-delay timers** and **deadband/hysteresis** against chattering and
# fleeting alarms, and EEMUA 191 puts a manageable steady-state load below roughly six alarms per
# operator-hour. `DEFAULT_DWELL_SECONDS` is the on-delay; `DEFAULT_MEMORY_SECONDS` is the
# hysteresis. Three seconds also carries operational meaning here: a worker in a marked zone for
# three continuous seconds is working there, not walking past its corner.
# ⚠️ The transfer from process alarms to vision alerts is an **analogy** and must be written as
# one; and the alarm-rate band is a design target, not something this project has measured under
# steady-state conditions.
DEFAULT_DWELL_SECONDS = 3.0
# How long a piece of PPE stays believed once it has been seen on a tracked person. Detection
# is per frame and imperfect: a helmet occluded for two frames does not leave the head, so
# without a memory the same worker flips between compliant and violating several times a
# second. One second spans the flicker without pretending PPE persists indefinitely.
DEFAULT_MEMORY_SECONDS = 1.0
# How long a tracked person may go unseen before their incident is considered over. Two
# seconds spans the intermittent detection F27 documented without bridging a genuine exit:
# a worker who leaves the zone and returns inside two seconds has not really left it.
DEFAULT_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class Alert:
    """One fired incident — the unit that reaches the screen, the log and the snapshot."""

    track_id: int
    zone: str
    missing: tuple[int, ...]
    started_at: float
    fired_at: float

    @property
    def dwell(self) -> float:
        """Seconds of continuous non-compliance in the zone before the alert fired."""
        return self.fired_at - self.started_at

    @property
    def missing_names(self) -> str:
        """The missing PPE as a readable string ("helmet+vest") for the banner and the log."""
        return "+".join(CLASS_NAMES[cls] for cls in self.missing)


@dataclass
class PPEMemory:
    """Short-term memory of what each tracked person was last seen wearing.

    A detector answers independently every frame, so a helmet that is occluded by a beam, or
    simply missed once, reads as a bare head — and the same worker flickers between compliant
    and violating several times a second. That is unusable in front of a supervisor, and it
    is not what the footage shows: **PPE does not come on and off at frame rate.**

    So a bound item is believed for :attr:`seconds` after it was last seen on that person.
    The trade is deliberate and worth stating: it suppresses flicker at the cost of taking up
    to a second to notice PPE genuinely being removed — acceptable when the dwell threshold
    already imposes a delay before anyone is alerted, and when the measured failure of this
    pipeline is over-flagging, not under-flagging.

    **This is a property of watching video, and it applies only to tracked people.** The
    violation axis and the demo evaluation score unrelated stills, where there is no "last
    frame" to remember and no identity to remember it against, so they are unaffected — the
    demo's agreement with the reported numbers is untouched.

    Set ``seconds`` to 0 to disable it and judge every frame on its own.
    """

    seconds: float = DEFAULT_MEMORY_SECONDS
    _seen: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)

    def observe(self, now: float, track_id: int, worn: tuple[int, ...] | set[int]) -> None:
        """Record what this person is wearing right now."""
        for cls in worn:
            self._seen[(track_id, cls)] = now

    def missing(self, now: float, track_id: int, required: tuple[int, ...]) -> tuple[int, ...]:
        """Required PPE this person is missing, after allowing for what was recently seen."""
        return tuple(
            cls
            for cls in required
            if now - self._seen.get((track_id, cls), float("-inf")) > self.seconds
        )

    def forget(self, before: float) -> None:
        """Drop entries older than ``before``, so a long run does not accumulate every track."""
        self._seen = {key: when for key, when in self._seen.items() if when >= before}


@dataclass
class _Incident:
    """The open, not-yet-ended non-compliance of one tracked person."""

    started_at: float
    last_seen: float
    missing: tuple[int, ...]
    fired: bool = False


@dataclass
class DwellTracker:
    """Per-track state machine converting frame-wise violations into debounced alerts.

    Args:
        dwell_seconds: Continuous non-compliance required before an alert fires. ``0`` fires
            on the first observation.
        grace_seconds: How long a track may go unobserved before its incident is closed.
        zone: Name of the zone being watched, carried onto every alert it produces.
    """

    dwell_seconds: float = DEFAULT_DWELL_SECONDS
    grace_seconds: float = DEFAULT_GRACE_SECONDS
    zone: str = "zone"
    _open: dict[int, _Incident] = field(default_factory=dict, repr=False)

    def update(self, now: float, violating: dict[int, tuple[int, ...]]) -> list[Alert]:
        """Advance the state machine by one frame and return any alerts that fired.

        Args:
            now: Timestamp of this frame, in seconds.
            violating: ``{track_id: missing PPE class ids}`` for every tracked person who is
                **in the zone and non-compliant right now**. A person who is compliant, outside
                the zone, or undetected this frame is simply absent from the mapping — the
                caller does not have to distinguish those, because all three mean "not an
                ongoing violation to count".

        Returns:
            Alerts that fired on this frame, at most one per track.
        """
        fired: list[Alert] = []

        for track_id, missing in violating.items():
            incident = self._open.get(track_id)
            if incident is None or now - incident.last_seen > self.grace_seconds:
                # A new incident: first sighting, or the previous one lapsed past the grace
                # window and this is a fresh entry rather than a continuation.
                incident = _Incident(started_at=now, last_seen=now, missing=missing)
                self._open[track_id] = incident
            else:
                incident.last_seen = now
                incident.missing = missing

            if not incident.fired and now - incident.started_at >= self.dwell_seconds:
                incident.fired = True
                fired.append(
                    Alert(
                        track_id=track_id,
                        zone=self.zone,
                        missing=missing,
                        started_at=incident.started_at,
                        fired_at=now,
                    )
                )

        # Close incidents nobody has reported for longer than the grace window. The same rule
        # covers compliance, leaving the zone and a dropped detection, because the caller
        # cannot always tell them apart — a worker who is not reported as violating simply
        # stops accumulating, and their incident ends once the gap outlasts the window. A
        # caller that *does* know the state changed calls `resolve` and ends it at once.
        for track_id, incident in list(self._open.items()):
            if track_id not in violating and now - incident.last_seen > self.grace_seconds:
                del self._open[track_id]

        return fired

    def resolve(self, track_id: int) -> None:
        """End a person's incident immediately (they became compliant, or left the zone).

        The caller uses this when it *knows* the state changed, rather than relying on the
        absence rule, so that a worker who complies can trigger a new alert straight away if
        they re-offend.
        """
        self._open.pop(track_id, None)

    @property
    def open_incidents(self) -> int:
        """How many incidents are currently being timed — the banner's "watching" count."""
        return len(self._open)
