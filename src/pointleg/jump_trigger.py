"""Deployment command clock shared with the fixed-jump observation contract."""

import math


class JumpTrigger:
    """Latch one trigger; repeated requests cannot restart an active jump."""

    duration = 3.

    def __init__(self):
        self.started_at = None
        self.armed = True
        self.last_time = None

    def update(self, signal, monotonic_seconds):
        if not math.isfinite(monotonic_seconds):
            raise ValueError("Jump clock must be finite")
        if self.last_time is not None and monotonic_seconds < self.last_time:
            raise ValueError("Jump clock must be monotonic")
        self.last_time = monotonic_seconds
        if not signal:
            self.armed = True
        if signal and self.armed and self.started_at is None:
            self.started_at = monotonic_seconds
            self.armed = False
        if self.started_at is None:
            return (0., 0., 0.)
        elapsed = monotonic_seconds - self.started_at
        if elapsed >= self.duration:
            self.finish()
            return (0., 0., 0.)
        return (1., elapsed / self.duration, 0.)

    def finish(self):
        """Call after four-foot stable landing or a runtime abort."""
        self.started_at = None
        self.armed = False
