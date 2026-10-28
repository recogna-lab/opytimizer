"""Stopping criteria for the optimization loop."""

import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from tqdm import tqdm


class _StoppingCriterion(ABC):
    """Base Class.  `should_stop` must be implemented in each subclass"""

    def __init__(self):
        self.pbar = None

    @abstractmethod
    def should_stop(self, opt) -> bool:
        """Return True when the criterio was satisfied"""

    def reset(self) -> None:
        """Internal State reinicialization."""
        if self.pbar:
            self.pbar.reset()

    def init_pbar(self, position: int) -> None:
        """Progress Bar initialization."""
        pass

    def update_pbar(self, opt) -> None:
        """Update the Progress Bar to the current optimization state."""
        pass

    def _advance_pbar(self, value: int) -> None:
        """Moves the progress bar to `value` without forcing a terminal write."""
        if self.pbar is None:
            return

        delta = value - self.pbar.n
        if delta >= 0:
            self.pbar.update(delta)
        else:
            # Backwards move (e.g. the patience counter was reset). `update()`
            # cannot express this, so redraw explicitly.
            self.pbar.n = value
            self.pbar.last_print_n = value
            self.pbar.refresh()

    def close_pbar(self) -> None:
        """Closes the Progress Bar safely"""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class MaxIterations(_StoppingCriterion):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def should_stop(self, opt) -> bool:
        return opt.total_iterations >= self.n

    def init_pbar(self, position: int) -> None:
        self.pbar = tqdm(
            total=self.n,
            desc="Max iterations",
            position=position,
            leave=True,
            ascii=True,
        )

    def update_pbar(self, opt) -> None:
        self._advance_pbar(opt.total_iterations)


class MaxEvaluations(_StoppingCriterion):
    """Soft check: verified at end of each iteration.
    For exact enforcement, use Function(budget=n) instead."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def should_stop(self, opt) -> bool:
        return opt.n_evals >= self.n

    def init_pbar(self, position) -> None:
        self.pbar = tqdm(
            total=self.n,
            desc="Max function evaluations",
            position=position,
            leave=True,
            ascii=True,
        )

    def update_pbar(self, opt) -> None:
        self._advance_pbar(opt.n_evals)


class NoImprovement(_StoppingCriterion):
    """`patience` non-improvement iterations.

    When the best fitness lives on a CUDA device (tensorized spaces), comparing
    it against a Python float forces a device -> host synchronization on every
    call, which drains the whole kernel queue. To avoid that, the running best
    and the non-improvement counter are kept ON the device and updated with
    async kernels; the counter is only read back every `check_every` calls.

    Args:
        patience: Number of consecutive non-improving iterations tolerated.
        min_delta: Minimum decrease that counts as an improvement.
        check_every: Read the counter back to the host every this many calls.
            The default (1) keeps the exact semantics of the original
            implementation. On GPU, a larger value (e.g. 10) removes most of the
            synchronizations at the price of overshooting the stop by at most
            `check_every - 1` iterations. It has no effect on the cost of a
            host-resident (numpy) run.
    """

    def __init__(
        self, patience: int, min_delta: float = 1e-8, check_every: int = 1
    ) -> None:
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.check_every = max(1, int(check_every))
        self._reset_state()

    def _reset_state(self) -> None:
        # `_best` / `_counter` are created lazily on the first call because their
        # array namespace (numpy vs cupy) depends on where the fitness lives.
        self._best = None
        self._counter = None
        self._calls = 0
        self._host_counter = 0  # last value actually read back to the host

    def should_stop(self, opt) -> bool:
        if opt.space.n_objectives != 1:
            return False

        fit = opt.space.best_agent.fit

        # CuPy arrays expose `__cuda_array_interface__`; host scalars/arrays do not.
        on_device = hasattr(fit, "__cuda_array_interface__")
        xp = opt.space.env.xp.__wrapped__ if on_device else np
        current = xp.asarray(fit)

        if self._best is None:
            self._best = xp.asarray(np.inf, dtype=current.dtype)
            self._counter = xp.zeros((), dtype=xp.int64)

        improved = (self._best - current) > self.min_delta
        self._best = xp.where(improved, current, self._best)
        self._counter = xp.where(improved, 0, self._counter + 1)

        self._calls += 1
        if self._calls % self.check_every:
            return False

        # The only device -> host transfer of this criterion.
        self._host_counter = int(self._counter)
        return self._host_counter >= self.patience

    def reset(self) -> None:
        self._reset_state()

    def init_pbar(self, position) -> None:
        self.pbar = tqdm(
            total=self.patience,
            desc=f"Non-improvement iterations ({self.min_delta})",
            position=position,
            leave=True,
            ascii=True,
        )

    def update_pbar(self, opt) -> None:
        # Uses the last counter value read back to the host, so drawing the bar
        # never triggers an extra device synchronization.
        self._advance_pbar(self._host_counter)


class _StoppingVessel:
    """Criteria aggregation."""

    def __init__(
        self,
        criteria: Optional[List[_StoppingCriterion]],
        postfix_interval: float = 0.25,
    ) -> None:
        self._criteria = criteria or []
        # Minimum number of seconds between two progress-bar postfix updates.
        self._postfix_interval = postfix_interval
        self._last_postfix = float("-inf")
        # Latest postfix request that has not been drawn yet (see `flush_postfix`).
        self._pending_postfix = None

    def should_stop(self, opt) -> bool:
        return any(c.should_stop(opt) for c in self._criteria)

    def reset(self) -> None:
        for c in self._criteria:
            c.reset()

    def init_pbars(self) -> None:
        for i, c in enumerate(self._criteria):
            c.init_pbar(position=i)

    def update_pbars(self, opt) -> None:
        for c in self._criteria:
            c.update_pbar(opt)

    def set_postfix(self, **kwargs) -> None:
        """Updates the postfix of every progress bar, throttled by wall-clock time.

        Values may be zero-argument callables, which are only evaluated when the
        postfix is actually going to be drawn.
        """
        pbars = [c.pbar for c in self._criteria if c.pbar]
        if not pbars:
            return

        # Always remember the latest request, even when it is throttled away: a short
        # run may end before the next draw is due, and the bar would otherwise be closed
        # showing a stale value from an early iteration (`flush_postfix`).
        self._pending_postfix = kwargs

        now = time.monotonic()
        if now - self._last_postfix < self._postfix_interval:
            return

        self._draw_postfix(pbars, kwargs, now)

    def _draw_postfix(self, pbars, kwargs, now) -> None:
        self._last_postfix = now
        self._pending_postfix = None

        values = {k: (v() if callable(v) else v) for k, v in kwargs.items()}
        for pbar in pbars:
            # refresh=False: the next throttled `update()` (or `close()`) draws it. The
            # default (refresh=True) would force a second terminal write per call.
            pbar.set_postfix(refresh=False, **values)

    def flush_postfix(self) -> None:
        """Draws the latest postfix request that the time throttle skipped, if any.

        Callable values are evaluated now, so the bar reports the state at the end of
        the run rather than at the moment of the last throttled update.
        """
        pbars = [c.pbar for c in self._criteria if c.pbar]
        if pbars and self._pending_postfix is not None:
            self._draw_postfix(pbars, self._pending_postfix, time.monotonic())

    def close_pbars(self) -> None:
        # Must run before closing: `close()` -> draws the final state of the bar.
        self.flush_postfix()

        for c in self._criteria:
            c.close_pbar()
