"""Backend-agnostic helpers shared by the block-based evolutionary algorithms."""

from typing import Any


def _categorical(probabilities: Any, shape: tuple, xp: Any) -> Any:
    """Samples categories from per-block probabilities.

    Args:
        probabilities: Normalized probabilities of shape (B, K).
        shape: Sample shape (B, ...), whose leading axis indexes the blocks.
        xp: Array module (numpy / cupy).

    Returns:
        (Any): Integer categories in [0, K) of shape `shape`.

    """

    cdf = xp.cumsum(probabilities, axis=1)
    cdf = cdf.reshape((cdf.shape[0],) + (1,) * (len(shape) - 1) + (cdf.shape[1],))
    u = xp.random.random(shape + (1,))

    return xp.minimum((u > cdf).sum(axis=-1), probabilities.shape[1] - 1)


def _regroup(X: Any, n_blocks: int) -> Any:
    """Equally divides every block among `n_blocks` next blocks.

    Block j of the output receives the j-th chunk of every input block, so each
    next block mixes solutions coming from all previous blocks.

    Args:
        X: Solutions of shape (B, S, D), with S divisible by `n_blocks`.
        n_blocks: Number of next blocks.

    Returns:
        (Any): Solutions of shape (n_blocks, B * S / n_blocks, D).

    """

    B, S, D = X.shape
    chunk = S // n_blocks
    X = X.reshape(B, n_blocks, chunk, D).transpose(1, 0, 2, 3)

    return X.reshape(n_blocks, B * chunk, D)


def _tournament(F: Any, mask: Any, n_select: int, xp: Any) -> Any:
    """m-tournament selection for several blocks at once (minimization).

    Args:
        F: Objective values of shape (N,).
        mask: Boolean mask of shape (B, 1, m_max): True for the first m_b
            competitors of block b, so each block can use its own m.
        n_select: Number of parents selected by each block.
        xp: Array module (numpy / cupy).

    Returns:
        (Any): Indices of the selected parents, of shape (B, n_select).

    """

    candidates = xp.random.randint(0, F.shape[0], size=(mask.shape[0], n_select, mask.shape[2]))
    fit = xp.where(mask, F[candidates], xp.inf)
    winners = xp.take_along_axis(candidates, fit.argmin(axis=2)[..., None], axis=2)

    return winners[..., 0]
