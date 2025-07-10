import copy
from abc import ABC, abstractmethod

import numpy as np

import opytimizer.utils.random as r
from opytimizer.core.agent import Agent


class BaseCrossover(ABC):
    """Abstract base class for crossover operators."""

    @abstractmethod
    def __call__(self, parent1, parent2, *args, **kwargs):
        pass


class BaseMutation(ABC):
    """Abstract base class for mutation operators."""

    @abstractmethod
    def __call__(self, vector, *args, **kwargs):
        pass


class ArithmeticCrossover(BaseCrossover):
    """Arithmetic crossover for real-valued vectors."""

    def __call__(self, parent1: Agent, parent2: Agent) -> tuple:
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        x1 = parent1.position
        x2 = parent2.position

        alpha = r.generate_uniform_random_number(0.0, 1.0, size=x1.shape)

        child1.position = alpha * x1 + (1 - alpha) * x2
        child2.position = alpha * x2 + (1 - alpha) * x1

        return child1, child2


class GaussianMutation(BaseMutation):
    """Gaussian mutation for real-valued vectors."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, agent: Agent) -> Agent:
        mutant = copy.deepcopy(agent)

        x = agent.position
        lb = agent.lb
        ub = agent.ub

        noise = r.generate_gaussian_random_number(
            mean=0.0, variance=self.std, size=x.shape
        )

        mutated = x + noise
        mutated = np.clip(mutated, lb, ub)

        mutant.position = mutated
        return mutant


class SBXCrossover(BaseCrossover):
    """Simulated Binary Crossover (SBX) for real-valued vectors."""

    def __init__(self, eta=20):
        self.eta = eta

    def __call__(self, parent1: Agent, parent2: Agent) -> tuple:
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        x1 = parent1.position.copy()
        x2 = parent2.position.copy()
        lb = parent1.lb
        ub = parent1.ub
        c1 = x1.copy()
        c2 = x2.copy()
        for j in range(x1.shape[0]):
            for d in range(x1.shape[1]):
                if np.random.rand() <= 0.5:
                    if abs(x1[j, d] - x2[j, d]) > 1e-10:
                        y1, y2 = (
                            (x1[j, d], x2[j, d])
                            if x1[j, d] < x2[j, d]
                            else (x2[j, d], x1[j, d])
                        )
                        rand = np.random.rand()
                        beta = 1.0 + (2.0 * (y1 - lb[j]) / (y2 - y1))
                        alpha = 2.0 - beta ** -(self.eta + 1)
                        if rand <= 1.0 / alpha:
                            betaq = (rand * alpha) ** (1.0 / (self.eta + 1))
                        else:
                            betaq = (1.0 / (2.0 - rand * alpha)) ** (
                                1.0 / (self.eta + 1)
                            )
                        c1[j, d] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (ub[j] - y2) / (y2 - y1))
                        alpha = 2.0 - beta ** -(self.eta + 1)
                        if rand <= 1.0 / alpha:
                            betaq = (rand * alpha) ** (1.0 / (self.eta + 1))
                        else:
                            betaq = (1.0 / (2.0 - rand * alpha)) ** (
                                1.0 / (self.eta + 1)
                            )
                        c2[j, d] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                        c1[j, d] = np.clip(c1[j, d], lb[j], ub[j])
                        c2[j, d] = np.clip(c2[j, d], lb[j], ub[j])
                    else:
                        c1[j, d] = x1[j, d]
                        c2[j, d] = x2[j, d]
                else:
                    c1[j, d] = x1[j, d]
                    c2[j, d] = x2[j, d]
        child1.position = c1
        child2.position = c2
        return child1, child2


class OnePointCrossover(BaseCrossover):
    """One-point crossover for binary or real-valued vectors."""

    def __call__(self, parent1: Agent, parent2: Agent) -> tuple:
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        p1 = parent1.position.copy()
        p2 = parent2.position.copy()
        lb = parent1.lb
        ub = parent1.ub

        point = np.random.randint(1, p1.shape[0])
        c1 = np.vstack((p1[:point, :], p2[point:, :]))
        c2 = np.vstack((p2[:point, :], p1[point:, :]))
        c1 = np.clip(c1, lb[:, None], ub[:, None])
        c2 = np.clip(c2, lb[:, None], ub[:, None])
        child1.position = c1
        child2.position = c2
        return child1, child2


class BitFlipMutation(BaseMutation):
    """Bit flip mutation for binary vectors."""

    def __call__(self, agent: Agent) -> Agent:
        mutant = copy.deepcopy(agent)
        x = agent.position
        m = x.copy()
        mask = np.random.rand(*x.shape) < 0.5
        m[mask] = 1 - m[mask]
        mutant.position = m
        return mutant


class PolynomialMutation(BaseMutation):
    """Polynomial mutation for real-valued vectors."""

    def __init__(self, eta=20):
        self.eta = eta

    def __call__(self, agent: Agent) -> Agent:
        mutant = copy.deepcopy(agent)
        x = agent.position.copy()
        lb = agent.lb
        ub = agent.ub
        m = x.copy()
        for j in range(x.shape[0]):
            for d in range(x.shape[1]):
                if np.random.rand() < 1.0 / (x.shape[0] * x.shape[1]):
                    delta1 = (m[j, d] - lb[j]) / (ub[j] - lb[j])
                    delta2 = (ub[j] - m[j, d]) / (ub[j] - lb[j])
                    rand = np.random.rand()
                    mut_pow = 1.0 / (self.eta + 1.0)
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta + 1))
                        deltaq = val**mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (
                            xy ** (self.eta + 1)
                        )
                        deltaq = 1.0 - val**mut_pow
                    m[j, d] += deltaq * (ub[j] - lb[j])
                    m[j, d] = np.clip(m[j, d], lb[j], ub[j])
        mutant.position = m
        return mutant
