import numpy as np
import pytest
from opytimizer.core.agent import Agent
import opytimizer.utils.exception as e
from opytimizer.utils.operators import (
    ArithmeticCrossover,
    BitFlipMutation,
    GaussianMutation,
    OnePointCrossover,
    PolynomialMutation,
    SBXCrossover,
)

class MockAgent:
    def __init__(self, n_vars, lb=0.0, ub=1.0):
        self.position = np.random.uniform(lb, ub, (n_vars, 1))
        self.lb = np.full((n_vars, 1), lb)
        self.ub = np.full((n_vars, 1), ub)

def test_base_crossover_properties():
    crossover = ArithmeticCrossover()
    
    assert crossover.rate == 1.0
    assert crossover.return_mode == 'both'

    crossover.rate = 0.5
    assert crossover.rate == 0.5

    crossover.return_mode = 'first'
    assert crossover.return_mode == 'first'

    with pytest.raises(e.ValueError):
        crossover.rate = 1.5

    with pytest.raises(e.ValueError):
        crossover.rate = -0.1

    with pytest.raises(ValueError):
        crossover.return_mode = 'invalid'

def test_arithmetic_crossover():
    parent1 = MockAgent(5)
    parent2 = MockAgent(5)
    
    crossover = ArithmeticCrossover(rate=1.0, gene_rate=1.0)
    children = crossover(parent1, parent2)

    assert len(children) == 2
    assert isinstance(children[0], Agent) or isinstance(children[0], MockAgent)
    assert children[0].position.shape == (5, 1)
    assert np.all(children[0].position >= parent1.lb)
    assert np.all(children[0].position <= parent1.ub)

def test_sbx_crossover():
    parent1 = MockAgent(10, 0, 10)
    parent2 = MockAgent(10, 0, 10)
    
    crossover = SBXCrossover(rate=1.0, eta=20)
    children = crossover(parent1, parent2)

    assert len(children) == 1
    assert children[0].position.shape == (10, 1)
    
    assert np.all(children[0].position >= parent1.lb)
    assert np.all(children[0].position <= parent1.ub)
    
    new_crossover = SBXCrossover(return_mode='both')
    new_children = new_crossover(parent1, parent2)

    assert len(new_children) == 2
    assert new_children[0].position.shape == (10, 1)
    assert np.all(new_children[1].position >= parent2.lb)
    assert np.all(new_children[1].position <= parent2.ub)

def test_sbx_crossover_return_modes():
    parent1 = MockAgent(5)
    parent2 = MockAgent(5)
    
    crossover = SBXCrossover(return_mode='first')
    children = crossover(parent1, parent2)
    assert len(children) == 1
    
    crossover.return_mode = 'random'
    children = crossover(parent1, parent2)
    assert len(children) == 1

def test_one_point_crossover():
    parent1 = MockAgent(10)
    parent2 = MockAgent(10)
    
    crossover = OnePointCrossover(rate=1.0)
    children = crossover(parent1, parent2)

    assert len(children) == 1
    assert children[0].position.shape == (10, 1)
    p1 = parent1.position.flatten()
    c1 = children[0].position.flatten()
    assert np.any(c1 == p1) 
    
    
    new_crossover = OnePointCrossover(rate=1.0, return_mode='both')
    new_children = new_crossover(parent1, parent2)

    assert len(new_children) == 2
    assert new_children[0].position.shape == (10, 1)
    assert new_children[1].position.shape == (10, 1)
    p1 = parent1.position.flatten()
    c1 = new_children[0].position.flatten()
    c2 = new_children[1].position.flatten()
    assert np.any(c1 == p1) 
    assert np.any(c2 == p1) 
    
    
    

def test_base_mutation_properties():
    mutation = GaussianMutation()
    
    assert mutation.rate == 0.025 # Default inherited
    
    mutation.rate = 0.5
    assert mutation.rate == 0.5
    
    with pytest.raises(e.ValueError):
        mutation.rate = 1.1

def test_gaussian_mutation():
    agent = MockAgent(10, 0, 10)
    mutation = GaussianMutation(std=1.0)
    mutation.rate = 1.0 
    
    mutant = mutation(agent)
    
    assert mutant.position.shape == (10, 1)
    assert np.all(mutant.position >= agent.lb)
    assert np.all(mutant.position <= agent.ub)
    assert not np.array_equal(mutant.position, agent.position)

def test_bit_flip_mutation():
    agent = MockAgent(10, 0, 1)
    agent.position = np.round(agent.position) 
    
    mutation = BitFlipMutation(rate=1.0)
    mutant = mutation(agent)

    assert mutant.position.shape == (10, 1)
    assert np.all(np.logical_or(mutant.position == 0, mutant.position == 1))
    
    diff = np.abs(mutant.position - agent.position)
    assert np.sum(diff) > 0

def test_polynomial_mutation():
    agent = MockAgent(10, 0, 10)
    mutation = PolynomialMutation(rate=1.0, eta=20)
    
    mutant = mutation(agent)
    
    assert mutant.position.shape == (10, 1)
    assert np.all(mutant.position >= agent.lb)
    assert np.all(mutant.position <= agent.ub)
    
    with pytest.raises(e.TypeError):
        mutation.eta = 20.5
        
    with pytest.raises(e.ValueError):
        mutation.eta = -1