"""Geometric Semantic Genetic Programming (Strongly-Typed).
"""

import operator
import random
from hashlib import sha1
from typing import Any, Dict, Optional, Type

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core.graph.node import GraphNode
from opytimizer.core.graph.space import _SingleObjectiveSpace
from opytimizer.core.graph.tree import Tree
from opytimizer.optimizers.single_objective.evolutionary.gp import GP
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class GSGP(GP):
    """A GSGP class, inherited from GP.


    References:
        A. Moraglio, K. Krawiec, and C. G. Johnson.
        Geometric semantic genetic programming.
        Lecture Notes in Computer Science (2012).

        G. H. de Rosa, J. P. Papa, and L. P. Papa.
        Feature selection using geometric semantic genetic programming.
        Proceedings of the Genetic and Evolutionary Computation Conference Companion (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: GP -> GSGP.")

        super(GSGP, self).__init__(params)

        self.ms = 0.1

        logger.info("Class overrided.")

    @property
    def ms(self) -> float:
        """Mutation step -- scales the `(TR1 - TR2)` random perturbation."""

        return self._ms

    @ms.setter
    def ms(self, ms: float) -> None:
        if not isinstance(ms, (float, int)):
            raise e.TypeError("`ms` should be a float or integer")
        if ms < 0:
            raise e.ValueError("`ms` should be >= 0")

        self._ms = ms

    def _hashed_name(self) -> str:
        """Builds a short, collision-unlikely name for an ad-hoc terminal,
        mirroring the original implementation's `sha1`-based naming."""

        value = r.generate_uniform_random_number().item()
        return sha1(repr(value).encode("ascii")).hexdigest()[:4]

    def _random_terminal_value(self, output_type: Type) -> Any:
        """Draws a fresh numeric value of `output_type`, passed through one
        of three bounding nonlinearities.

        Args:
            output_type: Type the returned value must be usable as (assumed
                numeric: `float`, `int`, or a numpy array).

        """

        value = r.generate_uniform_random_number().item()

        operator_id = r.generate_integer_random_number(0, 3)
        if operator_id == 0:
            value = np.exp(value)
        elif operator_id == 1:
            value = np.fabs(np.sin(value))
        elif operator_id == 2:
            value = np.cos(np.sin(value))

        if output_type is int:
            return int(value)
        if output_type is float:
            return float(value)

        return value

    def _mutate(self, tree: Tree) -> Tree:
        """Performs geometric semantic mutation on a single tree: replaces a randomly chosen node `T` with
        `SUM(T, MUL(ms, SUB(TR1, TR2)))`, where `TR1`/`TR2` are fresh
        random terminals of `T`'s own type.

        Args:
            tree: A `Tree` instance to be mutated.

        Returns:
            (Tree): A new, mutated tree (the input is left untouched).

        """

        mutated = tree.copy()
        target = random.choice(mutated.nodes)
        output_type = target.output_type

        terminal_1 = GraphNode(
            self._hashed_name(),
            value=self._random_terminal_value(output_type),
            output_type=output_type,
        )
        terminal_2 = GraphNode(
            self._hashed_name(),
            value=self._random_terminal_value(output_type),
            output_type=output_type,
        )

        sub_node = GraphNode(
            "SUB", value=operator.sub, output_type=output_type, is_terminal=False
        )
        sub_node.add_child(terminal_1)
        sub_node.add_child(terminal_2)

        ms_node = GraphNode("ms", value=self.ms, output_type=output_type)
        mul_node = GraphNode(
            "MUL", value=operator.mul, output_type=output_type, is_terminal=False
        )
        mul_node.add_child(ms_node)
        mul_node.add_child(sub_node)

        sum_node = GraphNode(
            "SUM", value=operator.add, output_type=output_type, is_terminal=False
        )
        sum_node.add_child(target.copy())
        sum_node.add_child(mul_node)

        mutated.replace_subtree(target, sum_node)

        return mutated

    def _mutation(self, space: _SingleObjectiveSpace) -> None:
        """Mutates a number of individuals pre-selected through a tournament procedure.

        Args:
            space: A TreeSpace object.

        """

        fitness = [agent.fit for agent in space.agents]
        n_individuals = int(space.n_agents * self.p_mutation)

        selected = g.tournament_selection(fitness, n_individuals)
        for s in selected:
            space.agents[s].position = self._mutate(space.agents[s].position)

    def _cross(self, father: Tree, mother: Tree) -> Tree:
        """Performs geometric semantic crossover: builds
        `SUM(MUL(TR, T1), MUL(1 - TR, T2))`, where `TR` is a fresh random
        gate value in `[0, 1]` and `T1`/`T2` are same-typed nodes picked
        from `father`/`mother`.

        Args:
            father: A father's tree to be crossed.
            mother: A mother's tree to be crossed.

        Returns:
            (Tree): A single offspring (a new tree; parents are untouched).

        """

        father_offspring = father.copy()
        mother_offspring = mother.copy()

        father_types = {n.output_type for n in father_offspring.nodes}
        mother_types = {n.output_type for n in mother_offspring.nodes}
        common_types = father_types & mother_types
        if not common_types:
            return father_offspring

        output_type = random.choice(list(common_types))
        sub_father = random.choice(father_offspring.nodes_of_type(output_type))
        sub_mother = random.choice(mother_offspring.nodes_of_type(output_type))

        gate_value = r.generate_uniform_random_number().item()  # already in [0, 1]

        gate_node = GraphNode(
            self._hashed_name(), value=gate_value, output_type=output_type
        )
        not_gate_node = GraphNode("~", value=1 - gate_value, output_type=output_type)

        left = GraphNode(
            "MUL", value=operator.mul, output_type=output_type, is_terminal=False
        )
        left.add_child(gate_node)
        left.add_child(sub_father.copy())

        right = GraphNode(
            "MUL", value=operator.mul, output_type=output_type, is_terminal=False
        )
        right.add_child(not_gate_node)
        right.add_child(sub_mother.copy())

        root = GraphNode(
            "SUM", value=operator.add, output_type=output_type, is_terminal=False
        )
        root.add_child(left)
        root.add_child(right)

        father_offspring.replace_subtree(sub_father, root)

        return father_offspring

    def _crossover(self, space: _SingleObjectiveSpace) -> None:
        """Crossover a number of individuals pre-selected through a tournament procedure.

        Args:
            space: A TreeSpace object.

        """

        fitness = [agent.fit for agent in space.agents]

        n_individuals = int(space.n_agents * self.p_crossover)
        if n_individuals % 2 != 0:
            n_individuals += 1

        selected = g.tournament_selection(fitness, n_individuals)
        for s in g.n_wise(selected):
            if s[0] == s[1]:
                continue

            space.agents[s[0]].position = self._cross(
                space.agents[s[0]].position, space.agents[s[1]].position
            )
