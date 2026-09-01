"""Particle Swarm Optimization-based algorithms.
"""

import copy
import time
from typing import Any, Dict, List, Optional

import numpy as np

from dataclasses import dataclass

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer, TensorizedOptimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import  _SingleObjectiveSpace, _SingleObjectiveTensorSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)



@dataclass
class PSO(Optimizer):
    """A PSO class, inherited from Optimizer.

    This is the designed class to define PSO-related
    variables and methods.

    References:
        J. Kennedy, R. C. Eberhart and Y. Shi. Swarm intelligence.
        Artificial Intelligence (2001).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> PSO.")

        super().__init__()

        self.w = 0.7
        self.c1 = 1.7
        self.c2 = 1.7

        self.build(params)

        logger.info("Class overrided.")

    @property
    def w(self) -> float:
        """Inertia weight."""

        return self._w

    @w.setter
    def w(self, w: float) -> None:
        if not isinstance(w, (float, int)):
            raise e.TypeError("`w` should be a float or integer")
        if w < 0:
            raise e.ValueError("`w` should be >= 0")

        self._w = w

    @property
    def c1(self) -> float:
        """Cognitive constant."""

        return self._c1

    @c1.setter
    def c1(self, c1: float) -> None:
        if not isinstance(c1, (float, int)):
            raise e.TypeError("`c1` should be a float or integer")
        if c1 < 0:
            raise e.ValueError("`c1` should be >= 0")

        self._c1 = c1

    @property
    def c2(self) -> float:
        """Social constant."""

        return self._c2

    @c2.setter
    def c2(self, c2: float) -> None:
        if not isinstance(c2, (float, int)):
            raise e.TypeError("`c2` should be a float or integer")
        if c2 < 0:
            raise e.ValueError("`c2` should be >= 0")

        self._c2 = c2

    @property
    def local_position(self) -> np.ndarray:
        """Array of velocities."""

        return self._local_position

    @local_position.setter
    def local_position(self, local_position: np.ndarray) -> None:
        if not isinstance(local_position, np.ndarray):
            raise e.TypeError("`local_position` should be a numpy array")

        self._local_position = local_position

    @property
    def velocity(self) -> np.ndarray:
        """Array of velocities."""

        return self._velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray) -> None:
        if not isinstance(velocity, np.ndarray):
            raise e.TypeError("`velocity` should be a numpy array")

        self._velocity = velocity

    def compile(self, space: _SingleObjectiveSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def evaluate(self, space: _SingleObjectiveSpace, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            fit = function(agent.position)
            if fit < agent.fit:
                agent.fit = fit
                self.local_position[i] = copy.deepcopy(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(self.local_position[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: _SingleObjectiveSpace) -> None:
        """Wraps Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        for i, agent in enumerate(space.agents):
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates agent's velocity (p. 294)
            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            # Updates agent's position (p. 294)
            agent.position += self.velocity[i]


_VELOCITY_UPDATE_KERNEL_SRC = r"""
extern "C" __global__
void velocity_update(
    double* __restrict__ vel,         
    const double* __restrict__ lbest,  
    const double* __restrict__ pos,    
    const double* __restrict__ gbest,  
    const double* __restrict__ r1,     
    const double* __restrict__ r2,     
    const double w,
    const double c1,
    const double c2,
    const int vars_dims,               
    const int N                        
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
 
    int agent_idx = idx / vars_dims;
    int var_idx   = idx % vars_dims;
 
    double v = w * vel[idx]
             + c1 * r1[agent_idx] * (lbest[idx] - pos[idx])
             + c2 * r2[agent_idx] * (gbest[var_idx] - pos[idx]);
    vel[idx] = v;
}
"""
 
_VELOCITY_UPDATE_KERNEL_F32_SRC = r"""
extern "C" __global__
void velocity_update_f32(
    float* __restrict__ vel,
    const float* __restrict__ lbest,
    const float* __restrict__ pos,
    const float* __restrict__ gbest,
    const float* __restrict__ r1,
    const float* __restrict__ r2,
    const float w,
    const float c1,
    const float c2,
    const int vars_dims,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
 
    int agent_idx = idx / vars_dims;
    int var_idx   = idx % vars_dims;
 
    float v = w * vel[idx]
            + c1 * r1[agent_idx] * (lbest[idx] - pos[idx])
            + c2 * r2[agent_idx] * (gbest[var_idx] - pos[idx]);
    vel[idx] = v;
}
"""


class PSOCuda(Optimizer, TensorizedOptimizer):
    """
        GPU-friendly, fully tensorized implementation of PSO (single-objective).

        All particle state -- current position, velocity, personal best
        (local) position/fitness, and the running global best -- lives on the
        GPU as `xp` tensors for the entire run. There is NO host <-> device
        synchronization inside `evaluate`/`update`; `space.agents` is read
        exactly once, at `compile` time, to seed the initial positions. The
        only place data is pulled back to the host is `sync_with_cpu`, meant
        to be called a single time, after the optimization loop has finished.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        logger.info("Overriding class: Optimizer -> PSO.")
        super().__init__()

        self.w = 0.7
        self.c1 = 1.7
        self.c2 = 1.7

        # Persistent GPU-resident state, populated in `compile`.
        self.position = None  # current particle positions (n_a, n_v)
        self.velocity = None  # particle velocities (n_a, n_v,)
        self.local_position = None   # personal-best (pbest) position (n_a, n_v)
        self.fit = None   # fitness of `self.position` (n_a,)
        self.local_fit = None   # personal-best fitness  (n_a,)
        self.global_best_position = None # gbest position  (n_v, n_d)
        self.global_best_fit = None   # gbest fitness  scalar tensor

        self.build(params)
        self._vel_kernel = None

        logger.info("Class overrided.")

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, w: float) -> None:
        if not isinstance(w, (float, int)):
            raise e.TypeError("`w` should be a float or integer")
        if w < 0:
            raise e.ValueError("`w` should be >= 0")
        self._w = w

    @property
    def c1(self) -> float:
        return self._c1

    @c1.setter
    def c1(self, c1: float) -> None:
        if not isinstance(c1, (float, int)):
            raise e.TypeError("`c1` should be a float or integer")
        if c1 < 0:
            raise e.ValueError("`c1` should be >= 0")
        self._c1 = c1

    @property
    def c2(self) -> float:
        return self._c2

    @c2.setter
    def c2(self, c2: float) -> None:
        if not isinstance(c2, (float, int)):
            raise e.TypeError("`c2` should be a float or integer")
        if c2 < 0:
            raise e.ValueError("`c2` should be >= 0")
        self._c2 = c2

    @property
    def local_position(self):
        return self._local_position

    @local_position.setter
    def local_position(self, local_position) -> None:
        self._local_position = local_position

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity) -> None:
        self._velocity = velocity

    def compile(self, space: _SingleObjectiveTensorSpace) -> None:
        xp = space.env.xp
        dtype = space.env.dtype
        n_a = space.n_agents
        n_v = space.n_variables
        n_d = space.n_dimensions

        
        
        self.local_position = space.X.copy()
        self.velocity = xp.zeros((n_a, n_v), dtype=dtype)


        self.local_fit = xp.full(n_a, xp.inf, dtype=dtype)

        self.global_best_position = xp.zeros((n_v, n_d), dtype=dtype)
        
        self.global_best_fit = xp.asarray(xp.inf, dtype=dtype)

        self._lb = xp.asarray(space.lb, dtype=dtype).reshape(1, n_v, 1)
        self._ub = xp.asarray(space.ub, dtype=dtype).reshape(1, n_v, 1)

        use_f64 = (dtype == xp.float64 or dtype == 'float64')
        if use_f64:
            self._vel_kernel = xp.RawKernel(_VELOCITY_UPDATE_KERNEL_SRC, 'velocity_update')
        else:
            self._vel_kernel = xp.RawKernel(_VELOCITY_UPDATE_KERNEL_F32_SRC, 'velocity_update_f32')

        self._vars_dims = n_v * n_d
        self._kernel_N = n_a * self._vars_dims
        self._block_size = 256
        self._kernel_grid = (self._kernel_N + self._block_size - 1) // self._block_size
        self._dtype = dtype

    def evaluate(self, space: _SingleObjectiveTensorSpace, function: Function) -> None:
        xp = space.env.xp

        space.F = function(space.X.squeeze(), xp)
        

        improved_mask = space.F < self.local_fit
      
        xp.copyto(self.local_position, space.X, where=improved_mask[:, None])
        xp.minimum(self.local_fit, space.F, out=self.local_fit)

        best_idx = xp.argmin(self.local_fit)
        best_fit_val = self.local_fit[best_idx]

        improved_global = best_fit_val < self.global_best_fit
        self.global_best_position = xp.where(
            improved_global, self.local_position[best_idx].reshape(-1,1), self.global_best_position
        )
       
        self.global_best_fit = xp.where(improved_global, best_fit_val, self.global_best_fit)

        space.best_agent.position = self.global_best_position.reshape(space.best_agent.position.shape)
        space.best_agent.fit = self.global_best_fit
        space.best_agent.ts = int(time.time())


    def update(self, space: _SingleObjectiveTensorSpace) -> None:
        xp = space.env.xp
        n_agents = space.n_agents
        dtype = self._dtype

        r1_buf = xp.random.uniform(0.0, 1.0, n_agents, dtype=(dtype if (dtype == 'float64') else xp.float32))
        r2_buf = xp.random.uniform(0.0, 1.0, n_agents, dtype=(dtype if (dtype == 'float64') else xp.float32))
        gbest_flat = self.global_best_position.ravel()

        scalar_type = np.dtype(dtype).type

        self._vel_kernel(
            (self._kernel_grid,), (self._block_size,),
            (self.velocity,
             self.local_position,
             space.X,
             gbest_flat,
             r1_buf,
             r2_buf,
             scalar_type(self.w),
             scalar_type(self.c1),
             scalar_type(self.c2),
             np.int32(self._vars_dims),
             np.int32(self._kernel_N))
        )
        

        space.X += self.velocity

        
       

    

class PSOTensor(Optimizer, TensorizedOptimizer):
    """
        GPU/CPU-agnostic, fully tensorized implementation of PSO (single-objective),
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        logger.info("Overriding class: Optimizer -> PSOTensor.")
        super().__init__()

        self.w = 0.7
        self.c1 = 1.7
        self.c2 = 1.7

        self.position = None
        self.velocity = None
        self.local_position = None
        self.fit = None
        self.local_fit = None
        self.global_best_position = None
        self.global_best_fit = None

        
        self._tmp_cognitive = None
        self._tmp_social = None
        self._r1 = None
        self._r2 = None

        self.build(params)

        logger.info("Class overrided.")

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, w: float) -> None:
        if not isinstance(w, (float, int)):
            raise e.TypeError("`w` should be a float or integer")
        if w < 0:
            raise e.ValueError("`w` should be >= 0")
        self._w = w

    @property
    def c1(self) -> float:
        return self._c1

    @c1.setter
    def c1(self, c1: float) -> None:
        if not isinstance(c1, (float, int)):
            raise e.TypeError("`c1` should be a float or integer")
        if c1 < 0:
            raise e.ValueError("`c1` should be >= 0")
        self._c1 = c1

    @property
    def c2(self) -> float:
        return self._c2

    @c2.setter
    def c2(self, c2: float) -> None:
        if not isinstance(c2, (float, int)):
            raise e.TypeError("`c2` should be a float or integer")
        if c2 < 0:
            raise e.ValueError("`c2` should be >= 0")
        self._c2 = c2

    @property
    def local_position(self):
        return self._local_position

    @local_position.setter
    def local_position(self, local_position) -> None:
        self._local_position = local_position

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity) -> None:
        self._velocity = velocity

    def compile(self, space: _SingleObjectiveTensorSpace) -> None:
        xp = space.env.xp
        dtype = space.env.dtype
        n_a = space.n_agents
        n_v = space.n_variables

        self.local_position = space.X.copy()  # (n_a, n_v)
        self.velocity = xp.zeros((n_a, n_v), dtype=dtype)

        self.local_fit = xp.full(n_a, xp.inf, dtype=dtype)

        self.global_best_position = xp.zeros((n_v, 1), dtype=dtype)
        self.global_best_fit = xp.asarray(xp.inf, dtype=dtype)

        self._lb = xp.asarray(space.lb, dtype=dtype).reshape(1, n_v)
        self._ub = xp.asarray(space.ub, dtype=dtype).reshape(1, n_v)

        self._dtype = dtype

       
        self._tmp_cognitive = xp.empty((n_a, n_v), dtype=dtype)
        self._tmp_social = xp.empty((n_a, n_v), dtype=dtype)
        self._r1 = xp.empty((n_a, 1), dtype=dtype)
        self._r2 = xp.empty((n_a, 1), dtype=dtype)
        self._gbest_row = xp.empty((1, n_v), dtype=dtype)  

        self._warmup(space)

    def _warmup(self, space: _SingleObjectiveTensorSpace) -> None:
        xp = space.env.xp
        n_a = space.n_agents
        dtype = self._dtype

        dummy_X = self.local_position.copy()
        dummy_local_position = self.local_position.copy()
        dummy_local_fit = self.local_fit.copy()
        dummy_F = xp.zeros(n_a, dtype=dtype)

        improved_mask = dummy_F < dummy_local_fit
        xp.copyto(dummy_local_position, dummy_X, where=improved_mask[:, None])
        xp.minimum(dummy_local_fit, dummy_F, out=dummy_local_fit)

        best_idx = xp.argmin(dummy_local_fit)
        best_fit_val = dummy_local_fit[best_idx]

        dummy_gbest_pos = self.global_best_position.copy()
        dummy_gbest_fit = self.global_best_fit.copy()

        improved_global = best_fit_val < dummy_gbest_fit
        candidate = dummy_local_position[best_idx].reshape(-1, 1)
        dummy_gbest_pos = xp.where(improved_global, candidate, dummy_gbest_pos)
        dummy_gbest_fit = xp.where(improved_global, best_fit_val, dummy_gbest_fit)

        xp.copyto(self._r1, xp.random.uniform(0.0, 1.0, self._r1.shape).astype(dtype))
        xp.copyto(self._r2, xp.random.uniform(0.0, 1.0, self._r2.shape).astype(dtype))
        xp.copyto(self._gbest_row, dummy_gbest_pos.reshape(1, -1))

        xp.subtract(dummy_local_position, dummy_X, out=self._tmp_cognitive)
        self._tmp_cognitive *= self._r1
        self._tmp_cognitive *= self.c1

        xp.subtract(self._gbest_row, dummy_X, out=self._tmp_social)
        self._tmp_social *= self._r2
        self._tmp_social *= self.c2

        dummy_velocity = xp.zeros_like(self.velocity)
        dummy_velocity *= self.w
        dummy_velocity += self._tmp_cognitive
        dummy_velocity += self._tmp_social

        dummy_X += dummy_velocity
        xp.clip(dummy_X, self._lb, self._ub, out=dummy_X)

        #if hasattr(xp, "cuda"):
            #xp.cuda.Stream.null.synchronize()

    def evaluate(self, space: _SingleObjectiveTensorSpace, function: Function) -> None:
        xp = space.env.xp

        space.F = function(space.X.squeeze(), xp)

        improved_mask = space.F < self.local_fit

        xp.copyto(self.local_position, space.X, where=improved_mask[:, None])
        xp.minimum(self.local_fit, space.F, out=self.local_fit)

        best_idx = xp.argmin(self.local_fit)
        best_fit_val = self.local_fit[best_idx]

        improved_global = best_fit_val < self.global_best_fit
        candidate = self.local_position[best_idx].reshape(-1, 1)
        self.global_best_position = xp.where(improved_global, candidate, self.global_best_position)
        self.global_best_fit = xp.where(improved_global, best_fit_val, self.global_best_fit)

        space.best_agent.position = self.global_best_position.reshape(space.best_agent.position.shape)
        space.best_agent.fit = self.global_best_fit
        #space.best_agent.ts = int(time.time())

    def update(self, space: _SingleObjectiveTensorSpace) -> None:
        xp = space.env.xp

        # Fill pre-allocated random buffers in-place 
        xp.copyto(self._r1, xp.random.uniform(0.0, 1.0, self._r1.shape).astype(self._dtype))
        xp.copyto(self._r2, xp.random.uniform(0.0, 1.0, self._r2.shape).astype(self._dtype))

        # gbest as a (1, n_v) row, written into a reused buffer
        xp.copyto(self._gbest_row, self.global_best_position.reshape(1, -1))

        # cognitive = c1 * r1 * (local_position - X) 
        xp.subtract(self.local_position, space.X, out=self._tmp_cognitive)
        self._tmp_cognitive *= self._r1
        self._tmp_cognitive *= self.c1

        # social = c2 * r2 * (gbest - X) 
        xp.subtract(self._gbest_row, space.X, out=self._tmp_social)
        self._tmp_social *= self._r2
        self._tmp_social *= self.c2

        # velocity = w*velocity + cognitive + social   
        self.velocity *= self.w
        self.velocity += self._tmp_cognitive
        self.velocity += self._tmp_social

        space.X += self.velocity

       

class AIWPSO(PSO):
    """An AIWPSO class, inherited from PSO.

    This is the designed class to define AIWPSO-related
    variables and methods.

    References:
        A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh.
        A novel particle swarm optimization algorithm with adaptive inertia weight.
        Applied Soft Computing (2011).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: PSO -> AIWPSO.")

        self.w_min = 0.1
        self.w_max = 0.9

        super(AIWPSO, self).__init__(params)

        logger.info("Class overrided.")

    @property
    def w_min(self) -> float:
        """Minimum inertia weight."""

        return self._w_min

    @w_min.setter
    def w_min(self, w_min: float) -> None:
        if not isinstance(w_min, (float, int)):
            raise e.TypeError("`w_min` should be a float or integer")
        if w_min < 0:
            raise e.ValueError("`w_min` should be >= 0")

        self._w_min = w_min

    @property
    def w_max(self) -> float:
        """Maximum inertia weight."""

        return self._w_max

    @w_max.setter
    def w_max(self, w_max: float) -> None:
        if not isinstance(w_max, (float, int)):
            raise e.TypeError("`w_max` should be a float or integer")
        if w_max < 0:
            raise e.ValueError("`w_max` should be >= 0")
        if w_max < self.w_min:
            raise e.ValueError("`w_max` should be >= `w_min`")

        self._w_max = w_max

    @property
    def fitness(self) -> List[float]:
        """List of fitnesses."""

        return self._fitness

    @fitness.setter
    def fitness(self, fitness: List[float]) -> None:
        if not isinstance(fitness, list):
            raise e.TypeError("`fitness` should be a list")

        self._fitness = fitness

    def _compute_success(self, agents: List[Agent]) -> None:
        """Computes the particles' success for updating inertia weight (eq. 16).

        Args:
            agents: List of agents.

        """

        p = 0

        for i, agent in enumerate(agents):
            if agent.fit < self.fitness[i]:
                p += 1

            self.fitness[i] = agent.fit

        self.w = (self.w_max - self.w_min) * (p / len(agents)) + self.w_min

    def update(self, space: _SingleObjectiveSpace, iteration: int) -> None:
        """Wraps Adaptive Inertia Weight Particle Swarm Optimization over all agents and variables.

        Args:
            space: Union[_SingleObjectiveSpace, _MultiObjectiveSpace] containing agents and update-related information.
            iteration: Current iteration.

        """

        if iteration == 0:
            self.fitness = [agent.fit for agent in space.agents]

        for i, agent in enumerate(space.agents):
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            agent.position += self.velocity[i]

        self._compute_success(space.agents)


class RPSO(PSO):
    """An RPSO class, inherited from Optimizer.

    This is the designed class to define RPSO-related
    variables and methods.

    References:
        M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi and J. P. Papa.
        Harnessing Particle Swarm Optimization Through Relativistic Velocity.
        IEEE Congress on Evolutionary Computation (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: PSO -> RPSO.")

        super(RPSO, self).__init__(params)

        logger.info("Class overrided.")

    @property
    def mass(self) -> np.ndarray:
        """Array of masses."""

        return self._mass

    @mass.setter
    def mass(self, mass: np.ndarray) -> None:
        if not isinstance(mass, np.ndarray):
            raise e.TypeError("`mass` should be a numpy array")

        self._mass = mass

    def compile(self, space: _SingleObjectiveSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.mass = r.generate_uniform_random_number(
            size=(space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: _SingleObjectiveSpace) -> None:
        """Wraps Relativistic Particle Swarm Optimization over all agents and variables.

        Args:
            space: Union[_SingleObjectiveSpace, _MultiObjectiveSpace] containing agents and update-related information.

        """

        max_velocity = np.max(self.velocity)

        for i, agent in enumerate(space.agents):
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates current agent velocity (eq. 11)
            gamma = 1 / np.sqrt(1 - (max_velocity**2 / c.LIGHT_SPEED**2))
            self.velocity[i] = (
                self.mass[i] * self.velocity[i] * gamma
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            agent.position += self.velocity[i]


class SAVPSO(PSO):
    """An SAVPSO class, inherited from Optimizer.

    This is the designed class to define SAVPSO-related
    variables and methods.

    References:
        H. Lu and W. Chen.
        Self-adaptive velocity particle swarm optimization for solving constrained optimization problems.
        Journal of global optimization (2008).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: PSO -> SAVPSO.")

        super(SAVPSO, self).__init__(params)

        logger.info("Class overrided.")

    def update(self, space: _SingleObjectiveSpace) -> None:
        """Wraps Self-adaptive Velocity Particle Swarm Optimization over all agents and variables.

        Args:
            space: Union[_SingleObjectiveSpace, _MultiObjectiveSpace] containing agents and update-related information.

        """

        positions = np.zeros(
            (space.agents[0].position.shape[0], space.agents[0].position.shape[1])
        )

        for agent in space.agents:
            positions += agent.position
        positions /= len(space.agents)

        for i, agent in enumerate(space.agents):
            idx = r.generate_integer_random_number(0, len(space.agents))

            # Updates current agent's velocity (eq. 8)
            r1 = r.generate_uniform_random_number()
            self.velocity[i] = (
                self.w
                * np.fabs(self.local_position[idx] - self.local_position[i])
                * np.sign(self.velocity[i])
                + r1 * (self.local_position[i] - agent.position)
                + (1 - r1) * (space.best_agent.position - agent.position)
            )

            agent.position += self.velocity[i]

            for j in range(agent.n_variables):
                r4 = r.generate_uniform_random_number(0, 1)

                if agent.position[j] > agent.ub[j]:
                    agent.position[j] = positions[j] + 1 * r4 * (
                        agent.ub[j] - positions[j]
                    )

                if agent.position[j] < agent.lb[j]:
                    agent.position[j] = positions[j] + 1 * r4 * (
                        agent.lb[j] - positions[j]
                    )


class VPSO(PSO):
    """A VPSO class, inherited from Optimizer.

    This is the designed class to define VPSO-related
    variables and methods.

    References:
        W.-P. Yang. Vertical particle swarm optimization algorithm and its application in soft-sensor modeling.
        International Conference on Machine Learning and Cybernetics (2007).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: PSO -> VPSO.")

        super(VPSO, self).__init__(params)

        logger.info("Class overrided.")

    @property
    def v_velocity(self) -> np.ndarray:
        """Array of vertical velocities."""

        return self._v_velocity

    @v_velocity.setter
    def v_velocity(self, v_velocity: np.ndarray) -> None:
        if not isinstance(v_velocity, np.ndarray):
            raise e.TypeError("`v_velocity` should be a numpy array")

        self._v_velocity = v_velocity

    def compile(self, space: _SingleObjectiveSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.v_velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: _SingleObjectiveSpace) -> None:
        """Wraps Vertical Particle Swarm Optimization over all agents and variables.

        Args:
            space: Union[_SingleObjectiveSpace, _MultiObjectiveSpace] containing agents and update-related information.

        """

        for i, agent in enumerate(space.agents):
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates current agent velocity (eq. 3)
            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            # Updates current agent vertical velocity (eq. 4)
            self.v_velocity[i] -= (
                np.dot(self.velocity[i].T, self.v_velocity[i])
                / (np.dot(self.velocity[i].T, self.velocity[i]) + c.EPSILON)
            ) * self.velocity[i]

            # Updates current agent position (eq. 5)
            r1 = r.generate_uniform_random_number()
            agent.position += r1 * self.velocity[i] + (1 - r1) * self.v_velocity[i]
