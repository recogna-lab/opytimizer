from opytimizer.spaces import HyperComplexSpace

# Defines the number of agents, decision variables,
# and search space dimensions
n_agents = 2
n_variables = 5
n_dimensions = 4
n_objectives = 1

# Creates the HyperComplexSpace
s = HyperComplexSpace(
    n_agents=n_agents,
    n_variables=n_variables,
    n_dimensions=n_dimensions,
    n_objectives=n_objectives,
)

# Prints out some properties
print(s.n_agents, s.n_variables, s.n_dimensions, s.n_objectives)
print(s.agents, s.best_agent)
print(s.best_agent.position)
