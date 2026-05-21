from opytimizer.optimizers.multi_objective.evolutionary import RVEA
from opytimizer.utils.reference_vectors import das_dennis

reference_vectors, _ = das_dennis(n_objectives=3, n_partitions=23)

# Creates a RVEA optimizer
o = RVEA(reference_vectors=reference_vectors)
