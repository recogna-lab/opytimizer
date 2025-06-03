import numpy as np


def arithmetic_crossover(parent1, parent2, crossover_rate=0.9):
    """
    Arithmetic crossover for real-valued vectors.

    For each variable, with probability `crossover_rate`,
    a linear combination of the parents' values is performed using a random alpha.
    Otherwise, the value is inherited from one of the parents.

    Args:
        parent1 (np.ndarray): First parent vector.
        parent2 (np.ndarray): Second parent vector.
        crossover_rate (float): Probability of crossover for each variable.

    Returns:
        tuple: Two children vectors (np.ndarray).
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    for j in range(len(parent1)):
        if np.random.random() < crossover_rate:
            alpha = np.random.random()
            child1[j] = alpha * parent1[j] + (1 - alpha) * parent2[j]
            child2[j] = alpha * parent2[j] + (1 - alpha) * parent1[j]
    return child1, child2


def sbx_crossover(parent1, parent2, lb, ub, crossover_rate=0.9, eta=20):
    """
    Simulated Binary Crossover (SBX) for real-valued vectors (Deb, 1995, 2002).

    Applies SBX with probability `crossover_rate` (per individual). Uses bounds for each gene.
    For each gene, if parents differ, applies SBX; otherwise, swaps genes. Children are clipped to bounds.

    Args:
        parent1 (np.ndarray): First parent vector.
        parent2 (np.ndarray): Second parent vector.
        lb (np.ndarray): Lower bounds for each gene.
        ub (np.ndarray): Upper bounds for each gene.
        crossover_rate (float): Probability of crossover per individual.
        eta (float): Distribution index for SBX.

    Returns:
        tuple: Two children vectors (np.ndarray).
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    if np.random.rand() <= crossover_rate:
        for i in range(parent1.size):
            x1 = parent1[i]
            x2 = parent2[i]
            if np.random.rand() <= 0.5:
                if abs(x1 - x2) > 1e-10:
                    y1, y2 = (x1, x2) if x1 < x2 else (x2, x1)
                    lbi = lb[i]
                    ubi = ub[i]
                    r = np.random.rand()

                    # Child 1
                    beta = 1.0 + (2.0 * (y1 - lbi) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    if r <= 1.0 / alpha:
                        betaq = (r * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - r * alpha)) ** (1.0 / (eta + 1.0))
                    c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))

                    # Child 2
                    beta = 1.0 + (2.0 * (ubi - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    if r <= 1.0 / alpha:
                        betaq = (r * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - r * alpha)) ** (1.0 / (eta + 1.0))
                    c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                    # Clipping
                    c1 = np.clip(c1, lbi, ubi)
                    c2 = np.clip(c2, lbi, ubi)

                    # Randomly assign
                    if np.random.rand() <= 0.5:
                        child1[i], child2[i] = c2, c1
                    else:
                        child1[i], child2[i] = c1, c2
                else:
                    child1[i], child2[i] = x2, x1
    return child1, child2


def one_point_crossover(parent1, parent2, lb=None, ub=None, crossover_rate=0.9):
    """
    One-point crossover for binary or real-valued vectors.

    With probability `crossover_rate`, selects a random crossover point and swaps the tails of the parents.
    If bounds are provided, clips the children to the allowed range.

    Args:
        parent1 (np.ndarray): First parent vector.
        parent2 (np.ndarray): Second parent vector.
        lb (np.ndarray or None): Lower bounds for each gene (optional).
        ub (np.ndarray or None): Upper bounds for each gene (optional).
        crossover_rate (float): Probability of crossover per individual.

    Returns:
        tuple: Two children vectors (np.ndarray).
    """
    parent1=parent1.flatten()
    parent2=parent2.flatten()
    child1 = parent1.copy()
    child2 = parent2.copy()
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))  # ponto de corte não pode ser 0
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        if lb is not None and ub is not None:
            child1 = np.clip(child1, lb, ub)
            child2 = np.clip(child2, lb, ub)
    return child1.reshape(-1,1), child2.reshape(-1,1)


def gaussian_mutation(vector, mutation_rate=0.1, std=0.1):
    """
    Gaussian mutation for real-valued vectors.

    For each variable, with probability `mutation_rate`,
    a Gaussian noise (mean 0, std `std`) is added to the variable value.

    Args:
        vector (np.ndarray): Vector to be mutated.
        mutation_rate (float): Probability of mutation for each variable.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Mutated vector.
    """
    mutant = vector.copy()
    for j in range(len(vector)):
        if np.random.random() < mutation_rate:
            mutant[j] += np.random.normal(0, std)
    return mutant


def bit_flip_mutation(vector, mutation_rate=0.01, **kwargs):
    """
    Bit flip mutation for binary vectors.

    For each variable, with probability `mutation_rate`,
    the bit is flipped (0 -> 1 or 1 -> 0).

    Args:
        vector (np.ndarray): Binary vector to be mutated.
        mutation_rate (float): Probability of mutation for each bit.

    Returns:
        np.ndarray: Mutated binary vector.
    """
    mutant = vector.copy()
    for j in range(len(vector)):
        if np.random.random() < mutation_rate:
            mutant[j] = 1 - mutant[j]
    return mutant


def polynomial_mutation(vector, lb, ub, mutation_rate=0.1, eta=20):
    """
    Polynomial mutation for real-valued vectors (Deb, 1996).

    For each variable, with probability `mutation_rate`, applies polynomial mutation
    using the distribution index `eta` and respecting the lower and upper bounds.

    Args:
        vector (np.ndarray): Vector to be mutated.
        lb (np.ndarray): Lower bounds for each gene.
        ub (np.ndarray): Upper bounds for each gene.
        mutation_rate (float): Probability of mutation for each variable.
        eta (float): Distribution index for the polynomial mutation.

    Returns:
        np.ndarray: Mutated vector.
    """
    mutant = vector.copy()
    for i in range(len(mutant)):
        if np.random.rand() < mutation_rate:
            x = mutant[i]
            lbi, ubi = lb[i], ub[i]
            delta1 = (x - lbi) / (ubi - lbi) if ubi > lbi else 0.0
            delta2 = (ubi - x) / (ubi - lbi) if ubi > lbi else 0.0
            u = np.random.rand()
            if u <= 0.5:
                delta_q = (2*u + (1 - 2*u) * (1 - delta1)**(eta + 1)) ** (1/(eta + 1)) - 1
            else:
                delta_q = 1 - (2*(1 - u) + 2*(u - 0.5)*(1 - delta2)**(eta + 1)) ** (1/(eta + 1))
            mutant[i] = x + delta_q * (ubi - lbi)
            mutant[i] = np.clip(mutant[i], lbi, ubi)
    return mutant 