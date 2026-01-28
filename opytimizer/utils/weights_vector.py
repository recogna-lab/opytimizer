import numpy as np
import math

def das_dennis(n_objectives: int, n_partitions: int) -> tuple:
    """
    Generates reference direction vectors (weight vectors) for Decomposition-based MHs
    using the Das-Dennis.

   
    
    Complexity: O(N), where N is the final number of weight vectors.

    Args:
        n_objectives (int): The dimensionality of the objective space (M).
        n_partitions (int): The number of divisions/partitions per objective axis (p).

    Returns:
        tuple:
            - weights (np.ndarray): A matrix of shape (N, M) where each row is a normalized weight vector.
            - n_points (int): The total number of generated vectors.
    """
    # Eege case handling
    if n_partitions == 0:
        return np.full((1, n_objectives), 1.0 / n_objectives), 1

    
    n_points = math.comb(n_partitions + n_objectives - 1, n_objectives - 1)

    
    # we allocate zeros once. No dynamic list appending or resizing occurs.
    weights = np.zeros((n_points, n_objectives))

    #  recursive function to fill the matrix "in-place"
    def _fill_weights(current_obj_idx, remaining_sum, start_row):
        """
        Internal recursive helper.
        
        Args:
            current_obj_idx: The column (objective) index we are currently filling (0 to M-1).
            remaining_sum: How many 'units' are left to distribute among remaining objectives.
            start_row: The row index in the 'weights' matrix where we start writing.
        """
        # Base Case: We are at the last objective (last column)
        if current_obj_idx == n_objectives - 1:
            # The last column strictly takes whatever sum is left to satisfy the partition count.
            # Due to the recursive structure, this single operation fills a specific slot.
            weights[start_row, current_obj_idx] = remaining_sum
            return

        
        current_row = start_row
        for value in range(remaining_sum + 1):
            
         
            next_objs = n_objectives - (current_obj_idx + 1)
            next_sum = remaining_sum - value
            
           
            num_combinations = math.comb(next_sum + next_objs - 1, next_objs - 1)

            if num_combinations > 0:
               
                weights[current_row : current_row + num_combinations, current_obj_idx] = value
                
               
                _fill_weights(current_obj_idx + 1, next_sum, current_row)
                
                #
                current_row += num_combinations

    # start the recursion from the first objective
    _fill_weights(0, n_partitions, 0)

    
    # divide by n_partitions so that the sum of every row equals 1.0
    weights = weights / n_partitions

    return weights, n_points