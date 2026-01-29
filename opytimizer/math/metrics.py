from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class BaseMetric(ABC):
    
    @property
    def pareto_front(self) -> np.ndarray:
        return self._pareto_front
    
    @pareto_front.setter
    def pareto_front(self, value) -> None:
        if hasattr(value, 'pareto_front'):
            points = np.array([agent.fit for agent in value.pareto_front])
        # agents list
        elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'fit'):
            points = np.array([agent.fit for agent in value])
        else:
             points = np.atleast_2d(value) 
             
        self._pareto_front = points
        
    @abstractmethod
    def __call__(self, pareto_front):
        pass

    @property
    def name(self):
        return type(self).__name__

class IGDMetric(BaseMetric):
    def __init__(self, pareto_optimal):
        self.pareto_optimal = np.atleast_2d(pareto_optimal)
        if self.pareto_optimal.size == 0:
            self.min_vals = None
            self.denom = None
            self.normalized_optimal = None
        else:
            self.min_vals = np.min(self.pareto_optimal, axis=0)
            self.max_vals = np.max(self.pareto_optimal, axis=0)
            self.denom = np.where(self.max_vals - self.min_vals == 0, 1.0, self.max_vals - self.min_vals)
            self.normalized_optimal = (self.pareto_optimal - self.min_vals) / self.denom

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.size == 0 or self.min_vals is None:
            return 0.0
        normalized_front = (self.pareto_front - self.min_vals) / self.denom
        distances = []
        for optimal in self.normalized_optimal:
            d = np.linalg.norm(normalized_front - optimal, axis=1)
            distances.append(np.min(d))
        return np.mean(distances)

class GDMetric(BaseMetric):
    def __init__(self, pareto_optimal):
        self.pareto_optimal = np.atleast_2d(pareto_optimal)
        if self.pareto_optimal.size == 0:
            self.min_vals = None
            self.denom = None
            self.normalized_optimal = None
        else:
            self.min_vals = np.min(self.pareto_optimal, axis=0)
            self.max_vals = np.max(self.pareto_optimal, axis=0)
            self.denom = np.where(self.max_vals - self.min_vals == 0, 1.0, self.max_vals - self.min_vals)
            self.normalized_optimal = (self.pareto_optimal - self.min_vals) / self.denom

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.size == 0 or self.min_vals is None:
            return 0.0
        normalized_front = (self.pareto_front - self.min_vals) / self.denom
        distances = []
        for solution in normalized_front:
            d = np.linalg.norm(self.normalized_optimal - solution, axis=1)
            distances.append(np.min(d))
        return np.mean(distances)


class SpreadMetric(BaseMetric):
    def __init__(self, pareto_optimal):
        self.pareto_optimal = np.atleast_2d(pareto_optimal)
        if self.pareto_optimal.size == 0:
            self.min_vals = None
            self.denom = None
            self.normalized_optimal = None
        else:
            self.min_vals = np.min(self.pareto_optimal, axis=0)
            self.max_vals = np.max(self.pareto_optimal, axis=0)
            self.denom = np.where(self.max_vals - self.min_vals == 0, 1.0, self.max_vals - self.min_vals)
            self.normalized_optimal = (self.pareto_optimal - self.min_vals) / self.denom

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.shape[0] < 2 or self.min_vals is None:
            return 0.0
        pf = (self.pareto_front - self.min_vals) / self.denom
        pf = pf[np.lexsort(np.rot90(pf))]
        df = np.linalg.norm(pf[1:] - pf[:-1], axis=1)
        d_mean = np.mean(df)
        d_f = np.min(np.linalg.norm(pf[0] - self.normalized_optimal, axis=1))
        d_l = np.min(np.linalg.norm(pf[-1] - self.normalized_optimal, axis=1))
        delta = (d_f + d_l + np.sum(np.abs(df - d_mean))) / (
            d_f + d_l + len(df) * d_mean
        )
        return delta

class ErrorRatioMetric(BaseMetric):
    def __init__(self, pareto_optimal, tol=1e-6):
        self.pareto_optimal = np.atleast_2d(pareto_optimal)
        self.tol = tol
        if self.pareto_optimal.size == 0:
            self.min_vals = None
            self.denom = None
            self.normalized_optimal = None
        else:
            self.min_vals = np.min(self.pareto_optimal, axis=0)
            self.max_vals = np.max(self.pareto_optimal, axis=0)
            self.denom = np.where(self.max_vals - self.min_vals == 0, 1.0, self.max_vals - self.min_vals)
            self.normalized_optimal = (self.pareto_optimal - self.min_vals) / self.denom

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.size == 0 or self.min_vals is None:
            return 0.0
        pf = (self.pareto_front - self.min_vals) / self.denom
        errors = 0
        for solution in pf:
            distances = np.linalg.norm(self.normalized_optimal - solution, axis=1)
            if np.min(distances) > self.tol:
                errors += 1
        return errors / len(pf)

class R2Metric(BaseMetric):
    def __init__(self, weight_vectors, ideal_point=None, nadir_point=None):
        self.weight_vectors = np.atleast_2d(weight_vectors)
        norms = np.linalg.norm(self.weight_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.weight_vectors = self.weight_vectors / norms
        self.ideal_point = ideal_point
        self.nadir_point = nadir_point

    def __call__(self, pareto_front):
        
        self.pareto_front = pareto_front
        if self.pareto_front.size == 0:
            return 0.0
        current_ideal = self.ideal_point if self.ideal_point is not None else np.min(self.pareto_front, axis=0)
        current_nadir = self.nadir_point if self.nadir_point is not None else np.max(self.pareto_front, axis=0)
        denom = np.where(current_nadir - current_ideal == 0, 1.0, current_nadir - current_ideal)
        normalized_pf = (self.pareto_front - current_ideal) / denom
        r2_values = []
        for w in self.weight_vectors:
            weighted_diff = normalized_pf * w
            tchebycheff = np.max(weighted_diff, axis=1)
            r2_values.append(np.min(tchebycheff))
        return np.mean(r2_values)

class MaximumSpreadMetric(BaseMetric):
    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.shape[0] < 2:
            return 0.0
        dists = np.linalg.norm(
            self.pareto_front[None, :, :] - self.pareto_front[:, None, :], axis=2
        )
        return np.max(dists)


class HypervolumeMetric(BaseMetric):
    def __init__(self, reference_point):
        self.reference_point = np.asarray(reference_point)

    def __call__(self, pareto_front):
        
        self.pareto_front = pareto_front
       
        # Validation and filtering
        if self.pareto_front.size == 0:
            return 0.0
            
        n_points, n_dims = self.pareto_front.shape
        
       
        if n_points <= 1:
            return 0.0
        # remove points worse than reference point
        is_worse = np.any(self.pareto_front > self.reference_point, axis=1)
        self.pareto_front= self.pareto_front[~is_worse]
        
        
        if self.pareto_front.shape[0] < 1:
            return 0.0

        # Dispatch based on dimensions
        if n_dims == 2:
            return self._calculate_hv2d(self.pareto_front)
        elif n_dims == 3:
            return self._calculate_hv3d(self.pareto_front)
        else:
            return 0.0 # >3D requires more complex algorithms 

    def _calculate_hv2d(self, front):
        """Calculates HV for 2D using sorting."""
        # sort by X axis
        sorted_indices = np.argsort(front[:, 0])
        sorted_front = front[sorted_indices]

        # filter dominated points to get a clean step-function
        non_dominated = []
        if len(sorted_front) > 0:
            non_dominated.append(sorted_front[0])
            for point in sorted_front[1:]:
                # Only add if better Y than the last added point
                if point[1] < non_dominated[-1][1]:
                    non_dominated.append(point)
        
        sorted_front = np.array(non_dominated)

        hv = 0.0
        prev_x = self.reference_point[0]
        ref_y = self.reference_point[1]

        # largest X to smallest X
        for point in reversed(sorted_front):
            width = prev_x - point[0]
            height = ref_y - point[1]
            
            if width > 0 and height > 0:
                hv += width * height
            
            prev_x = point[0]
            
        return hv

    def _calculate_hv3d(self, front):
        """Calculates HV for 3D using the slicing (integration) method."""
        # Sort by Z axis
        sorted_indices = np.argsort(front[:, 2])
        sorted_front = front[sorted_indices]
        
        volume = 0.0
        
        #  Iterate through Z slices
        for i in range(len(sorted_front)):
            z_curr = sorted_front[i, 2]
            
            # Determine Z height for this slice
            if i < len(sorted_front) - 1:
                z_next = sorted_front[i+1, 2]
            else:
                z_next = self.reference_point[2] 
            
            height = z_next - z_curr
            
            if height <= 0:
                continue
                
           # Project accumulated points (0 to i) onto 2D plane (X, Y)
            points_2d = sorted_front[0 : i+1, 0 : 2]
            
            # Calculate Area of the union of these 2D rectangles
            area = self._calculate_hv2d_projection(points_2d, self.reference_point[:2])
            
            # Add slice volume
            volume += area * height
            
        return volume

    def _calculate_hv2d_projection(self, points, ref_point):
        """Helper to calculate 2D area with explicit reference point."""
        sorted_indices = np.argsort(points[:, 0])
        sorted_front = points[sorted_indices]

        # filter non-dominated points (
        # 3D points projected to 2D might dominate each other
        non_dominated = []
        if len(sorted_front) > 0:
            non_dominated.append(sorted_front[0])
            for point in sorted_front[1:]:
                if point[1] < non_dominated[-1][1]:
                    non_dominated.append(point)
        
        sorted_front = np.array(non_dominated)
        
        area = 0.0
        prev_x = ref_point[0]
        ref_y = ref_point[1]
        
        # Calculate area
        for point in reversed(sorted_front):
            width = prev_x - point[0]
            height = ref_y - point[1]
            if width > 0 and height > 0:
                area += width * height
            prev_x = point[0]
            
        return area
    
    
    
def _check_dominance_worker(samples_chunk, pareto_front):
  
    is_dominated = np.zeros(len(samples_chunk), dtype=bool)
    
    for pareto_point in pareto_front:
       
        remaining = ~is_dominated
        if not np.any(remaining):
            break
            
        diff = pareto_point - samples_chunk[remaining]
        domination_mask = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
        is_dominated[remaining] = domination_mask
        
    return np.sum(is_dominated)
    
class MonteCarloHypervolumeMetric(BaseMetric):
    def __init__(self, n_samples: int = 10**6, 
                 lower_bound: Union[float, List[float]] = 0.0, 
                 upper_bound: Union[float, List[float]] = 1.0, 
                 parallel: bool = False, 
                 diff: float = 0.01):
        
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.n_vars = len(self.lower_bound) if self.lower_bound.ndim > 0 else 1
        self.n_samples = int(n_samples)
        self.diff = diff
        
        self.samples = np.random.uniform(
            low=self.lower_bound, 
            high=self.upper_bound, 
            size=(self.n_samples, self.n_vars)
        )
        
        self._exec = self._process_parallel if parallel else self._process_serial
        
        
    def _calculate_final_hv(self, count):
        rate = count / self.n_samples
        total_volume = np.prod((self.upper_bound + self.diff) - (self.lower_bound - self.diff))
        return rate * total_volume
    
    def _process_serial(self, pareto_front):
        count = _check_dominance_worker(self.samples, pareto_front)
        return self._calculate_final_hv(count)
        
        
                
    
    def _process_parallel(self, pareto_front):
        n_cores = multiprocessing.cpu_count()
        chunks = np.array_split(self.samples, n_cores)
        
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(_check_dominance_worker, chunk, pareto_front) for chunk in chunks]
            total_count = sum(f.result() for f in futures)
            
        return self._calculate_final_hv(total_count)
    
    def __call__(self, pareto_front):
       self.pareto_front = pareto_front
       return self._exec(self.pareto_front) 
   