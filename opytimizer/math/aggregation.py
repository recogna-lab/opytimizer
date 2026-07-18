from abc import  ABC, abstractmethod
class _BaseAggregation(ABC):

    @abstractmethod
    def __call__(self, obj_values, ref_vectors, xp, **kwargs):
        pass 


class WeightedSum(_BaseAggregation):
    def __call__(self, obj_values, ref_vectors, xp, **kwargs):
        
        obj_values_ = xp.atleast_2d(obj_values)
        ref_vectors_ = xp.atleast_2d(ref_vectors)

        result = xp.sum(ref_vectors_ * obj_values_, axis=1)
        return result.squeeze() if obj_values.ndim == 1 else result

class Tchebycheff(_BaseAggregation):

    def __call__(self, obj_values, ref_vectors, xp, **kwargs):
        z = kwargs.get('z')
        z_ = xp.atleast_2d(z)
        obj_values_ = xp.atleast_2d(obj_values)
        ref_vectors_ = xp.atleast_2d(ref_vectors)
        diff = xp.abs(obj_values_ - z_)
        scaled_diff = ref_vectors_ * diff
        result = xp.max(scaled_diff, axis=1)

        return result.squeeze() if obj_values.ndim == 1 else result


class PBI(_BaseAggregation):
    
    def __init__(self, theta: float = 5.0):
        self.theta = theta

    def __call__(self, obj_values, ref_vectors, xp, **kwargs):
        z = kwargs.get('z')
       
        obj_values_ = xp.atleast_2d(obj_values)
        ref_vectors_ = xp.atleast_2d(ref_vectors)
        z_ = xp.atleast_2d(z)

        d = obj_values_ - z_
        norm_w = xp.linalg.norm(ref_vectors_, axis=1, keepdims=True)
        norm_w = xp.where(norm_w == 0, 1e-10, norm_w)

        d1 = xp.sum(d * ref_vectors_, axis=1, keepdims=True) / norm_w
        projection = d1 * (ref_vectors_ / norm_w)
        d2 = xp.linalg.norm(d - projection, axis=1)

        result = d1.squeeze(axis=1) + self.theta * d2

        return result.squeeze() if obj_values.ndim == 1 else result



