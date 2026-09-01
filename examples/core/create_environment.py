from opytimizer.core import Environment

# CPU-based Environment 
cpu_env = Environment().set_backend('numpy').set_dtype('float64')

# CUDA-based Environment 
cuda_env = Environment().set_backend('cupy').set_dtype('float64')