from opytimizer.core import Environment

# CPU-based Environment 
cpu_env = Environment().set_backend('cpu').set_dtype('float64')

# CUDA-based Environment 
cuda_env = Environment().set_backend('cuda').set_dtype('float64')