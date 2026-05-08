from opytimizer.core import  MaxIterations, MaxEvaluations, NoImprovement

# 1 stopping condition
stopping_criterion = MaxIterations(100)

# Multiple criteria can be specified 
# The optimization terminates if any criterion is met

stoppping_criteria = [MaxEvaluations(1000), NoImprovement(patience=30, min_delta=10e-6)]

