import numpy as np
from qubots.base_problem import BaseProblem

class AckleyProblem(BaseProblem):
    """
    Ackley Function Problem:
      f(x) = -20*exp(-0.2*sqrt((1/n)*sum(x[i]^2))) - exp((1/n)*sum(cos(2*pi*x[i]))) + 20 + e
    Global minimum at x = [0,...,0] with f(x)=0.
    """
    def __init__(self, dim=2):
        self.dim = dim
        self.lower_bound = -32.768
        self.upper_bound = 32.768

    def evaluate_solution(self, solution) -> float:
        x = np.array(solution)
        n = self.dim
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2)/n))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x))/n)
        return float(term1 + term2 + 20 + np.e)

    def random_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim).tolist()