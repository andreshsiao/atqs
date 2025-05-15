import unittest
import numpy as np
from taq.ExecutionCostModel import AlmgrenChrissExecution

class TestAlmgrenChrissExecution(unittest.TestCase):
    def setUp(self):
        self.model = AlmgrenChrissExecution(T=1.0, N=13, X=1000, eta=0.142, beta=0.5, sigma=0.02, lam=0.01)

    def test_optimal_trajectory_shape(self):
        traj = self.model.optimal_trajectory()
        self.assertEqual(len(traj), 14)
        self.assertAlmostEqual(traj[0], self.model.X, delta=1e-5)
        self.assertAlmostEqual(traj[-1], 0.0, delta=1e-5)

    def test_cost_computation(self):
        traj = self.model.optimal_trajectory()
        E, V = self.model.compute_cost(traj)
        self.assertTrue(E > 0)
        self.assertTrue(V > 0)

    def test_total_objective_value(self):
        traj = self.model.optimal_trajectory()
        total = self.model.total_objective(traj)
        self.assertTrue(total > 0)

if __name__ == '__main__':
    unittest.main()
