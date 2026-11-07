import unittest
import numpy as np

from src.parte_iv import solve_lagrange, solve_lp
from src.parte_v import finite_difference_table, conditioning_numeric
from src.parte_vi import analytical_solution, two_batch_updates, batch_gd, minibatch_sgd


class TestSolutions(unittest.TestCase):
    def test_iv_a(self):
        r = solve_lagrange()
        self.assertAlmostEqual(r["x_star"], 4.0)
        self.assertAlmostEqual(r["y_star"], 2.0)
        self.assertAlmostEqual(r["lambda_star"], 8.0)
        self.assertAlmostEqual(r["C_star"], 24.0)
        self.assertAlmostEqual(r["dV_at_6"], 8.0)

    def test_iv_b(self):
        r = solve_lp()
        self.assertEqual(len(r["vertices"]), 5)
        self.assertTrue(r["unique"])
        x, y, z = r["optimum"]
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(z, 10.0)

    def test_v_a_orders(self):
        rows = finite_difference_table()
        self.assertAlmostEqual(rows[1]["p_plus"], 1.0, delta=0.03)
        self.assertAlmostEqual(rows[1]["p_central"], 2.0, delta=0.03)
        self.assertAlmostEqual(rows[2]["p_central"], 2.0, delta=0.03)

    def test_v_b(self):
        r = conditioning_numeric()
        self.assertTrue(r["cond_A"] > 1e8)
        self.assertTrue(r["cond_ridge"] < 1000)
        self.assertTrue(r["rel_x"] > 10)
        self.assertTrue(r["rel_x_ridge"] < 1e-3)

    def test_vi_analytical(self):
        r = analytical_solution()
        self.assertAlmostEqual(r["w_star"], 2.0)
        self.assertAlmostEqual(r["loss_star"], 0.0)

    def test_vi_two_steps(self):
        rows = two_batch_updates()
        self.assertAlmostEqual(rows[0]["loss"], 6.0)
        self.assertAlmostEqual(rows[1]["w"], 0.6)
        self.assertAlmostEqual(rows[1]["loss"], 2.94)
        self.assertAlmostEqual(rows[2]["w"], 1.02)
        self.assertAlmostEqual(rows[2]["loss"], 1.4406)

    def test_vi_training(self):
        w_batch, _ = batch_gd()
        self.assertAlmostEqual(w_batch, 2.0, delta=1e-4)
        for seed in range(5):
            w_sgd, _ = minibatch_sgd(seed)
            self.assertAlmostEqual(w_sgd, 2.0, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
