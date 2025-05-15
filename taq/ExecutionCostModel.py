import numpy as np

# This class implements the Almgren-Chriss optimal execution model.
# It computes the ideal trajectory to minimize trading costs and risk
# when executing a large order over a fixed time horizon.

class AlmgrenChrissExecution:
    # Initializes model parameters including market impact, volatility,
    # and risk aversion. These control how aggressively the model trades.
    def __init__(self, T: float, N: int, X: float, eta: float = 0.142, beta: float = 0.5, sigma: float = 0.02, lam: float = 0.01, daily_volume: float = 1_000_000):
        """
        Args:
            T (float): total trading time (e.g., 1.0 for a full day)
            N (int): number of discrete time steps (e.g., 13)
            X (float): total number of shares to trade
            eta (float): temporary impact coefficient
            beta (float): permanent impact coefficient
            sigma (float): volatility per sqrt(time step)
            lam (float): risk aversion parameter
            daily_volume (float): average daily trading volume for scaling participation rate
        """
        self.T = T
        self.N = N
        self.X = X
        self.eta = eta
        self.beta = beta
        self.sigma = sigma
        self.lam = lam
        self.dt = T / N
        self.daily_volume = daily_volume

    # Computes the optimal trading trajectory over N discrete time steps.
    # This is the theoretical schedule of remaining shares to hold over time,
    # derived from the Almgren-Chriss model using a hyperbolic sine solution.
    def optimal_trajectory(self) -> np.ndarray:
        """
        Computes the optimal execution trajectory using closed-form solution.
        Returns:
            np.ndarray: optimal holdings over time steps (length N+1)
        """
        tau = self.lam * self.sigma**2 / self.eta
        k = np.sqrt(tau)
        t = np.linspace(0, self.T, self.N + 1)
        holdings = self.X * np.sinh(k * (self.T - t)) / np.sinh(k * self.T)
        # [TERMINAL POINT] trajectory starts with full inventory and ends at zero
        return holdings

    # Calculates the expected transaction cost and its variance for
    # a given execution trajectory. Includes both market impact and risk.
    def compute_cost(self, trajectory: np.ndarray) -> tuple:
        """
        Compute expected cost and variance of a given trajectory.

        Args:
            trajectory (np.ndarray): holdings over time (length N+1)

        Returns:
            (float, float): expected cost E[C], variance V[C]
        """
        trading_rate = -(np.diff(trajectory) / self.dt)
        participation_rate = trading_rate / self.daily_volume
        E_C = np.sum(self.eta * participation_rate**2) * self.dt
        V_C = self.sigma**2 * np.sum(trajectory[:-1]**2) * self.dt
        # [PASSING POINT] both expected cost and variance should be positive if trajectory is non-trivial
        return E_C, V_C

    # Combines expected cost and risk into a single objective function
    # that can be used for optimization or evaluation.
    def total_objective(self, trajectory: np.ndarray) -> float:
        """
        Compute total execution cost = E[C] + lambda * V[C]

        Args:
            trajectory (np.ndarray): holdings over time (length N+1)

        Returns:
            float: total cost
        """
        E, V = self.compute_cost(trajectory)
        # [TERMINAL POINT] combines cost and variance into a single objective value for optimization
        return E + self.lam * V