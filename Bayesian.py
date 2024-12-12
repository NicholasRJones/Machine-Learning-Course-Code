import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm


# Define the noisy torque and thrust functions
def torque(D, n, alpha):
    torque_output = D + alpha / (8 - n)
    noise = np.random.normal(0, 0.2)
    return torque_output + noise


def thrust(D, n, alpha):
    dist_sq = (D - 3) ** 2 + (n - 5) ** 2 + (alpha - 7) ** 2
    thrust_output = 11 + dist_sq / 3 ** 2
    noise = np.random.normal(0, 0.2)
    return thrust_output + noise


def objective(D, n, alpha):
    return n * D * torque(D, n, alpha)


# Feasibility check function
def is_feasible(D, n, alpha):
    return 10 <= thrust(D, n, alpha) <= 12  # Define feasibility constraint


class BayesianOptimizationWithConstraints:
    def __init__(self, bounds, F, I, values):
        self.bounds = bounds
        self.F = np.array(F)  # Set of feasible points
        self.I = np.array(I)  # Set of infeasible points
        self.valuesF = np.array(values[:len(F)])  # Objective values of feasible points
        self.valuesI = np.array(values[len(F):])  # Objective values of infeasible points
        self.best = np.min(self.valuesF)  # Current best objective output
        self.best_input = self.F[np.argmin(self.valuesF)]  # Best input corresponding to the best objective
        self.besthist = [self.best]  # Store history
        self.feasiblehist = [[len(self.F), len(self.I)]]

    def update(self, x, objective_value, is_feasible_point):
        # Update feasible or infeasible points
        if is_feasible_point:
            self.F = np.vstack([self.F, x])
            self.valuesF = np.append(self.valuesF, objective_value)
            # Update best found solution
            if objective_value < self.best:
                self.best = objective_value
                self.best_input = x
        else:
            self.I = np.vstack([self.I, x])
            self.valuesI = np.append(self.valuesI, objective_value)
        self.feasiblehist.append([len(self.F), len(self.I)])
        self.besthist.append(self.best)

    # Define the Expected Improvement (EI) acquisition function
    def expected_improvement(self, x, model, y_min, constraint_model):
        # Predict the mean and variance for the objective and constraint at point x
        mu, sigma = model.predict(x.reshape(1, -1))
        mu_constraint, _ = constraint_model.predict(x.reshape(1, -1))

        # If the point is infeasible, we apply a penalty (we don't optimize in the infeasible region)
        if mu_constraint < 0:  # If constraint is violated, return a high penalty
            return 1e10  # Large penalty to avoid infeasible points

        # Expected Improvement (EI) formula:
        # EI = (y_min - mu) * normal_cdf((y_min - mu) / sigma) + sigma * normal_pdf((y_min - mu) / sigma)
        if sigma > 0:
            z = (y_min - mu) / sigma
            ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # We minimize the negative EI to maximize EI
        else:
            return 0  # No improvement if sigma is zero

    # Acquisition function: Use optimization to find the next point
    def acquire(self, model, constraint_model):
        # Current best objective value from feasible set
        y_min = np.min(self.valuesF)

        # Define bounds for optimization (the same bounds as defined earlier)
        bounds = self.bounds

        # Initial guess (could be the current best or random within bounds)
        x0 = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds])

        # Define the acquisition function to be minimized (negative EI)
        def obj_func(x):
            return self.expected_improvement(x, model, y_min, constraint_model)

        # Optimize the acquisition function using a method like L-BFGS-B
        result = minimize(obj_func, x0, bounds=self.bounds, method='L-BFGS-B')
        # Return the point with the best acquisition value
        return result.x

    def optimize(self, max_iterations=98):
        for iteration in range(max_iterations):
            # Step 1: Build Gaussian Process models for the objective and constraint
            x_train = np.vstack([self.F, self.I])
            y_train = np.hstack([self.valuesF, self.valuesI])

            # Gaussian Process for objective function
            kernel_obj = GPy.kern.Matern32(input_dim=3, lengthscale=1.0) + GPy.kern.White(input_dim=3, variance=0.2)
            model = GPy.models.GPRegression(x_train, y_train.reshape(-1, 1), kernel_obj)
            model.optimize(messages=False)  # Optimize the hyperparameters of the GP model

            # Gaussian Process for constraint function
            y_constraint = np.hstack([np.ones(len(self.F)), np.zeros(len(self.I))])  # 1 for feasible, 0 for infeasible
            kernel_constr = GPy.kern.Matern32(input_dim=3, lengthscale=1.0) + GPy.kern.White(input_dim=3, variance=0.2)
            constraint_model = GPy.models.GPClassification(x_train, y_constraint.reshape(-1, 1), kernel_constr)
            constraint_model.optimize(messages=False)  # Optimize the hyperparameters of the GP constraint model

            # Step 2: Optimize acquisition function
            new_x = self.acquire(model, constraint_model)

            # Step 3: Evaluate the new point
            D, n, alpha = new_x
            objective_value = objective(D, n, alpha)
            feasible = is_feasible(D, n, alpha)

            # Step 4: Update the sets (F, I) based on feasibility
            self.update(new_x, objective_value, feasible)

            print(f"Iteration {iteration + 1}, Objective Value: {objective_value}, Best Value: {self.best}")


# Initial points (feasible and infeasible)
bounds = [(1, 4), (1, 7), (0, 10)]  # Bounds for D, n, and alpha
F = [[2, 5, 7]]  # Feasible points
I = [[2, 2, 2]]  # Infeasible points
values = [objective(D, n, alpha) for D, n, alpha in F + I]  # Objective values for initial points

# Initialize and run the Bayesian Optimization
optimizer = BayesianOptimizationWithConstraints(bounds, F, I, values)
optimizer.optimize()

# Extract data from the optimizer instance
besthist = optimizer.besthist
feasiblehist = optimizer.feasiblehist


# Plot the besthist (Objective values)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)  # First subplot
plt.plot(range(1, len(besthist) + 1), besthist, color='b', label="Best Value")
plt.xlabel("Evaluations")
plt.ylabel("Objective")
plt.title("Best Objective Value for Feasible Points Over Iterations")
plt.grid(True)
plt.legend()

# Plot the feasiblehist (Feasible and Infeasible classifications)
plt.subplot(1, 2, 2)  # Second subplot
feasible_values = [x[0] for x in feasiblehist]  # Feasible evaluations
infeasible_values = [x[1] for x in feasiblehist]  # Infeasible evaluations

plt.plot(range(1, len(feasiblehist) + 1), feasible_values, color='g', label="Feasible")
plt.plot(range(1, len(feasiblehist) + 1), infeasible_values, color='r', label="Infeasible")
plt.xlabel("Evaluations")
plt.ylabel("Classifications")
plt.title("Feasible and Infeasible Classifications Over Iterations")
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
