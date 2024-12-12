import numpy as np
import GPy
from sklearn.svm import SVC
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


# SVM Classifier for feasibility
def svm_classifier(F, I, a):
    # Labels: Feasible points as 1, Infeasible points as 0
    y_F = np.ones(len(F))  # Feasible points are labeled as 1
    y_I = np.zeros(len(I))  # Infeasible points are labeled as 0

    # Combine the data and labels
    x_train = np.vstack([F, I])  # Stack feasible and infeasible points
    y_train = np.hstack([y_F, y_I])  # Stack the labels
    if a > 0:
        # Polynomial kernel SVM
        svm = SVC(kernel='poly', degree=4, C=max(2 * abs(a), 1e-2))
        svm.fit(x_train, y_train)
    else:
        # Polynomial kernel SVM
        svm = SVC(kernel='poly', degree=4, C=10)
        svm.fit(x_train, y_train)
    return svm


# Bayesian Optimization Class using GPy
class BayesianOptimization:
    def __init__(self, bounds, F, I, values, a_0=0.5):
        self.bounds = bounds
        self.F = F  # Set of feasible points
        self.I = I  # Set of infeasible points
        self.valuesF = values[:len(F)]  # Objective values of feasible points
        self.valuesI = values[len(F):]  # Objective values of infeasible points
        self.f = 1  # Number of feasible points
        self.a_0 = a_0  # Initial value of a
        self.a = -a_0  # Current value of a
        self.best = np.min(values[:len(F)])  # Current best objective output
        self.best_input = F[np.argmin(values[:len(F)])]  # Best input corresponding to the best objective
        self.svm = None  # Current classifier
        self.stag = None # Used for stopping criteria
        self.besthist = [self.best] # Store history
        self.feasiblehist = [[len(self.F), len(self.I)]]

    def update(self, x, is_feasible_point, objective_value):
        # Update feasible or infeasible points
        if is_feasible_point:
            self.F.append(x)
            self.valuesF.append(objective_value)
            if objective_value < self.best:
                self.best = objective_value
                self.best_input = x
            self.a = -self.a_0 / self.f  # Update a based on the number of feasible points
            self.f += 1
        else:
            self.I.append(x)
            self.valuesI.append(objective_value)
            self.a = 0
        self.feasiblehist.append([len(self.F), len(self.I)])
        self.besthist.append(self.best)

    def subproblem_constraints(self, x):
        return self.svm.decision_function(x.reshape(1, -1)) >= self.a

    def constrained_acquisition_function(self, x, model, y_min):
        mu, sigma = model.predict(x.reshape(1, -1))  # Get mean and std dev
        z = (y_min - mu) / sigma  # Expected decrease in the objective
        improvement = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        if self.subproblem_constraints(x):
            return -improvement
        else:
            violation = self.a - self.svm.decision_function(x.reshape(1, -1))  # If violated, return the violation value
            penalty = np.exp(violation)  # Penalty grows with the magnitude of the violation
            return -improvement + penalty  # Penalize more severely for large violations

    def optimize_acquisition_function(self, model):
        # Find the minimum observed value
        y_min = np.min(self.valuesF)

        bounds = self.bounds
        # Initial guess (could be the current best or random within bounds)
        x0 = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds])

        # Define the acquisition function to be minimized (negative EI)
        def obj_func(x):
            return self.constrained_acquisition_function(x, model, y_min)

        # Optimize the acquisition function using a method like L-BFGS-B
        result = minimize(obj_func, x0, bounds=self.bounds, method='L-BFGS-B')
        # Return the point with the best acquisition value
        return result.x

    def optimize(self, max_iterations=98):
        for iteration in range(max_iterations):
            # Step 1: Solve SVM with current F and I
            self.svm = svm_classifier(self.F, self.I, self.a)

            # Step 2: Solve the subproblem
            x_train = np.array(self.F + self.I)
            y_train = np.array(self.valuesF + self.valuesI)
            # Create the kernel for the Gaussian Process
            kernel = GPy.kern.Matern32(input_dim=3, lengthscale=1.0) + GPy.kern.White(input_dim=3, variance=0.2)
            # Initialize the Gaussian Process with GPy
            model = GPy.models.GPRegression(x_train, y_train.reshape(-1, 1), kernel)
            model.optimize(messages=False)  # Optimize the hyperparameters

            # Step 3: Optimize acquisition function
            x = self.optimize_acquisition_function(model)
            # Step 4: Evaluate the new point
            D, n, alpha = x
            objective_value = objective(D, n, alpha)
            feasible = is_feasible(D, n, alpha)

            # Step 5: Update the sets based on feasibility
            self.update(x, feasible, objective_value)

            print(f"Iteration {iteration + 1}, Current Objective: {objective_value}, Best Objective: {self.best}")


bounds = [(1, 4), (1, 7), (0, 10)]  # Bounds for n and alpha
F = [[2, 5, 7]]
I = [[2, 2, 2]]
values = [objective(D, n, alpha) for D, n, alpha in F + I]
optimizer = BayesianOptimization(bounds, F, I, values)
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
