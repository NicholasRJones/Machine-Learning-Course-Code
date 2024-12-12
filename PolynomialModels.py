import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
    return 10 <= thrust(D, n, alpha) <= 12

# Fit polynomial model to the torque or thrust function
def fit_polynomial_model(func, points):
    X = np.array(points)
    y = np.array([func(*point) for point in points])

    poly = PolynomialFeatures(degree=2)  # Using degree 2 for the polynomial
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    return poly, model

# Function to approximate the objective using polynomial models
def approximate_objective(D, n, alpha, torque_model, thrust_model):
    torque_poly, torque_reg = torque_model
    thrust_poly, thrust_reg = thrust_model

    # Generate feature vector for the current point
    point = np.array([[D, n, alpha]])
    point_poly = torque_poly.transform(point)

    # Predict the torque and thrust values
    torque_value = torque_reg.predict(point_poly)[0]
    thrust_value = thrust_reg.predict(thrust_poly.transform(point))[0]

    return n * D * torque_value, thrust_value

# Initialize lists to store statistics for plotting
besthist = [objective(2, 5, 7)]  # To store the best objective value of feasible points
feasiblehist = [[1, 1]]  # To store tuples of feasible and infeasible counts

# Run the optimization loop and collect statistics
def optimize():
    # Initial points
    points = [[2, 5, 7], [2, 2, 2]]
    best = besthist[0]

    # Fit initial polynomial models
    torque_model = fit_polynomial_model(torque, points)
    thrust_model = fit_polynomial_model(thrust, points)

    # Initialize feasible and infeasible counts
    feasible_count = 1
    infeasible_count = 1

    for iteration in range(1, 101):
        # Approximate the objective and feasibility constraint
        def objective_to_minimize(x):
            D, n, alpha = x
            obj, thrust_value = approximate_objective(D, n, alpha, torque_model, thrust_model)
            return obj

        # Feasibility constraint: 10 <= thrust <= 12
        def feasibility_constraint(x):
            D, n, alpha = x
            _, thrust_value = approximate_objective(D, n, alpha, torque_model, thrust_model)
            return 1 - abs(thrust_value - 11)

        # Define the bounds for the variables
        bounds = [(1, 4), (1, 7), (0, 10)]

        # Generate a random initial guess within the bounds
        initial_guess = [np.random.uniform(low, high) for low, high in bounds]

        # Optimization using Sequential Least Squares Programming (SLSQP)
        result = minimize(objective_to_minimize, initial_guess, bounds=bounds,
                          constraints={'type': 'ineq', 'fun': feasibility_constraint})

        # If the optimization converged, get the new point
        if result.success:
            new_point = result.x
            D, n, alpha = new_point
            # Evaluate the objective at the new point
            obj_value = objective(D, n, alpha)

            # Check feasibility and store the best feasible objective value
            if is_feasible(D, n, alpha):
                feasible_count += 1
                if obj_value < besthist[len(besthist) - 1]:
                    best = obj_value
            else:
                infeasible_count += 1
            besthist.append(best)

            # Store the feasible/infeasible counts
            feasiblehist.append((feasible_count, infeasible_count))

            # Store the new point
            points.append([D, n, alpha])

            # Refit the polynomial models with the updated points
            torque_model = fit_polynomial_model(torque, points)
            thrust_model = fit_polynomial_model(thrust, points)
        print(f"Iteration {iteration}, Current Objective: {obj_value}, Best Objective: {best}")


    # Return the history for plotting
    return besthist, feasiblehist

# Call the optimization function
besthist, feasiblehist = optimize()

# Plotting the results
plt.figure(figsize=(10, 5))

# Plot the besthist (Objective values for feasible points)
plt.subplot(1, 2, 1)  # First subplot
plt.plot(range(1, len(besthist) + 1), besthist, color='b', label="Best Feasible Value")
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
