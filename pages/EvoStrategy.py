from numpy import arange, exp, sqrt, cos, e, pi, meshgrid, asarray, cos, argsort
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import randn, rand, seed
import streamlit as st

# Objective function
def objective(v):
    x, y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# Check if a point is within the bounds of the search
def in_bounds(point, bounds):
    for d in range(len(bounds)):
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True

# Evolution strategy (mu, lambda) algorithm
def es_comma(objective, bounds, n_iter, step_size, mu, lam):
    best, best_eval = None, 1e+10
    n_children = int(lam / mu)
    population = list()
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    for epoch in range(n_iter):
        scores = [objective(c) for c in population]
        ranks = argsort(argsort(scores))
        selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
        children = list()
        for i in selected:
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + randn(len(bounds)) * step_size
                children.append(child)
        population = children
    return [best, best_eval]

# Evolution strategy (mu + lambda) algorithm
def es_plus(objective, bounds, n_iter, step_size, mu, lam):
    best, best_eval = None, 1e+10
    n_children = int(lam / mu)
    population = list()
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    for epoch in range(n_iter):
        scores = [objective(c) for c in population]
        ranks = argsort(argsort(scores))
        selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
        children = list()
        for i in selected:
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
            children.append(population[i])
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + randn(len(bounds)) * step_size
                children.append(child)
        population = children
    return [best, best_eval]

# Streamlit display
st.title("Ackley Function Optimization with Evolution Strategies")

# Plot the Ackley function surface
r_min, r_max = -5.0, 5.0
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
x, y = meshgrid(xaxis, yaxis)
results = objective([x, y])

fig = pyplot.figure()
axis = fig.add_subplot(111, projection='3d')
axis.plot_surface(x, y, results, cmap='jet')
st.pyplot(figure)

# Perform the evolution strategies
seed(1)
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
n_iter = 5000
step_size = 0.15
mu = 20
lam = 100

# Run (mu, lambda) strategy
best_comma, score_comma = es_comma(objective, bounds, n_iter, step_size, mu, lam)
st.write('Best solution (mu, lambda): f(%s) = %f' % (best_comma, score_comma))

# Run (mu + lambda) strategy
best_plus, score_plus = es_plus(objective, bounds, n_iter, step_size, mu, lam)
st.write('Best solution (mu + lambda): f(%s) = %f' % (best_plus, score_plus))
