# Multi-Objective Particle Swarm Optimization (MOPSO)

## Project Overview

This project implements a Multi-Objective Particle Swarm Optimization (MOPSO) algorithm based on Particle Swarm Optimization (PSO) for solving optimization problems with multiple conflicting objectives. The algorithm maintains an external archive to store non-dominated solutions (Pareto optimal solutions) and uses a crowding distance mechanism to maintain solution diversity.

## Algorithm Features

1. **Multi-Objective Optimization**: Simultaneously optimizes two conflicting objective functions
2. **External Archive Mechanism**: Maintains a set of non-dominated solutions (Pareto front)
3. **Crowding Distance Selection**: Uses crowding distance to maintain solution diversity
4. **Dynamic Visualization**: Real-time display of Pareto front evolution
5. **Chinese Font Support**: Automatic configuration of Chinese fonts to ensure proper display of Chinese characters in charts

## Algorithm Principles

### 1. Particle Swarm Optimization Basics
- Each particle represents a candidate solution
- Particles update their position and velocity based on personal best and global best
- Velocity update formula: `v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)`

### 2. Multi-Objective Processing Mechanism
- **Dominance Relationship Judgment**: Determines if one solution is dominated by another
- **External Archive**: Stores all non-dominated solutions (Pareto optimal solutions)
- **Crowding Distance**: Measures solution density in objective space to maintain diversity

### 3. Objective Function
Uses the ZDT1 test function:
- Objective 1: `f1 = x1` (minimize)
- Objective 2: `f2 = g * h`, where `g = 1 + 9*x2`, `h = 1 - sqrt(x1/g)` (minimize)
- Variable ranges: `x1 ∈ [0, 1]`, `x2 ∈ [0, 5]`

## Parameter Settings

```python
# Algorithm parameters
pop = 100           # Population size
dim = 2             # Problem dimension
maxIter = 100       # Maximum iterations
archive_size = 50   # Maximum archive capacity

# PSO parameters
w = 0.5             # Inertia weight
c1 = 1.0            # Individual learning factor
c2 = 1.0            # Social learning factor

# Boundary conditions
lb = [0, 0]         # Lower bounds
ub = [1, 5]         # Upper bounds
v_min = [-0.5, -0.5] # Minimum velocity
v_max = [0.5, 0.5]  # Maximum velocity
```

## Core Functions

### 1. `ini(size, lower_bound, upper_bound, dim)`
- **Function**: Initialize position or velocity
- **Parameters**: Population size, lower bound, upper bound, dimension
- **Returns**: Randomly initialized position/velocity matrix

### 2. `fun(position)`
- **Function**: Calculate objective function values
- **Parameters**: Particle position
- **Returns**: Array of two objective function values

### 3. `is_dominated(fitness1, fitness2)`
- **Function**: Determine dominance relationship
- **Parameters**: Two fitness vectors
- **Returns**: Boolean value indicating whether fitness1 is dominated by fitness2

### 4. `calculate_crowding_distance(fitness_vectors)`
- **Function**: Calculate crowding distance
- **Parameters**: Fitness vectors matrix
- **Returns**: List of crowding distances for each solution

### 5. `update_archive(archive, new_position, max_archive_size)`
- **Function**: Update external archive
- **Parameters**: Current archive, new position, maximum archive capacity
- **Returns**: Updated archive

## Running Process

### 1. Initialization Phase
- Initialize particle positions and velocities
- Calculate initial fitness
- Initialize personal best and external archive

### 2. Iterative Optimization Phase
- Update personal best solutions
- Update external archive (add non-dominated solutions, remove dominated ones)
- Use crowding distance selection to maintain archive diversity
- Update particle velocities and positions
- Dynamically update visualization every 10 iterations

### 3. Result Output Phase
- Display final Pareto front
- Output positions and fitness values of optimal solutions
- Save visualization results

## Installation and Running

### Environment Requirements
- Python 3.6+
- NumPy
- Matplotlib

### Install Dependencies
```bash
pip install numpy matplotlib
```

### Run the Program
```bash
python MOPSO.py
```

## Output Results

After running the program, it will display:
1. **Iteration Process Information**: Shows progress and archive size for each iteration
2. **Dynamic Visualization**: Updates Pareto front plot every 10 iterations
3. **Final Results**:
   - Number of Pareto front solutions
   - Positions and fitness values of top 10 optimal solutions
   - Final Pareto front visualization

## Visualization Features

1. **Dynamic Updates**: Updates chart every 10 iterations
2. **Fixed Axes**: Maintains consistent view range
3. **Chinese Labels**: Automatic configuration of Chinese fonts
4. **Interactive Mode**: Real-time display of optimization process

## Algorithm Advantages

1. **Efficiency**: PSO algorithm has fast convergence speed
2. **Diversity**: Crowding distance mechanism ensures solution diversity
3. **Practicality**: Suitable for real-world engineering multi-objective optimization problems
4. **Visualization**: Real-time display of optimization process for easy understanding and debugging

## Application Areas

- Engineering design optimization
- Resource allocation problems
- Scheduling optimization
- Machine learning hyperparameter tuning
- Financial portfolio optimization

## Extensions and Improvements

### Possible Improvement Directions
1. **Adaptive Parameters**: Implement adaptive inertia weight and learning factors
2. **More Objective Functions**: Extend to three or more objectives
3. **Constraint Handling**: Add constraint handling mechanisms
4. **Parallel Computing**: Utilize multi-core CPU for acceleration
5. **Other Test Functions**: Implement more standard test functions (e.g., ZDT2, ZDT3, etc.)

### Custom Objective Functions
To use custom objective functions, simply modify the implementation of the `fun()` function:
```python
def fun(position):
    # Custom objective functions
    f1 = ...  # First objective
    f2 = ...  # Second objective
    return np.array([f1, f2])
```

## Key Concepts Explained

### 1. Pareto Optimal Solution
In multi-objective optimization, a solution is called Pareto optimal if it is not worse than any other solution in all objectives and is better in at least one objective.

### 2. Dominance Relationship
Solution A dominates solution B if and only if:
- Solution A is not worse than solution B in all objectives
- Solution A is better than solution B in at least one objective

### 3. Crowding Distance
Used to measure the density of solutions in the objective space. Larger distances indicate sparser regions, which helps maintain solution set diversity.

### 4. External Archive
A collection used to store all non-dominated solutions (Pareto optimal solutions), dynamically updated during algorithm execution.

## Frequently Asked Questions

### Q1: How to modify the objective function?
A: Modify the calculation logic in the `fun()` function, ensuring it returns an array containing all objective function values.

### Q2: How to adjust algorithm parameters?
A: Modify the parameter settings in the main program, such as population size, iteration count, learning factors, etc.

### Q3: How to handle optimization problems with more dimensions?
A: Adjust the `dim` parameter and ensure the objective function can handle inputs of the corresponding dimension.

### Q4: How to save optimization results?
A: Add code at the end of the program to save solutions from the archive to a file.

## Code Examples

### Running the MOPSO Algorithm
```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Run the main program
if __name__ == "__main__":
    # You can add custom initialization code here
    # Then call the MOPSO algorithm
    print("Starting MOPSO algorithm...")
```

### Viewing Optimization Results
```python
# View solutions in the archive
for i, solution in enumerate(archive):
    position = solution[0]
    fitness = solution[1]
    print(f"Solution {i+1}: Position={position}, Fitness={fitness}")
```

## References

1. Coello, C. A. C., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives with particle swarm optimization.
2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
3. Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: Empirical results.

## License

This project is for learning and research purposes only.

## Author

afterglow321

## Changelog

- 2025-03-18: Initial version, basic MOPSO algorithm implementation
- 2025-03-18: Added dynamic visualization functionality
- 2025-03-18: Improved documentation and comments
- 2026-03-18: Created English documentation

## Contact

For questions or suggestions, please submit an issue through the project repository.