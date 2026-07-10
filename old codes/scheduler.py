import json
from ortools.sat.python import cp_model

print("1. Loading Geographic Matrix...")
with open('c_matrix.json', 'r') as f:
    C_matrix = json.load(f)

num_radars = len(C_matrix)
num_points = len(C_matrix[0])
num_time_slots = 24  # A full 24-hour horizon

# Map the budgets to your 5 specific radars (Max hours ON out of 24)
# Small (20), Medium (16), Large (12), Small (20), Medium (16)
budgets = [20, 16, 12, 20, 16]

model = cp_model.CpModel()

# --- 1. DEFINE VARIABLES ---
X = {} # X_it: Radar i ON at time t
for i in range(num_radars):
    for t in range(num_time_slots):
        X[(i, t)] = model.NewBoolVar(f'X_{i}_{t}')

V = {} # V_jt: Point j is a Void at time t
O = {} # O_jt: Number of extra overlapping radars at point j at time t
for j in range(num_points):
    for t in range(num_time_slots):
        V[(j, t)] = model.NewBoolVar(f'V_{j}_{t}')
        O[(j, t)] = model.NewIntVar(0, num_radars, f'O_{j}_{t}')

# --- 2. DEFINE CONSTRAINTS ---
print("2. Building Hardware & Coverage Constraints...")

# A. Energy Budget Constraint
for i in range(num_radars):
    model.Add(sum(X[(i, t)] for t in range(num_time_slots)) <= budgets[i])

# B. Coverage & Overlap Constraints (The core of your paper's math)
for t in range(num_time_slots):
    for j in range(num_points):
        # Calculate how many radars are looking at point j at time t
        coverage_sum = sum(C_matrix[i][j] * X[(i, t)] for i in range(num_radars))
        
        # coverage_sum >= 1 - V_jt
        model.Add(coverage_sum >= 1 - V[(j, t)])
        
        # O_jt >= coverage_sum - 1
        model.Add(O[(j, t)] >= coverage_sum - 1)

# --- 3. EXECUTE PHASE 1 ---
print("3. Executing Phase 1 (Minimizing Voids)...")
model.Minimize(sum(V[(j, t)] for j in range(num_points) for t in range(num_time_slots)))

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    min_voids = solver.ObjectiveValue()
    print(f"\n>> Phase 1 Complete! Minimum unavoidable void cells: {int(min_voids)}")
    
    # --- 4. EXECUTE PHASE 2 ---
    print("\n4. Executing Phase 2 (Minimizing Overlap)...")
    
    # Lock in the optimal security (you cannot sacrifice the void score to save power)
    model.Add(sum(V[(j, t)] for j in range(num_points) for t in range(num_time_slots)) == int(min_voids))
    
    # Change the objective to hunt for redundant overlap
    model.Minimize(sum(O[(j, t)] for j in range(num_points) for t in range(num_time_slots)))
    solver.Solve(model)
    
    print(f">> Phase 2 Complete! Total redundant overlap cells minimized to: {int(solver.ObjectiveValue())}")
    
    print("\n--- FINAL 24-HOUR RADAR SCHEDULE ---")
    for i in range(num_radars):
        schedule_array = [str(solver.Value(X[(i, t)])) for t in range(num_time_slots)]
        print(f"Radar {i+1} [Budget {budgets[i]:02d}/24]: {' '.join(schedule_array)}")
else:
    print("CRITICAL: Solver failed to find a feasible solution.")