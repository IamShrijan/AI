## BUILD A CLASS DEF HERE 
import time
import numpy as np
import random
from gridgame import *

class GridSolver:
    def __init__(self, gui=True, render_delay_sec=0.1, grid_size=10):
        setup(GUI=gui, render_delay_sec=render_delay_sec, gs=grid_size)
        self.state = execute('export')
        self.weights = [1, 2, 1, 1]
        self.gridSize = len(self.state[3])
    
    def reset_grid(self):
        setup(GUI=True, render_delay_sec=0, gs=10)
        self.state = execute('export')


    def valid_grid(self, grid):
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                color = grid[i, j]
                if color == -1:
                    continue
                if (i > 0 and grid[i - 1, j] == color) or \
                   (i < self.gridSize - 1 and grid[i + 1, j] == color) or \
                   (j > 0 and grid[i, j - 1] == color) or \
                   (j < self.gridSize - 1 and grid[i, j + 1] == color):
                    return False
        return True

    def fetch_color_palette(self, grid):
        empty_cells = 0
        colors_used = set()
        conflict = 0
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                color = grid[i, j]
                if color == -1:
                    empty_cells += 1
                else:
                    colors_used.add(color)
                    conflict += sum([
                        1 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        if 0 <= i + di < self.gridSize and 0 <= j + dj < self.gridSize and grid[i + di, j + dj] == color
                    ])
        return (len(colors_used), empty_cells, conflict)

    def get_move_set_to_move(self, pos_a, pos_b):
        x_diff = pos_a[1] - pos_b[1]
        y_diff = pos_a[0] - pos_b[0]
        return ['w'] * abs(x_diff) if x_diff > 0 else ['s'] * abs(x_diff) + \
               ['a'] * abs(y_diff) if y_diff > 0 else ['d'] * abs(y_diff)

    def objective_func(self, curr_state):
        color_palette, empty_cells, conflicts = self.fetch_color_palette(curr_state[3])
        shapes = len(curr_state[4])
        return (self.weights[0] * empty_cells) + (self.weights[1] * shapes) + \
               (self.weights[2] * conflicts**2) + (self.weights[3] * color_palette**2)

    def get_neighbors(self, state, strict=False):
        neighbors = []
        current_pos = state[0].copy()
        grid = state[3]
        moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        if not strict:
            moves.extend([(-2, 0), (2, 0), (0, 2), (0, -2)])
        
        for dy, dx in moves:
            new_pos = [current_pos[0] + dy, current_pos[1] + dx]
            if 0 <= new_pos[0] < self.gridSize and 0 <= new_pos[1] < self.gridSize:
                shape_range = range(3) if strict else [random.randint(0, len(shapes) - 1),random.randint(0, len(shapes) - 1)]
                for new_shape_index in shape_range:
                    for new_color_index in range(len(colors)):
                        new_grid = grid.copy()
                        new_placed_shapes = state[4].copy()
                        if canPlace(new_grid, shapes[new_shape_index], new_pos):
                            placeShape(new_grid, shapes[new_shape_index], new_pos, new_color_index)
                            new_placed_shapes.append((new_shape_index, new_pos.copy(), new_color_index))
                            new_state = (new_pos, new_shape_index, new_color_index, new_grid, new_placed_shapes, state[5])
                            moves = self.get_move_set_to_move(state[0], new_pos)
                            moves += ['h'] * ((new_shape_index - state[1]) % len(shapes))
                            moves += ['k'] * ((new_color_index - state[2]) % len(colors))
                            moves += ['p']
                            neighbors.append((new_state, moves))
        return neighbors

    def hill_climbing(self, state):
        current_state = state
        current_val = self.objective_func(state)
        current_moves = []
        no_neigh = 0
        no_place = 0
        strict = False

        for iter in range(1000):
            print("Iteration:",iter)
            neighbors = self.get_neighbors(current_state, strict)
            if not neighbors:
                print("NoNeigh,", no_neigh)
                no_neigh += 1
                if(no_neigh>3):
                    if(no_neigh>5):
                        print("IT IS STRICT NOW")
                        input("Press Enter")
                        strict = True
                        unfilledCells = [(x, y) for x in range(self.gridSize) for y in range(self.gridSize) if current_state[3][x, y] == -1]
                        if(unfilledCells!=[]):
                            random_pos = random.choice(unfilledCells)
                            print
                            move = self.get_move_set_to_move(current_state[0], random_pos)
                            for m in move:
                                current_state = execute(m)
                            input("Correct")
                        else:
                            break
                else:
                    x,y = random.randint(0, self.gridSize-1),random.randint(0, self.gridSize-1)
                    move = self.get_move_set_to_move(current_state[0],[x,y])
                    for m in move:
                        current_state = execute(m)
                    if(current_state[0]==-1):
                        print("GOOOOOODDDD")
                    else:
                        print("WHYyyyyyy soo many badddd")
            else:
                #no_neigh -= 1
                best_neighbor = min(neighbors, key=lambda x: self.objective_func(x[0]))
                best_neighbor_val = self.objective_func(best_neighbor[0])

                if best_neighbor_val <= current_val:
                    current_state = best_neighbor[0]
                    current_val = best_neighbor_val
                    current_moves = best_neighbor[1]
                    for m in current_moves:
                        current_state = execute(m)
                    no_place += 1
                else:
                    print("NOOO BEST NEIGH",best_neighbor_val,current_val)
                
                if no_place > 4:
                    no_place = 0
                    if not self.valid_grid(current_state[3]):
                        for _ in range(5):
                            current_state = execute('u')
                        current_val = self.objective_func(current_state)

        return current_state, current_moves

    def solve(self):
        best_solution = None
        best_score = float('inf')
        cur_state = self.state

        for _ in range(5):
            solution, moves = self.hill_climbing(cur_state)
            if solution[3].all() is None:
                print(cur_state)
            if checkGrid(solution[3]):
                break

        return solution

    def test(self):
        best_solutions = []
        
        for epoch in range(1):  # Run for 10 epochs
            print("Running Epoch:",epoch)
            
            
            # Move values closer to target
            learning_rate = 0.01
            self.weights = [1.5,0.0,1.0,1.0]
            prev_score = 45.634886020970164
            
            # Comment 1: Make it np array. 
            curState = self.state
            for _ in range(1):
                solution, moves = self.hill_climbing(curState)
                if solution[3].all() is None:
                    print(curState)
                #self.reset_grid()
                if checkGrid(solution[3]):
                    break
            
            score = self.objective_func(solution)
            best_solutions.append((score, solution, self.weights))
            
            

            score_difference = prev_score - score
            weight_updates = np.array([
                score_difference * self.weights[0],  # Update for empty cells
                score_difference * self.weights[1],  # Update for shapes
                score_difference * self.weights[2],  # Update for conflicts
                score_difference * self.weights[3]   # Update for color palette
            ])

            self.weights -= learning_rate*weight_updates
            # Ensure weights remain positive
            self.weights = np.maximum(self.weights, 0.1)     
            
            # Keep only the top 5 solutions
            #best_solutions = best_solutions[:5]
        # Sort solutions by score (lower is better)
        best_solutions.sort(key=lambda x: x[0])
        # Print the top 5 sets of values
        print("Top 5 sets of (a, b, c, d) values:")
        for i, (score, _, params) in enumerate(best_solutions, 1):
            print(f"{i}. {params} (Score: {score})")
        
        # Return the best solution
        return best_solutions[0][1]


    def run(self):
        start = time.time()
        solution = self.solve()
        end = time.time()

        if checkGrid(solution[3]):
            print("Solution found:")
        else:
            print("No valid solution found. Best attempt:")
        printGridState(solution[3])
        print(solution)

        np.savetxt('grid.txt', solution[3], fmt="%d")
        with open("shapes.txt", "w") as outfile:
            outfile.write(str(solution[4]))
        with open("time.txt", "w") as outfile:
            outfile.write(str(end - start))

# Usage
solver = GridSolver(gui=True, render_delay_sec=0.1, grid_size=10)
solver.test()

## ALGORITHM HILL CLIMBING
#def hillClimbing(state):
    # currentState = state
    # currentVal = ObjectiveFunc(state,weights)
    # currentMoves = []
    # for _ in range(10):
    #     N,M = getNeighbhor(currentState)
    #     nVal = ObjectiveFunc(N,weights)
    #     if(nVal < currentVal):
    #         currentState = N
    #         currentMoves = M
    #         currentVal = nVal
    # return currentState,currentMoves