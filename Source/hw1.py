import time
import numpy as np
from gridgame import *
import copy
import sys
import gridSolver

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.

##############################################################################################################################
setup(GUI = True, render_delay_sec = 0.1, gs = 10)

#loop_gui()
##############################################################################################################################

# Initialization

# shapePos is the current position of the brush.

# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py, and assignment instructions).

# currentColorIndex is the index of the current color being placed (order specified in gridgame.py, and assignment instructions).

# grid represents the current state of the board. 
    
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)

    # For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = execute('export')

#input()   # <-- workaround to prevent PyGame window from closing after execute() is called, for when GUI set to True. Uncomment to enable.
print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)


####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.

gridSolver

##########################################
# Write all your code in the area below. 
##########################################

##########################################
# Helper Functions 
##########################################
# Initialize random values for a, b, c, d

# def ValidGrid(grid): 
#     # Check that no adjacent cells have the same color
#     for i in range(len(grid)):
#         for j in range(len(grid)):
#             color = grid[i, j]
#             if color==-1:
#                 continue
#             if i > 0 and grid[i - 1, j] == color:
#                 return False
#             if i < len(grid) - 1 and grid[i + 1, j] == color:
#                 return False
#             if j > 0 and grid[i, j - 1] == color:
#                 return False
#             if j < len(grid) - 1 and grid[i, j + 1] == color:
#                 return False

#     return True

# def FetchColorPallate(grid):
#     emptyCells = 0
#     colorsUsed = set()
#     conflict = 0 
#     for i in range(len(grid)):
#         for j in range(len(grid)):
#             color = grid[i,j]
#             if color == -1:
#                 emptyCells +=1
#             else:
#                 colorsUsed.add(color)
#                 if i > 0 and grid[i - 1, j] == color:
#                     conflict+=1
#                 if i < len(grid) - 1 and grid[i + 1, j] == color:
#                     conflict+=1
#                 if j > 0 and grid[i, j - 1] == color:
#                     conflict+=1
#                 if j < len(grid) - 1 and grid[i, j + 1] == color:
#                     conflict+=1
#     colorPallate= len(colorsUsed)
#     return (colorPallate,emptyCells,conflict)

# def GetMoveSetToMove(posA,posB):
#     xDiff = posA[1] - posB[1]
#     yDiff = posA[0] - posB[0]
#     moveset = []
#     if(xDiff > 0):
#         moveset += ['w']*abs(xDiff)
#     elif(xDiff < 0):
#         moveset += ['s']*abs(xDiff)
#     if(yDiff > 0 ):
#         moveset += ['a']*abs(yDiff)
#     else:
#         moveset += ['d']*abs(yDiff)
#     return moveset


# ##########################################
# # Fetch Neighbors
# ##########################################
# #shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done

# def ObjectiveFunc(currState,weights):
#     #shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = currState
#     colorPallate,emptyCells,conflicts = FetchColorPallate(currState[3])
#     shapes = len(currState[4])
#     # Comment 2: NEED TO ADD ONE MORE WEIGHT FOR BIAS.
#     val = (weights[0]*emptyCells)+(weights[1]*shapes) +(weights[2]*conflicts**2) + (weights[3]*colorPallate**2)
#     return val

# def getNeighbhors(state, strict= False):
#     neighbors = []
#     current_pos = state[0].copy()
#     grid = state[3]

#     if (strict):
#         moves = [
#             (0, 0),   # current position
#             (-1, 0),  # up
#             (1, 0),   # down
#             (0, -1),  # left
#             (0, 1),   # right
#             ]
#         #print("ITS STRICT NOWW")
#         for dy, dx in moves:
#             new_pos = [current_pos[0] + dy, current_pos[1] + dx]
#             #print(new_pos)
#             # Check if the new position is within the grid
#             if 0 <= new_pos[0] < len(grid) and 0 <= new_pos[1] < len(grid):
#                 for new_shape_index in range(3):
#                     for new_color_index in range(len(colors)):
#                         new_grid = grid.copy()
#                         new_placed_shapes = state[4].copy()
#                         if canPlace(new_grid, shapes[new_shape_index], new_pos):
#                             placeShape(new_grid, shapes[new_shape_index], new_pos, new_color_index)
#                             new_placed_shapes.append((new_shape_index, new_pos.copy(), new_color_index))

#                             new_state = (new_pos, new_shape_index, new_color_index, new_grid, new_placed_shapes, state[5])
                            
#                             moves = GetMoveSetToMove(state[0], new_pos)
#                             moves += ['h'] * ((new_shape_index - state[1]) % len(shapes))
#                             moves += ['k'] * ((new_color_index - state[2]) % len(colors))
#                             moves += ['p']
#                             neighbors.append((new_state, moves))
#                 #print(neighbors)
#     else :
#         moves = [
#             (0, 0),   # current position
#             (-1, 0),  # up
#             (1, 0),   # down
#             (0, -1),  # left
#             (0, 1),  # right
#             (-2,0),
#             (2,0),
#             (0,2),
#             (0,-2)
#         ]

#         for dy, dx in moves:
#             new_pos = [current_pos[0] + dy, current_pos[1] + dx]
#             #print(new_pos)
#             # Check if the new position is within the grid
#             if 0 <= new_pos[0] < len(grid) and 0 <= new_pos[1] < len(grid):
#                 for new_color_index in range(len(colors)):  # Generate 2 neighbors for each position
#                     new_shape_index = random.randint(0, len(shapes) - 1)
#                     #new_color_index = random.randint(0, len(colors) - 1)
#                     new_grid = grid.copy()
#                     new_placed_shapes = state[4].copy()
#                     if canPlace(new_grid, shapes[new_shape_index], new_pos):
#                         placeShape(new_grid, shapes[new_shape_index], new_pos, new_color_index)
#                         new_placed_shapes.append((new_shape_index, new_pos.copy(), new_color_index))

#                         new_state = (new_pos, new_shape_index, new_color_index, new_grid, new_placed_shapes, state[5])
                        
#                         moves = GetMoveSetToMove(current_pos, new_pos)
#                         moves += ['h'] * ((new_shape_index - state[1]) % len(shapes))
#                         moves += ['k'] * ((new_color_index - state[2]) % len(colors))
#                         moves += ['p']
#                         neighbors.append((new_state, moves))

#     return neighbors

# #def getNeighbhor(state):
#     # neighbhor = copy.deepcopy(state)
#     # x,y = random.randint(0, len(grid)-1),random.randint(0, len(grid)-1)
#     # moves = []
#     # shape_index = random.randint(0, len(shapes) - 1)
#     # color_index = random.randint(0, len(colors) - 1)

#     # if(canPlace(neighbhor[3],shapes[shape_index],[x,y])):
#     #     placeShape(neighbhor[3],shapes[shape_index],[x,y],color_index)
#     #     placedShapesInNeighbor = neighbhor[4]+ [shapes[shape_index]]
#     #     neighbhor = ([x,y],shape_index,color_index,neighbhor[3],placedShapesInNeighbor,neighbhor[5])
#     #     moves = GetMoveSetToMove(state[0], [x,y])
#     #     moves += ['h']*((shape_index - state[1])%9)
#     #     moves += ['k']*((shape_index - state[2])%4)
#     #     moves += ['p']
#     # return (neighbhor,moves)

# def hillClimbing(state,weights):
#     currentState = state
#     currentVal = ObjectiveFunc(state,weights)
#     currentMoves = []
#     noNeigh = 0
#     noPlace = 0
#     noBestNeigh = 0
#     strict = False
#     for i in range(1000): 
#         print("This is iteration:",i)
#         neighbors = getNeighbhors(currentState,strict)
#         if(neighbors==[]):
#             noNeigh +=1
#             if(noNeigh>3):
#                 if(noNeigh>5):
#                     strict = True
#                     unfilledCells = [(x, y) for x in range(len(currentState[3])) for y in range(len(currentState[3])) if currentState[3][y, x] == -1]
#                     if(unfilledCells!=[]):
#                         random_pos = random.choice(unfilledCells)
#                         move = GetMoveSetToMove(currentState[0], random_pos)
#                         for m in move:
#                             currentState = execute(m)
#                     else:
#                         break
#                 else:
#                     x,y = random.randint(0, len(grid)-1),random.randint(0, len(grid)-1)
#                     move = GetMoveSetToMove(currentState[0],[x,y])
#                     for m in move:
#                         currentState = execute(m)
#             #print("CHANGE YOUR NEIGHBHOUR FUNCTION PLEASSSSEEE",noNeigh)
#         else:
#             noNeigh = 0
#             bestNeighbor = None
#             bestNeighborVal = float('inf')
#             bestMoves = []

#             for neighbor, moves in neighbors:
#                 neighborVal = ObjectiveFunc(neighbor,weights)
#                 if neighborVal < bestNeighborVal:
#                     bestNeighbor = neighbor
#                     bestNeighborVal = neighborVal
#                     bestMoves = moves
#             if(bestNeighbor):
#                 #print("CurrV",currentVal,"BestNVal",bestNeighborVal)
#                 if(bestNeighborVal <= currentVal):
#                     noBestNeigh = 0
#                     currentState = bestNeighbor
#                     currentVal = bestNeighborVal
#                     currentMoves = bestMoves
#                     for m in currentMoves:
#                         #print("Move Made",m)
#                         currentState = execute(m)
#                     noPlace+=1
#                 else:
#                     #noBestNeigh+=1
#                     print("Not the best Neigh")
#             if(noPlace>4):
#                 noPlace = 0
#                 if(ValidGrid(currentState[3])==False):
#                     for _ in range(5):
#                         currentState = execute('u')
#                     currentVal = ObjectiveFunc(currentState,weights)
                    

#     return currentState, currentMoves


# 


# # Call the solve function
# # solution = solve(State)

# # if checkGrid(solution[3]):
# #     print("Solution found:")
# #     printGridState(solution[3])
# # else:
# #     print("No valid solution found. Best attempt:")
# #     printGridState(solution[3])

# #print(solution)

# def solve(state):
#         bestSolution = None
#         bestScore = float('inf')
#         curState = state
#         weights = [1,2,1,1]

#         for _ in range(5):
#             solution,moves = hillClimbing(curState,weights)
#             if (solution[3].all() == None):
#                 print(curState)
#             if checkGrid(solution[3]):
#                 break
#             setup(GUI = True, render_delay_sec = 0.1, gs = 10)
#             curState = execute('e')

#         return solution

# State = (shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes  , done)
#test(State)


# N,M= getNeighbhors(State)
# for m1 in M:
#     print(m1)
#     State = execute(m1)
#     print(State)
# print(N)
# print(len(N[4]))


#solution = solve(State)

#if checkGrid(solution[3]):
#    print("Solution found:")
#    printGridState(solution[3])
#else:
#    print("No valid solution found. Best attempt:")
#    printGridState(solution[3])
#print(solution)

########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
