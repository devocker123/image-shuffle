import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import random
import heapq


#plot grid code here....
def plot_grid(grid:dict(), num_patches = 32):
    img = 0
    for i in range(0,num_patches):
        patch = grid[i,0]
        for j in range(1,num_patches):
            patch = np.hstack((patch,grid[i,j]))

        if (isinstance(img, int)):
            img = patch
        else:
            img = np.vstack((img,patch))
    plt.imshow(img)
    plt.axis(False)
    plt.show()




#random shuffling code was provided
def Random_Shuffle(linear:list(),blank_i,num_patches, x:int =10):

    prev_idx = 0
    for i in range(x):
        row = blank_i // num_patches
        col = blank_i % num_patches
        neighbor_indices = []
        # Check above
        if row > 0:
            neighbor_indices.append((row - 1) * num_patches + col)
        # Check left
        if col > 0:
            neighbor_indices.append(row * num_patches + (col - 1))
        # Check right
        if col < num_patches - 1:
            neighbor_indices.append(row * num_patches + (col + 1))
        # Check below
        if row < num_patches - 1:
            neighbor_indices.append((row + 1) * num_patches + col)

        if prev_idx in neighbor_indices:
            neighbor_indices.remove(prev_idx)
        new_blank = random.choice(neighbor_indices)
        prev_idx = blank_i
        temp = linear[blank_i]
        linear[blank_i] = linear[new_blank]
        linear[new_blank] = temp
        
        blank_i = new_blank

    return linear




# Create a 2D numpy array to store shuffled patch values
#this 2d array will serve as initial state for n puzzle game
#the values are sum of pixel value in shuffle_dictionary
shuffled_values_grid = np.zeros((4, 4), dtype=int)
#goal grid is the 2d numpy array for goal state
goal_grid = np.zeros((4,4), dtype=int)
linear = list()
def Image_Puzzle(path_img:str, patch_size:int = 16, depth_level=10):
    img = Image.open(path_img)
    img = img.resize((512,512))
    print(f"Original Image Size: {img.size}")
    #print("Original Image: ")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img = np.asarray(img)
    img_size = img.shape[0]

    num_patches = img_size//patch_size# 512 / 128 = 4
    

    for i,patch_height in enumerate(range(0, img_size, patch_size)):
        for j,patch_width in enumerate(range(0, img_size, patch_size)):
            image = img[patch_height:patch_height+patch_size,
                      patch_width:patch_width+patch_size]
            linear.append(image)
    linear[-1] = np.zeros_like(image)
    linear2 = linear.copy()

    linear2 = Random_Shuffle(linear2,len(linear2)-1,num_patches,depth_level)
    
    
    #Graph making
    shuffled_grid = {}
    original_grid = {}
    i = 0
    blank = (0,0)
    for x in range(0,num_patches):
        for y in range(0,num_patches):
            if np.array_equiv(linear2[i],np.zeros_like(image)):
                blank = (x,y)
            shuffled_grid[(x,y)] = linear2[i]
            original_grid[(x,y)] = linear[i]
            i+=1

    ##### addition of my code    
    # making numpy array-> goal state
    for x in range(4):
        for y in range(4):
            patch = original_grid[(x, y)]  # Get the shuffled patch from the dictionary
            # Calculate the assigned value for the patch based on its content
            if np.array_equiv(patch, np.zeros_like(image)):
                goal_value = 0  # Use 0 for the blank patch
            else:
                goal_value = np.sum(patch) # Use the sum of pixel values as the assigned value
            goal_grid[x, y] = goal_value
    

    # shuffled_values_grid with shuffled patch values
    for x in range(4):
        for y in range(4):
            patch = shuffled_grid[(x, y)]  # shuffled patch value from the dictionary
            # assigned value for the patch based on its content
            if np.array_equiv(patch, np.zeros_like(image)):
                assigned_value = 0  # Use 0 for the blank patch
            else:
                assigned_value = np.sum(patch) # Use the sum of pixel values as the assigned value
            shuffled_values_grid[x, y] = assigned_value

    #####
    print(f"Shuffled times: {depth_level}")
    plot_grid(shuffled_grid,num_patches)
    return original_grid, shuffled_grid , blank ,num_patches




img_path = r'C:\\Users\\mibra\\pictures\\cropped-My-image.jpg'
patch_size = 128
shuffels = 10    #ran this code with shuffle values 10, 15, 30      

#patch size greater than 128 not accepted, if facing difficulty you can try shuffels till 8, higher value preffered.
# If you have achieved solution try shuffling 13,14,15 times. Might result in bonus marks ;)




original_grid, shuffled_grid ,blank, num_patches= Image_Puzzle(img_path,patch_size,shuffels)


# img = Image.open(img_path)
# print(f"Original Image Size: {img.size}")


plot_grid(original_grid,num_patches)


plot_grid(shuffled_grid,num_patches)

print(goal_grid)# checking before and after values

# Now shuffled_values_grid contains the shuffled values of patches
print(shuffled_values_grid)
#if same then correct

#this dictionary will contain the pixel values of the reshuffled image
resequenced_dict = {}
#actually the reverse of what i did earlier
# matching the solved value(sum) of array with the original dictionary value and then reassigning
def re_seq(solved):
    for x in range(shuffled_values_grid.shape[0]):
        for y in range(shuffled_values_grid.shape[1]):
            assigned_value = shuffled_values_grid[x, y]
            if assigned_value == 0:
                # Use a blank patch (all zeros) for the empty space
                patch = np.zeros_like(original_grid[(0, 0)])  # Replace with the size of your patches
            else:
                # Get the original patch corresponding to the assigned value
                patch = None
                for key, value in original_grid.items():
                    if np.array_equal(np.sum(value), assigned_value):
                        patch = value
                        break
                if patch is None:
                    raise ValueError("No matching patch found for assigned value.")
            resequenced_dict[(x, y)] = patch
            
            
re_seq(shuffled_values_grid)


#shuffled_values_grid ---> initial state
#goal_grid ---> goal state
def bfs(initial_state, goal_state):
    queue = deque([(initial_state, [])])
    visited = set()

    while queue:
        current_state, path = queue.popleft()
        visited.add(tuple(map(tuple, current_state)))

        if np.array_equal(current_state, goal_state):
            return path

        # get neighbor states and add them to the queue
        successors = get_neighbors(current_state)
        for next_state, action in successors:
            if tuple(map(tuple, next_state)) not in visited:
                queue.append((next_state, path + [action]))

    return None  # No solution found

def get_neighbors(state):
    neighbors = []
    i, j = find_empty_space(state)
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible moves are: right, down, left, up

    for move in moves:
        ni, nj = i + move[0], j + move[1]
        if 0 <= ni < state.shape[0] and 0 <= nj < state.shape[1]:
            new_state = state.copy()
            new_state[i, j], new_state[ni, nj] = new_state[ni, nj], new_state[i, j]
            neighbors.append((new_state, move))
    
    return neighbors

def find_empty_space(state):
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i, j] == 0:
                return i, j

def print_state(state):
    for row in state:
        print(" ".join(map(str, row)))
    print("\n")

path = bfs(shuffled_values_grid, goal_grid.tolist())
if path:
    print("Solution found:")
    for step, action in enumerate(path):
        print(f"Step {step + 1}:")
        #first print array then plot then print move
        print_state(shuffled_values_grid)
        re_seq(shuffled_values_grid)# re shuffling function called
        plot_grid(resequenced_dict,num_patches)
        
        print(f"Action: Move {action} \n")
        i, j = find_empty_space(shuffled_values_grid)
        ni, nj = i + action[0], j + action[1]
        shuffled_values_grid[i, j], shuffled_values_grid[ni, nj] = shuffled_values_grid[ni, nj], shuffled_values_grid[i, j]
else:
    print("No solution found.")

print(shuffled_values_grid)

print(goal_grid)

re_seq(shuffled_values_grid)#one more time
plot_grid(resequenced_dict,num_patches)
print("Total steps : ",step+2)

# Brief Explanation why i chose Breadth-First Search

#In terms of the three factors which is considered when comparing algorithms for search,
#BFS ensures completeness, optimality and good space complexity compared to DFS and IDDLS.



# performing uninformed search with shuffle value = 30 is not giving any output


#performing Informed search with shuffle value = 30

shuffels = 10

original_grid, shuffled_grid ,blank, num_patches= Image_Puzzle(img_path,patch_size,shuffels)

plot_grid(original_grid,num_patches)


plot_grid(shuffled_grid,num_patches)


print(goal_grid)

print(shuffled_values_grid)

def heuristics(state, goal):
    distance = 0
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            value = state[i, j]
            if value != 0:  # blank tile
                goal_position = np.argwhere(goal == value)[0]
                distance += abs(i - goal_position[0]) + abs(j - goal_position[1])
    return distance


def apply_move(state, move):
    blank_i, blank_j = find_empty_space(state)
    new_i = blank_i + move[0]
    new_j = blank_j + move[1]
    new_state = state.copy()
    new_state[blank_i][blank_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[blank_i][blank_j]
    return new_state



def astar_search(initial_state, goal_state):
    open_set = []
    closed_set = set()
    came_from = {}  # Parent states

    heapq.heappush(open_set, (0, initial_state))  # (f-value, state)

    while open_set:
        _, current_state = heapq.heappop(open_set)

        if np.array_equal(current_state, goal_state):
            path = []
            while current_state.tobytes() in came_from:
                path.insert(0, current_state)
                current_state = came_from[current_state.tobytes()]
            path.insert(0, initial_state)
            return path

        closed_set.add(current_state.tobytes())

        for next_state, _ in get_neighbors(current_state):
            if next_state.tobytes() not in closed_set:
                f = heuristics(next_state, goal_state)
                heapq.heappush(open_set, (f, next_state))
                came_from[next_state.tobytes()] = current_state

    return None  # No solution found

result_state = astar(shuffled_values_grid, goal_grid)

# Print the solution path if found
if result_state:
    print("Solution found:")
    print(result_state)
else:
    print("No solution found.")

## My observations
# With shuffle value of 10 and using uninformed search Algorihtm - BFS the puzzle is solved relatively quickly beacuse the search 
# space is quite small. and since the BFS is optimal and complete algorithm it will give optimal solution in this case. 
# however if i increase the shuffle value to 13 or more the uninformed search algorithm will not give the answer
#
#therefore, informed search algorithm are used. When i increase the size of shuffle value to 30 and used the A star algorithm
# the puzzle -> (since i am not able to implement this part i will only use theoretical knowledge)
# the puzzle will be solved comparatively quickly using manhattan distance formula to calculate heuristics
# this informed search algorithms intelligently explore the paths.  

