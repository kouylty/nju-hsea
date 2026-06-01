import numpy as np
import logging
from copy import deepcopy
import random

log = logging.getLogger(__name__)


def swap_mutation(x: np.ndarray, repeats=5):
    assert x.ndim == 1
    next_x = x.copy()
    for _ in range(repeats):
        i, j = np.random.choice(range(len(x)), 2, replace=False)
        next_x[i], next_x[j] = next_x[j], next_x[i]
    return next_x

def insert_mutation(x: np.ndarray, repeats=5):
    assert x.ndim == 1
    next_x = x.copy()
    for _ in range(repeats):
        i, j = np.random.choice(range(len(next_x)), 2, replace=False)
        if i < j:
            next_x = np.concatenate((next_x[:i], next_x[i+1:j+1], next_x[i:i+1], next_x[j+1:]))
        else:
            next_x = np.concatenate((next_x[:j], next_x[i:i+1], next_x[j:i], next_x[i+1:]))
    return next_x


def reversal_mutation(x: np.ndarray, repeats=5):
    assert x.ndim == 1
    next_x = x.copy()
    for _ in range(repeats):
        i, j = np.random.choice(range(len(x)), 2, replace=False)
        # Ensure i < j for easier reversal
        if i > j:
            i, j = j, i
        # Reverse the sublist between indices i and j
        next_x[i:j+1] = next_x[i:j+1][::-1]
    return next_x

def shuffle_mutation(x: np.ndarray, repeats=5):
    assert x.ndim == 1
    next_x = x.copy()
    
    for _ in range(repeats):
        random_elements = []
        while len(random_elements) < 5:
            element = random.choice(next_x)
            if element not in random_elements:
                random_elements.append(element)

        loc_map = {}
        for element in random_elements:
            loc_map[element] = next_x.tolist().index(element)

        shuffled_elements = random_elements.copy()
        random.shuffle(shuffled_elements)

        for i in range(len(random_elements)):
            next_x[loc_map[random_elements[i]]] = shuffled_elements[i]

    return next_x

def shift_mutation(x: np.ndarray, repeats=5, shift_range=3):
    assert x.ndim == 1
    next_x = x.copy()

    for _ in range(repeats):
        # Step 1: Randomly select an element (macro_a)
        i = random.choice(range(len(x)))
        selected_element = next_x[i]

        # Step 2: Calculate the neighboring locations within the shift_range
        # The range can go from (i - shift_range) to (i + shift_range), ensuring the indices are valid
        neighbor_locs = list(range(max(0, i - shift_range), min(len(x), i + shift_range + 1)))
        # Remove the element itself (because we can't shift it to the same position)
        neighbor_locs.remove(i)

        # Step 3: Shuffle the neighboring positions
        random.shuffle(neighbor_locs)
        
        # Step 4: Try each neighbor location
        for loc in neighbor_locs:
            # Try moving the element to the new position
            new_x = next_x.copy()
            new_x[i], new_x[loc] = new_x[loc], new_x[i]  # Swap positions
            # If moving to this new location is feasible, update and return success
            if new_x[i] != selected_element:  # Simulating feasibility check
                next_x = new_x
                return next_x  # Successful mutation
            # If not feasible, restore to original state
            next_x = x.copy()
    return next_x  # If no successful move, return original sequence

def order_crossover(x1: np.ndarray, x2: np.ndarray):
    assert x1.ndim == 1 and x2.ndim == 1
    x_len = len(x1)
    next_x = np.zeros(x_len)
    
    i, j = np.random.choice(range(x_len), 2, replace=False)
    i, j = min(i, j), max(i, j)
    next_x[i: j] = x1[i: j]
    copy_idx = j
    for k in range(j, j+x_len-(j-i)):
        while x2[copy_idx] in next_x[i: j]:
            copy_idx = (copy_idx + 1) % x_len
        next_x[k % x_len] = x2[copy_idx]
        copy_idx = (copy_idx + 1) % x_len
    log.debug('next x: {}'.format(next_x))
    return next_x

def pmx_crossover(x1: np.ndarray, x2: np.ndarray):
    assert x1.ndim == 1 and x2.ndim == 1
    assert len(x1) == len(x2)
    
    x_len = len(x1)
    next_x = np.empty(x_len, dtype=x1.dtype)

    i, j = np.random.choice(x_len, 2, replace=False)
    i, j = min(i, j), max(i, j)
    
    next_x[i:j] = x1[i:j]
    segment = set(next_x[i:j])

    x1_segment = x1[i:j]
    x2_segment = x2[i:j]
    mapping = dict(zip(x1_segment, x2_segment))
    for k in list(range(j, x_len)) + list(range(0, i)):
        val = x2[k]

        while val in segment:
            val = mapping[val]
        
        next_x[k] = val
    
    log.debug(f'PMX offspring: {next_x}')
    return next_x

def cycle_crossover(x1: np.ndarray, x2: np.ndarray):
    assert x1.ndim == 1 and x2.ndim == 1
    assert len(x1) == len(x2)
    
    x_len = len(x1)
    next_x1 = np.empty_like(x1)
    next_x2 = np.empty_like(x2)
    visited = np.zeros(x_len, dtype=bool)
    cycle_num = 0  # Track cycle parity
    
    for start in range(x_len):
        if not visited[start]:
            current = start
            cycle_indices = []
            
            # Find cycle indices
            while True:
                cycle_indices.append(current)
                visited[current] = True
                x2_val = x2[current]
                current = np.where(x1 == x2_val)[0][0]  # Find position in x1
                if current == start:
                    break
            
            # Alternate parent based on cycle number
            if cycle_num % 2 == 0:
                next_x1[cycle_indices] = x1[cycle_indices]
                next_x2[cycle_indices] = x2[cycle_indices]
            else:
                next_x1[cycle_indices] = x2[cycle_indices]
                next_x2[cycle_indices] = x1[cycle_indices]
            
            cycle_num += 1
    
    selected_offspring = next_x1 if np.random.randint(2) == 0 else next_x2
    
    log.debug(f'Cycle CX offspring1: {next_x1}')
    log.debug(f'Cycle CX offspring2: {next_x2}')
    return selected_offspring