import numpy as np

def find_continuous_blocks(arr, threshold=1, return_longest_only=False):
    '''
    arr: 1D array or list
    threshold: 1 means truly continuous, anything larger allows for some slop/noise
    '''
    
    diffs = np.diff(arr)
    transitions = np.where(diffs > threshold)[0].tolist()
    transitions.append(-2)
    
    blocks = []
    last_transition = 0
    for transition in transitions:
        block = arr[last_transition+1:transition+1]
        blocks.append(block)
        last_transition = transition
    
    if return_longest_only:
        block_lengths = [len(block) for block in blocks]
        return blocks[np.argmax(block_lengths)]
    
    
    return blocks
    
