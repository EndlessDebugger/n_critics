import pickle 
from ncriticstask import NCriticsTask

with open('example_NCriticsTask.pkl', 'rb') as f:
    task = pickle.load(f)
    print(dir(task))
    print("Current prompt that is too long: ")
    print(task.prompt)
