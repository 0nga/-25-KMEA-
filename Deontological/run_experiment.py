import subprocess

# Parameters
prob = 0.1
num_gen = 500
savePassengers = 0 # with 0 i save pedestrians --> predAction = 1

# First command
command_1 = [
    "python3",
    "Deontological/ga_general.py",
    "-g", str(num_gen),
    "-p", "100",
    "-r",
    "-e", str(prob),
    "-s", str(savePassengers),
    "-o", "Deontological/outputTest"
]

# First command execution
print("Executing 1first command:", " ".join(command_1))
result_1 = subprocess.run(command_1)


# First command execution check
print("Command 1 successfully executed:", result_1.returncode == 0)

# Second command
command_2 = [
    "python3",
    "Deontological/plotAll.py",  
    "-g", str(num_gen) 
]

# Second command execution
print("Executing second command:", " ".join(command_2))
result_2 = subprocess.run(command_2)

# Second command execution check
print("Command 2  successfully executed:", result_2.returncode == 0)