import subprocess


# PArameters
prob = 0.1
num_gen = 500

# First command
command_1 = [
    "python3",
    "Utilitarian/ga_general.py",
    "-g", str(num_gen),
    "-p", "100",
    "-r",
    "-e", str(prob),
    "-o", "Utilitarian/outputTest"
]

# First command execution
print("Executing first command:", " ".join(command_1))
result_1 = subprocess.run(command_1)


# First command execution check
print("Command 1 successfully executed:", result_1.returncode == 0)

# Second command
command_2 = [
    "python3",
    "Utilitarian/plotAll.py",  
    "-g", str(num_gen) 
]

# Second command execution
print("Executing second command:", " ".join(command_2))
result_2 = subprocess.run(command_2)

# Second command execution check
print("Command 2  successfully executed:", result_2.returncode == 0)