import subprocess

# Parametri
prob = 0.1
num_gen = 500

# Primo comando
command_1 = [
    "python3",
    "fixedCode/ga_general.py",
    "-g", str(num_gen),
    "-p", "100",
    "-r",
    "-e", str(prob),
    "-o", "fixedCode/outputTest"
]

# Esecuzione del primo comando
print("Eseguo primo comando:", " ".join(command_1))
result_1 = subprocess.run(command_1)

# Controllo esito del primo comando
print("Comando 1 eseguito correttamente:", result_1.returncode == 0)

# Secondo comando
command_2 = [
    "python3",
    "fixedCode/plotAll.py",  
    "-g", str(num_gen) 
]

# Esecuzione del secondo comando
print("Eseguo secondo comando:", " ".join(command_2))
result_2 = subprocess.run(command_2)

# Controllo esito del secondo comando
print("Comando 2 eseguito correttamente:", result_2.returncode == 0)
