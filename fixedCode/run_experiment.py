import subprocess
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Run genetic algorithm experiment')
    parser.add_argument('-g', '--generations', type=int, default=20, help='Number of generations')
    parser.add_argument('-p', '--population', type=int, default=100, help='Population size')
    parser.add_argument('-e', '--prob', type=float, default=0.1, help='Probability value')
    parser.add_argument('-a', '--approach', choices=['Deontological', 'Utilitarian'], default='Deontological',
                      help='Approach type: Deontological or Utilitarian')
    args = parser.parse_args()

    # Select the appropriate GA file based on the approach
    ga_file = "fixedCode/ga_utilitarian.py" if args.approach == "Utilitarian" else "fixedCode/ga_general.py"

    # Comando corretto
    command = [
        "python3",
        ga_file,
        "-g", str(args.generations),
        "-p", str(args.population),
        "-r",
        "-e", str(args.prob),
        "-o", f"fixedCode/{args.approach}/test/outputTest"
    ]

    # Esecuzione
    print("Eseguo:", " ".join(command))
    result = subprocess.run(command)

    # Controllo esito
    print("Comando eseguito correttamente:", result.returncode == 0)

if __name__ == "__main__":
    main()


