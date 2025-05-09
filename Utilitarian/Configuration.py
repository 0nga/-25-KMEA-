class Configuration:

	def __init__(self):
		# Define Hyperparameters for NN
		#HIDDEN_LAYER_COUNT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		self.HIDDEN_LAYER_COUNT = [1]
		#HIDDEN_LAYER_NEURONS = [8, 16, 24, 32, 64, 128, 256, 512]
		self.HIDDEN_LAYER_NEURONS = [6]
		#HIDDEN_LAYER_RATE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
		self.HIDDEN_LAYER_RATE = [0.1]
		#HIDDEN_LAYER_ACTIVATIONS = ['tanh', 'relu', 'sigmoid', 'linear', 'softmax']
		self.HIDDEN_LAYER_ACTIVATIONS = ['relu']
		#HIDDEN_LAYER_TYPE = ['dense', 'dropout']
		self.HIDDEN_LAYER_TYPE = ['dense']
		self.MODEL_OPTIMIZER = ['adam', 'rmsprop']

		# Define Genetic Algorithm Parameters
		self.MAX_GENERATIONS = 3  # Max Number of Generations to Apply the Genetic Algorithm
		self.POPULATION_SIZE = 5  # Max Number of Individuals in Each Population
		self.best_ratio = 0.8
		#self.BEST_CANDIDATES_COUNT = int(self.POPULATION_SIZE * 0.4)  # Number of Best Candidates to Use
		self.random_ratio = 0.01
		#self.RANDOM_CANDIDATES_COUNT = int(self.POPULATION_SIZE * 0.1)  # Number of Random Candidates (From Entire Population of Generation) to Next Population
		self.OPTIMIZER_MUTATION_PROBABILITY = 0.1  # 10% of Probability to Apply Mutation on Optimizer Parameter
		self.HIDDEN_LAYER_MUTATION_PROBABILITY = 0.1  # 10% of Probability to Apply Mutation on Hidden Layer Quantity
		self.HIDDEN_LAYER_MUTATION_RANGE = 0.01  # Apply Mutation of 1%
		self.costPedestrian = 0.25	# cost of pedestrian c_s
		self.costPassengers = 0.25 	# cost of passengers c_t
		self.ALTRUISM = 0.1	# altruistic behavior
		self.numberOfPedestrians=5	#numberOfPedons
		self.numberOfPassengers=5	#numberOfPassengers
		self.probDeathPedestrians = 1.0	#probDeathPedestrians
		self.probDeathPassengers = 1.0	#probDeathPassengers
		self.STIGMA = -0.25
		self.HONOR = 0.25
		self.randomizeAltruism = False
		#self.path="/Users/aloreggia/Downloads/test/500ge/tournament_tanh_reward_pythonTest_altruism_"+str(self.ALTRUISM)+"_probPed_"+str(self.probDeathPedestrians)+"_pop_"+str(self.POPULATION_SIZE)+"_gen_"+str(self.MAX_GENERATIONS)
		#self.path="/Users/aloreggia/Downloads/test/500ge/tournament_tanh_reward_pythonTest_altruism_"+str(self.ALTRUISM)+"_general_pop_"+str(self.POPULATION_SIZE)+"_gen_"+str(self.MAX_GENERATIONS)
		self.set_path("/Users/onga/git/-25-KMEA-/Utilitarian/outputTest")
		self.set_best_candidates()
		self.set_random_candidates()
		
	def set_altruism(self, altruism):
		self.ALTRUISM = altruism
		#self.path="/Users/aloreggia/Downloads/test/500ge/tournament_tanh_reward_pythonTest_altruism_"+str(self.ALTRUISM)+"_probPed_"+str(self.probDeathPedestrians)+"_pop_"+str(self.POPULATION_SIZE)+"_gen_"+str(self.MAX_GENERATIONS)
		#self.path="/Users/aloreggia/Downloads/test/500ge/tournament_tanh_reward_pythonTest_altruism_"+str(self.ALTRUISM)+"_general_pop_"+str(self.POPULATION_SIZE)+"_gen_"+str(self.MAX_GENERATIONS)
		#self.set_path()
		
	def set_best_candidates(self):
		self.BEST_CANDIDATES_COUNT = max(1, int(self.POPULATION_SIZE * self.best_ratio))  # Number of Best Candidates to Use

	
	def set_random_candidates(self):
		self.RANDOM_CANDIDATES_COUNT = max(1, int(self.POPULATION_SIZE * self.random_ratio))  # Number of Random Candidates (From Entire Population of Generation) to Next Population

	def set_population_size(self, population):
		self.POPULATION_SIZE = population
		self.set_random_candidates()
		self.set_best_candidates()

	def set_path(self,path1=None):

		if path1:
			self.path = path1
		else:
			raise NameError('Path cannot be None')
		