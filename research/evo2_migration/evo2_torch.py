"""
Evo2: Simple Evolutionary Algorithm Model
PyTorch Implementation

This module implements the Evo2 evolutionary algorithm for optimization problems.
It uses a population-based approach with selection, crossover, and mutation operations.
"""

import torch
import numpy as np
import random


class Evo2Model:
    """
    Evo2 is a simple evolutionary algorithm model for optimization problems.
    
    The model maintains a population of candidate solutions (individuals),
    evaluates their fitness, selects parents based on fitness, creates new
    solutions through crossover and mutation, and replaces the old population.
    """
    
    def __init__(self, 
                 population_size=100, 
                 individual_size=10, 
                 mutation_rate=0.1, 
                 crossover_rate=0.7,
                 selection_pressure=2.0,
                 device='cpu'):
        """
        Initialize the Evo2 model.
        
        Args:
            population_size (int): Number of individuals in the population
            individual_size (int): Size of each individual (number of genes)
            mutation_rate (float): Probability of gene mutation
            crossover_rate (float): Probability of crossover between individuals
            selection_pressure (float): Selection pressure parameter
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.population_size = population_size
        self.individual_size = individual_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        self.device = device
        
        # Initialize population randomly between -1 and 1
        self.population = torch.rand((population_size, individual_size), 
                                     device=device) * 2 - 1
        self.fitness_scores = torch.zeros(population_size, device=device)
        
    def evaluate_fitness(self, fitness_function):
        """
        Evaluate the fitness of all individuals in the population.
        
        Args:
            fitness_function: Function that takes an individual and returns its fitness
        """
        for i in range(self.population_size):
            self.fitness_scores[i] = fitness_function(self.population[i])
        
    def select_parents(self):
        """
        Select parents for reproduction using tournament selection.
        
        Returns:
            Tensor containing selected parent indices
        """
        parent_indices = torch.zeros(self.population_size, dtype=torch.long, 
                                    device=self.device)
        
        for i in range(self.population_size):
            # Select random candidates for tournament
            candidates = torch.randint(0, self.population_size, (2,), device=self.device)
            # Select the one with better fitness (lower value for minimization problems)
            if self.fitness_scores[candidates[0]] < self.fitness_scores[candidates[1]]:
                parent_indices[i] = candidates[0]
            else:
                parent_indices[i] = candidates[1]
                
        return parent_indices
    
    def crossover(self, parents_indices):
        """
        Perform crossover operation between selected parents.
        
        Args:
            parents_indices: Tensor of parent indices
            
        Returns:
            Tensor containing the new population after crossover
        """
        new_population = torch.zeros_like(self.population)
        
        # Shuffle parents to create pairs
        shuffled_indices = parents_indices[torch.randperm(self.population_size)]
        
        for i in range(0, self.population_size, 2):
            parent1_idx = parents_indices[i]
            parent2_idx = shuffled_indices[i]
            
            # Check if crossover should occur
            if random.random() < self.crossover_rate and i+1 < self.population_size:
                # Single-point crossover
                crossover_point = random.randint(1, self.individual_size - 1)
                
                # Create two children
                new_population[i, :crossover_point] = self.population[parent1_idx, :crossover_point]
                new_population[i, crossover_point:] = self.population[parent2_idx, crossover_point:]
                
                if i+1 < self.population_size:
                    new_population[i+1, :crossover_point] = self.population[parent2_idx, :crossover_point]
                    new_population[i+1, crossover_point:] = self.population[parent1_idx, crossover_point:]
            else:
                # If no crossover, just copy the parents
                new_population[i] = self.population[parent1_idx]
                if i+1 < self.population_size:
                    new_population[i+1] = self.population[parent2_idx]
        
        return new_population
    
    def mutate(self, population):
        """
        Apply mutation to the population.
        
        Args:
            population: The population to mutate
            
        Returns:
            Mutated population
        """
        # Create a mask for mutation (True where mutation should occur)
        mutation_mask = torch.rand_like(population) < self.mutation_rate
        
        # Generate random values for mutation
        mutations = torch.randn_like(population) * 0.2
        
        # Apply mutations only where the mask is True
        mutated_population = torch.where(mutation_mask, population + mutations, population)
        
        return mutated_population
    
    def evolve(self, fitness_function, generations=100):
        """
        Evolve the population for a specified number of generations.
        
        Args:
            fitness_function: Function to evaluate individual fitness
            generations: Number of generations to evolve
            
        Returns:
            Best individual found and its fitness
        """
        for generation in range(generations):
            # Evaluate fitness
            self.evaluate_fitness(fitness_function)
            
            # Select parents
            parent_indices = self.select_parents()
            
            # Perform crossover
            new_population = self.crossover(parent_indices)
            
            # Perform mutation
            new_population = self.mutate(new_population)
            
            # Replace population
            self.population = new_population
        
        # Final evaluation
        self.evaluate_fitness(fitness_function)
        
        # Find the best individual
        best_idx = torch.argmin(self.fitness_scores)
        best_individual = self.population[best_idx]
        best_fitness = self.fitness_scores[best_idx]
        
        return best_individual, best_fitness


# Example usage
def example_usage():
    # Define a simple fitness function (minimize the sum of squares)
    def fitness_function(individual):
        return torch.sum(individual ** 2)
    
    # Initialize the model
    model = Evo2Model(
        population_size=100,
        individual_size=10,
        mutation_rate=0.1,
        crossover_rate=0.7
    )
    
    # Evolve the model
    best_solution, best_fitness = model.evolve(fitness_function, generations=100)
    
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")


if __name__ == "__main__":
    example_usage()