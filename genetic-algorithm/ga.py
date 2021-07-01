import numpy.random as rd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (19, 8)

def genetic(max_generations=100,population_size=10,print_info=False):
    population = gen_start_population(size=population_size)

    history_individual_vars=[sort_population(population)[-1]]
    history_fitness_vars=[f(sort_population(population)[-1])]
   
    for step in range(max_generations):
        if print_info:
          print_data(step, population)
        population = make_next_generation(population)
        history_individual_vars.append(sort_population(population)[-1])
        history_fitness_vars.append(f(sort_population(population)[-1]))
    best_individual = sort_population(population)[-1]
    best_fitness = f(best_individual)

    return best_individual , best_fitness, population,history_individual_vars,history_fitness_vars

def visualize(history_individual_vars,history_fitness_vars, best_individual,best_fitness,population):
    fig, axs = plt.subplots(3)
    axs[0].plot(history_individual_vars, 'r')
    axs[0].set_title("Изменение значений x")
    axs[1].plot(history_fitness_vars, 'b')
    axs[1].set_title("Изменение значений целевой функции")
    a, b=interval
    params= np.arange(a, b, 0.01)
    res=[f(x) for x in params]
    axs[2].plot(params, res,color='g')
    for individual in population:
        axs[2].scatter(individual, f(individual), s=30,color='b')
    axs[2].scatter(best_individual, best_fitness, s=30,color='r')
    axs[2].set_title("Результат")

def print_data(step, population):
    print("Итерация: ",step)
    print("Популяция : ",population)
    print("Лучшее значение целевой функции(приспособленности) в популяции = ", f(sort_population(population)[-1]))

def gen_start_population(size):
    # a,b = interval
    # population = []
    # for i in range(size):
    #     x= rd.uniform(a,b)
    #     population.append(x)
    # return population
    return [0.01]*size

def f(x):
    return x*x*(2+abs(math.sin(4*x)))

def make_choice_roulette(sorted_population):
    fitness_sum = sum(f(individual) for individual in sorted_population)

    rand_var = rd.uniform(0, 1)
    sum_probability = 0
    for individual in sorted_population:
        fitness = f(individual)
        probability = fitness / fitness_sum
        sum_probability += probability
        if rand_var <= sum_probability:
            return individual

def bound(x):
    a, b = interval
    return min(max(x, a), b)


def sort_population(population):
    # return sorted(population, key=f, reverse=True)
    return sorted(population, key=f)


def crossover(first_individual, second_individual):
    return (first_individual + second_individual) / 2


def mutate(individual):
    next_x = individual + rd.uniform(-0.5, 0.5)
    next_x = bound(next_x)
    return  next_x

def selection(previous_population):
    sorted_population = sort_population(previous_population)
    selected_individuals = []
    for i in range(population_size):
        selected_individuals.append(make_choice_roulette(sorted_population)) 
    return selected_individuals

def parents(selected_individuals):
    sorted_selection=sort_population(selected_individuals)
    first_parent=sorted_selection[-1]
    second_parent=sorted_selection[rd.randint(0,len(selected_individuals)-2)]
    return first_parent, second_parent

def make_next_generation(previous_population):
    next_generation = []
    selected_individuals = selection(previous_population)
    first_parent, second_parent = parents(selected_individuals)
    for i in range(population_size):
        individual = crossover(first_parent, second_parent)
        individual = mutate(individual)
        next_generation.append(individual)
    return next_generation

interval=(-10,10)
population_size=10
best_individual , best_fitness, population,history_individual_vars ,history_fitness_vars = genetic(max_generations=100,population_size=10,print_info=False)
print("-"*11,"Результат","-"*11)
print("Популяция : ",population)
print("Лучший элемент популяции = ",best_individual)
print("Значение целевой функции(приспособленности) = ", best_fitness)
visualize(history_individual_vars,history_fitness_vars, best_individual,best_fitness,population)