from django.views.decorators.csrf import csrf_exempt
from diet_decision_diabetics.models import Food
from diet_decision_diabetics.serializers import FoodsSerializers
from django.http import HttpResponse, JsonResponse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math

class Individual(object):
    chromosome_list = [] #list containing n food
    fitness = 0 #against function
    value = 0 #of the menu in the function
    fitness_relative = 0
    # The class "constructor" - It's actually an initializer 
    def __init__(self, chromosome_list, fitness, value, fitness_relative):
        self.chromosome_list = chromosome_list.copy()
        self.fitness = fitness
        self.value = value
        self.fitness_relative = fitness_relative
    def __repr__(self):
        return str(self.__dict__)

@csrf_exempt
def getDietAltered(request):#just want energy for this
    #get params
    n = int(request.POST.get("n"))
    chro_size = int(request.POST.get("chro_size"))
    option_stop = int(request.POST.get("option_stop"))
    stop = int(request.POST.get("stop"))
    crossover_prob = float(request.POST.get("crossover_prob"))
    mut_prob = float(request.POST.get("mut_prob"))
    option_trim = int(request.POST.get("option_trim")) # trim option: max_size or porcentaje
    poda = int(request.POST.get("poda")) # population
    energy = float(request.POST.get("energy"))

    bests = []

    population = [Individual([ FoodsSerializers(Food.objects.get(pk=random.randint(1,174))).data for food in range(chro_size)], 0, 0, 0) for i in range(n)] #creates n lists of menues
    population = evaluate(population, energy)
    bests.append(population[0].value)
    g=0
    if option_stop == 1:
        while g < stop:
            population = crossover(population, crossover_prob)
            population = mutate(population, mut_prob, chro_size, n)
            trimming(option_trim, poda, n, population)
            population = evaluate(population, energy)
            bests.append(population[0].value)
            g= g + 1
    if option_stop == 2:
        percentage = 3000
        while percentage > stop:
            population = crossover(population, crossover_prob)
            population = mutate(population, mut_prob, chro_size, n)
            trimming(option_trim, poda, n, population)
            population = evaluate(population, energy)
            bests.append(population[0].value)
            percentage = abs(energy - population[0].value)
            g= g + 1
    plot_last(energy,bests, g)
    return JsonResponse({'result' : population[0].chromosome_list})

def plot_last(energy, best, g):
    #plt.clf()
    fig2 = plt.figure()
    #limits of plot with input given by user
    plt.xlim(0, g)
    #labels
    plt.xlabel('Generación')
    plt.ylabel('kcal')
    #titles
    plt.title("Estadísticas")
    plt.grid(True)
    plt.axhline(y=energy, color='r', linestyle='-', label='Energía calculada')
    plt.plot(best,color='cornflowerblue', label='Mejor')
    fig2.legend(loc='upper left', fontsize='medium')
    plt.savefig('altered_chart.png')

def evaluate(population, energy):
    population = get_energy(population)
    for individual in population:
        individual.fitness = 1/(abs(energy - individual.value))
    evaluate_relative(population)
    population.sort(key=lambda x: x.fitness)
    return population[::-1].copy() #ordered from biggest to smallest

def evaluate_relative(population):
    total_fitness = sum(food.fitness for food in population)
    for food in population:
        food.fitness_relative = food.fitness / total_fitness

def get_energy(population):
    for individual in population:
        individual.value = 0 #resset value
        for food in individual.chromosome_list:
            individual.value += food['kcal'] #get value of each individual
    return population.copy()

def cross(child1, child2):
    child1_new= Individual(child1.chromosome_list.copy(), 0, 0, 0)
    child2_new= Individual(child2.chromosome_list.copy(), 0, 0, 0)
    last = int(len(child1.chromosome_list)/2)
    
    child1_new.chromosome_list[:last] = child1.chromosome_list[:last].copy()
    child1_new.chromosome_list[last:] = child2.chromosome_list[last:].copy()
    last = int(len(child1.chromosome_list)/2)
    child2_new.chromosome_list[:last] = child2.chromosome_list[:last].copy()
    child2_new.chromosome_list[:last] = child1.chromosome_list[last:].copy()
    return child1_new, child2_new

def crossover(offspring, crossover_prob):
    offspring_children = []
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if child1.fitness_relative >= crossover_prob:
            offspring1, offspring2 = cross(child1, child2)
            offspring_children.append(Individual(offspring1.chromosome_list.copy(), 0, 0, 0))
            offspring_children.append(Individual(offspring2.chromosome_list.copy(), 0, 0, 0))
    offspring.extend(offspring_children)
    offspring.sort(key=lambda x: x.fitness)
    return offspring[::-1].copy()

def trimming(option, poda, n, population):
    if option == 1:
        porcent = int(((100-poda) * n / 100))
        del population[porcent:]
    if option == 2:
        del population[poda:]
    return 0

def mutate(offspring, mut_prob, chro_size, n):
    genes_to_mutate = (len(offspring) * chro_size) * mut_prob
    changed= []
    i= 0
    rand = random.randint(1,len(offspring)*chro_size) #or it coul be n*chro_size
    while i < genes_to_mutate:
        if not(rand in changed):
            offspring[math.ceil(rand/chro_size)-1].chromosome_list[rand - (int(rand/chro_size) * chro_size)-1] = FoodsSerializers(Food.objects.get(pk=random.randint(1,174))).data
            changed.append(rand)
            i= i+1
        rand = random.randint(1,len(offspring)*chro_size)
    offspring.sort(key=lambda x: x.fitness)
    return offspring[::-1].copy()

def getPercentages(energy):
    protein = (energy * 0.15)/4
    fat = (energy * 0.25)/9
    carbs = (energy * 0.60)/4
    return [protein, fat, carbs]