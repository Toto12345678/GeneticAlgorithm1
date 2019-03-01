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

    # The class "constructor" - It's actually an initializer 
    def __init__(self, chromosome_list, fitness, value):
        self.chromosome_list = chromosome_list.copy()
        self.fitness = fitness
        self.value = value
    def __repr__(self):
        return str(self.__dict__)


@csrf_exempt
def calculate_nutritional_needs(request):
    #get parameters
    name = request.POST.get("name")
    age_factor = float(request.POST.get("age_factor"))
    sex = request.POST.get("sex")
    height = float(request.POST.get("height"))
    weight = float(request.POST.get("weight"))
    activity = float(request.POST.get("activity"))
    stress_factor = float(request.POST.get("stress_factor"))

    #process
    if sex == 'mujer':
        if height >= 150: 
            ideal_weight = 0.9 * (height-100)
            bmr = 25 * ideal_weight
        else:
            ideal_weight = height - 100
            bmr = 25 * ideal_weight
    if sex == 'hombre': 
        if height >= 160:
            ideal_weight = 0.9 * (height-100)
            bmr = 30 * ideal_weight
        else:
            ideal_weight = height - 100
            bmr = 30 * ideal_weight
    
    status_result = (weight / ideal_weight) * 100
    if status_result > 110:
        nutrition_status = -0.20
    if status_result <= 110 and status_result >= 90:
        nutrition_status = 0
    if status_result < 90:
        nutrition_status = 0.20

    energy = bmr + (activity * bmr) - (age_factor * bmr) + stress_factor + (nutrition_status * bmr)
    percentages= getPercentages(energy)
    return JsonResponse({
        'weight':weight,
        'bmr': bmr,
        'energy': energy,
        'protein': percentages[0],
        'fat': percentages[1],
        'carbs': percentages[2]
    })


@csrf_exempt
def getDiet(request):#just want energy for this
    # queryset = Food.objects.all()
    # serializer = FoodsSerializers(queryset, many = True) #get alldata in database
    # print(type(serializer.data[0]))
    #get params
    n = int(request.POST.get("n"))
    chro_size = int(request.POST.get("chro_size"))
    max_gen = int(request.POST.get("max_gen"))
    crossover_prob = float(request.POST.get("crossover_prob"))
    mut_prob = float(request.POST.get("mut_prob"))
    energy = float(request.POST.get("energy"))

    bests = []

    population = [Individual([ FoodsSerializers(Food.objects.get(pk=random.randint(1,174))).data for food in range(chro_size)], 0, 0) for i in range(n)] #creates n lists of menues
    population = evaluate(population, energy)
    population = roulette_select(population)
    bests.append(get_gists(population))
    g=0
    while g < max_gen:
        population = crossover(population, crossover_prob)
        trimming(n, population)
        population = mutate(population, mut_prob, chro_size, n)
        population = evaluate(population, energy)
        population = roulette_select(population)
        bests.append(get_gists(population))
        g= g + 1
    population.sort(key=lambda x: x.fitness)
    population = population[::-1].copy()
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
    #plt.plot(x_axis[0], y_axis[0], marker='o', markersize=7, color="#fe7f2d", linestyle='none')
    fig2.legend(loc='upper left', fontsize='medium')
    plt.savefig('chart_origin.png')


def get_gists(population):
    population.sort(key=lambda x: x.fitness)
    population_copy = population[::-1].copy() #ordered from biggest to smallest
    return population_copy[0].value

def getPercentages(energy):
    protein = (energy * 0.15)/4
    fat = (energy * 0.25)/9
    carbs = (energy * 0.60)/4
    return [protein, fat, carbs]

def evaluate(population, energy):
    population = get_energy(population)
    for individual in population:
        individual.fitness = 1/(abs(energy - individual.value))
    return population.copy()

def roulette_select(population):
    total_fitness = sum(food.fitness for food in population)
    relative_fitness = [f.fitness/total_fitness for f in population]
    probability_intervals= [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
    new_population = []
    i=0
    while i < len(population):
        r = random.uniform(0, 1)
        for (j, individual) in enumerate(population):
            if r <= probability_intervals[j]:
                new_population.append(individual)
                i = i + 1
                break
    return new_population

def get_energy(population):
    for individual in population:
        individual.value = 0 #resset value
        for food in individual.chromosome_list:
            individual.value += food['kcal'] #get value of each individual
    return population.copy()

def cross(child1, child2):
    child1_new= Individual(child1.chromosome_list.copy(), 0, 0)
    child2_new= Individual(child2.chromosome_list.copy(), 0, 0)
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
        if random.uniform(0, 1) <= crossover_prob:
            offspring1, offspring2 = cross(child1, child2)
            offspring_children.append(Individual(offspring1.chromosome_list.copy(), 0, 0))
            offspring_children.append(Individual(offspring2.chromosome_list.copy(), 0, 0))
    offspring.extend(offspring_children)
    offspring = offspring[::-1].copy() #invert so children don't go away
    return offspring

def trimming(n, population):
    del population[n:]
    return 0

def mutate(offspring, mut_prob, chro_size, n):
    genes_to_mutate = (n* chro_size) * mut_prob
    changed= []
    i= 0
    rand = random.randint(1,n*chro_size) #or it coul be n*chro_size
    while i < genes_to_mutate:
        if not(rand in changed):
            offspring[math.ceil(rand/chro_size)-1].chromosome_list[rand - (int(rand/chro_size) * chro_size)-1] = FoodsSerializers(Food.objects.get(pk=random.randint(1,174))).data
            changed.append(rand)
            i= i+1
        rand = random.randint(1,n*chro_size)
    return offspring