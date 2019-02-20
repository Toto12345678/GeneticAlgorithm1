'''Toto genetic algorithm'''
import base64
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from math import cos, sin
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import os

import random

class Individual(object):
    chromosome_list = []
    fitness = 0
    cdf = 0
    x = 0
    y = 0

    # The class "constructor" - It's actually an initializer 
    def __init__(self, chromosome_list, fitness, cdf, x, y):
        self.chromosome_list = chromosome_list.copy()
        self.fitness = fitness
        self.cdf = cdf
        self.x = x
        self.y = y
    def __repr__(self):
        return str(self.__dict__)

@csrf_exempt
def hola(request):
    a = int(request.POST.get("a")) # a
    b = int(request.POST.get("b")) # b
    c = int(request.POST.get("c")) # c
    d = int(request.POST.get("d")) # d
    n = int(request.POST.get("n")) # population
    option_trim = int(request.POST.get("option_trim")) # trim option: max_size or porcentaje
    poda = int(request.POST.get("poda")) # population
    option_stop = int(request.POST.get("option_stop")) # trim option: max_size or porcentaje
    stop = int(request.POST.get("stop")) # max generation
    mut_ind_prob = float(request.POST.get("mut_ind_prob")) # max generation
    mut_prob_gene = float(request.POST.get("mut_prob_gene")) # max generation

    chromosome_size = 32
    population = [Individual(np.random.randint(2, size=chromosome_size), 0, 0, 0, 0) for i in range(n)] #creates a list of 0 1

    population = evaluate(population, a, b, c, d)
    population = getCDF(population)
    #start cycle
    #values of each generation
    bests= []
    worsts= []
    means= []

    g=0
    if option_stop == 1:
        while g <= stop:
            g= g+1
            best,worst,mean = get_gists(population)
            bests.append(best)
            worsts.append(worst)
            means.append(mean)

            population = crossover(population)
            population = mutate(population,mut_ind_prob,mut_prob_gene)
            population = evaluate(population, a, b, c, d)
            population = getCDF(population)
            trimming(option_trim, poda, n, population)

            #chart
            plot_chart(population, a, b, c, d, g, not (g<=stop))

    if option_stop == 2:
        percentage = 0
        while percentage <= stop:
            g= g+1
            best,worst,mean = get_gists(population)
            bests.append(best)
            worsts.append(worst)
            means.append(mean)

            population = crossover(population)
            population = mutate(population,mut_ind_prob,mut_prob_gene)
            population = evaluate(population, a, b, c, d)
            population = getCDF(population)
            trimming(option_trim, poda, n, population)
            #evaluate stop condition
            values = [individual.fitness for individual in population.copy()] #extract the fitness values of each individual
            percentage = (values.count(max(values))/len(values))*100
            #print('Percentage of max value ('+str(max(values))+') is: '+str(percentage))

            #chart
            plot_chart(population, a, b, c, d, g, not (percentage<=stop))

    # #chart
    # plot_chart(population, a, b, c, d, g, True)
    create_video()
    plot_last(bests, means, worsts, g)
    print('************************************************************************************************************************************************************')
    print(population)
    img = cv2.imread('last_chart.png')
    return JsonResponse({
        'img': str(_encode_Base64(img=img)),
    })

def plot_last(best, mean, worst, g):
    plt.clf()
    fig2 = plt.figure()
    #limits of plot with input given by user
    plt.xlim(0, g)
    #labels
    plt.xlabel('Generación')
    plt.ylabel('Valores de la evaluación de la función')
    #titles
    plt.title("Estadísticas")
    plt.grid(True)
    plt.plot(best,color='cornflowerblue', label='Mejor')
    plt.plot(mean, color='g', label='Media')
    plt.plot(worst, color='m', label='Peor')
    fig2.legend(loc='upper left', fontsize='medium')
    plt.savefig('last_chart.png')

def plot_chart(population, a, b, c, d, g, end=False):
    #show chart
    y_axis = [individual.y for individual in population.copy()]
    x_axis = [individual.x for individual in population.copy()]

    plt.clf()
    fig = plt.figure()  # an empty figure with no axes
    #limits of plot with input given by user
    plt.xlim(a, b)
    plt.ylim(c, d)
    #labels
    plt.xlabel('X')
    plt.ylabel('Y')
    #titles
    plt.title("Valores por generación")
    fig.suptitle('Generación '+str(g))  # Add a title so we know which g it is
    plt.grid(True)
    # colors = x_axis
    # area = 25
    plt.scatter(x_axis, y_axis, s=60, c='#1789fc', alpha=0.4)
    #plt.plot(x_axis, y_axis, marker='o', markersize=10, color="#a1c181", linestyle='none')
    if end:
        plt.plot(x_axis[0], y_axis[0], marker='o', markersize=7, color="#fe7f2d", linestyle='none')

    plt.savefig('img'+str(g)+'.png', bbox_inches='tight')

def get_gists(population,):
    values = [individual.fitness for individual in population.copy()] #extract the values of each individual
    values.sort()
    return values[(len(values)-1)], values[0] ,np.mean(values)

def create_video():
    dir_path = '/Users/toto/Documents/IA/geneticAlgorithm-Back/geneticAlgorithm'
    images = []
    for f in os.listdir(dir_path):
        if f.startswith('img'):
            images.append(f)
    print(images)
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

    for image in images:
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
        out.write(frame) # Write out frame to video
    out.release()
    return 0

# the goal ('fitness') function to be maximized
def evaluate(population, a, b, c, d):
    delta_x = abs(b-a)/pow(2, 16)
    delta_y = abs(d-c)/pow(2, 16)

    # values evaluated in function f(x,y) = x^2 * (cos(x)+sin(y))
    for individual in population:
        split_list = split_seq(individual.chromosome_list.copy(), 2)
        individual.x = a + int(''.join(str(e) for e in split_list[0]), 2) * delta_x
        individual.y = c + int(''.join(str(e) for e in split_list[1]), 2) * delta_y
        individual.fitness = pow(individual.x,2) * ( cos(individual.x) + sin(individual.y))
    return population

def getCDF(population):
    values = [individual.fitness for individual in population.copy()] #extract the fitness values of each individual
    mean, std = np.mean(values), np.std(values, ddof=1)
    # print(values)
    # print('-------------------------------------------------------------------------------------------------------')
    # print('mean: '+str(mean)+'  std: '+str(std))
    standard_normal_dist = (values-mean)/std #get the standard normal distribution of the fitnesses
    cdf = scipy.stats.norm.cdf(standard_normal_dist) #get cdf

    # print('Normal standard dist: '+str(standard_normal_dist))
    # print('CDFs: '+str(cdf))
    # print('-------------------------------------------------------------------------------------------------------')
    i = 0
    for individual in population:
        individual.cdf = cdf[i]
        i= i+1
    population.sort(key=lambda x: x.fitness)
    population = population[::-1].copy()
    return population

def cross(child1, child2):
    child1_new= Individual(child1.chromosome_list.copy(), 0, 0, 0, 0)
    child2_new= Individual(child2.chromosome_list.copy(), 0, 0, 0, 0)
    size = min(len(child1.chromosome_list), len(child2.chromosome_list))
    cxpoints = [random.randint(1,size-1) for i in range(random.randint(1,5))] #generate crossing points and asign it to a list
    cxpoints.sort() #arrange list from small to big
    last = 0 #to store the last position when slicing the list
    for i in range(len(cxpoints)):
        #the actual crossing
        if (i % 2) != 0:
            child1_new.chromosome_list[last:cxpoints[i]] = child2.chromosome_list[last:cxpoints[i]].copy()
            child2_new.chromosome_list[last:cxpoints[i]] = child1.chromosome_list[last:cxpoints[i]].copy()
        last = cxpoints[i]
    if (len(cxpoints) % 2) != 0: #if not pair swap last parts because the first part is swapped the the last has to be swapped as well
        child1_new.chromosome_list[last:len(child1.chromosome_list)] = child2.chromosome_list[last:len(child2.chromosome_list)].copy()
        child2_new.chromosome_list[last:len(child2.chromosome_list)] = child1.chromosome_list[last:len(child1.chromosome_list)].copy()
    else: #if not pair don't swap last parts
        child1_new.chromosome_list[last:len(child1.chromosome_list)] = child1.chromosome_list[last:len(child1.chromosome_list)].copy()
        child2_new.chromosome_list[last:len(child2.chromosome_list)] = child2.chromosome_list[last:len(child2.chromosome_list)].copy()
    return child1_new, child2_new

def crossover(offspring):
    CXPB = 0.5
    offspring_children = []
    first_porcent = int((10 *len(offspring))/100)

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if child1.cdf > CXPB:
            offspring1, offspring2 = cross(child1, child1)
            offspring_children.append(Individual(offspring1.chromosome_list.copy(), 0, 0, 0, 0))
            offspring_children.append(Individual(offspring2.chromosome_list.copy(), 0, 0, 0, 0))
    offspring.extend(offspring_children)
    return offspring

def mutate(offspring,mut_ind_prob,mut_prob_gene):
    for mutant in offspring:
        if random.uniform(0, 1) <= mut_ind_prob:
            mutation(mutant, mut_prob_gene)
    return offspring

def mutation(mutant, mut_prob_gene):
    for i in range(len(mutant.chromosome_list)):
        if random.uniform(0, 1) <= mut_prob_gene:
            if mutant.chromosome_list[i] == 0:
                mutant.chromosome_list[i] = 1
            else:
                mutant.chromosome_list[i] = 0
    mutant.cdf = 0
    mutant.fitness = 0
    mutant.x = 0
    mutant.y = 0
    return mutant

def trimming(option, poda, n, population):
    if option == 1:
        porcent = int(((100-poda) * n / 100))
        del population[porcent:]
    if option ==2:
        del population[poda:]
    return 0

def split_seq(seq, size):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
        newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq

@csrf_exempt
def _encode_Base64(img=None):
    ''' Convert image file to b64 '''
    if img is not None:
        #prepare the image to be sent in b64
        buffer = cv2.imencode('.png', img)[1]
        jpg_as_txt = base64.b64encode(buffer)
        return jpg_as_txt
    else:
        return 0