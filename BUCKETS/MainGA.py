__author__ = 'julianagomes'
import sys
print sys.argv # returns: ['param.py']
import getopt
from pyevolve import G1DList
from pyevolve import GSimpleGA
from DataSet import *
import copy


GENERATIONS = 300 # Number of generations for Genetic Algorithm
POPULATION = 200  #Number of individuals
MUTATION_RATE = 0.315 #Probability of Mutation
CROSS_OVER_RATE =0.8  #Cross-over rate
SIZE_OF_CHROMOSOMES = 20 #the length of each chromosome = X genes (< val - better performance but few diversity, > val slow run but more diversity ...)

TrainSet = None #dataset of train data
TrainSetCopy = None #copy of train data dataset
TestSet = None #dataset of test data

iB = 1  # the b used in the bucket formula can take values from 1
sB = 13 # tO 13 ------> you can configure this number to see other results.

def eval_func(chromosome):
    """
        GA  fitness function
    """
    global TrainSet
    global TrainSetCopy
    TrainSetCopy = copy.copy(TrainSet)
    TrainSetCopy.dataSet = copy.copy(TrainSet.dataSet)
    TrainSetCopy.sort_dataset_by_chromosome(chromosome)
    currentNdcg = TrainSetCopy.get_ndcg()
    return currentNdcg


def main(argv):
    global TrainSet
    global TestSet

    letters = 'i:t'
    keywords = ['input=','test=']
    trainfile =''
    testfile=''

    #run the algorithm by: python MainGA --input=train.txt --test=test.txt
    try:
        opts,arg = getopt.getopt(sys.argv[1:],letters,keywords)
    except getopt.GetoptError:
        print 'GetoptError: -i <trainfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-i','--input'):
            trainfile = arg
        if opt in ('-t','--test'):
            testfile = arg

    if trainfile:
        trainF = open(trainfile, "r")
        TrainSet = DataSet(False,trainF,iB,sB)
        trainF.close()
        TrainSet.set_dataset_filename(str(trainfile))
        TrainSet.set_generations(GENERATIONS)
        TrainSet.set_pop_size(POPULATION)
        TrainSet.set_mutation_rate(MUTATION_RATE)
        genome = G1DList.G1DList(SIZE_OF_CHROMOSOMES)
        #* 2 to ascending and descending sorts, otherwise you'll have just ascending order
        genome.setParams(rangemin= 1, rangemax=(TrainSet.FeaturesNum)*2*sB)
        genome.evaluator.set(eval_func) #change here if you want a different fitness function
        ga = GSimpleGA.GSimpleGA(genome)
        ga.setGenerations(GENERATIONS)      #changes the # of generations(default 100)
        ga.setPopulationSize(POPULATION)    #changes the # of individuals(default 80)
        #ga.setMutationRate(MUTATION_RATE)   # --> use it when you want to change the Mutation Rate (default 0.02)
        #ga.setCrossoverRate(CROSS_OVER_RATE) # --> use it when you want to change the Crossover Rate (default 0.8)
        #ga.setMultiProcessing(True) # --> please read this: http://pyevolve.sourceforge.net/wordpress/?p=843

        ga.evolve(freq_stats=10)


        chromosome = ga.bestIndividual().getInternalList()  # the chromosome selected by GA
        #print chromosome ---> you can print the chromosome to see what was selected
        TrainSet.set_best_individual(chromosome)
        TrainSet.sort_dataset_by_chromosome(chromosome)
        TrainSet.write_scores(isTrain=True)


    else:
        sys.exit('GA_Algorithm: A train file is required.')


    if testfile:
        testF = open(testfile,'r')
        TestSet = DataSet(True,testF,iB,sB)
        testF.close()
        TestSet.set_dataset_filename(str(testfile))
        TestSet.set_generations(GENERATIONS)
        TestSet.set_pop_size(POPULATION)
        TestSet.set_mutation_rate(MUTATION_RATE)
        TestSet.isTest = True
        TestSet.sort_dataset_by_chromosome(TrainSet.get_best_individual()) #order the testset with chromosome found by GA with trainset
        TestSet.write_scores()


if __name__ == "__main__":
    main(sys.argv[1:])
