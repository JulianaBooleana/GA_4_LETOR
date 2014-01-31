from sys import stdout
import numpy as np
import re
import math
import copy
import subprocess
from Test import *
from Feature import *

#EVAL_TOOL CONSTANTS
SCRIPT_DATASET ="Eval-Score-4.0.pl"    # eval-score-mslr.pl for OSHUMED, Eval-Score-4.0.pl for MQ2008
RESULTS_P_QUERY = "0" #to return results averaged by query assign 0, otherwise assign 1

class DataSet:
    """

        This class is the representation of a DataSet
        The DataSet is in feature_Vectors plus id_query+id_doc and score
        contains all the operations to learn to rank with GA

    """

    isTest = False #True if the dataset is Test, False if it is Train
    DCG = 0 #Discounted cumulative Gain of a particular order
    perfectDCG = -1 #DCG of the perfect order
    FeaturesNum = 0 # Number of Features of a particular dataSet
    dataType = None #Numpy dataType of the structure of dataSet
    dataSet = None #the representation of the DataSet in features vector
    currentChromosome = None #the current chromosome
    BestIndividual = None #the best individual after all generations of GA
    DocCounter = 0 #a counter to count the # of docs of dataset
    
    MutationRate = None
    PopulationSize = None
    Generations = None

    DataSetFileName = None #typically test.txt or train.txt

    iB = None #b used in the formula of Feature.get_bucket()
    sB = None #
    MinMaxFeatures = {} #this var stores the min and max of each feature. It is used to calc the Feature- bucket formula

    def __init__(self, isT, data, iB, slot_size):
        """
         the class constructor creates the dataset with its features vectors
        """
        self.isTest = isT
        lineNumber = 1
        self.DocCounter = 0
        self.iB =iB
        self.sB = slot_size
        for queryDoc in data:
            if lineNumber==1:
                self.set_features_number(queryDoc)
                self.calc_data_type()
                self.dataSet = np.recarray((1,),self.dataType)
                self.dataSet[0] = self.parse_query_doc(queryDoc,lineNumber)
            else:
                self.dataSet = np.append(self.dataSet, self.parse_query_doc(queryDoc,lineNumber))
            lineNumber+=1


    def calc_data_type(self):
        """
            creates the dtype of the numpy array
        """
        self.dataType = [('query','i4'),('doc','S50'),('rel','i4'),('lineNum','i4'),('score','i4')]
        i = 1
        while i <= self.FeaturesNum:
            self.dataType.append(('f-'+str(i),'float64'))
            i+=1


    def get_doc_for_line(self,line):
        """
            returns the doc id
        """
        doc = line.index('#docid = ')
        doc_and_after = line[doc+9:]
        try:
            Space_i = doc_and_after.index(' ')
            doc_and_after =  doc_and_after[:Space_i]

            return doc_and_after
        except:
            doc_and_after.replace('\n','')
            doc_and_after.replace('\r','')
            return doc_and_after


    def get_query_id(self,line):
        """
            returns the query id
        """
        try:
            qid = line.index('qid:')
            query_id = line[qid+4:]
            space = query_id.index(' ')
            return query_id[:space]
        except:
            return ""


    def parse_query_doc(self, line, lineNum):
        """
            parses a line of the training files
        """
        self.DocCounter += 1
        doc =[]
        score  = int(line[0])
        query = self.get_query_id(line)
        doc.insert(0,query)
        doc.insert(1,str(self.DocCounter))
        doc.insert(2,score)
        doc.insert(3,lineNum)
        doc.insert(4,-1)
        fi = 5
        i=1
        while(i <= self.FeaturesNum):
            str_i = str(i)
            f_index = line.index(str_i+':')
            offset = len(str_i)+1
            feature = line[f_index+offset:]
            space_index = feature.index(' ')
            feature = feature[: space_index]
            ffeature = float(feature)
            doc.insert(fi,ffeature)
            if lineNum == 1:
                fval = Feature(ffeature,ffeature,self.iB)
                self.MinMaxFeatures[i] = fval
            else:
                self.MinMaxFeatures[i].update_min_max(ffeature)
            i+=1
            fi+=1

        dataset = np.array(tuple(doc),dtype=self.dataType)
        return dataset
    

    def sort_ideal(self):
        """
            sorts dataset according relevance
        """
        self.dataSet = np.sort(self.dataSet,order=['query','rel'])[::-1]



    def set_features_number(self,line):
        """
            calcs the number of features and sets info in dataSet
        """
        inv_line = line.split(' ')
        inv_line = inv_line[::-1]
        regex = re.compile('^[0-5]*[0-9]+:')
        for e in inv_line:
            if regex.match(e):
                i = e.index(':')
                e = e[:i]
                self.FeaturesNum = int(e)
                break


    def get_feature(self,c):
        """
            receives the gene and returns the Feature num in dataSet
        """
        TotalFeatNum = self.FeaturesNum
        num = math.fmod(c,TotalFeatNum)
        num = int(num)
        if num == 0:
            num = TotalFeatNum
        return num


    def get_b(self,g):
        """
            receives a gene from the chromosome and returns the b used in the bucket formula
        """
        i=0
        b = self.iB
        N = g/self.FeaturesNum
        rN = g%self.FeaturesNum
        if rN == 0:
            N -=1
        while i <= self.sB:
            if N == i or N == (self.sB +i):
                b+=i
                break
            i+=1
        return b


    def put_in_buckets(self,chromosome):
        """
            for each gene in chromosome it will buck the corresponding feature's
            values according its respective b
        """
        # ---> if you want to buck just the N features in chromosome#chromosome = chromomsome[:N]
        for c in chromosome:
            feature = self.get_feature(c)
            b = self.get_b(c)
            self.put_feature_in_buckets(feature,b)


    def put_feature_in_buckets(self,feature,b):
        """
            for each line of the dataset, update each value
            according its bucket
        """
        f_index = 'f-'+str(feature)
        self.MinMaxFeatures[feature].set_b(b)
        for el in self.dataSet:
            el[f_index] =  self.MinMaxFeatures[feature].get_bucket(el[f_index])


    def remove_duplicate_features(self,chromosome):
        """
            receives the chromosome and removes the duplicate features
        """
        freeDups = []
        newChromosome = []
        for gene in chromosome:
            feature = self.get_feature(gene)
            if feature not in freeDups:
                freeDups.append(feature)
                newChromosome.append(gene)
        return newChromosome


    def is_ascending(self,c):
        """
            receives a gene
            returns True if gene <= total # of features *sB-> order ascending
            returns False otherwise -> order descending
        """
        if c > (self.FeaturesNum*self.sB):
            return False
        else:
            return True


    def get_order_by_chromosome(self, N=-1):
        """
            returns a order to be used by the numpy.lexort method to sort the dataset based on the current chromosome
        """
        chromo_order = []
        chromosomeR = None
        if N <= 0:
            chromosomeR = self.currentChromosome[::-1]
        else:
            chromosomeR = self.currentChromosome[:N]
            chromosomeR = chromosomeR[::-1]
        for c in chromosomeR:
            f_index = 'f-'
            gene = int(c)
            n_feature = self.get_feature(gene)
            f_index += str(n_feature)
            if self.is_ascending(gene):
                chromo_order.append(self.dataSet[f_index])
            else:
                chromo_order.append(- self.dataSet[f_index])
        chromo_order.append(self.dataSet['query'])
        return chromo_order


    def sort_dataset_by_chromosome(self, chromosome):
        """
            receives a chromosome and sorts dataset according it
        """
        newChromosome = self.remove_duplicate_features(chromosome)
        self.set_current_chromosome(newChromosome)
        self.put_in_buckets(newChromosome)
        order = self.get_order_by_chromosome()
        self.dataSet = self.dataSet[np.lexsort(order)]



    def complex_sort(self,chromosome):
        """
            sorts the dataset step by step ; gene by gene
            use for debug it is useful to check the differences
        """
        i =1
        newChromosome = self.remove_duplicate_features(chromosome)
        self.set_current_chromosome(newChromosome)
        self.put_in_buckets(newChromosome)
        cromo_len = len(newChromosome)
        oldList = copy.copy(self.dataSet)
        while i <= cromo_len:
            current_order = self.get_order_by_chromosome(i)
            self.dataSet = self.dataSet[np.lexsort(current_order)]
            Test.checkOrderedList(self)
            n_diff = self.compare_the_differences(oldList)
            Test.print_differences_between_lists(self,n_diff,newChromosome[i-1])
            oldList = copy.copy(self.dataSet)
            i+=1
        fich=open('complex_sort.txt','a')
        fich.write(str(chromosome)+'original chromosome \n')
        fich.write(str(newChromosome)+' used chromosome \n')
        fich.close()


    def compare_the_differences(self, oldList):
        """
            method used for debug
            returns the # of differences between a previous sort and a current sort
        """
        diff = 0
        self.dataSet = self.assign_scores(self.dataSet)
        oldList = self.assign_scores(oldList)
        for el in oldList:
            arr_aux = np.array(self.dataSet[self.dataSet['lineNum']== el.lineNum], dtype = self.dataType)
            if arr_aux[0].score != el.score:
                #DEBUG CODE:
                if(el.lineNum != arr_aux[0].lineNum):
                    Test.report_error('compare_the_differences','comparing score elements with different lineNums')
                elif(el.query != arr_aux[0].query):
                    Test.report_error('compare_the_differences','comparing score elements with different queries')
                diff+=1
        return diff


    def assign_scores(self,corpus):
        """
            assigns the scores of the dataSet
            the eval_tools needs to know the scores of dataset
            to generate the evaluation metrics: NDCG & MAP
        """
        rank_pos =1
        current_q = None
        for q in corpus:
            if current_q != q.query:
                rank_pos = len(corpus[corpus['query'] == q.query])
                current_q = q.query
                q.score = rank_pos
            else:
                rank_pos -=1
                q.score = rank_pos
        return corpus


    def get_ndcg(self):
        """
            returns the value of NDCG of the current dataSet
        """
        DCGacm = 0.0
        query_counter = 0
        current_q = -1
        relevance_scores = None
        n = 1
        for qline in self.dataSet:
            if current_q != qline.query:
                if relevance_scores != None:
                    if(len(relevance_scores)< n):
                        DCGacm += Test.get_ndcg(relevance_scores,len(relevance_scores))
                    else:
                        DCGacm += Test.get_ndcg(relevance_scores,n)

                relevance_scores = []
                current_q = qline.query
                query_counter = query_counter+1
                n = 1
                #if n <= 10: -----> NDCG@10
            relevance_scores.append(qline.rel)
            n+=1
        DCGacm += Test.get_ndcg(relevance_scores,len(relevance_scores))
        ndcg= ((DCGacm)/float(query_counter))
        return ndcg

    
    def write_scores(self,isTrain=False):
        """
            writes the scores of the dataset, so eval_tools
            can generate the evaluation metrics NDCG+MAP
        """
        self.assign_scores(self.dataSet)
        filename = "scores"+self.DataSetFileName
        current_q = -1
        rank_pos = 1
        for q in self.dataSet:
            if current_q != q.query:
                rank_pos = len(self.dataSet[self.dataSet['query']==q.query])
                current_q = q.query
                q.score = rank_pos
            else:
                rank_pos -=1
                q.score = rank_pos
        test_set = np.sort(self.dataSet, order=['lineNum'])
        text_file = open(filename, "w")
        for qr in test_set:
            text_file.write(str(qr.score)+'\n')
        text_file.close()
        pipe = subprocess.Popen(["perl", SCRIPT_DATASET ,self.DataSetFileName , filename ,"metrics_"+self.DataSetFileName, RESULTS_P_QUERY], stdout = subprocess.PIPE)

    
    def set_best_individual(self,Best):
        """
            sets BestIndividual - chromosome (the best individual is the one selected by GA)
        """
        self.BestIndividual = Best


    def get_best_individual(self):
        """
            gets the BestIndividual property
        """
        return self.BestIndividual


    def set_current_chromosome(self,c):
        """
            sets the current chromosome
        """
        self.currentChromosome = c

    
    def get_current_chromosome(self):
        """
            gets the current chromosome
        """
        return self.currentChromosome

    def set_mutation_rate(self,mr):
        """
            sets the value of mut. rate
            just to have this info in dataset
        """
        self.MutationRate = mr


    def get_mutation_rate(self):
        """
            gets the current chromosome
        """
        return self.MutationRate

    def set_pop_size(self,ps):
        """
            sets the value of pop size
            just to have this info in dataset
        """
        self.PopulationSize = ps


    def get_pop_size(self):
        """
            gets the Population Size
        """
        return self.PopulationSize

    def set_generations(self,g):
        """
            sets the value of generations
            just to have this info in dataset
        """
        self.Generations = g


    def get_generations(self):
        """
            gets the # of Generations
        """
        return self.Generations

    def set_dataset_filename(self,name):
        self.DataSetFileName = name
