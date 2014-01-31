import math
from DataSet import *

class Test:
    """
        Static class, don't need to be instantiated
        Use this class just for debug

    """

    isFirstTime = True


    @staticmethod
    def get_max_ndcg(k, *ins):
        """
            from : https://github.com/linshuang/python/blob/master/ndcg.py
            calculates the maximum possible ndcg -> if all docs were sorted by its relevance
        """
        l = [i for i in ins]
        l = copy.copy(l[0])
        l.sort(None,None,True)
        max = 0.0
        for i in range(k):
            max += (math.pow(2, l[i])-1)/math.log(i+2,2)
        return max


    @staticmethod
    def get_ndcg(s, k):
        """
            form : https://github.com/linshuang/python/blob/master/ndcg.py
            calculates the ndcg of a list
        """
        z = Test.get_max_ndcg(k, s)
        dcg = 0.0
        for i in range(k):
            dcg += (math.pow(2, s[i])-1)/math.log(i+2,2)
        if z ==0:
            z = 1
        ndcg = dcg/z
        return ndcg


    @staticmethod
    def print_dcg(dcg,isPerfect):
        """
            stores the values of dcg in a file DCGs.txt
        """
        fich = open('DCGs.txt','a')
        if (isPerfect):
            fich.write('***** Perfect DCG '+str(dcg)+'\n')
        else:
            fich.write('other dcg '+str(dcg)+'\n')
        fich.close()


    @staticmethod
    def print_differences_between_lists(dataset,n_dif,ci,isTrain = False):
        """
            this method was used to print the differences between 2 lists,
            the old and the new one (used in DataSet complex_sort)
        """
        if isTrain:
            fname = 'evol chromosome sort TRAIN.txt'
        else:
            fname = 'evol chromosome sort TEST.txt'

        fich = open(fname,'a')
        if Test.isFirstTime:
            fich.write('Generations '+str(dataset.get_generations())+' '+' Mutation Rate'+str(dataset.get_mutation_rate())+'\n')
            fich.write(str(dataset.get_current_chromosome())+'\n')
            fich.write(Test.neat_chromosome(dataset)+'\n')
            Test.isFirstTime = False
        isAsc = ""
        fi = 0
        if ci != 0:
            fi = dataset.get_feature(ci)
        if dataset.is_ascending(ci):
            isAsc = 'ascending'
        else:
            isAsc = 'descending'

        ndcg = dataset.get_ndcg()

        fich.write('NDCG: '+str(ndcg)+' ordering by gene '+str(fi)+' order '+isAsc+' '+str(n_dif)+' differences found \n')
        fich.close()


    @staticmethod
    def neat_chromosome(dataset):
        """
            it uses the chromosome and neats its representation
            example: having 46 features and having this chromosome:
                input: [85,12,15,32]
                output: [-39.12,15,32]
        """
        c_str = ""
        i=0
        for c in dataset.get_current_chromosome():
            isAsc = dataset.is_ascending(c)
            c = dataset.get_feature(c)
            if not isAsc:
                c*=(-1) # each gene will have the signal - if its order is descending
            c_str += str(c)+" , "
            i+=1
        return c_str


    @staticmethod
    def check_ordered_list(dataset):
        """
            to see the value of features of the current ordering
        """
        
        crom = dataset.get_current_chromosome()
        sCromo = Test.neat_chromosome(dataset)
        
        fich = open('check_ordered_list.txt','a')
        fich.write(sCromo+'\n')
        old_query = -1
        for c in crom:
            f_index = 'f-'
            n_feature = dataset.getFeature(int(c))
            f_index+=str(n_feature)
            fich.write('\n')
            fich.write('\n')
            fich.write(f_index+': \n')
            
            for el in dataset.dataSet:
                if el['query'] != old_query:
                    fich.write('\n ******* new query: '+str(el['query'] )+'\n')
                    old_query = el['query']
                fich.write(str(el['lineNum'])+': '+str(el[f_index])+' ')
            fich.write('\n')
        fich.close()
        

    @staticmethod
    def report_error(entity, message):
        """
            to report a ERROR,
            put a condition on something that shouldn't happen,
            if it happens, report ERROR
        """
        fich = open('ERROR.txt','a')
        fich.write("Error on "+entity+" "+message)
        fich.close()