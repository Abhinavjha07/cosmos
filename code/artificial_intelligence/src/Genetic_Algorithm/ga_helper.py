import numpy as np;
import math;

class maths:
    def countcentres(chromosome):
        return chromosome[chromosome.argmax()] + 1

   
    def clustersrepresentation(chromosome, countclusters=None):
        if countclusters is None:
            countclusters = maths.countcentres(chromosome)

        # Initialize empty clusters
        clusters = [[] for i in range(countclusters)]

        # Fill clusters with index of data
        for idxdata in range(len(chromosome)):
            clusters[chromosome[idxdata]].append(idxdata)

        return clusters

    
    def getcentres(chromosomes, data, countclusters):
        

        centres = maths.calccenters(chromosomes, data, countclusters)

        return centres

    
    def calccenters(chromosomes, data, countclusters=None):
        

        if countclusters is None:
            countclusters = maths.countcentres(chromosomes[0])

        # Initialize center
        centers = np.zeros(shape=(len(chromosomes), countclusters, len(data[0])))

        for idxchromosome in range(len(chromosomes)):

            # Get count data in clusters
            countdataincluster = np.zeros(countclusters)

            # Next data point
            for idx in range(len(chromosomes[idxchromosome])):

                clusternum = chromosomes[idxchromosome][idx]

                centers[idxchromosome][clusternum] += data[idx]
                countdataincluster[clusternum] += 1

            for idxcluster in range(countclusters):
                if countdataincluster[idxcluster] != 0:
                    centers[idxchromosome][idxcluster] /= countdataincluster[idxcluster]

        return centers

    
    def calcprobabilityvector(fitness):
        

        # Get 1/fitness function
        invfitness = np.zeros(len(fitness))

        #
        for idx in range(len(invfitness)):

            if fitness[idx] != 0.0:
                invfitness[idx] = 1.0 / fitness[idx]
            else:
                invfitness[idx] = 0.0

        # Initialize vector
        prob = np.zeros(len(fitness))

        # Initialize first element
        prob[0] = invfitness[0]

        # Accumulate values in probability vector
        for idx in range(1, len(invfitness)):
            prob[idx] = prob[idx - 1] + invfitness[idx]

        # Normalize
        prob /= prob[-1]

        maths.setlastvaluetoone(prob)

        return prob


    def setlastvaluetoone(probabilities):
       
        # Start from the last elem
        backidx = - 1

        # All values equal to the last elem should be set to 1
        lastval = probabilities[backidx]

        # for all elements or if a elem not equal to the last elem
        for i in range(-1, -len(probabilities) - 1):
            if probabilities[backidx] == lastval:
                probabilities[backidx] = 1
            else:
                break

    def getuniform(probabilities):
        

        # Initialize return value
        residx = None

        # Get random num in range [0, 1)
        randomnum = np.random.rand()

        # Find segment with  val1 < randomnum < val2
        for idx in range(len(probabilities)):
            if randomnum < probabilities[idx]:
                residx = idx
                break

        return residx

 
class GA:
    

    def __init__(self, data, countclusters, chromosomecount, populationcount, countmutationgens=2,
                 coeffmutationcount=0.25, selectcoeff=1.0):
       
        # Initialize random
        np.random.seed()

        # Clustering data
        if type(data) is list:
            self.data = np.array(data)
        else:
            self.data = data

        # Count clusters
        self.countclusters = countclusters

        # Home many chromosome in population
        self.chromosomecount = chromosomecount

        # How many populations
        self.populationcount = populationcount

        # Count mutation genes
        self.countmutationgens = countmutationgens

        # Crossover rate
        self.crossoverrate = 1.0

        # Count of chromosome for mutation (range [0, 1])
        self.coeffmutationcount = coeffmutationcount

        # Exponential coeff for selection
        self.selectcoeff = selectcoeff

        # Result of clustering : best chromosome
        self.resultclustering = {'bestchromosome': [],
                                  'bestfitnessfunction': 0.0}


    def run(self):
        
        # Initialize population
        chromosomes = self.initpopulation(self.countclusters, len(self.data), self.chromosomecount)

        # Initialize the Best solution
        bestchromosome, bestff, firstfitnessfunctions \
            = self.getbestchromosome(chromosomes, self.data, self.countclusters)

        

        # Next population
        for idx in range(self.populationcount):

            # Select
            chromosomes = self.select(chromosomes, self.data, self.countclusters, self.selectcoeff)
            
            # Crossover
            self.crossover(chromosomes)

            # Mutation
            self.mutation(chromosomes, self.countclusters, self.countmutationgens, self.coeffmutationcount)

            # Update the Best Solution
            newbestchromosome, newbestff, fitnessfunctions \
                = self.getbestchromosome(chromosomes, self.data, self.countclusters)

            # Get best chromosome
            if newbestff < bestff:
                bestff = newbestff
                bestchromosome = newbestchromosome

            

        # Save result
        self.resultclustering['bestchromosome'] = bestchromosome
        self.resultclustering['bestfitnessfunction'] = bestff

        return bestchromosome, bestff





    def getclusters(self):
        return maths.clustersrepresentation(self.resultclustering['bestchromosome'], self.countclusters)


    
    def select(self,chromosomes, data, countclusters, selectcoeff):
       
        # Calc centers
        centres = maths.getcentres(chromosomes, data, countclusters)

        # Calc fitness functions
        fitness = GA.fitnessfunction(centres, data, chromosomes)

        for idx in range(len(fitness)):
            fitness[idx] = math.exp(1 + fitness[idx] * selectcoeff)

        # Calc probability vector
        probabilities = maths.calcprobabilityvector(fitness)

        # Select P chromosomes with probabilities
        newchromosomes = np.zeros(chromosomes.shape, dtype=np.int)
        
        # Selecting
        for idx in range(len(chromosomes)):
            newchromosomes[idx] = chromosomes[maths.getuniform(probabilities)]

        return newchromosomes


    def crossover(self,chromosomes):

        # Get pairs to Crossover
        pairstocrossover = np.array(range(len(chromosomes)))

        # Set random pairs
        np.random.shuffle(pairstocrossover)

        offsetinpair = int(len(pairstocrossover) / 2)

        # For each pair
        for idx in range(offsetinpair):

            # Generate random mask for crossover
            crossovermask = GA.getcrossovermask(len(chromosomes[idx]))

            # Crossover a pair
            GA.crossoverapair(chromosomes[pairstocrossover[idx]],
                                                chromosomes[pairstocrossover[idx + offsetinpair]],
                                                crossovermask)


    
    def mutation(self,chromosomes, countclusters, countgenformutation, coeffmutationcount):
        # Count gens in Chromosome
        countgens = len(chromosomes[0])

        # Get random chromosomes for mutation
        randomidxchromosomes = np.array(range(len(chromosomes)))
        np.random.shuffle(randomidxchromosomes)

        
        for idxchromosome in range(int(len(randomidxchromosomes) * coeffmutationcount)):

            
            for i in range(countgenformutation):

                # Get random gen
                gennum = np.random.randint(countgens)

                # Set random cluster
                chromosomes[randomidxchromosomes[idxchromosome]][gennum] = np.random.randint(countclusters)


    
    def crossoverapair(chromosome1, chromosome2, mask):
        
        for idx in range(len(chromosome1)):

            if mask[idx] == 1:
                # Swap values
                chromosome1[idx], chromosome2[idx] = chromosome2[idx], chromosome1[idx]


    
    def getcrossovermask(masklength):
        
        # Initialize mask
        mask = np.zeros(masklength)

        # Set a half of array to 1
        mask[:int(int(masklength) / 6)] = 1

        # Random shuffle
        np.random.shuffle(mask)

        return mask


    def initpopulation(self,countclusters, countdata, chromosomecount):
        population = np.random.randint(countclusters, size=(chromosomecount, countdata))

        return population


    
    def getbestchromosome(self,chromosomes, data, countclusters):
        # Calc centers
        centres = maths.getcentres(chromosomes, data, countclusters)

        # Calc Fitness functions
        fitnessfunctions = GA.fitnessfunction(centres, data, chromosomes)

        # Index of the best chromosome
        bestchromosomeidx = fitnessfunctions.argmin()

        # Get chromosome with the best fitness function
        return chromosomes[bestchromosomeidx], fitnessfunctions[bestchromosomeidx], fitnessfunctions


    
    def fitnessfunction(centres, data, chromosomes):
        # Get count of chromosomes and clusters
        countchromosome = len(chromosomes)

        # Initialize fitness function values
        fitnessfunction = np.zeros(countchromosome)

        # Calc fitness function for each chromosome
        for idxchromosome in range(countchromosome):

            # Get centers for a selected chromosome
            centresdata = np.zeros(data.shape)

            # Fill data centres
            for idx in range(len(data)):
                centresdata[idx] = centres[idxchromosome][chromosomes[idxchromosome][idx]]

            # Get distance for a chromosome
            fitnessfunction[idxchromosome] += np.sum(abs(data - centresdata))

        return fitnessfunction
