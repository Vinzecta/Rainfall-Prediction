import random
################################### Mutation ################################################
def swap_mutate(self):
    ''' Mutate the chromosome by swapping two cities
        Time complexity: O(1)
        Space complexity: O(1)
    '''
    idx1 = random.randint(0, len(self.chromosome) - 1)
    idx2 = random.randint(0, len(self.chromosome) - 1)
    self.chromosome[idx1], self.chromosome[idx2] = self.chromosome[idx2], self.chromosome[idx1]

def inverse_mutate(self):
    idx1, idx2 = sorted(random.sample(range(len(self.chromosome)), 2))
    substring = self.chromosome[idx1:idx2 + 1][::-1]
    self.chromosome = self.chromosome[:idx1] + substring + self.chromosome[idx2 + 1:]

def scramble_mutate(self):
    idx1, idx2 = sorted(random.sample(range(len(self.chromosome)), 2))
    substring = self.chromosome[idx1:idx2 + 1]
    random.shuffle(substring)
    self.chromosome = self.chromosome[:idx1] + substring + self.chromosome[idx2 + 1:]
