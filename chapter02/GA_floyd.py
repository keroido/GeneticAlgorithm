"""
フロイド問題
√1~√Nの数を２つに分けてそれぞれの集合の和の差が最も小さくなるようにする。
"""
import random
from bisect import bisect_left
from operator import attrgetter
from copy import deepcopy


N = 50
GEN_SIZE = 100
POP_SIZE = 100
CX_PROB = 0.9
MUT_PROB = 0.01
ELITE = 1
MIN_FIT = sum([(i+1)**(1/2) for i in range(N+1)])


class Individual:
    def __init__(self):
        self.gene = [(i+1)**(1/2)*random.choice((-1, 1)) for i in range(N+1)]
        self.evaluation = evaluate(self.gene)


class Population:
    def __init__(self):
        self.population = [Individual() for _ in range(POP_SIZE)]


def evaluate(ind):
    return abs(sum(ind))


def select(pop):
    pop.sort(key=attrgetter('evaluation'), reverse=True)
    fitness = sorted([MIN_FIT - p.evaluation for p in pop])
    roulet = [0]
    for fit in fitness:
        roulet.append(roulet[-1] + fit)
    roulet = roulet[1:]
    idx1 = bisect_left(roulet, random.uniform(roulet[0], roulet[-1]))
    idx2 = bisect_left(roulet, random.uniform(roulet[0], roulet[-1]))
    parent1, parent2 = pop[idx1], pop[idx2]
    return parent1, parent2


def crossover(parent1, parent2):
    offspring1, offspring2 = Individual(), Individual()
    offspring1.gene, offspring2.gene = [], []
    for p1, p2 in zip(parent1.gene, parent2.gene):
        if random.random() < 0.5:
            offspring1.gene.append(p1)
            offspring2.gene.append(p2)
        else:
            offspring2.gene.append(p1)
            offspring1.gene.append(p2)
    offspring1.evaluation = evaluate(offspring1.gene)
    offspring2.evaluation = evaluate(offspring2.gene)
    return offspring1, offspring2


def mutate(ind1, ind2):
    mutant1, mutant2 = Individual(), Individual()
    mutant1.gene, mutant2.gene = deepcopy(ind1.gene), deepcopy(ind2.gene)
    for i in range(N+1):
        if random.random() <= MUT_PROB:
            mutant1.gene[i] *= -1
            mutant2.gene[i] *= -1
    mutant1.evaluation = evaluate(mutant1.gene)
    mutant2.evaluation = evaluate(mutant2.gene)
    return mutant1, mutant2


def main():
    # 初期集団生成
    pop = Population()
    pop = pop.population
    fitness = [p.evaluation for p in pop]
    min_eval = min(fitness)
    max_eval = max(fitness)
    mean = sum(fitness) / len(fitness)
    print(f"========= Generation{0} =========")
    print(f"MAX: {max_eval}")
    print(f"MIN: {min_eval}")
    print(f"MEAN: {mean}")

    for i in range(1, GEN_SIZE+1):
        next_pop = []
        while len(next_pop) <= POP_SIZE:
            parent1, parent2 = select(pop)
            offspring1, offspring2 = crossover(parent1, parent2)
            ind1, ind2 = mutate(offspring1, offspring2)
            next_pop.append(ind1)
            next_pop.append(ind2)

        pop = deepcopy(next_pop)
        fitness = [p.evaluation for p in pop]
        min_eval = min(fitness)
        max_eval = max(fitness)
        mean = sum(fitness) / len(fitness)

        print()
        print(f"========= Generation{i} =========")
        print(f"MAX: {max_eval}")
        print(f"MIN: {min_eval}")
        print(f"MEAN: {mean}")

    pop.sort(key=attrgetter('evaluation'))
    best = pop[0].gene
    print()
    print("##======= last Generation =======##")
    print(f"MAX EVALUATION: {max_eval}")
    print(f"Best individual: ")
    print(best)


if __name__ == "__main__":
    main()

