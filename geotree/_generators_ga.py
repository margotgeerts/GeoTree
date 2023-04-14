from ._splits import *
from deap import creator, base, tools, algorithms
from ._tree import evaluate_split
import numpy
import warnings
import random

def euclidean(p1x, p1y, p2x, p2y):
    return ((p1x - p2x)**2 + (p1y - p2y)**2)**(.5)

def ga_evaluate_split(individual, ind_to_split_func, X, y, parent_mse, n, eval_ratio, random_state):
    if eval_ratio < 1.:
      n_samples = int(len(y)*eval_ratio) if (len(y)*eval_ratio) > 100 else len(y)
      idx_eval = random.sample(range(len(y)), n_samples)
      X_eval = X[idx_eval]
      y_eval = y[idx_eval]
    else:
      X_eval = X
      y_eval = y
    candidate_split = ind_to_split_func(individual)
    gain = evaluate_split(candidate_split, X_eval, y_eval, parent_mse, n)
    if isinstance(candidate_split, EllipseSplit):
        elli_gain = candidate_split.evaluate_ellipse(gain, len(y_eval), n)
        #if not elli_gain:
         #   print(candidate_split, gain, elli_gain)
        return elli_gain,
    
    return gain,

def random_uniform(low, up):
    return [np.random.uniform(l,u) for l,u in zip(low,up)]

def ga_find_best_split(
        X, y, parent_mse, n, 
        ind_to_split_func,
        low, up,
        eta=0.5,
        num_gens=100, num_pop=50,
        tournsize=5, 
        alpha=0.2,
        indpb=0.90, 
        cxpb=0.9, mutpb=0.50,
        hofsize=5,
        eval_ratio=1,
        random_state=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if hasattr(creator, "FitnessMax"): del creator.FitnessMax
        if hasattr(creator, "Individual"): del creator.Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random_uniform, low, up)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda individual : ga_evaluate_split(individual, ind_to_split_func, X, y, parent_mse, n, eval_ratio, random_state))
        toolbox.register("mate", tools.cxBlend, alpha=alpha)
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=eta, low=low, up=up, indpb=indpb)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)
        population = toolbox.population(n=num_pop)
        stats = tools.Statistics()
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        hall_of_fame = tools.HallOfFame(hofsize) if hofsize>0 else []
        population = algorithms.eaSimple(population, toolbox, cxpb, mutpb, num_gens, hall_of_fame, verbose=False)
        """for _ in range(num_gens):
            if hofsize>0:
                hall_of_fame.update(population)
            offspring = map(toolbox.clone, toolbox.select(population, len(population)))
            offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population = tools.selBest(offspring + hall_of_fame[:], len(population))"""
        top = tools.selBest(population[0], k=1)[0]
    return ind_to_split_func(top)

class OrthogonalSplitGenerator:

    def __init__(self, eta=0.5,
        num_gens=100, num_pop=50,
        tournsize=5, 
        alpha=0.05,
        indpb=0.90, 
        cxpb=0.9, mutpb=0.50,
        hofsize=5, regf=0,
        eval_ratio=1.):
        self.num_gens = num_gens
        self.num_pop = num_pop
        self.eta = eta
        self.tournsize = tournsize
        self.alpha = alpha
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.hofsize = hofsize
        self.regf = regf
        self.eval_ratio = eval_ratio
    
    def generate_candidates(self, X, y, parent_mse, features, geo_features, n, bbox, random_state):
        for f1 in features:
            def ind_to_split_func(individual):
                return OrthogonalSplit(f1, individual[0])
            yield(ga_find_best_split(X, y, parent_mse, n, ind_to_split_func,
                                     low=[min(X[:,f1])], up=[max(X[:,f1])],
                                     num_gens=self.num_gens, num_pop=self.num_pop,
                                     eta = self.eta, tournsize=self.tournsize, alpha = self.alpha,
                                    indpb= self.indpb,
                                     cxpb = self.cxpb, mutpb = self.mutpb,
                                     hofsize = self.hofsize,
                                     random_state=random_state,
                                     eval_ratio=self.eval_ratio
                                    ))

class DiagonalSplitGenerator:
    
    def __init__(self, eta=0.5,
        num_gens=100, num_pop=50,
        tournsize=5, 
        alpha=0.05,
        indpb=0.90, 
        cxpb=0.9, mutpb=0.50,
        hofsize=5, regf=0,
        eval_ratio=1.):
        self.num_gens = num_gens
        self.num_pop = num_pop
        self.eta = eta
        self.tournsize = tournsize
        self.alpha = alpha
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.hofsize = hofsize
        self.regf = regf
        self.eval_ratio = eval_ratio
    
    def generate_candidates(self, X, y, parent_mse, features, geo_features, n, bbox, random_state):
        if np.all(X[:,geo_features]==X[0,geo_features]):
            return None
        for i1, f1 in enumerate(geo_features):
            for f2 in geo_features[i1+1:]:
                def ind_to_split_func(individual):
                    # avoid division by zero
                    if individual[0] == individual[2]:
                        return None
                    slope = (individual[1] - individual[3]) / (individual[0] - individual[2])
                    intercept = individual[1] - slope * individual[0]
                    return DiagonalSplit(f1, f2, intercept, slope)
                min_f1 = min(X[:,f1])
                max_f1 = max(X[:,f1])
                min_f2 = min(X[:,f2])
                max_f2 = max(X[:,f2])
                yield(ga_find_best_split(X, y, parent_mse, n, ind_to_split_func,
                                         low=[min_f1, min_f2,min_f1, min_f2], 
                                         up=[max_f1, max_f2,max_f1, max_f2],
                                         num_gens=self.num_gens, num_pop=self.num_pop,
                                         eta = self.eta, tournsize=self.tournsize, alpha = self.alpha,
                                         indpb= self.indpb,
                                         cxpb = self.cxpb, mutpb = self.mutpb,
                                     hofsize = self.hofsize,
                                     random_state=random_state,
                                     eval_ratio=self.eval_ratio
                                    ))
                                

class EllipseSplitGenerator:
    
    def __init__(self, eta=0.5,
        num_gens=100, num_pop=50,
        tournsize=5, 
        alpha=0.05,
        indpb=0.90, 
        cxpb=0.9, mutpb=0.50,
        hofsize=5,
        regf = 0.,
        eval_ratio=1.):
        self.num_gens = num_gens
        self.num_pop = num_pop
        self.eta = eta
        self.tournsize = tournsize
        self.alpha = alpha
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.hofsize = hofsize
        self.regf = regf
        self.bbox = None
        self.eval_ratio = eval_ratio
    
    def generate_candidates(self, X, y, parent_mse, features, geo_features, n, bbox, random_state):
        self.bbox = bbox
        if np.all(X[:,geo_features]==X[0,geo_features]):
            return None
        for i1, f1 in enumerate(geo_features):
            for f2 in geo_features[i1+1:]:
                def ind_to_split_func(individual):
                    # calculate distance for each combination of points in individual
                    p1, p2, p3 = individual[0:2], individual[2:4], individual[4:6]
                    d12 = euclidean(p1[0],p1[1],p2[0],p2[1])
                    d13 = euclidean(p1[0],p1[1],p3[0],p3[1])
                    d23 = euclidean(p2[0],p2[1],p3[0],p3[1])
                    distances = [[d12,p1,p2,p3],[d13,p1,p3,p2],[d23,p2,p3,p1]]
                    # take the points with largest distance as focal points
                    min_d = min(distances, key=lambda x : x[0])[1:]
                    distance = euclidean(min_d[-1][0], min_d[-1][1], (min_d[0][0] + min_d[1][0])/2, (min_d[0][1] + min_d[1][1])/2)
                    return EllipseSplit(f1, f2, min_d[0], min_d[1], distance, self.bbox, self.regf)
                min_f1 = min(X[:,f1])
                max_f1 = max(X[:,f1])
                min_f2 = min(X[:,f2])
                max_f2 = max(X[:,f2])
                yield(ga_find_best_split(X, y, parent_mse, n, ind_to_split_func,
                                         low=[min_f1, min_f2,min_f1, min_f2,min_f1, min_f2], 
                                         up=[max_f1, max_f2,max_f1, max_f2,max_f1, max_f2],
                                         num_gens=self.num_gens, num_pop=self.num_pop,
                                         eta = self.eta, tournsize=self.tournsize, alpha = self.alpha,
                                         indpb= self.indpb,
                                         cxpb = self.cxpb, mutpb = self.mutpb,
                                     hofsize = self.hofsize,
                                     random_state=random_state,
                                     eval_ratio=self.eval_ratio
                                    ))
