from ._splits import *
from deap import creator, base, tools, algorithms
from ._tree import evaluate_split
import numpy
import warnings
import random

def euclidean(p1x, p1y, p2x, p2y):
    """
    Calculate the Euclidean distance between two points in a two-dimensional space.
    
    Parameters
    ----------
    p1x : float
        The x-coordinate of the first point, p1
    p1y : float
        The y-coordinate of the first point, p1
    p2x : float
        The x-coordinate of the second point, p2
    p2y : float
        The y-coordinate of the second point, p2

    Returns
    -------
    ((p1x - p2x)**2 + (p1y - p2y)**2)**(.5): float
        The Euclidean distance between two points (p1x, p1y) and (p2x, p2y).
    """
    return ((p1x - p2x)**2 + (p1y - p2y)**2)**(.5)

def ga_evaluate_split(individual, ind_to_split_func, X, y, parent_mse, n):
    """
    Evaluates the potential gain in mean squared error (MSE) from splitting a decision
    tree node based on the given candidate split. If the individual represents an ellipse
    split, then apply regularisation 
    
    Parameters
    ----------
    individual : array
        The individual that represents a potential split in a decision
        tree.
    ind_to_split_func : function
        A function that converts the individual to a Split object.
    X : np.array, shape (n_samples, n_features)
        A numpy array containing the input samples. 
    y : np.array, shape (n_samples,)
        The target variable.
    parent_mse : float
        The mean squared error of the parent node before the split.
    n : int
        The total number of samples in the dataset.

    Returns
    -------
    gain : np.float64
        The gain in mean squared error (MSE) resulting from the candidate split. 
    """
    candidate_split = ind_to_split_func(individual)
    gain = evaluate_split(candidate_split, X, y, parent_mse, n)
    if isinstance(candidate_split, EllipseSplit):
        elli_gain = candidate_split.evaluate_ellipse(gain, len(y), n)
        return elli_gain,
    
    return gain,

def random_uniform(low, up):
    """
    Generate a list of random numbers between the values specified in the
    "low" and "up" input lists.
    
    Parameters
    ----------
    low : array
        A list of lower bounds for each dimension of the desired random
        samples.
    up : array
        A list of upper bounds for each dimension of the desired random
        samples.

    Returns
    -------
    [np.random.uniform(l,u) for l,u in zip(low,up)] : list
        A list of random numbers generated using NumPy's `random.uniform` function. 
    """
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
    hofsize=5):
    """
    Find the best split for a decision tree using a genetic algorithm.
    
    Parameters
    ----------
    X : np.array, shape (n_samples, n_features)
        A numpy array containing the input samples. 
    y : np.array, shape (n_samples,)
        The target variable.
    parent_mse : float
        The mean squared error of the parent node before the split.
    n : int
        The total number of samples in the dataset.
    ind_to_split_func : function
        A function that converts the individual to a Split object.
    low : array
        A list of lower bounds for each dimension of the split individual.
    up : array
        A list of upper bounds for each dimension of the split individual.
    eta : float, default=0.5
        Crowding degree of the mutation. A high eta will produce a mutant resembling 
        its parent, while a small eta will produce a solution much more different.
    num_gens : int, default=100
        The number of generations.
    num_pop : int, default=50
        The number of individuals in each generation.
    tournsize : int, default=5
        The number of individuals participating in each tournament.
    alpha : float, default=0.2
        Extent of the interval in which the new values can be drawn for each attribute on 
        both side of the parents’ attributes.
    indpb : float, default=0.9
        The independent probability for each attribute of the individual to be mutated.
    cxpb : float, default=0.9
        The probability of mating two individuals.
    mutpb : float, default=0.5
        The probability of mutating an individual.
    hofsize : int, default=5
        The maximum number of individual to keep in the hall of fame.

    Based on DEAP documentation.

    Returns
    -------
    ind_to_split_func(top) : Split object
        The best individual (split) found by the genetic algorithm, which is 
        obtained by applying the `ind_to_split_func` function to the individual with the
        highest fitness score.
    """
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
        toolbox.register("evaluate", lambda individual : ga_evaluate_split(individual, ind_to_split_func, X, y, parent_mse, n))
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
        if hofsize > 0:
            population = population + hall_of_fame[:]
        top = tools.selBest(population[0], k=1)[0]
    return ind_to_split_func(top)

class OrthogonalSplitGenerator:
    """The OrthogonalSplitGenerator class.

    Parameters
    ----------
    
    eta : float, default=0.5
        Crowding degree of the mutation. A high eta will produce a mutant resembling 
        its parent, while a small eta will produce a solution much more different.
    num_gens : int, default=100
        The number of generations.
    num_pop : int, default=50
        The number of individuals in each generation.
    tournsize : int, default=5
        The number of individuals participating in each tournament.
    alpha : float, default=0.2
        Extent of the interval in which the new values can be drawn for each attribute on 
        both side of the parents’ attributes.
    indpb : float, default=0.9
        The independent probability for each attribute of the individual to be mutated.
    cxpb : float, default=0.9
        The probability of mating two individuals.
    mutpb : float, default=0.5
        The probability of mutating an individual.
    hofsize : int, default=5
        The maximum number of individual to keep in the hall of fame.
    regf : float
        A regularization parameter to punish for large ellipses. (Unused here)

    """

    def __init__(self, eta=0.5,
        num_gens=100, num_pop=50,
        tournsize=5, 
        alpha=0.05,
        indpb=0.90, 
        cxpb=0.9, mutpb=0.50,
        hofsize=5, regf=0):
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
    
    def generate_candidates(self, X, y, parent_mse, features, geo_features, n, bbox, random_state):
        """
        Generate orthogonal candidate splits using a genetic algorithm.
        
        Parameters
        ----------
        X : np.array, shape (n_samples, n_features)
            A numpy array containing the input samples. 
        y : np.array, shape (n_samples,)
            The target variable.
        parent_mse : float
            The mean squared error of the parent node before the split.
        n : int
            The total number of samples in the dataset.
        features : array-like
            A list of feature names or indices.
        geo_features : array-like
            A list of indices indicating a set of geospatial features.
        bbox : float
            Bounding box indiciating the spatial extent of the input data. (Unused here)
        random_state : RandomState instance
            Unused here

        Returns
        -------
        A generator that returns OrthogonalSplit objects.
        """
        for f1 in features:
            def ind_to_split_func(individual):
                return OrthogonalSplit(f1, individual[0])
            yield(ga_find_best_split(X, y, parent_mse, n, ind_to_split_func,
                                     low=[min(X[:,f1])], up=[max(X[:,f1])],
                                     num_gens=self.num_gens, num_pop=self.num_pop,
                                     eta = self.eta, tournsize=self.tournsize, alpha = self.alpha,
                                    indpb= self.indpb,
                                     cxpb = self.cxpb, mutpb = self.mutpb,
                                     hofsize = self.hofsize
                                    ))

class DiagonalSplitGenerator:
    """The DiagonalSplitGenerator class.

    Parameters
    ----------
    
    eta : float, default=0.5
        Crowding degree of the mutation. A high eta will produce a mutant resembling 
        its parent, while a small eta will produce a solution much more different.
    num_gens : int, default=100
        The number of generations.
    num_pop : int, default=50
        The number of individuals in each generation.
    tournsize : int, default=5
        The number of individuals participating in each tournament.
    alpha : float, default=0.2
        Extent of the interval in which the new values can be drawn for each attribute on 
        both side of the parents’ attributes.
    indpb : float, default=0.9
        The independent probability for each attribute of the individual to be mutated.
    cxpb : float, default=0.9
        The probability of mating two individuals.
    mutpb : float, default=0.5
        The probability of mutating an individual.
    hofsize : int, default=5
        The maximum number of individual to keep in the hall of fame.
    regf : float
        A regularization parameter to punish for large ellipses. (Unused here)

    """
    def __init__(self, eta=0.5,
        num_gens=100, num_pop=50,
        tournsize=5, 
        alpha=0.05,
        indpb=0.90, 
        cxpb=0.9, mutpb=0.50,
        hofsize=5, regf=0):
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
    
    def generate_candidates(self, X, y, parent_mse, features, geo_features, n, bbox, random_state):
        """
        Generate diagonal candidate splits using a genetic algorithm.
        
        Parameters
        ----------
        X : np.array, shape (n_samples, n_features)
            A numpy array containing the input samples. 
        y : np.array, shape (n_samples,)
            The target variable.
        parent_mse : float
            The mean squared error of the parent node before the split.
        n : int
            The total number of samples in the dataset.
        features : array-like
            A list of feature names or indices.
        geo_features : array-like
            A list of indices indicating a set of geospatial features.
        bbox : float
            Bounding box indiciating the spatial extent of the input data. (Unused here)
        random_state : RandomState instance
            Unused here    
        
        Returns
        -------
        A generator that returns DiagonalSplit objects.
        """
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
                                     hofsize = self.hofsize
                                    ))
                                

class EllipseSplitGenerator:
    """The EllipseSplitGenerator class.

    Parameters
    ----------
    
    eta : float, default=0.5
        Crowding degree of the mutation. A high eta will produce a mutant resembling 
        its parent, while a small eta will produce a solution much more different.
    num_gens : int, default=100
        The number of generations.
    num_pop : int, default=50
        The number of individuals in each generation.
    tournsize : int, default=5
        The number of individuals participating in each tournament.
    alpha : float, default=0.2
        Extent of the interval in which the new values can be drawn for each attribute on 
        both side of the parents’ attributes.
    indpb : float, default=0.9
        The independent probability for each attribute of the individual to be mutated.
    cxpb : float, default=0.9
        The probability of mating two individuals.
    mutpb : float, default=0.5
        The probability of mutating an individual.
    hofsize : int, default=5
        The maximum number of individual to keep in the hall of fame.
    regf : float
        A regularization parameter to punish for large ellipses.

    Attributes
    ---------
    bbox : float
        Bounding box indiciating the spatial extent of the input data.
    """
    def __init__(self, eta=0.5,
        num_gens=100, num_pop=50,
        tournsize=5, 
        alpha=0.05,
        indpb=0.90, 
        cxpb=0.9, mutpb=0.50,
        hofsize=5,
        regf = 0.):
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
    
    def generate_candidates(self, X, y, parent_mse, features, geo_features, n, bbox, random_state):
        """
        Generate ellipse candidate splits using a genetic algorithm.
        
        Parameters
        ----------
        X : np.array, shape (n_samples, n_features)
            A numpy array containing the input samples. 
        y : np.array, shape (n_samples,)
            The target variable.
        parent_mse : float
            The mean squared error of the parent node before the split.
        n : int
            The total number of samples in the dataset.
        features : array-like
            A list of feature names or indices.
        geo_features : array-like
            A list of indices indicating a set of geospatial features.
        bbox : float
            Bounding box indiciating the spatial extent of the input data.
        random_state : RandomState instance
            Unused here
        
        Returns
        -------
        A generator that returns EllipseSplit objects.
        """
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
                    # take the points with smallest distance as focal points
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
                                     hofsize = self.hofsize
                                    ))
