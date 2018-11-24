
import loader
import preprocessing
import algorithms as algs

from utils import plot_history
from estimator import compile_fit

# %% Initialization


simulation_num = 1

algorithms = [
    # 'regular',
    'siamese_double',
    'siamese_triplet',
]

datasets = [
    'gisette',
    'homus',
    'letter',
    'mnist',
    'nist',
    'pendigits',
    'satimage',
    'usps',
]

# %% Main Program


if __name__ == '__main__':

    # iterate over each dataset
    for db in datasets:
        dataset = loader.db_load(db)
        dataset = preprocessing.preproc(dataset)

        # use all algorithm for testing
        for alg in algorithms:
            algorithm = algs.get_algorithm(alg)
            transformed_db = algorithm.transform(dataset)

            histories = []
            # train neural network with the num of simulation
            for sim_num in range(simulation_num):
                algorithm = loader.alg_load(algorithm, dataset)
                history = compile_fit(algorithm, transformed_db)
                if not sim_num % 10:
                    algorithm.save_weights(db, sim_num)
                    plot_history(history, alg, db, sim_num)
                histories.append(history.history)
                algorithm.evaluate(transformed_db)
                algorithm.get_and_save_layer(transformed_db, dataset=dataset)
