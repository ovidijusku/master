import numpy as np
from time import perf_counter
from multiprocessing import Pool
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

many_elements = np.random.randint(low=-100, high=100, size=(1_000_000, 2))
few_elements = np.random.randint(low=-100, high=100, size=(15, 2))


def get_distances(x1):
    res = []
    for x2 in many_elements:
        subtracted = np.subtract(x1, x2)
        powered = np.power(subtracted, 2)
        res.append(np.sqrt(np.sum(powered)))
    return np.sum(res), x1


if __name__ == "__main__":
    results = []
    two_powers = [2**x for x in range(7)]
    for num_workers in two_powers:
        start_time = perf_counter()
        p = Pool(num_workers)
        distances = p.map(get_distances, few_elements.tolist())
        distances.sort(key=lambda x: x[0], reverse=True)
        print(distances[0])
        total_time = perf_counter() - start_time
        results.append(total_time)

    plt.plot(two_powers, results)
    plt.show()
    plt.savefig("times.png")
