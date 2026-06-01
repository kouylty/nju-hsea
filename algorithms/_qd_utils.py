import numpy as np 
from typing import List, Optional

class Archive:
    def __init__(
        self,
        fitnesses: List[float],
        descriptors: List[float],
        individuals: List[np.ndarray],
        labels: Optional[List[str]] = None,
        archive_size: Optional[int] = None
    ):
        self.fitnesses = fitnesses
        self.individuals = individuals
        self.descriptors = descriptors
        self.labels = labels
        self.archive_size = archive_size

    def __len__(self):
        return len(self.individuals)
    
    def __iter__(self):
        return iter(self.individuals)
    
    def __getitem__(self, key):
        return self.individuals[key]
    
    def update(
        self, 
        X: np.ndarray, 
        Y: float, 
        index: Optional[int] = None
    ):
        if index is None or index < 0:
            self.add_to_archive(X, Y)
        else:
            self.fitnesses[index] = Y 
            self.individuals[index] = X

    def add_to_archive(
        self,
        X: np.ndarray,
        Y: float
    ):
        self.individuals.append(X)
        self.fitnesses.append(Y)
        self.descriptors.append(0.)

        if self.archive_size is not None and len(self.individuals) > self.archive_size:
            min_idx = np.argmin(self.fitnesses)
            del self.individuals[min_idx]
            del self.fitnesses[min_idx]
            del self.descriptors[min_idx]

    

if __name__ == "__main__":
    size, dim = 1, 5
    fit = np.ones(dim)
    descriptors = np.ones(dim) * 0.5 
    individuals = list(np.random.rand(dim))
    print(individuals)
    arc = Archive(
        fitnesses=fit,
        descriptors=descriptors,
        individuals=individuals
    )
    for i in arc:
        print(i)

    
