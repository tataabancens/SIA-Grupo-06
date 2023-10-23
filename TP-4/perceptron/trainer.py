from abc import ABC, abstractmethod
from typing import Iterator


class Trainer(ABC):

    @abstractmethod
    def iterator(self, x_train, y_train, iters) -> Iterator:
        pass


class Batch(Trainer):
    def __init__(self):
        self.idx = 0

    def __str__(self):
        return "Batch"

    def iterator(self, x_train,  iters) -> Iterator:
        class CustomIterator:
            def __init__(self, x_train, iters):
                self.x_train = x_train
                self.iters = iters
                self.idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.idx < self.iters:
                    self.idx += 1
                    return zip(self.x_train)
                else:
                    raise StopIteration

        return CustomIterator(x_train, iters)


class MiniBatch(Trainer):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __str__(self):
        return f"MiniBatch({self.batch_size})"

    def iterator(self, x_train, iters) -> Iterator:
        class CustomIterator:
            def __init__(self, x_train, iters, batch_size: int):
                self.x_train = x_train
                self.batch_size = batch_size
                self.idx = 0
                self.counter = 0
                self.iters = iters

            def __iter__(self):
                return self

            def __next__(self):
                if self.counter >= self.iters:
                    raise StopIteration
                if self.idx >= len(self.x_train):
                    self.idx = 0
                start = self.idx
                end = min(self.idx + self.batch_size, len(self.x_train))
                self.idx = end
                self.counter += 1
                return self.x_train[start:end]

        return CustomIterator(x_train, iters, self.batch_size)


class Online(Trainer):
    def __str__(self):
        return "Online"

    def iterator(self, x_train,iters) -> Iterator:
        return MiniBatch(1).iterator(x_train, iters)
