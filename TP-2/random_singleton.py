import random as rd


class RandomSingleton(object):
    _instance = None
    random = rd

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RandomSingleton, cls).__new__(cls)
            try:
                cls.random.seed(kwargs["seed"])
            except KeyError:
                cls.random.seed(0)

        return cls._instance


if __name__ == "__main__":
    random_instance1 = RandomSingleton(seed=5)

    print(random_instance1.random.uniform(0, 1))
    print(random_instance1.random.uniform(0, 1))

    random_instance2 = RandomSingleton()
    print(random_instance2.random.uniform(0, 1))

