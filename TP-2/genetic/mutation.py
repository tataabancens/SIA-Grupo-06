from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Optional
from agent import Agent
from numpy import random


class Mutation(ABC):
    """
        Mutation method abstract class
    """

    @abstractmethod
    def mutate(self, agent: Agent, p_m: float) -> None:
        """
            Mutate the individuals from the agent. Returns the index of the mutated genes
        """


class OneGen(Mutation):
    """
        Se altera un solo gen con una probabilidad P_m.
    """

    def mutate(self, agent: Agent, p_m: float) -> None | List[int]:
        """
            Mutate the individuals from the agent
        """
        cromosome_amount = len(agent.cromosome)
        cromosome_to_mutate = random.default_rng().integers(
            0, cromosome_amount)

        mutate_probability = random.default_rng().uniform(0, 1)
        if mutate_probability <= p_m:
            agent.cromosome[cromosome_to_mutate] = random.default_rng().uniform(
                0, 1)
            return list(cromosome_to_mutate)
        return None


class LimitedMultiGen(Mutation):
    """
        Se selecciona una cantidad [1,M] (azarosa) de genes para mutar, con probabilidad P_m
    """

    def mutate(self, agent: Agent, p_m: float) -> None | List[int]:
        """
            Mutate the individuals from the agent. Returns the index of the mutated genes
        """
        cromosome_amount = len(agent.cromosome)

        cromosome_to_mutate_amount = random.default_rng().integers(
            1, cromosome_amount)

        cromosome_to_mutate = random.default_rng().integers(
            0, cromosome_amount, size=cromosome_to_mutate_amount)

        mutated_genes = []

        for _ in range(cromosome_to_mutate_amount):
            mutate_probability = random.default_rng().uniform(0, 1)
            if mutate_probability <= p_m:
                agent.cromosome[cromosome_to_mutate] = random.default_rng().uniform(
                    0, 1)
                mutated_genes.append(cromosome_to_mutate)

        return mutated_genes if len(mutated_genes) > 0 else None


class UniformMultiGen(Mutation):
    """
         Cada gen tiene una probabilidad P_m de ser mutado
    """

    def mutate(self, agent: Agent, p_m: float) -> None | List[int]:
        """
            Mutate the individuals from the agent. Returns the index of the mutated genes
        """
        cromosome_amount = len(agent.cromosome)

        mutated_genes = []

        for i in range(cromosome_amount):
            mutate_probability = random.default_rng().uniform(0, 1)
            if mutate_probability <= p_m:
                agent.cromosome[i] = random.default_rng().uniform(
                    0, 1)
                mutated_genes.append(i)

        return mutated_genes if len(mutated_genes) > 0 else None


class Complete(Mutation):
    """
        Con una probabilidad P_m se mutan todos los genes del individuo, acorde a la función de mutación definida para cada gen
    """

    def mutate(self, agent: Agent, p_m: float) -> None | List[int]:
        """
            Mutate the individuals from the agent. Returns the index of the mutated genes
        """
        cromosome_amount = len(agent.cromosome)

        mutated_genes = []

        mutate_probability = random.default_rng().uniform(0, 1)

        for i in range(cromosome_amount):
            if mutate_probability <= p_m:
                agent.cromosome[i] = random.default_rng().uniform(
                    0, 1)
                mutated_genes.append(i)

        return mutated_genes if len(mutated_genes) > 0 else None


class MutationOptions(Enum):
    """
        Selection options enum
    """
    ONE_GEN = "OneGen"
    LIMITED_MULTI_GEN = "LimitedMultiGen"
    UNIFORM_MULTI_GEN = "UniformMultiGen"
    COMPLETE = "Complete"

    @classmethod
    def from_string(cls, name: str) -> Optional['MutationOptions']:
        """
            Returns the enum value from the string
        """
        try:
            return MutationOptions(name)
        except ValueError:
            return None

    @classmethod
    def get_instance_from_name(cls, name: str) -> Optional[Mutation]:
        """
            Returns enum instance from the string
        """
        try:
            return MutationOptions.get_instance_from_type(MutationOptions(name))
        except ValueError:
            return None

    @classmethod
    def get_instance_from_type(cls, role_type: 'MutationOptions') -> Optional[Mutation]:
        """
            Returns enum instance from the enum
        """
        selected_class = globals()[role_type.value]
        return selected_class()