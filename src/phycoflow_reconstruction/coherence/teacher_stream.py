"""Deterministic full-membership condition stream for endpoint teacher visits."""
import torch


class TeacherStream:
    def __init__(self, population, seed):
        if population < 1:
            raise ValueError('teacher population must be positive')
        self.population = int(population)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.permutation = torch.empty(0, dtype=torch.long)
        self.cursor = 0
        self.visits = torch.zeros(population, dtype=torch.long)

    def next_batch(self, maximum):
        if maximum < 1:
            raise ValueError('batch cap must be positive')
        if self.cursor == len(self.permutation):
            self.permutation = torch.randperm(self.population, generator=self.generator)
            self.cursor = 0
        end = min(self.cursor + maximum, self.population)
        indices = self.permutation[self.cursor:end].clone()
        self.cursor = end
        self.visits[indices] += 1
        return indices

    def state_dict(self):
        return {'population':self.population, 'generator':self.generator.get_state().clone(),
                'permutation':self.permutation.clone(), 'cursor':self.cursor, 'visits':self.visits.clone()}

    def load_state_dict(self, state):
        p = state['permutation']
        visits = state['visits']
        cursor = state['cursor']
        if (state['population'] != self.population or p.dtype != torch.long or visits.dtype != torch.long
                or tuple(visits.shape) != (self.population,) or len(p) not in (0, self.population)
                or not isinstance(cursor, int) or not 0 <= cursor <= len(p) or (visits < 0).any()
                or (len(p) and not torch.equal(p.sort().values, torch.arange(self.population)))):
            raise ValueError('invalid teacher stream checkpoint')
        self.generator.set_state(state['generator'])
        self.permutation = p.clone()
        self.visits = visits.clone()
        self.cursor = cursor
