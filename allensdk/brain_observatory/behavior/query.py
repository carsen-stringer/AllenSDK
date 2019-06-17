from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import collections
import time
import pandas as pd

import logging
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)


def isstr(o):
  try:
    basestring
  except NameError:
    basestring = (str, bytes)
  return isinstance(o, basestring)

def generator_wrap(o):
  if not isstr(o):
    try:
      return iter(o)
    except TypeError:
      pass
  return iter([o])


class BaseSpecification:

    def __init__(self):

        self.logger = logging.getLogger('QUERY')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(ch)

    @abstractmethod
    def is_satisfied_by(self, candidate: Any) -> bool:
        raise NotImplementedError()

    def And(self, other: "BaseSpecification") -> "And":
        return self & other

    def __and__(self, other: "BaseSpecification") -> "And":
        return And(self, other)

    def Or(self, other: "BaseSpecification") -> "Or":
        return self | other

    def __or__(self, other: "BaseSpecification") -> "Or":
        return Or(self, other)

    def Not(self) -> "NotS":
        return ~self

    def __invert__(self) -> "NotS":
        return Not(self)

    def query(self, table, **kwargs):

        t0 = time.time()
        data = [x for x in table.to_dict(orient='records') if self.is_satisfied_by(x)]
        self.logger.info(time.time() - t0)
        
        return pd.DataFrame.from_records(data, **kwargs)


class And(BaseSpecification):
    queries: List[BaseSpecification]

    def __init__(self, *queries):
        super().__init__()
        self.queries = queries

    def is_satisfied_by(self, candidate: Any) -> bool:
        for q in self.queries:
            if not q.is_satisfied_by(candidate):
                return False
        return True


class Or(BaseSpecification):
    queries: List[BaseSpecification]

    def __init__(self, *queries):
        super().__init__()
        self.queries = queries

    def is_satisfied_by(self, candidate: Any) -> bool:
        return any([q.is_satisfied_by(candidate) for q in self.queries])


@dataclass(frozen=True)
class Not(BaseSpecification):
    subject: BaseSpecification

    def is_satisfied_by(self, candidate: Any) -> bool:
        return not self.subject.is_satisfied_by(candidate)


@dataclass(frozen=True)
class HasField(BaseSpecification):
    field: str

    def is_satisfied_by(self, candidate: Any) -> bool:
        return self.field in candidate


@dataclass(frozen=True)
class HasTimestamps(HasField):
    field: str = 'timestamps'


@dataclass(frozen=True)
class IsRunningSpeed(HasField):
    field: str = 'running_speed'


@dataclass(frozen=True)
class IsDff(HasField):
    field: str = 'dff'


@dataclass(frozen=True)
class StartTime(BaseSpecification):
    value: float

    def is_satisfied_by(self, candidate: Any) -> bool:
        if 'timestamps' in candidate:
            N = len(candidate['timestamps'])
            start_ii = np.searchsorted(candidate['timestamps'], self.value, 'left')
            candidate['timestamps'] = candidate['timestamps'][start_ii:]
            for key, val in candidate.items():
                if isinstance(val, (list, np.ndarray)) and len(val) == N:
                    candidate[key] = val[start_ii:]
            return True
        else:
            return False


@dataclass(frozen=True)
class StopTime(BaseSpecification):
    value: float

    def is_satisfied_by(self, candidate: Any) -> bool:
        if 'timestamps' in candidate:
            N = len(candidate['timestamps'])
            end_ii = np.searchsorted(candidate['timestamps'], self.value, 'right')
            candidate['timestamps'] = candidate['timestamps'][:end_ii]
            for key, val in candidate.items():
                if isinstance(val, (list, np.ndarray)) and len(val) == N:
                    candidate[key] = val[:end_ii]
            return True
        else:
            return False


@dataclass(frozen=True)
class TimestampInterval(BaseSpecification):
    start: float
    stop: float

    def is_satisfied_by(self, candidate: Any) -> bool:
        return StartTime(self.start).And(StopTime(self.stop)).is_satisfied_by(candidate)


@dataclass(frozen=True)
class StackedIntervals(BaseSpecification):
    start: list
    stop: list

    def is_satisfied_by(self, candidate: Any) -> bool:
        assert len(self.start) == len(self.stop)
        if 'timestamps' in candidate:
            N = len(candidate['timestamps'])
            k = len(self.start)
            starts_ii = np.searchsorted(candidate['timestamps'], self.start, 'left')
            ends_ii = np.searchsorted(candidate['timestamps'], self.stop, 'right')
            M = np.unique(ends_ii - starts_ii).max()

            new_timestamps = np.empty((k, M))
            new_timestamps.fill(np.nan)
            for ki, (start_ii, stop_ii) in enumerate(zip(starts_ii, ends_ii)):
                new_timestamps[ki, :stop_ii - start_ii] = candidate['timestamps'][start_ii:stop_ii]

            candidate['timestamps'] = new_timestamps
            for key, val in candidate.items():
                if isinstance(val, (list, np.ndarray)) and len(val) == N:


                    new_data = np.empty((k, M))
                    new_data.fill(np.nan)
                    for ki, (start_ii, stop_ii) in enumerate(zip(starts_ii, ends_ii)):
                        new_data[ki, :stop_ii - start_ii] = val[start_ii:stop_ii]

                    candidate[key] = new_data
            return True
        else:
            return False


        return StartTime(self.start).And(StopTime(self.stop)).is_satisfied_by(candidate)


@dataclass(frozen=True)
class Equals(BaseSpecification):
    val: Any
    key: str

    def is_satisfied_by(self, candidate: Any) -> bool:
        return self.key in candidate and candidate[self.key] == self.val


@dataclass(frozen=True)
class CellRoiId(Equals):
    val: Any
    key: str = 'cell_roi_id'


@dataclass(frozen=True)
class ValueRange(BaseSpecification):
    key: str
    start: float
    stop: float

    def is_satisfied_by(self, candidate: Any) -> bool:
        return self.key in candidate and self.start < candidate[self.key] and candidate[self.key] < self.stop

@dataclass(frozen=True)
class Same(BaseSpecification):
    key1: str
    key2: str

    def is_satisfied_by(self, candidate: Any) -> bool:
        return self.key1 in candidate and self.key2 in candidate and candidate[self.key2] == candidate[self.key2]

@dataclass(frozen=True)
class Different(BaseSpecification):
    key1: str
    key2: str

    def is_satisfied_by(self, candidate: Any) -> bool:
        return self.key1 in candidate and self.key2 in candidate and candidate[self.key1] != candidate[self.key2]




import pytest

@pytest.fixture
def records():
    return [
        {'A': 1},
        {'B': 2},
        {'A': 1, 'B': 2},
        ]


@pytest.mark.parametrize('query, number_satisfied', [
    (And(Equals(1, 'A'), Equals(1, 'B')), 0),
    (And(Equals(1, 'A'), Equals(1, 'A')), 2),
    (And(Equals(1, 'A'), Equals(2, 'B'), Equals(1, 'A')), 1),
    ])
def test_and(query, number_satisfied, records):
    assert len([x for x in records if query.is_satisfied_by(x)]) == number_satisfied


@pytest.mark.parametrize('query, number_satisfied', [
    (Or(Equals(1, 'A'), Equals(1, 'B')), 2),
    (Equals(1, 'A') | Equals(1, 'B'), 2),
    (Or(Equals(1, 'A'), Equals(2, 'B'), Equals(3, 'C')), 3),
    (Or(Equals(3, 'A'), Equals(1, 'A'), Equals(2, 'B')), 3),
    ])
def test_or(query, number_satisfied, records):
    assert len([x for x in records if query.is_satisfied_by(x)]) == number_satisfied


@pytest.mark.parametrize('query, number_satisfied', [
    (Equals(1, 'A'), 2),
    (~Equals(1, 'A'), 1),
    (Not(Equals(1, 'A')), 1),
    (Equals(1, 'A').Not(), 1),
    ])
def test_not(query, number_satisfied, records):
    S = len([x for x in records if (query).is_satisfied_by(x)])
    Sc = len([x for x in records if (~query).is_satisfied_by(x)])
    assert S == number_satisfied
    assert S + Sc == len(records)

