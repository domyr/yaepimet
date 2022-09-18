import os
import pickle
from dataclasses import dataclass
from typing import Union, List, Type

import numpy
import pandas


@dataclass
class Parameter:
    """
    Class to keep all data defining a parameter: name, display_name, list_of_choices, default_index, dtype
    """
    name: str
    display_name: str
    list_of_choices: List[Union[str, float, int]]
    default_index: int
    dtype: Type

    def default_value(self):
        return self.list_of_choices[self.default_index]

    def __post_init__(self):
        assert self.default_index < len(self.list_of_choices)


@dataclass
class ConfigHandler:
    """
    Class that contains config in the form of a parameter list. That is for each element in PARAMETERS it is assumed
    that a respective field is defined in the dataclass. Moreover PATH defines binding to file on disk.
    Load/save method load a respecitve object from PATH.
    """

    PATH: str
    PARAMETERS: List[Parameter]

    @classmethod
    def load(cls):
        if not os.path.isfile(cls.PATH):
            obj = cls(**{**{k.name: k.default_value() for k in cls.PARAMETERS}, "PATH": cls.PATH,
                         "PARAMETERS": cls.PARAMETERS})
            obj.save()
            return obj
        else:
            with open(cls.PATH, "rb") as f:
                config = pickle.load(f)
            return cls(**{**config, "PATH": cls.PATH, "PARAMETERS": cls.PARAMETERS})

    def to_dict(self):
        return {k.name: self.__getattribute__(k.name) for k in self.PARAMETERS}

    def save(self):
        with open(self.PATH, "wb") as f:
            pickle.dump(self.to_dict(), f)


class DataBuffer:
    """
    Convenience class to handle all kinds of buffers/fifos. Defined by list of columns, cache_size.
    Default implementation is a fifo.
    In addition time_col and time_range can be provided. In this case data is merged using the time_col and filtered
    by the provided time_range.
    Morever a list of groupby_cols can be provided to restrict the size of each group to be group_size.
    """
    df: pandas.DataFrame

    def __init__(self, columns: List[str], cache_size: int, time_col: Union[str, None] = None,
                 time_range: Union[int, float, None] = None, groupby_cols: Union[List[str], None] = None,
                 group_size: Union[int, None] = None):

        if len(columns) != len(set(columns)):
            raise ValueError(f"Expected numeric_columns {columns} to be unique.")

        if time_col is not None:
            if not time_col in columns:
                raise ValueError(
                    f"Expected time_col {time_col} to be in numeric_columns {columns}.")

            if not isinstance(time_range, int):
                raise ValueError(f"Expected time_range {time_range} to be of type int.")

        if groupby_cols is not None:
            if not set(groupby_cols) - set(columns) == set():
                raise ValueError(
                    f"Expected groupby_cols {groupby_cols} to contained in provided {columns}")

            if not isinstance(group_size, int):
                raise ValueError(f"Expected group_size {group_size} to be of type int.")

        first_column_block = [time_col] if time_col is not None else []
        second_column_block = [x for x in groupby_cols if
                               not x in first_column_block] if groupby_cols is not None else []

        self.columns = first_column_block + second_column_block + [x for x in columns if x not in first_column_block
                                                                   + second_column_block]

        self.cache_size = cache_size

        self.df = None
        self.reset()

        self.time_col = time_col
        self.time_range = time_range
        self.groupby_cols = groupby_cols
        self.group_size = group_size

    def reset(self):
        if self.df is None:
            values = numpy.ones(shape=(self.cache_size, len(self.columns))) * numpy.nan
            self.df = pandas.DataFrame(data=values, columns=self.columns)
        else:
            self.df[:] = numpy.nan

    def refresh(self):
        """Implements the logic to refresh if time_col/time_range or groupby_cols/group_size are provided."""
        df: pandas.DataFrame = self.df

        if self.time_col is not None and self.time_range is not None:
            df = df.sort_values(self.time_col, ascending=True)
            df = df[df[self.time_col] > self.df[self.time_col].max() - self.time_range]

        if self.groupby_cols is not None and self.group_size is not None:
            df = df.groupby(self.groupby_cols, as_index=False).tail(self.group_size)

        df = df[-self.cache_size:]
        df.reset_index(drop=True, inplace=True)
        self.df = df

    def _validate(self, dg: pandas.DataFrame):
        """Checks if columns of dg and self.df coincide."""
        assert len(dg.columns) == len(self.df.columns)
        assert set(dg.columns) == set(self.df.columns)

    def ingest(self, dg: pandas.DataFrame):
        """Updates self.df using dg."""
        self._validate(dg)

        if self.time_col is not None and dg[self.time_col].min() <= self.df[self.time_col].max():
            self.df = pandas.concat([self.df[self.columns], dg[self.columns]], ignore_index=True)
            self.refresh()
        else:
            self.df = self.df.shift(-len(dg))
            self.df.iloc[-len(dg):] = dg[self.columns].values
            self.refresh()
