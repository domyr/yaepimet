import os.path
import pickle
import unittest
from dataclasses import dataclass

import numpy
import pandas

from lib.utils import Parameter, ConfigHandler, DataBuffer


class TestAudio(unittest.TestCase):

    def test_config(self):
        var_1_parameter = Parameter("var_1", "description 1", [1, 2, 3], 0, int)
        var_2_parameter = Parameter("var_2", "description 1", [1, 2, 3, -1], 1, int)
        var_3_parameter = Parameter("var_3", "description 1", [1, 2, 3, 4], 2, float)

        @dataclass
        class Config1(ConfigHandler):
            PATH = "./data/tmp/config1.pkl"
            PARAMETERS = [var_1_parameter, var_2_parameter]

            var_1: var_1_parameter.dtype
            var_2: var_2_parameter.dtype

        @dataclass
        class Config2(ConfigHandler):
            PATH = "./data/tmp/config2.pkl"
            PARAMETERS = [var_3_parameter]

            var_3: var_3_parameter.dtype

        config_1 = Config1.load()
        config_2 = Config2.load()
        assert os.path.isfile(Config1.PATH)

        with open(Config1.PATH, "rb") as f:
            config_3 = pickle.load(f)
        with open(Config2.PATH, "rb") as f:
            config_4 = pickle.load(f)

        config_5 = Config1(**{"var_1": 3, "var_2": -1, "PATH": Config1.PATH, "PARAMETERS": Config1.PARAMETERS})
        config_5.save()

        config_6 = Config1.load()

        assert config_1.to_dict() == {'var_1': 1, 'var_2': 2}, config_1.to_dict()
        assert config_2.to_dict() == {'var_3': 3}
        assert config_1.to_dict() == config_3
        assert config_2.to_dict() == config_4
        assert config_5.to_dict() == config_6.to_dict()
        assert config_5.to_dict() == {'var_1': 3, 'var_2': -1}



    def test_data_buffer(self):
        databuffer = DataBuffer(columns=["ts", "group", "val"], cache_size=4, time_col="ts", time_range=5,
                                groupby_cols=["group"], group_size=1)
        df_ = pandas.DataFrame({"ts": [1, 2, 3, 4], "group": [1, 2, 1, 2], "val": [3, 5, -1, -2]})
        databuffer.ingest(df_)
        assert numpy.all(databuffer.df.values == numpy.array([[3.0, 1.0, -1.0], [4.0, 2.0, -2.0]]))

        databuffer = DataBuffer(columns=["ts", "group", "val"], cache_size=4, time_col="ts", time_range=5,
                                groupby_cols=["group"], group_size=2)
        df_ = pandas.DataFrame({"ts": [4, 2, 3, 1], "group": [1, 2, 1, 2], "val": [3, 5, -1, -2]})
        databuffer.ingest(df_)

        assert numpy.all(
            databuffer.df.values == numpy.array([[1.0, 2.0, -2.0], [2., 2., 5.], [3.0, 1.0, -1.0], [4., 1., 3.]]))

        df_ = pandas.DataFrame({"ts": [3.5, 6], "group": [1, 2], "val": [99, 99]})
        databuffer.ingest(df_)

        assert numpy.all(
            databuffer.df.values == numpy.array([[2.0, 2.0, 5.0], [3.5, 1., 99.], [4.0, 1.0, 3.0], [6., 2., 99.]]))

        databuffer = DataBuffer(columns=["val"], cache_size=6)
        df_ = pandas.DataFrame({"val": [1, 2, 3, 4]})
        databuffer.ingest(df_)

        df_ = pandas.DataFrame({"val": [5, 6]})
        databuffer.ingest(df_)
        assert numpy.all(
            databuffer.df.values == numpy.array([[1.], [2.], [3.], [4.], [5.], [6.]])), databuffer.df.values

        df_ = pandas.DataFrame({"val": [7, 8]})
        databuffer.ingest(df_)
        assert numpy.all(
            databuffer.df.values == numpy.array([[3.], [4.], [5.], [6.], [7.], [8.]])), databuffer.df.values

    def tearDown(self) -> None:

        if os.path.isfile("./data/tmp/config1.pkl"):
            os.remove("./data/tmp/config1.pkl")
        if os.path.isfile("./data/tmp/config2.pkl"):
            os.remove("./data/tmp/config2.pkl")

if __name__ == '__main__':
    unittest.main()
