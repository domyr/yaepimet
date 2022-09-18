import pickle
import unittest

import numpy

from lib.audio import res_cent_to_dt, hz_to_note, read_udp_package, audio_from_udp, compute_yin


class TestAudio(unittest.TestCase):

    def test_res_cent_to_dt(self):
        assert numpy.isclose(res_cent_to_dt(1, 440), 3.9334, atol=1e-4)
        print(res_cent_to_dt(.1, 60))

    def test_hz_to_note(self):
        assert hz_to_note(440) == "A4"
        assert hz_to_note(numpy.nan) == "NULL"

    def test_read_udp_package(self):
        with open("./data/udp_packages.pkl", "rb") as f:
            all_data = pickle.load(f)

        single_package = all_data[0]
        audio, t0 = read_udp_package(single_package)

        assert numpy.all(audio[:5] == numpy.array([ 119.,  846., 1562., 2204., 2678.])), audio[:5]

        audio, t0 = audio_from_udp(all_data)

        df_yin = compute_yin(audio, t0, sr=44100, base_frequency=442, hop_length=2048 // 2,
                             frame_length=2 * 2048)
        assert df_yin["note0"].mode().values[0] == "C5"

        assert abs(len(df_yin) - len(audio) // (2048 // 2)) <= 1, (len(df_yin), len(audio) // (2048 // 2))


if __name__ == '__main__':
    unittest.main()
