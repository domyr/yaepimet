from typing import Tuple, List

import librosa
import numpy
import pandas
from scapy.all import RTP


def res_cent_to_dt(res_cent: float, f: float) -> float:
    """
    Returns minimum time for FFT in order to achieve the given resolution in cent relative to the frequency f.
    """
    assert res_cent > -100
    assert res_cent < 100
    assert f > 0

    res_hz = (2 ** (res_cent / 1200) - 1) * f
    dt = 1 / res_hz
    return dt


def hz_to_note(f: float) -> str:
    """
    Wrapper around librosa.hz_to_note to catch exception.
    """
    try:
        note = librosa.hz_to_note(f)
    except Exception as _:
        note = "NULL"
    return note


def read_udp_package(data: bytes) -> Tuple[numpy.array, int]:
    """
    Interprets UDP package from RTP and returns audio signal as well as initial time (epoch unix time).
    Assumes payload type to be sl16 currently.
    """

    rtp = RTP(data)

    #assert rtp.payload_type == 11
    # According to https://en.wikipedia.org/wiki/RTP_payload_formats this refers to 44.1kHu, single channel, 16 bit audio.

    ts = rtp.time
    payload = rtp.payload.load

    #audio = [int(payload[i:i + 2].hex(), 16) for i in range(0, len(payload), 2)]
    audio = [int.from_bytes(payload[i:i + 2], byteorder='little', signed=True) for i in range(0, len(payload), 2)]
    audio = numpy.array(audio, numpy.int16)
    audio = numpy.array(audio, numpy.float64)
    # other fields: rtp.numsync, rtp.sequence, rtp.timestamp, rtp.sourcesync, rtp.sync
    return audio, ts


def audio_from_udp(all_data: List[bytes]) -> Tuple[numpy.array, int]:
    """Processes a batch of udp packages and returns concatenated audio signal as well as initial time."""
    list_audio, list_ts = zip(*[read_udp_package(data) for data in all_data])
    audio = numpy.hstack(list_audio)
    t0 = list_ts[0]
    return audio, t0


def compute_yin(audio: numpy.array, t0: int, sr: int = 44100, base_frequency: int = 442, hop_length=2048 // 2,
                frame_length=2 * 2048):
    """Uses librosa.yin to compute sequence of pitches and puts result into dataframe."""

    f0s = librosa.yin(audio[:], frame_length=frame_length, hop_length=hop_length,
                      sr=sr, fmin=librosa.note_to_hz('C2'),
                      fmax=librosa.note_to_hz('C7'))

    times = librosa.times_like(f0s, sr=sr, hop_length=hop_length)
    times = times + t0

    result = [{"dt": max(times) - min(times), "f0": f0 if not numpy.isnan(f0) else -1,
               "note0": hz_to_note(f0),
               "pitch0": librosa.pitch_tuning(f0 * 440 / base_frequency),
               "t": t} for f0, t in zip(f0s, times)]
    df_result = pandas.DataFrame(result)
    return df_result


def compute_fft(audio: numpy.array, sampling_rate: int):
    """Convience wrapper for librosa.stft with suitable parameters. Returns magnitudes and frequencies."""

    audio = audio[~numpy.isnan(audio)]
    n_fft = 2 ** (int(numpy.log2(len(audio)))) * 2 ** 3

    hop_length = n_fft
    win_length = n_fft

    stft_raw = librosa.stft(audio, hop_length=hop_length, n_fft=n_fft, win_length=win_length, center=True)
    stft = numpy.abs(stft_raw)

    frequencies = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    magnitudes = numpy.mean(stft, axis=1)

    return magnitudes, frequencies


def compute_if(audio: numpy.array, sampling_rate, n_steps=10, hop_length=2048 // 32):
    """Convience wrapper for librosa.reassigned_spectrogram with suitable parameters.
    Returns magnitudes and frequencies."""

    audio = audio[~numpy.isnan(audio)]
    n_fft = len(audio) // 1 - n_steps * hop_length

    frequencies, times, magnitudes = librosa.reassigned_spectrogram(audio, center=False, hop_length=hop_length,
                                                                    n_fft=n_fft, sr=sampling_rate)

    # magnitudes = numpy.mean(mags, 1)
    # frequencies = numpy.mean(frequs, 1)
    magnitudes = magnitudes.flatten()
    frequencies = frequencies.flatten()

    return magnitudes, frequencies
