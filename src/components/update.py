from dataclasses import dataclass
from typing import List

import librosa
import numpy
import pandas

from components.config import AppConfig, InternalAppConfig
from lib.audio import audio_from_udp, compute_yin, res_cent_to_dt, compute_fft, compute_if
from lib.utils import DataBuffer


@dataclass
class ComputationState:
    buffer_fft_computation: DataBuffer
    current_filling_buffer_audio: int
    target_buffer_size_if: int
    target_buffer_size_fft: int

    def just_computed_fft(self):
        val_1 = self.buffer_fft_computation.df["val"].iloc[-2]
        val_2 = self.buffer_fft_computation.df["val"].iloc[-1]
        return (not val_1) and val_2


def update_from_data(all_data: List[bytes], buffer_yin: DataBuffer, buffer_rolling_yin: DataBuffer,
                     buffer_audio: DataBuffer, buffer_records: DataBuffer,
                     buffer_fft_result: DataBuffer, state: ComputationState, app_config: AppConfig,
                     internal_app_config: InternalAppConfig,
                     ):
    sampling_rate = app_config.sampling_rate
    pitch_tuning = app_config.pitch_tuning
    resolution_fft_cent = app_config.resolution_fft_cent
    resolution_if_cent = app_config.resolution_if_cent

    do_fft = app_config.do_fft

    audio, t0 = audio_from_udp(all_data)

    df_yin_ = compute_yin(audio, t0, sampling_rate, pitch_tuning)
    buffer_yin.ingest(df_yin_)

    last_note0 = buffer_rolling_yin.df.tail(1)["note0"].values[0]
    f0 = buffer_yin.df.tail(100)["f0"].median()
    pitch0 = librosa.pitch_tuning(f0 * 440 / pitch_tuning)
    note0 = librosa.hz_to_note(f0 * 440 / pitch_tuning)
    buffer_rolling_yin.ingest(pandas.DataFrame([{"f0": f0, "pitch0": pitch0, "note0": note0}]))

    dt_for_resolution = res_cent_to_dt(resolution_fft_cent, f0)
    target_buffer_size_fft = max(int(dt_for_resolution * sampling_rate), 3 * sampling_rate)
    target_buffer_size_fft = min(target_buffer_size_fft, buffer_audio.cache_size)
    if not do_fft:
        target_buffer_size_fft = numpy.nan

    dt_for_if = res_cent_to_dt(resolution_if_cent, f0)
    buffer_size_tenth = max(int(dt_for_if * sampling_rate), 3 * sampling_rate)

    target_buffer_size_if = max(int(buffer_size_tenth / 10), 3 * sampling_rate)
    target_buffer_size_if = min(target_buffer_size_if, buffer_audio.cache_size)

    state.target_buffer_size_fft = target_buffer_size_fft
    state.target_buffer_size_if = target_buffer_size_if

    if note0 != last_note0:
        buffer_audio.reset()
        state.buffer_fft_computation.ingest(pandas.DataFrame([{"val": False}]))
        state.current_filling_buffer_audio = 0
    else:
        df_audio = pandas.DataFrame({"val": audio})
        buffer_audio.ingest(df_audio)
        state.current_filling_buffer_audio += len(df_audio)

    last_fft_computation_state = state.buffer_fft_computation.df["val"].values[-1]

    compute_now = (not last_fft_computation_state) and (
            (do_fft and state.current_filling_buffer_audio > .9 * max(target_buffer_size_if, target_buffer_size_fft)) or
            ((not do_fft) and state.current_filling_buffer_audio > .9 * target_buffer_size_if))

    if compute_now:

        fl = 2 ** (internal_app_config.lower_bound_frequency_cent / 1200) * f0
        fu = 2 ** (internal_app_config.upper_bound_frequency_cent / 1200) * f0

        if do_fft:
            magnitudes_fft, frequencies_fft = compute_fft(
                buffer_audio.df["val"].values[-int(.9 * target_buffer_size_fft):],
                sampling_rate)
        else:
            magnitudes_fft = numpy.array([])
            frequencies_fft = numpy.array([])

        magnitudes_if, frequencies_if = compute_if(buffer_audio.df["val"].values[-int(.9 * target_buffer_size_if):],
                                                   sampling_rate)

        def eval_chart(frequencies, vals, fu, fl):
            m = (frequencies < fu) & (frequencies > fl)
            cents = numpy.log2(frequencies[m] / pitch_tuning + 1e-10) * 1200
            cents = cents - numpy.round(cents / 100) * 100

            if m.sum() == 0:
                return numpy.array([]), numpy.array([]), numpy.nan
            else:
                argmax_cents = cents[vals[m] == vals[m].max()]
                if len(argmax_cents) == 0:
                    argmax_cents = numpy.nan
                else:
                    argmax_cents = argmax_cents[0]
                x = cents
                y = vals[m]
                m_ = (cents >= internal_app_config.lower_bound_frequency_cent) & (
                        cents <= internal_app_config.upper_bound_frequency_cent)

                x_ = x[m_]
                y_ = y[m_]
                return x_, y_, argmax_cents

        x0, y0, argmax0_cents = eval_chart(frequencies_fft, magnitudes_fft, fu, fl)
        x1, y1, argmax1_cents = eval_chart(frequencies_fft, magnitudes_fft, 2 * fu, 2 * fl)

        x0_if, y0_if, argmax0_cents_if = eval_chart(frequencies_if, magnitudes_if, fu, fl)
        x1_if, y1_if, argmax1_cents_if = eval_chart(frequencies_if, magnitudes_if, 2 * fu, 2 * fl)

        df0 = pandas.DataFrame({"type": "f0", "x": x0, "y": y0 / (numpy.sqrt(sum(y0 ** 2)))})
        df1 = pandas.DataFrame({"type": "f1", "x": x1, "y": y1 / (numpy.sqrt(sum(y1 ** 2)))})
        df0_if = pandas.DataFrame({"type": "f0_if", "x": x0_if, "y": y0_if / (numpy.sqrt(sum(y0_if ** 2)))})
        df1_if = pandas.DataFrame({"type": "f1_if", "x": x1_if, "y": y1_if / (numpy.sqrt(sum(y1_if ** 2)))})

        df_fft = pandas.concat([df0, df1, df0_if, df1_if], ignore_index=True)
        buffer_fft_result.reset()
        buffer_fft_result.ingest(df_fft)

        this_record = pandas.DataFrame([{"note0": note0, "f0": numpy.round(f0, 2),
                                         "pitch0": numpy.round(pitch0 * 100, 2),
                                         "fft pitch 0": numpy.round(argmax0_cents, 2),
                                         "fft pitch 1": numpy.round(argmax1_cents, 2),
                                         "if pitch 0": numpy.round(argmax0_cents_if, 2),
                                         "if pitch 1": numpy.round(argmax1_cents_if, 2),
                                         }])

        buffer_records.ingest(this_record)

        state.buffer_fft_computation.ingest(pandas.DataFrame([{"val": True}]))
    else:
        state.buffer_fft_computation.ingest(pandas.DataFrame([{"val": last_fft_computation_state}]))

    return buffer_yin, buffer_rolling_yin, buffer_audio, buffer_records, buffer_fft_result, state
