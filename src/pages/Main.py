import asyncio
import socket

import altair as alt
import numpy
import pandas
import streamlit as st

from components.config import AppConfig, InternalAppConfig
from components.update import ComputationState, update_from_data
from lib.utils import DataBuffer

internal_app_config = InternalAppConfig.load()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((internal_app_config.udp_ip, internal_app_config.udp_port))


def plot(buffer_yin: DataBuffer, buffer_rolling_yin: DataBuffer, buffer_audio: DataBuffer, buffer_records: DataBuffer,
         buffer_fft_result: DataBuffer,
         state: ComputationState, app_config: AppConfig, internal_app_config: InternalAppConfig, placeholder_main):
    with placeholder_main.container():
        last_record = buffer_rolling_yin.df.iloc[-1]
        note0 = last_record["note0"]
        f0 = last_record["f0"]
        pitch0 = last_record["pitch0"]
        buffer_audio.df["val"].notnull().sum()

        metric_columns = st.columns(5)
        metric_columns[1].metric(app_config.PARAMETERS[0].display_name, app_config.pitch_tuning)
        metric_columns[2].metric(app_config.PARAMETERS[1].display_name, app_config.resolution_fft_cent)
        metric_columns[3].metric(app_config.PARAMETERS[2].display_name, app_config.resolution_if_cent)
        metric_columns[4].metric(app_config.PARAMETERS[3].display_name, app_config.sampling_rate)

        first_row = st.columns(5)

        first_row[0].metric(f"Note",
                            note0,
                            delta=None, delta_color="normal", help=None)

        first_row[1].metric(f"Frequency (Hz)",
                            numpy.round(f0, 2),
                            delta=None, delta_color="normal", help=None)

        first_row[2].metric(f"Pitch (cent)",
                            numpy.round(pitch0 * 100, 2),
                            delta=None, delta_color="normal", help=None)

        first_row[3].metric(f"Buffer Fill Level for FFT (%)",
                            numpy.round(state.current_filling_buffer_audio / state.target_buffer_size_fft * 100, 0),
                            delta=None, delta_color="normal", help=None)

        first_row[4].metric(f"Buffer Fill Level for IFT (%)",
                            numpy.round(state.current_filling_buffer_audio / state.target_buffer_size_if * 100, 0),
                            delta=None, delta_color="normal", help=None)

        second_row = st.columns(2)

        df: pandas.Frame = buffer_records.df
        m = ~df.isnull().all(axis=1)
        if m.sum() > 0:
            second_row[0].dataframe(data=df[m][::-1], width=None, height=None)

        df = buffer_fft_result.df
        m = ~df.isnull().all(axis=1)
        if m.sum() > 0:
            chart = alt.Chart(df[m]).mark_line().encode(
                x=alt.X('x'),
                y=alt.Y('y'),
                color=alt.Color("type")
            )
            second_row[1].altair_chart(chart, use_container_width=True)


async def produce(queue):
    while True:
        try:
            data, addr = sock.recvfrom(65507)
            _ = await queue.put(data)
            _ = await asyncio.sleep(0.001)
        except Exception as e:
            print("Producer Exception", e)


async def consume(queue):
    buffer_yin = DataBuffer(columns=["t", "note0", "f0", "pitch0", "dt"],
                            cache_size=10000,
                            time_col="t",
                            time_range=15 * 60,
                            groupby_cols=["note0"],
                            group_size=500)

    buffer_audio = DataBuffer(columns=["val"], cache_size=1200000)  # 1300000 corresponds to 1 cent resolution for C2

    buffer_rolling_yin = DataBuffer(columns=["note0", "f0", "pitch0"], cache_size=100)
    buffer_rolling_yin.ingest(pandas.DataFrame([{"note0": "UNK", "f0": 0, "pitch0": -100}]))

    buffer_records = DataBuffer(
        columns=["note0", "f0", "pitch0", "fft pitch 0", "fft pitch 1", "if pitch 0", "if pitch 1"],
        cache_size=1000)

    buffer_fft_result = DataBuffer(columns=["x", "y", "type"], cache_size=300000)

    buffer_fft_computation = DataBuffer(columns=["val"], cache_size=5)
    buffer_fft_computation.ingest(pandas.DataFrame([{"val": False}]))

    state = ComputationState(buffer_fft_computation=buffer_fft_computation,
                             current_filling_buffer_audio=-1,
                             target_buffer_size_if=-1,
                             target_buffer_size_fft=-1)
    app_config = AppConfig.load()

    while True:
        try:
            all_data = []
            for _ in range(internal_app_config.batch_size):
                item = await queue.get()
                all_data += [item]
                queue.task_done()

            buffer_yin, buffer_rolling_yin, buffer_audio, buffer_records, buffer_fft_result, state = \
                update_from_data(all_data, buffer_yin, buffer_rolling_yin, buffer_audio, buffer_records,
                                 buffer_fft_result, state, app_config, internal_app_config)

            plot(buffer_yin, buffer_rolling_yin, buffer_audio, buffer_records, buffer_fft_result,
                 state, app_config, internal_app_config, placeholder_main)

        except Exception as e:
            print("Consumer Exception", e)


async def run():
    queue = asyncio.Queue()

    consumer = asyncio.create_task(consume(queue))
    consumers = [consumer]
    _ = await produce(queue)
    _ = await queue.join()

    consumer.cancel()
    _ = await asyncio.gather(*consumers, return_exceptions=True)


placeholder_main = st.empty()
asyncio.run(run(), debug=False)
