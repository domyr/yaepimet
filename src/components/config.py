from dataclasses import dataclass

from lib.utils import Parameter, ConfigHandler

pitch_tuning_parameter = Parameter("pitch_tuning", "a (Hz)", [435, 440, 442], 2, int)
resolution_cent_fft_parameter = Parameter("resolution_fft_cent", "Resolution in cent (FFT)", [.1, .25, .5, 1, 2, 4], 3,
                                          float)
resolution_cent_if_parameter = Parameter("resolution_if_cent", "Resolution in cent (IF)", [.1, .25, .5, 1, 2, 4], 0,
                                         float)
sampling_rate_parameter = Parameter("sampling_rate", "Sampling rate", [22050, 44100, 48000], 1, int)
do_fft_parameter = Parameter("do_fft", "Compute full FFT", [True, False], 0, bool)

batch_size_parameter = Parameter("batch_size", "Batch size of UDP packages (1 package corresponds to 0.015s)",
                                 [20], 0, int)
udp_port_parameter = Parameter("udp_port", "Port for UDP", [5005], 0, int)
udp_ip_parameter = Parameter("udp_ip", "IP for UDP", ["0.0.0.0"], 0, str)

lower_bound_frequency_cent_parameter = Parameter("lower_bound_frequency_cent",
                                                 "lower bound in cent for frequency domain plot", [-50, -25], 1, int)
upper_bound_frequency_cent_parameter = Parameter("upper_bound_frequency_cent",
                                                 "upper bound in cent for frequency domain plot", [25, 50], 0, int)


@dataclass
class AppConfig(ConfigHandler):
    PATH = "/app/config.pkl"
    PARAMETERS = [pitch_tuning_parameter, resolution_cent_fft_parameter, resolution_cent_if_parameter,
                  sampling_rate_parameter, do_fft_parameter]

    pitch_tuning: pitch_tuning_parameter.dtype
    resolution_fft_cent: resolution_cent_fft_parameter.dtype
    resolution_if_cent: resolution_cent_if_parameter.dtype
    sampling_rate: sampling_rate_parameter.dtype
    do_fft: do_fft_parameter.dtype


@dataclass
class InternalAppConfig(ConfigHandler):
    PATH = "/app/internalappconfig.pkl"
    PARAMETERS = [batch_size_parameter, udp_ip_parameter, udp_port_parameter, lower_bound_frequency_cent_parameter,
                  upper_bound_frequency_cent_parameter]

    batch_size: batch_size_parameter.dtype
    udp_ip: udp_ip_parameter.dtype
    udp_port: udp_port_parameter.dtype
    lower_bound_frequency_cent: lower_bound_frequency_cent_parameter.dtype
    upper_bound_frequency_cent: upper_bound_frequency_cent_parameter.dtype
