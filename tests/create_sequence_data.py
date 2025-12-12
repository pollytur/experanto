import shutil
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import yaml

from experanto.interpolators import Interpolator

SEQUENCE_ROOT = Path("tests/sequence_data")


@contextmanager
def create_sequence_data(
    n_signals=10,
    shifts_per_signal=False,
    use_mem_mapped=False,
    t_end=10.0,
    sampling_rate=10.0,
    contain_nans=False,
    start_time=None,
    irregular_timestamps=False,
    large_phase_shifts=False,
):
    """
    Create sequence data for testing interpolators.

    Parameters
    ----------
    n_signals : int
        Number of signals/channels.
    shifts_per_signal : bool
        If True, creates phase shifts per signal.
    use_mem_mapped : bool
        If True, uses memory-mapped file for data.
    t_end : float
        Duration of the sequence (end_time = start_time + t_end).
    sampling_rate : float
        Sampling rate in Hz.
    contain_nans : bool
        If True, randomly inserts NaNs into data.
    start_time : float or None
        Start time for the sequence. If None, a random non-negative start time
        is generated (between 0 and 10).
    irregular_timestamps : bool
        If True, generates irregular timestamps by adding random jitter.
    large_phase_shifts : bool
        If True, generates phase shifts that can be larger than 1/sampling_rate
        (above the target sampling rate). This tests the case where phase shifts
        can cause different signals to return different index ranges.
    """
    try:
        SEQUENCE_ROOT.mkdir(parents=True, exist_ok=True)
        (SEQUENCE_ROOT / "meta").mkdir(parents=True, exist_ok=True)

        # Generate start_time if not provided (random non-negative value)
        if start_time is None:
            start_time = np.random.rand() * 10.0  # Random start between 0 and 10

        # end_time is start_time + t_end
        end_time = start_time + t_end

        meta = {
            "start_time": start_time,
            "end_time": end_time,
            "modality": "sequence",
            "sampling_rate": sampling_rate,
            "phase_shift_per_signal": shifts_per_signal,
            "is_mem_mapped": use_mem_mapped,
            "n_signals": n_signals,
        }

        n_samples = (
            int((meta["end_time"] - meta["start_time"]) * meta["sampling_rate"]) + 1
        )

        # todo - think if it should be an assert for n_samples or not
        if irregular_timestamps and n_samples >= 2:
            # Generate irregular timestamps with random jitter, ensuring monotonicity
            regular_spacing = (meta["end_time"] - meta["start_time"]) / (n_samples - 1)
            jitter_scale = regular_spacing * 0.3  # Up to 30% jitter per interval
            # Generate jitter for each interval (n_samples - 1 intervals)
            timestamps = np.zeros(n_samples)
            timestamps[0] = meta["start_time"]
            for i in range(1, n_samples - 1):
                jitter = np.random.uniform(-jitter_scale, jitter_scale)
                timestamps[i] = timestamps[i - 1] + regular_spacing + jitter
            # Set the last timestamp exactly to end_time to maintain the interval
            timestamps[-1] = meta["end_time"]
            assert (
                np.diff(timestamps) > 0
            ).all(), "time is not monotonically growing anymore"
        else:
            timestamps = np.linspace(
                meta["start_time"],
                meta["end_time"],
                n_samples,
            )
        np.save(SEQUENCE_ROOT / "timestamps.npy", timestamps)
        meta["n_timestamps"] = len(timestamps)

        data = np.random.rand(len(timestamps), n_signals)

        if contain_nans:
            nan_indices = np.random.choice(
                data.size, size=int(0.1 * data.size), replace=False
            )
            data.flat[nan_indices] = np.nan

        if not use_mem_mapped:
            np.save(SEQUENCE_ROOT / "data.npy", data)
        else:
            filename = SEQUENCE_ROOT / "data.mem"

            fp = np.memmap(filename, dtype=data.dtype, mode="w+", shape=data.shape)
            fp[:] = data[:]
            fp.flush()  # Ensure data is written to disk
            del fp
        meta["dtype"] = str(data.dtype)

        shifts = None
        if shifts_per_signal:
            if large_phase_shifts:
                # Generate phase shifts that can be larger than 1/sampling_rate
                # This means shifts can span multiple sampling intervals
                # Range: 0 to 3 * 1/sampling_rate (up to 3 sampling intervals)
                shifts = np.random.rand(n_signals) * 3.0 / meta["sampling_rate"]
            else:
                # Original behavior: shifts smaller than one sampling interval
                shifts = np.random.rand(n_signals) / meta["sampling_rate"] * 0.9
            np.save(SEQUENCE_ROOT / "meta" / "phase_shifts.npy", shifts)

        with open(SEQUENCE_ROOT / "meta.yml", "w") as f:
            yaml.dump(meta, f)

        yield timestamps, data, shifts
    finally:
        shutil.rmtree(SEQUENCE_ROOT)


@contextmanager
def sequence_data_and_interpolator(data_kwargs=None, interp_kwargs=None):
    data_kwargs = data_kwargs or {}
    interp_kwargs = interp_kwargs or {}
    with create_sequence_data(**data_kwargs) as (timestamps, data, shifts):
        with closing(
            Interpolator.create("tests/sequence_data", **interp_kwargs)
        ) as seq_interp:
            yield timestamps, data, shifts, seq_interp
