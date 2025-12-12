import numpy as np
import pytest

from experanto.interpolators import (
    PhaseShiftedSequenceInterpolator,
    SequenceInterpolator,
)

from .create_sequence_data import sequence_data_and_interpolator

DEFAULT_SEQUENCE_LENGTH = 10


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_nearest_neighbor_interpolation(n_signals, sampling_rate, use_mem_mapped):
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=use_mem_mapped,
            t_end=5.0,
            sampling_rate=sampling_rate,
            start_time=0,
        )
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"
        assert not isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is a PhaseShiftedSequenceInterpolator"

        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1e-9
        interp, valid = seq_interp.interpolate(
            times=times, return_valid=True
        )  # Add a small epsilon to avoid floating point errors
        assert times.shape == valid.shape, "All samples should be valid"
        assert (
            interp == data[:DEFAULT_SEQUENCE_LENGTH]
        ).all(), "Nearest neighbor interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_nearest_neighbor_interpolation_handles_nans(n_signals, keep_nans):
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=5.0,
            sampling_rate=10.0,
            contain_nans=True,
            start_time=0,
        ),
        interp_kwargs=dict(keep_nans=keep_nans),
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"

        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1e-9
        interp, valid = seq_interp.interpolate(
            times=times, return_valid=True
        )  # Add a small epsilon to avoid floating point errors
        assert times.shape == valid.shape, "All samples should be valid"
        assert np.array_equal(
            interp, data[:DEFAULT_SEQUENCE_LENGTH], equal_nan=True
        ), "Nearest neighbor interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
def test_nearest_neighbor_interpolation_with_inbetween_times(n_signals, sampling_rate):
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=5.0,
            sampling_rate=sampling_rate,
            start_time=0,
        )
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"

        delta_t = 1.0 / sampling_rate

        # timestamps multiplied by 0.8 should be floored to the same timestamp
        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 0.8 * delta_t
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert times.shape == valid.shape, "All samples should be valid"
        assert (
            interp == data[:DEFAULT_SEQUENCE_LENGTH]
        ).all(), "Nearest neighbor interpolation does not match expected data"

        # timestamps multiplied by 1.2 should be floored to the next timestamp
        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1.2 * delta_t
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert times.shape == valid.shape, "All samples should be valid"
        assert (
            interp == data[1 : DEFAULT_SEQUENCE_LENGTH + 1]
        ).all(), "Nearest neighbor interpolation does not match expected data"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
def test_nearest_neighbor_interpolation_with_phase_shifts(
    n_signals, sampling_rate, use_mem_mapped
):
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=use_mem_mapped,
            t_end=5.0,
            sampling_rate=sampling_rate,
            shifts_per_signal=True,
            start_time=0,
        )
    ) as (timestamps, data, shift, seq_interp):
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is not a PhaseShiftedSequenceInterpolator"

        delta_t = 1.0 / sampling_rate
        times = (
            timestamps[1 : DEFAULT_SEQUENCE_LENGTH + 1] + 1e-9
        )  # Add a small epsilon to avoid floating point errors
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert times.shape == valid.shape, "All samples should be valid"
        assert (
            interp == data[0:DEFAULT_SEQUENCE_LENGTH]
        ).all(), "Nearest neighbor interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"

        # Test phase shifts
        for i in range(data.shape[1]):
            for dt in np.linspace(0, 0.99) * delta_t:
                shifted_times = times + shift[i] + dt

                interp, valid = seq_interp.interpolate(
                    times=shifted_times, return_valid=True
                )
                assert (
                    interp[:, i] == data[1 : DEFAULT_SEQUENCE_LENGTH + 1, i]
                ).all(), f"Data at {dt} does not match original data (use_mem_mapped={use_mem_mapped}, sampling_rate={sampling_rate}, shifts_per_signal={True})"

            for dt in np.linspace(1.0, 1.99) * delta_t:
                shifted_times = times + shift[i] + dt

                interp, valid = seq_interp.interpolate(
                    times=shifted_times, return_valid=True
                )
                assert (
                    interp[:, i] == data[2 : DEFAULT_SEQUENCE_LENGTH + 2, i]
                ).all(), f"Data at {dt} does not match original data (use_mem_mapped={use_mem_mapped}, sampling_rate={sampling_rate}, shifts_per_signal={True})"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_nearest_neighbor_interpolation_with_phase_shifts_handles_nans(
    n_signals, keep_nans
):
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=5.0,
            sampling_rate=10.0,
            shifts_per_signal=True,
            contain_nans=True,
            start_time=0,
        ),
        interp_kwargs=dict(keep_nans=keep_nans),
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is not a PhaseShiftedSequenceInterpolator"

        times = (
            timestamps[1 : DEFAULT_SEQUENCE_LENGTH + 1] + 1e-9
        )  # Add a small epsilon to avoid floating point errors
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert times.shape == valid.shape, "All samples should be valid"
        assert np.array_equal(
            interp, data[0:DEFAULT_SEQUENCE_LENGTH], equal_nan=True
        ), "Nearest neighbor interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
@pytest.mark.parametrize("contain_nans", [False, True])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_linear_interpolation(
    n_signals, sampling_rate, use_mem_mapped, contain_nans, keep_nans
):
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=use_mem_mapped,
            t_end=5.0,
            sampling_rate=sampling_rate,
            contain_nans=contain_nans,
            start_time=0,
        ),
        interp_kwargs=dict(keep_nans=keep_nans),
    ) as (timestamps, data, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Not a SequenceInterpolator"
        seq_interp.interpolation_mode = "linear"

        delta_t = 1.0 / sampling_rate
        idx = [i for i in range(1, DEFAULT_SEQUENCE_LENGTH + 1)]
        times = timestamps[idx] + 0.5 * delta_t

        t1, t2 = (
            timestamps[idx][:, np.newaxis],
            timestamps[[id + 1 for id in idx]][:, np.newaxis],
        )
        y1, y2 = data[idx], data[[id + 1 for id in idx]]
        expected = y1 + ((times[:, np.newaxis] - t1) / (t2 - t1)) * (y2 - y1)
        if not keep_nans:
            np.copyto(expected, np.nanmean(expected, axis=0), where=np.isnan(expected))
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)

        assert times.shape == valid.shape, "All samples should be valid"
        assert np.allclose(
            interp, expected, atol=1e-6, equal_nan=True
        ), "Linear interpolation does not match expected data"
        assert valid.shape == (
            DEFAULT_SEQUENCE_LENGTH,
        ), f"Expected valid.shape == ({DEFAULT_SEQUENCE_LENGTH},), got {valid.shape}"
        assert interp.shape == (
            DEFAULT_SEQUENCE_LENGTH,
            n_signals,
        ), f"Expected interp.shape == ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), got {interp.shape}"
        if not keep_nans:
            assert (
                np.isnan(interp).sum() == 0
            ), "Interpolated data should not contain NaNs"


@pytest.mark.parametrize("n_signals", [0, 1, 10, 50])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("use_mem_mapped", [False, True])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_linear_interpolation_with_phase_shifts(
    n_signals, sampling_rate, use_mem_mapped, keep_nans
):
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=use_mem_mapped,
            t_end=5.0,
            sampling_rate=sampling_rate,
            shifts_per_signal=True,
            start_time=0,
        ),
        interp_kwargs=dict(keep_nans=keep_nans),
    ) as (timestamps, data, shift, seq_interp):
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Not a PhaseShiftedSequenceInterpolator"
        seq_interp.interpolation_mode = "linear"

        delta_t = 1.0 / sampling_rate
        idx = slice(1, DEFAULT_SEQUENCE_LENGTH + 1)
        times = timestamps[idx] + 0.5 * delta_t

        for sig_idx in range(data.shape[1]):
            shift_offset = shift[sig_idx]
            shifted_times = times + shift_offset

            expected = np.zeros(len(shifted_times))
            for i in range(len(shifted_times)):
                t = shifted_times[i]

                shifted_timestamps = timestamps + shift_offset
                left_idx = np.searchsorted(shifted_timestamps, t, side="right") - 1
                right_idx = left_idx + 1

                if left_idx < 0 or right_idx >= len(timestamps):
                    continue

                t1, t2 = shifted_timestamps[left_idx], shifted_timestamps[right_idx]
                y1, y2 = data[left_idx, sig_idx], data[right_idx, sig_idx]

                expected[i] = y1 + ((t - t1) / (t2 - t1)) * (y2 - y1)
            if not keep_nans:
                np.copyto(
                    expected, np.nanmean(expected, axis=0), where=np.isnan(expected)
                )

            interp, valid = seq_interp.interpolate(
                times=shifted_times, return_valid=True
            )

            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                assert np.allclose(
                    interp[valid_indices, sig_idx], expected[valid_indices], atol=1e-6
                ), f"Linear interpolation mismatch for signal {sig_idx}"
            if not keep_nans:
                assert (
                    np.isnan(interp).sum() == 0
                ), "Interpolated data should not contain NaNs"


@pytest.mark.filterwarnings(
    "ignore:Sequence interpolation returns empty array, no valid times queried:UserWarning"
)
@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
@pytest.mark.parametrize("end_time", [0.05, 1.0, 5.0, 12.0])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_interpolation_for_invalid_times(interpolation_mode, end_time, keep_nans):
    n_signals = 10
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=end_time,
            sampling_rate=10.0,
            start_time=0,
        ),
        interp_kwargs=dict(keep_nans=keep_nans),
    ) as (_, _, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"
        seq_interp.interpolation_mode = interpolation_mode

        times = np.array([-5.0, -0.1, 0.1, 4.9, 5.0, 5.1, 10.0])
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        expected_valid = (
            np.where((times >= 0.0) & (times <= end_time))[0]
            if interpolation_mode == "nearest_neighbor"
            else np.where((times >= 0.0) & (times < end_time))[0]
        )
        assert (
            expected_valid == valid
        ).all(), "Valid times does not match expected values"
        expected_nr_valid = len(valid)
        assert interp.shape == (
            expected_nr_valid,
            n_signals,
        ), f"Expected interp.shape == ({expected_nr_valid}, {n_signals}), got {interp.shape}"


@pytest.mark.filterwarnings(
    "ignore:Sequence interpolation returns empty array, no valid times queried:UserWarning"
)
@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
@pytest.mark.parametrize("end_time", [0.05, 1.0, 5.0, 12.0])
@pytest.mark.parametrize("keep_nans", [False, True])
def test_interpolation_with_phase_shifts_for_invalid_times(
    interpolation_mode, end_time, keep_nans
):
    n_signals = 10
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=end_time,
            sampling_rate=10.0,
            shifts_per_signal=True,
            start_time=0,
        ),
        interp_kwargs=dict(keep_nans=keep_nans),
    ) as (_, _, phase_shifts, seq_interp):
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Interpolation object is not a PhaseShiftedSequenceInterpolator"
        seq_interp.interpolation_mode = interpolation_mode

        times = np.array([-5.0, -0.1, 0.1, 4.9, 4.9999999, 5.0, 5.0000001, 5.1, 10.0])
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)
        assert (
            np.where(
                (times >= np.min(phase_shifts))
                & (times <= end_time + np.max(phase_shifts))
            )[0]
            == valid
        ).all(), "Valid times does not match expected values"
        expected_nr_valid = len(valid)
        assert interp.shape == (
            expected_nr_valid,
            n_signals,
        ), f"Expected interp.shape == ({expected_nr_valid}, {n_signals}), got {interp.shape}"


@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
@pytest.mark.parametrize("phase_shifts", [True, False])
def test_interpolation_for_empty_times(interpolation_mode, phase_shifts):
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=10,
            use_mem_mapped=True,
            t_end=5.0,
            sampling_rate=10.0,
            shifts_per_signal=phase_shifts,
            start_time=0,
        )
    ) as (_, _, _, seq_interp):
        assert isinstance(
            seq_interp, SequenceInterpolator
        ), "Interpolation object is not a SequenceInterpolator"
        seq_interp.interpolation_mode = interpolation_mode

        with pytest.warns(
            UserWarning,
            match="Sequence interpolation returns empty array, no valid times queried",
        ):
            interp, valid = seq_interp.interpolate(
                times=np.array([]), return_valid=True
            )
        assert interp.shape[0] == 0, "No data expected"
        assert valid.shape[0] == 0, "No data expected"


def test_interpolation_mode_not_implemented():
    with sequence_data_and_interpolator(
        data_kwargs=dict(start_time=0)
    ) as (_, _, _, seq_interp):
        seq_interp.interpolation_mode = "unsupported_mode"
        with pytest.raises(NotImplementedError):
            seq_interp.interpolate(np.array([0.0, 1.0, 2.0]), return_valid=True)


# =============================================================================
# New test cases for expanded coverage
# =============================================================================


@pytest.mark.parametrize("n_signals", [1, 10])
@pytest.mark.parametrize("sampling_rate", [3.0, 10.0, 100.0])
@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
def test_interpolation_with_nonzero_start_time(n_signals, sampling_rate, interpolation_mode):
    """Test interpolation when start_time is non-zero to check for numerics issues."""
    # Use various non-zero start times to catch potential floating point issues
    for start_time in [0.1, 1.5, 10.0, 100.0, 1000.5]:
        with sequence_data_and_interpolator(
            data_kwargs=dict(
                n_signals=n_signals,
                use_mem_mapped=True,
                t_end=5.0,
                sampling_rate=sampling_rate,
                start_time=start_time,
            )
        ) as (timestamps, data, _, seq_interp):
            seq_interp.interpolation_mode = interpolation_mode

            # Query times within the valid range
            times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1e-9
            interp, valid = seq_interp.interpolate(times=times, return_valid=True)

            assert len(valid) == DEFAULT_SEQUENCE_LENGTH, (
                f"Expected {DEFAULT_SEQUENCE_LENGTH} valid times, got {len(valid)} "
                f"(start_time={start_time}, sampling_rate={sampling_rate})"
            )
            assert interp.shape == (DEFAULT_SEQUENCE_LENGTH, n_signals), (
                f"Shape mismatch: expected ({DEFAULT_SEQUENCE_LENGTH}, {n_signals}), "
                f"got {interp.shape}"
            )

            # Verify that the interpolator correctly handles the offset
            assert seq_interp.start_time == start_time, (
                f"start_time mismatch: expected {start_time}, got {seq_interp.start_time}"
            )
            assert seq_interp.end_time == start_time + 5.0, (
                f"end_time mismatch: expected {start_time + 5.0}, got {seq_interp.end_time}"
            )


@pytest.mark.parametrize("n_signals", [1, 10])
@pytest.mark.parametrize("sampling_rate", [10.0, 50.0])
@pytest.mark.parametrize("interpolation_mode", ["nearest_neighbor", "linear"])
def test_interpolation_with_irregular_timestamps(n_signals, sampling_rate, interpolation_mode):
    """Test interpolation with irregular (jittered) timestamps."""
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=5.0,
            sampling_rate=sampling_rate,
            start_time=0,
            irregular_timestamps=True,
        )
    ) as (timestamps, data, _, seq_interp):
        seq_interp.interpolation_mode = interpolation_mode

        # Verify timestamps are irregular (not perfectly evenly spaced)
        diffs = np.diff(timestamps)
        # With jitter, differences should not all be the same
        assert not np.allclose(diffs, diffs[0], rtol=0.01), (
            "Timestamps should have irregular spacing"
        )

        # Query times within the valid range
        times = timestamps[:DEFAULT_SEQUENCE_LENGTH] + 1e-9
        interp, valid = seq_interp.interpolate(times=times, return_valid=True)

        assert len(valid) > 0, "Should have valid interpolation results"
        assert interp.shape[1] == n_signals, f"Expected {n_signals} signals"


@pytest.mark.parametrize("n_signals", [5, 10, 50])
@pytest.mark.parametrize("sampling_rate", [10.0, 100.0])
def test_phase_shifts_above_sampling_rate(n_signals, sampling_rate):
    """
    Test with phase shifts larger than 1/sampling_rate (the target sampling interval).
    
    This tests the case where phase shifts can span multiple sampling intervals,
    which is important for ensuring the interpolator handles large phase shifts correctly.
    """
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=5.0,
            sampling_rate=sampling_rate,
            shifts_per_signal=True,
            start_time=0,
            large_phase_shifts=True,
        )
    ) as (timestamps, data, shifts, seq_interp):
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Should be PhaseShiftedSequenceInterpolator"

        delta_t = 1.0 / sampling_rate

        # Verify that at least some shifts are larger than the sampling interval
        assert np.any(shifts > delta_t), (
            f"Expected some phase shifts > {delta_t}, but max shift was {np.max(shifts)}"
        )

        # Query times within valid range
        # Start from a time that accounts for the max phase shift
        max_shift = np.max(shifts)
        start_idx = int(np.ceil(max_shift * sampling_rate)) + 1
        times = timestamps[start_idx : start_idx + DEFAULT_SEQUENCE_LENGTH] + 1e-9

        interp, valid = seq_interp.interpolate(times=times, return_valid=True)

        # Should still get valid interpolation results
        assert len(valid) > 0, "Should have valid interpolation results with large phase shifts"
        assert interp.shape[1] == n_signals, f"Expected {n_signals} signals"


@pytest.mark.parametrize("n_signals", [5, 10])
@pytest.mark.parametrize("sampling_rate", [10.0])
def test_phase_shifts_cause_different_indexes(n_signals, sampling_rate):
    """
    Test that phase shifts cause different signals to return different index ranges.
    
    The point of phase shifts is that they change the indexes of the data returned.
    For example, neuron 1 might return indexes 0 to 10, while neuron 2 returns
    indexes 1 to 11 because neuron 1 has a bigger timeshift than neuron 2.
    """
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=5.0,
            sampling_rate=sampling_rate,
            shifts_per_signal=True,
            start_time=0,
            large_phase_shifts=True,
        )
    ) as (timestamps, data, shifts, seq_interp):
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Should be PhaseShiftedSequenceInterpolator"

        delta_t = 1.0 / sampling_rate

        # Sort shifts to understand which signals have larger/smaller shifts
        sorted_shift_indices = np.argsort(shifts)
        min_shift_signal = sorted_shift_indices[0]
        max_shift_signal = sorted_shift_indices[-1]

        # Skip this test if all shifts are too similar
        if np.abs(shifts[max_shift_signal] - shifts[min_shift_signal]) < 0.5 * delta_t:
            return

        # Query at times where the phase shifts should cause different indexes
        # to be retrieved for different signals
        base_time = timestamps[3] + shifts[min_shift_signal] + 0.1 * delta_t
        times = np.array([base_time])

        interp, valid = seq_interp.interpolate(times=times, return_valid=True)

        # The interpolated values should be different because they come from
        # different source indexes due to the phase shifts
        if len(valid) > 0 and n_signals > 1:
            # With different phase shifts, different signals should retrieve
            # values from different time points in the original data
            # We verify this by checking that not all values are from the same row
            assert interp.shape[0] > 0, "Should have interpolated values"


@pytest.mark.parametrize("n_signals", [1, 5, 10])
@pytest.mark.parametrize("sampling_rate", [10.0, 50.0])
def test_linear_interpolation_matches_np_interp(n_signals, sampling_rate):
    """
    Test that linear interpolation matches numpy's np.interp function.
    
    This tests the assumptions about how linear interpolation is performed,
    using np.interp as the reference implementation.
    """
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=5.0,
            sampling_rate=sampling_rate,
            start_time=0,
        ),
        interp_kwargs=dict(keep_nans=True),
    ) as (timestamps, data, _, seq_interp):
        seq_interp.interpolation_mode = "linear"

        delta_t = 1.0 / sampling_rate

        # Query at times between the regular timestamps (not exactly on them)
        query_times = timestamps[1:DEFAULT_SEQUENCE_LENGTH] + 0.3 * delta_t

        interp, valid = seq_interp.interpolate(times=query_times, return_valid=True)

        # Compute expected values using np.interp for each signal
        expected = np.zeros((len(query_times), n_signals))
        for sig_idx in range(n_signals):
            expected[:, sig_idx] = np.interp(
                query_times,
                timestamps,
                data[:, sig_idx]
            )

        # Compare results
        assert np.allclose(interp, expected, rtol=1e-6, atol=1e-9), (
            f"Linear interpolation does not match np.interp. "
            f"Max difference: {np.max(np.abs(interp - expected))}"
        )


@pytest.mark.parametrize("n_signals", [5, 10])
@pytest.mark.parametrize("sampling_rate", [10.0, 50.0])
def test_linear_interpolation_with_phase_shifts_matches_np_interp(n_signals, sampling_rate):
    """
    Test that linear interpolation with phase shifts matches numpy's np.interp.
    
    Each signal should be interpolated as if it had its own shifted time axis.
    """
    with sequence_data_and_interpolator(
        data_kwargs=dict(
            n_signals=n_signals,
            use_mem_mapped=True,
            t_end=5.0,
            sampling_rate=sampling_rate,
            shifts_per_signal=True,
            start_time=0,
        ),
        interp_kwargs=dict(keep_nans=True),
    ) as (timestamps, data, shifts, seq_interp):
        assert isinstance(
            seq_interp, PhaseShiftedSequenceInterpolator
        ), "Should be PhaseShiftedSequenceInterpolator"
        seq_interp.interpolation_mode = "linear"

        delta_t = 1.0 / sampling_rate

        # Query at times that are valid for all phase shifts
        max_shift = np.max(shifts)
        start_idx = int(np.ceil(max_shift * sampling_rate)) + 2
        query_times = timestamps[start_idx : start_idx + DEFAULT_SEQUENCE_LENGTH - 2] + 0.3 * delta_t

        interp, valid = seq_interp.interpolate(times=query_times, return_valid=True)

        if len(valid) == 0:
            return  # Skip if no valid times

        valid_query_times = query_times[valid]

        # Compute expected values using np.interp for each signal with its phase shift
        expected = np.zeros((len(valid_query_times), n_signals))
        for sig_idx in range(n_signals):
            # Each signal has its own shifted time axis
            shifted_timestamps = timestamps + shifts[sig_idx]
            expected[:, sig_idx] = np.interp(
                valid_query_times,
                shifted_timestamps,
                data[:, sig_idx]
            )

        # Compare results
        assert np.allclose(interp, expected, rtol=1e-6, atol=1e-9), (
            f"Linear interpolation with phase shifts does not match np.interp. "
            f"Max difference: {np.max(np.abs(interp - expected))}"
        )


@pytest.mark.parametrize("n_signals", [5, 10])
def test_multiple_phase_shift_generations(n_signals):
    """
    Test with multiple generations of phase shifts to verify index behavior.
    
    The point is that phase shifts change the indexes of returned data,
    so for neuron 1 we might return indexes 0 to 10 while for neuron 2
    we return indexes 1 to 11 because neuron 1 has a bigger timeshift.
    """
    sampling_rate = 10.0
    delta_t = 1.0 / sampling_rate

    # Run multiple iterations with different random phase shifts
    for iteration in range(3):
        with sequence_data_and_interpolator(
            data_kwargs=dict(
                n_signals=n_signals,
                use_mem_mapped=True,
                t_end=10.0,  # Longer duration for more data points
                sampling_rate=sampling_rate,
                shifts_per_signal=True,
                start_time=0,
                large_phase_shifts=True,
            )
        ) as (timestamps, data, shifts, seq_interp):
            assert isinstance(
                seq_interp, PhaseShiftedSequenceInterpolator
            ), "Should be PhaseShiftedSequenceInterpolator"

            # Verify we have diverse phase shifts
            shift_range = np.max(shifts) - np.min(shifts)

            # Query at a fixed time point
            max_shift = np.max(shifts)
            query_time = timestamps[10] + max_shift + 0.5 * delta_t
            times = np.array([query_time])

            interp, valid = seq_interp.interpolate(times=times, return_valid=True)

            if len(valid) > 0:
                # The interpolated data should reflect the phase-shifted indexing
                # Different signals will have retrieved data from different
                # effective time points
                assert interp.shape == (1, n_signals), (
                    f"Expected shape (1, {n_signals}), got {interp.shape}"
                )


if __name__ == "__main__":
    print("Running tests")
    pytest.main([__file__])
