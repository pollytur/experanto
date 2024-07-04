from pathlib import Path
import numpy as np
from abc import abstractmethod
import yaml
import numpy.lib.format as fmt
import os
import yaml
import warnings
import re


class TimeInterval:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end

    def __contains__(self, time):
        return self.start <= time < self.end

    def intersect(self, times):
        return (times >= self.start) & (times < self.end)
        
        
    def __repr__(self) -> str:
        return f"TimeInterval [{self.start}, {self.end})"


class Interpolator:

    def __init__(self, root_folder: str) -> None:
        self.root_folder = Path(root_folder)
        meta = self.load_meta()
        self.start_time = meta["start_time"]
        self.end_time = meta["end_time"]
        # Valid interval can be different to start time and end time. 
        self.valid_interval = TimeInterval(self.start_time, self.end_time)

    def load_meta(self):
        with open(self.root_folder / "meta.yml") as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        return meta

    @abstractmethod
    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...
        # returns interpolated signal and boolean mask of valid samples

    def __contains__(self, times: np.ndarray):
        return np.any(self.valid_times(times))

    @staticmethod
    def create(root_folder: str) -> "Interpolator":
        with open(Path(root_folder) / 'meta.yml', 'r') as file:
            meta_data = yaml.safe_load(file)
        modality = meta_data.get('modality')
        class_name = modality.capitalize() + "Interpolator"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](root_folder)

    def __contains__(self, times: np.ndarray):
        return np.any((times >= self.timestamps[0]) & (times <= self.timestamps[-1]))
    
    def valid_times(self, times: np.ndarray) -> np.ndarray:
        return self.valid_interval.intersect(times)


class SequenceInterpolator(Interpolator):

    def __init__(self, root_folder: str) -> None:
        super().__init__(root_folder)
        meta = self.load_meta()
        self.time_delta = 1./meta["sampling_rate"]

        self.use_phase_shifts = meta["phase_shift_per_signal"]
        if meta["phase_shift_per_signal"]:
            self._phase_shifts = np.load(self.root_folder / "meta/phase_shifts.npy")
            self.valid_interval = TimeInterval(
                self.start_time + np.max(self._phase_shifts),
                self.end_time + np.min(self._phase_shifts),
            )
        self._data = np.load(self.root_folder / "data.npy")

    def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = self.valid_times(times)
        valid_times = times[valid]

        idx = np.round((valid_times[:, np.newaxis] - self._phase_shifts[np.newaxis, :] - self.start_time) / self.time_delta).astype(int)
        return np.take_along_axis(self._data, idx, axis=0), valid


class ScreenInterpolator(Interpolator):

    def __init__(self, root_folder: str) -> None:
        super().__init__(root_folder)
        self.timestamps = np.load(self.root_folder / "timestamps.npy")
        self._parse_meta()

        # create mapping from image index to file index
        self._data_files = [Path(root_folder) / "data" / (m.file_name + ".npy") for m in self.meta]
        self._num_frames = [m.num_frames for m in self.meta]
        self._first_frame_idx = [m.first_frame for m in self.meta]
        self._data_file_idx = np.concatenate([np.full(m.num_frames, i) for i, m in enumerate(self.meta)])

        self._image_size = self.meta[0].image_size
        assert np.all([m.image_size == self._image_size for m in self.meta]), 'All files must have the same image size'

    def _parse_meta(self) -> None:
        # Function to check if a file is a numbered yml file
        def is_numbered_yml(file_name):
            return re.fullmatch(r'\d{5}\.yml', file_name) is not None

        # Get block subfolders and sort by number
        meta_files = [f for f in (self.root_folder / "meta").iterdir() if f.is_file() and is_numbered_yml(f.name)]
        meta_files.sort(key=lambda f: int(os.path.splitext(f.name)[0]))

        self.meta = []
        for f in meta_files:
            self.meta.append(ScreenMeta.create(f))

    def interpolate(self, times: np.ndarray) -> tuple:
        valid = self.valid_times(times)
        valid_times = times[valid]
        valid_times += 1e-6 # add small offset to avoid numerical issues

        assert np.all(np.diff(valid_times) > 0), "Times must be sorted"
        idx = np.searchsorted(self.timestamps, valid_times) - 1 # convert times to frame indices
        assert np.all((idx >= 0) & (idx < len(self.timestamps))), "All times must be within the valid range"
        data_file_idx = self._data_file_idx[idx]
        
        # Go through files, load them and extract all frames
        unique_file_idx = np.unique(data_file_idx)
        out = np.zeros([len(valid_times)] + list(self._image_size))
        for u_idx in unique_file_idx:
            data = np.load(self._data_files[u_idx])
            idx_for_this_file = np.where(self._data_file_idx[idx] == u_idx)
            out[idx_for_this_file] = data[idx[idx_for_this_file] - self._first_frame_idx[u_idx]]

        return out, valid
    

class ScreenMeta():
    def __init__(self, file_name: str, data: dict, image_size: tuple, first_frame: int, num_frames: int) -> None:
        self.file_name = file_name
        self._data = data
        self.modality = data.get('modality')
        self.image_size = image_size
        self.first_frame = first_frame
        self.num_frames = num_frames

    @staticmethod
    def create(file_name: str) -> "ScreenMeta":
        with open(file_name, 'r') as file:
            meta_data = yaml.safe_load(file)
        modality = meta_data.get('modality')
        class_name = modality.capitalize() + "Meta"
        assert class_name in globals(), f"Unknown modality: {modality}"
        return globals()[class_name](Path(file_name).stem, meta_data)
    

class ImageMeta(ScreenMeta):
    def __init__(self, file_name, data) -> None:
        super().__init__(file_name, data, tuple(data.get("image_size")), data.get("first_frame"), 2)


class VideoMeta(ScreenMeta):
    def __init__(self, file_name, data) -> None:
        super().__init__(file_name, data, tuple(data.get("image_size")), data.get("first_frame"), data.get("num_frames"))

 