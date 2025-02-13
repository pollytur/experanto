import yaml
from pathlib import Path
import numpy as np

SEQUENCE_ROOT = Path("tests/sequence_data")

def create_sequence_data(n_signals = 10, shifts_per_signal = False, use_mem_mapped = False, t_end = 10.0, sampling_rate = 10.0):
    SEQUENCE_ROOT.mkdir(parents=True, exist_ok=True)
    (SEQUENCE_ROOT / "meta").mkdir(parents=True, exist_ok=True)
    
    meta = {
        "start_time": 0,
        "end_time": t_end,
        "modality": "sequence",
        "sampling_rate": sampling_rate,
        "phase_shift_per_signal": shifts_per_signal,
        "is_mem_mapped": use_mem_mapped,
        "n_signals": n_signals,
    }
    
        
    timestamps = np.linspace(meta["start_time"], meta["end_time"], int((meta["end_time"] - meta["start_time"]) * meta["sampling_rate"]) + 1)
    np.save(SEQUENCE_ROOT / "timestamps.npy", timestamps)
    meta["n_timestamps"] = len(timestamps)
    

    data = np.random.rand(len(timestamps), n_signals)
    if not use_mem_mapped:
        np.save(SEQUENCE_ROOT / "data.npy", data)
    else:
        filename = SEQUENCE_ROOT / "data.npy"
        
        fp = np.memmap(filename, dtype=data.dtype, mode='w+', shape=data.shape)
        fp[:] = data[:]
        fp.flush()  # Ensure data is written to disk
    meta["dtype"] = str(data.dtype)
    
    if shifts_per_signal:
        shifts = np.random.rand(n_signals) / meta["sampling_rate"] * 0.9
        np.save(SEQUENCE_ROOT / "meta" / "phase_shifts.npy", shifts)

    with open(SEQUENCE_ROOT / "meta.yml", "w") as f:
        yaml.dump(meta, f)


    return timestamps, data, shifts if shifts_per_signal else None
    
if __name__ == "__main__":
    create_sequence_data()