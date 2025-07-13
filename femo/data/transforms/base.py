import os
from ...logger import LOGGER
import struct
import pandas as pd
import numpy as np
import mmap
from numba import njit
from abc import ABC, abstractmethod
from dataclasses import dataclass

class BaseTransform(ABC):

    @property
    def sensor_map(self):
        return self._sensor_map

    @property
    def logger(self):
        return LOGGER
    
    @property
    def sensor_selection(self) -> list:
        return self._sensor_selection
    
    @property
    def sensor_freq(self) -> int:
        return self._sensor_freq
    
    @property
    def sensation_freq(self) -> int:
        return self._sensation_freq
    
    @property
    def sensors(self) -> list:
        sensors = []
        for s in self.sensor_selection:
            if '_left' in s or '_right' in s:
                key, side = s.rsplit('_', 1)
                index = 0 if side == 'left' else 1
                sensors.append(self.sensor_map[key][index])
            else:
                sensors.extend(self.sensor_map[s])
        return sorted(sensors)

    @property
    def num_sensors(self) -> int:
        return len(self.sensors)
    
    @property
    def use_all_sensors(self) -> bool:
        return True if len(self.sensor_selection) == len(self.sensor_map) else False
    
    def __init__(self,
                 description: str = "Only large piezos from belt types 'A' and 'C'",
                 sensor_freq: int = 1024,
                 sensation_freq: int = 1024,
                 sensor_map: dict = {
                     'accelerometer': ['sensor_1', 'sensor_2'],
                     'piezoelectric_large': ['sensor_3', 'sensor_6'],
                     'other': ['sensor_4', 'sensor_5']
                 },
                 sensor_selection: list = ['piezoelectric_large']) -> None:
        super().__init__()

        self._description = description
        self._sensor_map = sensor_map
        self._sensor_selection = sensor_selection
        self._sensor_freq = sensor_freq
        self._sensation_freq = sensation_freq
    
    def __repr__(self):
        return self._description

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    
    @abstractmethod
    def transform(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class Header:
    start_time: int
    end_time: int
    freqpiezo: int
    freqaccel: int
    freqimu: int
    freqforce: int


@njit
def parse_records(buf, hdr_end,
                  piezo_buf, accel_buf, imu_buf, force_buf, ts_buf, btn_buf):
    pi = ai = ii = fi = ti = rec = 0
    pos = hdr_end
    N = buf.shape[0]
    while pos < N:
        d = buf[pos]
        pos += 1

        if d & 0x80:
            p0 = buf[pos] | (buf[pos+1]<<8)
            p1 = buf[pos+2] | (buf[pos+3]<<8)
            p2 = buf[pos+4] | (buf[pos+5]<<8)
            p3 = buf[pos+6] | (buf[pos+7]<<8)
            piezo_buf[pi, 0] = rec
            piezo_buf[pi, 1] = p0
            piezo_buf[pi, 2] = p1
            piezo_buf[pi, 3] = p2
            piezo_buf[pi, 4] = p3
            pi += 1
            pos += 8

        if d & 0x40:
            a0 = buf[pos] | (buf[pos+1]<<8)
            a1 = buf[pos+2] | (buf[pos+3]<<8)
            a2 = buf[pos+4] | (buf[pos+5]<<8)
            a3 = buf[pos+6] | (buf[pos+7]<<8)
            a4 = buf[pos+8] | (buf[pos+9]<<8)
            a5 = buf[pos+10] | (buf[pos+11]<<8)
            accel_buf[ai, 0] = rec
            accel_buf[ai, 1] = a0
            accel_buf[ai, 2] = a1
            accel_buf[ai, 3] = a2
            accel_buf[ai, 4] = a3
            accel_buf[ai, 5] = a4
            accel_buf[ai, 6] = a5
            ai += 1
            pos += 12

        if d & 0x20:
            imu_buf[ii, 0] = rec
            for k in range(10):
                low  = buf[pos + 2*k]
                high = buf[pos + 2*k + 1]
                val  = (high<<8) | low
                if val & 0x8000:
                    val -= 1 << 16
                imu_buf[ii, k+1] = val
            ii += 1
            pos += 20

        if d & 0x10:
            v = buf[pos] | (buf[pos+1]<<8)
            force_buf[fi, 0] = rec
            force_buf[fi, 1] = v
            fi += 1
            pos += 2

        if d & 0x08:
            s = (buf[pos]   | (buf[pos+1]<<8) |
                 (buf[pos+2]<<16)|(buf[pos+3]<<24))
            m = buf[pos+4] | (buf[pos+5]<<8)
            ts_buf[ti, 0] = rec
            ts_buf[ti, 1] = s
            ts_buf[ti, 2] = m
            ti += 1
            pos += 6

        btn_buf[rec, 0] = rec
        btn_buf[rec, 1] = d & 0x1
        rec += 1

    return pi, ai, ii, fi, ti, rec


class FeMoData:
    """
    V1-format reader for .dat sensor logs:
      - reads header
      - counts records
      - allocates exact-size buffers
      - memory-maps + JIT-parses in one pass
      - slices to actual lengths and builds DataFrames
    """

    def __init__(self, inputfile: str):
        self.inputfile = inputfile
        self.header: Header = None
        self.dataframes: dict[str, pd.DataFrame] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._read()
        self._build_dataframes()

    def _read(self):
        # 1) Read header
        with open(self.inputfile, 'rb') as f:
            st_sec, st_ms = struct.unpack('<LH', f.read(6))
            en_sec, en_ms = struct.unpack('<LH', f.read(6))
            f_pz, = struct.unpack('<H', f.read(2))
            f_ac, = struct.unpack('<H', f.read(2))
            f_imu,= struct.unpack('<H', f.read(2))
            f_f,   = struct.unpack('<H', f.read(2))
            header_end = f.tell()

        self.header = Header(
            start_time=st_sec*1000 + st_ms,
            end_time=en_sec*1000   + en_ms,
            freqpiezo=f_pz,
            freqaccel=f_ac,
            freqimu=f_imu,
            freqforce=f_f
        )

        # 2) Count total records
        with open(self.inputfile, 'rb') as f:
            total_recs = self._count_records(f, header_end)

        # 3) Allocate exact buffers
        piezo_buf = np.empty((total_recs, 5),  dtype=np.int32)
        accel_buf = np.empty((total_recs, 7),  dtype=np.int32)
        imu_buf   = np.empty((total_recs,11),  dtype=np.int32)
        force_buf = np.empty((total_recs, 2),  dtype=np.int32)
        ts_buf    = np.empty((total_recs, 3),  dtype=np.int64)
        btn_buf   = np.empty((total_recs, 2),  dtype=np.int32)

        # 4) Memory-map and copy into a NumPy array
        with open(self.inputfile, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            buf = np.frombuffer(mm, dtype=np.uint8).copy()
            mm.close()

        # 5) JIT-parse all records
        pi, ai, ii, fi, ti, rec = parse_records(
            buf, header_end,
            piezo_buf, accel_buf, imu_buf, force_buf, ts_buf, btn_buf
        )

        # 6) Slice to actual lengths
        self._arrays = {
            'piezo':          piezo_buf[:pi],
            'accel': accel_buf[:ai],
            'imu':            imu_buf[:ii],
            'force':          force_buf[:fi],
            'timestamp':      ts_buf[:ti],
            'button':         btn_buf[:rec],
        }

    def _count_records(self, f, offset: int) -> int:
        f.seek(offset)
        cnt = 0
        while True:
            b = f.read(1)
            if not b:
                break
            d = b[0]
            cnt += 1
            if d & 0x80:
                f.seek(8, os.SEEK_CUR)
            if d & 0x40:
                f.seek(12, os.SEEK_CUR)
            if d & 0x20:
                f.seek(20, os.SEEK_CUR)
            if d & 0x10:
                f.seek(2, os.SEEK_CUR)
            if d & 0x08:
                f.seek(6, os.SEEK_CUR)
        return cnt

    def _build_dataframes(self):
        a = self._arrays
        self.dataframes['piezos'] = (
            pd.DataFrame(a['piezo'],
                         columns=['measurement_index','p1','p2','p3','p4'])
             .set_index('measurement_index')
        )
        self.dataframes['accel'] = (
            pd.DataFrame(a['accel'],
                         columns=['measurement_index','x1','y1','z1','x2','y2','z2'])
             .set_index('measurement_index')
        )
        self.dataframes['imu'] = (
            pd.DataFrame(a['imu'],
                         columns=[
                             'measurement_index',
                             'rotation_r','rotation_i','rotation_j','rotation_k',
                             'magnet_x','magnet_y','magnet_z',
                             'accel_x','accel_y','accel_z'
                         ])
             .set_index('measurement_index')
        )
        self.dataframes['force'] = (
            pd.DataFrame(a['force'],
                         columns=['measurement_index','f'])
             .set_index('measurement_index')
        )
        self.dataframes['timestamp'] = (
            pd.DataFrame(a['timestamp'],
                         columns=['measurement_index','sec','millis'])
             .set_index('measurement_index')
        )
        self.dataframes['push_button'] = (
            pd.DataFrame(a['button'],
                         columns=['measurement_index','button'])
             .set_index('measurement_index')
        )

    def get(self, name: str) -> pd.DataFrame | None:
        return self.dataframes.get(name)

    def to_parquet(self, folder: str = '') -> None:
        if not folder:
            folder = os.path.join(os.getcwd(), self.inputfile + '_parquet')
        os.makedirs(folder, exist_ok=True)
        for key, df in self.dataframes.items():
            df.to_parquet(os.path.join(folder, f"{key}.parquet"))

'''    
Byte Ordering (Endianness):
On Windows, the default byte order might match the format expected by your data structure, but on Linux (Ubuntu), 
the default might differ, causing the unpacking to fail if the byte order is not explicitly specified.
You can specify the byte order in your struct.unpack format string. Use < for little-endian (common in x86, x86-64 architectures) 
or > for big-endian to ensure consistency across platforms. For example, if your data is in little-endian format, modify your format string to "<LH".
'''