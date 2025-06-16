import os
import struct
import numpy as np
import mmap
from numba import njit
from dataclasses import dataclass


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


class FeMoDataV1:
    """
    V1-format reader for .dat sensor logs:
      - reads header
      - counts records
      - allocates exact-size buffers
      - memory-maps + JIT-parses in one pass
      - stores raw arrays for DataLoader consumption
    """

    def __init__(self, inputfile: str) -> None:
        self.inputfile: str = inputfile
        self.header: Header = None
        self._arrays: dict[str, np.ndarray] = {}
        self._read()

    def _read(self) -> None:
        # 1) Read header
        with open(self.inputfile, 'rb') as file_handle:
            start_seconds, start_milliseconds = struct.unpack('<LH', file_handle.read(6))
            end_seconds, end_milliseconds = struct.unpack('<LH', file_handle.read(6))
            frequency_piezo, = struct.unpack('<H', file_handle.read(2))
            frequency_accel, = struct.unpack('<H', file_handle.read(2))
            frequency_imu, = struct.unpack('<H', file_handle.read(2))
            frequency_force, = struct.unpack('<H', file_handle.read(2))
            header_end_offset = file_handle.tell()

        self.header = Header(
            start_time=start_seconds * 1000 + start_milliseconds,
            end_time=end_seconds * 1000 + end_milliseconds,
            freqpiezo=frequency_piezo,
            freqaccel=frequency_accel,
            freqimu=frequency_imu,
            freqforce=frequency_force
        )

        # 2) Count total records
        with open(self.inputfile, 'rb') as file_handle:
            total_record_count = self._count_records(file_handle, header_end_offset)

        # 3) Allocate exact buffers
        piezo_buffer = np.empty((total_record_count, 5), dtype=np.int32)
        accel_buffer = np.empty((total_record_count, 7), dtype=np.int32)
        imu_buffer = np.empty((total_record_count, 11), dtype=np.int32)
        force_buffer = np.empty((total_record_count, 2), dtype=np.int32)
        timestamp_buffer = np.empty((total_record_count, 3), dtype=np.int64)
        button_buffer = np.empty((total_record_count, 2), dtype=np.int32)

        # 4) Memory-map and copy into a NumPy array
        with open(self.inputfile, 'rb') as file_handle:
            memory_map = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            file_buffer = np.frombuffer(memory_map, dtype=np.uint8).copy()
            memory_map.close()

        # 5) JIT-parse all records
        (actual_piezo_count, actual_accel_count, actual_imu_count, 
         actual_force_count, actual_timestamp_count, actual_record_count) = parse_records(
            file_buffer, header_end_offset,
            piezo_buffer, accel_buffer, imu_buffer, force_buffer, timestamp_buffer, button_buffer
        )

        # 6) Slice to actual lengths and store arrays
        self._arrays = {
            'piezo': piezo_buffer[:actual_piezo_count],
            'accel': accel_buffer[:actual_accel_count],
            'imu': imu_buffer[:actual_imu_count],
            'force': force_buffer[:actual_force_count],
            'timestamp': timestamp_buffer[:actual_timestamp_count],
            'button': button_buffer[:actual_record_count],
        }

    def _count_records(self, file_handle, offset: int) -> int:
        """Count the total number of records in the file."""
        file_handle.seek(offset)
        record_count = 0
        while True:
            byte_data = file_handle.read(1)
            if not byte_data:
                break
            descriptor_byte = byte_data[0]
            record_count += 1
            if descriptor_byte & 0x80:
                file_handle.seek(8, os.SEEK_CUR)
            if descriptor_byte & 0x40:
                file_handle.seek(12, os.SEEK_CUR)
            if descriptor_byte & 0x20:
                file_handle.seek(20, os.SEEK_CUR)
            if descriptor_byte & 0x10:
                file_handle.seek(2, os.SEEK_CUR)
            if descriptor_byte & 0x08:
                file_handle.seek(6, os.SEEK_CUR)
        return record_count
    
'''    
Byte Ordering (Endianness):
On Windows, the default byte order might match the format expected by your data structure, but on Linux (Ubuntu), 
the default might differ, causing the unpacking to fail if the byte order is not explicitly specified.
You can specify the byte order in your struct.unpack format string. Use < for little-endian (common in x86, x86-64 architectures) 
or > for big-endian to ensure consistency across platforms. For example, if your data is in little-endian format, modify your format string to "<LH".
'''
