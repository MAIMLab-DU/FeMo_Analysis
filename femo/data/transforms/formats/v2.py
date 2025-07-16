import numpy as np
import struct
from dataclasses import dataclass
# Import the C++ module we built
import femo_parser_cpp as femo_parser_backend


@dataclass
class FileHeader:
    """Class to hold the V2 file header information."""
    start_time: int
    end_time: int
    freqpiezo: int
    freqaccel: int
    freqimu: int
    freqforce: int
    last_sync_time: int
    device_identifier: str
    device_mac_address: str
    

class FeMoDataV2:
    """
    A high-performance parser for v1.4 FeMo data files

    - Uses a C++ backend for fast, corruption-resilient parsing.
    - Presents data as raw NumPy arrays for DataLoader consumption.
    - Provides utility methods for data access.
    """
    def __init__(self, file_path: str, parse_on_init: bool = True):
        """
        Initializes the parser for a given file path.

        Args:
            file_path (str): The path to the .bin sensor data file.
            parse_on_init (bool): If True, automatically parses the entire file upon creation.
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string.")
        self.file_path: str = file_path

        # --- Data containers ---
        self.raw_result = None
        self.header = None
        self._arrays: dict[str, np.ndarray] = {}

        if parse_on_init:
            self.parse()

    def parse(self, start_byte: int = -1, end_byte: int = -1, max_packets: int = -1):
        """
        Parses the file and loads the data into the object's attributes.

        Args:
            start_byte (int, optional): The byte offset to start parsing from. Defaults to -1 (start of data).
            end_byte (int, optional): The byte offset to stop parsing at. Defaults to -1 (end of file).
            max_packets (int, optional): The maximum number of packets to read. Defaults to -1 (no limit).
        
        Returns:
            Itself (self), to allow for method chaining.
        """
        # Call the high-performance C++ function
        self.raw_result = femo_parser_backend.parse_femo_file(
            self.file_path,
            start_byte=start_byte,
            end_byte=end_byte,
            max_packets=max_packets
        )
        
        # Process the raw C++ output into final NumPy arrays
        self._build_results()        
        return self

    def _build_results(self) -> None:
        """
        Internal method to build the .metadata and ._arrays attributes
        from the raw parser output.
        """
        if not self.raw_result or self.raw_result.packets_found == 0:
            print("Warning: No valid packets found. Data arrays will be empty.")
            return

        # 1. --- Build Metadata ---
        file_header = self.raw_result.file_header
        self.header = FileHeader(
                start_time=file_header.startTimestamp*1000 + file_header.startMs,
                end_time=file_header.finishTimestamp*1000 + file_header.finishMs,
                freqpiezo=file_header.piezoFreq,
                freqimu=file_header.IMUFreq,
                freqaccel=file_header.accelerometerFreq,  # Not used in V2
                freqforce=file_header.forceSensorFreq,  # Not used in V2
                last_sync_time=file_header.lastSyncTimestamp*1000 + file_header.lastSyncMs,
                device_identifier=file_header.deviceId.strip('\x00'),
                device_mac_address=':'.join(f'{byte:02X}' for byte in file_header.macAddress)
            )

        # 2. --- Build NumPy Arrays (matching V1 structure) ---
        num_packets: int = self.raw_result.packets_found
        
        # Button data: [record_index, button_state]
        # Button data is per-packet, but we need to expand it to match the sample rate
        # Each packet has 64 samples, so repeat each button state 64 times
        # Alternative could be repeating only 0s (up for discussion)
        button_buffer = np.column_stack([
            np.arange(num_packets * 64, dtype=np.int32),
            np.repeat(self.raw_result.button_pressed.astype(np.int32), 64)
        ])
        
        # Timestamp data: [record_index, timestamp_seconds, timestamp_milliseconds] - Unused
        timestamp_buffer = np.column_stack([
            np.arange(num_packets, dtype=np.int32),
            self.raw_result.packet_timestamps.astype(np.int64)
        ])
        
        # Piezo data: [record_index, piezo_left, piezo_right]
        piezo_buffer = np.column_stack([
            np.arange(num_packets * 64, dtype=np.int32),
            self.raw_result.piezo1_data,
            self.raw_result.piezo2_data,
        ])
        
        # IMU data: [record_index, quat_r, quat_i, quat_j, quat_k, mag_x, mag_y, mag_z, accel_x, accel_y, accel_z] - matching V1 format
        # Need confirmation on scaling factors
        imu_buffer = np.column_stack([
            np.arange(0, num_packets * 64, 8, dtype=np.int32),
            (self.raw_result.quat_r),   # quat_r
            (self.raw_result.quat_i),   # quat_i
            (self.raw_result.quat_j),   # quat_j
            (self.raw_result.quat_k),   # quat_k
            (self.raw_result.mag_x),    # mag_x
            (self.raw_result.mag_y),    # mag_y
            (self.raw_result.mag_z),    # mag_z
            (self.raw_result.accel_x),  # accel_x
            (self.raw_result.accel_y),  # accel_y
            (self.raw_result.accel_z)   # accel_z
        ])
        
        # Store arrays with same keys as V1
        self._arrays = {
            'piezo': piezo_buffer,
            'accel': np.empty((0, 7), dtype=np.int32),  # Not used in V2
            'imu': imu_buffer,
            'force': np.empty((0, 2), dtype=np.int32),  # Not used in V2
            'timestamp': timestamp_buffer,
            'button': button_buffer
        }

    def __repr__(self) -> str:
        if not self.raw_result:
            return f"<FeMoDataV2 for '{self.file_path}' (unparsed)>"
        return (f"<FeMoDataV2 for '{self.file_path}'>\n"
                f"  - Metadata: {self.header}\n"
                f"  - Available arrays: {list(self._arrays.keys())}")


def create_dummy_file(filepath: str, num_packets: int) -> None:
    """Creates a valid, dummy v1.4 sensor file for testing."""
    print(f"Creating dummy file '{filepath}' with {num_packets} packets...")
    with open(filepath, 'wb') as file_handle:
        # 1. File Header (48 bytes)
        file_handle.write(struct.pack('<I', 0x48414148))  # HAAH
        file_handle.write(struct.pack('<LHLH', 1700000000, 500, 1700000000 + num_packets // 16, 500))
        file_handle.write(struct.pack('<HHHH', 1024, 128, 128, 0))  # Frequencies
        file_handle.write(struct.pack('<LHI', 0, 0, 0))  # Sync time, bytes written
        file_handle.write(b'DV1\x00')  # Device ID
        file_handle.write(b'\xDE\xAD\xBE\xEF\x00\x00')  # MAC
        file_handle.write(struct.pack('<I', 0x485A5A48))  # HZZH
        
        # 2. Data Packets (424 bytes each)
        for packet_index in range(num_packets):
            # Packet Header (8 bytes)
            file_handle.write(struct.pack('<HBLB', 0x504B, packet_index % 256, packet_index * 62, 0))
            
            # Piezo Data (256 bytes)
            for sample_index in range(64):
                # Sine wave for some interesting data
                piezo1_value = int(32768 + 20000 * np.sin(2 * np.pi * (packet_index * 64 + sample_index) / 1024.0))
                piezo2_value = int(32768 + 15000 * np.cos(2 * np.pi * (packet_index * 64 + sample_index) / 512.0))
                file_handle.write(struct.pack('<HH', piezo1_value, piezo2_value))
            
            # IMU Data (160 bytes)
            imu_sample_data = struct.pack('<hhhhhhhhhh', 10000, 0, 0, 0, 50, 60, 70, 0, 0, 9810)
            for _ in range(8):
                file_handle.write(imu_sample_data)
                
        # 3. End Signature (4 bytes)
        file_handle.write(struct.pack('<I', 0x445A5A44))  # DZZD


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Check that required libraries are available before proceeding
    if femo_parser_backend is None or plt is None:
        print("\nAborting test script due to missing libraries.")
    else:
        dummy_file_path: str = "test_sensor_data.bin"
        num_packets_to_create: int = 200  # about 12.5 seconds of data
        
        try:
            # 1. Generate a test file
            create_dummy_file(dummy_file_path, num_packets_to_create)
            
            # 2. Initialize the parser and parse the file
            print(f"\nParsing '{dummy_file_path}' with FeMoDataV2...")
            femo_data_parser: FeMoDataV2 = FeMoDataV2(dummy_file_path)
            print(femo_data_parser)

            # 3. Access raw arrays for plotting
            piezo_sensor_array: np.ndarray = femo_data_parser._arrays.get('piezo')

            if piezo_sensor_array is not None and piezo_sensor_array.size > 0:
                print("\nPlotting 'piezo1' data channel...")
                
                # 4. Calculate time-based x-axis (equivalent to DataFrame multi-index)
                # Each packet contains 64 samples, so we need to create sample indices
                number_of_packets: int = len(piezo_sensor_array)
                samples_per_packet: int = 64
                
                # Create time axis: packet_index * samples_per_packet + sample_index
                packet_indices: np.ndarray = piezo_sensor_array[:, 0]  # First column is packet index
                sample_indices_in_packet: np.ndarray = np.zeros(number_of_packets, dtype=np.int32)
                
                # Calculate equivalent DataFrame multi-index time axis
                time_axis_samples: np.ndarray = packet_indices * samples_per_packet + sample_indices_in_packet
                
                # Convert to seconds if needed (assuming sampling frequency from metadata)
                piezo_frequency_hz: int = femo_data_parser.header.freqpiezo
                time_axis_seconds: np.ndarray = time_axis_samples.astype(np.float64) / piezo_frequency_hz
                
                # Extract piezo1 data (column 1 in the array)
                piezo1_sensor_values: np.ndarray = piezo_sensor_array[:, 1]
                
                # 5. Plot the data using time in seconds (limited samples for clarity)
                max_samples_to_plot: int = min(1000, len(piezo_sensor_array))
                
                plt.figure(figsize=(12, 6))
                plt.plot(time_axis_seconds[:max_samples_to_plot], 
                        piezo1_sensor_values[:max_samples_to_plot], 
                        label='Piezo 1')
                
                plt.title(f"Piezo 1 Data from '{dummy_file_path}' (first {max_samples_to_plot} samples)")
                plt.xlabel("Time (seconds)")
                plt.ylabel("ADC Value")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                print("Showing plot. Close the plot window to exit the script.")
                plt.show()
            else:
                print("\nCould not plot data: 'piezo' array is empty or missing.")

        except Exception as parsing_error:
            print(f"\nAn error occurred during the test run: {parsing_error}")
            import traceback
            traceback.print_exc()
