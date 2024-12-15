import os
from ...logger import LOGGER
import struct
import pandas as pd
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


class FeMoData:
    def __init__(self, inputfile):
        self.inputfile=inputfile
        self.piezo_list = []
        self.accelerometer_list = []
        self.imu_list = []
        self.force_list = []
        self.timestamp_list = []
        self.button_list = []

        self.dataframes ={}
        self.header = None

        self._read()
        self.fill_dataframes()


    def fill_dataframes(self):
        self.dataframes["piezos"] = (pd.DataFrame(self.piezo_list,
                                                 columns=['measurement_index','p1','p2','p3','p4'])
                                                 .set_index("measurement_index")
                                                 )
        self.dataframes["accelerometers"] =( pd.DataFrame(self.accelerometer_list,
                                                         columns=['measurement_index','x1','y1','z1','x2', 'y2','z2'])
                                                         .set_index("measurement_index")
                                                         )
        self.dataframes["imu"] = (pd.DataFrame(self.imu_list,
                                               columns=['measurement_index','rotation_r','rotation_i','rotation_j', 'rotation_k',
                                                'magnet_x','magnet_y','magnet_z',
                                                'accel_x','accel_y','accel_z'])
                                                .set_index("measurement_index")
                                                )
        self.dataframes["force"] = (pd.DataFrame(self.force_list,
                                                 columns=['measurement_index','f'])
                                                 .set_index("measurement_index")
                                                 )
        self.dataframes["timestamp"] = (pd.DataFrame(self.timestamp_list,
                                                     columns=['measurement_index','sec','millis']
                                                     )
                                                     .set_index("measurement_index")
                                                     )
        self.dataframes["push_button"] = (pd.DataFrame(self.button_list,
                                                       columns=['measurement_index','button']
                                                       )
                                                       .set_index("measurement_index")
                                                    )
        
    def get(self,dataframe):
        return self.dataframes.get(dataframe, None)
        

    def to_parquet(self, folder=''):
        if not folder:
            folder = os.path.join(os.getcwd(),self.inputfile+"_parquet")
        if not os.path.exists(folder):
            os.mkdir(folder)
        for key, frame in self.dataframes.items():
            frame.to_parquet(os.path.join(folder,key+".parquet"))


    def _read(self):

        unpack_piezo = struct.Struct('HHHH').unpack
        unpack_accelerometer = struct.Struct("HHHHHH").unpack
        unpack_imu = struct.Struct("hhhhhhhhhh").unpack
        unpack_u2 = struct.Struct("H").unpack
        unpack_timestamp = struct.Struct("LH").unpack

        with open(self.inputfile, 'rb') as file:
            file.seek(0, 2)
            file_len = file.tell()  # noqa: F841
            file.seek(0)

            self.header = Header(struct.unpack("<LH",file.read(6)),
                                 struct.unpack("<LH",file.read(6)),
                                 struct.unpack("<H",file.read(2)),
                                 struct.unpack("<H",file.read(2)),
                                 struct.unpack("<H",file.read(2)),
                                 struct.unpack("<H",file.read(2))
                                 )

            

            measurement_index = 0

            descriptor = file.read(1)

            while descriptor:
                
                if (descriptor[0] >> 7) & 1:
                    self.piezo_list.append( [measurement_index] +  list(unpack_piezo(file.read(8))))

                if (descriptor[0] >> 6) & 1:
                    self.accelerometer_list.append( [measurement_index] +  list(unpack_accelerometer(file.read(12))))
                
                if (descriptor[0] >> 5) & 1:
                    self.imu_list.append( [measurement_index] + list(unpack_imu(file.read(20))))

                if (descriptor[0] >> 4) & 1:
                    self.force_list.append( [measurement_index] +  list(unpack_u2(file.read(2))))
                
                if (descriptor[0] >> 3) & 1:
                    self.timestamp_list.append( [measurement_index] +  list(unpack_timestamp(file.read(6))))

                self.button_list.append( [measurement_index,  descriptor[0] & 1])
                measurement_index += 1
                descriptor = file.read(1)

'''    
Byte Ordering (Endianness):
On Windows, the default byte order might match the format expected by your data structure, but on Linux (Ubuntu), 
the default might differ, causing the unpacking to fail if the byte order is not explicitly specified.
You can specify the byte order in your struct.unpack format string. Use < for little-endian (common in x86, x86-64 architectures) 
or > for big-endian to ensure consistency across platforms. For example, if your data is in little-endian format, modify your format string to "<LH".
'''