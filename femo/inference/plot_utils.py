import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Union, List, Tuple
import pandas as pd


def trim_data(data: np.ndarray, sampling_freq: int) -> np.ndarray:
    """Trim edge data based on duration: 30s for long recordings, 5s for short."""
    duration_sec: float = len(data) / sampling_freq
    removal_period: int = 30 if duration_sec > 300 else 5
    trim_samples: int = int(removal_period * sampling_freq)
    
    return data[trim_samples:-trim_samples]


def determine_time_unit_and_scale(data_length: int, sensor_freq: int) -> Tuple[str, float, str]:
    """
    Determine the appropriate time unit and scaling factor based on data duration.
    
    Args:
        data_length: Number of data points
        sensor_freq: Sensor frequency in Hz
        
    Returns:
        Tuple of (unit, scale_factor, unit_label)
    """
    duration_seconds = data_length / sensor_freq
    
    if duration_seconds < 120:  # Less than 2 minutes
        return "sec", 1.0, "Time (seconds)"
    elif duration_seconds < 7200:  # Less than 2 hours
        return "min", 60.0, "Time (minutes)"
    else:
        return "hour", 3600.0, "Time (hours)"
    

def create_time_axis(data_length: int, sensor_freq: int, 
                     force_unit: str = None) -> Tuple[np.ndarray, str]:
    """
    Create time axis with appropriate scaling.
    
    Args:
        data_length: Number of data points
        sensor_freq: Sensor frequency in Hz
        force_unit: Force specific unit ('sec', 'min', 'hour')
        
    Returns:
        Tuple of (time_array, unit_label)
    """
    if force_unit:
        unit = force_unit
        if unit == "sec":
            scale_factor, unit_label = 1.0, "Time (seconds)"
        elif unit == "min":
            scale_factor, unit_label = 60.0, "Time (minutes)"
        elif unit == "hour":
            scale_factor, unit_label = 3600.0, "Time (hours)"
        else:
            raise ValueError(f"Unknown unit: {force_unit}")
    else:
        unit, scale_factor, unit_label = determine_time_unit_and_scale(data_length, sensor_freq)
    
    time_axis = np.arange(data_length) / sensor_freq / scale_factor
    return time_axis, unit_label


def create_figure(figsize: tuple = (16, 15), 
                  num_subplots: int = 8) -> tuple:
    
    # Plot num_sensors and 2 more rows (detections) of subplots
    fig, axes = plt.subplots(num_subplots, 1, 
                             figsize=figsize, sharex=True)
    return fig, axes


def plot_imu_rotation(axes: np.ndarray,
                      axis_idx: int,
                      imu_rotation_data: pd.DataFrame,
                      sensor_freq: int,
                      preprocessed: bool = False,
                      force_unit: str = None) -> np.ndarray:
    """
    Plot IMU rotation data (roll, pitch, yaw) on a single subplot.
    
    Args:
        axes: Array of matplotlib axes
        axis_idx: Index of the axis to plot on
        imu_rotation_data: DataFrame containing roll, pitch, yaw columns
        sensor_freq: Sensor frequency in Hz
        force_unit: Force specific time unit
        
    Returns:
        Updated axes array
    """
    if axes is None:
        raise ValueError('axes must be specified')
    
    if not preprocessed:
        imu_rotation_data = trim_data(imu_rotation_data, sensor_freq)
        
    time_axis, unit_label = create_time_axis(
        len(imu_rotation_data), sensor_freq, force_unit
    )
    
    # Plot roll, pitch, yaw with different colors
    axes[axis_idx].plot(time_axis, imu_rotation_data['roll'].values, 
                        color='red', label='Roll', linewidth=1)
    axes[axis_idx].plot(time_axis, imu_rotation_data['pitch'].values, 
                        color='green', label='Pitch', linewidth=1)
    axes[axis_idx].plot(time_axis, imu_rotation_data['yaw'].values, 
                        color='blue', label='Yaw', linewidth=1)
    
    axes[axis_idx].set_ylabel('Rotation (°)')
    axes[axis_idx].set_title('IMU Rotation Data')
    axes[axis_idx].legend(loc='upper right')
    axes[axis_idx].grid(True, alpha=0.3)
    
    return axes


def plot_imu_acceleration(axes: np.ndarray,
                          axis_idx: int,
                          imu_acceleration_data: np.ndarray,
                          sensor_freq: int,
                          preprocessed: bool = False,
                          force_unit: str = None) -> np.ndarray:
    """
    Plot IMU acceleration data.
    
    Args:
        axes: Array of matplotlib axes
        axis_idx: Index of the axis to plot on
        imu_acceleration_data: Array of acceleration values
        sensor_freq: Sensor frequency in Hz
        force_unit: Force specific time unit
        
    Returns:
        Updated axes array
    """
    if axes is None:
        raise ValueError('axes must be specified')
    
    if not preprocessed:
        imu_acceleration_data = trim_data(imu_acceleration_data, sensor_freq)
        
    time_axis, unit_label = create_time_axis(
        len(imu_acceleration_data), sensor_freq, force_unit
    )
    
    axes[axis_idx].plot(time_axis, imu_acceleration_data, 
                        color='purple', linewidth=1)
    axes[axis_idx].set_ylabel('Acceleration (m/s²)')
    axes[axis_idx].set_title('IMU Acceleration Data')
    axes[axis_idx].grid(True, alpha=0.3)
    
    return axes


def plot_sensor_data(axes: np.ndarray,
                     axis_idx: int,
                     sensor_data: Union[List, np.ndarray],
                     sensor_name: str,
                     preprocessed: bool = False,
                     sensor_freq: int = None,
                     force_unit: str = None):
    """
    Plot sensor data with automatic time unit determination.
    
    Args:
        axes: Array of matplotlib axes
        axis_idx: Index of the axis to plot on
        data: Sensor data array
        sensor_name: Name of the sensor for labeling
        sensor_freq: Sensor frequency in Hz (defaults to instance frequency)
        force_unit: Force specific time unit
        
    Returns:
        Updated axes array
    """
    if axes is None:
        raise ValueError('axes must be specified')
    
    if not preprocessed:
        sensor_data = trim_data(sensor_data, sensor_freq)
    
    time_axis, unit_label = create_time_axis(len(sensor_data), sensor_freq, force_unit)
    
    axes[axis_idx].plot(time_axis, sensor_data, label=sensor_name)
    axes[axis_idx].set_ylabel("Signal")
    axes[axis_idx].set_xlabel("")
    axes[axis_idx].legend()
    axes[axis_idx].grid(True, alpha=0.3)

    return axes


def plot_detections(axes: np.ndarray,
                    axis_idx: int,
                    detection_map: Union[List, np.ndarray],
                    det_type: str,
                    xlabel: str,
                    ylabel: str,
                    sensor_freq: int = None,
                    force_unit: str = None):
    """
    Plot detection maps with automatic time unit determination.
    
    Args:
        axes: Array of matplotlib axes
        axis_idx: Index of the axis to plot on
        detection_map: Binary detection map
        det_type: Type of detection for labeling
        xlabel: X-axis label
        ylabel: Y-axis label
        sensor_freq: Sensor frequency in Hz (defaults to instance frequency)
        force_unit: Force specific time unit
        
    Returns:
        Updated axes array
    """
    if axes is None:
        raise ValueError('axes must be specified')
    
    time_axis, unit_label = create_time_axis(len(detection_map), sensor_freq, force_unit)
    
    axes[axis_idx].plot(time_axis, detection_map, label=det_type)
    axes[axis_idx].set_ylabel(ylabel)
    
    # Use provided xlabel or auto-generated unit label
    if xlabel:
        axes[axis_idx].set_xlabel(xlabel)
    else:
        axes[axis_idx].set_xlabel(unit_label)
        
    axes[axis_idx].legend()
    axes[axis_idx].grid(True, alpha=0.3)

    return axes


def plot_sensation_map(axes: np.ndarray,
                       axis_idx: int,
                       sensation_data: Union[List, np.ndarray],
                       sensor_freq: int = None,
                       force_unit: str = None) -> np.ndarray:
    """
    Plot sensation map data with special styling.
    
    Args:
        axes: Array of matplotlib axes
        axis_idx: Index of the axis to plot on
        sensation_data: Sensation map data array
        sensor_freq: Sensor frequency in Hz (defaults to sensation frequency)
        force_unit: Force specific time unit
        
    Returns:
        Updated axes array
    """
    if axes is None:
        raise ValueError('axes must be specified')
    
    time_axis, unit_label = create_time_axis(len(sensation_data), sensor_freq, force_unit)
    
    axes[axis_idx].plot(time_axis, sensation_data, label="Sensation", 
                        color='orange', linewidth=2)
    axes[axis_idx].set_ylabel("Sensation")
    axes[axis_idx].set_xlabel("")
    axes[axis_idx].legend()
    axes[axis_idx].grid(True, alpha=0.3)

    return axes


def save_figure(fig: Figure, filename: str) -> None:
    assert fig is not None, "fig is not specified"
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)