#ifndef FEMO_PARSER_H
#define FEMO_PARSER_H

#include <vector>
#include <string>
#include <cstdint>

// --- Data Structures Mirroring the File Format ---

// Note: The __attribute__((packed)) or #pragma pack(1) is crucial.
#pragma pack(push, 1)

// 3.1. Sync Header (15 bytes) - Firmware should automatically remove this, but we're still compatible with it just in case.
struct SyncHeader {
    uint16_t signature;           // 0xA55E
    uint32_t total_chunks;
    uint32_t chunks_synced;
    uint32_t upload_version;
    uint8_t  is_synced;
};

// 3.2. File Header (48 bytes)
struct FileHeader {
    uint32_t start_signature;     // 0x48414148 "HAAH"
    uint32_t startTimestamp;
    uint16_t startMs;
    uint32_t finishTimestamp;
    uint16_t finishMs;
    uint16_t piezoFreq;
    uint16_t accelerometerFreq;
    uint16_t IMUFreq;
    uint16_t forceSensorFreq;
    uint32_t lastSyncTimestamp;
    uint16_t lastSyncMs;
    uint32_t bytesWritten;
    char     deviceId[4];
    uint8_t  macAddress[6];
    uint32_t end_signature;       // 0x485A5A48 "HZZH"
};

// 3.3.1. Packet Header (8 bytes)
struct PacketHeader {
    uint16_t signature;           // 0x504B "PK"
    uint8_t  sequence;
    uint32_t timestamp_ms;
    uint8_t  flags;
};

// 3.3.2. Sensor Data (416 bytes)
struct PiezoSample {
    uint16_t piezo1;
    uint16_t piezo2;
};

struct IMUSample {
    int16_t quat_r;
    int16_t quat_i;
    int16_t quat_j;
    int16_t quat_k;
    int16_t mag_x;
    int16_t mag_y;
    int16_t mag_z;
    int16_t accel_x;
    int16_t accel_y;
    int16_t accel_z;
};

struct SensorData {
    PiezoSample piezo_samples[64];
    IMUSample   imu_samples[8];
};

// 3.3. Full Data Packet (424 bytes)
struct DataPacket {
    PacketHeader header;
    SensorData   payload;
};

#pragma pack(pop)


// --- Data Structure for Returning Results to Python ---

struct ParseResult {
    // Metadata from the File Header
    FileHeader file_header;
    bool sync_header_present = false;
    SyncHeader sync_header;

    // Parsed packet data
    std::vector<uint32_t> packet_timestamps;
    std::vector<bool> button_pressed;
    
    // Piezo data will be a flattened 2D array (NumPy will reshape it)
    std::vector<uint16_t> piezo1_data;
    std::vector<uint16_t> piezo2_data;

    // IMU data
    std::vector<int16_t> quat_r;
    std::vector<int16_t> quat_i;
    std::vector<int16_t> quat_j;
    std::vector<int16_t> quat_k;

    std::vector<int16_t> mag_x;
    std::vector<int16_t> mag_y;
    std::vector<int16_t> mag_z;
    
    std::vector<int16_t> accel_x;
    std::vector<int16_t> accel_y;
    std::vector<int16_t> accel_z;

    // Diagnostics
    long long total_bytes_read = 0;
    int packets_found = 0;
    int corrupted_packets_skipped = 0;
};


// --- Main Parser Function Declaration ---

ParseResult parse_femo_file(
    const std::string& path,
    long long start_byte = -1,
    long long end_byte = -1,
    int max_packets = -1
);

#endif // FEMO_PARSER_H
