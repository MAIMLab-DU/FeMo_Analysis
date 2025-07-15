#include "parser.h"
#include <fstream>
#include <stdexcept>
#include <iostream> // For potential debugging

// Implementation of the main parser function
ParseResult parse_femo_file(const std::string& path, long long start_byte, long long end_byte, int max_packets) {
    
    // 1. --- INITIALIZATION ---
    // Create a result object to store all our parsed data.
    ParseResult result;

    // Open the file for reading in binary mode.
    std::ifstream file(path, std::ios::binary | std::ios::ate); // ate: open at end to get size
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open file " + path);
    }

    // Get the total size of the file.
    long long file_size = file.tellg();
    file.seekg(0, std::ios::beg); // Go back to the beginning

    // 2. --- HEADER PARSING ---
    // The file must contain at least a FileHeader to be valid.
    if (file_size < sizeof(FileHeader)) {
        throw std::runtime_error("Error: File is too small to be a valid FeMo data file.");
    }

    // Check for Sync Header (0xA55E)
    uint16_t initial_signature;
    file.read(reinterpret_cast<char*>(&initial_signature), sizeof(initial_signature));
    if (initial_signature == 0xA55E) {
        result.sync_header_present = true;
        file.seekg(0, std::ios::beg); // Go back to read the full header
        file.read(reinterpret_cast<char*>(&result.sync_header), sizeof(SyncHeader));
    } else {
        // No sync header, rewind to the beginning to read the file header.
        file.seekg(0, std::ios::beg);
    }
    
    // At this point, the file pointer is at the start of the FileHeader.
    long long file_header_start_pos = file.tellg();
    file.read(reinterpret_cast<char*>(&result.file_header), sizeof(FileHeader));

    // CRITICAL VALIDATION: Check the FileHeader's signatures. If these are wrong,
    // the file is fundamentally corrupt or not the right format.
    if (result.file_header.start_signature != 0x48414148 /* HAAH */ || result.file_header.end_signature != 0x485A5A48 /* HZZH */) {
        throw std::runtime_error("Error: Invalid FileHeader signature. This may not be a FeMo data file.");
    }

    // 3. --- DATA PACKET PARSING LOOP ---
    
    // Determine the byte range to parse for data packets.
    // The data section starts right after the file header.
    long long loop_start_pos = file_header_start_pos + sizeof(FileHeader);
    
    // If user provided a start_byte, use it, but ensure it's after the header.
    if (start_byte > loop_start_pos) {
        loop_start_pos = start_byte;
    }

    // The loop should end at the user-specified end_byte or the end of the file.
    long long loop_end_pos = file_size;
    if (end_byte > 0 && end_byte < file_size) {
        loop_end_pos = end_byte;
    }

    // Seek to the determined starting position for the data packet loop.
    file.seekg(loop_start_pos, std::ios::beg);

    DataPacket current_packet;
    
    while (file.tellg() < loop_end_pos && (max_packets == -1 || result.packets_found < max_packets)) {
        
        long long current_pos = file.tellg();

        // Ensure we have enough bytes left for at least a packet header.
        if (current_pos + sizeof(PacketHeader) > loop_end_pos) {
            break; 
        }

        // Read the packet signature to validate.
        uint16_t packet_signature;
        file.read(reinterpret_cast<char*>(&packet_signature), sizeof(packet_signature));

        if (packet_signature == 0x504B /* PK */) {
            // SIGNATURE MATCHED: We found a valid packet.
            
            // Check if there's enough space for the rest of the packet payload.
            if (current_pos + sizeof(DataPacket) > loop_end_pos) {
                // Not enough bytes left for a full packet, so we stop.
                break;
            }

            // Seek back to the beginning of the signature and read the whole packet.
            file.seekg(current_pos);
            file.read(reinterpret_cast<char*>(&current_packet), sizeof(DataPacket));
            
            result.packets_found++;
            
            // --- Extract data and push it into our result vectors ---
            result.packet_timestamps.push_back(current_packet.header.timestamp_ms);
            result.button_pressed.push_back(current_packet.header.flags & 0x01);

            // Extract Piezo data (64 samples per packet)
            for (int i = 0; i < 64; ++i) {
                result.piezo1_data.push_back(current_packet.payload.piezo_samples[i].piezo1);
                result.piezo2_data.push_back(current_packet.payload.piezo_samples[i].piezo2);
            }

            // Extract IMU data (8 samples per packet)
            for (int i = 0; i < 8; ++i) {
                const auto& imu = current_packet.payload.imu_samples[i];
                result.quat_r.push_back(imu.quat_r);
                result.quat_i.push_back(imu.quat_i);
                result.quat_j.push_back(imu.quat_j);
                result.quat_k.push_back(imu.quat_k);
                result.mag_x.push_back(imu.mag_x);
                result.mag_y.push_back(imu.mag_y);
                result.mag_z.push_back(imu.mag_z);
                result.accel_x.push_back(imu.accel_x);
                result.accel_y.push_back(imu.accel_y);
                result.accel_z.push_back(imu.accel_z);
            }

        } else {
            // SIGNATURE MISMATCH: Corruption detected.
            result.corrupted_packets_skipped++;
            
            // The file pointer is already advanced by 2 bytes from our signature read.
            // We don't need to do anything else. The loop will continue, effectively
            // scanning from the next byte for a valid 'PK' signature.
            // For robustness, we'll manually seek forward 1 byte from the start of the bad read attempt.
            file.seekg(current_pos + 1);
        }
    }
    
    // 4. --- FINALIZATION ---
    result.total_bytes_read = file.tellg();
    file.close();

    return result;
}
