#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // Needed for automatic conversion of std::vector, etc.
#include <pybind11/numpy.h>    // Needed for interfacing with NumPy

#include "parser.h"

namespace py = pybind11;

// The PYBIND11_MODULE macro creates a function that will be called when
// the Python interpreter imports the module. The module name (femo_parser_cpp)
// is passed as the first argument.
PYBIND11_MODULE(femo_parser_cpp, m) {
    m.doc() = "High-performance C++ parser for FeMo sensor data files"; // Optional module docstring

    // --- Bind C++ Structs to Python Classes ---

    py::class_<SyncHeader>(m, "SyncHeader")
        .def_readonly("signature", &SyncHeader::signature)
        .def_readonly("total_chunks", &SyncHeader::total_chunks)
        .def_readonly("chunks_synced", &SyncHeader::chunks_synced)
        .def_readonly("upload_version", &SyncHeader::upload_version)
        .def_readonly("is_synced", &SyncHeader::is_synced)
        // Add a __repr__ for nice printing in Python
        .def("__repr__",
             [](const SyncHeader &sh) {
                 return "<SyncHeader signature=" + std::to_string(sh.signature) + ">";
             });

    py::class_<FileHeader>(m, "FileHeader")
        .def_readonly("start_signature", &FileHeader::start_signature)
        .def_readonly("startTimestamp", &FileHeader::startTimestamp)
        .def_readonly("startMs", &FileHeader::startMs)
        .def_readonly("finishTimestamp", &FileHeader::finishTimestamp)
        .def_readonly("finishMs", &FileHeader::finishMs)
        .def_readonly("piezoFreq", &FileHeader::piezoFreq)
        .def_readonly("accelerometerFreq", &FileHeader::accelerometerFreq)
        .def_readonly("IMUFreq", &FileHeader::IMUFreq)
        .def_readonly("forceSensorFreq", &FileHeader::forceSensorFreq)
        .def_readonly("lastSyncTimestamp", &FileHeader::lastSyncTimestamp)
        .def_readonly("lastSyncMs", &FileHeader::lastSyncMs)
        .def_readonly("bytesWritten", &FileHeader::bytesWritten)
        .def_readonly("deviceId", &FileHeader::deviceId)
        // .def_readonly("macAddress", &FileHeader::macAddress)
        .def_property_readonly("macAddress", [](const FileHeader &fh) {
                return py::bytes(reinterpret_cast<const char*>(fh.macAddress), 6);
            })
        .def_readonly("end_signature", &FileHeader::end_signature)
        .def("__repr__",
             [](const FileHeader &fh) {
                 return "<FileHeader deviceId='" + std::string(fh.deviceId, 4) + "'>";
             });
             
    // This is the main result object that our function will return.
    // We bind it and all its fields so we can access them in Python.
    py::class_<ParseResult>(m, "ParseResult")
        .def_readonly("file_header", &ParseResult::file_header)
        .def_readonly("sync_header_present", &ParseResult::sync_header_present)
        .def_readonly("sync_header", &ParseResult::sync_header)
        .def_readonly("total_bytes_read", &ParseResult::total_bytes_read)
        .def_readonly("packets_found", &ParseResult::packets_found)
        .def_readonly("corrupted_packets_skipped", &ParseResult::corrupted_packets_skipped)
        // For the vector members, we expose them as NumPy arrays.
        // This is the key for performance: pybind11 can create a NumPy array
        // that wraps the C++ vector's data without copying it.
        .def_property_readonly("packet_timestamps", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.packet_timestamps));
        })
        .def_property_readonly("button_pressed", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.button_pressed));
        })
        .def_property_readonly("piezo1_data", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.piezo1_data));
        })
        .def_property_readonly("piezo2_data", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.piezo2_data));
        })
        // ... and so on for all other data vectors ...
        .def_property_readonly("quat_r", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.quat_r));
        })
        .def_property_readonly("quat_i", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.quat_i));
        })
        .def_property_readonly("quat_j", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.quat_j));
        })
        .def_property_readonly("quat_k", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.quat_k));
        })
        .def_property_readonly("mag_x", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.mag_x));
        })
        .def_property_readonly("mag_y", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.mag_y));
        })
        .def_property_readonly("mag_z", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.mag_z));
        })
        .def_property_readonly("accel_x", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.accel_x));
        })
        .def_property_readonly("accel_y", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.accel_y));
        })
        .def_property_readonly("accel_z", [](py::object &obj) {
            ParseResult &r = obj.cast<ParseResult&>();
            return py::array(py::cast(r.accel_z));
        });


    // --- Bind the Main Function ---
    // This exposes our C++ parse_femo_file function to Python.
    m.def("parse_femo_file", &parse_femo_file, "The main file parsing function.",
          // Use py::arg to define the names of the arguments as they will appear in Python.
          // This also allows us to set default values.
          py::arg("path"),
          py::arg("start_byte") = -1,
          py::arg("end_byte") = -1,
          py::arg("max_packets") = -1
    );
}

