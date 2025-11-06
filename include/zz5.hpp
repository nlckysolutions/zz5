#pragma once
#include <cstdint>
#include <string>

namespace zz5 {

struct Result {
    bool        found = false;
    std::string text;
    std::string digest_hex;
    uint64_t    tries = 0;
    double      seconds = 0.0;
    double      speed_hps = 0.0;
};

struct Config {
    std::string prefix = "";
    int         blocks = 8192;
    int         threads = 256;
    uint64_t    tries_per_thread = 4096;
    double      max_seconds = 0.0;
};

Result find(const std::string& targetHex, int suffixLen, const Config& cfg);

}
