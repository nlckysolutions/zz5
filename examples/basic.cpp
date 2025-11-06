#include "zz5.hpp"
#include <iostream>

int main() {
    zz5::Config cfg;
    cfg.prefix = "nlckysolutions_";
    cfg.blocks = 8192;
    cfg.threads = 256;
    cfg.tries_per_thread = 4096;
    cfg.max_seconds = 5.0; // stop after 5 sec if not found

    auto res = zz5::find("deadbeef", /*suffixLen*/8, cfg);

    std::cout << "found=" << res.found
              << " tries=" << res.tries
              << " speed=" << (uint64_t)res.speed_hps << " H/s\n";
    if (res.found) {
        std::cout << "text=" << res.text << "\n";
        std::cout << "md5=" << res.digest_hex << "\n";
    }
}
