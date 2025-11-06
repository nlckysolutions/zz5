// zz5.cu — library build of zz5 — MD5 vanity hash finder (FAST; counter-based; fixed length; compile-time prefix)
// Copyleft NlckySolutions - Nov. 2025 - GPLv3

#include "zz5.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <stdexcept>

// ==================== Tuning ====================
static constexpr int MAX_SUFFIX_LEN = 40;

// --------------------------------- Utilities ---------------------------------
#define CHECK_CUDA(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e)); }}while(0)

static inline uint32_t hex_nibble(char c){
    if (c>='0'&&c<='9') return (uint32_t)(c - '0');
    if (c>='a'&&c<='f') return 10u + (uint32_t)(c - 'a');
    if (c>='A'&&c<='F') return 10u + (uint32_t)(c - 'A');
    return 0u;
}

// --------------------------------- MD5 consts --------------------------------
__constant__ uint32_t c_MD5_K[64] = {
  0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,
  0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
  0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,
  0x6b901122,0xfd987193,0xa679438e,0x49b40821,
  0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,
  0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
  0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,
  0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
  0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,
  0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
  0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,
  0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
  0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,
  0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
  0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,
  0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

__constant__ uint32_t c_MD5_S[64] = {
  7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
  5,9,14,20,5,9,14,20,5,9,14,20,
  4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
  6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
};

// ----------------------- Device constants (runtime) ----------------------
__constant__ char     c_prefix[64];
__constant__ int      c_prefixLen;
__constant__ uint32_t c_baseW[16];
__constant__ int      c_sufWIdx[MAX_SUFFIX_LEN];
__constant__ uint8_t  c_sufShift[MAX_SUFFIX_LEN];
__constant__ int      c_suffixLen;
__constant__ int      c_totalLen;
__constant__ uint8_t  c_strideDigits[MAX_SUFFIX_LEN];

// --------------------------------- Device state --------------------------------
struct DevResult {
  int found;
  int len;
  char s[64];
  uint8_t digest[16];
};
__device__ DevResult dev_result;
__device__ unsigned long long dev_hashes = 0ULL;

// --------------------------------- Intrinsics ---------------------------------
__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t r){
    uint32_t y;
    asm volatile ("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(y) : "r"(x), "r"(r));
    return y;
}

// --------------------------------- MD5 core -----------------------------------
__device__ __forceinline__ void md5_from_words(const uint32_t w[16], uint8_t out[16]) {
    uint32_t a = 0x67452301u, b = 0xEFCDAB89u, c = 0x98BADCFEu, d = 0x10325476u;

    #pragma unroll
    for (int i=0;i<64;i++){
        uint32_t F, g;
        if (i < 16)      { F = (b & c) | ((~b) & d);           g = i; }
        else if (i < 32) { F = (d & b) | ((~d) & c);           g = (5*i + 1) & 15; }
        else if (i < 48) { F = b ^ c ^ d;                      g = (3*i + 5) & 15; }
        else             { F = c ^ (b | (~d));                 g = (7*i) & 15; }

        uint32_t tmp = d;
        d = c; c = b;
        uint32_t add = a + F + c_MD5_K[i] + w[g];
        b = b + rotl32(add, c_MD5_S[i]);
        a = tmp;
    }

    a += 0x67452301u; b += 0xEFCDAB89u; c += 0x98BADCFEu; d += 0x10325476u;

    uint32_t outs[4] = { a,b,c,d };
    #pragma unroll
    for (int j=0;j<4;j++){
        out[j*4+0] = (uint8_t)(outs[j] & 0xFF);
        out[j*4+1] = (uint8_t)((outs[j]>>8) & 0xFF);
        out[j*4+2] = (uint8_t)((outs[j]>>16)& 0xFF);
        out[j*4+3] = (uint8_t)((outs[j]>>24)& 0xFF);
    }
}

// ------------------------- Base-26 helpers ----------------------
__device__ __forceinline__ void add_stride_base26(uint8_t *digits, int S){
    uint32_t carry = 0;
    #pragma unroll
    for (int i=0;i<MAX_SUFFIX_LEN;i++){
        if (i >= S) break;
        uint32_t sum = (uint32_t)digits[i] + (uint32_t)c_strideDigits[i] + carry;
        digits[i] = (uint8_t)(sum % 26u);
        carry     = (sum / 26u);
    }
}
__device__ __forceinline__ void index_to_base26(uint64_t idx, uint8_t *digits, int S){
    #pragma unroll
    for (int i=0;i<MAX_SUFFIX_LEN;i++){
        if (i >= S) break;
        digits[i] = (uint8_t)(idx % 26u);
        idx /= 26u;
    }
}

// --------------------------------- Kernel -------------------------------------
__global__ __launch_bounds__(256,2)
void md5_vanity_kernel(const uint8_t* __restrict__ d_target,
                       int match_len, uint8_t last_mask,
                       uint64_t tries_per_thread,
                       unsigned long long counter_base)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t digits[MAX_SUFFIX_LEN];
    index_to_base26(counter_base + (unsigned long long)tid, digits, c_suffixLen);

    unsigned long long local_count = 0ULL;

    for (uint64_t it = 0; it < tries_per_thread; ++it){
        if ((it & 0xFFu) == 0u && dev_result.found) break;

        uint32_t w[16];
        #pragma unroll
        for (int i=0;i<16;i++) w[i] = c_baseW[i];

        #pragma unroll 8
        for (int j=0;j<MAX_SUFFIX_LEN;j++){
            if (j >= c_suffixLen) break;
            const uint32_t ch = (uint32_t)('a' + digits[j]);
            const int wi = c_sufWIdx[j];
            const uint8_t sh = c_sufShift[j];
            w[wi] |= (ch << sh);
        }

        uint8_t dg[16];
        md5_from_words(w, dg);
        local_count++;

        bool ok = true;
        if (match_len > 0) {
            const int last = match_len - 1;
            #pragma unroll
            for (int i=0;i<16;i++){
                if (i >= last) break;
                if (dg[i] != d_target[i]) { ok = false; break; }
            }
            if (ok) {
                if ( ((uint8_t)(dg[last] & last_mask)) != d_target[last]) ok = false;
            }
        }

        if (ok){
            if (atomicCAS(&dev_result.found, 0, 1) == 0){
                const int L = c_totalLen;
                #pragma unroll
                for (int i=0;i<64;i++){ if (i>=c_prefixLen) break; dev_result.s[i] = c_prefix[i]; }
                #pragma unroll
                for (int j=0;j<MAX_SUFFIX_LEN;j++){
                    if (j >= c_suffixLen) break;
                    dev_result.s[c_prefixLen + j] = (char)('a' + digits[j]);
                }
                dev_result.s[L] = 0;
                dev_result.len = L;
                #pragma unroll
                for (int k=0;k<16;k++) dev_result.digest[k] = dg[k];
            }
            break;
        }

        add_stride_base26(digits, c_suffixLen);
    }

    // Warp-then-block reduction into dev_hashes
    unsigned long long v = local_count;
    for (int off=16; off>0; off>>=1)
        v += __shfl_down_sync(0xffffffff, v, off);

    __shared__ unsigned long long warpSums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();
    if (warp == 0){
        unsigned long long wv = (lane < (blockDim.x + 31) / 32) ? warpSums[lane] : 0ULL;
        for (int off=16; off>0; off>>=1)
            wv += __shfl_down_sync(0xffffffff, wv, off);
        if (lane == 0) atomicAdd(&dev_hashes, wv);
    }
}

// --------------------------------- Host helpers --------------------------------
static void build_baseW_and_positions(const char* prefixStr, int prefixLen, int suffixLen,
                                      uint32_t baseW[16],
                                      int sufWIdx[MAX_SUFFIX_LEN],
                                      uint8_t sufShift[MAX_SUFFIX_LEN],
                                      int &totalLen)
{
    for (int i=0;i<16;i++) baseW[i] = 0u;

    // pack prefix bytes
    for (int i=0;i<prefixLen;i++){
        int wi = i >> 2;
        int sh = (i & 3) * 8;
        baseW[wi] |= ((uint32_t)(uint8_t)prefixStr[i]) << sh;
    }

    // suffix byte positions
    for (int j=0;j<suffixLen;j++){
        int idx = prefixLen + j;
        int wi  = idx >> 2;
        int sh  = (idx & 3) * 8;
        sufWIdx[j]  = wi;
        sufShift[j] = (uint8_t)sh;
    }

    totalLen = prefixLen + suffixLen;

    // 0x80 at byte totalLen
    {
        int i = totalLen;
        int wi = i >> 2;
        int sh = (i & 3) * 8;
        baseW[wi] |= ((uint32_t)0x80) << sh;
    }

    // bit length in w[14..15]
    const uint64_t bitlen = (uint64_t)totalLen * 8ULL;
    baseW[14] = (uint32_t)(bitlen & 0xFFFFFFFFu);
    baseW[15] = (uint32_t)(bitlen >> 32);
}

static inline std::string hex_of_digest(const uint8_t dg[16]){
    static const char* h="0123456789abcdef";
    std::string s; s.resize(32);
    for (int i=0;i<16;i++){ s[2*i]=h[(dg[i]>>4)&0xF]; s[2*i+1]=h[dg[i]&0xF]; }
    return s;
}

// --------------------------------- Public API ---------------------------------
namespace zz5 {

Result find(const std::string& targetHex, int suffixLen, const Config& cfg)
{
    if (suffixLen <= 0 || suffixLen > MAX_SUFFIX_LEN)
        throw std::invalid_argument("suffixLen must be 1..40");
    if ((int)cfg.prefix.size() + suffixLen > 55)
        throw std::invalid_argument("prefix+suffix must be <= 55 bytes (single MD5 block)");
    if (cfg.blocks <= 0 || cfg.threads <= 0)
        throw std::invalid_argument("blocks/threads must be > 0");

    // Parse target hex (up to 32 hex chars -> 16 bytes)
    int hexlen = (int)targetHex.size();
    if (hexlen > 32) hexlen = 32;
    const int full_bytes = hexlen / 2;
    const int has_half   = hexlen & 1;
    const int match_len  = full_bytes + has_half;

    uint8_t target_bytes[16] = {0};
    for (int i=0;i<full_bytes;i++){
        uint8_t hi = (uint8_t)hex_nibble(targetHex[2*i]);
        uint8_t lo = (uint8_t)hex_nibble(targetHex[2*i+1]);
        target_bytes[i] = (uint8_t)((hi<<4)|lo);
    }
    uint8_t last_mask = 0xFF;
    if (has_half){
        uint8_t hi = (uint8_t)hex_nibble(targetHex[2*full_bytes]);
        target_bytes[full_bytes] = (uint8_t)(hi<<4);
        last_mask = 0xF0;
    }

    // Build constants
    const int prefixLenHost = (int)cfg.prefix.size();
    uint32_t baseW[16];
    int      sufWIdx[MAX_SUFFIX_LEN];
    uint8_t  sufShift[MAX_SUFFIX_LEN];
    int      totalLen = 0;
    build_baseW_and_positions(cfg.prefix.c_str(), prefixLenHost, suffixLen,
                              baseW, sufWIdx, sufShift, totalLen);

    // Push to device
    CHECK_CUDA(cudaMemcpyToSymbol(c_prefix,     cfg.prefix.data(),            prefixLenHost));
    CHECK_CUDA(cudaMemcpyToSymbol(c_prefixLen,  &prefixLenHost,               sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_baseW,      baseW,                        sizeof(baseW)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_sufWIdx,    sufWIdx,                      sizeof(int)*MAX_SUFFIX_LEN));
    CHECK_CUDA(cudaMemcpyToSymbol(c_sufShift,   sufShift,                     sizeof(uint8_t)*MAX_SUFFIX_LEN));
    CHECK_CUDA(cudaMemcpyToSymbol(c_suffixLen,  &suffixLen,                   sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(c_totalLen,   &totalLen,                    sizeof(int)));

    uint8_t* d_target = nullptr;
    if (match_len){
        CHECK_CUDA(cudaMalloc(&d_target, match_len));
        CHECK_CUDA(cudaMemcpy(d_target, target_bytes, match_len, cudaMemcpyHostToDevice));
    } else {
        CHECK_CUDA(cudaMalloc(&d_target, 1));
    }

    DevResult zero = {};
    unsigned long long z = 0ULL;
    CHECK_CUDA(cudaMemcpyToSymbol(dev_result, &zero, sizeof(DevResult)));
    CHECK_CUDA(cudaMemcpyToSymbol(dev_hashes, &z,    sizeof(unsigned long long)));

    // Precompute stride digits in base-26: stride = blocks*threads
    const unsigned long long grid_size =
        (unsigned long long)cfg.blocks * (unsigned long long)cfg.threads;
    uint8_t h_strideDigits[MAX_SUFFIX_LEN]; for (int i=0;i<MAX_SUFFIX_LEN;i++) h_strideDigits[i]=0;
    {
        unsigned long long tmp = grid_size;
        for (int i=0;i<suffixLen;i++){
            h_strideDigits[i] = (uint8_t)(tmp % 26u);
            tmp /= 26u;
            if (tmp==0ULL) break;
        }
    }
    CHECK_CUDA(cudaMemcpyToSymbol(c_strideDigits, h_strideDigits, sizeof(uint8_t)*MAX_SUFFIX_LEN));

    auto t0 = std::chrono::high_resolution_clock::now();
    //auto last = t0;
    unsigned long long counter_base = 0ULL;

    Result out;
    while (true){
        md5_vanity_kernel<<<cfg.blocks, cfg.threads>>>(
            d_target, match_len, last_mask,
            cfg.tries_per_thread,
            counter_base
        );
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        counter_base += grid_size * cfg.tries_per_thread;

        unsigned long long total = 0ULL;
        CHECK_CUDA(cudaMemcpyFromSymbol(&total, dev_hashes, sizeof(unsigned long long)));

        DevResult host_res;
        CHECK_CUDA(cudaMemcpyFromSymbol(&host_res, dev_result, sizeof(DevResult), 0, cudaMemcpyDeviceToHost));

        auto now = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(now - t0).count();

        if (host_res.found){
            out.found      = true;
            out.text       = std::string(host_res.s, host_res.len);
            out.digest_hex = hex_of_digest(host_res.digest);
            out.tries      = total;
            out.seconds    = secs;
            out.speed_hps  = (secs>0)? (double)total / secs : 0.0;
            break;
        }

        if (cfg.max_seconds > 0.0 && secs >= cfg.max_seconds){
            out.found      = false;
            out.text.clear();
            out.digest_hex.clear();
            out.tries      = total;
            out.seconds    = secs;
            out.speed_hps  = (secs>0)? (double)total / secs : 0.0;
            break;
        }
    }

    cudaFree(d_target);
    return out;
}

} // namespace zz5
