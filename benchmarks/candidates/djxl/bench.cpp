// djxl candidate: decode a JPEG XL file the way the `djxl` CLI does.
//
// The reference `djxl` tool decodes into a PackedPixelFile that preserves the
// codestream's native sample type (e.g. uint16 for a 10-bit image) and all
// channels (color + alpha). We reproduce those output semantics through the
// libjxl C API: native data type via JXL_BIT_DEPTH_FROM_CODESTREAM and the full
// channel count, rather than forcing float32 RGB like the bare `libjxl`
// candidate. (Homebrew's jpeg-xl ships the static extras codec library but not
// its headers, so we cannot link `jxl::extras::DecodeImageJXL` directly; the
// C API path produces the same pixel buffer djxl would write.)
//
// It emits a single JSON line matching the jxlit benchmark schema.

#include <jxl/decode.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/types.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kWarmupDecodes = 3;

struct Args {
  std::string file;
  int iterations = 100;
  int threads = 0;  // 0 == auto
  std::string action = "decode_cpu";
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--file") {
      args.file = next("--file");
    } else if (arg == "--iterations") {
      args.iterations = std::stoi(next("--iterations"));
    } else if (arg == "--threads") {
      args.threads = std::stoi(next("--threads"));
    } else if (arg == "--action") {
      args.action = next("--action");
    } else if (arg == "--layout" || arg == "--hardware" || arg == "--destination") {
      next(arg.c_str());  // accepted for a uniform CLI, not used here
    } else if (arg == "--no-telemetry") {
      // flag with no value; accepted and ignored
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      std::exit(1);
    }
  }
  if (args.file.empty()) {
    std::fprintf(stderr, "--file is required\n");
    std::exit(1);
  }
  if (args.iterations <= 0) {
    std::fprintf(stderr, "--iterations must be greater than 0\n");
    std::exit(1);
  }
  return args;
}

std::vector<uint8_t> read_file(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "failed to open %s\n", path.c_str());
    std::exit(1);
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (size > 0 && std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
    std::fprintf(stderr, "failed to read %s\n", path.c_str());
    std::exit(1);
  }
  std::fclose(f);
  return bytes;
}

// Selects the native libjxl output sample type for the codestream, matching how
// djxl preserves the original precision.
JxlDataType native_data_type(const JxlBasicInfo& info) {
  if (info.exponent_bits_per_sample > 0) return JXL_TYPE_FLOAT;
  if (info.bits_per_sample > 8) return JXL_TYPE_UINT16;
  return JXL_TYPE_UINT8;
}

size_t bytes_per_sample(JxlDataType type) {
  switch (type) {
    case JXL_TYPE_FLOAT:
      return 4;
    case JXL_TYPE_UINT16:
      return 2;
    default:
      return 1;
  }
}

// Decodes the first frame into the codestream's native sample type and channel
// count. Returns false on error.
bool decode_once(const uint8_t* data, size_t size, void* runner,
                 std::vector<uint8_t>& out, uint32_t& width, uint32_t& height,
                 uint32_t& channels) {
  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  if (!dec) return false;

  bool ok = true;
  if (JxlDecoderSetParallelRunner(dec, JxlResizableParallelRunner, runner) !=
          JXL_DEC_SUCCESS ||
      JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
          JXL_DEC_SUCCESS ||
      JxlDecoderSetInput(dec, data, size) != JXL_DEC_SUCCESS) {
    JxlDecoderDestroy(dec);
    return false;
  }
  JxlDecoderCloseInput(dec);

  JxlPixelFormat format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_ERROR) {
      ok = false;
      break;
    }
    if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) {
        ok = false;
        break;
      }
      width = info.xsize;
      height = info.ysize;
      channels = info.num_color_channels + (info.alpha_bits > 0 ? 1 : 0);
      format.num_channels = channels;
      format.data_type = native_data_type(info);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size = 0;
      if (JxlDecoderImageOutBufferSize(dec, &format, &buffer_size) !=
          JXL_DEC_SUCCESS) {
        ok = false;
        break;
      }
      out.resize(buffer_size);
      if (JxlDecoderSetImageOutBuffer(dec, &format, out.data(), buffer_size) !=
          JXL_DEC_SUCCESS) {
        ok = false;
        break;
      }
      // Keep the codestream's native bit depth for integer outputs, as djxl does.
      if (format.data_type != JXL_TYPE_FLOAT) {
        JxlBitDepth bit_depth;
        bit_depth.type = JXL_BIT_DEPTH_FROM_CODESTREAM;
        bit_depth.bits_per_sample = 0;
        bit_depth.exponent_bits_per_sample = 0;
        JxlDecoderSetImageOutBitDepth(dec, &bit_depth);
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      break;  // first frame decoded; that is the image we benchmark
    } else if (status == JXL_DEC_SUCCESS) {
      break;
    } else {
      ok = false;
      break;
    }
  }

  JxlDecoderDestroy(dec);
  return ok;
}

double percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  if (sorted.size() == 1) return sorted[0];
  double rank = (p / 100.0) * (sorted.size() - 1);
  size_t lower = static_cast<size_t>(rank);
  size_t upper = std::min(lower + 1, sorted.size() - 1);
  double weight = rank - lower;
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  std::vector<uint8_t> bytes = read_file(args.file);

  void* runner = JxlResizableParallelRunnerCreate(nullptr);
  if (!runner) {
    std::fprintf(stderr, "failed to create parallel runner\n");
    return 1;
  }
  uint32_t threads = args.threads > 0
                         ? static_cast<uint32_t>(args.threads)
                         : std::max(1u, std::thread::hardware_concurrency());
  JxlResizableParallelRunnerSetThreads(runner, threads);

  std::vector<uint8_t> pixels;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  for (int i = 0; i < kWarmupDecodes; ++i) {
    if (!decode_once(bytes.data(), bytes.size(), runner, pixels, width, height,
                     channels)) {
      std::fprintf(stderr, "warmup decode failed\n");
      JxlResizableParallelRunnerDestroy(runner);
      return 1;
    }
  }

  std::vector<double> latencies_ms;
  latencies_ms.reserve(args.iterations);
  auto decode_start = std::chrono::steady_clock::now();
  for (int i = 0; i < args.iterations; ++i) {
    auto start = std::chrono::steady_clock::now();
    if (!decode_once(bytes.data(), bytes.size(), runner, pixels, width, height,
                     channels)) {
      std::fprintf(stderr, "decode failed\n");
      JxlResizableParallelRunnerDestroy(runner);
      return 1;
    }
    std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    latencies_ms.push_back(elapsed.count());
    volatile uint8_t sink = pixels.empty() ? 0 : pixels[0];
    (void)sink;
  }
  std::chrono::duration<double> decode_seconds =
      std::chrono::steady_clock::now() - decode_start;

  JxlResizableParallelRunnerDestroy(runner);

  std::vector<double> sorted = latencies_ms;
  std::sort(sorted.begin(), sorted.end());
  double mean = 0.0;
  for (double v : latencies_ms) mean += v;
  mean /= latencies_ms.size();
  double megapixels = static_cast<double>(width) * height / 1e6;

  std::printf(
      "{\"decoder\":\"djxl\",\"action\":\"%s\",\"hardware\":\"cpu\","
      "\"destination\":\"cpu\",\"iterations\":%d,\"width\":%u,\"height\":%u,"
      "\"channels\":%u,\"megapixels\":%.6f,\"decode_seconds\":%.6f,"
      "\"latency_ms\":{\"mean\":%.3f,\"p50\":%.3f,\"p95\":%.3f,\"min\":%.3f,"
      "\"max\":%.3f}}\n",
      args.action.c_str(), args.iterations, width, height, channels, megapixels,
      decode_seconds.count(), mean, percentile(sorted, 50.0),
      percentile(sorted, 95.0), sorted.front(), sorted.back());
  return 0;
}
