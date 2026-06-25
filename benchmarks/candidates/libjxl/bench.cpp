// libjxl candidate: decode a JPEG XL file with the raw libjxl C API.
//
// This is the fastest libjxl library path: it talks directly to `JxlDecoder`
// with a resizable thread-pool runner and asks for interleaved float32 RGB,
// which is what an ML / GPU-upload pipeline typically wants. It emits a single
// JSON line matching the jxlit benchmark schema so the compare orchestrator can
// aggregate it alongside every other candidate.

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

// Tolerant parser: understands the flags this candidate needs and accepts (then
// ignores) the rest of the shared benchmark flags so every candidate can be
// launched with an identical argument list.
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

// Decodes the first frame to interleaved float32 RGB. Returns false on error.
bool decode_once(const uint8_t* data, size_t size, void* runner,
                 std::vector<float>& out, uint32_t& width, uint32_t& height) {
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

  const JxlPixelFormat format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
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
      out.resize(static_cast<size_t>(width) * height * 3);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size = 0;
      if (JxlDecoderImageOutBufferSize(dec, &format, &buffer_size) !=
          JXL_DEC_SUCCESS) {
        ok = false;
        break;
      }
      out.resize(buffer_size / sizeof(float));
      if (JxlDecoderSetImageOutBuffer(dec, &format, out.data(), buffer_size) !=
          JXL_DEC_SUCCESS) {
        ok = false;
        break;
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

  std::vector<float> pixels;
  uint32_t width = 0;
  uint32_t height = 0;
  for (int i = 0; i < kWarmupDecodes; ++i) {
    if (!decode_once(bytes.data(), bytes.size(), runner, pixels, width, height)) {
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
    if (!decode_once(bytes.data(), bytes.size(), runner, pixels, width, height)) {
      std::fprintf(stderr, "decode failed\n");
      JxlResizableParallelRunnerDestroy(runner);
      return 1;
    }
    std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    latencies_ms.push_back(elapsed.count());
    // Keep the decoded buffer observable so the decode is not elided.
    volatile float sink = pixels.empty() ? 0.0f : pixels[0];
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
      "{\"decoder\":\"libjxl\",\"action\":\"%s\",\"hardware\":\"cpu\","
      "\"destination\":\"cpu\",\"iterations\":%d,\"width\":%u,\"height\":%u,"
      "\"channels\":3,\"megapixels\":%.6f,\"decode_seconds\":%.6f,"
      "\"latency_ms\":{\"mean\":%.3f,\"p50\":%.3f,\"p95\":%.3f,\"min\":%.3f,"
      "\"max\":%.3f}}\n",
      args.action.c_str(), args.iterations, width, height, megapixels,
      decode_seconds.count(), mean, percentile(sorted, 50.0),
      percentile(sorted, 95.0), sorted.front(), sorted.back());
  return 0;
}
