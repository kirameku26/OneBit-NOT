#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static int load_bit(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open model: " + path);
    char bit = 0;
    if (!file.get(bit) || (bit != 0 && bit != 1) || file.peek() != EOF) {
        throw std::runtime_error("Invalid model: expected exactly one byte (0 or 1)");
    }
    return static_cast<unsigned char>(bit);
}

static int predict(ggml_backend_t backend, int x, int weight_bit) {
    constexpr size_t tensor_count = 16;
    const size_t ctx_bytes = tensor_count * ggml_tensor_overhead() + ggml_graph_overhead();
    std::vector<uint8_t> arena(ctx_bytes);
    ggml_init_params params = {ctx_bytes, arena.data(), true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    // Float32 is the compute representation. Only the learned bit is stored.
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * product = ggml_mul(ctx, input, weight);
    ggml_tensor * sum = ggml_add(ctx, product, bias);
    ggml_tensor * result = ggml_step(ctx, sum);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_backend_t backends[] = {backend};
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!scheduler) { ggml_free(ctx); throw std::runtime_error("scheduler initialization failed"); }

    bool ok = ggml_backend_sched_alloc_graph(scheduler, graph);
    if (!ok) {
        ggml_backend_sched_free(scheduler);
        ggml_free(ctx);
        throw std::runtime_error("graph allocation failed");
    }
    const float x_float = static_cast<float>(x);
    const float w_float = weight_bit == 1 ? -1.0f : 1.0f;
    const float b_float = 0.5f;
    ggml_backend_tensor_set(input, &x_float, 0, sizeof(x_float));
    ggml_backend_tensor_set(weight, &w_float, 0, sizeof(w_float));
    ggml_backend_tensor_set(bias, &b_float, 0, sizeof(b_float));

    const ggml_status status = ggml_backend_sched_graph_compute(scheduler, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(scheduler);
        ggml_free(ctx);
        throw std::runtime_error("graph computation failed");
    }
    float output = 0;
    ggml_backend_tensor_get(result, &output, 0, sizeof(output));
    ggml_backend_sched_free(scheduler);
    ggml_free(ctx);
    return output > 0.5f ? 1 : 0;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2 && argc != 3) {
            std::cerr << "Usage: onebit-not-ggml MODEL.BIN [0|1]\n";
            return 2;
        }
        const int bit = load_bit(argv[1]);
        ggml_backend_load_all();
        ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend) throw std::runtime_error("ggml CPU backend unavailable");
        if (argc == 3) {
            const std::string arg = argv[2];
            if (arg != "0" && arg != "1") {
                ggml_backend_free(backend);
                throw std::runtime_error("Input must be 0 or 1");
            }
            const int x = arg[0] - '0';
            std::cout << x << " -> " << predict(backend, x, bit) << '\n';
        } else {
            for (int x = 0; x <= 1; ++x) {
                std::cout << x << " -> " << predict(backend, x, bit) << '\n';
            }
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
