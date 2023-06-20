#include "ggml-sycl.h"

#include <cstdio>
#include <algorithm>
#include <cassert>
#include <atomic>
#include <mutex>
#include <memory>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

static auto get_device_selector(std::string deviceName) {
  using Sel = sycl::ext::oneapi::filter_selector;
  return [selector = Sel(std::move(deviceName))](
             const sycl::device &dev) -> int { return selector(dev); };
}

template <typename F> static auto catch_all(F &&func) {
  try {
    return func();
  } catch (const std::exception &e) {
    fprintf(stdout, "An exception was thrown: %s\n", e.what());
    fflush(stdout);
    abort();
  } catch (...) {
    fprintf(stdout, "An unknown exception was thrown\n");
    fflush(stdout);
    abort();
  }
}

static size_t next_pow2(size_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if (sizeof(v) > 4) {
        v |= v >> 32;
    }
    v++;
    return v;
}

static void* check_ptr(void* ptr, size_t size, size_t align, const char* type) {
    if (!ptr) {
        fprintf(stderr, "SYCL: Failed to allocate %s memory: %zu %zu\n",
                type, size, align);
        abort();
    }
    return ptr;
}

static void* alloc_shared(sycl::queue& queue, size_t size, size_t align) {
    size = std::max(std::max(size, align), (size_t) 1);

    void* mem;
    if (align == 0) {
        mem = sycl::malloc_shared(size, queue);
    } else {
        mem = sycl::aligned_alloc_shared(align, size, queue);
    }

    return check_ptr(mem, size, align, "shared");
}

static void* alloc_device(sycl::queue& queue, size_t size, size_t align) {
    size = std::max(std::max(size, align), (size_t) 1);

    void* mem;
    if (align == 0) {
        mem = sycl::malloc_device(size, queue);
    } else {
        mem = sycl::aligned_alloc_device(align, size, queue);
    }

    return check_ptr(mem, size, align, "device");
}


namespace {
struct local_context;
struct global_context {
    global_context() {
        // TODO: Unhardcode deivce
        device = sycl::device{get_device_selector("level_zero:gpu:0")};
        queue = sycl::queue{device};
    }

    local_context* get_context();
    void return_context(local_context* ctx);

    sycl::device device;
    sycl::queue queue;
    std::atomic<int> requestsInFlight{0};
    std::mutex lock;
    std::unique_ptr<local_context> lctx;
};

struct local_context {
    local_context(global_context& ctx) {
        queue = sycl::queue{ctx.queue.get_context(), ctx.device};
    }
    ~local_context() {
        free_scratch();
    }

    void* get_scratch(size_t size) {
        if (size <= scratchSize)
            return scratch;

        size = next_pow2(size);
        free_scratch();
        scratch = alloc_device(queue, size, 0);
        scratchSize = size;
        return scratch;
    }

    void free_scratch() {
        if (scratch) {
            sycl::free(scratch, queue);
            scratch = nullptr;
        }
    }

    sycl::queue queue;
    std::vector<sycl::event> deps;
    void* scratch = nullptr;
    size_t scratchSize = 0;
    std::unique_ptr<local_context> next;
};

local_context* global_context::get_context() {
    std::unique_lock<std::mutex> l(lock); // TODO: atomics?
    if (!lctx)
        return new local_context(*this);

    auto ret = lctx.release();
    lctx = std::move(ret->next);
    return ret;
}

void global_context::return_context(local_context* ctx) {
    assert(ctx && "Invalid local context");
    assert(!ctx.next && "Invalid local context next");
    ctx->deps.clear();
    std::unique_lock<std::mutex> l(lock); // TODO: atomics?
    ctx->next = std::move(lctx);
    lctx.reset(ctx);
}

struct local_context_guard {
    local_context_guard(global_context& c): ctx(c) {
        ++ctx.requestsInFlight;
        lctx = ctx.get_context();
        assert(lctx);
    }
    ~local_context_guard() {
        assert(lctx);
        ctx.return_context(lctx);
        --ctx.requestsInFlight;
    }

    global_context& ctx;
    local_context* lctx = nullptr;
};
}

static global_context* g_context = nullptr;
static global_context& get_context() {
    assert(g_context && "Context is not initialized");
    return *g_context;
}

extern "C" void ggml_sycl_init() {
    assert(!g_context && "Context is already initialized");
    catch_all([&]() {
        g_context = new global_context;
    });
}

static bool matmul_f16_f32_f32(global_context& ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, void * wdata, size_t wsize);

extern "C" bool ggml_sycl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize) {
    return catch_all([&]() {
        auto checkTypes = [&](ggml_type a, ggml_type b, ggml_type c) {
            return src0->type == a && src1->type == b && dst->type == c;
        };
#define T(a) GGML_TYPE_##a
    if (checkTypes(T(F16), T(F32), T(F32))) return matmul_f16_f32_f32(get_context(), src0, src1, dst, wdata, wsize);
#undef T
        return false;
    });
}

extern "C" void* ggml_sycl_alloc_shared(size_t size, size_t align) {
    return alloc_shared(get_context().queue, size, align);
}

extern "C" void ggml_sycl_free(void* ptr) {
    if (ptr) {
        sycl::free(ptr, get_context().queue);
    }
}

extern "C" void ggml_sycl_transform_tensor(void * data, struct ggml_tensor * tensor) {
    // printf("ggml_sycl_transform_tensor %d\n", (int)(ggml_is_contiguous(tensor)));
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne3 = tensor->ne[3];

    const ggml_type type = tensor->type;
    const size_t size = ggml_type_size(type) * ne0 * ne1 * ne2 * ne3 / ggml_blck_size(type);

    auto &ctx = get_context();
    return catch_all([&]() {
        auto mem = sycl::malloc_shared(size, ctx.queue);
        if (!mem) {
            fprintf(stderr, "SYCL: Failed to allocate shared memory\n");
            abort();
        }
        ctx.queue.memcpy(mem, data, size).wait();
        tensor->data = mem;
    });

}

template<typename Src, typename Dst>
static sycl::event convert_type_2d(
        sycl::queue queue, const void* src, void* dst, int64_t ne00,
        int64_t ne01, int64_t nb00, int64_t nb01) {
    auto dst_typed = static_cast<Dst*>(dst);
    return queue.submit([&](sycl::handler& h) {
        sycl::range<2> r{ne00, ne01};
        h.parallel_for(r, [=](sycl::item<2> idx) {
            auto i00 = idx.get_id(0);
            auto i01 = idx.get_id(1);
            auto dst_id = i00 + i01 * ne00;
            dst_typed[dst_id] = (Dst)*(Src*) ((const char *) src + i01*nb01 + i00*nb00);
        });
    });
}


static bool check_strides(const ggml_tensor * tensor) {
    return tensor->nb[0] == ggml_type_size(tensor->type);
}

static bool matmul_f16_f32_f32(global_context& ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, void * wdata, size_t wsize) {
    // printf("matmul_f16_f32_f32 %d\n", (int)checkStrides(src0));
    if (!check_strides(src0))
        return false;

    const auto ne00 = src0->ne[0];
    const auto ne01 = src0->ne[1];
    const auto ne02 = src0->ne[2];
    const auto ne03 = src0->ne[3];

    const auto ne10 = src1->ne[0];
    const auto ne11 = src1->ne[1];
    const auto ne12 = src1->ne[2];
    const auto ne13 = src1->ne[3];

    const auto ne0  = dst->ne[0];
    const auto ne1  = dst->ne[1];
    const auto ne2  = dst->ne[2];
    const auto ne3  = dst->ne[3];

    (void)ne00;(void)ne01;(void)ne02;(void)ne03;
    (void)ne10;(void)ne11;(void)ne12;(void)ne13;
    (void)ne0 ;(void)ne1 ;(void)ne2 ;(void)ne3;

    const auto nb00 = src0->nb[0];
    const auto nb01 = src0->nb[1];
    const auto nb02 = src0->nb[2];
    const auto nb03 = src0->nb[3];

    const auto nb10 = src1->nb[0];
    const auto nb11 = src1->nb[1];
    const auto nb12 = src1->nb[2];
    const auto nb13 = src1->nb[3];

    const auto nb0  = dst->nb[0];
    const auto nb1  = dst->nb[1];
    const auto nb2  = dst->nb[2];
    const auto nb3  = dst->nb[3];

    (void)nb00;(void)nb01;(void)nb02;(void)nb03;
    (void)nb10;(void)nb11;(void)nb12;(void)nb13;
    (void)nb0 ;(void)nb1 ;(void)nb2 ;(void)nb3 ;

    // if (ne11 * ne01 * ne10 < 32*32*32)
    //     return false;

    const int64_t scratch_local_size = ne10 * ne11 * sizeof(sycl::half);
    const int64_t scratch_size = scratch_local_size * ne02 * ne03;

    local_context_guard g(ctx);
    auto &queue = g.lctx->queue;
    auto &deps = g.lctx->deps;
    assert(deps.empty());

    auto scratch = (char *) g.lctx->get_scratch(scratch_size);
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            auto x  = (sycl::half *) ((char *) src0->data + i02*nb02 + i03*nb03);
            auto y  =      (float *) ((char *) src1->data + i02*nb12 + i03*nb13);
            auto sc = (sycl::half *) (scratch + i02*scratch_local_size + i03*ne02*scratch_local_size);

            deps.emplace_back(
                convert_type_2d<float, sycl::half>(queue, y, sc, ne10, ne11,
                                                   nb10, nb11));

            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            namespace blas = oneapi::mkl::blas::row_major;
            blas::gemm(queue,
                oneapi::mkl::transpose::N,
                oneapi::mkl::transpose::T,
                ne11, ne01, ne10,
                1.0f,   sc, ne10,
                         x, nb01 / sizeof(sycl::half),
                0.0f,    d, ne01,
                deps);
            deps.clear();
        }
    }
    queue.wait();
    return true;
}
