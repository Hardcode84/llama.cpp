#include "ggml-sycl.h"

#include <cstdio>
#include <cassert>
#include <atomic>
#include <mutex>
#include <memory>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

static auto getDeviceSelector(std::string deviceName) {
  using Sel = sycl::ext::oneapi::filter_selector;
  return [selector = Sel(std::move(deviceName))](
             const sycl::device &dev) -> int { return selector(dev); };
}

template <typename F> static auto catchAll(F &&func) {
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


namespace {
struct LocalContext;
struct Context {
    Context() {
        // TODO: Unhardcode deivce
        // device = sycl::device{getDeviceSelector("level_zero:gpu:0")};
        device = sycl::device{getDeviceSelector("opencl:cpu:0")};
        queue = sycl::queue{device};
    }

    LocalContext* getContext();
    void returnContext(LocalContext* ctx);

    sycl::device device;
    sycl::queue queue;
    std::atomic<int> requestsInFlight{0};
    std::mutex lock;
    std::unique_ptr<LocalContext> lctx;
};

struct LocalContext {
    LocalContext(Context& ctx) {
        queue = sycl::queue{ctx.queue.get_context(), ctx.device};
    }
    ~LocalContext() {
        freeScratch();
    }

    void* getScratch(size_t size) {
        if (size <= scratchSize)
            return scratch;

        freeScratch();
        scratch = sycl::malloc_device(size, queue);
        if (!scratch) {
            fprintf(stderr, "SYCL: Failed to allocate scratch memory\n");
            abort();
        }

        scratchSize = size;
        return scratch;
    }

    void freeScratch() {
        if (scratch) {
            sycl::free(scratch, queue);
            scratch = nullptr;
        }
    }

    sycl::queue queue;
    std::vector<sycl::event> deps;
    void* scratch = nullptr;
    size_t scratchSize = 0;
    std::unique_ptr<LocalContext> next;
};

LocalContext* Context::getContext() {
    std::unique_lock<std::mutex> l(lock); // TODO: atomics?
    if (!lctx)
        return new LocalContext(*this);

    auto ret = lctx.release();
    lctx = std::move(ret->next);
    return ret;
}

void Context::returnContext(LocalContext* ctx) {
    assert(ctx && "Invalid local context");
    assert(!ctx.next && "Invalid local context next");
    ctx->deps.clear();
    std::unique_lock<std::mutex> l(lock); // TODO: atomics?
    ctx->next = std::move(lctx);
    lctx.reset(ctx);
}

struct LocalContextGuard {
    LocalContextGuard(Context& c): ctx(c) {
        ++ctx.requestsInFlight;
        lctx = ctx.getContext();
        assert(lctx);
    }
    ~LocalContextGuard() {
        assert(lctx);
        ctx.returnContext(lctx);
        --ctx.requestsInFlight;
    }

    Context& ctx;
    LocalContext* lctx = nullptr;
};
}

static Context* context = nullptr;

extern "C" void ggml_sycl_init() {
    printf("ggml_cycl_init\n");
    catchAll([&]() {
        assert(!context && "Context is already initialized");
        context = new Context;
    });
}

static bool matmul_f16_f32_f32(Context& ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, void * wdata, size_t wsize);

extern "C" bool ggml_sycl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize) {
    assert(context && "Context is not initialized");
    catchAll([&]() {
        auto checkTypes = [&](ggml_type a, ggml_type b, ggml_type c) {
            return src0->type == a && src1->type == b && dst->type == c;
        };
#define T(a) GGML_TYPE_##a
    if (checkTypes(T(F16), T(F32), T(F32))) return matmul_f16_f32_f32(*context, src0, src1, dst, wdata, wsize);
#undef T
    });
    return false;
}


template<typename Src, typename Dst>
static sycl::event convertType2d(sycl::queue queue, const void* src, void* dst, int64_t ne00, int64_t ne01, int64_t nb00, int64_t nb01) {
    auto dstTyped = static_cast<Dst*>(dst);
    return queue.submit([&](sycl::handler& h) {
        sycl::range<2> r{ne00, ne01};
        h.parallel_for<class type_convert>(r, [=](sycl::item<2> idx) {
            auto i00 = idx.get_id(0);
            auto i01 = idx.get_id(1);
            auto dstId = i00 + i01 * ne00;
            dstTyped[dstId] = (Dst)*(Src*) ((const char *) src + i01*nb01 + i00*nb00);
        });
    });
}


static bool checkStrides(const ggml_tensor * tensor) {
    return tensor->nb[0] == ggml_type_size(tensor->type);
}

static bool matmul_f16_f32_f32(Context& ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, void * wdata, size_t wsize) {
    printf("matmul_f16_f32_f32 %d\n", (int)checkStrides(src0));
    if (!checkStrides(src0))
        return false;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];
    const int64_t ne3  = dst->ne[3];

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int64_t scratchLocalSize = ne00 * ne01 * sizeof(sycl::half);
    const int64_t scratchSize = scratchLocalSize * ne02 * ne03;

    printf("matmul_f16_f32_f32 1\n"); fflush(stdout);

    LocalContextGuard g(ctx);
    printf("matmul_f16_f32_f32 2\n"); fflush(stdout);
    auto &queue = g.lctx->queue;
    auto &deps = g.lctx->deps;
    auto scratch = (char *) g.lctx->getScratch(scratchSize);
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            auto x = (sycl::half *) ((char *) src0->data + i03*nb03 + i02*nb02);
            auto y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);
            auto sc = (sycl::half *) (scratch + scratchLocalSize * i02 + ne02 * scratchLocalSize);
            printf("matmul_f16_f32_f32 3\n"); fflush(stdout);
            deps.clear();
            deps.emplace_back(convertType2d<float, sycl::half>(queue, y, sc, ne10, ne11, nb10, nb11));
            printf("matmul_f16_f32_f32 4\n"); fflush(stdout);

            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            namespace blas = oneapi::mkl::blas::row_major;
            blas::gemm(queue,
                oneapi::mkl::transpose::N,
                oneapi::mkl::transpose::T,
                ne11, ne01, ne10,
                1.0f,   sc, ne10,
                         x, ne00,
                0.0f,    d, ne01,
                deps);
            printf("matmul_f16_f32_f32 5\n"); fflush(stdout);
        }
    }
    printf("matmul_f16_f32_f32 6\n"); fflush(stdout);
    queue.wait();
    printf("matmul_f16_f32_f32 7\n"); fflush(stdout);
    return true;
}