#include <torch/extension.h>
#include <sycl/sycl.hpp>

// Forward declare the kernel wrapper
template<typename scalar_t>
void selective_scan_kernel(
    sycl::queue &q,
    scalar_t* x,
    scalar_t* dt,
    scalar_t* A,
    scalar_t* B,
    scalar_t* C,
    scalar_t* out,
    int batch, int seqlen, int dim, int dstate);

torch::Tensor selective_scan_sycl(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    auto batch = x.size(0);
    auto seqlen = x.size(1);
    auto dim = x.size(2);
    auto dstate = B.size(2); // Assuming B is [batch, seqlen, dstate]

    auto out = torch::empty_like(x);

    // Initialize SYCL queue targeting GPU
    auto q = sycl::queue(sycl::gpu_selector_v);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "selective_scan_sycl", ([&] {
        selective_scan_kernel<scalar_t>(
            q,
            x.data_ptr<scalar_t>(),
            dt.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch, seqlen, dim, dstate
        );
    }));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan", &selective_scan_sycl, "Selective Scan (SYCL)");
}
