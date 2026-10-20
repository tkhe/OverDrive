#include <cuda_runtime_api.h>

namespace overdrive
{
    int get_cudart_version()
    {
        return CUDART_VERSION;
    }
}  // namespace overdrive
