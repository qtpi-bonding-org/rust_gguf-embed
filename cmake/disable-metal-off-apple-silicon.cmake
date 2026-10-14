# llama-cpp-sys-2 (vendored llama.cpp) inherits ggml's own CMake default of
# GGML_METAL=ON on every Apple platform, regardless of CPU arch and of the
# runtime `n_gpu_layers`. On an Intel Mac that compiles Metal in and
# initializes it even when `gpu_layers_override` forces CPU-only, which has
# produced NaN embeddings (AMD Radeon Pro 5300M) and, with the llama.cpp
# snapshot in llama-cpp-sys-2 0.1.159, a silent multi-minute stall in
# ggml_metal_library_compile_all at backend init.
#
# Wired in by .cargo/config.toml (CMAKE_TOOLCHAIN_FILE is one of the CMAKE_*
# variables llama-cpp-sys-2's build.rs forwards to CMake), so it only applies
# when building this crate on its own (its tests, examples). Consumers set
# their own: episutra-frb/cmake/disable-metal-when-cpu-only.cmake.
#
# Apple Silicon keeps Metal. Everything else gets it forced off; `option()`
# never overrides an existing CACHE entry, so setting it here wins.
if(DEFINED CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
    set(_gguf_embed_target_arch "${CMAKE_OSX_ARCHITECTURES}")
elseif(DEFINED CMAKE_SYSTEM_PROCESSOR AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "")
    set(_gguf_embed_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
else()
    set(_gguf_embed_target_arch "${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()

if(NOT _gguf_embed_target_arch MATCHES "^(arm64|aarch64)$")
    set(GGML_METAL OFF CACHE BOOL
        "Disabled by gguf-embed/cmake/disable-metal-off-apple-silicon.cmake" FORCE)
endif()
