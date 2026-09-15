#pragma once

#include "allocator.hpp"

namespace chaosuan::core::allocators {
class NaiveAllocator : public MemoryAllocator {
public:
    NaiveAllocator(const ChaosuanRuntimeAPI *runtime_api);
    ~NaiveAllocator() = default;
    std::byte *allocate(size_t size) override;
    void release(std::byte *memory) override;
};
} // namespace chaosuan::core::allocators