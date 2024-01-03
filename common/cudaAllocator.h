#pragma once

#include "allocator.h"
#include "../runtime/bufferManager.h"

#include <cuda_runtime.h>
#include <unordered_map>

namespace bitfusion
{

    namespace common
    {

        class CudaAllocator : public IAllocator
        {
        public:
            explicit CudaAllocator(runtime::BufferManager bufferManager);

            ~CudaAllocator() override = default;

            void free(void** ptr) override;

        protected:
            bool contains(void const* ptr) const override
            {
                return mPointerMapping.find(ptr) != mPointerMapping.end();
            }

            ReallocType reallocType(void const* ptr, size_t size) const override;

            void* malloc(size_t size, bool setZero) override;

            void memSet(void* ptr, int val, size_t size) override;

        private:
            runtime::BufferManager mBufferManager;
            std::unordered_map<void const*, runtime::BufferManager::IBufferPtr> mPointerMapping{};
        };

    }
}