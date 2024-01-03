#pragma once

#include <cassert>
#include <string>

#include "logger.h"

namespace bitfusion
{
    namespace common
    {

        enum class ReallocType
        {
            INCREASE,
            REUSE,
            DECREASE,
        };

        class IAllocator
        {
        public:
            virtual ~IAllocator() = default;

            IAllocator(const IAllocator&) = delete;
            IAllocator& operator=(const IAllocator&) = delete;

            template <typename T>
            [[nodiscard]] T* reMalloc(T* ptr, size_t sizeBytes, const bool setZero = true)
            {
                TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
                auto const sizeAligned = ((sizeBytes + 31) / 32) * 32;
                if (contains(ptr))
                {
                    auto const realloc = reallocType(ptr, sizeAligned);
                    if (realloc == ReallocType::INCREASE)
                    {
                        TLLM_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", ptr);
                        free(&ptr);
                        return reinterpret_cast<T*>(malloc(sizeAligned, setZero));
                    }
                    else if (realloc == ReallocType::DECREASE)
                    {
                        TLLM_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", ptr);
                        free(&ptr);
                        return reinterpret_cast<T*>(malloc(sizeAligned, setZero));
                    }
                    else
                    {
                        assert(realloc == ReallocType::REUSE);
                        TLLM_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", ptr, sizeAligned);
                        if (setZero)
                        {
                            memSet(ptr, 0, sizeAligned);
                        }
                        return ptr;
                    }
                }
                else
                {
                    TLLM_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", ptr);
                    return reinterpret_cast<T*>(malloc(sizeAligned, setZero));
                }
            }

            virtual void free(void** ptr) = 0;

            template <typename T>
            void free(T** ptr)
            {
                free(reinterpret_cast<void**>(ptr));
            }

        protected:
            IAllocator() = default;

            virtual void* malloc(std::size_t size, bool setZero) = 0;
            virtual void memSet(void* ptr, int val, std::size_t size) = 0;

            virtual bool contains(void const* ptr) const = 0;
            virtual ReallocType reallocType(void const* ptr, size_t size) const = 0;
        };
    }
}