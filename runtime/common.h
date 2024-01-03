#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace bitfusion::runtime
{

using SizeType = std::int32_t;

using TokenIdType = std::int32_t;

template <typename T>
using StringPtrMap = std::unordered_map<std::string, std::shared_ptr<T>>;

}
