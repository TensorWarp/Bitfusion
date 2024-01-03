
#pragma once

#include "../../runtime/bufferManager.h"
#include "../../runtime/iTensor.h"

#include <string>

namespace bitfusion::runtime::utils
{

[[nodiscard]] ITensor::UniquePtr loadNpy(BufferManager& manager, const std::string& npyFile, const MemoryType where);

void saveNpy(BufferManager& manager, ITensor const& tensor, const std::string& filename);

}
