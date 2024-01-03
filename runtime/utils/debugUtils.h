#pragma once

#include "../../runtime/bufferManager.h"
#include "../../runtime/runtimeKernels.h"

namespace bitfusion::runtime
{
namespace utils
{

bool tensorHasNan(const IBuffer& tensor, BufferManager& manager);

}
}
