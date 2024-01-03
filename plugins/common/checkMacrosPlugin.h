#pragma once

#include "../../common/assert.h"
#include "../../common/cudaUtils.h"

namespace bitfusion::plugins
{

void logError(const char* msg, const char* file, const char* fn, int line);

void caughtError(const std::exception& e);

}
