#include "envUtils.h"
#include "logger.h"
#include <cstdlib>

namespace bitfusion::common
{

    bool getEnvMmhaMultiblockDebug()
    {
        static bool init = false;
        static bool forceMmhaMaxSeqLenTile = false;
        if (!init)
        {
            init = true;
            const char* enable_mmha_debug_var = std::getenv("TRENABLE_MMHA_MULTI_BLOCK_DEBUG");
            if (enable_mmha_debug_var)
            {
                if (enable_mmha_debug_var[0] == '1' && enable_mmha_debug_var[1] == '\0')
                {
                    forceMmhaMaxSeqLenTile = true;
                }
            }
        }
        return forceMmhaMaxSeqLenTile;
    }

    int getEnvMmhaBlocksPerSequence()
    {
        static bool init = false;
        static int mmhaBlocksPerSequence = 0;
        if (!init)
        {
            init = true;
            const char* mmhaBlocksPerSequenceEnv = std::getenv("TRMMHA_BLOCKS_PER_SEQUENCE");
            if (mmhaBlocksPerSequenceEnv)
            {
                mmhaBlocksPerSequence = std::atoi(mmhaBlocksPerSequenceEnv);
                if (mmhaBlocksPerSequence <= 0)
                {
                    LOG_WARNING("Invalid value for TRMMHA_BLOCKS_PER_SEQUENCE. Will use default values instead!");
                }
            }
        }
        return mmhaBlocksPerSequence;
    }

}