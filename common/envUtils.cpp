#include "envUtils.h"
#include "logger.h"
#include <cstdlib>
#include <optional>
#include <string_view>

namespace bitfusion::common
{
    namespace
    {
        std::optional<bool> forceMmhaMaxSeqLenTile;
        std::optional<int> mmhaBlocksPerSequence;
        
        /// <summary>
        /// Retrieves a boolean value from the environment variable.
        /// </summary>
        /// <param name="envVarName">The name of the environment variable.</param>
        /// <returns>An optional boolean value. If the variable is not set or not equal to "1", returns nullopt.</returns>
        std::optional<bool> getEnvBool(const char* envVarName)
        {
            if (const char* envValue = std::getenv(envVarName); envValue)
            {
                return std::string_view(envValue) == "1";
            }
            return std::nullopt;
        }

        /// <summary>
        /// Retrieves an integer value from the environment variable.
        /// </summary>
        /// <param name="envVarName">The name of the environment variable.</param>
        /// <returns>An optional integer value. If the variable is not set or not a valid integer, returns nullopt.</returns>
        std::optional<int> getEnvInt(const char* envVarName)
        {
            if (const char* envValue = std::getenv(envVarName); envValue)
            {
                try
                {
                    return std::stoi(envValue);
                }
                catch (const std::invalid_argument&)
                {
                    LOG_WARNING("Invalid value for ", envVarName, ". Will use default values instead!");
                }
            }
            return std::nullopt;
        }
    }

    /// <summary>
    /// Retrieves the environment variable TRENABLE_MMHA_MULTI_BLOCK_DEBUG and returns its boolean value.
    /// </summary>
    /// <returns>The boolean value of the environment variable or false if not set.</returns>
    bool getEnvMmhaMultiblockDebug()
    {
        if (!forceMmhaMaxSeqLenTile.has_value())
        {
            forceMmhaMaxSeqLenTile = getEnvBool("TRENABLE_MMHA_MULTI_BLOCK_DEBUG");
        }
        return forceMmhaMaxSeqLenTile.value_or(false);
    }

    /// <summary>
    /// Retrieves the environment variable TRMMHA_BLOCKS_PER_SEQUENCE and returns its integer value.
    /// </summary>
    /// <returns>The integer value of the environment variable or 0 if not set or not a valid integer.</returns>
    int getEnvMmhaBlocksPerSequence()
    {
        if (!mmhaBlocksPerSequence.has_value())
        {
            mmhaBlocksPerSequence = getEnvInt("TRMMHA_BLOCKS_PER_SEQUENCE");
        }
        return mmhaBlocksPerSequence.value_or(0);
    }
}