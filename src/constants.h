#pragma once
#include <cstdint>

constexpr float epsilon = 1e-6f;
constexpr int tileSizeX = 16;
constexpr int tileSizeY = 16;
constexpr int numLayers = 4;
constexpr int numAttentionHeads = 8;
constexpr int numHierarchicalLevels = 3;
constexpr int numCrossAttentionHeads = 4;
constexpr int numAxialAttentionHeads = 4;
constexpr float MIN_ACTIVATION = 0.000001f;
constexpr float MAX_ACTIVATION = 0.999999f;
constexpr float MIN_FLOAT = -99999999.0f;
constexpr int MAX_VALUE = 100;
#define ERRORSCALEF 0.5f
#define ONEOVERERRORSCALE (1.0 / ERRORSCALEF)
constexpr auto NUM_GPUS = 4;
constexpr auto NUM_NODES = 2;
constexpr auto NUM_ITERATIONS = 5;
constexpr auto NUM_COLLECTIVES = 3;
constexpr auto NUM_TASKS = 6;
constexpr auto NUM_PARTITIONS = 8;
constexpr auto NUM_COMM_GROUPS = 2;
constexpr auto NUM_BARRIERS = 3;
constexpr auto NUM_BETA_ROUNDS = 4;
const int TILE_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 128;
constexpr int NUM_STREAMS_PER_GPU = 4;
constexpr int NUM_WORKER_THREADS = 8;
static const float VERSION = 0.9f;
#define MIN_ERROR (float)__exp10f(-20.0)
constexpr uint32_t ALIGNMENT = 512;
constexpr uint32_t ALIGNMENT_MASK = ALIGNMENT - 1;
