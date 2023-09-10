#pragma once

#include "../system/GpuTypes.h"
#include "../system/Types.h"

struct CDL
{
	/// <summary>
	/// Initializes a new instance of the CDL class.
	/// </summary>
	CDL();

	/// <summary>
	/// Loads a JSON file.
	/// </summary>
	/// <param name="fname">The name of the JSON file to load.</param>
	/// <returns>The result of the JSON loading operation.</returns>
	int Load_JSON(const std::string& fname);

	/// <summary>
	/// The name of the network file.
	/// </summary>
	std::string _networkFileName;

	/// <summary>
	/// The random seed value.
	/// </summary>
	int _randomSeed;

	/// <summary>
	/// The mode of operation.
	/// </summary>
	Mode _mode;

	/// <summary>
	/// The name of the data file.
	/// </summary>
	std::string _dataFileName;

	/// <summary>
	/// The number of epochs for training.
	/// </summary>
	int _epochs;

	/// <summary>
	/// The batch size for training.
	/// </summary>
	int _batch;

	/// <summary>
	/// The learning rate (alpha) for training.
	/// </summary>
	float _alpha;

	/// <summary>
	/// The regularization parameter (lambda) for training.
	/// </summary>
	float _lambda;

	/// <summary>
	/// The momentum parameter (mu) for training.
	/// </summary>
	float _mu;

	/// <summary>
	/// The interval for updating the learning rate (alpha).
	/// </summary>
	int _alphaInterval;

	/// <summary>
	/// The multiplier for updating the learning rate (alpha).
	/// </summary>
	float _alphaMultiplier;

	/// <summary>
	/// The optimizer used for training.
	/// </summary>
	TrainingMode _optimizer;

	/// <summary>
	/// The name of the checkpoint file.
	/// </summary>
	std::string _checkpointFileName;

	/// <summary>
	/// The interval for saving checkpoint files during training.
	/// </summary>
	int _checkpointInterval;

	/// <summary>
	/// Indicates whether to shuffle the data indexes during training.
	/// </summary>
	bool _shuffleIndexes;

	/// <summary>
	/// The name of the results file.
	/// </summary>
	std::string _resultsFileName;
};