#pragma once

#include <fstream>

/// <summary>
/// Abstract class for reading data.
/// </summary>
class DataReader
{
public:

    /// <summary>
    /// Reads a row of data from the data source.
    /// </summary>
    /// <param name="key">A pointer to store the key from the row.</param>
    /// <param name="vector">A pointer to store the vector data from the row.</param>
    /// <returns>True if a row was successfully read, false if the end of the data source is reached or an error occurs.</returns>
    virtual bool readRow(std::string* key, float* vector) = 0;

    /// <summary>
    /// Gets the number of rows in the data source.
    /// </summary>
    /// <returns>The number of rows.</returns>
    uint32_t getRows() const;

    /// <summary>
    /// Gets the number of columns in the data source.
    /// </summary>
    /// <returns>The number of columns.</returns>
    int getColumns() const;

    /// <summary>
    /// Virtual destructor for the DataReader class.
    /// </summary>
    virtual ~DataReader()
    {
    }

protected:
    uint32_t rows; ///< The number of rows in the data source.
    int columns;   ///< The number of columns in the data source.
};

/// <summary>
/// Class for reading data from a text file.
/// </summary>
class TextFileDataReader : public DataReader
{
public:

    /// <summary>
    /// Constructor for TextFileDataReader.
    /// </summary>
    /// <param name="fileName">The name of the text file to read from.</param>
    /// <param name="keyValueDelimiter">The delimiter character separating key and value in each row (default is '\t').</param>
    /// <param name="vectorDelimiter">The delimiter character separating vector elements in each row (default is ' ').</param>
    TextFileDataReader(const std::string& fileName, char keyValueDelimiter = '\t', char vectorDelimiter = ' ');

    /// <summary>
    /// Reads a row of data from the text file.
    /// </summary>
    /// <param name="key">A pointer to store the key from the row.</param>
    /// <param name="vector">A pointer to store the vector data from the row.</param>
    /// <returns>True if a row was successfully read, false if the end of the file is reached or an error occurs.</returns>
    bool readRow(std::string* key, float* vector);

    /// <summary>
    /// Static method to find the dimensions (rows and columns) of the data in a text file.
    /// </summary>
    /// <param name="fileName">The name of the text file to analyze.</param>
    /// <param name="rows">A reference to store the number of rows found in the file.</param>
    /// <param name="columns">A reference to store the number of columns found in the file.</param>
    /// <param name="keyValueDelimiter">The delimiter character separating key and value in each row (default is '\t').</param>
    /// <param name="vectorDelimiter">The delimiter character separating vector elements in each row (default is ' ').</param>
    static void findDataDimensions(const std::string& fileName, uint32_t& rows, int& columns, char keyValueDelimiter =
        '\t', char vectorDelimiter = ' ');

    /// <summary>
    /// Destructor for TextFileDataReader.
    /// </summary>
    ~TextFileDataReader();

  private:
    std::string fileName;
    std::fstream fileStream;
    char keyValueDelimiter;
    char vectorDelimiter;
};