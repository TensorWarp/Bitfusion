#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>

class DataReader {
public:
    uint32_t getRows() const;
    int getColumns() const;
    virtual bool readRow(std::string* key, float* vector) = 0;
    virtual ~DataReader() = default;

protected:
    uint32_t rows = 0;
    int columns = 0;
};

class TextFileDataReader : public DataReader {
public:
    TextFileDataReader(const std::string& fileName, char keyValueDelimiter, char vectorDelimiter);

    bool readRow(std::string* key, float* vector) override;

    ~TextFileDataReader() {
        fileStream.close();
    }

private:
    std::string fileName;
    std::ifstream fileStream;
    char keyValueDelimiter;
    char vectorDelimiter;

    void findDataDimensions(const std::string& fileName, char keyValueDelimiter, char vectorDelimiter);
};

TextFileDataReader::TextFileDataReader(const std::string& fileName, char keyValueDelimiter, char vectorDelimiter) :
    fileName(fileName),
    fileStream(fileName, std::ios_base::in),
    keyValueDelimiter(keyValueDelimiter),
    vectorDelimiter(vectorDelimiter) {
    findDataDimensions(fileName, keyValueDelimiter, vectorDelimiter);
}

const int failure = 1;
const int success = 0;

int splitKeyVector(const std::string& line, std::string& key, std::string& vector, char keyValueDelimiter) {
    int keyValDelimIndex = line.find_first_of(keyValueDelimiter);

    if (keyValDelimIndex == std::string::npos) {
        return failure;
    }

    key = line.substr(0, keyValDelimIndex);
    vector = line.substr(static_cast<std::basic_string<char, std::char_traits<char>, std::allocator<char>>::size_type>(keyValDelimIndex) + 1, line.size());
    return success;
}

void TextFileDataReader::findDataDimensions(const std::string& fileName, char keyValueDelimiter, char vectorDelimiter) {
    std::ifstream fs(fileName, std::ios_base::in);

    rows = 0;
    columns = 0;

    std::string line;
    while (std::getline(fs, line)) {
        if (line.empty()) {
            continue;
        }

        ++rows;

        std::string key;
        std::string vectorStr;

        if (splitKeyVector(line, key, vectorStr, keyValueDelimiter)) {
            throw std::invalid_argument("Malformed line: " + line);
        }

        std::stringstream vectorStrStream(vectorStr);
        std::string elementStr;
        int columnsInRow = 0;
        while (std::getline(vectorStrStream, elementStr, vectorDelimiter)) {
            ++columnsInRow;
        }

        if (columns == 0) {
            columns = columnsInRow;
        }
        else if (columns != columnsInRow) {
            throw std::invalid_argument("Inconsistent number of columns detected: " + std::to_string(columnsInRow));
        }
    }

    fs.close();
}

bool TextFileDataReader::readRow(std::string* key, float* vector) {
    std::string line;
    if (std::getline(fileStream, line)) {
        std::string vectorStr;
        splitKeyVector(line, *key, vectorStr, keyValueDelimiter);

        std::stringstream vectorStrStream(vectorStr);
        std::string elementStr;
        size_t idx;

        for (int i = 0; std::getline(vectorStrStream, elementStr, vectorDelimiter); ++i) {
            try {
                vector[i] = std::stof(elementStr, &idx);
                if (idx != elementStr.size()) {
                    throw std::invalid_argument("Malformed vector element: " + elementStr);
                }
            }
            catch (const std::exception& e) {
                throw std::invalid_argument("Error parsing as float: " + elementStr + ", Column " + std::to_string(i) + " of: " + line);
            }
        }

        return true;
    }
    else {
        return false;
    }
}