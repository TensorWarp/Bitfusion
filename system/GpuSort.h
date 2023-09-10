#pragma once

#include <memory>

/// <summary>
/// Template class for performing GPU sorting operations.
/// </summary>
/// <typeparam name="KeyType">The data type of the keys to be sorted.</typeparam>
/// <typeparam name="ValueType">The data type associated with the keys.</typeparam>
template<typename KeyType, typename ValueType> class GpuSort
{
private:
    unsigned int _items;
    unsigned int _itemStride;
    std::unique_ptr<GpuBuffer<KeyType>> _pbKey;
    KeyType* _pKey0;
    KeyType* _pKey1;
    std::unique_ptr<GpuBuffer<ValueType>> _pbValue;
    ValueType* _pValue0;
    ValueType* _pValue1;
    size_t _tempBytes;
    std::unique_ptr<GpuBuffer<char>> _pbTemp;
    KeyType* _pKey;
    ValueType* _pValue;


    
public:
    
    /// <summary>
    /// Constructor for the GpuSort class.
    /// </summary>
    /// <param name="items">The number of items to be sorted.</param>
    GpuSort(unsigned int items) :
    _items(items),
    _itemStride(((items + 511) >> 9) << 9),
    _pbKey(new GpuBuffer<KeyType>(_itemStride * 2)),
    _pKey0(_pbKey->_pDevData),
    _pKey1(_pbKey->_pDevData + _itemStride),
    _pbValue(new GpuBuffer<ValueType>(_itemStride * 2)),
    _pValue0(_pbValue->_pDevData),
    _pValue1(_pbValue->_pDevData + _itemStride),
    _tempBytes(kInitSort(_items, _pbValue.get(), _pbKey.get())),
    _pbTemp(new GpuBuffer<char>(_tempBytes))
    {
        _pKey = _pKey0;
        _pValue = _pValue0;      
    }

    /// <summary>
    /// Destructor for the GpuSort class.
    /// </summary>
    ~GpuSort()
    {
    }
    
    /// <summary>
    /// Sorts the data using GPU-based sorting.
    /// </summary>
    /// <returns>True if the sorting operation was successful, otherwise false.</returns>
    bool Sort() { return kSort(_items, _pKey0, _pKey1, _pValue0, _pValue1, _pbTemp->_pDevData, _tempBytes); }
    
    /// <summary>
    /// Gets the GPU buffer containing keys.
    /// </summary>
    /// <returns>A pointer to the GPU buffer.</returns>
    GpuBuffer<KeyType>* GetKeyBuffer() { return _pbKey.get(); }
    
    /// <summary>
    /// Gets the GPU buffer containing associated values.
    /// </summary>
    /// <returns>A pointer to the GPU buffer.</returns>
    GpuBuffer<ValueType>* GetValueBuffer() { return _pbValue.get(); }
    
    /// <summary>
    /// Gets a pointer to the keys data on the GPU.
    /// </summary>
    /// <returns>A pointer to the keys data.</returns>
    KeyType* GetKeyPointer() { return _pKey;}
    
    /// <summary>
    /// Gets a pointer to the associated values data on the GPU.
    /// </summary>
    /// <returns>A pointer to the values data.</returns>
    ValueType* GetValuePointer() { return _pValue; }
};
