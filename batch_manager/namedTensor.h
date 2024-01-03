#pragma once

#include "../runtime/iTensor.h"

#include <string>

namespace bitfusion::batch_manager
{
    template <typename TTensor>
    class GenericNamedTensor
    {
    public:
        using TensorPtr = TTensor;

        TensorPtr tensor;
        std::string name;

        GenericNamedTensor() = default;
        ~GenericNamedTensor() = default;

        GenericNamedTensor(TensorPtr _tensor, std::string _name)
            : tensor{ std::move(_tensor) }
            , name{ std::move(_name) }
        {
        }

        explicit GenericNamedTensor(std::string _name)
            : tensor{}
            , name{ std::move(_name) }
        {
        }

        TensorPtr operator()()
        {
            return tensor;
        }

        TensorPtr const& operator()() const
        {
            return tensor;
        }
    };

    class NamedTensor : public GenericNamedTensor<bitfusion::runtime::ITensor::SharedPtr>
    {
    public:
        using Base = GenericNamedTensor<bitfusion::runtime::ITensor::SharedPtr>;
        using TensorPtr = Base::TensorPtr;

        NamedTensor(
            nvinfer1::DataType _type, std::vector<int64_t> const& _shape, std::string _name, const void* _data = nullptr);

        NamedTensor(TensorPtr _tensor, std::string _name)
            : Base(std::move(_tensor), std::move(_name)) {};

        explicit NamedTensor(std::string _name)
            : Base(std::move(_name)) {};

        [[nodiscard]] std::vector<int64_t> serialize() const;

        static NamedTensor deserialize(const int64_t* packed);
    };
}