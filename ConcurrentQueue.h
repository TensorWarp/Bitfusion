#pragma once

template <typename T>
class ConcurrentQueue {
public:
    ConcurrentQueue() = default;

    void Push(const T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(value);
        condition_.notify_one();
    }

    bool TryPop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = queue_.front();
        queue_.pop();
        return true;
    }

    bool Empty() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t Size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;
};