#pragma once

#include <deque>
#include <mutex>
#include <functional>
#include <future>
#include <thread>
#include <vector>
#include <condition_variable>

class ThreadPool {
public:
    ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this]() { return stop || !tasks.empty(); });
                        if (stop && tasks.empty())
                            return;
                        task = std::move(tasks.front());
                        tasks.pop_front();
                    }
                    task();
                }
                });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
    }

    template <typename Function>
    void AddTask(Function&& func) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace_back([func]() { func(); });
        }
        condition.notify_one();
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
        using return_type = typename std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace_back([task]() { (*task)(); });
        }
        condition.notify_one();
        return result;
    }

    template <typename Function>
    void Execute(Function&& func) {
        AddTask(std::forward<Function>(func));
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
    }

private:
    std::vector<std::thread> threads;
    std::deque<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};
