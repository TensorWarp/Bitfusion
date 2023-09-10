#pragma once

#include <deque>
#include <mutex>
#include <functional>
#include <future>
#include <thread>
#include <vector>
#include <condition_variable>

#include "../utils/ConfigException.h"

/// <summary>
/// Thread pool implementation for concurrent task execution.
/// </summary>
class ThreadPool {
public:
    /// <summary>
    /// Constructs a thread pool with the default number of threads (equal to the number of CPU cores).
    /// </summary>
    ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

    /// <summary>
    /// Constructs a thread pool with the specified number of threads.
    /// </summary>
    /// <param name="numThreads">The number of threads in the thread pool.</param>
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

    /// <summary>
    /// Destroys the thread pool and joins all worker threads.
    /// </summary>
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

    /// <summary>
    /// Execute a function asynchronously in the thread pool.
    /// </summary>
    /// <typeparam name="F">The type of the function to execute.</typeparam>
    /// <typeparam name="Args">The types of arguments for the function.</typeparam>
    /// <param name="f">The function to execute.</param>
    /// <param name="args">The arguments for the function.</param>
    template <class F, class... Args>
    void Execute(F&& f, Args&&... args) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stop) {
                throw ConfigException("ThreadPool is stopped.", DataSetEnums::ErrorCode::ThreadPoolError);
            }
            tasks.emplace_back([task]() { (*task)(); });
        }
        condition.notify_one();
    }


    /// <summary>
    /// Adds a task to the thread pool for execution.
    /// </summary>
    /// <typeparam name="Function">The type of the task function.</typeparam>
    /// <param name="func">The task function to be executed.</param>
    template <typename Function>
    void AddTask(Function&& func) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::forward<Function>(func));
        }
        condition.notify_one();
    }

    /// <summary>
    /// Enqueues a task with arguments for execution and returns a future for the result.
    /// </summary>
    /// <typeparam name="F">The type of the task function.</typeparam>
    /// <typeparam name="Args">The types of the arguments to the task function.</typeparam>
    /// <param name="f">The task function to be executed.</param>
    /// <param name="args">The arguments to be passed to the task function.</param>
    /// <returns>A future for the result of the task.</returns>
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

private:
    std::vector<std::thread> threads;
    std::deque<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};