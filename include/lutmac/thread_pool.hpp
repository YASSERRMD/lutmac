#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace lutmac {

class ThreadPool {
public:
  static ThreadPool &instance() {
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
  }

  template <typename F> void parallel_for(size_t start, size_t end, F &&func) {
    if (end <= start)
      return;

    size_t n = end - start;
    size_t num_workers = std::min(n, workers_.size());
    size_t chunk = (n + num_workers - 1) / num_workers;

    std::atomic<size_t> completed{0};

    for (size_t t = 0; t < num_workers; ++t) {
      size_t chunk_start = start + t * chunk;
      size_t chunk_end = std::min(chunk_start + chunk, end);

      if (chunk_start >= end)
        break;

      enqueue([&func, chunk_start, chunk_end, &completed] {
        for (size_t i = chunk_start; i < chunk_end; ++i) {
          func(i);
        }
        completed.fetch_add(1, std::memory_order_release);
      });
    }

    // Wait for completion
    while (completed.load(std::memory_order_acquire) < num_workers) {
      std::this_thread::yield();
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : workers_) {
      t.join();
    }
  }

private:
  ThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty())
              return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  void enqueue(std::function<void()> task) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      tasks_.push(std::move(task));
    }
    cv_.notify_one();
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_;
};

} // namespace lutmac
