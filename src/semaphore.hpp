#ifndef __SEMAPHORE_HPP
#define __SEMAPHORE_HPP

#include <mutex>
#include <condition_variable>

class semaphore {
public:
	std::mutex mutex_;
	std::condition_variable condition_;
	uint32_t count_ = 0; // Initialized as locked.

public:
	semaphore() = default;
	semaphore(uint32_t nCnt) : count_(nCnt) {}

	void unlock() {
		std::lock_guard<decltype(mutex_)> lock(mutex_);
		++count_;
		condition_.notify_one();
	}

	void lock() {
		std::unique_lock<decltype(mutex_)> lock(mutex_);
		while(!count_) // Handle spurious wake-ups.
			condition_.wait(lock);
		--count_;
	}

	void set_count(unsigned long nCnt) {
		std::lock_guard<decltype(mutex_)> lock(mutex_);
		count_ = nCnt;
	}
};

#endif