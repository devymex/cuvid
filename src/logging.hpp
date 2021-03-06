/*!
 *  Copyright (c) 2015 by Contributors
 * \file logging.h
 * \brief defines logging macros of dmlc
 *  allows use of GLOG, fall back to internal
 *  implementation when disabled
 */
#ifndef _LOGGING_HDR
#define _LOGGING_HDR
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <cstring>
#include <cmath>
#include <chrono>

#ifdef DMLC_LOG_STACK_TRACE
#include <execinfo.h>
#include <cxxabi.h>
#define DMLC_LOG_STACK_TRACE_SIZE 10
#endif

#define DMLC_THROW_EXCEPTION noexcept(false)
#define DMLC_LOG_FATAL_THROW 1

namespace dmlc {
/*!
 * \brief exception class that will be thrown by
 *  default logger if DMLC_LOG_FATAL_THROW == 1
 */
struct Error : public std::runtime_error {
  /*!
   * \brief constructor
   * \param s the error message
   */
  explicit Error(const std::string &s) : std::runtime_error(s) {}
};
}  // namespace dmlc

// #if DMLC_USE_GLOG
// #include <glog/logging.h>
//
// namespace dmlc {
// /*!
//  * \brief optionally redirect to google's init log
//  * \param argv0 The arguments.
//  */
// inline void InitLogging(const char* argv0) {
//   google::InitGoogleLogging(argv0);
// }
// }  // namespace dmlc
//
// #else
// use a light version of glog
#include <assert.h>
#include <iostream>
#include <sstream>
#include <ctime>

#if defined(_MSC_VER)
#pragma warning(disable : 4722)
#pragma warning(disable : 4068)
#endif

namespace dmlc {
inline void InitLogging(const char*) {
  // DO NOTHING
}

class LogCheckError {
 public:
  LogCheckError() : str(nullptr) {}
  explicit LogCheckError(const std::string& str_) : str(new std::string(str_)) {}
  LogCheckError(const LogCheckError& other) = delete;
  LogCheckError(LogCheckError&& other) : str(other.str) {
    other.str = nullptr;
  }
  ~LogCheckError() { if (str != nullptr) delete str; }
  operator bool() const { return str != nullptr; }
  LogCheckError& operator=(const LogCheckError& other) = delete;
  LogCheckError& operator=(LogCheckError&& other) = delete;
  std::string* str;
};

#ifndef DMLC_GLOG_DEFINED

#ifndef _LIBCPP_SGX_NO_IOSTREAMS
#define DEFINE_CHECK_FUNC(name, op)                               \
  template <typename X, typename Y>                               \
  inline LogCheckError LogCheck##name(const X& x, const Y& y) {   \
    if (x op y) return LogCheckError();                           \
    std::ostringstream os;                                        \
    os << " (" << x << " vs. " << y << ") ";  /* CHECK_XX(x, y) requires x and y can be serialized to string. Use CHECK(x OP y) otherwise. NOLINT(*) */ \
    return LogCheckError(os.str());                               \
  }                                                               \
  inline LogCheckError LogCheck##name(int x, int y) {             \
    return LogCheck##name<int, int>(x, y);                        \
  }
#else
#define DEFINE_CHECK_FUNC(name, op)                               \
  template <typename X, typename Y>                               \
  inline LogCheckError LogCheck##name(const X& x, const Y& y) {   \
    if (x op y) return LogCheckError();                           \
    return LogCheckError("Error.");                               \
  }                                                               \
  inline LogCheckError LogCheck##name(int x, int y) {             \
    return LogCheck##name<int, int>(x, y);                        \
  }
#endif

#define CHECK_BINARY_OP(name, op, x, y)                               \
  if (dmlc::LogCheckError _check_err = dmlc::LogCheck##name(x, y))    \
    dmlc::LogMessageFatal(__FILE__, __LINE__).stream()                \
      << "Check failed: " << #x " " #op " " #y << *(_check_err.str) << ": "

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
DEFINE_CHECK_FUNC(_LT, <)
DEFINE_CHECK_FUNC(_GT, >)
DEFINE_CHECK_FUNC(_LE, <=)
DEFINE_CHECK_FUNC(_GE, >=)
DEFINE_CHECK_FUNC(_EQ, ==)
DEFINE_CHECK_FUNC(_NE, !=)
#pragma GCC diagnostic pop

// Always-on checking
#define CHECK(x)                                           \
  if (!(x))                                                \
    dmlc::LogMessageFatal(__FILE__, __LINE__).stream()     \
      << "Check failed: " #x << ": "
#define CHECK_LT(x, y) CHECK_BINARY_OP(_LT, <, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(_GT, >, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(_LE, <=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(_GE, >=, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, x, y)
#define CHECK_NOTNULL(x) \
  ((x) == NULL ? dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check  notnull: "  #x << ' ', (x) : (x)) // NOLINT(*)
// Debug-only checking.
#ifdef NDEBUG
#define DCHECK(x) \
  while (false) CHECK(x)
#define DCHECK_LT(x, y) \
  while (false) CHECK((x) < (y))
#define DCHECK_GT(x, y) \
  while (false) CHECK((x) > (y))
#define DCHECK_LE(x, y) \
  while (false) CHECK((x) <= (y))
#define DCHECK_GE(x, y) \
  while (false) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) \
  while (false) CHECK((x) == (y))
#define DCHECK_NE(x, y) \
  while (false) CHECK((x) != (y))
#else
#define DCHECK(x) CHECK(x)
#define DCHECK_LT(x, y) CHECK((x) < (y))
#define DCHECK_GT(x, y) CHECK((x) > (y))
#define DCHECK_LE(x, y) CHECK((x) <= (y))
#define DCHECK_GE(x, y) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) CHECK((x) == (y))
#define DCHECK_NE(x, y) CHECK((x) != (y))
#endif  // NDEBUG

#if DMLC_LOG_CUSTOMIZE
#define GLOG_INFO dmlc::CustomLogMessage(__FILE__, __LINE__)
#else
#define GLOG_INFO dmlc::LogMessage(__FILE__, __LINE__)
#endif
#define GLOG_ERROR GLOG_INFO
#define GLOG_WARNING GLOG_INFO
#define GLOG_FATAL dmlc::LogMessageFatal(__FILE__, __LINE__)
#define GLOG_QFATAL GLOG_FATAL

// Poor man version of VLOG
#define VLOG(x) GLOG_INFO.stream()

#define LOG(severity) GLOG_##severity.stream()
#define LG GLOG_INFO.stream()
#define LOG_IF(severity, condition) \
  !(condition) ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)

#ifdef NDEBUG
#define LOG_DFATAL GLOG_ERROR
#define DFATAL ERROR
#define DLOG(severity) true ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)
#define DLOG_IF(severity, condition) \
  (true || !(condition)) ? (void)0 : dmlc::LogMessageVoidify() & LOG(severity)
#else
#define LOG_DFATAL GLOG_FATAL
#define DFATAL FATAL
#define DLOG(severity) LOG(severity)
#define DLOG_IF(severity, condition) LOG_IF(severity, condition)
#endif

// Poor man version of LOG_EVERY_N
#define LOG_EVERY_N(severity, n) LOG(severity)

#endif  // DMLC_GLOG_DEFINED

class DateLogger {
 public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char* HumanDate() {
	auto now = std::chrono::system_clock::now();
	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
			now.time_since_epoch()).count();
	auto dateTime = std::chrono::system_clock::to_time_t(now);
	struct tm local;
#if defined(_WIN32)
	localtime_s(&local, &dateTime);
#else
	localtime_r(&dateTime, &local);
#endif
	snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d.%03d", local.tm_hour,
			local.tm_min, local.tm_sec, (int)(millis % 1000));
	return buffer_;
  }

 private:
  char buffer_[64];
};

#ifndef _LIBCPP_SGX_NO_IOSTREAMS
class LogMessage {
 public:
  LogMessage(const char* file, int line)
      :
#ifdef __ANDROID__
        log_stream_(std::cout)
#else
        log_stream_(std::cerr)
#endif
  {
#if !defined(_WIN32)
	char cSlash = '/';
#else
	char cSlash = '\\';
#endif
  	const char *pFn = strrchr(file, cSlash);
  	if (pFn != nullptr) {
  		file = pFn + 1;
  	}
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  ~LogMessage() { log_stream_ << '\n'; }
  std::ostream& stream() { return log_stream_; }

 protected:
  std::ostream& log_stream_;

 private:
  DateLogger pretty_date_;
  LogMessage(const LogMessage&);
  void operator=(const LogMessage&);
};

// customized logger that can allow user to define where to log the message.
class CustomLogMessage {
 public:
  CustomLogMessage(const char* file, int line) {
    log_stream_ << "[" << DateLogger().HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  ~CustomLogMessage() {
    Log(log_stream_.str());
  }
  std::ostream& stream() { return log_stream_; }
  /*!
   * \brief customized logging of the message.
   * This function won't be implemented by libdmlc
   * \param msg The message to be logged.
   */
  static void Log(const std::string& msg);

 private:
  std::ostringstream log_stream_;
};
#else
class DummyOStream {
 public:
  template <typename T>
  DummyOStream& operator<<(T _) { return *this; }
  inline std::string str() { return ""; }
};
class LogMessage {
 public:
  LogMessage(const char* file, int line) : log_stream_() {}
  DummyOStream& stream() { return log_stream_; }

 protected:
  DummyOStream log_stream_;

 private:
  LogMessage(const LogMessage&);
  void operator=(const LogMessage&);
};
#endif



#if DMLC_LOG_STACK_TRACE
inline std::string Demangle(char const *msg_str) {
  using std::string;
  string msg(msg_str);
  size_t symbol_start = string::npos;
  size_t symbol_end = string::npos;
  if ( ((symbol_start = msg.find("_Z")) != string::npos)
       && (symbol_end = msg.find_first_of(" +", symbol_start)) ) {
    string left_of_symbol(msg, 0, symbol_start);
    string symbol(msg, symbol_start, symbol_end - symbol_start);
    string right_of_symbol(msg, symbol_end);

    int status = 0;
    size_t length = string::npos;
    std::unique_ptr<char, void (*)(void *__ptr)> demangled_symbol =
        {abi::__cxa_demangle(symbol.c_str(), 0, &length, &status), &std::free};
    if (demangled_symbol && status == 0 && length > 0) {
      string symbol_str(demangled_symbol.get());
      std::ostringstream os;
      os << left_of_symbol << symbol_str << right_of_symbol;
      return os.str();
    }
  }
  return string(msg_str);
}

// By default skip the first frame because
// that belongs to ~LogMessageFatal
inline std::string StackTrace(
    size_t start_frame = 1,
    const size_t stack_size = DMLC_LOG_STACK_TRACE_SIZE) {
  using std::string;
  std::ostringstream stacktrace_os;
  std::vector<void*> stack(stack_size);
  int nframes = backtrace(stack.data(), static_cast<int>(stack_size));
  stacktrace_os << "Stack trace:\n";
  char **msgs = backtrace_symbols(stack.data(), nframes);
  if (msgs != nullptr) {
    for (int frameno = start_frame; frameno < nframes; ++frameno) {
      string msg = dmlc::Demangle(msgs[frameno]);
      stacktrace_os << "  [bt] (" << frameno - start_frame << ") " << msg << "\n";
    }
  }
  free(msgs);
  string stack_trace = stacktrace_os.str();
  return stack_trace;
}

#else  // DMLC_LOG_STACK_TRACE is off

inline std::string demangle(char const* msg_str) {
  return std::string();
}

inline std::string StackTrace(size_t start_frame = 1,
                              const size_t stack_size = 0) {
  return std::string("Stack trace not available when "
  "DMLC_LOG_STACK_TRACE is disabled at compile time.");
}

#endif  // DMLC_LOG_STACK_TRACE

#if defined(_LIBCPP_SGX_NO_IOSTREAMS)
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) : LogMessage(file, line) {}
  ~LogMessageFatal() {
    abort();
  }
 private:
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#elif DMLC_LOG_FATAL_THROW == 0
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) : LogMessage(file, line) {}
  ~LogMessageFatal() {
    log_stream_ << "\n" << StackTrace() << "\n";
    abort();
  }

 private:
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#else
class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  std::ostringstream &stream() { return log_stream_; }
  ~LogMessageFatal() DMLC_THROW_EXCEPTION {
#if DMLC_LOG_STACK_TRACE
    log_stream_ << "\n" << StackTrace() << "\n";
#endif

    // throwing out of destructor is evil
    // hopefully we can do it here
    // also log the message before throw
#if DMLC_LOG_BEFORE_THROW
    LOG(ERROR) << log_stream_.str();
#endif
    throw Error(log_stream_.str());
  }

 private:
  std::ostringstream log_stream_;
  DateLogger pretty_date_;
  LogMessageFatal(const LogMessageFatal&);
  void operator=(const LogMessageFatal&);
};
#endif

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
#if !defined(_LIBCPP_SGX_NO_IOSTREAMS)
  void operator&(std::ostream&) {}
#endif
};

}  // namespace dmlc

inline int64_t GetNowTimeSec() {
	return std::chrono::duration_cast<std::chrono::seconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
}

inline int64_t GetNowTimeMS() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
}

inline int64_t GetNowTimeUS() {
	return std::chrono::duration_cast<std::chrono::microseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
}

inline int64_t GetNowTimeNS() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
}

#endif
