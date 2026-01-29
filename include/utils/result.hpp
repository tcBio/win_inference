#pragma once

#include <string>
#include <variant>
#include <optional>
#include <stdexcept>

namespace qwen {

/// Error information for failed operations
struct Error {
    std::string code;
    std::string message;
    std::string details;

    Error() = default;
    Error(std::string c, std::string m, std::string d = "")
        : code(std::move(c)), message(std::move(m)), details(std::move(d)) {}

    static Error internal(const std::string& msg) {
        return Error{"internal_error", msg};
    }

    static Error invalid_request(const std::string& msg) {
        return Error{"invalid_request_error", msg};
    }

    static Error model_error(const std::string& msg) {
        return Error{"model_error", msg};
    }

    static Error resource_exhausted(const std::string& msg) {
        return Error{"resource_exhausted", msg};
    }

    static Error timeout(const std::string& msg) {
        return Error{"timeout", msg};
    }

    static Error cancelled(const std::string& msg) {
        return Error{"cancelled", msg};
    }
};

/// Result type for operations that can fail
template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Error error) : data_(std::move(error)) {}

    [[nodiscard]] bool ok() const { return std::holds_alternative<T>(data_); }
    [[nodiscard]] bool is_error() const { return std::holds_alternative<Error>(data_); }

    [[nodiscard]] const T& value() const& {
        if (is_error()) {
            throw std::runtime_error("Result::value() called on error: " + error().message);
        }
        return std::get<T>(data_);
    }

    [[nodiscard]] T& value() & {
        if (is_error()) {
            throw std::runtime_error("Result::value() called on error: " + error().message);
        }
        return std::get<T>(data_);
    }

    [[nodiscard]] T value() && {
        if (is_error()) {
            throw std::runtime_error("Result::value() called on error: " + error().message);
        }
        return std::move(std::get<T>(data_));
    }

    [[nodiscard]] const Error& error() const& {
        if (ok()) {
            throw std::runtime_error("Result::error() called on success");
        }
        return std::get<Error>(data_);
    }

    [[nodiscard]] T value_or(T default_value) const {
        return ok() ? value() : std::move(default_value);
    }

    template <typename Func>
    auto map(Func&& func) const -> Result<decltype(func(std::declval<T>()))> {
        if (ok()) {
            return func(value());
        }
        return error();
    }

    template <typename Func>
    auto and_then(Func&& func) const -> decltype(func(std::declval<T>())) {
        if (ok()) {
            return func(value());
        }
        return error();
    }

private:
    std::variant<T, Error> data_;
};

/// Specialization for void results
template <>
class Result<void> {
public:
    Result() : error_(std::nullopt) {}
    Result(Error error) : error_(std::move(error)) {}

    [[nodiscard]] bool ok() const { return !error_.has_value(); }
    [[nodiscard]] bool is_error() const { return error_.has_value(); }

    void value() const {
        if (is_error()) {
            throw std::runtime_error("Result::value() called on error: " + error().message);
        }
    }

    [[nodiscard]] const Error& error() const {
        if (!is_error()) {
            throw std::runtime_error("Result::error() called on success");
        }
        return *error_;
    }

    static Result success() { return Result(); }

private:
    std::optional<Error> error_;
};

// Convenience type alias
using VoidResult = Result<void>;

}  // namespace qwen
