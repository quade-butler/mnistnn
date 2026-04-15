#ifndef BASE_H
#define BASE_H

#include <cstdint> // std::uint8_t, uint32_t
#include <vector>  // std::vector, std::size_t
#include <span>
#include <string>
#include <stdexcept>

// Types
using Size	 = std::size_t;
using Pixel	 = std::uint8_t;
using String = std::string;

template<typename T>
using Vector = std::vector<T>;

template<typename T>
using Span = std::span<T>;


// Errors
using runtime_error = std::runtime_error;


#endif // BASE_H
