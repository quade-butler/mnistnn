#ifndef BASE_H
#define BASE_H

#include <cstdint> // std::uint8_t, uint32_t
#include <vector>  // std::vector, std::size_t
#include <span>

namespace mnistnn
{
    // Types
    using Size	 = std::size_t;
    using Pixel	 = std::uint8_t;
    using Label  = std::uint8_t;

    template<typename T>
    using Vector = std::vector<T>;

    template<typename T>
    using Span = std::span<T>;

} // end namespace mnistnn
#endif // BASE_H
