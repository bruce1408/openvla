#pragma once

#include <cstdint>
#include <vector>

namespace openvla
{

struct ActionMetadata
{
    std::int32_t effectiveVocabSize{};
    std::vector<float> binCenters;
    std::vector<float> q01;
    std::vector<float> q99;
    std::vector<std::uint8_t> mask;
};

class ActionDecoder
{
public:
    explicit ActionDecoder(ActionMetadata metadata);

    [[nodiscard]] std::vector<float> decode(
        std::vector<std::int32_t> const& tokenIds) const;

private:
    ActionMetadata metadata_;
};

}  // namespace openvla
