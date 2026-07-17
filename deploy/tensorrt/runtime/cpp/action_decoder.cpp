#include "action_decoder.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace openvla
{

ActionDecoder::ActionDecoder(ActionMetadata metadata)
    : metadata_(std::move(metadata))
{
    auto const actionDim = metadata_.q01.size();
    if (metadata_.effectiveVocabSize <= 0 || metadata_.binCenters.empty())
    {
        throw std::invalid_argument("Action metadata has an empty vocabulary or bin table");
    }
    if (metadata_.q99.size() != actionDim || metadata_.mask.size() != actionDim)
    {
        throw std::invalid_argument("q01, q99, and mask must have the same action dimension");
    }
}

std::vector<float> ActionDecoder::decode(
    std::vector<std::int32_t> const& tokenIds) const
{
    if (tokenIds.size() != metadata_.q01.size())
    {
        throw std::invalid_argument("Token count does not match the configured action dimension");
    }

    std::vector<float> actions(tokenIds.size());
    auto const maxBin = static_cast<std::int32_t>(metadata_.binCenters.size() - 1U);
    for (std::size_t index = 0; index < tokenIds.size(); ++index)
    {
        auto discrete = metadata_.effectiveVocabSize - tokenIds[index] - 1;
        discrete = std::clamp(discrete, 0, maxBin);
        auto const normalized = metadata_.binCenters[static_cast<std::size_t>(discrete)];
        if (metadata_.mask[index] != 0U)
        {
            actions[index] = 0.5F * (normalized + 1.0F)
                * (metadata_.q99[index] - metadata_.q01[index])
                + metadata_.q01[index];
        }
        else
        {
            actions[index] = normalized;
        }
    }
    return actions;
}

}  // namespace openvla
