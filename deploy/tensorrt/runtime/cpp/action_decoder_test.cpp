#include "action_decoder.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

int main()
{
    openvla::ActionMetadata metadata;
    metadata.effectiveVocabSize = 32000;
    metadata.binCenters = {-0.5F, 0.0F, 0.5F};
    metadata.q01 = {-2.0F, -1.0F};
    metadata.q99 = {2.0F, 1.0F};
    metadata.mask = {1U, 0U};

    openvla::ActionDecoder decoder(std::move(metadata));
    auto const action = decoder.decode(std::vector<std::int32_t>{31999, 31997});
    assert(action.size() == 2U);
    assert(std::abs(action[0] - (-1.0F)) < 1.0e-6F);
    assert(std::abs(action[1] - 0.5F) < 1.0e-6F);
    return 0;
}
