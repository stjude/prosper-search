import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "figures"))
from fig5_ot_correspondence_absolute_candidate import fixed_display_values


def test_same_cost_has_same_color_position_across_pairs():
    first = fixed_display_values([0.1, 0.4, 0.8])
    second = fixed_display_values([0.3, 0.4, 0.9])
    assert first[1] == second[1] == 0.6
    np.testing.assert_allclose(fixed_display_values([0, 1]), [1, 0])


@pytest.mark.parametrize("cost", [[-0.01], [1.01], [float("nan")]])
def test_values_outside_display_contract_are_not_silently_clipped(cost):
    with pytest.raises(ValueError, match="without clipping"):
        fixed_display_values(cost)
