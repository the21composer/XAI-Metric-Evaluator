from .faithfulness import Faithfulness
from .monotonicity import Monotonicity
from .infidelity import Infidelity

available_metrics = {
    "faithfulness": Faithfulness,
    "monotonicity": Monotonicity,
    "infidelity": Infidelity
}