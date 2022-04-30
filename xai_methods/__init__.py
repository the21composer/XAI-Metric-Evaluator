from .xai_lime import LimeXAI
from .xai_shap import ShapXAI, KernelShap

available_methods = {
    "shap": ShapXAI,
    "kernelshap": KernelShap,
    "lime": LimeXAI,
}