from .synthetic_gaussian import GaussianLinearRegression, GaussianNonlinearAdditiveRegression, GaussianPiecewiseConstantRegression, GaussianLinearBinary, GaussianNonlinearAdditiveBinary, GaussianPiecewiseConstantBinary
from .synthetic_mixture import GMLinearRegression, GMNonlinearAdditiveRegression, GMPiecewiseConstantRegression
from .custom_dataset import CustomDataset
from .synthetic_multinomial import MultinomialLinearRegression

available_datasets = {
    "regression": {
        "gaussianLinear": GaussianLinearRegression,
        "gaussianNonLinearAdditive": GaussianNonlinearAdditiveRegression,
        "gaussianPiecewiseConstant": GaussianPiecewiseConstantRegression,
        "mixtureLinear": GMLinearRegression,
        "mixtureNonLinearAdditive": GMNonlinearAdditiveRegression,
        "mixturePiecewiseConstant": GMPiecewiseConstantRegression
    },
    "classification": {
        "gaussianLinear": GaussianLinearBinary,
        "gaussianNonLinearAdditive": GaussianNonlinearAdditiveBinary,
        "gaussianPiecewiseConstant": GaussianPiecewiseConstantBinary,
    },
}