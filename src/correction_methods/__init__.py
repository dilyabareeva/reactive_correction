from src.correction_methods.base_correction_method import Vanilla
from src.correction_methods.clarc.clarc import (
    PClarcFullFeature,
    RClarcFullFeature,
    ACRClarcFullFeature,
    ACCRClarcFullFeature,
)


def get_correction_method(method_name):
    CORRECTION_METHODS = {
        "vanilla": Vanilla,
        "p-clarc": PClarcFullFeature,
        "r-clarc": RClarcFullFeature,
        "acr-clarc": ACRClarcFullFeature,
        "accr-clarc": ACCRClarcFullFeature,
    }

    assert (
        method_name in CORRECTION_METHODS.keys()
    ), f"Correction method '{method_name}' unknown, choose one of {list(CORRECTION_METHODS.keys())}"
    return CORRECTION_METHODS[method_name]
