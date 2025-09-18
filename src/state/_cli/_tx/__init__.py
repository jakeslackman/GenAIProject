import argparse as ap

from ._infer import add_arguments_infer, run_tx_infer
from ._infer_with_hooks import add_arguments_infer_with_hooks, run_tx_infer as run_tx_infer_with_hooks
from ._predict import add_arguments_predict, run_tx_predict
from ._predict_with_hooks import add_arguments_predict as add_arguments_predict_with_hooks, run_tx_predict as run_tx_predict_with_hooks
from ._predict_heat_map import add_arguments_predict as add_arguments_predict_heat_map, run_tx_predict as run_tx_predict_heat_map
from ._preprocess_infer import add_arguments_preprocess_infer, run_tx_preprocess_infer
from ._preprocess_train import add_arguments_preprocess_train, run_tx_preprocess_train
from ._test import add_arguments_predict as add_arguments_test, run_tx_predict as run_tx_test
from ._train import add_arguments_train, run_tx_train

__all__ = [
    "run_tx_train",
    "run_tx_predict",
    "run_tx_predict_with_hooks",
    "run_tx_predict_heat_map",
    "run_tx_test",
    "run_tx_infer",
    "run_tx_infer_with_hooks",
    "run_tx_preprocess_train",
    "run_tx_preprocess_infer",
    "add_arguments_tx",
]


def add_arguments_tx(parser: ap.ArgumentParser):
    """"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    add_arguments_train(subparsers.add_parser("train", add_help=False))
    add_arguments_predict(subparsers.add_parser("predict"))
    add_arguments_predict_with_hooks(subparsers.add_parser("predict_with_hooks"))
    add_arguments_predict_heat_map(subparsers.add_parser("predict_heat_map"))
    add_arguments_test(subparsers.add_parser("test"))
    add_arguments_infer(subparsers.add_parser("infer"))
    add_arguments_infer_with_hooks(subparsers.add_parser("infer_with_hooks"))
    add_arguments_preprocess_train(subparsers.add_parser("preprocess_train"))
    add_arguments_preprocess_infer(subparsers.add_parser("preprocess_infer"))
