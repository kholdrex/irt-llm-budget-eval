"""Command-line entry point: irteval <command> [options]."""
import argparse

from . import analysis, budget, prepare, report, rq1, rq2

DATASETS = ("openllm", "embedllm")


def _dataset_name(value):
    if value not in DATASETS:
        raise argparse.ArgumentTypeError(f"unknown data set {value!r}; choose from {', '.join(DATASETS)}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog="irteval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="download the response data and build the matrices")
    p.add_argument("datasets", nargs="*", type=_dataset_name, help="openllm and/or embedllm (default: both)")
    for name, helptext in (("rq2", "held-out response prediction and hyperparameter selection"),
                           ("rq1", "item difficulty analyses (run after rq2)")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("dataset", choices=["d1", "d2"])
    p = sub.add_parser("budget", help="reduced-budget evaluation of new models (run after rq2)")
    p.add_argument("dataset", choices=["d1", "d2"])
    p.add_argument("--scheme", choices=["grouped", "random"], default="grouped")
    p.add_argument("--folds", type=int, nargs="*", choices=range(5), help="subset of the five folds")
    sub.add_parser("analyze", help="aggregate results into CSV tables and figures")
    sub.add_parser("report", help="paper tables and the signed-error figure (run after analyze)")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare.main(args.datasets or DATASETS)
    elif args.command == "rq2":
        rq2.main(args.dataset)
    elif args.command == "rq1":
        rq1.main(args.dataset)
    elif args.command == "budget":
        budget.main(args.dataset, args.scheme, args.folds)
    elif args.command == "analyze":
        analysis.main()
    elif args.command == "report":
        report.main()


if __name__ == "__main__":
    main()
