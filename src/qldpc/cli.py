"""Command-line interface for building and benchmarking qLDPC codes."""

from __future__ import annotations

import argparse
import json

from .experiments import run_sweep
from .registry import available_codes, build_code
from .simulation import code_capacity_ler
from .types import SweepConfig


def _cmd_list(_: argparse.Namespace) -> None:
    print("preset codes:", ", ".join(available_codes()))
    print("toric codes:  toric:<distance>  (e.g. toric:3, toric:5)")


def _cmd_info(args: argparse.Namespace) -> None:
    code = build_code(args.code)
    print(code.summary())


def _cmd_decode(args: argparse.Namespace) -> None:
    code = build_code(args.code)
    estimate = code_capacity_ler(
        code, p=args.p, shots=args.shots, seed=args.seed, backend=args.backend
    )
    print(f"{code.summary()}  p={args.p}")
    print(
        f"logical error rate = {estimate.logical_error_rate:.4e} "
        f"[{estimate.ci_low:.4e}, {estimate.ci_high:.4e}]  ({args.shots} shots)"
    )


def _cmd_sweep(args: argparse.Namespace) -> None:
    config = SweepConfig(
        codes=args.codes.split(","),
        error_rates=[float(x) for x in args.p.split(",")],
        shots=args.shots,
        seed=args.seed,
        backend=args.backend,
    )
    result = run_sweep(config)
    payload = json.loads(result.model_dump_json())
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.output}")
    else:
        print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="qLDPC code builder and BP+OSD benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available code families").set_defaults(func=_cmd_list)

    info = sub.add_parser("info", help="print [[n, k]] for a code")
    info.add_argument("code")
    info.set_defaults(func=_cmd_info)

    decode = sub.add_parser("decode", help="estimate the code-capacity logical error rate")
    decode.add_argument("code")
    decode.add_argument("--p", type=float, default=0.03)
    decode.add_argument("--shots", type=int, default=2000)
    decode.add_argument("--seed", type=int, default=2026)
    decode.add_argument("--backend", default="auto", choices=["auto", "scratch", "ldpc"])
    decode.set_defaults(func=_cmd_decode)

    sweep = sub.add_parser("sweep", help="sweep codes and physical error rates")
    sweep.add_argument("--codes", default="bb72,toric:5")
    sweep.add_argument("--p", default="0.02,0.03,0.04")
    sweep.add_argument("--shots", type=int, default=2000)
    sweep.add_argument("--seed", type=int, default=2026)
    sweep.add_argument("--backend", default="auto", choices=["auto", "scratch", "ldpc"])
    sweep.add_argument("--output", default="")
    sweep.set_defaults(func=_cmd_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
