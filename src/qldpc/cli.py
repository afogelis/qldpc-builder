"""Command-line interface for building and benchmarking qLDPC codes."""

from __future__ import annotations

import argparse
import json

from .experiments import run_sweep
from .registry import available_codes, build_code
from .simulation import code_capacity_ler, phenomenological_ler
from .types import SweepConfig


def _cmd_list(_: argparse.Namespace) -> None:
    print("preset codes:", ", ".join(available_codes()))
    print("toric codes:  toric:<distance>  (e.g. toric:3, toric:5)")


def _cmd_info(args: argparse.Namespace) -> None:
    code = build_code(args.code)
    print(code.summary())
    stats = code.connectivity()
    print(
        "connectivity:",
        f"max check weight X/Z = {stats.max_x_check_weight}/{stats.max_z_check_weight},",
        f"max qubit degree = {stats.max_qubit_degree}",
    )
    report = code.distance(search_trials=args.search_trials, seed=args.seed)
    print("distance:", report.summary())


def _cmd_decode(args: argparse.Namespace) -> None:
    code = build_code(args.code)
    if args.noise == "phenomenological":
        estimate = phenomenological_ler(
            code,
            px=args.px,
            pz=args.pz,
            shots=args.shots,
            seed=args.seed,
            backend=args.backend,
        )
        noise_text = f"px={args.px}, pz={args.pz}"
    else:
        estimate = code_capacity_ler(
            code, p=args.p, shots=args.shots, seed=args.seed, backend=args.backend
        )
        noise_text = f"p={args.p}"
    print(f"{code.summary()}  {noise_text}")
    print(
        f"logical error rate = {estimate.logical_error_rate:.4e} "
        f"[{estimate.ci_low:.4e}, {estimate.ci_high:.4e}]  ({args.shots} shots)"
    )


def _cmd_export(args: argparse.Namespace) -> None:
    code = build_code(args.code)
    target = code.export_matrices(
        args.output,
        include_stim=args.stim,
        px=args.px,
        pz=args.pz,
        stim_mode=args.stim_mode,
    )
    print(f"exported {code.name} to {target}")


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

    info = sub.add_parser("info", help="print [[n, k, d]] and connectivity for a code")
    info.add_argument("code")
    info.add_argument("--search-trials", type=int, default=4000)
    info.add_argument("--seed", type=int, default=2026)
    info.set_defaults(func=_cmd_info)

    decode = sub.add_parser("decode", help="estimate the logical error rate")
    decode.add_argument("code")
    decode.add_argument("--noise", choices=["code_capacity", "phenomenological"], default="code_capacity")
    decode.add_argument("--p", type=float, default=0.03, help="X error rate for code-capacity mode")
    decode.add_argument("--px", type=float, default=0.03, help="X rate for phenomenological mode")
    decode.add_argument("--pz", type=float, default=0.001, help="Z rate for phenomenological mode")
    decode.add_argument("--shots", type=int, default=2000)
    decode.add_argument("--seed", type=int, default=2026)
    decode.add_argument("--backend", default="auto", choices=["auto", "scratch", "ldpc"])
    decode.set_defaults(func=_cmd_decode)

    export = sub.add_parser("export", help="write parity-check matrices for decoder-benchmark")
    export.add_argument("code")
    export.add_argument("--output", default="artifacts/export")
    export.add_argument("--stim", action="store_true", help="also write syndrome.stim")
    export.add_argument("--px", type=float, default=0.03, help="X error rate embedded in syndrome.stim")
    export.add_argument("--stim-mode", default="x_only", choices=["x_only"])
    export.set_defaults(func=_cmd_export)

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
