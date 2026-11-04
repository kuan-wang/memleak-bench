#!/usr/bin/env python3
"""Run the pinned Letta server (letta==0.16.8) on a local, isolated Postgres.

Letta 0.16.8's server is Postgres-only and its schema is created by the
alembic migrations in the source tree (not shipped in the wheel). This script
mirrors the official Docker startup (`letta/server/startup.sh` + `init.sql`):

  1. Start Postgres 16 + pgvector from the `pgserver` wheel on 127.0.0.1.
  2. Create the `letta` database/schema with the vector extension.
  3. Run `alembic upgrade head` from the letta 0.16.8 source checkout.
  4. exec `letta server`.

Keys are read from the repo .env and passed only through the child process
environment; nothing is printed. Host-injected ANTHROPIC_*/OPENAI_* variables
(e.g. a desktop-app ANTHROPIC_BASE_URL) are dropped.

Usage (from the repo root, with the dedicated server venv):
    .venv-letta-server/bin/python scripts/text_comparison/launch_letta_server.py \
        --state-dir <dir> [--pg-port 5544] [--port 8283] [--no-anthropic]
    .venv-letta-server/bin/python scripts/text_comparison/launch_letta_server.py \
        --state-dir <dir> --stop-db
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LETTA_SRC = ROOT / ".cache" / "letta-0.16.8-src"  # git tag 0.16.8, commit 1131535716e8
LETTA_BIN = ROOT / ".venv-letta-server" / "bin" / "letta"
ALEMBIC_BIN = ROOT / ".venv-letta-server" / "bin" / "alembic"
PASS_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def dotenv() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def pg_bin(name: str) -> str:
    import pgserver  # noqa: F401  (only to locate the bundled binaries)
    return str(Path(pgserver.__file__).parent / "pginstall" / "bin" / name)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def psql(port: int, db: str, sql: str) -> str:
    return run([pg_bin("psql"), "-h", "127.0.0.1", "-p", str(port), "-U", "postgres",
                "-d", db, "-v", "ON_ERROR_STOP=1", "-tAc", sql]).stdout.strip()


def start_db(state: Path, port: int) -> None:
    pgdata = state / "pgdata"
    if not (pgdata / "PG_VERSION").exists():
        pgdata.mkdir(parents=True, exist_ok=True)
        run([pg_bin("initdb"), "-D", str(pgdata), "--auth=trust", "--auth-local=trust",
             "--encoding=utf8", "-U", "postgres"])
    status = subprocess.run([pg_bin("pg_ctl"), "-D", str(pgdata), "status"],
                            capture_output=True, text=True)
    if status.returncode != 0:
        run([pg_bin("pg_ctl"), "-D", str(pgdata), "-w", "-l", str(state / "postgres.log"),
             "-o", f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories=''",
             "start"])
    if psql(port, "postgres", "SELECT 1 FROM pg_database WHERE datname='letta'") != "1":
        psql(port, "postgres", "CREATE DATABASE letta")
        # Equivalent of the official init.sql with db_name=db_user-owner postgres.
        for sql in ("CREATE SCHEMA letta", "ALTER DATABASE letta SET search_path TO letta",
                    "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA letta",
                    "DROP SCHEMA IF EXISTS public CASCADE"):
            psql(port, "letta", sql)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--pg-port", type=int, default=5544)
    parser.add_argument("--port", type=int, default=8283)
    parser.add_argument("--no-anthropic", action="store_true",
                        help="Do not pass an Anthropic key (development with OpenAI models only)")
    parser.add_argument("--stop-db", action="store_true")
    args = parser.parse_args()
    state = args.state_dir.resolve()

    if args.stop_db:
        subprocess.run([pg_bin("pg_ctl"), "-D", str(state / "pgdata"), "-m", "fast", "stop"])
        return

    state.mkdir(parents=True, exist_ok=True)
    start_db(state, args.pg_port)

    keys = dotenv()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "OPENAI_", "CLAUDE", "LETTA_"))}
    for name in PASS_KEYS:
        if name == "ANTHROPIC_API_KEY" and args.no_anthropic:
            continue
        if keys.get(name):
            env[name] = keys[name]
    env.update(
        LETTA_DIR=str(state / "letta-home"),
        LETTA_PG_URI=f"postgresql://postgres@127.0.0.1:{args.pg_port}/letta",
        LETTA_DISABLE_TRACING="true",
        LETTA_TELEMETRY_ENABLE_DATADOG="false",
    )
    migrate = subprocess.run([str(ALEMBIC_BIN), "upgrade", "head"], cwd=LETTA_SRC, env=env,
                             capture_output=True, text=True)
    (state / "alembic.log").write_text(migrate.stdout + migrate.stderr)
    if migrate.returncode != 0:
        sys.exit(f"alembic upgrade failed; see {state / 'alembic.log'}")
    os.chdir(state)  # the server writes openapi_letta.json into its working directory
    os.execve(str(LETTA_BIN), [str(LETTA_BIN), "server", "--host", "127.0.0.1",
                               "--port", str(args.port)], env)


if __name__ == "__main__":
    main()
