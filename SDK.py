"""Deep Agent SDK — host-side external library.

Place this file anywhere on your machine (notebook dir, ~/libs/, etc.).
Set the environment variable DEEP_AGENT_PROJECT to the absolute path of the
Docker project folder before importing:

    import os
    os.environ["DEEP_AGENT_PROJECT"] = "/path/to/Deep agent"
    from SDK import create_agent, load_agent, Agent

Tools and schema for each agent are defined by passing file paths at creation time:

    create_agent(
        name="my_agent",
        business_rules="rules.md",
        tools_file="my_tools.py",   # .py with @tool-decorated functions
        schema_file="my_schema.py", # .py with one Pydantic BaseModel subclass
    )
"""
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    root = os.environ.get("DEEP_AGENT_PROJECT")
    if not root:
        raise RuntimeError(
            "Set the DEEP_AGENT_PROJECT environment variable to the project directory. "
            "Example:  os.environ['DEEP_AGENT_PROJECT'] = '/path/to/Deep agent'"
        )
    return Path(root).resolve()


def _agents_dir() -> Path:
    return _project_root() / "agents"


def _output_dir() -> Path:
    return _project_root() / "output"


def _compose(*args: str) -> None:
    cmd = ["docker", "compose", "run", "--rm", "agent", *args]
    print("$", " ".join(shlex.quote(a) for a in cmd))
    proc = subprocess.run(cmd, cwd=_project_root(), stderr=subprocess.PIPE, text=True)
    if proc.stderr:
        print(proc.stderr, end="")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError(
            f"docker compose exited {proc.returncode}. Last stderr lines:\n"
            + "\n".join(tail)
        )


def _latest_complete_run_dir(name: str) -> Path:
    base = _output_dir() / name
    if not base.exists():
        raise FileNotFoundError(f"No runs for agent {name!r} in {base}")
    candidates = [
        p for p in base.iterdir()
        if p.is_dir() and (
            (p / "result.md").exists()
            or (p / "result.txt").exists()
            or (p / "structured.json").exists()
            or (p / "interrupt.json").exists()
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No completed run under {base} (missing result.md / result.txt / structured.json)"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


@dataclass
class RunResult:
    output: str
    output_dir: Path
    structured: dict | None = None
    interrupt: list[dict] | None = None
    thread_id: str | None = None


def _read_run_dir(latest: Path) -> RunResult:
    sj = latest / "structured.json"
    md_path = latest / "result.md"
    txt_path = latest / "result.txt"
    itr_path = latest / "interrupt.json"

    structured = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else None

    if md_path.exists():
        body = md_path.read_text(encoding="utf-8")
    elif txt_path.exists():
        body = txt_path.read_text(encoding="utf-8")
    elif structured is not None:
        body = (
            "# Agent output\n\n```json\n"
            + json.dumps(structured, ensure_ascii=False, indent=2)
            + "\n```\n"
        )
    else:
        raise FileNotFoundError(
            f"No result.md, result.txt, or structured.json in {latest}"
        )

    interrupt = None
    thread_id = None
    if itr_path.exists():
        data = json.loads(itr_path.read_text(encoding="utf-8"))
        interrupt = data.get("interrupts")
        thread_id = data.get("thread_id")

    return RunResult(
        output=body, output_dir=latest, structured=structured,
        interrupt=interrupt, thread_id=thread_id,
    )


@dataclass
class Agent:
    name: str

    @property
    def root(self) -> Path:
        return _agents_dir() / self.name

    def run(self, sample) -> RunResult:
        _compose("run", "--name", self.name, "--input", str(sample))
        return _read_run_dir(_latest_complete_run_dir(self.name))

    def resume(self, thread_id: str, decisions) -> RunResult:
        """Resume an interrupted HITL run.

        decisions: a dict like {"type": "approve"} or a list of such dicts.
        """
        if isinstance(decisions, dict):
            decisions = [decisions]
        payload = json.dumps(decisions, ensure_ascii=False)
        _compose(
            "resume", "--name", self.name,
            "--thread-id", thread_id,
            "--decisions", payload,
        )
        return _read_run_dir(_latest_complete_run_dir(self.name))

    def train(self, samples: list) -> None:
        _compose("train", "--name", self.name, "--samples", *(str(s) for s in samples))

    def test(self, samples: list) -> None:
        _compose("test", "--name", self.name, "--samples", *(str(s) for s in samples))


def create_agent(
    name: str,
    business_rules=None,
    process=None,
    tool_tips=None,
    tools_file: str | Path | None = None,
    schema_file: str | Path | None = None,
    overwrite: bool = False,
) -> Agent:
    """Create a new named agent.

    Args:
        name: agent name (used as directory under agents/).
        business_rules: path to a .md file or inline text.
        process: path to process.md or inline text.
        tool_tips: path to tool_tips.md or inline text.
        tools_file: path to a .py file containing @tool-decorated functions.
                    Copied to agents/<name>/custom_tools.py and loaded at runtime.
        schema_file: path to a .py file containing a Pydantic BaseModel subclass.
                     Copied to agents/<name>/custom_schema.py and loaded at runtime.
        overwrite: recreate the agent directory if it already exists.
    """
    # copy custom files into the agent dir before running create in the container
    agent_dir = _agents_dir() / name
    if agent_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Agent already exists: {agent_dir}. Pass overwrite=True to recreate."
            )
        shutil.rmtree(agent_dir)
    agent_dir.mkdir(parents=True)

    if tools_file is not None:
        shutil.copy2(Path(tools_file), agent_dir / "custom_tools.py")

    if schema_file is not None:
        shutil.copy2(Path(schema_file), agent_dir / "custom_schema.py")

    args = ["create", "--name", name]
    if business_rules is not None:
        args += ["--business-rules", str(business_rules)]
    if process is not None:
        args += ["--process", str(process)]
    if tool_tips is not None:
        args += ["--tool-tips", str(tool_tips)]
    args.append("--overwrite")  # dir already created above

    _compose(*args)
    return Agent(name=name)


def load_agent(name: str) -> Agent:
    agent_dir = _agents_dir() / name
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent not found: {agent_dir}")
    return Agent(name=name)


if __name__ == "__main__":
    # Quick smoke-test — adjust paths as needed
    agent = create_agent(
        name="charter_v1",
        business_rules="agent_init/buisness_rules.md",
        overwrite=True,
    )
    agent = load_agent("charter_v1")
    result = agent.run("/workspace/input/charter.pdf")
    print(result.output[:500])
