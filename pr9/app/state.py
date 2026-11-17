from dataclasses import dataclass, field


@dataclass
class Step:
    n: int
    name: str
    kind: str
    status: str
    detail: str | None = None
    input: dict | list | str | None = None
    output: dict | list | str | None = None
    elapsed: float | None = None
    usage: dict | None = None
    at: str | None = None


@dataclass
class Run:
    run_id: str
    customer_id: str
    text: str
    status: str = "running"
    route: str | None = None
    steps: list[Step] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    pending: dict | None = None
    decision: dict | None = None
    reply: str | None = None
    stopped: str | None = None
    usage: dict = field(default_factory=dict)
    elapsed: dict = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None