"""Static wiring checks for the OrbitApp controllers (no Panda3D needed).

A reference to a controller method that does not exist -- or a call through the app
to a method that moved into a controller -- only fails at runtime, possibly on a code
path no scenario reaches (an error branch, a rarely pressed button). These tests read
the source instead.
"""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
RENDER = ROOT / "orbitsim" / "render"
CONTROLLERS = RENDER / "controllers"


def _controller_members():
    """controller attribute name on OrbitApp -> names defined on that controller class."""
    app_src = (RENDER / "app.py").read_text(encoding="utf-8")
    attr_of = {cls: attr for attr, cls in re.findall(r"self\.(\w+) = (\w+)\(self\)", app_src)}
    members = {}
    for path in CONTROLLERS.glob("*.py"):
        for cls in [n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.ClassDef)]:
            names = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
            names |= {t.id for m in cls.body if isinstance(m, ast.Assign)
                      for t in m.targets if isinstance(t, ast.Name)}
            members[attr_of[cls.name]] = names
    return members


def _app_methods():
    tree = ast.parse((RENDER / "app.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OrbitApp")
    return {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}


def _sources():
    files = [RENDER / "app.py", *CONTROLLERS.glob("*.py"), *(ROOT / "tests").rglob("*.py")]
    return [(p, p.read_text(encoding="utf-8")) for p in files]


def test_every_controller_reference_resolves():
    members = _controller_members()
    assert members, "no controllers found"
    pat = re.compile(r"\.(" + "|".join(map(re.escape, members)) + r")\.([A-Za-z_]\w*)")
    unresolved = [
        f"{path.relative_to(ROOT)}:{i}: {attr}.{name}"
        for path, text in _sources()
        for i, line in enumerate(text.splitlines(), 1)
        for attr, name in pat.findall(line)
        if name not in members[attr]
    ]
    assert not unresolved, "\n".join(unresolved)


def test_no_call_through_the_app_to_a_moved_method():
    moved = set().union(*_controller_members().values()) - _app_methods() - {"__init__"}
    pat = re.compile(r"self\.app\.(" + "|".join(map(re.escape, sorted(moved))) + r")\(")
    stale = [
        f"{path.relative_to(ROOT)}:{i}: {line.strip()}"
        for path, text in _sources()
        for i, line in enumerate(text.splitlines(), 1)
        if pat.search(line)
    ]
    assert not stale, "\n".join(stale)
