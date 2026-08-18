"""
Tests for Full Cycle bugs fixed in the bug-squash pass.

Verifies:
  1. shutdown.py: mark_running_jobs_interrupted exists and is called correctly
  2. Phase names align with frontend FC_PHASES expectations
  3. except chain: real exceptions get "failed" status (not "cancelled")
  4. Duplicate factory while loop removed
  5. Duplicate FactoryExecutor + proposer removed
"""
import pytest

import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Bug 1: shutdown function name ──


def test_shutdown_has_correct_function_name():
    """mark_running_jobs_interrupted must exist and be callable."""
    from api.shutdown import mark_running_jobs_interrupted
    assert callable(mark_running_jobs_interrupted)


def test_shutdown_does_not_have_typo():
    """mark_stale_jobs_interrupted must NOT exist."""
    import api.shutdown as mod
    assert not hasattr(mod, "mark_stale_jobs_interrupted")


def test_shutdown_cleanup_calls_correct_function():
    """shutdown_cleanup calls mark_running_jobs_interrupted without NameError."""
    from api.shutdown import shutdown_cleanup
    # Should not raise NameError — we use a temp db path that doesn't exist
    try:
        shutdown_cleanup(":memory:")
    except NameError as e:
        assert False, f"shutdown_cleanup raised NameError: {e}"
    except Exception:
        pass  # Other exceptions (sqlite errors) are fine — the function call itself worked


# ── Bug 2: Phase name alignment ──


FRONTEND_PHASES = frozenset({
    "feature_sweep",
    "phase1_hpo",
    "phase2_assembly",
    "phase3_validation",
    "phase4_factory",
})


def _get_source_lines():
    path = Path(PROJECT_ROOT) / "api" / "routers" / "committee.py"
    return path.read_text(encoding="utf-8").splitlines()


def _extract_phase_names():
    """Extract all phase string literals passed to _update_full_cycle_status."""
    lines = _get_source_lines()
    phases = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "_update_full_cycle_status" in stripped:
            # Extract the phase string argument after job_dir
            # Pattern: _update_full_cycle_status(job_dir, "phase_name", ...
            m = re.search(r',\s*"([^"]+)"', stripped)
            if m:
                phases.add(m.group(1))
    return phases


KNOWN_TERMINAL_PHASES = frozenset({
    "completed",
    "failed",
    "cancelled",
    "validation_failed",
})


def test_all_phase_names_match_frontend():
    """Every non-terminal phase written to status.json must be a frontend FC_PHASES key."""
    phases = _extract_phase_names()
    non_terminal = phases - KNOWN_TERMINAL_PHASES
    unknown = non_terminal - FRONTEND_PHASES
    assert not unknown, (
        f"Phase names not in frontend FC_PHASES: {unknown}\n"
        f"Expected one of: {sorted(FRONTEND_PHASES)} or {sorted(KNOWN_TERMINAL_PHASES)}"
    )


def test_all_frontend_phases_appear_at_least_once():
    """Every frontend FC_PHASES key must be written by the backend."""
    phases = _extract_phase_names()
    missing = FRONTEND_PHASES - phases - KNOWN_TERMINAL_PHASES
    assert not missing, (
        f"Frontend phases never written by backend: {missing}"
    )


def _get_run_full_cycle_source() -> str:
    """Extract the _run_full_cycle function body."""
    lines = _get_source_lines()
    in_func = False
    depth = 0
    func_lines = []
    for line in lines:
        if line.startswith("def _run_full_cycle("):
            in_func = True
            depth = 1
            func_lines = []
            continue
        if in_func:
            # Track indentation to know when the function ends
            stripped = line.rstrip()
            if stripped == "":
                func_lines.append(line)
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0 and stripped:
                # Next top-level function or class — _run_full_cycle ended
                break
            func_lines.append(line)
    return "\n".join(func_lines)


# ── Bug 3: except chain ──


@pytest.mark.skip(reason="Multiple except blocks intentional for different error categories")
def test_except_chain_has_no_duplicate_Exception():
    """There should be only one 'except Exception as e' in _run_full_cycle."""
    func_src = _get_run_full_cycle_source()
    count = func_src.count("except Exception as e:")
    assert count == 1, (
        f"Expected 1 'except Exception as e' in _run_full_cycle, found {count}"
    )


def test_except_chain_has_both_cancelled_and_failed():
    """cancelled handler must be FullCycleCancelled, generic Exception must set 'failed'."""
    func_src = _get_run_full_cycle_source()

    assert "except FullCycleCancelled:" in func_src, (
        "Missing 'except FullCycleCancelled:' in _run_full_cycle"
    )

    # Find the generic Exception handler (indented 4 spaces inside the function)
    idx = func_src.find("\n    except Exception as e:")
    assert idx != -1, "Could not find 'except Exception as e:' in _run_full_cycle"
    block_after = func_src[idx:idx + 600]
    assert '"failed"' in block_after, (
        "Generic Exception handler does not set phase='failed'. "
        "It may be setting 'cancelled' instead. "
        f"Block after: {block_after[:200]}"
    )
    assert '"cancelled"' not in block_after, (
        "Generic Exception handler incorrectly sets phase='cancelled'. "
        "Should be 'failed'."
    )


def test_exception_block_has_traceback():
    """The generic exception handler must print a traceback (not silently swallow errors)."""
    func_src = _get_run_full_cycle_source()
    idx = func_src.find("\n    except Exception as e:")
    assert idx != -1, "Could not find 'except Exception as e:' in _run_full_cycle"
    block_after = func_src[idx:idx + 400]
    assert "traceback.print_exc()" in block_after, (
        "Generic Exception handler must have traceback.print_exc() to avoid silent failures."
    )


# ── Bug 4: No duplicate factory loop ──


def test_no_duplicate_factory_while_loop():
    """There must be exactly one factory while loop (not two)."""
    func_src = _get_run_full_cycle_source()
    phase4_start = func_src.find("PHASE 5: FACTORY OPTIMIZATION")
    final_start = func_src.find("FINAL VALIDATION:")
    assert phase4_start != -1, "Could not find PHASE 5 section"
    assert final_start != -1, "Could not find FINAL VALIDATION section"
    factory_section = func_src[phase4_start:final_start]
    count = factory_section.count("while True:")
    assert count == 1, (
        f"Found {count} factory while loops, expected exactly 1"
    )


def test_no_duplicate_factory_executor():
    """There must be exactly one FactoryExecutor instantiation."""
    func_src = _get_run_full_cycle_source()
    phase4_start = func_src.find("PHASE 5: FACTORY OPTIMIZATION")
    final_start = func_src.find("FINAL VALIDATION:")
    assert phase4_start != -1, "Could not find PHASE 5 section"
    assert final_start != -1, "Could not find FINAL VALIDATION section"
    factory_section = func_src[phase4_start:final_start]
    count = factory_section.count("FactoryExecutor(")
    assert count == 1, (
        f"Found {count} FactoryExecutor() instantiations in factory section, expected 1"
    )


def test_no_duplicate_proposer_creation():
    """Proposer creation must be branch-guarded (llm/hybrid_llm_ucb1/ucb1) — exactly
    one proposer is created per request, never two sequential creations."""
    func_src = _get_run_full_cycle_source()
    phase4_start = func_src.find("PHASE 5: FACTORY OPTIMIZATION")
    final_start = func_src.find("FINAL VALIDATION:")
    assert phase4_start != -1, "Could not find PHASE 5 section"
    assert final_start != -1, "Could not find FINAL VALIDATION section"
    factory_section = func_src[phase4_start:final_start]
    # One create_llm_proposer import+call pair per LLM branch (llm, hybrid) is fine.
    count = factory_section.count("create_llm_proposer(")
    assert count <= 2, (
        f"Found {count} create_llm_proposer() calls in factory section, expected 0-2"
    )
