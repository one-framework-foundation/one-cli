# Aggregate sandbox quota envelope

`onecli.sandbox.quota.prepare` is a trusted application API, independent of model inputs.
It returns command argv and a minimal user-bus environment for a unique transient
systemd user service. The caller captures bounded output and always invokes
`cleanup()` in a finally block, even when launch, output capture or execution fails.

Before executing the supplied absolute command, an isolated Python interpreter
checks its actual cgroup v2 membership and reads back memory.max, memory.swap.max,
pids.max and cpu.max. Invalid or absent enforcement exits 125 before the command.
Every descendant shares the aggregate memory/process and CPU-rate envelope.
RuntimeMaxSec plus control-group stop/kill bounds lifetime; CPU rate times runtime
bounds available CPU time, subject to kernel scheduling/accounting granularity.
No delegated cgroup or cgroup path may be exposed inside the actual sandbox.

This envelope is for trusted host-selected bwrap argv. It does not isolate arbitrary
calling process commands, limit aggregate disk storage, cap concurrent independent jobs, or
make provider billing guarantees. Snapshot and artifact handling remain separate.
A same-UID hostile same-user process, modified system executables or user manager, and
root compromise are outside this boundary. No command is released on a failed
readback. No fallback runs without quota enforcement.

## Release and delayed dispatch

After Popen, the host must call `launch.authorize(timeout=5)` before collecting the
result. A private, per-launch control file coordinates verifier readiness, release
and cancellation with an exclusive flock. Cleanup records cancellation before
contacting systemd. A service whose StartTransientUnit was delayed therefore sees
cancellation and cannot execute its payload, even if cleanup originally observed
no unit. The verifier keeps the lock through exec; CLOEXEC releases it atomically
at process replacement. Cancellation and payload release cannot cross unnoticed.
Cleanup deletes the control file once the unit is stopped, and removes the
controls directory when it is empty. That is safe for a delayed manager request:
its verifier finds no control file and exits before running anything, with
status 1 and a Python traceback rather than the 125 a refused release returns.
`control_root` may select a canonical private
host directory (created with mode 0700), separate from removable result snapshots.
