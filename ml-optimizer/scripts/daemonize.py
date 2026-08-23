#!/usr/bin/env python3
"""Double-fork daemonizer: runs CMD fully detached from the invoking session.

Classic POSIX double-fork: the intermediate child exits immediately, so the
grandchild is orphaned and reparented to init before it execs. Process-tree
reapers walking descendants of the calling shell will not find it.
"""

import os
import sys


def main():
    args = sys.argv[1:]
    out_path = None
    if args and args[0] == "--out":
        out_path = args[1]
        args = args[2:]
    if not args:
        sys.exit("usage: daemonize.py [--out FILE] CMD [ARGS...]")

    pid = os.fork()
    if pid > 0:
        # First parent: reap the intermediate child and return immediately.
        os.waitpid(pid, 0)
        print(f"[DAEMONIZE] launched {args} (orphaned)")
        return

    os.setsid()
    if os.fork() > 0:
        os._exit(0)  # intermediate child exits; grandchild reparents to init

    fd0 = os.open(os.devnull, os.O_RDONLY)
    os.dup2(fd0, 0)
    if out_path:
        fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    else:
        fdn = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fdn, 1)
        os.dup2(fdn, 2)

    os.execvp(args[0], args)


main()
