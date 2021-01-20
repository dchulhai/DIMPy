#!/usr/bin/env python3

from .dimpy import run_from_command_line

if __name__ == '__main__':
    try:
        run_from_command_line()
    except KeyboardInterrupt:
        exit(1) 
