"""
logger.py — Structured pipeline logger (ported from notebook PipelineLogger)
Faithful port of the PipelineLogger class defined in Cell 9a / 9b / 15.
"""

import datetime
from typing import Optional

class PipelineLogger:
    _LEVELS = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3}
    _ICONS  = {'DEBUG': '..', 'INFO': 'OK', 'WARN': '!!', 'ERROR': 'XX'}
    _min    = 'INFO'

    # In-memory log buffer so /api/logs can return recent entries
    _buffer: list = []
    _MAX_BUFFER = 500

    @classmethod
    def set_level(cls, level: str):
        cls._min = level

    @classmethod
    def log(cls, step: str, msg: str, level: str = 'INFO') -> None:
        if cls._LEVELS.get(level, 0) < cls._LEVELS.get(cls._min, 0):
            return
        ts   = datetime.datetime.now().strftime('%H:%M:%S')
        icon = cls._ICONS.get(level, '  ')
        line = f'[{ts}][{icon}][{step:<16s}] {msg}'
        print(line)
        entry = {'ts': ts, 'level': level, 'step': step, 'msg': msg}
        cls._buffer.append(entry)
        if len(cls._buffer) > cls._MAX_BUFFER:
            cls._buffer = cls._buffer[-cls._MAX_BUFFER:]

    @classmethod
    def get_recent(cls, n: int = 100) -> list:
        return cls._buffer[-n:]

    @classmethod
    def clear(cls):
        cls._buffer.clear()


# Module-level shorthand  (same pattern as notebook)
log = PipelineLogger.log
