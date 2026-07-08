"""Cloud-native orchestration for the autonomous 4B data loop (Mac-independent).

Everything here runs on Modal (chess-instructor-2): a self-perpetuating loop that
trains -> evaluates (the P1 eval app) -> improves the data deterministically ->
repeats, with all state on a Modal Volume so a laptop restart never stops it.
"""
