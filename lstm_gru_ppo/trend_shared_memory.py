import struct
import time
from multiprocessing import shared_memory

import numpy as np


class TrendSharedMemory:
    """
    Shared memory layout:
    - metadata (fixed 64 bytes): <Q d f I I
      seq(uint64), timestamp(float64), confidence(float32), rows(uint32), cols(uint32)
    - matrix payload: rows * cols * float32

    Lock-free protocol (single writer / multi reader):
    1) writer sets odd seq (writing)
    2) writer updates payload
    3) writer sets even seq (committed)
    Reader accepts only if seq_before == seq_after and seq is even.
    """

    META_FMT = "<QdfII"
    META_SIZE = 64

    def __init__(self, name, rows, cols, create=False):
        self.name = name
        self.rows = int(rows)
        self.cols = int(cols)

        payload_size = self.rows * self.cols * 4
        self.total_size = self.META_SIZE + payload_size

        if create:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.total_size)
            self._write_meta(seq=0, ts=0.0, conf=0.0, rows=self.rows, cols=self.cols)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()

    def _write_meta(self, seq, ts, conf, rows, cols):
        packed = struct.pack(self.META_FMT, int(seq), float(ts), float(conf), int(rows), int(cols))
        self.shm.buf[: len(packed)] = packed

    def _read_meta(self):
        size = struct.calcsize(self.META_FMT)
        seq, ts, conf, rows, cols = struct.unpack(self.META_FMT, self.shm.buf[:size])
        return seq, ts, conf, rows, cols

    def write_matrix(self, matrix, confidence=0.0):
        mat = np.asarray(matrix, dtype=np.float32)
        if mat.shape != (self.rows, self.cols):
            raise ValueError(f"matrix shape mismatch: got {mat.shape}, expect {(self.rows, self.cols)}")

        seq0, _, _, _, _ = self._read_meta()
        start_seq = seq0 + 1 if seq0 % 2 == 0 else seq0 + 2

        # Mark as writing.
        self._write_meta(start_seq, time.time(), confidence, self.rows, self.cols)

        payload = mat.tobytes(order="C")
        start = self.META_SIZE
        end = start + len(payload)
        self.shm.buf[start:end] = payload

        # Mark as committed.
        self._write_meta(start_seq + 1, time.time(), confidence, self.rows, self.cols)

    def read_matrix(self):
        size = self.rows * self.cols * 4
        start = self.META_SIZE
        end = start + size

        seq1, ts1, conf1, r1, c1 = self._read_meta()
        payload = bytes(self.shm.buf[start:end])
        seq2, ts2, conf2, r2, c2 = self._read_meta()

        if r1 != self.rows or c1 != self.cols or r2 != self.rows or c2 != self.cols:
            return None, 0.0, 0.0, 0

        # Must be stable and committed (even seq).
        if seq1 != seq2 or (seq2 % 2) != 0:
            return None, 0.0, 0.0, 0

        mat = np.frombuffer(payload, dtype=np.float32).reshape(self.rows, self.cols)
        return mat, float(ts2), float(conf2), int(seq2)
