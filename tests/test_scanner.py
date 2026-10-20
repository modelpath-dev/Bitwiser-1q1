import struct
import tempfile
import unittest
from pathlib import Path

import scanner


def _gguf_kv_str(key: str, value: str) -> bytes:
    k, v = key.encode(), value.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v


def _gguf_kv_u32(key: str, value: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", value)


def write_gguf(path: Path, file_type: int) -> None:
    kvs = [_gguf_kv_str("general.architecture", "llama"), _gguf_kv_u32("general.file_type", file_type)]
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    path.write_bytes(header + b"".join(kvs))


class ReadGgufMetadataTest(unittest.TestCase):
    def quant_for(self, file_type: int) -> str | None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf"
            write_gguf(p, file_type)
            return scanner._read_gguf_metadata(p).get("quant")

    def test_known_quant_types(self):
        self.assertEqual(self.quant_for(1), "F16")
        self.assertEqual(self.quant_for(15), "Q4_K_M")

    def test_bf16_source_is_recognised(self):
        # LLAMA_FTYPE_MOSTLY_BF16 = 32 in llama.h; HF conversions commonly ship as BF16.
        self.assertEqual(self.quant_for(32), "BF16")

    def test_ternary_and_one_bit_types_are_recognised(self):
        self.assertEqual(self.quant_for(36), "TQ1_0")
        self.assertEqual(self.quant_for(40), "Q1_0")


if __name__ == "__main__":
    unittest.main()
