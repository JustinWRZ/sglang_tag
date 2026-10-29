import os
import unittest
from unittest import mock

from sglang_h3_compat.flags import (
    env_enabled,
    ffmpeg_path,
    nvenc_cq,
    vae_bf16_enabled,
)


class FlagsTest(unittest.TestCase):
    def test_env_enabled_defaults_and_values(self):
        self.assertTrue(env_enabled("H3_COMPAT_MISSING_FLAG", True))
        self.assertFalse(env_enabled("H3_COMPAT_MISSING_FLAG", False))
        with mock.patch.dict(os.environ, {"H3_COMPAT_TEST_FLAG": "true"}):
            self.assertTrue(env_enabled("H3_COMPAT_TEST_FLAG", False))
        with mock.patch.dict(os.environ, {"H3_COMPAT_TEST_FLAG": "0"}):
            self.assertFalse(env_enabled("H3_COMPAT_TEST_FLAG", True))

    def test_vae_bf16_gate(self):
        with mock.patch.dict(os.environ, {"MINIMAX_H3_VAE_ENCODE_DTYPE": "bf16"}):
            self.assertTrue(vae_bf16_enabled())
        with mock.patch.dict(os.environ, {"MINIMAX_H3_VAE_ENCODE_DTYPE": "fp32"}):
            self.assertFalse(vae_bf16_enabled())

    def test_nvenc_overrides(self):
        with mock.patch.dict(os.environ, {"H3_COMPAT_FFMPEG_PATH": "/opt/ffmpeg"}):
            self.assertEqual(ffmpeg_path(), "/opt/ffmpeg")
        with mock.patch.dict(os.environ, {"H3_COMPAT_NVENC_CQ": "31"}):
            self.assertEqual(nvenc_cq(), 31)
        with mock.patch.dict(os.environ, {"H3_COMPAT_NVENC_CQ": "invalid"}):
            self.assertEqual(nvenc_cq(), 25)


if __name__ == "__main__":
    unittest.main()
