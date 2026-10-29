import math
import unittest

from sglang_h3_compat.native import (
    ErSdeStepScalars,
    build_nvenc_command,
    build_state_manifest_json,
    er_sde_step_scalars,
    frame_count_from_video_latent_t,
    native_version,
    prepare_sigmas,
    validate_continuation_json,
)

SIGMAS = [
    0.9999916659027076,
    0.9844,
    0.9627,
    0.9304,
    0.878,
    0.7579,
    0.5527,
    0.2368,
    0.0,
]

EXPECTED = [
    (False, 1.309908534722834e-13, 0.015600026585161686, None, 0.9843999743461609),
    (False, 0.5622528791427612, 0.028528837487101555, -0.7112250328063965, 0.7876860499382019),
    (False, 0.7521670460700989, 0.04154415428638458, -0.3184700310230255, 0.5842148661613464),
    (False, 0.8072152733802795, 0.06581779569387436, -0.24590037763118744, 0.4547892212867737),
    (False, 0.7555352449417114, 0.14992469549179077, -0.3888469338417053, 0.3665596544742584),
    (False, 0.6678386926651001, 0.2856162488460541, -0.35246098041534424, 0.22200094163417816),
    (False, 0.39919641613960266, 0.5846394300460815, -0.38764119148254395, 0.08598870784044266),
    (True, 0.0, 0.0, None, 0.0),
]


class NativeCoreTest(unittest.TestCase):
    def test_native_version(self):
        self.assertEqual(native_version(), "0.3.1")

    def test_prepare_sigmas(self):
        video, audio = prepare_sigmas(
            video=[1.0, 0.0],
            audio=None,
            sampler="er_sde",
            num_steps=1,
            flow_shift=12.0,
            audio_flow_shift=3.0,
        )
        self.assertEqual(len(video), 2)
        self.assertEqual(len(audio), 2)
        self.assertLess(video[0], 1.0)
        self.assertEqual(video[-1], 0.0)
        self.assertEqual(audio[-1], 0.0)

    def test_er_sde_step_scalars_matches_torch_reference(self):
        values = er_sde_step_scalars(SIGMAS, max_stage=3, num_integration_points=200.0)
        self.assertEqual(len(values), len(EXPECTED))
        for scalar, expected in zip(values, EXPECTED):
            self.assertIsInstance(scalar, ErSdeStepScalars)
            self.assertEqual(scalar.is_final, expected[0])
            for actual, reference in zip(
                (
                    scalar.stage1_x,
                    scalar.stage1_d,
                    scalar.stage2_x,
                    scalar.noise_scale,
                ),
                expected[1:],
            ):
                if reference is None:
                    self.assertIsNone(actual)
                    continue
                self.assertTrue(
                    math.isclose(actual, reference, rel_tol=2e-6, abs_tol=2e-6),
                    msg=f"{actual} != {reference}",
                )

    def test_state_manifest_json(self):
        value = build_state_manifest_json(
            """
            {
              "protocol": "h3-native-av-state/v1",
              "artifact_kind": "h3-native-av-state",
              "created_at_unix": 1,
              "request": {},
              "tensor_spec": {},
              "geometry": {},
              "sampling": {},
              "video_timeline": {},
              "audio_timeline": {},
              "continuation": null,
              "validity": {},
              "delivery": {},
              "export_elapsed_s": 0.1
            }
            """
        )
        self.assertIn('"artifact_kind"', value)

    def test_frame_count(self):
        self.assertEqual(frame_count_from_video_latent_t(37), 124)
        with self.assertRaises(ValueError):
            frame_count_from_video_latent_t(3)

    def test_nvenc_command(self):
        command = build_nvenc_command(
            [
                "/usr/bin/ffmpeg",
                "-y",
                "-vcodec",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "25",
                "out.mp4",
            ],
            "/custom/ffmpeg",
            25,
        )
        self.assertEqual(
            command,
            [
                "/custom/ffmpeg",
                "-y",
                "-vcodec",
                "h264_nvenc",
                "-rc",
                "vbr",
                "-cq",
                "25",
                "-pix_fmt",
                "yuv420p",
                "out.mp4",
            ],
        )

    def test_continuation_report(self):
        report_json = validate_continuation_json(
            """
            {
              "continuation": {
                "state_path": "/tmp/a.state.safetensors",
                "anchor_latent_frames": 5
              },
              "manifest": {
                "protocol": "h3-native-av-state/v1",
                "geometry": {"latent_t": 37, "latent_h": 48, "latent_w": 84},
                "sampling": {"sampler": "er_sde", "sigmas": [1.0, 0.0]}
              },
              "target": {"latent_t": 37, "latent_h": 48, "latent_w": 84},
              "current_sampler": "er_sde",
              "current_sigmas": [1.0, 0.0],
              "anchor_row_count": 5040
            }
            """
        )
        self.assertIn('"resolved_anchor_latent_frames":5', report_json)
        self.assertIn('"mode":"masked_prefix"', report_json)


if __name__ == "__main__":
    unittest.main()
