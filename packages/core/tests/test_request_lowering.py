import unittest
from types import SimpleNamespace

from sglang_h3_compat.request_lowering import lower_video_request_kwargs_after


class RequestLoweringTest(unittest.TestCase):
    def test_extras_are_lifted_into_diffusers_kwargs(self):
        request = SimpleNamespace(
            model_extra={
                "minimax_h3_sampler": "er_sde",
                "minimax_h3_sigmas": {"video": [1.0, 0.0]},
                "export_state": True,
                "h3_continuation": {
                    "state_path": "/tmp/a.state.safetensors",
                    "anchor_latent_frames": 5,
                },
            }
        )
        result = lower_video_request_kwargs_after(
            {"diffusers_kwargs": {"existing": 1}},
            None,
            request,
            {},
        )
        self.assertEqual(
            result["diffusers_kwargs"],
            {
                "existing": 1,
                "minimax_h3_sampler": "er_sde",
                "minimax_h3_sigmas": {"video": [1.0, 0.0]},
                "export_state": True,
                "h3_continuation": {
                    "state_path": "/tmp/a.state.safetensors",
                    "anchor_latent_frames": 5,
                },
            },
        )

    def test_rejects_unknown_sampler(self):
        request = SimpleNamespace(model_extra={"minimax_h3_sampler": "dpmsolver"})
        with self.assertRaisesRegex(ValueError, "minimax_h3_sampler"):
            lower_video_request_kwargs_after({}, None, request, {})

    def test_keeps_result_when_no_compat_fields_are_present(self):
        request = SimpleNamespace(model_extra={})
        result = {"diffusers_kwargs": {"existing": 1}}
        self.assertIs(
            lower_video_request_kwargs_after(result, None, request, {}), result
        )


if __name__ == "__main__":
    unittest.main()
