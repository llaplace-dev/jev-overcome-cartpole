#!/usr/bin/env python3
"""Run a predictive CartPole agent with Jev through TypeSafe's API.

This starts from v2 and adds the CartPole time scale plus explicit short-horizon
state estimates for position and pole angle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


TYPESAFE_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
CARTPOLE_ENV_ID = "CartPole-v1"
ACTION_TO_ID = {"push_left": 0, "push_right": 1}
RIGHT_ACTION_THRESHOLD = 0.5
TRANSIENT_STATUS_CODES = frozenset(
    (408, 429, 500, 502, 503, 504, 520, 521, 522, 524)
)

# These are the CartPole-specific prompts used by jev_deep_rl.  The only
# adaptation is that this file asks a noul question: true means push_right and
# false means push_left.  The original repository asks the equivalent Choice
# question with the same goal, action descriptions, and instructions.
CARTPOLE_GOAL = (
    "Keep the pole upright and the cart on the track for as long as possible. "
    "An episode fails beyond +/-2.4 meters or +/-12 degrees. "
    "Positive position, velocity, angle, and angular velocity point right."
)

CARTPOLE_ACTION_DESCRIPTIONS = {
    "push_left": "Apply a leftward force to the cart.",
    "push_right": "Apply a rightward force to the cart.",
}

CARTPOLE_CONTROL_GUIDANCE = (
    "Predict the next few states before choosing an action. Consider the current "
    "cart position, cart velocity, pole angle, and pole angular velocity. "
    "Account for gravity, acceleration, and inertia: a force changes acceleration "
    "and velocity, and its effect persists into subsequent observations. "
    "Correct before the cart or pole approaches a failure boundary; do not wait "
    "until a boundary is crossed. Prioritize preventing an imminent pole-angle "
    "failure, even if the cart is not perfectly centered. Avoid switching left "
    "and right because of tiny observation changes; keep a correction direction "
    "until the predicted motion justifies changing it."
)

CARTPOLE_ACTION_GUIDANCE = (
    "Action 0 (push_left) applies a leftward force and action 1 (push_right) "
    "applies a rightward force. Positive cart position and velocity point right. "
    "Positive pole angle means the pole leans right, and positive pole angular "
    "velocity means the rightward lean is increasing. As a default correction, "
    "a pole leaning right calls for push_right to move the cart under the pole; "
    "a pole leaning left calls for push_left. When the angle is near zero, use "
    "pole angular velocity to anticipate the next lean."
)

CARTPOLE_SAFETY_GUIDANCE = (
    "CartPole fails when abs(cart_position_m) exceeds 2.4 meters or abs(" 
    "pole_angle_deg) exceeds 12 degrees, and CartPole-v1 truncates at 500 steps. "
    "Treat abs(cart_position_m) >= 1.8 meters and abs(pole_angle_deg) >= 6.9 "
    "degrees as warning zones. Start correcting inside a warning zone; never "
    "wait until a hard boundary is crossed."
)

CARTPOLE_PRIORITY_GUIDANCE = (
    "Use this priority order: first prevent an imminent pole-angle failure, then "
    "prevent an imminent cart-position failure, then reduce pole angle and "
    "angular velocity, and only then return the cart smoothly toward the center. "
    "Do not sacrifice a stable pole merely to center the cart immediately. "
    "The objective is to survive the complete 500-step episode."
)

CARTPOLE_PREDICTION_GUIDANCE = (
    "Each action controls approximately 0.02 seconds of motion. For an early "
    "warning, estimate the next state before accounting for the full force "
    "response: predicted_cart_position_m is approximately cart_position_m + "
    "0.02 * cart_velocity_m_per_s, and predicted_pole_angle_deg is approximately "
    "pole_angle_deg + 0.02 * pole_angular_velocity_deg_per_s. These are trend "
    "estimates, not exact dynamics. Compare push_left and push_right over the "
    "next few steps, and use the action whose predicted trend moves away from "
    "the nearest failure boundary. If angle and angular velocity have the same "
    "sign, the pole is continuing to fall that way and correction is urgent; if "
    "they have opposite signs, avoid overcorrecting while the pole is recovering."
)

ACTION_INSTRUCTIONS = (
    "Which single action should be applied for the next environment step "
    "to pursue `goal`, given `observation`? "
    "Use `observation.description` when present to interpret the coordinates, "
    "controlled object, motion, and game phase. "
    "Consider current motion as well as position. Select from the supplied actions. "
    "Missing observations are unknown, not zero. "
    + CARTPOLE_CONTROL_GUIDANCE
    + " "
    + CARTPOLE_ACTION_GUIDANCE
    + " "
    + CARTPOLE_SAFETY_GUIDANCE
    + " "
    + CARTPOLE_PRIORITY_GUIDANCE
    + " "
    + CARTPOLE_PREDICTION_GUIDANCE
    + " For this noul question, true means push_right and false means push_left."
)

ACTION_CRITERIA = {
    "true": CARTPOLE_ACTION_DESCRIPTIONS["push_right"],
    "false": CARTPOLE_ACTION_DESCRIPTIONS["push_left"],
}


def make_state(
    step: int,
    observation: Sequence[float],
) -> Dict[str, Any]:
    """Build the CartPole state expected by the Jev prompt."""

    return {
        "environment": CARTPOLE_ENV_ID,
        "goal": CARTPOLE_GOAL,
        "step": int(step),
        "observation": _encode_observation(observation),
    }


class TypeSafeError(RuntimeError):
    """Raised when TypeSafe cannot return a valid Jev decision."""


@dataclass(frozen=True)
class JevDecision:
    """A typed Jev noul answer mapped to a CartPole action."""

    action_name: str
    action: int
    probabilities: Dict[str, float]


def _as_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _extract_noul(payload: Mapping[str, Any]) -> JevDecision:
    """Read Jev's yes probability and map it to push_left/push_right."""

    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        raise TypeSafeError("Jev response is missing an 'answers' object")

    answer = answers.get("action")
    if not isinstance(answer, Mapping):
        raise TypeSafeError("Jev response is missing answers.action")

    right_probability = _as_float(answer.get("noul"))
    if right_probability is None or not 0.0 <= right_probability <= 1.0:
        raise TypeSafeError(
            "Jev response contains invalid answers.action.noul: {!r}".format(
                answer.get("noul")
            )
        )

    action_name = (
        "push_right" if right_probability >= RIGHT_ACTION_THRESHOLD else "push_left"
    )
    return JevDecision(
        action_name=action_name,
        action=ACTION_TO_ID[action_name],
        probabilities={
            "push_left": 1.0 - right_probability,
            "push_right": right_probability,
        },
    )


class TypeSafeJev:
    """Small dependency-free client for TypeSafe's official Jev endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        base_url: str = TYPESAFE_SYSTEM_ONE_URL,
    ) -> None:
        if not api_key.strip():
            raise ValueError("TYPESAFE_API_KEY is empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not base_url.strip():
            raise ValueError("TypeSafe API base URL is empty")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.base_url = base_url

    def choose(self, state: Mapping[str, Any]) -> JevDecision:
        """Ask the control-guided CartPole noul question for the next action."""

        payload = {
            "model": self.model,
            "state": dict(state),
            "questions": {
                "action": {
                    "type": "noul",
                    "instructions": ACTION_INSTRUCTIONS,
                    "criteria": dict(ACTION_CRITERIA),
                }
            },
        }
        return _extract_noul(self._post_json(payload))

    def _post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )

        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
                    status = getattr(response, "status", 200)
                if not 200 <= status < 300:
                    raise TypeSafeError(
                        "TypeSafe returned HTTP {}: {}".format(
                            status, response_body[:500]
                        )
                    )
                parsed = json.loads(response_body)
                if not isinstance(parsed, Mapping):
                    raise TypeSafeError("TypeSafe returned a non-object JSON response")
                return parsed
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code not in TRANSIENT_STATUS_CODES or attempt + 1 >= self.max_attempts:
                    raise TypeSafeError(
                        "TypeSafe returned HTTP {}: {}".format(
                            exc.code, error_body[:500]
                        )
                    ) from exc
            except urllib.error.URLError as exc:
                if attempt + 1 >= self.max_attempts:
                    raise TypeSafeError(
                        "Could not reach TypeSafe: {}".format(exc.reason)
                    ) from exc
            except json.JSONDecodeError as exc:
                raise TypeSafeError("TypeSafe returned invalid JSON") from exc

            time.sleep(self.backoff_seconds * (2**attempt))

        raise TypeSafeError("TypeSafe request failed after retries")


def _encode_observation(observation: Sequence[float]) -> Dict[str, float]:
    """Use the same named, unit-bearing CartPole state as jev_deep_rl."""

    try:
        values = [float(value) for value in observation]
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected four finite CartPole observation values.") from exc
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected four finite CartPole observation values.")

    position, velocity, angle, angular_velocity = values
    return {
        "cart_position_m": position,
        "cart_velocity_m_per_s": velocity,
        "pole_angle_deg": math.degrees(angle),
        "pole_angular_velocity_deg_per_s": math.degrees(angular_velocity),
    }


def _capture_frame(env: Any, frame_index: int) -> Any:
    """Render one RGB frame and keep it distinct in the GIF."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "GIF recording requires Pillow. Install it with: "
            "python3 -m pip install Pillow"
        ) from exc

    frame = env.render()
    if frame is None:
        raise RuntimeError("Gymnasium did not return an RGB frame for GIF recording")
    image = Image.fromarray(frame).convert("RGB")

    # Prevent GIF encoders from merging repeated environment frames.
    marker_width = min(16, image.width)
    marker_bits = format(
        frame_index % (1 << marker_width), "0{}b".format(marker_width)
    )
    draw = ImageDraw.Draw(image)
    for x, bit in enumerate(marker_bits):
        color = (255, 255, 255) if bit == "1" else (0, 0, 0)
        draw.point((x, 0), fill=color)
        if image.height > 1:
            draw.point((x, 1), fill=color)

    return image.convert("P", palette=Image.ADAPTIVE, colors=256)


def _save_gif(frames: Sequence[Any], output_path: str, fps: int) -> None:
    """Save frames as a looping GIF and verify its encoded frame count."""

    if not frames:
        raise ValueError("Cannot save an empty CartPole GIF")
    if fps < 1:
        raise ValueError("GIF FPS must be >= 1")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "GIF recording requires Pillow. Install it with: "
            "python3 -m pip install Pillow"
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / float(fps))))
    frames[0].save(
        str(output),
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    with Image.open(str(output)) as saved:
        encoded_frame_count = int(getattr(saved, "n_frames", 1))
    if encoded_frame_count != len(frames):
        raise RuntimeError(
            "GIF encoder wrote {} frames; expected {}".format(
                encoded_frame_count, len(frames)
            )
        )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predictive CartPole prompt derived from v2 "
            "through TypeSafe's official API."
        )
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes (default: 1)."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional local step limit; CartPole-v1 itself truncates at 500 steps.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed environment resets.")
    parser.add_argument("--render", action="store_true", help="Render CartPole in a window.")
    parser.add_argument(
        "--gif",
        metavar="PATH",
        default=None,
        help="Save captured CartPole frames as a GIF at PATH.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=20,
        help="Playback frame rate for the GIF (default: 20).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every Jev action.")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "TYPESAFE_MODEL", os.environ.get("JEV_MODEL", DEFAULT_MODEL)
        ),
        help="TypeSafe model id (default: %(default)s).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TYPESAFE_BASE_URL", TYPESAFE_SYSTEM_ONE_URL),
        help="TypeSafe System One URL (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        import gymnasium as gym
    except ImportError:
        print(
            "gymnasium is not installed. Run: python3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not api_key:
        print("Set TYPESAFE_API_KEY before running the agent.", file=sys.stderr)
        return 1
    if args.episodes < 1:
        print("--episodes must be >= 1.", file=sys.stderr)
        return 1
    if args.max_steps is not None and args.max_steps < 1:
        print("--max-steps must be >= 1.", file=sys.stderr)
        return 1
    if args.gif_fps < 1:
        print("--gif-fps must be >= 1.", file=sys.stderr)
        return 1
    if args.gif and args.render:
        print("--render is ignored when --gif is enabled.", file=sys.stderr)

    client = TypeSafeJev(
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        base_url=args.base_url,
    )
    render_mode = "rgb_array" if args.gif else ("human" if args.render else None)
    env = gym.make(CARTPOLE_ENV_ID, render_mode=render_mode)
    gif_frames = []

    try:
        for episode in range(args.episodes):
            reset_seed = None if args.seed is None else args.seed + episode
            observation, _ = env.reset(seed=reset_seed)
            state = make_state(step=0, observation=observation)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                decision = client.choose(state)
                next_observation, reward, terminated, truncated, _ = env.step(
                    decision.action
                )
                reward = float(reward)
                terminated = bool(terminated)
                truncated = bool(truncated)
                steps += 1
                total_reward += reward

                if args.gif:
                    gif_frames.append(_capture_frame(env, len(gif_frames)))

                state = make_state(
                    step=steps,
                    observation=next_observation,
                )

                if args.verbose:
                    print(
                        "episode={} step={} action={} (id={}) probabilities={}".format(
                            episode + 1,
                            steps,
                            decision.action_name,
                            decision.action,
                            decision.probabilities,
                        )
                    )

                if args.max_steps is not None and steps >= args.max_steps:
                    break

            local_limit = (
                args.max_steps is not None
                and steps >= args.max_steps
                and not (terminated or truncated)
            )
            print(
                "episode={} steps={} reward={:.0f}{}".format(
                    episode + 1,
                    steps,
                    total_reward,
                    " (local step limit)" if local_limit else "",
                )
            )
    except TypeSafeError as exc:
        print("Jev request failed: {}".format(exc), file=sys.stderr)
        return 1
    finally:
        env.close()

    if args.gif:
        try:
            _save_gif(gif_frames, args.gif, args.gif_fps)
        except (OSError, RuntimeError, ValueError) as exc:
            print("Could not save GIF: {}".format(exc), file=sys.stderr)
            return 1
        print("gif_saved={} frames={}".format(args.gif, len(gif_frames)))

    return 0


def main() -> int:
    return run(_make_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
