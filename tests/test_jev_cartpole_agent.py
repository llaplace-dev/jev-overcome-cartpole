import unittest

from jev_cartpole_agent_perfect import (
    ACTION_TO_ID,
    CARTPOLE_FAILURE_CONDITIONS,
    CARTPOLE_PHYSICS,
    CARTPOLE_PHYSICS_NO_FORMULA,
    JevCartPoleAgent,
    JevDecision,
    TypeSafeJev,
    _extract_noul,
)


class FakeClient:
    def __init__(self):
        self.states = []

    def choose(self, state):
        self.states.append(state)
        return JevDecision(
            action_label="right",
            action=ACTION_TO_ID["right"],
            confidence=0.9,
            probabilities={"left": 0.1, "right": 0.9},
            raw_answer={"type": "noul", "noul": 0.9},
        )


class JevCartPoleAgentTests(unittest.TestCase):
    def test_request_declares_a_noul_question(self):
        client = TypeSafeJev(api_key="test-key")
        captured = {}

        def fake_post(payload):
            captured.update(payload)
            return {"answers": {"action": {"type": "noul", "noul": 0.75}}}

        client._post_json = fake_post
        decision = client.choose({"current_observation": {"cart_position": 0.0}})

        self.assertEqual(decision.action, 1)
        self.assertEqual(captured["questions"]["action"]["type"], "noul")
        self.assertIn("criteria", captured["questions"]["action"])
        self.assertIn("true", captured["questions"]["action"]["criteria"])
        self.assertIn("false", captured["questions"]["action"]["criteria"])

    def test_ablation_prompt_omits_explicit_prediction_formula(self):
        client = TypeSafeJev(api_key="test-key", ablation_no_formula=True)
        captured = {}

        def fake_post(payload):
            captured.update(payload)
            return {"answers": {"action": {"type": "noul", "noul": 0.25}}}

        client._post_json = fake_post
        client.choose(
            {
                "current_observation": {
                    "cart_position": 0.0,
                    "cart_velocity": 0.0,
                    "pole_angle_radians": 0.01,
                    "pole_angular_velocity": 0.0,
                },
                "previous_transitions": [],
                "physics": CARTPOLE_PHYSICS_NO_FORMULA,
            }
        )

        instructions = captured["questions"]["action"]["instructions"]
        prediction_rule = captured["state"]
        self.assertNotIn("cart_position + 0.02 * cart_velocity", prediction_rule)
        self.assertNotIn("pole_angle_radians + 0.02 * pole_angular_velocity", prediction_rule)
        self.assertNotIn("cart_position + 0.02 * cart_velocity", instructions)
        self.assertNotIn("pole_angle_radians + 0.02 * pole_angular_velocity", instructions)

    def test_extracts_noul_probability_and_maps_true_to_right(self):
        decision = _extract_noul(
            {
                "answers": {
                    "action": {
                        "type": "noul",
                        "noul": 0.81,
                    }
                }
            }
        )
        self.assertEqual(decision.action, 1)
        self.assertEqual(decision.action_label, "right")
        self.assertAlmostEqual(decision.probabilities["right"], 0.81)

    def test_maps_false_noul_to_left(self):
        decision = _extract_noul(
            {
                "answers": {
                    "action": {
                        "type": "noul",
                        "noul": 0.2,
                    }
                }
            }
        )
        self.assertEqual(decision.action, 0)
        self.assertEqual(decision.action_label, "left")

    def test_agent_serializes_cartpole_observation_and_history(self):
        client = FakeClient()
        agent = JevCartPoleAgent(client, history_size=1)
        agent.reset()
        decision = agent.act([0.1, -0.2, 0.03, 0.04], step=7)

        self.assertEqual(decision.action, 1)
        self.assertEqual(client.states[0]["step"], 7)
        self.assertEqual(client.states[0]["current_observation"]["pole_angle_radians"], 0.03)
        self.assertEqual(client.states[0]["previous_transitions"], [])
        self.assertEqual(
            client.states[0]["failure_conditions"], CARTPOLE_FAILURE_CONDITIONS
        )
        self.assertIn("2.4", client.states[0]["failure_conditions"]["terminated"]["cart_position"])
        self.assertIn("0.2094395", client.states[0]["failure_conditions"]["terminated"]["pole_angle_radians"])
        self.assertEqual(client.states[0]["physics"], CARTPOLE_PHYSICS)
        self.assertIn("Gravity", client.states[0]["physics"]["gravity"])
        self.assertIn("inertia", client.states[0]["physics"]["inertia"])
        self.assertIn("acceleration", client.states[0]["physics"]["acceleration"])

        agent.record_transition(
            step=7,
            observation=[0.1, -0.2, 0.03, 0.04],
            decision=decision,
            next_observation=[0.11, -0.1, 0.031, 0.05],
            reward=1.0,
            terminated=False,
            truncated=False,
        )
        agent.act([0.11, -0.1, 0.031, 0.05], step=8)
        transition = client.states[1]["previous_transitions"][0]
        self.assertEqual(transition["step"], 7)
        self.assertEqual(transition["action"], {"label": "right", "id": 1})
        self.assertEqual(transition["observation_after"]["cart_position"], 0.11)

    def test_history_keeps_all_transitions_in_the_episode(self):
        client = FakeClient()
        agent = JevCartPoleAgent(client, history_size=6)
        agent.reset()

        for step in range(6):
            observation = [float(step), 0.0, 0.01, 0.0]
            next_observation = [float(step) + 0.1, 0.0, 0.02, 0.0]
            decision = agent.act(observation, step=step)
            agent.record_transition(
                step=step,
                observation=observation,
                decision=decision,
                next_observation=next_observation,
                reward=1.0,
                terminated=False,
                truncated=False,
            )

        agent.act([6.0, 0.0, 0.01, 0.0], step=6)
        history = client.states[-1]["previous_transitions"]
        self.assertEqual(len(history), 6)
        self.assertEqual([item["step"] for item in history], [0, 1, 2, 3, 4, 5])

    def test_default_history_size_sends_only_current_observation(self):
        client = FakeClient()
        agent = JevCartPoleAgent(client)
        agent.reset()

        decision = agent.act([0.0, 0.0, 0.01, 0.0], step=0)
        agent.record_transition(
            step=0,
            observation=[0.0, 0.0, 0.01, 0.0],
            decision=decision,
            next_observation=[0.01, 0.1, 0.011, 0.01],
            reward=1.0,
            terminated=False,
            truncated=False,
        )
        agent.act([0.01, 0.1, 0.011, 0.01], step=1)

        self.assertEqual(client.states[-1]["previous_transitions"], [])

    def test_history_size_keeps_only_the_most_recent_transitions(self):
        client = FakeClient()
        agent = JevCartPoleAgent(client, history_size=2)
        agent.reset()

        for step in range(4):
            observation = [float(step), 0.0, 0.01, 0.0]
            decision = agent.act(observation, step=step)
            agent.record_transition(
                step=step,
                observation=observation,
                decision=decision,
                next_observation=[float(step) + 0.1, 0.0, 0.02, 0.0],
                reward=1.0,
                terminated=False,
                truncated=False,
            )

        agent.act([4.0, 0.0, 0.01, 0.0], step=4)
        history = client.states[-1]["previous_transitions"]
        self.assertEqual([item["step"] for item in history], [2, 3])


if __name__ == "__main__":
    unittest.main()
