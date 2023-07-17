import unittest
import random
import torch
from gymnasium import spaces
from scratch import Network, replayMemory, DQN


# class TestNetwork(unittest.TestCase):
#     def setUp(self):
#         self.input_states = 16
#         self.output_actions = 4
#         self.network = Network(self.input_states, self.output_actions)

#     def test_forward_input(self):
#         state = torch.randn(1, self.input_states)
#         # print("STATE: ", state)
#         result = self.network.forward(state)
#         self.assertIsInstance(result, torch.Tensor)

#     def test_forward_output(self):
#         state = torch.randn(1, self.input_states)
#         result = self.network.forward(state)
#         # print("RESULT: ", result)
#         self.assertEqual(result.shape, (1, self.output_actions))


# class TestReplayMemory(unittest.TestCase):
#     state = torch.randn(4)
#     action = torch.tensor(0)
#     reward = torch.tensor(0.5)
#     next_state = torch.randn(4)
#     transition = (state, action, reward, next_state)

#     def setUp(self):
#         self.capacity = 100
#         self.memory = replayMemory(self.capacity)
#         self.input_space = spaces.Box(low=0, high=1, shape=(4,), dtype=float)
#         self.output_space = spaces.Discrete(2)

#     def test_push_input_type(self):
#         self.memory.push(self.transition)

#         self.assertIsInstance(self.memory.memory[0][0], torch.Tensor)
#         self.assertIsInstance(self.memory.memory[0][1], torch.Tensor)
#         self.assertIsInstance(self.memory.memory[0][2], torch.Tensor)
#         self.assertIsInstance(self.memory.memory[0][3], torch.Tensor)

#     def test_push_output_format(self):
#         self.memory.push(self.transition)

#         self.assertEqual(self.memory.memory[0][0].shape, (4,))
#         self.assertEqual(self.memory.memory[0][1].shape, ())
#         self.assertEqual(self.memory.memory[0][2].shape, ())
#         self.assertEqual(self.memory.memory[0][3].shape, (4,))

#     def test_sample_input_type(self):
#         batch_size = 32
#         self.memory.memory = [
#             (torch.randn(4), torch.tensor(0), torch.tensor(0.5), torch.randn(4))
#             for _ in range(batch_size)
#         ]

#         result = self.memory.sample(batch_size)

#         self.assertIsInstance(result[0], torch.Tensor)
#         self.assertIsInstance(result[1], torch.Tensor)
#         self.assertIsInstance(result[2], torch.Tensor)
#         self.assertIsInstance(result[3], torch.Tensor)

#     def test_sample_output_format(self):
#         batch_size = 32
#         self.memory.memory = [
#             (torch.randn(4), torch.tensor(0), torch.tensor(0.5), torch.randn(4))
#             for _ in range(batch_size)
#         ]

#         result = self.memory.sample(batch_size)

#         self.assertEqual(result[0].shape, (batch_size, 4))
#         self.assertEqual(result[1].shape, (batch_size,))
#         self.assertEqual(result[2].shape, (batch_size,))
#         self.assertEqual(result[3].shape, (batch_size, 4))


class TestDQN(unittest.TestCase):
    def setUp(self):
        self.input_states = 16
        self.output_actions = 4
        self.gamma = 0.9
        self.dqn = DQN(self.input_states, self.output_actions, self.gamma)
        self.input_space = spaces.Box(
            low=0, high=1, shape=(self.input_states,), dtype=float
        )

    def test_select_action_input_output_types(self):
        state = torch.randn(self.input_states)
        print("STATE: ", state)
        action = self.dqn.select_action(state)
        print("ACTION: ", action)
        print("ACTION DATA: ", action.item())

        self.assertIsInstance(state, torch.Tensor)
        self.assertIsInstance(action, int)

    # def test_learn_input_types(self):
    #     batch_state = torch.randn(32, self.input_states)
    #     batch_next_state = torch.randn(32, self.input_states)
    #     batch_reward = torch.randn(32)
    #     batch_action = torch.randint(self.output_actions, (32,))

    #     self.dqn.learn(batch_state, batch_next_state, batch_reward, batch_action)

    #     self.assertIsInstance(batch_state, torch.Tensor)
    #     self.assertIsInstance(batch_next_state, torch.Tensor)
    #     self.assertIsInstance(batch_reward, torch.Tensor)
    #     self.assertIsInstance(batch_action, torch.Tensor)

    # def test_update_input_output_types(self):
    #     reward = 0.5
    #     new_signal = [0.1] * self.input_states

    #     action = self.dqn.update(reward, new_signal)

    #     self.assertIsInstance(reward, float)
    #     self.assertIsInstance(new_signal, list)
    #     self.assertIsInstance(action, int)

    # def test_score_output_type(self):
    #     score = self.dqn.score()

    #     self.assertIsInstance(score, float)

    # def test_save_load(self):
    #     self.dqn.save()

    #     dqn2 = DQN(self.input_states, self.output_actions, self.gamma)
    #     dqn2.load()

    #     self.assertEqual(self.dqn.model.state_dict(), dqn2.model.state_dict())
    #     self.assertEqual(self.dqn.optimizer.state_dict(), dqn2.optimizer.state_dict())


if __name__ == "__main__":
    unittest.main()
