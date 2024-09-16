# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import random
import torch
import numpy as np
from HandyRL.handyrl.util import softmax
from HandyRL.handyrl.envs.dots_and_boxes import Environment as DotsnBoxesEnv
from MinMax.BasicPlayers import RandomPlayer


class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return [0.0]


class RuleBasedAgent(RandomAgent):
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action(player, key=self.key)
        else:
            return random.choice(env.legal_actions(player))


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        if v is not None:
            print('v = %f' % v)
        if prob is not None:
            print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, temperature=0.0, observation=True):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
        self.temperature = temperature
        self.observation = observation

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()

    def plan(self, obs):
        outputs = self.model.inference(obs, self.hidden)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False):
        obs = env.observation(player)
        outputs = self.plan(obs)
        actions = env.legal_actions(player)
        p = outputs['policy']
        v = outputs.get('value', None)
        mask = np.ones_like(p)
        mask[actions] = 0
        p = p - mask * 1e32

        if show:
            print_outputs(env, softmax(p), v)

        if self.temperature == 0:
            ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
            return ap_list[0][0]
        else:
            return random.choices(np.arange(len(p)), weights=softmax(p / self.temperature))[0]

    def observe(self, env, player, show=False):
        v = None
        if self.observation:
            obs = env.observation(player)
            outputs = self.plan(obs)
            v = outputs.get('value', None)
            if show:
                print_outputs(env, None, v)
        return v


class EnsembleAgent(Agent):
    def reset(self, env, show=False):
        self.hidden = [model.init_hidden() for model in self.model]

    def plan(self, obs):
        outputs = {}
        for i, model in enumerate(self.model):
            o = model.inference(obs, self.hidden[i])
            for k, v in o.items():
                if k == 'hidden':
                    self.hidden[i] = v
                else:
                    outputs[k] = outputs.get(k, []) + [v]
        for k, vl in outputs.items():
            outputs[k] = np.mean(vl, axis=0)
        return outputs


class SoftAgent(Agent):
    def __init__(self, model):
        super().__init__(model, temperature=1.0)


class RLPlayer(RandomPlayer):
    def __init__(self, index, colour):
        model_path = 'D:\\PycharmProjects\\EmperorJin\\HandyRL\\models\\2.pth'
        self.index = index
        self.colour = colour
        self.env = DotsnBoxesEnv()
        self.model = self.env.net()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def chooseMove(self, game):
        """
        game: class Game
        return: (direction, x, y)
        """
        try:
            raw_state = game.grid
        except Exception:
            raw_state = game
        state = self.ui2env(raw_state)
        print(state)
        state = torch.tensor(state).reshape((1, 1, 11, 6))
        outputs = self.model(state)
        p_ = outputs['policy'].detach()[0]
        p_ = p_.numpy()
        _legal_actions = game.get_all_legal_moves()
        legal_actions = [self.env.site2num(self.ui_action2env_action(a)) for a in _legal_actions]
        action_mask = np.ones_like(p_) * 1e32
        action_mask[legal_actions] = 0
        p = softmax(p_ - action_mask)
        action = random.choices(legal_actions, weights=p[legal_actions])[0]
        site_action = self.env.num2site(action)

        legal_form_action = self.env_action2ui_action(site_action)

        return legal_form_action

    # convert ui-type state to env-type
    def ui2env(self, raw: list):
        d, x, y = 2, 6, 5
        state = np.zeros((11, 6), dtype=np.float32)
        state[::2, -1] = -1
        for i in range(d):
            for j in range(x):
                for k in range(y):
                    if i == 0:
                        state[j*2][k] = -1 if raw[i][j][k].owner == 2 else raw[i][j][k].owner
                    else:  # i = 1
                        state[2*k+1][j] = -1 if raw[i][j][k].owner == 2 else raw[i][j][k].owner

        return state

    def ui_action2env_action(self, action):
        _d, _x, _y = action
        if _d == 0:
            x = 2 * _x
            y = _y
        else:  # i = 1
            x = 2 * _y + 1
            y = _x

        return [x, y]

    def env_action2ui_action(self, env_action):
        x, y = env_action
        if x % 2 == 0:
            _d = 0
            _x = x // 2
            _y = y
        else:
            _d = 1
            _x = y
            _y = x // 2

        legal_form = (_d, _x, _y)

        return legal_form

if __name__ == '__main__':
    agent = RLPlayer(0, 0)
    raw_state = [
        [[0] * 5, [1, 0, 0, 1, 0], [1, 1, 0, 0, 1], [0] * 5, [1] * 5, [1] * 5],
        [[1] * 5, [1, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0] * 5, [0] * 5, [0] * 5]
    ]
    _action = agent.chooseMove(raw_state)
    print(_action)
