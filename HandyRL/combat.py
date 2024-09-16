import numpy as np
import torch
from handyrl.agent import Agent
from handyrl.evaluation import load_model
from handyrl.envs.dots_and_boxes import Environment as DotsnBoxesEnv
from handyrl.envs.dots_and_boxes import SimpleConv2dModel



if __name__ == '__main__':
    env = DotsnBoxesEnv()
    model_path = 'models/latest.pth'
    model = env.net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

