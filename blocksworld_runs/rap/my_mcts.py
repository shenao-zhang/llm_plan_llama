# 1. copy paste mcts.py

from colorama import init as colorama_init
colorama_init()
from colorama import Fore
from colorama import Style
import math
import random
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Optional


class MCTSNode(ABC):
    @abstractmethod
    def find_children(self):
        return set()

    @abstractmethod
    def find_one_child(self) -> 'MCTSNode':
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self):
        return True

    @property
    @abstractmethod
    def reward(self):
        return 0

    @property
    @abstractmethod
    def visited(self):
        return 0


class MCTS:
    def __init__(self, w_exp=1, discount=1, prior=False, aggr_reward='sum', aggr_child='max'):
        self.Q: dict[MCTSNode, float] = defaultdict(lambda : 0.) # sum of past returns
        self.N: dict[MCTSNode, int] = defaultdict(lambda : 0) # num of visits paid to states
        self.M: dict[MCTSNode, float] = defaultdict(lambda : -math.inf) # max return achieved previously
        self.children = dict()
        self.w_exp = w_exp
        self.discount = discount
        self.prior = prior
        self.aggr_reward = aggr_reward
        self.aggr_child = aggr_child

    def rollout(self, node: MCTSNode):
        if self.prior: # It seems that we always set prior to be True
            path = self._select_prior(node)
        else:
            path = self._select(node)
            self._expand(path[-1])
            self._simulate(path)
        self._back_propagate(path)

    def _select_prior(self, node: MCTSNode):
        path = [node]
        while not node.is_terminal:
            self._expand(node)
            if len(self.children[node]) == 0:
                return path
            node = self._uct_select(node)
            path.append(node)
        self._expand(node)
        return path

    def _select(self, node: MCTSNode):
        path = []
        while True:
            path.append(node)
            if node not in self.children or node.is_terminal:
                return path
            for child in self.children[node]:
                if child not in self.children.keys():
                    path.append(child)
                    return path
            node = self._uct_select(node)

    def _expand(self, node: MCTSNode):
        if node not in self.children:
            self.children[node] = node.find_children()

    @staticmethod
    def _simulate(path: list[MCTSNode]):
        node = path[-1]
        while not node.is_terminal:
            node = node.find_one_child()
            if node:
                path.append(node)
            else:
                break

    def max_terminal(self, cur: MCTSNode):
        if cur.is_terminal:
            if cur.visited:
                return cur, cur.reward
            else:
                return cur, -math.inf
        if cur not in self.children:
            return cur, -math.inf
        max_n, max_r = max((self.max_terminal(child) for child in self.children[cur]), key=lambda x: x[1])
        return max_n, max_r + cur.reward

    def max_mean_terminal(self, cur: MCTSNode, sum=0., cnt=0):
        if cur.is_terminal:
            if cur.visited:
                return cur, (sum + cur.reward) / (cnt + 1)
            else:
                return cur, -math.inf
        if cur not in self.children or not self.children[cur]:
            return cur, -math.inf
        
        
        return max((self.max_mean_terminal(child, sum + cur.reward, cnt + 1) for child in self.children[cur]), key=lambda x: x[1])

    def _back_propagate(self, path: list[MCTSNode], reward=0.):
        print(f'Start back propagate...')
        coeff = 1
        for node in reversed(path): # For nodes in the path, if node2 is the descendant of node1, then node2 has one more (action, state) pair.
            print(f'{Fore.BLUE}current node.prompt={node.prompt}')
            print(f'{Fore.CYAN}current node.reward={node.reward}')
            print(f'{Fore.CYAN}current node.r0{node._r0}')
            print(f'{Fore.CYAN}current node.r1{node._r1}{Style.RESET_ALL}') # TODO: delete this later
            reward = reward * self.discount + node.reward # accumulate the reward --> get a return at each step. So we are back propagating reward to calculate return
            coeff = coeff * self.discount + 1
            if self.aggr_reward == 'mean':
                c_reward = reward / coeff
            else:
                c_reward = reward
            if node not in self.N:
                self.Q[node] = c_reward
            else:
                self.Q[node] += c_reward
            self.N[node] += 1
            self.M[node] = max(self.M[node], c_reward)

    def _uct(self, node: MCTSNode, log_n_f: float):
        # print("# in _uct (reward, uct)")
        if self.prior and self.N[node] == 0:
            # print("## unexplored: ", node.reward, node.reward + self.w_exp * math.sqrt(log_n_f))
            return node.reward + self.w_exp * math.sqrt(log_n_f)
        if self.aggr_child == 'max':
            # print("## explored: ", self.N[node], self.M[node], self.w_exp * math.sqrt(log_n_f / self.N[node]))
            return self.M[node] + self.w_exp * math.sqrt(log_n_f / self.N[node])
        elif self.aggr_child == 'mean':
            return self.Q[node] / self.N[node] + self.w_exp * math.sqrt(log_n_f / self.N[node])

    def _uct_select(self, node: MCTSNode):
        if self.prior and self.N[node] == 0:
            log_n = math.log(1)
        else:
            log_n = math.log(self.N[node])
        return max(self.children[node], key=lambda n: self._uct(n, log_n))
