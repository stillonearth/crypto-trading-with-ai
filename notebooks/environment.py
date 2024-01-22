import numpy as np
import csv
import datetime

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    def create_crypto_dataset(self):
        print("[*] Generating Bitcoin Dataset")

        btc_hourly = []
        with open("./datasets/Binance_BTC_USDT_1h.csv") as csvfile:
            tick_reader = csv.DictReader(csvfile, delimiter=",")
            for row in tick_reader:
                date = datetime.datetime.strptime(row["Date"][:19], "%Y-%m-%d %H:%M:%S")
                open_price = float(row["Open"])
                unix_time = int(datetime.datetime.timestamp(date))
                btc_hourly.append((unix_time, open_price))

        btc_hourly = sorted(btc_hourly, key=lambda r: r[0])
        rates = np.array([r[1] for r in btc_hourly])
        dates = np.array([r[0] for r in btc_hourly])

        return (dates, rates)

    def __init__(self, render_mode=None, initial_liquidity=0.0, initial_stock=0.0):
        self.observation_space = spaces.Dict(
            {
                "market": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=float),
                "forecast": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=float),
                "liquidity": spaces.Box(0, np.inf, shape=(1,), dtype=float),
                "stock": spaces.Box(0, np.inf, shape=(1,), dtype=float),
            }
        )

        self.action_space = spaces.Box(0, np.inf, shape=(2,), dtype=float)
        self.render_mode = render_mode
        self.liquidity = initial_liquidity
        self.stock = initial_stock
        self.t = 0
        self.stock_data = self.create_crypto_dataset()

    def _get_obs(self):
        return {
            "market": self.stock_data[self.t][1],
            "forecast": 0.0,
            "liquidity": self.liquidity,
            "stock": self.stock,
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.t = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _estimate_portfolio(self):
        return self.liquidity + self.stock * self.stock_data[self.t][1]

    def step(self, action):
        previous_portfolio_estimate = self._estimate_portfolio()

        self.liquidity -= action[0]
        stock_delta = action[0] / self.stock_data[1]
        self.stock += stock_delta

        self.stock -= action[1]
        liquidity_delta = self.stock_data[1] / action[1]
        self.liquidity += liquidity_delta

        new_portfolio_estimate = self._estimate_portfolio()

        gain_percent = (
            new_portfolio_estimate - previous_portfolio_estimate
        ) / previous_portfolio_estimate

        terminated = self.t == len(self.stock_data) - 1

        return self._get_obs(), gain_percent, terminated, False, self._get_info()

    def render(self):
        pass

    def close(self):
        pass
