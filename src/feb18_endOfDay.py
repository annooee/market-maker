import pandas as pd
import gym
from gym import spaces
import numpy as np
import heapq

def read_csv(file_path):
    return pd.read_csv(file_path)
def process_data(df):
    # Assuming your dataframe is named df
    # Extract the relevant columns
    bids_prices = [f"bids[{i}].price" for i in range(25)]
    bids_amounts = [f"bids[{i}].amount" for i in range(25)]
    asks_prices = [f"asks[{i}].price" for i in range(25)]
    asks_amounts = [f"asks[{i}].amount" for i in range(25)]

    bid_price_data = df[bids_prices].values
    bid_amount_data = df[bids_amounts].values
    ask_price_data = df[asks_prices].values
    ask_amount_data = df[asks_amounts].values

    num_entries = len(bid_price_data)


    return bid_price_data, bid_amount_data, ask_price_data, ask_amount_data, num_entries


# Example usage


####ENVIORNMENT#####
class MarketMakingEnvironment(gym.Env):
    #def __init__(self, bid_prices, bid_amounts, ask_prices, ask_amounts):
    def __init__(self, file_paths):
        print("testing")
        super(MarketMakingEnvironment, self).__init__()

        self.bid_prices = []
        self.bid_amounts = []
        self.ask_prices = []
        self.ask_amounts = []

        #need to keep track of how many lines in this file
        self.num_entries = 0

        self.file_paths = file_paths
        self.current_step = 0
        self.current_file_index = 0

        #also updates how many num_entries for a specific file
        self.load_data()


        self.position = None
        self.start_line = 0
        self.episode_num = 0
        self.total_reward = 0

        self.commission = 0.01
        self.totalMoneyMadeFromMatching = 0


        #for market making only !!!
        self.inventory = 10000
        self.cash = 2500

        # Action space: Continuous action ranging from -1 to 1
        self.action_space = spaces.Box(low=-500, high=500, shape=(1,), dtype=np.float32)

        # Observation space (changes to 4, )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 25), dtype=np.float32)

    def load_data(self):
        if self.current_file_index < len(self.file_paths):
            file_path = self.file_paths[self.current_file_index]
            print("\n\nNEW FILE\n\n",file_path,"\n\n")
            df = read_csv(file_path)
            self.bid_prices, self.bid_amounts, self.ask_prices, self.ask_amounts, self.num_entries = process_data(df)
            print("Number of Entries in this file: ", self.num_entries)

            self.current_step = 0
            self.current_file_index += 1
        else:
            raise Exception("All files has been processed")
    def reset(self):
        self.start_line = self.episode_num * 1000
        self.position = None
        self.episode_num += 1

        #if self.current_step >= len(self.bid_prices):
        if self.current_step >= 999:
            self.load_data()

        return self.get_observation()


    def get_observation(self):
        idx = self.start_line + self.current_step
        return np.array([self.bid_prices[idx],
                         self.bid_amounts[idx],
                         self.ask_prices[idx],
                         self.ask_amounts[idx]], dtype=np.float32)
    def matching(self, line):
        #bid heap (max heap)
        heap1 = list(zip([-price for price in self.bid_prices[line]], self.bid_amounts[line]))
        heapq.heapify(heap1)
        #ask heap (min heap)
        heap2 = list(zip(self.ask_prices[line], self.ask_amounts[line]))
        heapq.heapify(heap2)

        while heap1 and heap2:
            max_bid = heapq.heappop(heap1)
            min_ask = heapq.heappop(heap2)

            bid_price, bid_volume = -max_bid[0], max_bid[1]  # -max bid to reverse ordering in the heap so highest bid price is closer to the root, so highest price gets popped first
            ask_price, ask_volume = min_ask[0], min_ask[1]

            effective_bid_price = bid_price + self.commission
            effective_ask_price = ask_price - self.commission

            spread = effective_bid_price - effective_ask_price

            if spread > 0:
                matched_volume = min(bid_volume, ask_volume)
                remaining_bid_volume = bid_volume - matched_volume
                remaining_ask_volume = ask_volume - matched_volume

                if remaining_bid_volume > 0:
                    heapq.heappush(heap1, (-bid_price, remaining_bid_volume))
                if remaining_ask_volume > 0:
                    heapq.heappush(heap2, (ask_price, remaining_ask_volume))

                #print(f"Matched: {matched_volume} at price {ask_price + spread / 2}")
                self.totalMoneyMadeFromMatching += matched_volume*spread
            else:
                #dont need to process the rest of info
                return

    def get_average(self, idx):
        bid_total_value = sum([self.bid_prices[idx][i] * self.bid_amounts[idx][i] for i in range(25)])
        bid_total_amount = sum(self.bid_amounts[idx])
        ask_total_value = sum([self.ask_prices[idx][i] * self.ask_amounts[idx][i] for i in range(25)])
        ask_total_amount = sum(self.ask_amounts[idx])
        bid_avg = bid_total_value / bid_total_amount if bid_total_amount != 0 else 0
        ask_avg = ask_total_value / ask_total_amount if ask_total_amount != 0 else 0
        return bid_avg, ask_avg

    def step(self, action):
        action = action[0]
        reward = 0
        #find curr line
        idx = self.start_line + self.current_step
        #matching part
        self.matching(idx)
        #market making part
        bid_avg, ask_avg = self.get_average(idx)
        new_bid_avg, new_ask_avg = self.get_average(idx+10)

        #if we buy, that means more inventory (- means we bidding so buy)
        self.inventory -= action
        #assume it always get brought/sold
        # if action < 0, bid (that means we buy)
        # if action > 0, ask (that means we sell)
        if action < 0:
            diff = new_bid_avg-bid_avg
            reward += diff*-action

            #changes to our cash amount
            self.cash += action * bid_avg
        elif action > 0:
            diff = ask_avg-new_ask_avg
            reward += diff*action

            #changes to our cash amount
            self.cash += action * ask_avg
        else:
            #punish for not putting out bidding/asking price
            reward -= 1
        self.total_reward += reward

        self.current_step += 1
        episode_length = 1000
        done = self.current_step % episode_length == 0
        ##done = self.current_step >= len(self.bid_prices) - 1
        if done:
            #handle end of day here
            #assume this is end of day, so we need to get inventory back to 10000
            #sell/buy @ average price
            print("Done 1 DAY!!!")
            print(f"Processing CSV line: {self.start_line + self.current_step}")
            print(f"total rewards: {self.total_reward}")
            print(f"money made from matching: {self.totalMoneyMadeFromMatching}")
            print(f"currentInventory: {self.inventory}")
            print(f"current Cash: {self.cash}")
        reward = float(reward)
        return self.get_observation(), reward, done, {}

    def render(self, mode='human'):
        # Just printing for now
        print(f"Step: {self.current_step}, Bid: {self.bid_prices[self.current_step]}, Ask: {self.ask_prices[self.current_step]}")

    def close(self):
        pass

from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

#"C:\Users\eric7\OneDrive\Desktop\ML_data"
file_paths = [
    f"C:\\ericDataFiles\\ML_data\\binance_book_snapshot_25_{date.strftime('%Y-%m-%d')}_ADAUSDT.csv.gz"
    for date in pd.date_range(start="2023-04-14", end="2023-09-08")
]
###Initialize environment ###
env = DummyVecEnv([lambda: MarketMakingEnvironment(file_paths)])

##DDPG
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)


print("Training started")
model.learn(total_timesteps=10000)
print("Training completed")

print("Testing started")

obs = env.reset()
cumulative_reward = 0


#should train the

# C:\ericDataFiles\ML_data\binance_book_snapshot_25_2023-04-30_ADAUSDT.csv.gz  on the 6000th training step
# starts at 4-14, 10+6 ==== > 4-30... means code transition is correct for testing
#rewards still low, need to improve step function
for i in range(10000):
    print(f"TESTING LOOP --- {i}")
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    #print(f"obs:{obs}, rewards{rewards}, done:{done}")
    cumulative_reward += rewards
    env.render()

print("Total Reward after 100 steps:", cumulative_reward)