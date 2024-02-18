# To constraint that the agent cannot borrow money
# if we buy, that means more inventory (- means we bidding so buy)
self.inventory -= action
# if action < 0, bid (that means we buy)
if action < 0:
    # Calculate the potential cash balance after the purchase
    potential_cash = self.cash + action * bid_avg
    # Check if we have enough cash to cover the purchase
    if potential_cash < 0:
        # Not enough cash - this would mean borrowing money, apply penalty
        reward = -100
    else:
        # Enough cash - proceed with purchase
        diff_per = (new_bid_avg - bid_avg) / bid_avg
        reward += diff_per * -action
        self.cash = potential_cash  # Update cash only if the action is valid
# if action > 0, ask (that means we sell)
elif action > 0:
    diff_per = (ask_avg - new_ask_avg) / new_ask_avg
    reward += diff_per * action
    # Update cash amount from selling stock
    self.cash += action * ask_avg
else:
    # No action taken, apply slight penalty for inactivity
    reward = -0.001
self.total_reward += reward
# End of the day settlement
episode_length = 1000  # num of lines per day - 1
self.current_step += 1
8:23
But after I fully think, I feel like we could limit each action by how much can be bought or sold in one step can be helpful to bond large actions. So I limit 10% cash for each purchase as an example, we could also try from 2% to 20%
8:26
# Limit buying amount (ex. smaller than 10% of Asset) to Avoid Bankruptcy
# To constraint that the agent cannot borrow money and cannot buy more than 10% of the current cash balance
# (we could use 20% 15% 10% 5% to test)
# if we buy, that means more inventory (- means we bidding so buy)
self.inventory -= action
# if action < 0, bid (that means we buy)
if action < 0:
    # Calculate the maximum allowable purchase based on current cash
    max_allowable_purchase = -0.1 * self.cash  # 10% of current cash balance
    # Calculate the potential cash balance after the purchase
    potential_cash = self.cash + action * bid_avg
    # Check if the purchase exceeds the allowable limit
    if action < max_allowable_purchase:
        # Purchase exceeds allowable purchase, apply penalty
        reward = -1
    else:
        # Valid purchase within the constraints
        diff_per = (new_bid_avg - bid_avg) / bid_avg
        reward += diff_per * -action
        self.cash = potential_cash  # Update cash only if the action is valid
# if action > 0, ask (that means we sell)
elif action > 0:
    diff_per = (ask_avg - new_ask_avg) / new_ask_avg
    reward += diff_per * action
    # Update cash amount from selling stock
    self.cash += action * ask_avg
else:
    # No action taken, apply slight penalty for inactivity
    reward = -0.001
self.total_reward += reward
# End of the day settlement
episode_length = 1000  # num of lines per day - 1
self.current_step += 1