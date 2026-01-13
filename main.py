from data_loader import get_price_panel
from config import UNIVERSE

prices = get_price_panel(use_cache=False, universe=UNIVERSE)
print(prices.shape)
print(prices.head())
print(prices.tail())
