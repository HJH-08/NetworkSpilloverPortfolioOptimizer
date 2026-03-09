from data_loader import get_price_panel
from config import UNIVERSE


def main() -> None:
    prices = get_price_panel(use_cache=True, universe=UNIVERSE)
    print("Shape:", prices.shape)
    print(prices.head())
    print(prices.tail())


if __name__ == "__main__":
    main()
