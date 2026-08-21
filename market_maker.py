import math
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

AJARAI_NAME: Final[str] = "AJR"
AJARAI_UNDERLYING_ID: Final[int] = 2
FED_FUNDS_RATE_NAME: Final[str] = "FED"
FED_FUNDS_RATE_UNDERLYING_ID: Final[int] = 1
RATE_STRIKE_GRID: Final[float] = 0.25
THERIODIC_NAME: Final[str] = "THR"
THERIODIC_UNDERLYING_ID: Final[int] = 3

UNDERLYING_NAME_BY_ID: Final[dict[int, str]] = {
    AJARAI_UNDERLYING_ID: AJARAI_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME,
    THERIODIC_UNDERLYING_ID: THERIODIC_NAME,
}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class BinaryOption:
    legs: "tuple[OptionLeg, ...]"
    option_id: int
    steps_until_expiry: int
    strike: float

    def __post_init__(self) -> None:
        if self.steps_until_expiry < 0:
            raise ValueError("Steps until expiry must be non-negative")

        if not self.legs:
            raise ValueError("Binary option must have at least one leg")

        underlying_ids: list[int] = [leg.underlying_id for leg in self.legs]
        if len(underlying_ids) != len(set(underlying_ids)):
            raise ValueError("Binary option legs must reference distinct underlyings")

        if any(leg.weight == 0 for leg in self.legs):
            raise ValueError("Binary option leg weights must be non-zero")

    def __str__(self) -> str:
        terms: list[str] = []
        for index, leg in enumerate(self.legs):
            name: str = UNDERLYING_NAME_BY_ID.get(leg.underlying_id, str(leg.underlying_id))
            magnitude: float = abs(leg.weight)
            magnitude_str: str = "" if magnitude == 1 else f"{magnitude:.2f}*"
            if index == 0:
                sign: str = "-" if leg.weight < 0 else ""
            else:
                sign = " - " if leg.weight < 0 else " + "
            terms.append(f"{sign}{magnitude_str}{name}")
        observable_expression: str = "".join(terms)
        return f"{self.option_id} ({self.steps_until_expiry}d {observable_expression} >= {self.strike:.2f})"

    def advance_step(self) -> "BinaryOption":
        if self.steps_until_expiry == 0:
            return self

        return replace(self, steps_until_expiry=self.steps_until_expiry - 1)

    def contract_matches(self, other: "BinaryOption") -> bool:
        return replace(other, option_id=self.option_id) == self

    def expiry_valuation(self, value_by_underlying_id: dict[int, float]) -> float:
        return 1.0 if self.observable_value(value_by_underlying_id) >= self.strike else 0.0

    def observable_value(self, value_by_underlying_id: dict[int, float]) -> float:
        return sum(leg.weight * value_by_underlying_id[leg.underlying_id] for leg in self.legs)


@dataclass(frozen=True)
class FokOrder:
    counterparty_id: int
    option_id: int
    order_type: "OrderType"
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("FOK order price must be non-negative")

        if self.quantity <= 0:
            raise ValueError("FOK order quantity must be positive")


@dataclass(frozen=True)
class MarketHistory:
    values_by_underlying_id: dict[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        lengths: set[int] = {len(values) for values in self.values_by_underlying_id.values()}
        if len(lengths) > 1:
            raise ValueError("All underlyings must have the same number of historical days")

        if lengths and next(iter(lengths)) <= 0:
            raise ValueError("Market history must contain at least one day")

    @property
    def num_days(self) -> int:
        if not self.values_by_underlying_id:
            return 0
        return len(next(iter(self.values_by_underlying_id.values())))


@dataclass(frozen=True)
class MarketParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_up_probability: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float

    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self) -> None:
        if self.rate_step <= 0:
            raise ValueError("Rate step must be positive")

        if self.rate_up_probability <= 0 or self.rate_down_probability <= 0:
            raise ValueError("Rate up/down probabilities must both be positive")

        if self.rate_up_probability + self.rate_down_probability > 1:
            raise ValueError("Rate up/down probabilities must not sum to more than 1")

        if self.rate_target < 0:
            raise ValueError("Rate target must be non-negative")

        if not (0 <= self.rate_reversion_strength <= 1):
            raise ValueError("Rate reversion strength must be between 0 and 1")

        if self.ajarai_idio_std_dev < 0 or self.theriodic_idio_std_dev < 0 or self.sector_std_dev < 0:
            raise ValueError("Standard deviations must be non-negative")

    def advance_company_value(
        self,
        current_value: float,
        rate_change: float,
        sector_shock: float,
        *,
        drift: float,
        rate_beta: float,
        sector_beta: float,
        idio_std_dev: float,
    ) -> float:
        idiosyncratic_shock: float = random.gauss(mu=0.0, sigma=idio_std_dev)
        log_return: float = drift + (rate_beta * rate_change) + (sector_beta * sector_shock) + idiosyncratic_shock
        return round(current_value * math.exp(log_return), 2)

    def advance_rate(self, rate_value: float) -> float:
        up_probability, down_probability = self.tilted_rate_probabilities(rate_value)
        draw: float = random.random()
        if draw < up_probability:
            return self.next_rate_value(rate_value, 1)

        if draw < up_probability + down_probability:
            return self.next_rate_value(rate_value, -1)

        return rate_value

    def advance_step(self, value_by_underlying_id: dict[int, float]) -> dict[int, float]:
        current_rate_value: float = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_value: float = self.advance_rate(current_rate_value)
        rate_change: float = round(rate_value - current_rate_value, 2)
        sector_shock: float = random.gauss(mu=0.0, sigma=self.sector_std_dev)
        return {
            FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
            AJARAI_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[AJARAI_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.ajarai_drift,
                rate_beta=self.ajarai_rate_beta,
                sector_beta=self.ajarai_sector_beta,
                idio_std_dev=self.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[THERIODIC_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.theriodic_drift,
                rate_beta=self.theriodic_rate_beta,
                sector_beta=self.theriodic_sector_beta,
                idio_std_dev=self.theriodic_idio_std_dev,
            ),
        }

    def next_rate_value(self, rate_value: float, num_grid_steps: int) -> float:
        return max(round(rate_value + num_grid_steps * self.rate_step, 2), 0.0)

    def tilted_rate_probabilities(self, rate_value: float) -> tuple[float, float]:
        tilt: float = self.rate_reversion_strength * (self.rate_target - rate_value)
        up_probability: float = min(max(self.rate_up_probability + tilt, 0.0), 1.0)
        down_probability: float = min(max(self.rate_down_probability - tilt, 0.0), 1.0 - up_probability)
        return up_probability, down_probability


@dataclass(frozen=True)
class OptionLeg:
    underlying_id: int
    weight: float


class OrderType(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Position:
    def __init__(self) -> None:
        self.option_quantity_by_option_id: dict[int, int] = defaultdict(int)

    def add_option_quantity(self, option_id: int, quantity: int) -> None:
        self.option_quantity_by_option_id[option_id] += quantity


@dataclass(frozen=True)
class Quote:
    bid_price: float
    bid_quantity: int
    offer_price: float
    offer_quantity: int

    def __post_init__(self) -> None:
        if self.bid_quantity <= 0 or self.offer_quantity <= 0:
            raise ValueError("Quote quantities must be positive")

        if not (0.0 <= self.bid_price <= 1.0 and 0.0 <= self.offer_price <= 1.0):
            raise ValueError("Quote prices must be between 0 and 1")

        if self.bid_price >= self.offer_price:
            raise ValueError("Quote bid price must be less than offer price")

        if any(abs(round(price * 100) - price * 100) > 1e-6 for price in (self.bid_price, self.offer_price)):
            raise ValueError("Quote prices must be in whole pennies (multiples of 0.01)")


@dataclass(frozen=True)
class Underlying:
    name: str
    underlying_id: int
    value: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Underlying):
            return False
        return self.underlying_id == other.underlying_id

# ============================================================================
# YOUR MARKET MAKER -- fill in the six stubbed methods below
# ============================================================================


class MarketMaker:
    LIVE_PATHS: Final[int] = 4_000
    THEO_PATHS: Final[int] = 25_000
    MONTE_CARLO_SEED: Final[int] = 8_675_309
    HALF_SPREAD: Final[float] = 0.01
    FOK_EDGE: Final[float] = 0.01
    BASE_QUANTITY: Final[int] = 2
    INVENTORY_LIMIT: Final[int] = 10
    SKEW_PER_CONTRACT: Final[float] = 0.005
    MAX_INVENTORY_SKEW: Final[float] = 0.05
    MIN_ESTIMATION_TRANSITIONS: Final[int] = 5
    PROBABILITY_FLOOR: Final[float] = 0.001
    RATE_REVERSION_PRIOR_TRANSITIONS: Final[float] = 50.0
    DRIFT_SCALE: Final[float] = 0.005
    MAX_CACHED_RANDOM_STEPS: Final[int] = 10
    EPSILON: Final[float] = 1e-12
    MAX_SIMULATED_VALUE: Final[float] = 1e300

    def __init__(
        self,
        underlying_initial_state: list[Underlying],
        option_initial_state: list[BinaryOption],
        cash_balance: float,
    ) -> None:
        self.underlying_state: list[Underlying] = underlying_initial_state
        self.active_option_state: list[BinaryOption] = option_initial_state
        self.cash_balance: float = cash_balance
        self.position: Position = Position()
        self.remaining_risk_budget: float = max(0.0, cash_balance)
        self.long_quantity_by_option_id: dict[int, int] = defaultdict(int)
        self.short_quantity_by_option_id: dict[int, int] = defaultdict(int)
        self.estimated_parameters: MarketParameters = self.default_parameters()
        self.live_price_by_option: dict[BinaryOption, float] = {}
        self.live_terminal_values_by_expiry: dict[
            int, list[dict[int, float]]
        ] = {}
        self.live_standard_draws: list[tuple[float, float, float, float]] = []

    def on_step_advance(self,new_underlying_state: list[Underlying], new_option_state: list[BinaryOption]
    ) -> None:
        self.release_expiry_collateral(new_underlying_state, new_option_state)
        self.underlying_state = new_underlying_state
        self.active_option_state = new_option_state
        self.live_price_by_option.clear()
        self.live_terminal_values_by_expiry.clear()

    def on_trade(
        self, option: BinaryOption, price: float, quantity: int, counterparty_id: int) -> None:
        self.position.add_option_quantity(option.option_id, quantity)
        if quantity > 0:
            self.long_quantity_by_option_id[option.option_id] += quantity
            maximum_loss = max(0.0, price) * quantity
        else:
            self.short_quantity_by_option_id[option.option_id] += abs(quantity)
            maximum_loss = max(0.0, 1.0 - price) * abs(quantity)
        self.remaining_risk_budget = max(0.0, self.remaining_risk_budget - maximum_loss)

    def release_expiry_collateral(self, new_underlying_state: list[Underlying],
        new_option_state: list[BinaryOption]) -> None:
        new_options_by_id = {option.option_id: option for option in new_option_state}
        new_values = {
            underlying.underlying_id: underlying.value
            for underlying in new_underlying_state
        }
        for option in self.active_option_state:
            next_option = new_options_by_id.get(option.option_id)
            still_active = (
                next_option is not None and next_option.steps_until_expiry > 0
            )
            if option.steps_until_expiry != 1 or still_active:
                continue

            payoff = option.expiry_valuation(new_values)
            option_id = option.option_id
            credit = self.long_quantity_by_option_id.get(option_id, 0) * payoff
            credit += self.short_quantity_by_option_id.get(option_id, 0) * (
                1.0 - payoff
            )
            self.remaining_risk_budget += credit
            self.long_quantity_by_option_id.pop(option_id, None)
            self.short_quantity_by_option_id.pop(option_id, None)
            self.position.option_quantity_by_option_id.pop(option_id, None)

    @property
    def name(self) -> str:
        return "Clever Market Making Bot"

    def price_option(self, option: BinaryOption) -> float:
        if option not in self.live_price_by_option:
            self.live_price_by_option[option] = self.price_with_parameters(
                self.estimated_parameters, option, self.LIVE_PATHS
            )
        return self.live_price_by_option[option]

    def price_option_from_parameters(
            self, market_parameters: MarketParameters, option: BinaryOption) -> float:
        return self.price_with_parameters(market_parameters, option, self.THEO_PATHS)

    def quote(self, option: BinaryOption, counterparty_id: int) -> Quote:
        reservation_price = self.reservation_price(option)
        bid_price = max(
            0.0,
            min(0.99, math.floor((reservation_price - self.HALF_SPREAD) * 100 + self.EPSILON)/ 100))
        offer_price = max(
            0.01,
            min(1.0, math.ceil((reservation_price + self.HALF_SPREAD) * 100 - self.EPSILON)/ 100))

        if bid_price >= offer_price:
            if offer_price < 1.0:
                offer_price = round(offer_price + 0.01, 2)
            else:
                bid_price = round(bid_price - 0.01, 2)

        position = self.get_position(option.option_id)
        bid_quantity = self.safe_quote_quantity(
            inventory_room=self.INVENTORY_LIMIT - position,
            maximum_loss_per_contract=bid_price,
        )
        offer_quantity = self.safe_quote_quantity(
            inventory_room=self.INVENTORY_LIMIT + position,
            maximum_loss_per_contract=1.0 - offer_price,
        )

        if bid_quantity == 0:
            bid_price = 0.0
            bid_quantity = 1
        if offer_quantity == 0:
            offer_price = 1.0
            offer_quantity = 1

        return Quote(
            bid_price=round(bid_price, 2),
            bid_quantity=bid_quantity,
            offer_price=round(offer_price, 2),
            offer_quantity=offer_quantity,
        )

    def respond_to_fok(self, option: BinaryOption, fok_order: FokOrder) -> bool:
        """Decline all fill-or-kill orders; this strategy supplies RFQ liquidity only."""
        return False

    def warm_up(self, market_history: MarketHistory) -> None:
        defaults = self.default_parameters()
        estimates: dict[str, float] = vars(defaults).copy()
        histories = market_history.values_by_underlying_id
        rates = histories.get(FED_FUNDS_RATE_UNDERLYING_ID)
        self.add_rate_estimates(estimates, rates)
        company_results = self.company_estimates(histories, rates)
        self.add_company_estimates(estimates, company_results)
        self.estimated_parameters = self.valid_parameters(estimates, defaults)
        self.live_price_by_option.clear()
        self.live_terminal_values_by_expiry.clear()

    def add_rate_estimates(self, estimates: dict[str, float], 
        rates: tuple[float, ...] | None) -> None:
        if rates is None:
            return
        rate_estimate = self.estimate_rate_parameters(rates)
        if rate_estimate is None:
            return
        rate_step = self.estimate_rate_step(rates)
        if rate_step is not None:
            estimates["rate_step"] = rate_step
        up_probability, down_probability, reversion_strength = rate_estimate
        estimates["rate_up_probability"] = up_probability
        estimates["rate_down_probability"] = down_probability
        estimates["rate_reversion_strength"] = reversion_strength

    def company_estimates(self, histories: dict[int, tuple[float, ...]],
        rates: tuple[float, ...] | None) -> dict[int, tuple[float, float, dict[int, float]]]:

        if rates is None:
            return {}
        results: dict[int, tuple[float, float, dict[int, float]]] = {}

        for underlying_id in (AJARAI_UNDERLYING_ID, THERIODIC_UNDERLYING_ID):
            values = histories.get(underlying_id)
            if values is None:
                continue
            result = self.estimate_company_parameters(rates, values)
            if result is not None:
                results[underlying_id] = result
        return results

    def add_company_estimates(self, estimates: dict[str, float],
        company_results: dict[int, tuple[float, float, dict[int, float]]]) -> None:

        ajarai_result = company_results.get(AJARAI_UNDERLYING_ID)
        theriodic_result = company_results.get(THERIODIC_UNDERLYING_ID)
        if ajarai_result is not None:
            estimates["ajarai_drift"] = ajarai_result[0]
            estimates["ajarai_rate_beta"] = ajarai_result[1]
        if theriodic_result is not None:
            estimates["theriodic_drift"] = theriodic_result[0]
            estimates["theriodic_rate_beta"] = theriodic_result[1]

        if ajarai_result is not None and theriodic_result is not None:
            self.set_joint_residual_estimates(
                estimates, ajarai_result[2], theriodic_result[2]
            )
        else:
            if ajarai_result is not None:
                estimates["ajarai_sector_beta"] = 0.0
                estimates["ajarai_idio_std_dev"] = self.residual_std_dev(
                    list(ajarai_result[2].values())
                )
            if theriodic_result is not None:
                estimates["theriodic_sector_beta"] = 0.0
                estimates["theriodic_idio_std_dev"] = self.residual_std_dev(
                    list(theriodic_result[2].values())
                )

    @staticmethod
    def valid_parameters(estimates: dict[str, float], 
        defaults: MarketParameters) -> MarketParameters:
        try:
            candidate = MarketParameters(**estimates)
        except (TypeError, ValueError):
            return defaults
        return (
            candidate
            if all(
                math.isfinite(getattr(candidate, field_name))
                for field_name in estimates
            )
            else defaults
        )

    @staticmethod
    def default_parameters() -> MarketParameters:
        return MarketParameters(
            ajarai_drift=0.0,
            ajarai_idio_std_dev=0.02,
            ajarai_rate_beta=0.0,
            ajarai_sector_beta=0.01,
            rate_down_probability=0.20,
            rate_reversion_strength=0.05,
            rate_up_probability=0.20,
            sector_std_dev=1.0,
            theriodic_drift=0.0,
            theriodic_idio_std_dev=0.02,
            theriodic_rate_beta=0.0,
            theriodic_sector_beta=0.01,
            rate_step=RATE_STRIKE_GRID,
            rate_target=2.0,
        )

    def current_values(self) -> dict[int, float]:
        return {
            underlying.underlying_id: underlying.value
            for underlying in self.underlying_state
        }

    def price_with_parameters(self, market_parameters: MarketParameters, 
                              option: BinaryOption, number_of_paths: int) -> float:
        current_values = self.current_values()
        if option.steps_until_expiry == 0:
            return option.expiry_valuation(current_values)

        if all(
            leg.underlying_id == FED_FUNDS_RATE_UNDERLYING_ID for leg in option.legs
        ):
            probability = self.price_rate_option_exactly(
                market_parameters, option, current_values
            )
        else:
            probability = self.price_company_option_by_simulation(
                market_parameters,
                option,
                current_values,
                number_of_paths,
            )
        return min(max(probability, 0.0), 1.0)

    @staticmethod
    def price_rate_option_exactly(market_parameters: MarketParameters, option: BinaryOption,
        current_values: dict[int, float]) -> float:
        
        initial_rate = current_values[FED_FUNDS_RATE_UNDERLYING_ID]
        probability_by_rate: dict[float, float] = {initial_rate: 1.0}
        for _ in range(option.steps_until_expiry):
            next_probability_by_rate: dict[float, float] = defaultdict(float)
            for rate_value, state_probability in probability_by_rate.items():
                up_probability, down_probability = (
                    market_parameters.tilted_rate_probabilities(rate_value)
                )
                next_probability_by_rate[
                    market_parameters.next_rate_value(rate_value, 1)
                ] += state_probability * up_probability
                next_probability_by_rate[
                    market_parameters.next_rate_value(rate_value, -1)
                ] += state_probability * down_probability
                next_probability_by_rate[rate_value] += state_probability * (
                    1.0 - up_probability - down_probability
                )
            probability_by_rate = next_probability_by_rate

        return sum(
            state_probability
            for rate_value, state_probability in probability_by_rate.items()
            if option.expiry_valuation({FED_FUNDS_RATE_UNDERLYING_ID: rate_value})
        )

    def price_company_option_by_simulation(self, market_parameters: MarketParameters,
        option: BinaryOption, current_values: dict[int, float], number_of_paths: int
    ) -> float:
        use_shared_path_cache = (
            market_parameters is self.estimated_parameters
            and number_of_paths == self.LIVE_PATHS
            and current_values == self.current_values()
            and self.active_company_option_count(option.steps_until_expiry) > 1
        )
        terminal_values: Iterable[dict[int, float]]
        if use_shared_path_cache:
            cached_terminal_values = self.live_terminal_values_by_expiry.get(
                option.steps_until_expiry
            )
            if cached_terminal_values is None:
                cached_terminal_values = list(
                    self.simulate_terminal_values(
                        market_parameters,
                        current_values,
                        option.steps_until_expiry,
                        number_of_paths,
                    )
                )
                self.live_terminal_values_by_expiry[
                    option.steps_until_expiry
                ] = cached_terminal_values
            terminal_values = cached_terminal_values
        else:
            terminal_values = self.simulate_terminal_values(
                market_parameters,
                current_values,
                option.steps_until_expiry,
                number_of_paths,
            )
        in_the_money_paths = sum(
            int(option.expiry_valuation(path_values))
            for path_values in terminal_values
        )
        return in_the_money_paths / number_of_paths

    def active_company_option_count(self, steps_until_expiry: int) -> int:
        return sum(
            1
            for option in self.active_option_state
            if option.steps_until_expiry == steps_until_expiry
            and any(
                leg.underlying_id != FED_FUNDS_RATE_UNDERLYING_ID
                for leg in option.legs
            )
        )

    def simulate_terminal_values(self, market_parameters: MarketParameters,
        current_values: dict[int, float], steps_until_expiry: int, 
        number_of_paths: int) -> Iterator[dict[int, float]]:
        
        standard_draws = self.simulation_standard_draws(
            steps_until_expiry, number_of_paths
        )
        initial_rate = current_values[FED_FUNDS_RATE_UNDERLYING_ID]
        initial_ajarai = current_values[AJARAI_UNDERLYING_ID]
        initial_theriodic = current_values[THERIODIC_UNDERLYING_ID]
        rate_transition_by_value: dict[
            float, tuple[float, float, float, float]
        ] = {}
        tilted_rate_probabilities = market_parameters.tilted_rate_probabilities
        next_rate_value = market_parameters.next_rate_value
        advance_company = self.advance_company_with_shock
        sector_std_dev = market_parameters.sector_std_dev
        ajarai_drift = market_parameters.ajarai_drift
        ajarai_rate_beta = market_parameters.ajarai_rate_beta
        ajarai_sector_beta = market_parameters.ajarai_sector_beta
        ajarai_idio_std_dev = market_parameters.ajarai_idio_std_dev
        theriodic_drift = market_parameters.theriodic_drift
        theriodic_rate_beta = market_parameters.theriodic_rate_beta
        theriodic_sector_beta = market_parameters.theriodic_sector_beta
        theriodic_idio_std_dev = market_parameters.theriodic_idio_std_dev

        for _ in range(number_of_paths):
            rate_value = initial_rate
            ajarai_value = initial_ajarai
            theriodic_value = initial_theriodic

            for _ in range(steps_until_expiry):
                (
                    rate_draw,
                    sector_standard_shock,
                    ajarai_standard_shock,
                    theriodic_standard_shock,
                ) = next(standard_draws)
                rate_transition = rate_transition_by_value.get(rate_value)
                if rate_transition is None:
                    up_probability, down_probability = tilted_rate_probabilities(
                        rate_value
                    )
                    rate_transition = (
                        up_probability,
                        down_probability,
                        next_rate_value(rate_value, 1),
                        next_rate_value(rate_value, -1),
                    )
                    rate_transition_by_value[rate_value] = rate_transition
                (
                    up_probability,
                    down_probability,
                    up_rate_value,
                    down_rate_value,
                ) = rate_transition
                if rate_draw < up_probability:
                    next_rate = up_rate_value
                elif rate_draw < up_probability + down_probability:
                    next_rate = down_rate_value
                else:
                    next_rate = rate_value

                rate_change = round(next_rate - rate_value, 2)
                sector_shock = (
                    0.0
                    + sector_standard_shock * sector_std_dev
                )
                ajarai_value = advance_company(
                    ajarai_value,
                    rate_change,
                    sector_shock,
                    0.0
                    + ajarai_standard_shock
                    * ajarai_idio_std_dev,
                    drift=ajarai_drift,
                    rate_beta=ajarai_rate_beta,
                    sector_beta=ajarai_sector_beta,
                )
                theriodic_value = advance_company(
                    theriodic_value,
                    rate_change,
                    sector_shock,
                    0.0
                    + theriodic_standard_shock
                    * theriodic_idio_std_dev,
                    drift=theriodic_drift,
                    rate_beta=theriodic_rate_beta,
                    sector_beta=theriodic_sector_beta,
                )
                rate_value = next_rate
            yield {
                FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
                AJARAI_UNDERLYING_ID: ajarai_value,
                THERIODIC_UNDERLYING_ID: theriodic_value,
            }

    def simulation_standard_draws(self, steps_until_expiry: int, 
        number_of_paths: int) -> Iterator[tuple[float, float, float, float]]:

        number_of_draws = steps_until_expiry * number_of_paths
        use_live_cache = (
            number_of_paths == self.LIVE_PATHS
            and steps_until_expiry <= self.MAX_CACHED_RANDOM_STEPS
        )
        if not use_live_cache:
            return self.generate_standard_draws(number_of_draws)
        if len(self.live_standard_draws) < number_of_draws:
            return self.generate_and_cache_standard_draws(number_of_draws)
        return iter(self.live_standard_draws)

    def generate_and_cache_standard_draws(self, number_of_draws: int
        ) -> Iterator[tuple[float, float, float, float]]:

        generated_draws = []
        for standard_draw in self.generate_standard_draws(number_of_draws):
            generated_draws.append(standard_draw)
            if (
                len(generated_draws) == number_of_draws
                and len(self.live_standard_draws) < number_of_draws
            ):
                self.live_standard_draws = generated_draws
            yield standard_draw

    def generate_standard_draws(self, 
        number_of_draws: int) -> Iterator[tuple[float, float, float, float]]:

        random_generator = random.Random(self.MONTE_CARLO_SEED)
        for _ in range(number_of_draws):
            yield (
                random_generator.random(),
                random_generator.gauss(0.0, 1.0),
                random_generator.gauss(0.0, 1.0),
                random_generator.gauss(0.0, 1.0),
            )

    @classmethod
    def advance_company_with_generator(cls, random_generator: random.Random, current_value: float,
        rate_change: float, sector_shock: float, *, drift: float, rate_beta: float, sector_beta: float,
        idio_std_dev: float) -> float:

        idiosyncratic_shock = random_generator.gauss(0.0, idio_std_dev)
        return cls.advance_company_with_shock(
            current_value,
            rate_change,
            sector_shock,
            idiosyncratic_shock,
            drift=drift,
            rate_beta=rate_beta,
            sector_beta=sector_beta,
        )

    @classmethod
    def advance_company_with_shock(cls, current_value: float, rate_change: float, sector_shock: float, 
            idiosyncratic_shock: float, *, drift: float, rate_beta: float, sector_beta: float) -> float:

        log_return = (
            drift
            + rate_beta * rate_change
            + sector_beta * sector_shock
            + idiosyncratic_shock
        )
        if current_value == 0.0:
            return 0.0
        try:
            next_value = current_value * math.exp(log_return)
        except OverflowError:
            next_value = cls.MAX_SIMULATED_VALUE if log_return > 0.0 else 0.0
        if not math.isfinite(next_value) or next_value >= cls.MAX_SIMULATED_VALUE:
            return cls.MAX_SIMULATED_VALUE if next_value > 0.0 else 0.0
        return round(max(0.0, next_value), 2)

    def reservation_price(self, option: BinaryOption) -> float:
        fair_value = self.price_option(option)
        inventory = self.get_position(option.option_id)
        inventory_skew = min(
            max(inventory * self.SKEW_PER_CONTRACT, -self.MAX_INVENTORY_SKEW),
            self.MAX_INVENTORY_SKEW,
        )
        return min(max(fair_value - inventory_skew, 0.0), 1.0)

    def get_position(self, option_id: int) -> int:
        return self.position.option_quantity_by_option_id.get(option_id, 0)

    def estimate_rate_step(self, rate_values: tuple[float, ...]) -> float | None:
        move_counts: dict[float, int] = defaultdict(int)
        for index in range(len(rate_values) - 1):
            current_rate = rate_values[index]
            next_rate = rate_values[index + 1]
            if not math.isfinite(current_rate) or not math.isfinite(next_rate):
                continue
            move = round(abs(next_rate - current_rate), 2)
            if move > self.EPSILON:
                move_counts[move] += 1
        if not move_counts:
            return None
        most_common_move = max(
            move_counts, key=lambda move: (move_counts[move], -move)
        )
        return most_common_move if move_counts[most_common_move] >= 2 else None

    def safe_quote_quantity(self, *, inventory_room: int, maximum_loss_per_contract: float) -> int:
        if inventory_room <= 0:
            return 0
        if maximum_loss_per_contract <= self.EPSILON:
            affordable_quantity = self.BASE_QUANTITY
        elif self.remaining_risk_budget <= self.EPSILON:
            return 0
        else:
            affordable_quantity = math.floor(
                (self.remaining_risk_budget + self.EPSILON) / maximum_loss_per_contract
            )
        return max(0, min(self.BASE_QUANTITY, inventory_room, affordable_quantity))

    def estimate_rate_parameters(self, rate_values: tuple[float, ...]) -> tuple[float, float, float] | None:
        valid_transitions = [
            (rate_values[index], rate_values[index + 1])
            for index in range(len(rate_values) - 1)
            if math.isfinite(rate_values[index])
            and math.isfinite(rate_values[index + 1])
        ]
        if len(valid_transitions) < self.MIN_ESTIMATION_TRANSITIONS:
            return None

        target = 2.0
        distances: list[float] = []
        up_indicators: list[float] = []
        down_indicators: list[float] = []
        for current_rate, next_rate in valid_transitions:
            distances.append(target - current_rate)
            up_indicators.append(
                1.0 if next_rate > current_rate + self.EPSILON else 0.0
            )
            down_indicators.append(
                1.0 if next_rate < current_rate - self.EPSILON else 0.0
            )

        try:
            mean_distance = self.mean(distances)
            variance_distance = self.variance(distances)
            mean_up = self.mean(up_indicators)
            mean_down = self.mean(down_indicators)
            if variance_distance <= self.EPSILON:
                reversion_strength = 0.0
            else:
                covariance_up = self.covariance(distances, up_indicators)
                covariance_down = self.covariance(distances, down_indicators)
                reversion_strength = min(
                    max(0.0, (covariance_up - covariance_down) / (2.0 * variance_distance)),
                    1.0,
                )
                history_weight = len(valid_transitions) / (
                    len(valid_transitions) + self.RATE_REVERSION_PRIOR_TRANSITIONS
                )
                reversion_strength *= history_weight
            up_probability = mean_up - reversion_strength * mean_distance
            down_probability = mean_down + reversion_strength * mean_distance
        except (OverflowError, ValueError):
            return None

        if not all(
            math.isfinite(value)
            for value in (up_probability, down_probability, reversion_strength)
        ):
            return None
        up_probability = max(up_probability, self.PROBABILITY_FLOOR)
        down_probability = max(down_probability, self.PROBABILITY_FLOOR)
        probability_sum = up_probability + down_probability
        if probability_sum > 0.999:
            scale = 0.999 / probability_sum
            up_probability *= scale
            down_probability *= scale
        return up_probability, down_probability, reversion_strength

    def estimate_company_parameters(self, rate_values: tuple[float, ...], 
                company_values: tuple[float, ...]) -> tuple[float, float, dict[int, float]] | None:
        
        observations: list[tuple[int, float, float]] = []
        for index in range(1, min(len(rate_values), len(company_values))):
            previous_value = company_values[index - 1]
            current_value = company_values[index]
            if (
                previous_value <= 0.0
                or current_value <= 0.0
                or not math.isfinite(previous_value)
                or not math.isfinite(current_value)
                or not math.isfinite(rate_values[index - 1])
                or not math.isfinite(rate_values[index])
            ):
                continue
            rate_change = rate_values[index] - rate_values[index - 1]
            log_return = math.log(current_value) - math.log(previous_value)
            if math.isfinite(rate_change) and math.isfinite(log_return):
                observations.append((index, rate_change, log_return))

        if len(observations) < self.MIN_ESTIMATION_TRANSITIONS:
            return None

        rate_changes = [observation[1] for observation in observations]
        log_returns = [observation[2] for observation in observations]
        try:
            variance_rate_change = self.variance(rate_changes)
            has_rate_variation = variance_rate_change > self.EPSILON
            if not has_rate_variation:
                rate_beta = 0.0
            else:
                rate_beta = (
                    self.covariance(rate_changes, log_returns) / variance_rate_change
                )
            drift = self.mean(log_returns) - rate_beta * self.mean(rate_changes)
            preliminary_residuals = {
                index: log_return - drift - rate_beta * rate_change
                for index, rate_change, log_return in observations
            }
            residual_variance = self.variance(list(preliminary_residuals.values()))
            observation_count = len(observations)
            mean_rate_change = self.mean(rate_changes)
            centered_rate_sum = variance_rate_change * observation_count
            if has_rate_variation:
                error_variance = (
                    residual_variance * observation_count / (observation_count - 2)
                )
                drift_standard_error_squared = error_variance * (
                    1.0 / observation_count
                    + mean_rate_change**2 / centered_rate_sum
                )
            else:
                error_variance = (
                    residual_variance * observation_count / (observation_count - 1)
                )
                drift_standard_error_squared = error_variance / observation_count
            drift_scale_squared = self.DRIFT_SCALE**2
            drift *= drift_scale_squared / (
                drift_scale_squared + drift_standard_error_squared
            )
            residuals = {
                index: log_return - drift - rate_beta * rate_change
                for index, rate_change, log_return in observations
            }
        except (OverflowError, ValueError):
            return None
        if not math.isfinite(drift) or not math.isfinite(rate_beta):
            return None
        if not all(math.isfinite(residual) for residual in residuals.values()):
            return None
        return drift, rate_beta, residuals

    def set_joint_residual_estimates(self, estimates: dict[str, float], ajarai_residuals: dict[int, float],
        theriodic_residuals: dict[int, float]) -> None:

        common_indices = sorted(ajarai_residuals.keys() & theriodic_residuals.keys())
        if len(common_indices) < self.MIN_ESTIMATION_TRANSITIONS:
            return

        ajarai_values = [ajarai_residuals[index] for index in common_indices]
        theriodic_values = [theriodic_residuals[index] for index in common_indices]
        try:
            ajarai_variance = max(0.0, self.variance(ajarai_values))
            theriodic_variance = max(0.0, self.variance(theriodic_values))
            covariance = self.covariance(ajarai_values, theriodic_values)
        except (OverflowError, ValueError):
            return
        if not all(
            math.isfinite(value)
            for value in (ajarai_variance, theriodic_variance, covariance)
        ):
            return

        estimates["sector_std_dev"] = 1.0
        if ajarai_variance >= theriodic_variance and ajarai_variance > self.EPSILON:
            ajarai_loading = math.sqrt(ajarai_variance)
            theriodic_loading = covariance / ajarai_loading
            estimates["ajarai_sector_beta"] = ajarai_loading
            estimates["theriodic_sector_beta"] = theriodic_loading
            estimates["ajarai_idio_std_dev"] = 0.0
            estimates["theriodic_idio_std_dev"] = math.sqrt(
                max(0.0, theriodic_variance - theriodic_loading**2)
            )
        elif theriodic_variance > self.EPSILON:
            theriodic_loading = math.sqrt(theriodic_variance)
            ajarai_loading = covariance / theriodic_loading
            estimates["ajarai_sector_beta"] = ajarai_loading
            estimates["theriodic_sector_beta"] = theriodic_loading
            estimates["ajarai_idio_std_dev"] = math.sqrt(
                max(0.0, ajarai_variance - ajarai_loading**2)
            )
            estimates["theriodic_idio_std_dev"] = 0.0
        else:
            estimates["ajarai_sector_beta"] = 0.0
            estimates["theriodic_sector_beta"] = 0.0
            estimates["ajarai_idio_std_dev"] = 0.0
            estimates["theriodic_idio_std_dev"] = 0.0

    @classmethod
    def residual_std_dev(cls, residuals: list[float]) -> float:
        return math.sqrt(max(0.0, cls.variance(residuals))) if residuals else 0.0

    @staticmethod
    def mean(values: list[float]) -> float:
        return math.fsum(values) / len(values)

    @classmethod
    def variance(cls, values: list[float]) -> float:
        if not values:
            return 0.0
        mean_value = cls.mean(values)
        return math.fsum((value - mean_value) ** 2 for value in values) / len(values)

    @classmethod
    def covariance(cls, left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        left_mean = cls.mean(left)
        right_mean = cls.mean(right)
        return math.fsum(
            (left_value - left_mean) * (right_value - right_mean)
            for left_value, right_value in zip(left, right)
        ) / len(left)
