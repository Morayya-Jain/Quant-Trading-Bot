from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import market_maker

STARTING_CASH = 5.0
OPTION_LIFETIME = 3
BASELINE_REF = "2be0a5b"
REPOSITORY_DIR = Path(__file__).resolve().parent
UNDERLYING_START_VALUES = {1: 2.0, 2: 100.0, 3: 100.0}
PROFILE_NAMES = ("noise", "informed", "adverse", "fok")


@dataclass(frozen=True)
class Regime:
    name: str
    warmup_days: int
    parameter_overrides: dict[str, float]


@dataclass(frozen=True)
class OptionSpec:
    option_id: int
    legs: tuple[tuple[int, float], ...]
    strike: float


@dataclass(frozen=True)
class Scenario:
    history: dict[int, tuple[float, ...]]
    start_values: dict[int, float]
    future_values: tuple[dict[int, float], ...]
    options: tuple[OptionSpec, ...]
    parameters: dict[str, float]


@dataclass(frozen=True)
class SessionResult:
    pnl: float
    bankrupt: bool
    trade_count: int


@dataclass(frozen=True)
class Strategy:
    name: str
    module: types.ModuleType
    market_maker_class: type[Any]


def default_parameters() -> dict[str, float]:
    return {
        "ajarai_drift": 0.002,
        "ajarai_idio_std_dev": 0.03,
        "ajarai_rate_beta": -0.08,
        "ajarai_sector_beta": 0.04,
        "rate_down_probability": 0.15,
        "rate_reversion_strength": 0.03,
        "rate_up_probability": 0.25,
        "sector_std_dev": 0.7,
        "theriodic_drift": -0.001,
        "theriodic_idio_std_dev": 0.024,
        "theriodic_rate_beta": 0.04,
        "theriodic_sector_beta": 0.028,
        "rate_step": 0.25,
        "rate_target": 2.0,
    }


def regimes() -> tuple[Regime, ...]:
    return (
        Regime("typical-short", 25, {}),
        Regime("typical-medium", 100, {}),
        Regime("typical-long", 250, {}),
        Regime(
            "high-volatility",
            100,
            {
                "ajarai_idio_std_dev": 0.08,
                "ajarai_sector_beta": 0.07,
                "sector_std_dev": 1.0,
                "theriodic_idio_std_dev": 0.07,
                "theriodic_sector_beta": 0.06,
            },
        ),
        Regime(
            "mean-reverting-rates",
            100,
            {
                "rate_down_probability": 0.12,
                "rate_reversion_strength": 0.10,
                "rate_up_probability": 0.12,
            },
        ),
        Regime(
            "zero-drift",
            100,
            {"ajarai_drift": 0.0, "theriodic_drift": 0.0},
        ),
        Regime(
            "correlated-spread",
            100,
            {
                "ajarai_idio_std_dev": 0.02,
                "ajarai_sector_beta": 0.06,
                "sector_std_dev": 1.0,
                "theriodic_idio_std_dev": 0.025,
                "theriodic_sector_beta": 0.05,
            },
        ),
    )


def load_baseline_module() -> types.ModuleType:
    resolved = subprocess.run(
        [
            "git",
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{BASELINE_REF}^{{commit}}",
        ],
        check=True,
        capture_output=True,
        cwd=REPOSITORY_DIR,
        text=True,
    )
    commit_hash = resolved.stdout.strip()
    completed = subprocess.run(
        [
            "git",
            "show",
            "--end-of-options",
            f"{commit_hash}:market_maker.py",
        ],
        check=True,
        capture_output=True,
        cwd=REPOSITORY_DIR,
        text=True,
    )
    module_name = f"market_maker_{commit_hash[:12]}"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    exec(  # noqa: S102 - the harness intentionally loads a local Git revision.
        compile(completed.stdout, f"{module_name}.py", "exec"), module.__dict__
    )
    return module


class NetRiskMarketMaker(market_maker.MarketMaker):
    def on_trade(
        self,
        option: market_maker.BinaryOption,
        price: float,
        quantity: int,
        counterparty_id: int,
    ) -> None:
        self.cash_balance -= price * quantity
        self.position.add_option_quantity(option.option_id, quantity)
        self.update_remaining_risk_budget()

    def release_expiry_collateral(
        self,
        new_underlying_state: list[market_maker.Underlying],
        new_option_state: list[market_maker.BinaryOption],
    ) -> None:
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
            self.cash_balance += self.get_position(option_id) * payoff
            self.long_quantity_by_option_id.pop(option_id, None)
            self.short_quantity_by_option_id.pop(option_id, None)
            self.position.option_quantity_by_option_id.pop(option_id, None)
        self.update_remaining_risk_budget()

    def quote(
        self, option: market_maker.BinaryOption, counterparty_id: int
    ) -> market_maker.Quote:
        reservation_price = self.reservation_price(option)
        bid_price = max(
            0.0,
            min(
                0.99,
                math.floor((reservation_price - self.HALF_SPREAD) * 100 + self.EPSILON)
                / 100,
            ),
        )
        offer_price = max(
            0.01,
            min(
                1.0,
                math.ceil((reservation_price + self.HALF_SPREAD) * 100 - self.EPSILON)
                / 100,
            ),
        )
        if bid_price >= offer_price:
            if offer_price < 1.0:
                offer_price = round(offer_price + 0.01, 2)
            else:
                bid_price = round(bid_price - 0.01, 2)

        position = self.get_position(option.option_id)
        bid_quantity = self.safe_net_quote_quantity(
            option.option_id,
            self.INVENTORY_LIMIT - position,
            bid_price,
            1,
        )
        offer_quantity = self.safe_net_quote_quantity(
            option.option_id,
            self.INVENTORY_LIMIT + position,
            offer_price,
            -1,
        )
        if bid_quantity == 0:
            bid_price = 0.0
            bid_quantity = 1
        if offer_quantity == 0:
            offer_price = 1.0
            offer_quantity = 1
        return market_maker.Quote(
            round(bid_price, 2),
            bid_quantity,
            round(offer_price, 2),
            offer_quantity,
        )

    def respond_to_fok(
        self,
        option: market_maker.BinaryOption,
        fok_order: market_maker.FokOrder,
    ) -> bool:
        if fok_order.option_id != option.option_id:
            return False
        reservation_price = self.reservation_price(option)
        if fok_order.order_type == market_maker.OrderType.BUY:
            signed_quantity = -fok_order.quantity
            has_edge = (
                fok_order.price + self.EPSILON >= reservation_price + self.FOK_EDGE
            )
        elif fok_order.order_type == market_maker.OrderType.SELL:
            signed_quantity = fok_order.quantity
            has_edge = (
                fok_order.price <= reservation_price - self.FOK_EDGE + self.EPSILON
            )
        else:
            return False
        resulting_position = self.get_position(option.option_id) + signed_quantity
        return (
            abs(resulting_position) <= self.INVENTORY_LIMIT
            and self.can_afford_trade(
                option.option_id, fok_order.price, signed_quantity
            )
            and has_edge
        )

    def update_remaining_risk_budget(self) -> None:
        short_liability = math.fsum(
            max(0, -quantity)
            for quantity in self.position.option_quantity_by_option_id.values()
        )
        self.remaining_risk_budget = max(0.0, self.cash_balance - short_liability)

    def projected_risk_budget(
        self, option_id: int, price: float, quantity: int
    ) -> float:
        current_position = self.get_position(option_id)
        resulting_position = current_position + quantity
        total_short_liability = math.fsum(
            max(0, -position)
            for position in self.position.option_quantity_by_option_id.values()
        )
        return (
            self.cash_balance
            - price * quantity
            - total_short_liability
            + max(0, -current_position)
            - max(0, -resulting_position)
        )

    def can_afford_trade(self, option_id: int, price: float, quantity: int) -> bool:
        return self.projected_risk_budget(option_id, price, quantity) >= -self.EPSILON

    def safe_net_quote_quantity(
        self,
        option_id: int,
        inventory_room: int,
        price: float,
        quantity_sign: int,
    ) -> int:
        maximum_quantity = max(0, min(self.BASE_QUANTITY, inventory_room))
        for quantity in range(maximum_quantity, 0, -1):
            if self.can_afford_trade(option_id, price, quantity_sign * quantity):
                return quantity
        return 0


def cached_market_maker(
    base_class: type[Any], price_cache: dict[tuple[Any, ...], float]
) -> type[Any]:
    class CachedMarketMaker(base_class):
        def price_option(self, option: Any) -> float:
            parameter_key = tuple(sorted(vars(self.estimated_parameters).items()))
            value_key = tuple(
                sorted(
                    (underlying.underlying_id, underlying.value)
                    for underlying in self.underlying_state
                )
            )
            option_key = (
                option.steps_until_expiry,
                option.strike,
                tuple((leg.underlying_id, leg.weight) for leg in option.legs),
            )
            key = (self.LIVE_PATHS, parameter_key, value_key, option_key)
            if key not in price_cache:
                price_cache[key] = super().price_option(option)
            return price_cache[key]

    return CachedMarketMaker


def make_parameters(
    module: types.ModuleType, parameter_values: dict[str, float]
) -> Any:
    return module.MarketParameters(**parameter_values)


def make_underlyings(module: types.ModuleType, values: dict[int, float]) -> list[Any]:
    return [
        module.Underlying("FED", 1, values[1]),
        module.Underlying("AJR", 2, values[2]),
        module.Underlying("THR", 3, values[3]),
    ]


def make_options(
    module: types.ModuleType,
    option_specs: tuple[OptionSpec, ...],
    steps_until_expiry: int,
) -> list[Any]:
    return [
        module.BinaryOption(
            tuple(
                module.OptionLeg(underlying_id, weight)
                for underlying_id, weight in option_spec.legs
            ),
            option_spec.option_id,
            steps_until_expiry,
            option_spec.strike,
        )
        for option_spec in option_specs
    ]


def make_scenario(regime: Regime, seed: int) -> Scenario:
    parameter_values = default_parameters()
    parameter_values.update(regime.parameter_overrides)
    parameters = make_parameters(market_maker, parameter_values)
    values = UNDERLYING_START_VALUES.copy()
    histories = {underlying_id: [value] for underlying_id, value in values.items()}
    state_before = random.getstate()
    random.seed(100_000 + 1_009 * seed + regime.warmup_days)
    try:
        for day_number in range(regime.warmup_days):
            values = parameters.advance_step(values)
            for underlying_id, value in values.items():
                histories[underlying_id].append(value)
        start_values = values.copy()
        future_values = []
        for day_number in range(OPTION_LIFETIME):
            values = parameters.advance_step(values)
            future_values.append(values)
    finally:
        random.setstate(state_before)

    option_specs = (
        OptionSpec(1_001, ((1, 1.0),), start_values[1] + 0.25),
        OptionSpec(1_002, ((2, 1.0),), start_values[2]),
        OptionSpec(1_003, ((3, 1.0),), start_values[3]),
        OptionSpec(
            1_004,
            ((2, 1.0), (3, -1.0)),
            start_values[2] - start_values[3],
        ),
    )
    return Scenario(
        history={
            underlying_id: tuple(history)
            for underlying_id, history in histories.items()
        },
        start_values=start_values,
        future_values=tuple(future_values),
        options=option_specs,
        parameters=parameter_values,
    )


def option_payoff(option_spec: OptionSpec, values: dict[int, float]) -> float:
    observable = math.fsum(
        weight * values[underlying_id] for underlying_id, weight in option_spec.legs
    )
    return 1.0 if observable >= option_spec.strike else 0.0


def reference_prices(
    scenario: Scenario,
    values: dict[int, float],
    steps_until_expiry: int,
    number_of_paths: int,
    price_cache: dict[tuple[Any, ...], float],
) -> dict[int, float]:
    parameters = make_parameters(market_maker, scenario.parameters)
    options = make_options(market_maker, scenario.options, steps_until_expiry)
    maker = market_maker.MarketMaker(
        make_underlyings(market_maker, values), options, STARTING_CASH
    )
    prices = {}
    for option in options:
        key = (
            tuple(sorted(scenario.parameters.items())),
            tuple(sorted(values.items())),
            option.option_id,
            steps_until_expiry,
            number_of_paths,
        )
        if key not in price_cache:
            price_cache[key] = maker.price_with_parameters(
                parameters, option, number_of_paths
            )
        prices[option.option_id] = price_cache[key]
    return prices


def exact_available_capital(cash: float, positions: dict[int, int]) -> float:
    return cash - math.fsum(max(0, -quantity) for quantity in positions.values())


def record_trade(
    cash: float,
    positions: dict[int, int],
    option_id: int,
    price: float,
    quantity: int,
) -> tuple[float, bool]:
    cash -= price * quantity
    positions[option_id] += quantity
    return cash, exact_available_capital(cash, positions) < -1e-9


def execute_quote_flow(
    profile_name: str,
    maker: Any,
    options: list[Any],
    fair_prices: dict[int, float],
    terminal_values: dict[int, float],
    option_specs_by_id: dict[int, OptionSpec],
    random_generator: random.Random,
    cash: float,
    positions: dict[int, int],
) -> tuple[float, bool, int]:
    bankrupt = False
    trade_count = 0
    for option in options:
        quote = maker.quote(option, counterparty_id=101)
        fair_price = fair_prices[option.option_id]
        if profile_name == "noise":
            valuation = fair_price + random_generator.gauss(0.0, 0.04)
        elif profile_name == "informed":
            valuation = fair_price
        else:
            payoff = option_payoff(
                option_specs_by_id[option.option_id], terminal_values
            )
            valuation = 0.25 * fair_price + 0.75 * payoff
        valuation = min(max(valuation, 0.0), 1.0)
        requested_quantity = random_generator.randint(1, 3)
        if valuation >= quote.offer_price:
            price = quote.offer_price
            quantity = -min(requested_quantity, quote.offer_quantity)
        elif valuation <= quote.bid_price:
            price = quote.bid_price
            quantity = min(requested_quantity, quote.bid_quantity)
        else:
            continue
        maker.on_trade(option, price, quantity, counterparty_id=101)
        cash, trade_bankruptcy = record_trade(
            cash, positions, option.option_id, price, quantity
        )
        bankrupt = bankrupt or trade_bankruptcy
        trade_count += 1
    return cash, bankrupt, trade_count


def execute_fok_flow(
    strategy: Strategy,
    maker: Any,
    options: list[Any],
    fair_prices: dict[int, float],
    random_generator: random.Random,
    cash: float,
    positions: dict[int, int],
) -> tuple[float, bool, int]:
    bankrupt = False
    trade_count = 0
    for option in options:
        counterparty_buys = random_generator.random() < 0.5
        valuation = fair_prices[option.option_id] + random_generator.gauss(0.0, 0.035)
        valuation = min(max(valuation, 0.0), 1.0)
        quantity = random_generator.randint(1, 3)
        if counterparty_buys:
            order_type = strategy.module.OrderType.BUY
            price = min(1.0, math.ceil(valuation * 100.0 - 1e-12) / 100.0)
            signed_quantity = -quantity
        else:
            order_type = strategy.module.OrderType.SELL
            price = max(0.0, math.floor(valuation * 100.0 + 1e-12) / 100.0)
            signed_quantity = quantity
        order = strategy.module.FokOrder(
            202, option.option_id, order_type, price, quantity
        )
        if not maker.respond_to_fok(option, order):
            continue
        maker.on_trade(option, price, signed_quantity, counterparty_id=202)
        cash, trade_bankruptcy = record_trade(
            cash,
            positions,
            option.option_id,
            price,
            signed_quantity,
        )
        bankrupt = bankrupt or trade_bankruptcy
        trade_count += 1
    return cash, bankrupt, trade_count


def run_session(
    strategy: Strategy,
    scenario: Scenario,
    profile_name: str,
    seed: int,
    reference_path_count: int,
    strategy_price_cache: dict[tuple[Any, ...], float],
    reference_price_cache: dict[tuple[Any, ...], float],
) -> SessionResult:
    module = strategy.module
    options = make_options(module, scenario.options, OPTION_LIFETIME)
    maker_class = cached_market_maker(strategy.market_maker_class, strategy_price_cache)
    maker = maker_class(
        make_underlyings(module, scenario.start_values),
        options,
        STARTING_CASH,
    )
    maker.warm_up(module.MarketHistory(scenario.history))
    cash = STARTING_CASH
    positions: dict[int, int] = defaultdict(int)
    bankrupt = False
    trade_count = 0
    current_values = scenario.start_values
    option_specs_by_id = {
        option_spec.option_id: option_spec for option_spec in scenario.options
    }
    profile_number = PROFILE_NAMES.index(profile_name) + 1
    random_generator = random.Random(900_000 + 7_919 * seed + profile_number)

    for day_number, next_values in enumerate(scenario.future_values):
        steps_until_expiry = OPTION_LIFETIME - day_number
        fair_prices = reference_prices(
            scenario,
            current_values,
            steps_until_expiry,
            reference_path_count,
            reference_price_cache,
        )
        if profile_name == "fok":
            cash, new_bankruptcy, new_trades = execute_fok_flow(
                strategy,
                maker,
                options,
                fair_prices,
                random_generator,
                cash,
                positions,
            )
        else:
            cash, new_bankruptcy, new_trades = execute_quote_flow(
                profile_name,
                maker,
                options,
                fair_prices,
                scenario.future_values[-1],
                option_specs_by_id,
                random_generator,
                cash,
                positions,
            )
        bankrupt = bankrupt or new_bankruptcy
        trade_count += new_trades

        next_options = [option.advance_step() for option in options]
        maker.on_step_advance(make_underlyings(module, next_values), next_options)
        for option in options:
            if option.steps_until_expiry != 1:
                continue
            payoff = option_payoff(option_specs_by_id[option.option_id], next_values)
            cash += positions.pop(option.option_id, 0) * payoff
        bankrupt = bankrupt or exact_available_capital(cash, positions) < -1e-9
        current_values = next_values
        options = next_options

    return SessionResult(cash - STARTING_CASH, bankrupt, trade_count)


def summarize(results: list[SessionResult]) -> tuple[float, float, int, int]:
    return (
        math.fsum(result.pnl for result in results) / len(results),
        min(result.pnl for result in results),
        sum(result.bankrupt for result in results),
        sum(result.trade_count for result in results),
    )


def candidate_is_accepted(
    gross_results: list[SessionResult], candidate_results: list[SessionResult]
) -> tuple[bool, float]:
    if len(gross_results) != len(candidate_results) or not gross_results:
        raise ValueError("paired strategy results must be non-empty and equally sized")
    gross_summary = summarize(gross_results)
    candidate_summary = summarize(candidate_results)
    worst_paired_delta = min(
        candidate.pnl - gross.pnl
        for gross, candidate in zip(gross_results, candidate_results)
    )
    accepted = (
        candidate_summary[2] == 0
        and candidate_summary[0] > gross_summary[0]
        and worst_paired_delta >= -(0.01 * STARTING_CASH)
    )
    return accepted, worst_paired_delta


def run_benchmark(seed_count: int, reference_path_count: int) -> bool:
    baseline_module = load_baseline_module()
    strategies = (
        Strategy(
            f"commit-{BASELINE_REF}", baseline_module, baseline_module.MarketMaker
        ),
        Strategy("gross-current", market_maker, market_maker.MarketMaker),
        Strategy("net-experiment", market_maker, NetRiskMarketMaker),
    )
    strategy_price_caches: dict[str, dict[tuple[Any, ...], float]] = {
        strategy.name: {} for strategy in strategies
    }
    reference_price_cache: dict[tuple[Any, ...], float] = {}
    results_by_strategy: dict[str, list[SessionResult]] = {
        strategy.name: [] for strategy in strategies
    }
    start_time = time.perf_counter()

    for seed in range(seed_count):
        for regime in regimes():
            scenario = make_scenario(regime, seed)
            for profile_name in PROFILE_NAMES:
                for strategy in strategies:
                    result = run_session(
                        strategy,
                        scenario,
                        profile_name,
                        seed,
                        reference_path_count,
                        strategy_price_caches[strategy.name],
                        reference_price_cache,
                    )
                    results_by_strategy[strategy.name].append(result)

    print("strategy          mean pnl   worst pnl   bankruptcies   trades")
    summaries = {}
    for strategy in strategies:
        summary = summarize(results_by_strategy[strategy.name])
        summaries[strategy.name] = summary
        mean_pnl, worst_pnl, bankruptcies, trade_count = summary
        print(
            f"{strategy.name:<17} {mean_pnl:>8.4f}   {worst_pnl:>9.4f}"
            f"   {bankruptcies:>12}   {trade_count:>6}"
        )

    accepted, worst_paired_delta = candidate_is_accepted(
        results_by_strategy["gross-current"],
        results_by_strategy["net-experiment"],
    )
    elapsed = time.perf_counter() - start_time
    print(
        f"sessions per strategy: {len(results_by_strategy['net-experiment'])}; "
        f"elapsed: {elapsed:.2f}s"
    )
    print(f"worst paired PnL delta: {worst_paired_delta:.4f}")
    print(
        "acceptance: "
        + ("PASS" if accepted else "FAIL")
        + " (zero bankruptcies, higher mean PnL, each paired loss within 1% capital)"
    )
    return accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seeded comparison of binary-option market-maker policies."
    )
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--reference-paths", type=int, default=8_000)
    args = parser.parse_args()
    if args.seeds <= 0 or args.reference_paths <= 0:
        parser.error("--seeds and --reference-paths must be positive")
    return args


def main() -> int:
    args = parse_args()
    accepted = run_benchmark(args.seeds, args.reference_paths)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
