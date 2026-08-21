import math
import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from unittest.mock import patch

from market_maker import (
    AJARAI_UNDERLYING_ID,
    FED_FUNDS_RATE_UNDERLYING_ID,
    THERIODIC_UNDERLYING_ID,
    BinaryOption,
    FokOrder,
    MarketHistory,
    MarketMaker,
    MarketParameters,
    OptionLeg,
    OrderType,
    Quote,
    Underlying,
)


def make_parameters(**overrides: float) -> MarketParameters:
    values = {
        "ajarai_drift": 0.0,
        "ajarai_idio_std_dev": 0.08,
        "ajarai_rate_beta": 0.0,
        "ajarai_sector_beta": 0.03,
        "rate_down_probability": 0.30,
        "rate_reversion_strength": 0.0,
        "rate_up_probability": 0.20,
        "sector_std_dev": 1.0,
        "theriodic_drift": 0.0,
        "theriodic_idio_std_dev": 0.08,
        "theriodic_rate_beta": 0.0,
        "theriodic_sector_beta": 0.03,
        "rate_step": 0.25,
        "rate_target": 2.0,
    }
    values.update(overrides)
    return MarketParameters(**values)


def make_option(
    option_id: int = 101,
    *,
    underlying_id: int = AJARAI_UNDERLYING_ID,
    strike: float = 100.0,
    steps: int = 1,
    weight: float = 1.0,
) -> BinaryOption:
    return BinaryOption(
        legs=(OptionLeg(underlying_id=underlying_id, weight=weight),),
        option_id=option_id,
        steps_until_expiry=steps,
        strike=strike,
    )


def make_market_maker(
    *, cash: float = 100.0, options: list[BinaryOption] | None = None
) -> MarketMaker:
    if options is None:
        options = [make_option()]
    return MarketMaker(
        underlying_initial_state=[
            Underlying("FED", FED_FUNDS_RATE_UNDERLYING_ID, 2.0),
            Underlying("AJR", AJARAI_UNDERLYING_ID, 100.0),
            Underlying("THR", THERIODIC_UNDERLYING_ID, 100.0),
        ],
        option_initial_state=options,
        cash_balance=cash,
    )


def estimated_parameters(market_maker: MarketMaker) -> MarketParameters:
    """Find the private parameter snapshot without prescribing its attribute name."""
    candidates = [
        value
        for value in vars(market_maker).values()
        if isinstance(value, MarketParameters)
    ]
    if not candidates:
        raise AssertionError(
            "MarketMaker should retain its current MarketParameters estimate"
        )
    return candidates[-1]


def assert_valid_parameters(
    test_case: unittest.TestCase, parameters: MarketParameters
) -> None:
    for field in fields(MarketParameters):
        test_case.assertTrue(
            math.isfinite(getattr(parameters, field.name)),
            f"{field.name} must be finite",
        )
    test_case.assertGreater(parameters.rate_up_probability, 0.0)
    test_case.assertGreater(parameters.rate_down_probability, 0.0)
    test_case.assertLessEqual(
        parameters.rate_up_probability + parameters.rate_down_probability, 1.0
    )
    test_case.assertGreater(parameters.rate_step, 0.0)
    test_case.assertGreaterEqual(parameters.rate_target, 0.0)
    test_case.assertGreaterEqual(parameters.rate_reversion_strength, 0.0)
    test_case.assertLessEqual(parameters.rate_reversion_strength, 1.0)
    test_case.assertGreaterEqual(parameters.ajarai_idio_std_dev, 0.0)
    test_case.assertGreaterEqual(parameters.theriodic_idio_std_dev, 0.0)
    test_case.assertGreaterEqual(parameters.sector_std_dev, 0.0)


def history_from_returns(
    ajarai_returns: list[float], theriodic_returns: list[float]
) -> MarketHistory:
    ajarai_values = [100.0]
    theriodic_values = [100.0]
    for ajarai_return, theriodic_return in zip(ajarai_returns, theriodic_returns):
        ajarai_values.append(round(ajarai_values[-1] * math.exp(ajarai_return), 2))
        theriodic_values.append(
            round(theriodic_values[-1] * math.exp(theriodic_return), 2)
        )
    return MarketHistory(
        {
            FED_FUNDS_RATE_UNDERLYING_ID: (2.0,) * len(ajarai_values),
            AJARAI_UNDERLYING_ID: tuple(ajarai_values),
            THERIODIC_UNDERLYING_ID: tuple(theriodic_values),
        }
    )


def reference_prices(
    parameters: MarketParameters,
    options: list[BinaryOption],
    *,
    number_of_paths: int = 80_000,
) -> list[float]:
    """Independent seeded oracle using the supplied market process itself."""
    if not options:
        raise ValueError("reference pricing requires at least one option")
    if len({option.steps_until_expiry for option in options}) != 1:
        raise ValueError("reference-priced options must have the same expiry")

    initial_values = {
        FED_FUNDS_RATE_UNDERLYING_ID: 2.0,
        AJARAI_UNDERLYING_ID: 100.0,
        THERIODIC_UNDERLYING_ID: 100.0,
    }
    hit_counts = [0] * len(options)
    state_before = random.getstate()
    random.seed(19_847)
    try:
        for _ in range(number_of_paths):
            values = initial_values.copy()
            for _ in range(options[0].steps_until_expiry):
                values = parameters.advance_step(values)
            for index, option in enumerate(options):
                hit_counts[index] += int(option.expiry_valuation(values))
    finally:
        random.setstate(state_before)
    return [hit_count / number_of_paths for hit_count in hit_counts]


class MarketMakerPricingTests(unittest.TestCase):
    def test_reference_prices_rejects_an_empty_option_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one option"):
            reference_prices(make_parameters(), [])

    def test_reference_prices_rejects_mixed_expiries(self) -> None:
        options = [make_option(option_id=1, steps=1), make_option(option_id=2, steps=2)]

        with self.assertRaisesRegex(ValueError, "same expiry"):
            reference_prices(make_parameters(), options)

    def test_name_is_fixed_and_nonempty(self) -> None:
        self.assertEqual(make_market_maker().name, "SimpleSafeMM")

    def test_expired_option_returns_exact_payoff(self) -> None:
        option_in = make_option(strike=100.0, steps=0)
        option_out = make_option(option_id=102, strike=100.01, steps=0)
        market_maker = make_market_maker(options=[option_in, option_out])
        parameters = make_parameters()

        self.assertEqual(market_maker.price_option(option_in), 1.0)
        self.assertEqual(market_maker.price_option(option_out), 0.0)
        self.assertEqual(
            market_maker.price_option_from_parameters(parameters, option_in), 1.0
        )
        self.assertEqual(
            market_maker.price_option_from_parameters(parameters, option_out), 0.0
        )

    def test_one_step_fed_option_uses_exact_transition_probability(self) -> None:
        option = make_option(
            underlying_id=FED_FUNDS_RATE_UNDERLYING_ID,
            strike=2.25,
            steps=1,
        )
        market_maker = make_market_maker(options=[option])

        price = market_maker.price_option_from_parameters(make_parameters(), option)

        self.assertAlmostEqual(price, 0.20, places=12)

    def test_company_pricing_is_deterministic_id_independent_and_rng_safe(self) -> None:
        option = make_option(strike=100.0, steps=3)
        equivalent_option = replace(option, option_id=999)
        market_maker = make_market_maker(options=[option])
        parameters = make_parameters()
        random.seed(87341)
        state_before = random.getstate()

        first = market_maker.price_option_from_parameters(parameters, option)
        second = market_maker.price_option_from_parameters(parameters, option)
        equivalent = market_maker.price_option_from_parameters(
            parameters, equivalent_option
        )

        self.assertEqual(first, second)
        self.assertEqual(first, equivalent)
        self.assertEqual(random.getstate(), state_before)
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 1.0)

    def test_symmetric_one_step_company_probability_is_close_to_half(self) -> None:
        option = make_option(strike=100.0, steps=1)
        parameters = make_parameters(
            ajarai_idio_std_dev=0.10,
            ajarai_sector_beta=0.0,
        )

        price = make_market_maker(options=[option]).price_option_from_parameters(
            parameters, option
        )

        self.assertAlmostEqual(price, 0.5, delta=0.025)

    def test_identical_companies_with_only_shared_shocks_preserve_spread(self) -> None:
        option = BinaryOption(
            legs=(
                OptionLeg(AJARAI_UNDERLYING_ID, 1.0),
                OptionLeg(THERIODIC_UNDERLYING_ID, -1.0),
            ),
            option_id=202,
            steps_until_expiry=3,
            strike=0.0,
        )
        parameters = make_parameters(
            ajarai_idio_std_dev=0.0,
            theriodic_idio_std_dev=0.0,
            ajarai_sector_beta=0.05,
            theriodic_sector_beta=0.05,
        )

        price = make_market_maker(options=[option]).price_option_from_parameters(
            parameters, option
        )

        self.assertEqual(price, 1.0)

    def test_live_pricing_is_repeatable_under_concurrent_calls(self) -> None:
        option = make_option(strike=103.0, steps=2)
        market_maker = make_market_maker(options=[option])

        with ThreadPoolExecutor(max_workers=4) as executor:
            prices = list(
                executor.map(lambda _: market_maker.price_option(option), range(8))
            )

        self.assertTrue(all(price == prices[0] for price in prices))
        self.assertGreaterEqual(prices[0], 0.0)
        self.assertLessEqual(prices[0], 1.0)

    def test_company_prices_match_independent_supplied_process_oracle(self) -> None:
        options = [
            make_option(option_id=301, strike=101.0, steps=3),
            make_option(
                option_id=302,
                underlying_id=THERIODIC_UNDERLYING_ID,
                strike=99.0,
                steps=3,
            ),
            BinaryOption(
                legs=(
                    OptionLeg(AJARAI_UNDERLYING_ID, 1.0),
                    OptionLeg(THERIODIC_UNDERLYING_ID, -1.0),
                ),
                option_id=303,
                steps_until_expiry=3,
                strike=2.0,
            ),
        ]
        parameters = make_parameters(
            ajarai_drift=0.002,
            ajarai_idio_std_dev=0.06,
            ajarai_rate_beta=-0.04,
            ajarai_sector_beta=0.04,
            rate_down_probability=0.20,
            rate_reversion_strength=0.03,
            rate_up_probability=0.25,
            sector_std_dev=0.8,
            theriodic_drift=-0.001,
            theriodic_idio_std_dev=0.05,
            theriodic_rate_beta=0.03,
            theriodic_sector_beta=0.02,
        )
        expected_prices = reference_prices(parameters, options)
        market_maker = make_market_maker(options=options)

        actual_prices = [
            market_maker.price_option_from_parameters(parameters, option)
            for option in options
        ]

        for actual, expected in zip(actual_prices, expected_prices):
            self.assertAlmostEqual(actual, expected, delta=0.02)

    def test_extreme_finite_drift_does_not_overflow_pricing(self) -> None:
        option = make_option(strike=100.0, steps=1)
        market_maker = make_market_maker(options=[option])

        price = market_maker.price_option_from_parameters(
            make_parameters(
                ajarai_drift=1_000.0,
                ajarai_idio_std_dev=0.0,
                ajarai_sector_beta=0.0,
            ),
            option,
        )

        self.assertEqual(price, 1.0)

    def test_zero_company_value_remains_zero_under_extreme_positive_drift(self) -> None:
        option = make_option(strike=0.01, steps=1)
        market_maker = MarketMaker(
            underlying_initial_state=[
                Underlying("FED", FED_FUNDS_RATE_UNDERLYING_ID, 2.0),
                Underlying("AJR", AJARAI_UNDERLYING_ID, 0.0),
                Underlying("THR", THERIODIC_UNDERLYING_ID, 100.0),
            ],
            option_initial_state=[option],
            cash_balance=100.0,
        )

        price = market_maker.price_option_from_parameters(
            make_parameters(
                ajarai_drift=1_000.0,
                ajarai_idio_std_dev=0.0,
                ajarai_sector_beta=0.0,
            ),
            option,
        )

        self.assertEqual(price, 0.0)


class MarketProcessTests(unittest.TestCase):
    def test_supplied_market_process_advances_all_underlyings_together(self) -> None:
        parameters = make_parameters(
            ajarai_idio_std_dev=0.0,
            ajarai_sector_beta=0.0,
            sector_std_dev=0.0,
            theriodic_idio_std_dev=0.0,
            theriodic_sector_beta=0.0,
        )
        values = {
            FED_FUNDS_RATE_UNDERLYING_ID: 2.0,
            AJARAI_UNDERLYING_ID: 100.0,
            THERIODIC_UNDERLYING_ID: 100.0,
        }

        with (
            patch("market_maker.random.random", return_value=0.1),
            patch("market_maker.random.gauss", return_value=0.0),
        ):
            next_values = parameters.advance_step(values)

        self.assertEqual(next_values[FED_FUNDS_RATE_UNDERLYING_ID], 2.25)
        self.assertEqual(next_values[AJARAI_UNDERLYING_ID], 100.0)
        self.assertEqual(next_values[THERIODIC_UNDERLYING_ID], 100.0)


class MarketMakerNamingTests(unittest.TestCase):
    def test_custom_class_and_instance_names_do_not_use_single_underscores(
        self,
    ) -> None:
        def is_single_underscore_name(name: str) -> bool:
            return name.startswith("_") and not name.startswith("__")

        class_names = sorted(
            name for name in vars(MarketMaker) if is_single_underscore_name(name)
        )
        instance_names = sorted(
            name
            for name in vars(make_market_maker())
            if is_single_underscore_name(name)
        )

        self.assertEqual(class_names, [])
        self.assertEqual(instance_names, [])


class MarketMakerWarmUpTests(unittest.TestCase):
    def test_extreme_finite_rate_history_retains_fallback_parameters(self) -> None:
        market_maker = make_market_maker()
        fallback_parameters = estimated_parameters(market_maker)
        rates = (0.0, 1e308, 0.0, 1e308, 0.0, 1e308)
        history = MarketHistory(
            {
                FED_FUNDS_RATE_UNDERLYING_ID: rates,
                AJARAI_UNDERLYING_ID: (100.0,) * len(rates),
                THERIODIC_UNDERLYING_ID: (100.0,) * len(rates),
            }
        )

        market_maker.warm_up(history)

        parameters = estimated_parameters(market_maker)
        self.assertEqual(parameters, fallback_parameters)
        assert_valid_parameters(self, parameters)

    def test_short_or_incomplete_history_keeps_valid_fallbacks(self) -> None:
        histories = [
            MarketHistory({}),
            MarketHistory({FED_FUNDS_RATE_UNDERLYING_ID: (2.0, 2.25, 2.0)}),
            MarketHistory(
                {
                    FED_FUNDS_RATE_UNDERLYING_ID: (2.0,) * 6,
                    AJARAI_UNDERLYING_ID: (100.0, 0.0, 100.0, 100.0, 100.0, 100.0),
                    THERIODIC_UNDERLYING_ID: (100.0,) * 6,
                }
            ),
        ]

        for history in histories:
            with self.subTest(history=history.values_by_underlying_id):
                market_maker = make_market_maker()
                market_maker.warm_up(history)
                parameters = estimated_parameters(market_maker)
                assert_valid_parameters(self, parameters)
                self.assertGreaterEqual(market_maker.price_option(make_option()), 0.0)
                self.assertLessEqual(market_maker.price_option(make_option()), 1.0)

    def test_company_drift_is_estimated_from_static_rate_history(self) -> None:
        market_maker = make_market_maker()
        history = history_from_returns([0.01] * 8, [-0.005] * 8)

        market_maker.warm_up(history)
        parameters = estimated_parameters(market_maker)

        self.assertAlmostEqual(parameters.ajarai_drift, 0.01, delta=0.001)
        self.assertAlmostEqual(parameters.theriodic_drift, -0.005, delta=0.001)
        assert_valid_parameters(self, parameters)

    def test_residual_factor_loadings_reproduce_covariance_sign(self) -> None:
        signals = [0.02, -0.01, 0.015, -0.02, 0.01, -0.015, 0.025, -0.005]
        for multiplier, expected_sign in ((1.0, 1), (-1.0, -1)):
            with self.subTest(multiplier=multiplier):
                history = history_from_returns(
                    signals, [multiplier * signal for signal in signals]
                )
                market_maker = make_market_maker()
                market_maker.warm_up(history)
                parameters = estimated_parameters(market_maker)
                loading_product = (
                    parameters.ajarai_sector_beta * parameters.theriodic_sector_beta
                )

                self.assertEqual(1 if loading_product > 0 else -1, expected_sign)
                assert_valid_parameters(self, parameters)

    def test_rate_estimates_remain_valid_for_varied_and_static_series(self) -> None:
        rate_series = (
            2.0,
            2.25,
            2.0,
            1.75,
            2.0,
            2.0,
            2.25,
            2.0,
            1.75,
        )
        for rates in (rate_series, (2.0,) * len(rate_series)):
            with self.subTest(rates=rates):
                history = MarketHistory(
                    {
                        FED_FUNDS_RATE_UNDERLYING_ID: rates,
                        AJARAI_UNDERLYING_ID: (100.0,) * len(rates),
                        THERIODIC_UNDERLYING_ID: (100.0,) * len(rates),
                    }
                )
                market_maker = make_market_maker()

                market_maker.warm_up(history)
                parameters = estimated_parameters(market_maker)

                self.assertEqual(parameters.rate_step, 0.25)
                self.assertEqual(parameters.rate_target, 2.0)
                assert_valid_parameters(self, parameters)

    def test_static_rate_history_estimates_near_zero_move_probabilities(self) -> None:
        rates = (2.0,) * 12
        history = MarketHistory(
            {
                FED_FUNDS_RATE_UNDERLYING_ID: rates,
                AJARAI_UNDERLYING_ID: (100.0,) * len(rates),
                THERIODIC_UNDERLYING_ID: (100.0,) * len(rates),
            }
        )
        option = make_option(
            underlying_id=FED_FUNDS_RATE_UNDERLYING_ID,
            strike=2.25,
            steps=1,
        )
        market_maker = make_market_maker(options=[option])

        market_maker.warm_up(history)
        parameters = estimated_parameters(market_maker)

        self.assertAlmostEqual(parameters.rate_up_probability, 0.001)
        self.assertAlmostEqual(parameters.rate_down_probability, 0.001)
        self.assertAlmostEqual(market_maker.price_option(option), 0.001)

    def test_rate_estimation_skips_isolated_nonfinite_observations(self) -> None:
        rates = (2.0, 2.25, 2.5, 2.75, 3.0, math.nan, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25)
        history = MarketHistory(
            {
                FED_FUNDS_RATE_UNDERLYING_ID: rates,
                AJARAI_UNDERLYING_ID: (100.0,) * len(rates),
                THERIODIC_UNDERLYING_ID: (100.0,) * len(rates),
            }
        )
        market_maker = make_market_maker()

        market_maker.warm_up(history)

        self.assertGreater(estimated_parameters(market_maker).rate_up_probability, 0.5)

    def test_company_rate_beta_is_recovered_from_synthetic_history(self) -> None:
        rates = [2.0]
        rate_moves = [0.25, -0.25, -0.25, 0.25] * 10
        for rate_move in rate_moves:
            rates.append(round(rates[-1] + rate_move, 2))
        ajarai_returns = [0.001 + 0.04 * move for move in rate_moves]
        theriodic_returns = [-0.002 - 0.03 * move for move in rate_moves]
        history = history_from_returns(ajarai_returns, theriodic_returns)
        history = MarketHistory(
            {
                **history.values_by_underlying_id,
                FED_FUNDS_RATE_UNDERLYING_ID: tuple(rates),
            }
        )
        market_maker = make_market_maker()

        market_maker.warm_up(history)
        parameters = estimated_parameters(market_maker)

        self.assertAlmostEqual(parameters.ajarai_drift, 0.001, delta=0.001)
        self.assertAlmostEqual(parameters.ajarai_rate_beta, 0.04, delta=0.005)
        self.assertAlmostEqual(parameters.theriodic_drift, -0.002, delta=0.001)
        self.assertAlmostEqual(parameters.theriodic_rate_beta, -0.03, delta=0.005)

    def test_large_history_is_processed_without_failure(self) -> None:
        num_days = 10_001
        history = MarketHistory(
            {
                FED_FUNDS_RATE_UNDERLYING_ID: (2.0,) * num_days,
                AJARAI_UNDERLYING_ID: (100.0,) * num_days,
                THERIODIC_UNDERLYING_ID: (100.0,) * num_days,
            }
        )
        market_maker = make_market_maker()

        market_maker.warm_up(history)

        assert_valid_parameters(self, estimated_parameters(market_maker))

    def test_extreme_positive_company_history_does_not_overflow_log_returns(
        self,
    ) -> None:
        values = (1e308, 1e-308, 1e308, 1e-308, 1e308, 1e-308)
        history = MarketHistory(
            {
                FED_FUNDS_RATE_UNDERLYING_ID: (2.0,) * len(values),
                AJARAI_UNDERLYING_ID: values,
                THERIODIC_UNDERLYING_ID: (100.0,) * len(values),
            }
        )
        market_maker = make_market_maker()

        market_maker.warm_up(history)

        assert_valid_parameters(self, estimated_parameters(market_maker))


class MarketMakerQuoteTests(unittest.TestCase):
    def test_position_storage_does_not_collide_with_quote_lookup(self) -> None:
        option = make_option()
        market_maker = make_market_maker(options=[option])
        market_maker.position.add_option_quantity(option.option_id, 1)

        with patch.object(market_maker, "price_option", return_value=0.50):
            quote = market_maker.quote(option, counterparty_id=77)

        self.assertIsInstance(quote, Quote)

    def test_quote_is_two_cents_around_fair_value(self) -> None:
        option = make_option()
        market_maker = make_market_maker(options=[option])

        with patch.object(market_maker, "price_option", return_value=0.50):
            quote = market_maker.quote(option, counterparty_id=77)

        self.assertEqual(quote, Quote(0.48, 2, 0.52, 2))

    def test_quotes_remain_valid_at_probability_boundaries(self) -> None:
        option = make_option()
        for fair_value in (0.0, 0.001, 0.999, 1.0):
            with self.subTest(fair_value=fair_value):
                market_maker = make_market_maker(options=[option])
                with patch.object(
                    market_maker, "price_option", return_value=fair_value
                ):
                    quote = market_maker.quote(option, counterparty_id=77)

                self.assertGreaterEqual(quote.bid_price, 0.0)
                self.assertLess(quote.bid_price, quote.offer_price)
                self.assertLessEqual(quote.offer_price, 1.0)
                self.assertEqual(round(quote.bid_price * 100), quote.bid_price * 100)
                self.assertEqual(
                    round(quote.offer_price * 100), quote.offer_price * 100
                )

    def test_inventory_skew_moves_both_prices_away_from_existing_position(self) -> None:
        option = make_option()
        flat = make_market_maker(options=[option])
        long = make_market_maker(options=[option])
        short = make_market_maker(options=[option])
        long.position.add_option_quantity(option.option_id, 4)
        short.position.add_option_quantity(option.option_id, -4)

        with patch.object(MarketMaker, "price_option", return_value=0.50):
            flat_quote = flat.quote(option, 1)
            long_quote = long.quote(option, 1)
            short_quote = short.quote(option, 1)

        self.assertLess(long_quote.bid_price, flat_quote.bid_price)
        self.assertLess(long_quote.offer_price, flat_quote.offer_price)
        self.assertGreater(short_quote.bid_price, flat_quote.bid_price)
        self.assertGreater(short_quote.offer_price, flat_quote.offer_price)

    def test_inventory_limits_disable_only_the_risk_increasing_side(self) -> None:
        option = make_option()
        long = make_market_maker(options=[option])
        short = make_market_maker(options=[option])
        long.position.add_option_quantity(option.option_id, 10)
        short.position.add_option_quantity(option.option_id, -10)

        with patch.object(MarketMaker, "price_option", return_value=0.50):
            long_quote = long.quote(option, 1)
            short_quote = short.quote(option, 1)

        self.assertEqual((long_quote.bid_price, long_quote.bid_quantity), (0.0, 1))
        self.assertEqual(long_quote.offer_quantity, 2)
        self.assertEqual(
            (short_quote.offer_price, short_quote.offer_quantity), (1.0, 1)
        )
        self.assertEqual(short_quote.bid_quantity, 2)

    def test_quantities_are_bounded_by_full_fill_risk_budget(self) -> None:
        option = make_option()
        market_maker = make_market_maker(cash=0.75, options=[option])

        with patch.object(market_maker, "price_option", return_value=0.50):
            quote = market_maker.quote(option, 1)

        self.assertEqual(quote.bid_quantity, 1)
        self.assertEqual(quote.offer_quantity, 1)

    def test_exhausted_budget_produces_inert_two_sided_quote(self) -> None:
        option = make_option()
        market_maker = make_market_maker(cash=10.0, options=[option])
        market_maker.remaining_risk_budget = 0.0

        with patch.object(market_maker, "price_option", return_value=0.50):
            quote = market_maker.quote(option, 1)

        self.assertEqual(quote, Quote(0.0, 1, 1.0, 1))


class MarketMakerFokAndAccountingTests(unittest.TestCase):
    def test_fok_side_interpretation_and_edge_thresholds(self) -> None:
        option = make_option()
        market_maker = make_market_maker(options=[option])

        with patch.object(market_maker, "price_option", return_value=0.50):
            self.assertTrue(
                market_maker.respond_to_fok(
                    option, FokOrder(1, option.option_id, OrderType.BUY, 0.52, 1)
                )
            )
            self.assertTrue(
                market_maker.respond_to_fok(
                    option, FokOrder(1, option.option_id, OrderType.SELL, 0.48, 1)
                )
            )
            self.assertFalse(
                market_maker.respond_to_fok(
                    option, FokOrder(1, option.option_id, OrderType.BUY, 0.51, 1)
                )
            )
            self.assertFalse(
                market_maker.respond_to_fok(
                    option, FokOrder(1, option.option_id, OrderType.SELL, 0.49, 1)
                )
            )

    def test_fok_rejects_wrong_contract_inventory_breach_and_full_fill_risk(
        self,
    ) -> None:
        option = make_option()

        wrong_id_maker = make_market_maker(options=[option])
        with patch.object(wrong_id_maker, "price_option", return_value=0.50):
            self.assertFalse(
                wrong_id_maker.respond_to_fok(
                    option, FokOrder(1, 999, OrderType.BUY, 0.80, 1)
                )
            )

        inventory_maker = make_market_maker(options=[option])
        inventory_maker.position.add_option_quantity(option.option_id, 8)
        with patch.object(inventory_maker, "price_option", return_value=0.50):
            self.assertFalse(
                inventory_maker.respond_to_fok(
                    option, FokOrder(1, option.option_id, OrderType.SELL, 0.48, 3)
                )
            )

        risk_maker = make_market_maker(cash=1.0, options=[option])
        with patch.object(risk_maker, "price_option", return_value=0.50):
            self.assertFalse(
                risk_maker.respond_to_fok(
                    option, FokOrder(1, option.option_id, OrderType.BUY, 0.52, 3)
                )
            )
            self.assertFalse(
                risk_maker.respond_to_fok(
                    option, FokOrder(1, option.option_id, OrderType.SELL, 0.48, 3)
                )
            )

    def test_responding_to_fok_never_reserves_budget_or_changes_position(self) -> None:
        option = make_option()
        market_maker = make_market_maker(cash=10.0, options=[option])
        budget_before = market_maker.remaining_risk_budget
        position_before = market_maker.position.option_quantity_by_option_id[
            option.option_id
        ]

        with patch.object(market_maker, "price_option", return_value=0.50):
            accepted = market_maker.respond_to_fok(
                option, FokOrder(1, option.option_id, OrderType.BUY, 0.80, 2)
            )

        self.assertTrue(accepted)
        self.assertEqual(market_maker.remaining_risk_budget, budget_before)
        self.assertEqual(
            market_maker.position.option_quantity_by_option_id[option.option_id],
            position_before,
        )

    def test_trade_callback_updates_position_and_conservative_risk_budget(self) -> None:
        option = make_option()
        market_maker = make_market_maker(cash=10.0, options=[option])

        market_maker.on_trade(option, price=0.30, quantity=2, counterparty_id=1)
        self.assertEqual(
            market_maker.position.option_quantity_by_option_id[option.option_id], 2
        )
        self.assertAlmostEqual(market_maker.remaining_risk_budget, 9.40)

        market_maker.on_trade(option, price=0.80, quantity=-3, counterparty_id=2)
        self.assertEqual(
            market_maker.position.option_quantity_by_option_id[option.option_id], -1
        )
        self.assertAlmostEqual(market_maker.remaining_risk_budget, 8.80)


class MarketMakerIntegrationTests(unittest.TestCase):
    def test_seeded_miniature_session_stays_valid_and_solvent(self) -> None:
        option = make_option(
            underlying_id=FED_FUNDS_RATE_UNDERLYING_ID,
            strike=3.0,
            steps=1,
        )
        market_maker = make_market_maker(cash=5.0, options=[option])
        market_maker.warm_up(
            MarketHistory(
                {
                    FED_FUNDS_RATE_UNDERLYING_ID: (2.0,) * 8,
                    AJARAI_UNDERLYING_ID: (100.0,) * 8,
                    THERIODIC_UNDERLYING_ID: (100.0,) * 8,
                }
            )
        )

        initial_quote = market_maker.quote(option, counterparty_id=11)
        self.assertIsInstance(initial_quote, Quote)
        accepted = market_maker.respond_to_fok(
            option,
            FokOrder(11, option.option_id, OrderType.BUY, 1.0, 1),
        )
        self.assertTrue(accepted)
        market_maker.on_trade(option, price=1.0, quantity=-1, counterparty_id=11)

        expired_option = option.advance_step()
        market_maker.on_step_advance(
            [
                Underlying("FED", FED_FUNDS_RATE_UNDERLYING_ID, 2.25),
                Underlying("AJR", AJARAI_UNDERLYING_ID, 100.0),
                Underlying("THR", THERIODIC_UNDERLYING_ID, 100.0),
            ],
            [expired_option],
        )

        self.assertEqual(market_maker.price_option(expired_option), 0.0)
        self.assertGreaterEqual(market_maker.remaining_risk_budget, 0.0)
        final_quote = market_maker.quote(expired_option, counterparty_id=11)
        self.assertLess(final_quote.bid_price, final_quote.offer_price)


if __name__ == "__main__":
    unittest.main()
