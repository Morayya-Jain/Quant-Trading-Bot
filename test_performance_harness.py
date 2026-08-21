import unittest
from unittest.mock import patch

from market_maker import (
    AJARAI_UNDERLYING_ID,
    FED_FUNDS_RATE_UNDERLYING_ID,
    THERIODIC_UNDERLYING_ID,
    BinaryOption,
    FokOrder,
    OptionLeg,
    OrderType,
    Quote,
    Underlying,
)
from performance_harness import (
    NetRiskMarketMaker,
    SessionResult,
    candidate_is_accepted,
)


def make_option(
    option_id: int = 101,
    underlying_id: int = AJARAI_UNDERLYING_ID,
    steps: int = 1,
) -> BinaryOption:
    return BinaryOption((OptionLeg(underlying_id, 1.0),), option_id, steps, 100.0)


def make_net_market_maker(
    cash: float, options: list[BinaryOption]
) -> NetRiskMarketMaker:
    return NetRiskMarketMaker(
        [
            Underlying("FED", FED_FUNDS_RATE_UNDERLYING_ID, 2.0),
            Underlying("AJR", AJARAI_UNDERLYING_ID, 100.0),
            Underlying("THR", THERIODIC_UNDERLYING_ID, 100.0),
        ],
        options,
        cash,
    )


class PerformanceHarnessAcceptanceTests(unittest.TestCase):
    def test_candidate_passes_at_the_exact_paired_loss_tolerance(self) -> None:
        gross_results = [
            SessionResult(pnl=0.0, bankrupt=False, trade_count=1),
            SessionResult(pnl=0.0, bankrupt=False, trade_count=1),
        ]
        candidate_results = [
            SessionResult(pnl=0.10, bankrupt=False, trade_count=2),
            SessionResult(pnl=-0.05, bankrupt=False, trade_count=2),
        ]

        accepted, worst_paired_delta = candidate_is_accepted(
            gross_results, candidate_results
        )

        self.assertTrue(accepted)
        self.assertAlmostEqual(worst_paired_delta, -0.05)

    def test_paired_regression_fails_even_when_global_worst_pnl_improves(self) -> None:
        gross_results = [
            SessionResult(pnl=2.0, bankrupt=False, trade_count=1),
            SessionResult(pnl=-5.0, bankrupt=False, trade_count=1),
        ]
        candidate_results = [
            SessionResult(pnl=0.12, bankrupt=False, trade_count=2),
            SessionResult(pnl=-2.90, bankrupt=False, trade_count=2),
        ]

        accepted, worst_paired_delta = candidate_is_accepted(
            gross_results, candidate_results
        )

        self.assertFalse(accepted)
        self.assertAlmostEqual(worst_paired_delta, -1.88)

    def test_unpaired_results_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty and equally sized"):
            candidate_is_accepted(
                [SessionResult(pnl=0.0, bankrupt=False, trade_count=0)],
                [],
            )

    def test_bankruptcy_or_non_improving_mean_rejects_candidate(self) -> None:
        gross_results = [SessionResult(0.0, False, 1)]

        bankrupt, bankrupt_delta = candidate_is_accepted(
            gross_results, [SessionResult(0.10, True, 2)]
        )
        equal_mean, equal_mean_delta = candidate_is_accepted(
            gross_results, [SessionResult(0.0, False, 2)]
        )

        self.assertFalse(bankrupt)
        self.assertFalse(equal_mean)
        self.assertAlmostEqual(bankrupt_delta, 0.10)
        self.assertAlmostEqual(equal_mean_delta, 0.0)


class NetRiskExperimentTests(unittest.TestCase):
    def test_offsetting_fills_release_collateral_and_lock_in_spread(self) -> None:
        option = make_option()
        maker = make_net_market_maker(1.0, [option])

        maker.on_trade(option, price=0.40, quantity=1, counterparty_id=1)
        maker.on_trade(option, price=0.60, quantity=-1, counterparty_id=2)

        self.assertEqual(maker.get_position(option.option_id), 0)
        self.assertAlmostEqual(maker.cash_balance, 1.20)
        self.assertAlmostEqual(maker.remaining_risk_budget, 1.20)

    def test_risk_reducing_quote_and_fok_work_without_free_collateral(self) -> None:
        option = make_option()
        maker = make_net_market_maker(0.40, [option])
        maker.on_trade(option, price=0.60, quantity=-1, counterparty_id=1)
        self.assertAlmostEqual(maker.remaining_risk_budget, 0.0)

        with patch.object(maker, "price_option", return_value=0.50):
            quote = maker.quote(option, counterparty_id=2)
            accepts_fok = maker.respond_to_fok(
                option,
                FokOrder(2, option.option_id, OrderType.SELL, 0.49, 1),
            )

        self.assertEqual(quote, Quote(0.49, 2, 1.0, 1))
        self.assertTrue(accepts_fok)

    def test_uncovered_quote_and_fok_stop_before_capital_turns_negative(
        self,
    ) -> None:
        option = make_option()
        maker = make_net_market_maker(0.50, [option])

        with patch.object(maker, "price_option", return_value=0.50):
            quote = maker.quote(option, counterparty_id=1)
            accepts_long = maker.respond_to_fok(
                option,
                FokOrder(1, option.option_id, OrderType.SELL, 0.49, 2),
            )
            accepts_short = maker.respond_to_fok(
                option,
                FokOrder(1, option.option_id, OrderType.BUY, 0.51, 2),
            )

        self.assertEqual(quote, Quote(0.49, 1, 0.51, 1))
        self.assertFalse(accepts_long)
        self.assertFalse(accepts_short)

    def test_short_liabilities_remain_separate_across_options(self) -> None:
        first_option = make_option(option_id=101)
        second_option = make_option(
            option_id=102, underlying_id=THERIODIC_UNDERLYING_ID
        )
        maker = make_net_market_maker(2.0, [first_option, second_option])

        maker.on_trade(first_option, price=0.80, quantity=-2, counterparty_id=1)
        maker.on_trade(second_option, price=0.25, quantity=-2, counterparty_id=2)

        self.assertAlmostEqual(maker.cash_balance, 4.10)
        self.assertAlmostEqual(maker.remaining_risk_budget, 0.10)

    def test_expiry_settles_net_cash_and_releases_liability(self) -> None:
        option = make_option(steps=1)
        long_maker = make_net_market_maker(0.40, [option])
        short_maker = make_net_market_maker(0.40, [option])
        long_maker.on_trade(option, price=0.40, quantity=1, counterparty_id=1)
        short_maker.on_trade(option, price=0.60, quantity=-1, counterparty_id=2)
        next_underlyings = [
            Underlying("FED", FED_FUNDS_RATE_UNDERLYING_ID, 2.0),
            Underlying("AJR", AJARAI_UNDERLYING_ID, 101.0),
            Underlying("THR", THERIODIC_UNDERLYING_ID, 100.0),
        ]

        long_maker.on_step_advance(next_underlyings, [])
        short_maker.on_step_advance(next_underlyings, [])

        self.assertAlmostEqual(long_maker.cash_balance, 1.0)
        self.assertAlmostEqual(long_maker.remaining_risk_budget, 1.0)
        self.assertAlmostEqual(short_maker.cash_balance, 0.0)
        self.assertAlmostEqual(short_maker.remaining_risk_budget, 0.0)


if __name__ == "__main__":
    unittest.main()
