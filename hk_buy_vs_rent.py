#!/usr/bin/env python3
"""
Hong Kong Property: Buy vs Rent Analysis

Compares the financial outcomes of buying vs renting property in Hong Kong,
including opportunity cost of capital invested in risk assets.

Author: Claude (for HK relocation planning)
"""

import json
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Assumptions:
    """All model assumptions in one place for easy adjustment"""

    # Property details
    property_value_hkd: float = 50_000_000  # HK$50m property

    # Buy scenario
    deposit_pct: float = 0.30  # 30% deposit (HKMA max LTV 70% as of Oct 2024)
    mortgage_rate_annual: float = 0.04  # 4% p.a. (current HK rates ~3.5-4.5%)
    mortgage_term_years: int = 25
    stamp_duty_rate: float = 0.0425  # 4.25% AVD
    legal_buying_costs_pct: float = 0.01  # ~1% legal/conveyancing
    annual_maintenance_pct: float = 0.005  # 0.5% of property value
    management_fee_monthly_hkd: float = 8_000  # Building management fee
    rates_pct_of_rateable: float = 0.05  # Government rates
    rateable_value_pct: float = 0.04  # Rateable value as % of property value (approx)
    property_appreciation_annual: float = 0.03  # 3% p.a. (conservative for HK)
    selling_costs_pct: float = 0.02  # Agent fees, legal on exit

    # Rent scenario
    monthly_rent_hkd: float = 150_000  # HK$150k/month for comparable property
    rent_inflation_annual: float = 0.02  # 2% annual rent increase
    rent_deposit_months: int = 2  # Standard 2 months deposit

    # Investment counterfactual
    investment_return_annual: float = 0.20  # 20% p.a. as specified

    # Analysis period
    analysis_years: int = 10  # Model out to 10 years

    # Currency
    hkd_to_gbp: float = 0.10  # Approximate conversion


def calculate_monthly_mortgage_payment(
    principal: float,
    annual_rate: float,
    years: int
) -> float:
    """Calculate monthly mortgage payment using standard amortization formula"""
    if annual_rate == 0:
        return principal / (years * 12)

    monthly_rate = annual_rate / 12
    num_payments = years * 12

    payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
              ((1 + monthly_rate)**num_payments - 1)

    return payment


def calculate_mortgage_balance(
    principal: float,
    annual_rate: float,
    years: int,
    months_paid: int
) -> float:
    """Calculate remaining mortgage balance after n months"""
    if annual_rate == 0:
        return principal - (principal / (years * 12)) * months_paid

    monthly_rate = annual_rate / 12
    num_payments = years * 12
    monthly_payment = calculate_monthly_mortgage_payment(principal, annual_rate, years)

    # Remaining balance formula
    balance = principal * ((1 + monthly_rate)**months_paid) - \
              monthly_payment * (((1 + monthly_rate)**months_paid - 1) / monthly_rate)

    return max(0, balance)


def calculate_mortgage_interest_paid(
    principal: float,
    annual_rate: float,
    years: int,
    months_paid: int
) -> float:
    """Calculate total interest paid over n months"""
    monthly_payment = calculate_monthly_mortgage_payment(principal, annual_rate, years)
    total_paid = monthly_payment * months_paid
    principal_paid = principal - calculate_mortgage_balance(principal, annual_rate, years, months_paid)
    return total_paid - principal_paid


def run_buy_scenario(a: Assumptions, hold_years: int) -> dict:
    """
    Calculate total costs and equity position for buying scenario
    """
    months = hold_years * 12

    # Initial costs
    deposit = a.property_value_hkd * a.deposit_pct
    stamp_duty = a.property_value_hkd * a.stamp_duty_rate
    legal_costs = a.property_value_hkd * a.legal_buying_costs_pct
    initial_cash_outlay = deposit + stamp_duty + legal_costs

    # Mortgage
    loan_amount = a.property_value_hkd * (1 - a.deposit_pct)
    monthly_mortgage = calculate_monthly_mortgage_payment(
        loan_amount, a.mortgage_rate_annual, a.mortgage_term_years
    )
    total_mortgage_payments = monthly_mortgage * months
    interest_paid = calculate_mortgage_interest_paid(
        loan_amount, a.mortgage_rate_annual, a.mortgage_term_years, months
    )
    remaining_balance = calculate_mortgage_balance(
        loan_amount, a.mortgage_rate_annual, a.mortgage_term_years, months
    )

    # Ongoing costs (summed over holding period, with property value appreciation)
    total_maintenance = 0
    total_management = 0
    total_rates = 0

    for year in range(hold_years):
        property_value_year = a.property_value_hkd * ((1 + a.property_appreciation_annual) ** year)
        total_maintenance += property_value_year * a.annual_maintenance_pct
        total_management += a.management_fee_monthly_hkd * 12
        total_rates += property_value_year * a.rateable_value_pct * a.rates_pct_of_rateable

    # Exit
    final_property_value = a.property_value_hkd * ((1 + a.property_appreciation_annual) ** hold_years)
    selling_costs = final_property_value * a.selling_costs_pct

    # Net proceeds on sale
    gross_sale_proceeds = final_property_value
    net_sale_proceeds = gross_sale_proceeds - remaining_balance - selling_costs

    # Total cash flows
    total_cash_out = (
        initial_cash_outlay +
        total_mortgage_payments +
        total_maintenance +
        total_management +
        total_rates
    )

    total_cash_in = net_sale_proceeds

    # Unrecoverable costs (money that doesn't come back)
    unrecoverable = (
        stamp_duty +
        legal_costs +
        interest_paid +
        total_maintenance +
        total_management +
        total_rates +
        selling_costs
    )

    # Net position
    net_wealth_change = total_cash_in - total_cash_out + deposit  # Add back deposit as it's returned via equity

    return {
        "hold_years": hold_years,
        "initial_cash_outlay": initial_cash_outlay,
        "deposit": deposit,
        "stamp_duty": stamp_duty,
        "legal_costs": legal_costs,
        "monthly_mortgage": monthly_mortgage,
        "total_mortgage_payments": total_mortgage_payments,
        "interest_paid": interest_paid,
        "total_maintenance": total_maintenance,
        "total_management": total_management,
        "total_rates": total_rates,
        "final_property_value": final_property_value,
        "property_gain": final_property_value - a.property_value_hkd,
        "remaining_mortgage": remaining_balance,
        "selling_costs": selling_costs,
        "net_sale_proceeds": net_sale_proceeds,
        "total_cash_out": total_cash_out,
        "unrecoverable_costs": unrecoverable,
        "net_wealth_change": net_wealth_change,
    }


def run_rent_scenario(a: Assumptions, hold_years: int) -> dict:
    """
    Calculate total costs for renting, plus investment returns on capital
    """
    months = hold_years * 12

    # Capital that would have been used for deposit + buying costs
    capital_available = (
        a.property_value_hkd * a.deposit_pct +  # Deposit
        a.property_value_hkd * a.stamp_duty_rate +  # Stamp duty saved
        a.property_value_hkd * a.legal_buying_costs_pct  # Legal costs saved
    )

    # Rent payments (with inflation)
    total_rent = 0
    for year in range(hold_years):
        annual_rent = a.monthly_rent_hkd * 12 * ((1 + a.rent_inflation_annual) ** year)
        total_rent += annual_rent

    # Rent deposit (returned at end, no return assumed)
    rent_deposit = a.monthly_rent_hkd * a.rent_deposit_months

    # Investment returns on capital
    # Compound the capital that would have gone to deposit/stamp duty
    investment_end_value = capital_available * ((1 + a.investment_return_annual) ** hold_years)
    investment_gain = investment_end_value - capital_available

    # Additionally, model the monthly cash flow difference
    # Approximate: assume mortgage payment vs rent difference is invested monthly
    # (This is a simplification - in practice you'd model monthly)

    # Net position
    net_wealth_change = investment_gain - total_rent + capital_available

    return {
        "hold_years": hold_years,
        "capital_available_to_invest": capital_available,
        "total_rent_paid": total_rent,
        "rent_deposit": rent_deposit,
        "investment_end_value": investment_end_value,
        "investment_gain": investment_gain,
        "net_wealth_change": net_wealth_change,
    }


def compare_scenarios(a: Assumptions) -> List[dict]:
    """Run both scenarios for each year and compare"""
    results = []

    for year in range(1, a.analysis_years + 1):
        buy = run_buy_scenario(a, year)
        rent = run_rent_scenario(a, year)

        buy_advantage = buy["net_wealth_change"] - rent["net_wealth_change"]

        results.append({
            "year": year,
            "buy": buy,
            "rent": rent,
            "buy_advantage_hkd": buy_advantage,
            "buy_advantage_gbp": buy_advantage * a.hkd_to_gbp,
            "recommendation": "BUY" if buy_advantage > 0 else "RENT",
        })

    return results


def print_summary(a: Assumptions, results: List[dict]):
    """Print a formatted summary of results"""

    print("=" * 80)
    print("HONG KONG PROPERTY: BUY VS RENT ANALYSIS")
    print("=" * 80)
    print()

    print("KEY ASSUMPTIONS")
    print("-" * 40)
    print(f"Property Value:          HK${a.property_value_hkd:,.0f} (£{a.property_value_hkd * a.hkd_to_gbp:,.0f})")
    print(f"Monthly Rent:            HK${a.monthly_rent_hkd:,.0f} (£{a.monthly_rent_hkd * a.hkd_to_gbp:,.0f})")
    print(f"Deposit:                 {a.deposit_pct:.0%}")
    print(f"Mortgage Rate:           {a.mortgage_rate_annual:.1%} p.a.")
    print(f"Property Appreciation:   {a.property_appreciation_annual:.1%} p.a.")
    print(f"Rent Inflation:          {a.rent_inflation_annual:.1%} p.a.")
    print(f"Investment Return:       {a.investment_return_annual:.0%} p.a.")
    print(f"Stamp Duty:              {a.stamp_duty_rate:.2%}")
    print()

    # Initial costs breakdown
    deposit = a.property_value_hkd * a.deposit_pct
    stamp_duty = a.property_value_hkd * a.stamp_duty_rate
    legal = a.property_value_hkd * a.legal_buying_costs_pct
    total_upfront = deposit + stamp_duty + legal

    print("UPFRONT COSTS (BUY)")
    print("-" * 40)
    print(f"Deposit (40%):           HK${deposit:,.0f}")
    print(f"Stamp Duty (4.25%):      HK${stamp_duty:,.0f}")
    print(f"Legal Costs (~1%):       HK${legal:,.0f}")
    print(f"TOTAL UPFRONT:           HK${total_upfront:,.0f} (£{total_upfront * a.hkd_to_gbp:,.0f})")
    print()

    print("YEAR-BY-YEAR COMPARISON")
    print("-" * 80)
    print(f"{'Year':<6} {'Buy Net':<18} {'Rent Net':<18} {'Advantage':<18} {'Verdict':<10}")
    print("-" * 80)

    for r in results:
        buy_net = r["buy"]["net_wealth_change"]
        rent_net = r["rent"]["net_wealth_change"]
        adv = r["buy_advantage_hkd"]

        print(f"{r['year']:<6} HK${buy_net:>14,.0f} HK${rent_net:>14,.0f} HK${adv:>14,.0f} {r['recommendation']:<10}")

    print()

    # Detailed breakdown for key years
    for target_year in [6, 10]:
        if target_year <= a.analysis_years:
            r = results[target_year - 1]
            print("=" * 80)
            print(f"DETAILED BREAKDOWN: YEAR {target_year}")
            print("=" * 80)
            print()

            print("BUY SCENARIO")
            print("-" * 40)
            b = r["buy"]
            print(f"Initial Cash Outlay:     HK${b['initial_cash_outlay']:>15,.0f}")
            print(f"  - Deposit:             HK${b['deposit']:>15,.0f}")
            print(f"  - Stamp Duty:          HK${b['stamp_duty']:>15,.0f}")
            print(f"  - Legal Costs:         HK${b['legal_costs']:>15,.0f}")
            print()
            print(f"Monthly Mortgage:        HK${b['monthly_mortgage']:>15,.0f}")
            print(f"Total Mortgage Paid:     HK${b['total_mortgage_payments']:>15,.0f}")
            print(f"  - Interest Portion:    HK${b['interest_paid']:>15,.0f}")
            print()
            print(f"Ongoing Costs:")
            print(f"  - Maintenance:         HK${b['total_maintenance']:>15,.0f}")
            print(f"  - Management Fees:     HK${b['total_management']:>15,.0f}")
            print(f"  - Rates:               HK${b['total_rates']:>15,.0f}")
            print()
            print(f"Property Value (exit):   HK${b['final_property_value']:>15,.0f}")
            print(f"Property Gain:           HK${b['property_gain']:>15,.0f}")
            print(f"Remaining Mortgage:      HK${b['remaining_mortgage']:>15,.0f}")
            print(f"Selling Costs:           HK${b['selling_costs']:>15,.0f}")
            print(f"Net Sale Proceeds:       HK${b['net_sale_proceeds']:>15,.0f}")
            print()
            print(f"UNRECOVERABLE COSTS:     HK${b['unrecoverable_costs']:>15,.0f}")
            print(f"NET WEALTH CHANGE:       HK${b['net_wealth_change']:>15,.0f}")
            print()

            print("RENT SCENARIO")
            print("-" * 40)
            rt = r["rent"]
            print(f"Capital Invested:        HK${rt['capital_available_to_invest']:>15,.0f}")
            print(f"Total Rent Paid:         HK${rt['total_rent_paid']:>15,.0f}")
            print(f"Investment End Value:    HK${rt['investment_end_value']:>15,.0f}")
            print(f"Investment Gain:         HK${rt['investment_gain']:>15,.0f}")
            print()
            print(f"NET WEALTH CHANGE:       HK${rt['net_wealth_change']:>15,.0f}")
            print()

            print("COMPARISON")
            print("-" * 40)
            print(f"Buy Advantage:           HK${r['buy_advantage_hkd']:>15,.0f}")
            print(f"                         £{r['buy_advantage_gbp']:>16,.0f}")
            print(f"RECOMMENDATION:          {r['recommendation']}")
            print()


def sensitivity_analysis(base: Assumptions) -> None:
    """Run sensitivity analysis on key variables"""

    print("=" * 80)
    print("SENSITIVITY ANALYSIS (6-YEAR HORIZON)")
    print("=" * 80)
    print()

    hold_years = 6

    # Investment return sensitivity
    print("INVESTMENT RETURN SENSITIVITY")
    print("-" * 60)
    print(f"{'Return':<15} {'Buy Advantage':<20} {'Recommendation':<15}")
    print("-" * 60)

    for return_rate in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        a = Assumptions(investment_return_annual=return_rate)
        buy = run_buy_scenario(a, hold_years)
        rent = run_rent_scenario(a, hold_years)
        adv = buy["net_wealth_change"] - rent["net_wealth_change"]
        rec = "BUY" if adv > 0 else "RENT"
        print(f"{return_rate:.0%}            HK${adv:>15,.0f}    {rec}")

    print()

    # Property appreciation sensitivity
    print("PROPERTY APPRECIATION SENSITIVITY")
    print("-" * 60)
    print(f"{'Appreciation':<15} {'Buy Advantage':<20} {'Recommendation':<15}")
    print("-" * 60)

    for prop_rate in [-0.02, 0.00, 0.02, 0.03, 0.05, 0.08]:
        a = Assumptions(property_appreciation_annual=prop_rate)
        buy = run_buy_scenario(a, hold_years)
        rent = run_rent_scenario(a, hold_years)
        adv = buy["net_wealth_change"] - rent["net_wealth_change"]
        rec = "BUY" if adv > 0 else "RENT"
        print(f"{prop_rate:+.0%}            HK${adv:>15,.0f}    {rec}")

    print()

    # Rent level sensitivity (as % of property value - yield)
    print("RENTAL YIELD SENSITIVITY (rent as % of property value p.a.)")
    print("-" * 60)
    print(f"{'Yield':<15} {'Monthly Rent':<20} {'Buy Advantage':<20} {'Rec':<10}")
    print("-" * 60)

    for yield_rate in [0.025, 0.030, 0.036, 0.040, 0.045, 0.050]:
        monthly_rent = (base.property_value_hkd * yield_rate) / 12
        a = Assumptions(monthly_rent_hkd=monthly_rent)
        buy = run_buy_scenario(a, hold_years)
        rent = run_rent_scenario(a, hold_years)
        adv = buy["net_wealth_change"] - rent["net_wealth_change"]
        rec = "BUY" if adv > 0 else "RENT"
        print(f"{yield_rate:.1%}           HK${monthly_rent:>12,.0f}     HK${adv:>12,.0f}    {rec}")

    print()


def breakeven_analysis(a: Assumptions) -> None:
    """Find breakeven points for key variables"""

    print("=" * 80)
    print("BREAKEVEN ANALYSIS (6-YEAR HORIZON)")
    print("=" * 80)
    print()

    hold_years = 6

    # Find breakeven investment return
    for return_rate in range(1, 50):
        rate = return_rate / 100
        test_a = Assumptions(investment_return_annual=rate)
        buy = run_buy_scenario(test_a, hold_years)
        rent = run_rent_scenario(test_a, hold_years)
        if rent["net_wealth_change"] > buy["net_wealth_change"]:
            print(f"Investment Return Breakeven: ~{rate:.0%} p.a.")
            print(f"  Above this return, RENT is better")
            print(f"  Below this return, BUY is better")
            break

    print()

    # Find breakeven property appreciation
    for prop_rate in range(-10, 20):
        rate = prop_rate / 100
        test_a = Assumptions(property_appreciation_annual=rate)
        buy = run_buy_scenario(test_a, hold_years)
        rent = run_rent_scenario(test_a, hold_years)
        if buy["net_wealth_change"] > rent["net_wealth_change"]:
            print(f"Property Appreciation Breakeven: ~{rate:.0%} p.a.")
            print(f"  Above this appreciation, BUY is better")
            print(f"  Below this appreciation, RENT is better")
            break

    print()


def main():
    """Main entry point"""

    # Create base assumptions
    assumptions = Assumptions()

    # Run comparison
    results = compare_scenarios(assumptions)

    # Print summary
    print_summary(assumptions, results)

    # Sensitivity analysis
    sensitivity_analysis(assumptions)

    # Breakeven analysis
    breakeven_analysis(assumptions)

    print("=" * 80)
    print("NOTES")
    print("=" * 80)
    print("""
1. This model assumes you sell the property at the end of the holding period.
   If keeping for tax residency, the comparison changes (no selling costs,
   ongoing rental income vs mortgage payments).

2. The 20% p.a. investment return is aggressive. Historical equity returns
   are closer to 7-10% p.a. Adjust the assumption to match your expectations.

3. Mortgage rates in HK are currently ~3.5-4.5% (HIBOR-linked). This model
   uses 4% but rates may vary over your holding period.

4. Property appreciation in HK has been negative recently (-5% to -15% from
   peak). The 3% assumption is long-term average; short-term may differ.

5. Rental yields in HK luxury segment are typically 2.5-3.5% gross.
   The model uses HK$150k/month on HK$50m = 3.6% yield.

6. Tax implications are not modelled. HK has no capital gains tax, so
   property gains are tax-free. Investment gains are also tax-free in HK.

7. Currency risk is not modelled. If you're earning in HKD and property/
   investments are in HKD, this is neutral. If you have GBP liabilities
   or plan to repatriate, consider FX risk.

8. HKD/USD carry: HKD is pegged to USD. HK mortgage rates (HIBOR-linked)
   can differ from US rates. If investing in USD assets while borrowing
   HKD, any rate differential is implicitly captured in the investment
   return vs mortgage rate spread. The peg eliminates FX risk between
   HKD and USD, but carries tail risk if the peg ever breaks.

9. Opportunity cost of time/hassle in property ownership is not quantified.
""")


if __name__ == "__main__":
    main()
