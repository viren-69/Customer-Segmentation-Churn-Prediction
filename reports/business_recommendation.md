# Business recommendation: reducing churn and growing repeat purchases

*One-page summary for management. The analysis is in [notebooks/](../notebooks/).*

## Telco: who leaves, and what to do about it

**Situation.** 26.5% of our 7,043 customers churned last quarter. Churn is concentrated, not random:

| Group | Churn rate |
|---|---|
| Month-to-month contract | **42.7%** (vs 2.8% on two-year contracts) |
| Month-to-month **and** fiber optic (2,128 customers) | **54.6%** |
| First 6 months as a customer | **52.9%** (vs 9.5% after 4 years) |
| No tech support add-on | 31.2% (vs 15.2% with it) |

**The model.** A logistic regression scores each customer's churn risk. On customers it never saw during training, it ranks churners above loyal customers 85% of the time (ROC-AUC 0.85). Contacting customers with a risk of 21% or more reaches **86% of the customers who would churn**, while contacting less than half of the base.

**Value.** On the 1,409-customer test group, under the assumptions below, a model-targeted retention campaign is worth about **$52.6K**, vs $44.0K at the default cut-off and $27.5K for contacting everyone. Across the whole base that's roughly five times as much.
*Assumptions: $50 cost per contacted customer; the offer keeps 30% of real churners; a kept customer is worth 12 months of their bill. These are placeholders. Across 27 combinations of cost ($25–100), save rate (20–50%) and value (6–24 months), model targeting always beat contacting everyone. But the best risk cut-off moved between 5% and 83%, so **finance's real figures are needed before launch**.*

### Recommendations
1. **Convert month-to-month customers to one-year contracts**, starting with fiber customers. The model estimates their churn risk would fall from 43% to 29%. Because contract choice isn't random, **A/B test the offer first**: give a contract discount to a random half and compare.
2. **Fix the first 6 months:** a welcome call, a 30-day check-in, and early help with setup. Over half of new customers leave in this window.
3. **Include tech support / online security free for the first 3 months** on fiber plans. Customers with these add-ons churn half as often.
4. **Investigate fiber:** it's the premium product, yet its customers churn at twice the DSL rate. Check pricing against competitors and outage/speed complaints.
5. **Run the model weekly.** Give agents each at-risk customer's top reasons (SHAP), so the offer matches the reason (contract discount vs support vs price).
6. *Lower priority:* nudge electronic-check payers to auto-pay. Their high churn is mostly explained by their contracts and tenure, so expect a modest effect.

## Olist marketplace: customer segments (RFM)

Only **3%** of customers ever buy twice, so growth depends on second purchases.

| Segment | % customers | % revenue | Action |
|---|---|---|---|
| Recent big spenders | 29% | **42%** | Second-purchase offer within 30 days; cross-sell |
| Lapsed big spenders | 19% | **36%** | Personal win-back offer; ask why they left |
| Recent small spenders | 30% | 9% | Free-shipping threshold, bundles |
| Lapsed small spenders | 19% | 7% | Cheap automated emails only |
| Repeat buyers | 3% | 6% | Loyalty perks; reviews and referrals |

**Priority:** the two "big spender" segments hold **78% of revenue** in 48% of customers. Focus campaign budget there.

## Next steps
- A/B test the contract offer and the onboarding programme; measure churn after 90 days.
- Score customers monthly and retrain the model quarterly; alert if accuracy drops.
- Replace the profit assumptions with finance's real offer cost and customer value.
