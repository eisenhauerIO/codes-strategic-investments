import pulp as lp
import math

# --- 1. Helper Functions ---

def calculate_gamma(confidence_score):
    if not (0 <= confidence_score <= 1):
        raise ValueError("Confidence score must be between 0 and 1.")
    return 1 - confidence_score

def calculate_effective_returns(initiatives, confidence_penalty_func=calculate_gamma):
    scenarios = ['best', 'med', 'worst']
    for initiative in initiatives:
        c_i = initiative['confidence']
        gamma_i = confidence_penalty_func(c_i)
        initiative['gamma'] = gamma_i
        effective_returns = {}
        R_base_map = {
            'best': initiative['R_best'],
            'med': initiative['R_med'],
            'worst': initiative['R_worst']
        }
        for scenario_name in scenarios:
            R_ij_base = R_base_map[scenario_name]
            effective_returns[scenario_name] = (1 - gamma_i) * R_ij_base + gamma_i * initiative['R_worst']
        initiative['effective_returns'] = effective_returns
    return initiatives

def calculate_optimal_scenario_returns(initiatives, total_budget):
    scenarios = ['best', 'med', 'worst']
    V_j_star = {}
    print("\n--- Calculating Optimal Scenario Returns (V_j_star) ---")
    for scenario_name in scenarios:
        prob_scenario = lp.LpProblem(f"Optimal_Return_Scenario_{scenario_name}", lp.LpMaximize)
        y = lp.LpVariable.dicts("Select", [i['id'] for i in initiatives], 0, 1, lp.LpBinary)
        prob_scenario += lp.lpSum(y[i['id']] * i['effective_returns'][scenario_name] for i in initiatives)
        prob_scenario += lp.lpSum(y[i['id']] * i['cost'] for i in initiatives) <= total_budget
        try:
            prob_scenario.solve(lp.PULP_CBC_CMD(msg=False))
        except Exception as e:
            print(f"Error solving for scenario {scenario_name}: {e}")
            V_j_star[scenario_name] = -math.inf
            continue
        if lp.LpStatus[prob_scenario.status] == 'Optimal':
            V_j_star[scenario_name] = lp.value(prob_scenario.objective)
            print(f"  Scenario '{scenario_name}': V_j_star = {V_j_star[scenario_name]:.2f}")
        else:
            V_j_star[scenario_name] = -math.inf
            print(f"  Scenario '{scenario_name}': Problem status = {lp.LpStatus[prob_scenario.status]}")
    return V_j_star

# --- 2. Main Optimization Function ---

def solve_minimax_regret_optimization(
    initiatives_data, 
    total_budget, 
    min_confidence_threshold, 
    min_portfolio_worst_return,
    confidence_penalty_func=calculate_gamma
):
    eligible_initiatives = [i for i in initiatives_data if i['confidence'] >= min_confidence_threshold]
    if not eligible_initiatives:
        return {
            'status': 'No Eligible Initiatives',
            'min_max_regret': None,
            'selected_initiatives': [],
            'total_cost': 0,
            'total_actual_returns': {'best': 0, 'med': 0, 'worst': 0},
            'v_j_star': {},
            'regrets_for_selected_portfolio': {}
        }

    processed_initiatives = calculate_effective_returns(eligible_initiatives, confidence_penalty_func)
    V_j_star = calculate_optimal_scenario_returns(processed_initiatives, total_budget)

    if any(val == -math.inf for val in V_j_star.values()):
        return {
            'status': 'Error in V_j_star calculation',
            'min_max_regret': None,
            'selected_initiatives': [],
            'total_cost': 0,
            'total_actual_returns': {'best': 0, 'med': 0, 'worst': 0},
            'v_j_star': V_j_star,
            'regrets_for_selected_portfolio': {}
        }

    print("\n--- Formulating Minimax Regret Problem ---")
    prob = lp.LpProblem("Minimax_Regret_Investment_Portfolio", lp.LpMinimize)
    x = lp.LpVariable.dicts("Select", [i['id'] for i in processed_initiatives], 0, 1, lp.LpBinary)
    theta = lp.LpVariable("Max_Regret", lowBound=0)
    prob += theta

    scenarios = ['best', 'med', 'worst']
    for scenario_name in scenarios:
        prob += theta >= V_j_star[scenario_name] - lp.lpSum(
            x[i['id']] * i['effective_returns'][scenario_name] for i in processed_initiatives)

    prob += lp.lpSum(x[i['id']] * i['cost'] for i in processed_initiatives) <= total_budget
    prob += lp.lpSum(x[i['id']] * i['R_worst'] for i in processed_initiatives) >= min_portfolio_worst_return

    print("Solving the main optimization problem...")
    try:
        prob.solve(lp.PULP_CBC_CMD(msg=False))
    except Exception as e:
        return {
            'status': f"Error solving main problem: {e}",
            'min_max_regret': None,
            'selected_initiatives': [],
            'total_cost': 0,
            'total_actual_returns': {'best': 0, 'med': 0, 'worst': 0},
            'v_j_star': V_j_star,
            'regrets_for_selected_portfolio': {}
        }

    results = {}
    results['status'] = lp.LpStatus[prob.status]
    results['min_max_regret'] = lp.value(prob.objective) if prob.status == lp.LpStatusOptimal else None

    selected_initiatives = []
    total_cost = 0
    total_actual_returns = {s: 0 for s in scenarios}
    regrets_for_selected_portfolio = {s: 0 for s in scenarios}

    if prob.status == lp.LpStatusOptimal:
        for i in processed_initiatives:
            if x[i['id']].varValue > 0.5:
                selected_initiatives.append(i['id'])
                total_cost += i['cost']
                for scenario_name in scenarios:
                    total_actual_returns[scenario_name] += i['effective_returns'][scenario_name]

        for scenario_name in scenarios:
            regrets_for_selected_portfolio[scenario_name] = V_j_star[scenario_name] - total_actual_returns[scenario_name]

    results['selected_initiatives'] = selected_initiatives
    results['total_cost'] = total_cost
    results['total_actual_returns'] = total_actual_returns
    results['v_j_star'] = V_j_star
    results['regrets_for_selected_portfolio'] = regrets_for_selected_portfolio

    return results
