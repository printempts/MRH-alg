import math
import sys
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import time
import matplotlib.pyplot as plt

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper

def load_dataset(file):
    """ Loads the raw-unstructured data from the text file"""
    try:
        dt_raw = pd.read_csv(file, header=None)
        return dt_raw
    except:
        print(
            "Oops! Error on file import. Make sure you have a folder named 'datasets' with the assets inside and try again...")

def find_largest_n_elements(arr, n):
    sorted_indices = [idx for idx, _ in sorted(enumerate(arr), key=lambda x: x[1], reverse=True)]
    largest_n_indices = sorted_indices[:n]
    return largest_n_indices

def data_preprocessing(dataset):
    assets_num = int(dataset[0][0])

    ### dataset with details of each available asset (expected return, standard deviation)
    ds_asset_details = dataset.iloc[1:assets_num + 1, 0]
    ds_asset_details = ds_asset_details.str.split(' ', expand=True, n=2)
    ds_asset_details.drop(ds_asset_details.columns[[0]], axis=1, inplace=True)
    ds_asset_details.columns = ['ExpReturn', 'StDev']
    # Convert both columns from string to float
    ds_asset_details['ExpReturn'] = ds_asset_details['ExpReturn'].astype(float)
    ds_asset_details['StDev'] = ds_asset_details['StDev'].astype(float)

    ds_correlations = dataset.iloc[assets_num + 1:, 0]
    ds_correlations = ds_correlations.str.split(' ', expand=True, n=3)
    ds_correlations.drop(ds_correlations.columns[[0]], axis=1, inplace=True)
    ds_correlations.columns = ['Asset1', 'Asset2', 'Correlation']

    ds_correlations['Asset1'] = ds_correlations['Asset1'].astype(int)
    ds_correlations['Asset2'] = ds_correlations['Asset2'].astype(int)
    ds_correlations['Correlation'] = ds_correlations['Correlation'].astype(float)

    ds_rho = pd.DataFrame(index=range(1, assets_num + 1), columns=range(1, assets_num + 1))
    for i in range(len(ds_correlations)):
        ds_rho.iloc[ds_correlations.iloc[i, 0] - 1, ds_correlations.iloc[i, 1] - 1] = ds_correlations.iloc[i, 2]
        ds_rho.iloc[ds_correlations.iloc[i, 1] - 1, ds_correlations.iloc[i, 0] - 1] = ds_correlations.iloc[i, 2]

    rho = np.array(ds_rho.iloc[0].tolist())
    for i in range(1, len(ds_rho)):
        rho = np.append(rho, ds_rho.iloc[i].tolist(), axis=0)
    rho = rho.reshape((assets_num, assets_num))

    return assets_num, ds_asset_details, ds_correlations, rho

#   处理文件得到assets_num, asset_return, asset_cov, asset_cov_inv
def get_inf(file):
    dt_raw = load_dataset(file)
    assets_num, asset_details, asset_correlations, asset_rho = data_preprocessing(dt_raw)
    asset_return = np.array(asset_details.ExpReturn.tolist())
    asset_return = np.around(asset_return,6)
    std = np.array(asset_details.StDev.tolist())
    std = np.around(std,6)
    correlation = np.array(asset_rho)
    correlation = np.around(correlation,6)
    asset_cov = np.asarray(correlation * std * std.reshape((std.shape[0], 1)))
    asset_cov_inv = np.linalg.inv(asset_cov)
    asset_cov = np.around(asset_cov, 18)
    asset_cov_inv = np.around(asset_cov_inv, 18)
    return assets_num, asset_return, asset_cov, asset_cov_inv


#   to solve primal problem (PB)
def primal_solution_B(total_rho, k, assets_num, asset_return, asset_cov):
    port_opt = Model(name='PortfolioOptimization')
    port_opt.context.cplex_parameters.threads = 1

    x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}
    y = {i: port_opt.binary_var(name='y_{0}'.format(i)) for i in range(0, assets_num)}

    port_opt.minimize(
        port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)))

    port_opt.add_constraint(port_opt.sum(x[i] for i in range(0, assets_num)) == 1)
    port_opt.add_constraint(port_opt.sum(y[i] for i in range(0, assets_num)) == k)
    port_opt.add_constraint(port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)) == total_rho)
    for i in range(0, assets_num):
        port_opt.add_constraint(x[i] >= 0.01 * y[i])
        port_opt.add_constraint(x[i] <= y[i])

    port_opt.solve()

    values_primal = []
    for i in range(0, assets_num):
        values_primal.append(port_opt.solution.get_value("y_{0}".format(i)))

   
    return port_opt.objective_value, values_primal


#   to solve relaxed primal problem (RB)
def primal_solution_relaxed_B(total_rho, k, assets_num, asset_return, asset_cov):
    port_opt = Model(name='PortfolioOptimization')
    port_opt.context.cplex_parameters.threads = 1
    x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}
    y = {i: port_opt.continuous_var(name='y_{0}'.format(i)) for i in range(0, assets_num)}

    port_opt.minimize(
        port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)))

    port_opt.add_constraint(port_opt.sum(x[i] for i in range(0, assets_num)) == 1)
    port_opt.add_constraint(port_opt.sum(y[i] for i in range(0, assets_num)) == k)
    port_opt.add_constraint(port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)) == total_rho)
    for i in range(0, assets_num):
        port_opt.add_constraint(x[i] >= 0.01 * y[i])
    for i in range(0, assets_num):
        port_opt.add_constraint(x[i] <= y[i])
    for i in range(0, assets_num):
        port_opt.add_constraint(y[i] <= 1)
    for i in range(0, assets_num):
        port_opt.add_constraint(y[i] >= 0)

    port_opt.solve()

    values = []
    for i in range(0, assets_num):
        values.append(port_opt.solution.get_value("y_{0}".format(i)))
    values_relaxed = [0] * assets_num

    
    for i in find_largest_n_elements(values, 10):
        values_relaxed[i] = 1

    
    return port_opt.objective_value, values_relaxed


#   to solve dual problem (DB)
def dual_solution_B(total_rho, k, assets_num, asset_return, asset_cov_inv):
    port_opt_duel = Model(name='PortfolioOptimizationDuel')
    port_opt_duel.context.cplex_parameters.threads = 1
    lam = {i: port_opt_duel.continuous_var(name='lambda_{0}'.format(i), lb=0) for i in range(0, assets_num)}
    miu = {i: port_opt_duel.continuous_var(name='miu_{0}'.format(i), lb=0) for i in range(0, assets_num)}
    gama = port_opt_duel.continuous_var(name='gama', lb=-float("inf"))
    alpha = port_opt_duel.continuous_var(name='alpha', lb=-float("inf"))
    beta = port_opt_duel.continuous_var(name='beta', lb=-float("inf"))
    x = {i: port_opt_duel.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}

    port_opt_duel.maximize(port_opt_duel.sum(
        0.25 * asset_cov_inv[i][j] * (lam[j] - miu[j] - alpha * asset_return[j] - beta) * (
                    lam[i] - miu[i] - alpha * asset_return[i] - beta) + 0.5 * (
                    alpha * asset_return[i] + beta - lam[i] + miu[i]) * asset_cov_inv[i][j] * (
                    lam[j] - miu[j] - alpha * asset_return[j] - beta) for i in range(0, assets_num) for j in
        range(0, assets_num))+ port_opt_duel.sum(gama+0.01*lam[i]-miu[i] for i in range(0, assets_num)) -alpha * total_rho - beta - k * gama)
    

    for i in range(0, assets_num):
        port_opt_duel.add_constraint(lam[i] * 0.01 - miu[i] + gama <= 0)
        port_opt_duel.add_constraint(x[i] == port_opt_duel.sum(
            0.5 * asset_cov_inv[i][j] * (lam[j] - miu[j] - alpha * asset_return[j] - beta) for j in
            range(0, assets_num)))
        port_opt_duel.add_constraint(x[i]>=0)

    port_opt_duel.solve()
    values = []
    for i in range(0, assets_num):
        values.append(port_opt_duel.solution.get_value("x_{0}".format(i)))
    values_duel = [0] * assets_num

    
    for i in find_largest_n_elements(values, 10):
        values_duel[i] = 1

    
    return port_opt_duel.objective_value, values_duel

#   此函数是先计算结果并比较
def comparison_B(file, file_ef,file_comparison):
    dt = pd.read_csv(file_ef)
    dt = dt.iloc[0:dt.size + 1, 0]
    dt = dt.str.split(' ', expand=True)
    xx = dt[2].astype(float)
    yy = dt[4].astype(float)
    k = 10

    y_arr_dual = []
    y_arr_dual_values = []
    y_arr = []
    y_arr_values = []
    y_arr_relaxed = []
    y_arr_relaxed_values = []
    relaxed_comparison = []
    dual_comparison = []

    assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(file)
    i=0
    for total_rho in xx:
        i=i+1
        try:
            objective_primal, values_primal = primal_solution_B(total_rho, k, assets_num, asset_return, asset_cov)
            y_arr.append(objective_primal)
            y_arr_values.append(values_primal)
        except:
            values_primal = np.NaN
            y_arr.append(np.NaN)
            y_arr_values.append(np.NaN)

        try:
            objective_relaxed, values_relaxed = primal_solution_relaxed_B(total_rho, k, assets_num, asset_return,
                                                                        asset_cov)
            y_arr_relaxed.append(objective_relaxed)
            y_arr_relaxed_values.append(values_relaxed)
        except:
            values_relaxed = np.NaN
            y_arr_relaxed.append(np.NaN)
            y_arr_relaxed_values.append(np.NaN)

        try:
            objective_dual, values_dual = dual_solution_B(total_rho, k, assets_num, asset_return, asset_cov_inv)
            y_arr_dual.append(objective_dual)
            y_arr_dual_values.append(values_dual)
            print("primal")
        except:
            values_dual = np.NaN
            y_arr_dual.append(np.NaN)
            y_arr_dual_values.append(np.NaN)
            print("no primal")

        try:
            print(sum(np.array(values_primal) * np.array(values_relaxed)))
            relaxed_comparison.append(sum(np.array(values_primal) * np.array(values_relaxed)))
            print("relaxed")
        except:
            relaxed_comparison.append(np.NaN)
            print("no relaxed")

        try:
            print(sum(np.array(values_primal) * np.array(values_dual)))
            dual_comparison.append(sum(np.array(values_primal) * np.array(values_dual)))
            print("dual")
        except:
            dual_comparison.append(np.NaN)
            print("no dual")
        print(i/2000)
    y_arr = pd.Series(y_arr)
    y_arr_dual = pd.Series(y_arr_dual)
    y_arr_relaxed = pd.Series(y_arr_relaxed)
    y_arr_values = pd.Series(y_arr_values)
    y_arr_dual_values = pd.Series(y_arr_dual_values)
    y_arr_relaxed_values = pd.Series(y_arr_relaxed_values)
    relaxed_comparison = pd.Series(relaxed_comparison)
    dual_comparison = pd.Series(dual_comparison)


    data = pd.concat([xx, yy, y_arr,y_arr_relaxed, y_arr_dual, y_arr_values, y_arr_relaxed_values, y_arr_dual_values,relaxed_comparison, dual_comparison], axis=1)
    data.to_csv(file_comparison)

#   GA的构成
def crossover(mother, father):
    long = len(mother)

    child = [0]*long
    rate = np.array(mother)+np.array(father)
    one_index = np.random.choice(np.arange(long), size=10, replace=False, p=rate/(sum(rate)))
    for i in one_index:
        child[i] = 1

    return child

#   GA的mutation部分
def mutation(child):
    zero_index = np.random.choice([i for i, num in enumerate(child) if num == 0])
    one_index = np.random.choice([i for i, num in enumerate(child) if num == 1])

    child[zero_index], child[one_index] = child[one_index], child[zero_index]
    return child

#   subset optimization
def y_fixed(assets_num, asset_return, asset_cov, y, total_rho):
    port_opt = Model(name='PortfolioOptimization')

    x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}

    port_opt.minimize(
        port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)))

    port_opt.add_constraint(port_opt.sum(x[i] for i in range(0, assets_num)) == 1)
    port_opt.add_constraint(port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)) == total_rho)
    for i in range(0, assets_num):
        port_opt.add_constraint(x[i] >= 0.01 * y[i])
    for i in range(0, assets_num):
        port_opt.add_constraint(x[i] <= y[i])

    port_opt.solve()

    return port_opt.objective_value

#   GA的计算fitness部分
def fitness(pop, total_rho,file):
    long = len(pop)
    pop_fitness = []
    assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(file)
    port_opt = Model(name='PortfolioOptimization')
    port_opt.context.cplex_parameters.threads = 1
    x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}

    port_opt.minimize(
        port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)))
    for y in pop:

        port_opt.clear_constraints()
        port_opt.add_constraint(port_opt.sum(x[i] for i in range(0, assets_num)) == 1)
        port_opt.add_constraint(port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)) == total_rho)
        for i in range(0, assets_num):
            port_opt.add_constraint(x[i] >= 0.01 * y[i])
        for i in range(0, assets_num):
            port_opt.add_constraint(x[i] <= y[i])
        try:
            port_opt.solve()
            pop_fitness.append(port_opt.objective_value)
        except:
            pop_fitness.append(-1)


    max_value = max(pop_fitness)
    if max_value == -1:
        sys.exit(0)
    for j in [p for p, num in enumerate(pop_fitness) if num == -1]:
        pop_fitness[j] = max_value

    min_index = np.argmin(pop_fitness)
    min_value = pop_fitness[min_index]
    min_port = pop[min_index]
    pop_fitness = [max_value - l for l in pop_fitness]
    find_largest_n_elements(pop_fitness, 1)
    if sum(pop_fitness) == 0:
        pop_fitness = [max_value] * long


    port_opt.clear()
    port_opt.end()
    return pop_fitness, min_value, min_port

#   GA的计算selection部分
def select(pop, pop_fitness):
    new_pop = []
    long = len(pop)

    selection = np.random.choice(np.arange(long), size=long, replace=True,
                                 p=np.array(pop_fitness) / (sum(np.array(pop_fitness))))
    for o in np.arange(long):
        new_pop.append(pop[selection[o]])
    return new_pop

#   GA中需要几代进行迭代
def new_generation(pop, crossover_rate, mutation_rate, min_port, min_value, total_rho,file):
    new_pop = []
    long = len(pop)
    for i in range(long):
        father = pop[i].copy()
        if np.random.rand() < crossover_rate:
            mother = pop[np.random.randint(long)]
            child = crossover(mother, father)
        else:
            child = father
        if np.random.rand() < mutation_rate:
            child = mutation(child)
        new_pop.append(child)
    new_pop[0] = min_port.copy()

    pop_fitness, value, port = fitness(new_pop, total_rho,file)
    if min_value > value:
        min_value = value
        min_port = port.copy()

    new_pop = select(new_pop, pop_fitness)
    return new_pop, min_port, min_value

#   GA的本体
def GA_B(file, file_ef,file_experiment):
    dt = pd.read_csv(file_ef)
    dt = dt.iloc[0:dt.size + 1, 0]
    dt = dt.str.split(' ', expand=True)
    xx = dt[2].astype(float)
    yy = dt[4].astype(float)
    k = 10
    x=[]
    y=[]
    y_arr_dual = []
    y_arr = []
    y_arr_relaxed = []
    ga_arr = []
    ga_port = []

    assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(file)

    for n in range(50):
        print(n)
        index = 38+40*n
        total_rho = xx[index]
        x.append(total_rho)
        y.append(yy[index])

        try:
            objective_primal, port_primal = primal_solution_B(total_rho, k, assets_num, asset_return, asset_cov)
            y_arr.append(objective_primal)
        except:
            y_arr.append(np.nan)
            y_arr_relaxed.append(np.nan)
            y_arr_dual.append(np.nan)
            ga_port.append(np.nan)
            ga_arr.append(np.nan)
            continue


        try:
            objective_relaxed, port_relaxed = primal_solution_relaxed_B(total_rho, k, assets_num, asset_return, asset_cov)
        except:
            y_arr_relaxed.append(np.nan)
            y_arr_dual.append(np.nan)
            ga_port.append(np.nan)
            ga_arr.append(np.nan)
            continue

        try:
            objective_dual, port_dual = dual_solution_B(total_rho, k, assets_num, asset_return, asset_cov_inv)
        except:
            y_arr_relaxed.append(np.nan)
            y_arr_dual.append(np.nan)
            ga_port.append(np.nan)
            ga_arr.append(np.nan)
            continue
        y_arr_relaxed.append(objective_relaxed)
        y_arr_dual.append(objective_dual)

        try:
            value_1 = y_fixed(assets_num,asset_return,asset_cov,port_relaxed,10,total_rho)
        except:
            try:
                value_2 = y_fixed(assets_num,asset_return,asset_cov,port_dual,10,total_rho)
                min_value = value_2
                min_port = port_dual
            except:
                min_value = 1
                min_port = port_relaxed
        try:
            value_2 = y_fixed(assets_num, asset_return, asset_cov, port_dual, 10, total_rho)
            if value_1 < value_2:
                min_value = value_1
                min_port = port_relaxed
            else:
                min_value = value_2
                min_port = port_dual
        except:
            min_value = value_1
            min_port = port_relaxed

        pop = [port_relaxed] * 50 + [port_dual] * 50

        for m in range(20):
            pop, min_port, min_value = new_generation(pop, 0.8, 0.3, min_port, min_value, total_rho,file)
        ga_port.append(min_port)
        ga_arr.append(min_value)
    x=pd.Series(x)
    y=pd.Series(y)
    y_arr = pd.Series(y_arr)
    y_arr_dual = pd.Series(y_arr_dual)
    y_arr_relaxed = pd.Series(y_arr_relaxed)
    ga_port = pd.Series(ga_port)
    ga_arr = pd.Series(ga_arr)
    data = pd.concat([x, y, y_arr, y_arr_dual, y_arr_relaxed, ga_arr, ga_port], axis=1)
    data.to_csv(file_experiment)

#   这个函数是为了比较两个投资组合的重合数量
def compare_port(port_1,port_2):
    return sum(np.array(port_1)*np.array(port_2))

#   接下来则是基于Chang的模型，即带tradeoff（lambda）的模型。
#   to solve relaxed problem RA
def relaxed_solution_A(k, assets_num, asset_return, asset_cov,lam):
    port_opt = Model(name='PortfolioOptimization')
    port_opt.context.cplex_parameters.threads = 1
    x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}
    y = {i: port_opt.continuous_var(name='y_{0}'.format(i)) for i in range(0, assets_num)}

    port_opt.minimize(
        lam * port_opt.sum(
            x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)) - (
                    1 - lam) * port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)))
    port_opt.add_constraint(port_opt.sum(x[i] for i in range(0, assets_num)) == 1)
    port_opt.add_constraint(port_opt.sum(y[i] for i in range(0, assets_num)) == k)
    for i in range(0, assets_num):
        port_opt.add_constraint(x[i] >= 0.01 * y[i])
    for i in range(0, assets_num):
        port_opt.add_constraint(x[i] <= y[i])
    for i in range(0, assets_num):
        port_opt.add_constraint(y[i] <= 1)
    for i in range(0, assets_num):
        port_opt.add_constraint(y[i] >= 0)

    port_opt.solve()

    values = []
    for i in range(0, assets_num):
        values.append(port_opt.solution.get_value("y_{0}".format(i)))
    values_relaxed = [0] * assets_num
    for i in find_largest_n_elements(values, 10):
        values_relaxed[i] = 1

    return port_opt.objective_value, values_relaxed


#   to solve dual problem DA
def dual_solution_A(k, assets_num, asset_return, asset_cov_inv,lam):
    if lam == 0 :
        return primal_solution_A(k, assets_num, asset_return, asset_cov_inv,lam)[0],primal_solution_A(k, assets_num, asset_return, asset_cov_inv,lam)[1]
    port_opt_duel = Model(name='PortfolioOptimizationDuel')
    sigma = {i: port_opt_duel.continuous_var(name='sigma_{0}'.format(i), lb=0) for i in range(0, assets_num)}
    miu = {i: port_opt_duel.continuous_var(name='miu_{0}'.format(i), lb=0) for i in range(0, assets_num)}
    gama = port_opt_duel.continuous_var(name='gama', lb=-float("inf"))
    beta = port_opt_duel.continuous_var(name='beta', lb=-float("inf"))
    x = {i: port_opt_duel.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}

    port_opt_duel.maximize(port_opt_duel.sum(
        0.25 * (1/lam) * asset_cov_inv[i][j] * (sigma[j] - miu[j] +(1-lam)* asset_return[j] - beta) * (
                    sigma[i] - miu[i] +(1-lam) * asset_return[i] - beta) + 0.5 * (1/lam)* (
                    -(1-lam)* asset_return[i] + beta - sigma[i] + miu[i]) * asset_cov_inv[i][j] * (
                    sigma[j] - miu[j] +(1-lam)* asset_return[j] - beta) for i in range(0, assets_num) for j in
        range(0, assets_num))+port_opt_duel.sum(gama+0.01*sigma[i]-miu[i] for i in range(0, assets_num))  - beta - k * gama)

    for i in range(0, assets_num):
        port_opt_duel.add_constraint(sigma[i] * 0.01 - miu[i] + gama <= 0)
        port_opt_duel.add_constraint(x[i] == port_opt_duel.sum(
            0.5 * asset_cov_inv[i][j] * (sigma[j] - miu[j] +(1-lam) * asset_return[j] - beta) for j in
            range(0, assets_num)))
        port_opt_duel.add_constraint(x[i] >=0)
    try:
        port_opt_duel.solve()
    except:
        print(sys.exc_info())
    values = []
    for i in range(0, assets_num):
        values.append(port_opt_duel.solution.get_value("x_{0}".format(i)))
    values_duel = [0] * assets_num
    for i in find_largest_n_elements(values, 10):
        values_duel[i] = 1

    return port_opt_duel.objective_value, values_duel

#   to solve primal problem PA
def primal_solution_A(k, assets_num, asset_return, asset_cov,lam):
    port_opt = Model(name='PortfolioOptimization')
    x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}
    y = {i: port_opt.binary_var(name='y_{0}'.format(i)) for i in range(0, assets_num)}
    port_opt.minimize(
        lam*port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num))-(1-lam)*port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)))
    port_opt.add_constraint(port_opt.sum(x[i] for i in range(0, assets_num)) == 1)
    port_opt.add_constraint(port_opt.sum(y[i] for i in range(0, assets_num)) == k)
    for i in range(0, assets_num):
        port_opt.add_constraint(x[i] >= 0.01 * y[i])
        port_opt.add_constraint(x[i] <= y[i])
    port_opt.solve()
    variance = port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)).solution_value
    mean_return = port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)).solution_value
    values_primal=[]
    for i in range(0, assets_num):
        values_primal.append(port_opt.solution.get_value("y_{0}".format(i)))

    return port_opt.objective_value, values_primal,variance,mean_return

#   模型的比较
def comparison_A(file, file_ef,comparisons):
    dt = pd.read_csv(file_ef)
    dt = dt.iloc[0:dt.size + 1, 0]
    dt = dt.str.split(' ', expand=True)
    xx = dt[2].astype(float)
    yy = dt[4].astype(float)
    k = 10

    y_arr_dual = []
    y_arr_dual_values = []
    y_arr = []
    y_arr_values = []
    y_arr_relaxed = []
    y_arr_relaxed_values = []
    relaxed_comparison = []
    dual_comparison = []

    assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(file)
    i=0
    E = 50
    #   注意此处lambda需要刨去lambda=0的情况，然后间隔为0.02，直到lambda=1得到50个点
    for e in range(50):
        lam = (e+1)/E
        print(lam)
        try:
            objective_primal, values_primal,_,_ = primal_solution_A(k, assets_num, asset_return, asset_cov,lam)
            y_arr.append(objective_primal)
            y_arr_values.append(values_primal)
            print("primal")
        except:
            values_primal = np.NaN
            y_arr.append(np.NaN)
            y_arr_values.append(np.NaN)


        try:
            objective_relaxed, values_relaxed = relaxed_solution_A(k, assets_num, asset_return,
                                                                        asset_cov,lam)
            y_arr_relaxed.append(objective_relaxed)
            y_arr_relaxed_values.append(values_relaxed)
            print("relaxed")
        except:
            values_relaxed = np.NaN
            y_arr_relaxed.append(np.NaN)
            y_arr_relaxed_values.append(np.NaN)

        try:
            objective_dual, values_dual = dual_solution_A(k, assets_num, asset_return, asset_cov_inv,lam)
            y_arr_dual.append(objective_dual)
            y_arr_dual_values.append(values_dual)
            print("dual")
        except:
            values_dual = np.NaN
            y_arr_dual.append(np.NaN)
            y_arr_dual_values.append(np.NaN)
            print("no dual")

        try:
            relaxed_comparison.append(sum(np.array(values_primal) * np.array(values_relaxed)))
        except:
            relaxed_comparison.append(np.NaN)

        try:
            dual_comparison.append(sum(np.array(values_primal) * np.array(values_dual)))
        except:
            dual_comparison.append(np.NaN)

    y_arr = pd.Series(y_arr)
    y_arr_dual = pd.Series(y_arr_dual)
    y_arr_relaxed = pd.Series(y_arr_relaxed)
    y_arr_values = pd.Series(y_arr_values)
    y_arr_dual_values = pd.Series(y_arr_dual_values)
    y_arr_relaxed_values = pd.Series(y_arr_relaxed_values)
    relaxed_comparison = pd.Series(relaxed_comparison)
    dual_comparison = pd.Series(dual_comparison)

    data = pd.concat([pd.Series(range(50)), y_arr, y_arr_dual, y_arr_relaxed, y_arr_values, y_arr_dual_values, y_arr_relaxed_values,
                       dual_comparison,relaxed_comparison], axis=1)
    data.to_csv(comparisons)

#   新的一版subset optimization函数
def y_fixed_re(assets_num, asset_return, asset_cov, y,lam):
    port_opt = Model(name='PortfolioOptimization')
    port_opt.context.cplex_parameters.threads = 1
    k = 10
    result = []
    for i in range(assets_num):
        if y[i]==1:
            result.append(i)
        else:
            continue

    asset_cov = asset_cov[result]
    asset_cov = asset_cov[:,result]
    asset_return = asset_return[result]
    x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(k)}
    port_opt.minimize(
        lam*port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(k) for j in range(k))-(1-lam)*port_opt.sum(x[i] * asset_return[i] for i in range(k)))

    port_opt.add_constraint(port_opt.sum(x[i] for i in range(k)) == 1)
    for i in range(0, k):
        port_opt.add_constraint(x[i] >= 0.01)
    for i in range(0, k):
        port_opt.add_constraint(x[i] <= 1)

    port_opt.solve()
    return port_opt.objective_value


def fitness_re(pop,assets_num, asset_return, asset_cov,lam):
    k = 10
    long = len(pop)
    pop_fitness = []
    port_opt = Model(name='PortfolioOptimization')
    port_opt.context.cplex_parameters.threads = 1
    x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(k)}
    for y in pop:
        result = []
        for i in range(assets_num):
            if y[i] == 1:
                result.append(i)
            else:
                continue
        asset_cov_1 = asset_cov[result]
        asset_cov_1 = asset_cov_1[:, result]
        asset_return_1 = asset_return[result]
        port_opt.minimize(
            lam * port_opt.sum(x[i] * x[j] * asset_cov_1[i][j] for i in range(k) for j in range(k)) - (
                        1 - lam) * port_opt.sum(x[i] * asset_return_1[i] for i in range(k)))

        port_opt.add_constraint(port_opt.sum(x[i] for i in range(k)) == 1)
        for i in range(0, k):
            port_opt.add_constraint(x[i] >= 0.01)
        for i in range(0, k):
            port_opt.add_constraint(x[i] <= 1)
        try:
            port_opt.solve()
            pop_fitness.append(port_opt.objective_value)
        except:
            pop_fitness.append(-1)
        port_opt.clear_constraints()
        port_opt.clear_multi_objective()
    max_value = max(pop_fitness)
    if max_value == -1:
        sys.exit(0)
    for j in [p for p, num in enumerate(pop_fitness) if num == -1]:
        pop_fitness[j] = max_value
    min_index = np.argmin(pop_fitness)
    min_value = pop_fitness[min_index]
    min_port = pop[min_index]
    pop_fitness = [max_value - l for l in pop_fitness]
    find_largest_n_elements(pop_fitness, 1)
    if sum(pop_fitness) == 0:
        pop_fitness = [max_value] * long
    return pop_fitness, min_value, min_port

#   与select函数一致
def select_re(pop, pop_fitness):
    new_pop = []
    long = len(pop)

    selection = np.random.choice(np.arange(long), size=long, replace=True,
                                 p=np.array(pop_fitness) / (sum(np.array(pop_fitness))))
    for o in np.arange(long):
        new_pop.append(pop[selection[o]])
    return new_pop

#   与new_generation函数一致
def new_generation_re(pop, crossover_rate, mutation_rate, min_port, min_value,assets_num, asset_return, asset_cov,lam):
    # print(min_port)
    new_pop = []
    long = len(pop)
    for i in range(long):
        father = pop[i].copy()
        if np.random.rand() < crossover_rate:
            mother = pop[np.random.randint(long)]
            child = crossover(mother, father)
        else:
            child = father
        if np.random.rand() < mutation_rate:
            child = mutation(child)
        new_pop.append(child)

    new_pop[0] = min_port.copy()

    pop_fitness, value, port = fitness_re(new_pop,assets_num, asset_return, asset_cov,lam)
    if min_value > value:
        min_value = value
        min_port = port.copy()

    new_pop = select_re(new_pop, pop_fitness)
    return new_pop, min_port, min_value

#   寻找目标投资组合的k=1的邻域
def k_search_1(port):
    pool = []
    port_re = port.copy()
    for i in range(10):
        m = port.index(1)
        port[m] = 2
        for j in range(len(port)-10):
            n = port.index(-1*i)
            port[n]=-1*(i+1)
            pool1 = port_re.copy()
            pool1[m]=0
            pool1[n]=1
            pool.append(pool1.copy())
    for i in range(len(port)):
        port[i] = port_re[i]
    return pool

#   目标投资组合的k=1的邻域内是否有更好的组合，如果有则作为新的目标投资组合再进行新的查找
def ga_vns(assets_num, asset_return, asset_cov,y,lam):
    min_port = y.copy()
    min_value = y_fixed_re(assets_num, asset_return, asset_cov, min_port, lam)
    for p in range(4):
        index = 0
        port_pool = k_search_1(min_port)
        for l in port_pool:
            try:
                min_value_search = y_fixed_re(assets_num, asset_return, asset_cov, l, lam)
                if min_value_search < min_value:
                    min_port = l.copy()
                    min_value = min_value_search
                    index = 1
            except:
                continue
        if index == 0:
            break
    return min_port,min_value

#   这里将GA和vns合并在一起了
def GA_vns_A(file, file_ef,file_experiment_re):
    dt = pd.read_csv(file_ef)
    dt = dt.iloc[0:dt.size + 1, 0]
    dt = dt.str.split(' ', expand=True)
    xx = dt[2].astype(float)
    yy = dt[4].astype(float)
    k = 10
    x=[]
    y=[]
    y_arr_dual = []
    y_arr = []
    y_arr_relaxed = []
    ga_arr = []
    ga_port = []
    ga_vns_port = []
    ga_vns_value = []

    assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(file)
    E = 51
    for e in range(1,51):
        lam = e/(E-1)
        print(lam)
        x.append(lam)
        y.append(primal_solution_A(k,assets_num,asset_return,asset_cov,lam)[0])


        objective_primal, port_primal,variance,mean = primal_solution_A(k, assets_num, asset_return, asset_cov,lam)
        y_arr.append(objective_primal)



        objective_relaxed, port_relaxed = relaxed_solution_A(k, assets_num, asset_return, asset_cov,lam)

        objective_dual, port_dual = dual_solution_A(k, assets_num, asset_return, asset_cov_inv,lam)

        y_arr_relaxed.append(objective_relaxed)
        y_arr_dual.append(objective_dual)


        value_1 = y_fixed_re(assets_num,asset_return,asset_cov,port_relaxed,lam)
        value_2 = y_fixed_re(assets_num,asset_return,asset_cov,port_dual,lam)


        if value_1 < value_2:
            min_value = value_1
            min_port = port_relaxed
        else:
            min_value = value_2
            min_port = port_dual


        pop = [port_relaxed] * 50 + [port_dual] * 50

        for m in range(20):
            pop, min_port, min_value = new_generation_re(pop, 0.8, 0.3, min_port, min_value, assets_num, asset_return, asset_cov,lam)
        ga_port.append(min_port.copy())
        ga_arr.append(min_value)

        #   后面的ga_vns函数就是vns的部分
        vns_port, vns_value = ga_vns(assets_num, asset_return, asset_cov,min_port,lam)
        ga_vns_port.append(vns_port.copy())
        ga_vns_value.append(vns_value)
    x=pd.Series(x)
    y=pd.Series(y)
    y_arr = pd.Series(y_arr)
    y_arr_dual = pd.Series(y_arr_dual)
    y_arr_relaxed = pd.Series(y_arr_relaxed)
    ga_port = pd.Series(ga_port)
    ga_arr = pd.Series(ga_arr)
    ga_vns_port = pd.Series(ga_vns_port)
    ga_vns_value = pd.Series(ga_vns_value)
    data = pd.concat([x, y, y_arr, y_arr_dual, y_arr_relaxed, ga_arr, ga_port,ga_vns_value,ga_vns_port], axis=1)
    data.to_csv(file_experiment_re)



#   后面是实际使用以及计算MEAPE等measures时用的代码
ls_files_experiments_re  = [ "D:\WorkingFiles\Journal\cplex\experiment1_re.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_re.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_re.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_re.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_re.csv"
            ]

ls_files_experiments_re1  = [ "D:\WorkingFiles\Journal\cplex\experiment1_re1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_re1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_re1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_re1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_re1.csv"
            ]


#   找k=2的邻域
def k_search_2(port):
    pool = k_search_1(port)
    indices_1 = []
    indices_0 = []
    for i,num in enumerate(port):
        if num == 1:
            indices_1.append(i)
        if num == 0:
            indices_0.append(i)
    for a in range(1,10):
        for b in range(a+1,11):
            m = indices_1[a-1]
            n = indices_1[b-1]
            for c in range(1,len(port) - 10):
                for d in range(c+1,len(port) - 9):
                    p = indices_0[c-1]
                    q = indices_0[d-1]
                    pool1=port.copy()
                    pool1[m] = 0
                    pool1[n] = 0
                    pool1[p] = 1
                    pool1[q] = 1
                    pool.append(pool1.copy())
    return pool

#   找k=3的邻域
def k_search_3(port):
    pool = k_search_2(port)
    indices_1 = []
    indices_0 = []
    for i,num in enumerate(port):
        if num == 1:
            indices_1.append(i)
        if num == 0:
            indices_0.append(i)
    for a in range(1,9):
        for b in range(a+1,10):
            for c in range(b+1,11):
                m = indices_1[a-1]
                n = indices_1[b-1]
                l = indices_1[c-1]
                for d in range(1,len(port) - 11):
                    for e in range(d+1,len(port) - 10):
                        for f in range(e + 1, len(port) - 9):
                            p = indices_0[d-1]
                            q = indices_0[e-1]
                            w = indices_0[f-1]
                            pool1=port.copy()
                            pool1[m] = 0
                            pool1[n] = 0
                            pool1[l] = 0
                            pool1[p] = 1
                            pool1[q] = 1
                            pool1[w] = 1
                            pool.append(pool1.copy())
    return pool

#   找k=4的邻域
def k_search_4(port):
    pool = k_search_3(port)
    indices_1 = []
    indices_0 = []
    for i,num in enumerate(port):
        if num == 1:
            indices_1.append(i)
        if num == 0:
            indices_0.append(i)
    for a in range(1,8):
        for b in range(a+1,9):
            for c in range(b+1,10):
                for d in range(c + 1, 11):
                    m = indices_1[a-1]
                    n = indices_1[b-1]
                    l = indices_1[c-1]
                    o = indices_1[d-1]
                    for e in range(1,len(port) - 12):
                        for f in range(e+1,len(port) - 11):
                            for g in range(f + 1, len(port) - 10):
                                for h in range(g + 1, len(port) - 9):
                                    p = indices_0[e-1]
                                    q = indices_0[f-1]
                                    w = indices_0[g-1]
                                    s = indices_0[h-1]
                                    pool1=port.copy()
                                    pool1[m] = 0
                                    pool1[n] = 0
                                    pool1[l] = 0
                                    pool1[o] = 0
                                    pool1[p] = 1
                                    pool1[q] = 1
                                    pool1[w] = 1
                                    pool1[s] = 1
                                    pool.append(pool1.copy())
    return pool

#   下面的函数是用于比较启发式算法和直接k=4邻域内搜索的时间效率
def vns_GA_time_compare(file,file_ef):
    dt = pd.read_csv(file_ef)
    dt = dt.iloc[0:dt.size + 1, 0]
    dt = dt.str.split(' ', expand=True)
    k = 10
    timer1=[]
    timer2=[]
    timer3=[]
    assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(file)
    E = 51
    for e in range(1, 51):
        lam = e / (E - 1)
        time_start = time.time()
        objective_dual, port_dual = dual_solution_A(k, assets_num, asset_return, asset_cov_inv, lam)
        time_end = time.time()
        timer1.append(time_end-time_start)
        time_start = time.time()
        best_value = y_fixed_re(assets_num, asset_return, asset_cov,port_dual,lam)
        best_port = port_dual.copy()
        for y in k_search_4(port_dual):
            value = y_fixed_re(assets_num, asset_return, asset_cov,y,lam)
            if value<best_value:
                best_value=value
                best_port=y
            else:
                continue
        time_end = time.time()
        timer2.append(time_end-time_start)
        time_start = time.time()
        objective_relaxed, port_relaxed = relaxed_solution_A(k, assets_num, asset_return, asset_cov,lam)
        value_1 = y_fixed_re(assets_num, asset_return, asset_cov, port_relaxed, lam)
        value_2 = y_fixed_re(assets_num, asset_return, asset_cov, port_dual, lam)
        if value_1 < value_2:
            min_value = value_1
            min_port = port_relaxed
        else:
            min_value = value_2
            min_port = port_dual
        pop = [port_relaxed] * 50 + [port_dual] * 50
        for m in range(20):
            pop, min_port, min_value = new_generation_re(pop, 0.8, 0.3, min_port, min_value, assets_num, asset_return,
                                                         asset_cov, lam)
        time_end = time.time()
        timer3.append(time_end - time_start)
        print("dual耗时 ",timer1[e-1],"vns耗时 ",timer2[e-1],"vnsga耗时 ",timer3[e-1])

#   ls_files，ls_files_ef文件就是从OR library下载下来的文件
ls_files = [ "D:\WorkingFiles\Journal\cplex\port1.txt"
            ,"D:\WorkingFiles\Journal\cplex\port2.txt"
            ,"D:\WorkingFiles\Journal\cplex\port3.txt"
            ,"D:\WorkingFiles\Journal\cplex\port4.txt"
            ,"D:\WorkingFiles\Journal\cplex\port5.txt"
            ]
ls_files_ef = [ "D:\WorkingFiles\Journal\cplex\portef1.txt"
            ,"D:\WorkingFiles\Journal\cplex\portef2.txt"
            ,"D:\WorkingFiles\Journal\cplex\portef3.txt"
            ,"D:\WorkingFiles\Journal\cplex\portef4.txt"
            ,"D:\WorkingFiles\Journal\cplex\portef5.txt"
            ]
ls_files_experiments_re_final_51 = [ "D:\WorkingFiles\Journal\cplex\experiment1_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_final51.csv"
            ]

#   计算GA_VNS_A的MPE以及绘制图，得到的结果复制黏贴到ls_files_experiments_re_final_51原文件里面
for i in range(5):
    print("第{0}个数据集：".format(i+1))
    data1 = pd.read_csv(ls_files_ef[i],encoding='gb2312')
    data2 = pd.read_csv(ls_files_experiments_re_final_51[i],encoding='gb2312')

    data1 = data1.iloc[0:data1.size + 1, 0]
    data1 = data1.str.split(' ', expand=True)
    xx = data1[2].astype(float)
    yy = data1[4].astype(float)
    ga = data2.iloc[0:data2.size,11]
    x = data2.iloc[0:data2.size,10]
    MPE = []
    for j in range(50):
        if np.isnan(ga[j]):
            continue
        x_direct = x[j]
        y_direct = ga[j]
        for n in range(1998):
            if xx[n]>=x_direct and xx[n+1]<=x_direct:
                y_coordinate = yy[n]+(x_direct-xx[n])*(yy[n]-yy[n+1])/(xx[n]-xx[n+1])
                y_MPE = 100*abs(math.sqrt(y_direct)-math.sqrt(y_coordinate))/math.sqrt(y_coordinate)
                y_MPE_1 = 100 * abs(y_direct - y_coordinate) / y_coordinate
                break
            else:
                continue
        for n in range(1998):
            if yy[n]>=y_direct and yy[n+1]<=y_direct:
                x_coordinate = xx[n]+(y_direct-yy[n])*(xx[n]-xx[n+1])/(yy[n]-yy[n+1])
                x_MPE = 100*abs(x_direct-x_coordinate)/x_coordinate
                break
            else:
                continue
        MPE.append(min(x_MPE,y_MPE))
        print(x_MPE,y_MPE,y_MPE_1,min(x_MPE,y_MPE))
    print("mean:",sum(MPE)/len(MPE),",median", np.median(MPE),",min:",min(MPE),",max:",max(MPE))
    plt.xlabel('Variance')
    plt.ylabel('Return')
    plt.plot(yy,xx,label="Unconstrained Effient Frontier")
    plt.scatter(ga,x,marker="x", label="GA_VNS_A",c="red")
    plt.show()

ls_files_experiments = [ "D:\WorkingFiles\Journal\cplex\experiment1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5.csv"
            ]

#   绘制GA_B图，得到的结果复制黏贴到ls_files_experiments_re_final_51原文件里面
for file in ls_files_experiments:
    dt = pd.read_csv(file,encoding='gb2312')
    ga=dt.iloc[0:dt.size+1,6]
    port = dt.iloc[0:dt.size+1,7]
    x=dt.iloc[0:dt.size+1,1]
    dual=dt.iloc[0:dt.size+1,5]
    relaxed=dt.iloc[0:dt.size+1,4]
    primal=dt.iloc[0:dt.size+1,3]
    unconstrained = dt.iloc[0:dt.size+1,2]
    plt.xlabel('Variance')
    plt.ylabel('Return')
    plt.plot(unconstrained,x,label="Unconstrained Effient Frontier")
    plt.plot(ga,x,".",label = "GA_B")
    plt.legend(loc='best',fontsize=12)
    plt.show()




ls_files_experiments_vns = [ "D:\WorkingFiles\Journal\cplex\experiment1_vns.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_vns.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_vns.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_vns.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_vns.csv"
            ]
ls_files_experiments_re_final = [ "D:\WorkingFiles\Journal\cplex\experiment1_final.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_final.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_final.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_final.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_final.csv"
            ]
ls_files_experiments_re_final_51 = [ "D:\WorkingFiles\Journal\cplex\experiment1_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_final51.csv"
            ]

#
# file = "D:\WorkingFiles\Journal\cplex\port1.txt"
# file_ef = "D:\WorkingFiles\Journal\cplex\portef1.txt"
# GA(file,file_ef)
#
ls_files = [ "D:\WorkingFiles\Journal\cplex\port1.txt"
            ,"D:\WorkingFiles\Journal\cplex\port2.txt"
            ,"D:\WorkingFiles\Journal\cplex\port3.txt"
            ,"D:\WorkingFiles\Journal\cplex\port4.txt"
            ,"D:\WorkingFiles\Journal\cplex\port5.txt"
            ]
ls_files_ef = [ "D:\WorkingFiles\Journal\cplex\portef1.txt"
            ,"D:\WorkingFiles\Journal\cplex\portef2.txt"
            ,"D:\WorkingFiles\Journal\cplex\portef3.txt"
            ,"D:\WorkingFiles\Journal\cplex\portef4.txt"
            ,"D:\WorkingFiles\Journal\cplex\portef5.txt"
            ]
ls_files_experiments = [ "D:\WorkingFiles\Journal\cplex\experiment1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5.csv"
            ]
# for i in [2,3]:
#     file = ls_files[i]
#     file_ef = ls_files_ef[i]
#     experiment = ls_files_experiments[i]
#     GA(file,file_ef,experiment)




#
ls_files_experiments = [ "D:\WorkingFiles\Journal\cplex\experiment1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5.csv"
            ]


# for i in [3]:
#     assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(ls_files[i])
#     dt = pd.read_csv(ls_files_experiments[i])
#     ga=dt.iloc[0:dt.size+1,6]
#     port = dt.iloc[0:dt.size+1,7]
#     x=dt.iloc[0:dt.size+1,1]
#     print("第{0}个数据集".format(i+1))
#     jieguo = []
#     for i in range(50):
#         if np.isnan(ga[i]):
#             continue
#         port_ga = eval(port[i])
#         total_rho = x[i]
#         objective,port_primal = primal_solution(total_rho,10,assets_num,asset_return,asset_cov)
#         jieguo.append(int(compare_port(port_ga,port_primal)))
#     print(jieguo)

# assets_num, asset_return, asset_cov, asset_cov_inv = get_inf("D:\WorkingFiles\Journal\cplex\port1.txt")
# # start_time1 = time.time()
# port_opt = Model(name='PortfolioOptimization')
# port_opt.context.cplex_parameters.threads = 1
# x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}
# y=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
# port_opt.minimize(
#     port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)))
# port_opt.clear_constraints()
# port_opt.add_constraint(port_opt.sum(x[i] for i in range(0, assets_num)) == 1)
# port_opt.add_constraint(port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)) == 0.0103274424)
# for i in range(0, assets_num):
#     port_opt.add_constraint(x[i] >= 0.01 * y[i])
# for i in range(0, assets_num):
#     port_opt.add_constraint(x[i] <= y[i])
# port_opt.solve()
# print(port_opt.objective_value)


# data2 = pd.read_csv("D:\WorkingFiles\Journal\cplex\comparison4.csv")
#
#
# data1 = pd.read_csv("D:\WorkingFiles\Journal\cplex\portef4.txt")
#
# data1 = data1.iloc[0:data1.size + 1, 0]
# data1 = data1.str.split(' ', expand=True)
# xx = data1[2].astype(float)
# yy = data1[4].astype(float)
# ga = data2.iloc[0:data2.size + 1, 3]
# x=data2.iloc[0:data2.size + 1, 1]

# yy=np.sqrt(yy)
# ga=np.sqrt(ga)
# x=np.sqrt(x)

# for j in range(1999):
#     if np.isnan(ga[j]):
#         continue
#     x_direct = x[j]
#     y_direct = ga[j]
#     for n in range(1998):
#         if xx[n]>=x_direct and xx[n+1]<=x_direct:
#             y_coordinate = yy[n]+(x_direct-xx[n])*(yy[n]-yy[n+1])/(xx[n]-xx[n+1])
#             y_MPE = 100*abs(math.sqrt(y_direct)-math.sqrt(y_coordinate))/math.sqrt(y_coordinate)
#             y_MPE_1 = 100 * abs(y_direct - y_coordinate) / y_coordinate
#             break
#         else:
#             continue
#     for n in range(1998):
#         if yy[n]>=y_direct and yy[n+1]<=y_direct:
#             x_coordinate = xx[n]+(y_direct-yy[n])*(xx[n]-xx[n+1])/(yy[n]-yy[n+1])
#             x_MPE = 100*abs(x_direct-x_coordinate)/x_coordinate
#             break
#         else:
#             continue
#     print(x_MPE,y_MPE,y_MPE_1,min(x_MPE,y_MPE))


# for i in range(5):
#     print("第{0}个数据集：".format(i+1))
#     data1 = pd.read_csv(ls_files_ef[i],encoding='gb2312')
#     data2 = pd.read_csv(ls_files_experiments[i],encoding='gb2312')
#
#     data1 = data1.iloc[0:data1.size + 1, 0]
#     data1 = data1.str.split(' ', expand=True)
#     xx = data1[2].astype(float)
#     yy = data1[4].astype(float)
#
#     ga = data2.iloc[0: 60, 6]
#     x=data2.iloc[0:60,1]
#
#     # yy=np.sqrt(yy)
#     # ga=np.sqrt(ga)
#     # x=np.sqrt(x)
#
#     for j in range(50):
#         if np.isnan(ga[j]):
#             continue
#         x_direct = x[j]
#         y_direct = ga[j]
#         for n in range(1998):
#             if xx[n]>=x_direct and xx[n+1]<=x_direct:
#                 y_coordinate = yy[n]+(x_direct-xx[n])*(yy[n]-yy[n+1])/(xx[n]-xx[n+1])
#                 y_MPE = 100*abs(math.sqrt(y_direct)-math.sqrt(y_coordinate))/math.sqrt(y_coordinate)
#                 y_MPE_1 = 100 * abs(y_direct - y_coordinate) / y_coordinate
#                 break
#             else:
#                 continue
#         for n in range(1998):
#             if yy[n]>=y_direct and yy[n+1]<=y_direct:
#                 x_coordinate = xx[n]+(y_direct-yy[n])*(xx[n]-xx[n+1])/(yy[n]-yy[n+1])
#                 x_MPE = 100*abs(x_direct-x_coordinate)/x_coordinate
#                 break
#             else:
#                 continue
#         print(x_MPE,y_MPE,y_MPE_1,min(x_MPE,y_MPE))

# for i in range(5):
#     assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(ls_files[i])
#     print("第{0}个数据集：".format(i+1))
#     data1 = pd.read_csv(ls_files_ef[i],encoding='gb2312')
#     data1 = data1.iloc[0:data1.size + 1, 0]
#     data1 = data1.str.split(' ', expand=True)
#     xx = data1[2].astype(float)
#     yy = data1[4].astype(float)
#
#     # xx=np.sqrt(xx)
#     # yy=np.sqrt(yy)
#     # ga=np.sqrt(ga)
#     # x=np.sqrt(x)
#
#     for j in range(data1.size):
#         x_direct = xx[j]
#         try:
#             value, port = primal_solution(x_direct,10,assets_num, asset_return, asset_cov)
#             y_direct=value
#         except:
#             y_direct = np.nan
#         if np.isnan(y_direct):
#             continue
#         y_coordinate = yy[j]
#         # print(x_direct,y_direct,y_coordinate)
#         y_MPE = 100*abs(y_direct-y_coordinate)/y_coordinate
#
#         for n in range(data1.size):
#             if yy[n]>=y_direct and yy[n+1]<=y_direct:
#                 x_coordinate = xx[n]+(y_direct-yy[n])*(xx[n]-xx[n+1])/(yy[n]-yy[n+1])
#                 x_MPE = 100*abs(x_direct-x_coordinate)/x_coordinate
#                 break
#             else:
#                 continue
#         print(x_MPE,  y_MPE)

# ls_comparison=[ "D:\WorkingFiles\Journal\cplex\comparison1.csv"
#             ,"D:\WorkingFiles\Journal\cplex\comparison2.csv"
#             ,"D:\WorkingFiles\Journal\cplex\comparison3.csv"
#             ,"D:\WorkingFiles\Journal\cplex\comparison4.csv"
#             ,"D:\WorkingFiles\Journal\cplex\comparison5.csv"
#             ]
# for i in [1,2,3,4]:
#     print("第{0}个数据集：".format(i+1))
#     data3 = pd.read_csv(ls_comparison[i],encoding='gb2312')
#     xx = data3.iloc[0:data3.size + 1, 1]
#     yy = data3.iloc[0:data3.size + 1, 2]
#     y = data3.iloc[0:data3.size + 1, 3]
#     for j in range(xx.size):
#         x_direct = xx[j]
#         y_direct = y[j]
#         if np.isnan(y_direct):
#             continue
#         y_coordinate = yy[j]
#         # print(x_direct,y_direct,y_coordinate)
#         y_MPE = 100*abs(y_direct-y_coordinate)/y_coordinate
#
#         for n in range(xx.size):
#             if yy[n]>=y_direct and yy[n+1]<=y_direct:
#                 x_coordinate = xx[n]+(y_direct-yy[n])*(xx[n]-xx[n+1])/(yy[n]-yy[n+1])
#                 x_MPE = 100*abs(x_direct-x_coordinate)/x_coordinate
#                 break
#             else:
#                 continue
#         print(x_MPE,  y_MPE)

# for i in range(5):
#     if i ==1:
#         continue
#     print("第{0}个数据集：".format(i+1))
#     data3 = pd.read_csv(ls_comparison[i], encoding='gb2312')
#     values = data3.iloc[0:data3.size + 1, 6]
#     sum_value=0*np.array(eval(values[1998]))
#     for j in range(values.size):
#         if pd.isna(values[j]):
#             continue
#         value = np.array(eval(values[j]))
#         sum_value = sum_value+value
#     print(list(map(int,list(sum_value))))



ls_files_experiments_re  = [ "D:\WorkingFiles\Journal\cplex\experiment1_re.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_re.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_re.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_re.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_re.csv"
            ]

ls_files_experiments_re1  = [ "D:\WorkingFiles\Journal\cplex\experiment1_re1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_re1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_re1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_re1.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_re1.csv"
            ]


ls_files_experiments_vns = [ "D:\WorkingFiles\Journal\cplex\experiment1_vns.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_vns.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_vns.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_vns.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_vns.csv"
            ]

ls_files_experiments_re_final_51 = [ "D:\WorkingFiles\Journal\cplex\experiment1_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment2_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment3_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment4_final51.csv"
            ,"D:\WorkingFiles\Journal\cplex\experiment5_final51.csv"
            ]
# for i in [2]:
#     file = ls_files[i]
#     file_ef = ls_files_ef[i]
#     experiment = ls_files_experiments_re_final[i]
#     vns(file,file_ef,experiment)



# for i in range(5):
#     file = ls_files[i]
#     file_ef = ls_files_ef[i]
#     experiment = ls_files_experiments_re_final_51[i]
#     GA_vns_re(file,file_ef,experiment)

# assets_num, asset_return, asset_cov, asset_cov_inv = get_inf("D:\WorkingFiles\Journal\cplex\port1.txt")
# k=10
# E = 50
# for e in range(50):
#     lam = (E-e)/(E-1)
#     variance, total_return = primal_solution_re(k,assets_num, asset_return, asset_cov,lam)
# for i in [3,4]:
#     print("第{0}个数据集：".format(i+1))
#     data1 = pd.read_csv(ls_files_ef[i],encoding='gb2312')
#     data2 = pd.read_csv(ls_files_experiments[i],encoding='gb2312')
#
#     data1 = data1.iloc[0:data1.size + 1, 0]
#     data1 = data1.str.split(' ', expand=True)
#     xx = data1[2].astype(float)
#     yy = data1[4].astype(float)
#     ga = []
#     x = []
#     assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(ls_files[i])
#     k = 10
#     E = 50
#     for e in range(50):
#         print(e)
#         lam = e/(E - 1)
#         _,_,variance, total_return = primal_solution_re(k, assets_num, asset_return, asset_cov, lam)
#         ga.append(variance)
#         x.append(total_return)
#     # yy=np.sqrt(yy)
#     # ga=np.sqrt(ga)
#     # x=np.sqrt(x)
#     MPE = []
#     for j in range(50):
#         if np.isnan(ga[j]):
#             continue
#         x_direct = x[j]
#         y_direct = ga[j]
#         for n in range(1998):
#             if xx[n]>=x_direct and xx[n+1]<=x_direct:
#                 y_coordinate = yy[n]+(x_direct-xx[n])*(yy[n]-yy[n+1])/(xx[n]-xx[n+1])
#                 y_MPE = 100*abs(math.sqrt(y_direct)-math.sqrt(y_coordinate))/math.sqrt(y_coordinate)
#                 y_MPE_1 = 100 * abs(y_direct - y_coordinate) / y_coordinate
#                 break
#             else:
#                 continue
#         for n in range(1998):
#             if yy[n]>=y_direct and yy[n+1]<=y_direct:
#                 x_coordinate = xx[n]+(y_direct-yy[n])*(xx[n]-xx[n+1])/(yy[n]-yy[n+1])
#                 x_MPE = 100*abs(x_direct-x_coordinate)/x_coordinate
#                 break
#             else:
#                 continue
#         MPE.append(min(x_MPE,y_MPE))
#         print(x_direct,y_direct,x_MPE,y_MPE,y_MPE_1,min(x_MPE,y_MPE))
#     print("min:",min(MPE),",max:",max(MPE),",mean:",sum(MPE)/len(MPE),",median", np.median(MPE))
#     plt.plot(yy,xx,label="Unconstrained Effient Frontier")
#     plt.scatter(ga,x,marker="x", label="Optimal",c="red")
#     plt.show()

# for a in range(5):
#     comparison_re(ls_files[a],ls_files_ef[a],ls_comparison_re[a])
#
# for i in [2,3,4]:
#     print("第{0}个数据集".format(i+1))
#     assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(ls_files[i])
#     dt = pd.read_csv(ls_comparison_lambda_final[i])
#     dt1 = pd.read_csv(ls_comparison_lambda_final[i])
#     port_ga = dt.iloc[0:dt.size,5]
#     port_optimal = dt1.iloc[0:dt1.size,5]
#     lam = 0
#     for y in port_ga:
#         y = eval(y)
#         lam = lam+1/50
#         port_opt = Model(name='PortfolioOptimization')
#         port_opt.context.cplex_parameters.threads = 1
#
#         x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}
#         port_opt.minimize(
#             lam * port_opt.sum(
#                 x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)) - (
#                         1 - lam) * port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)))
#
#         port_opt.add_constraint(port_opt.sum(x[i] for i in range(0, assets_num)) == 1)
#         for i in range(0, assets_num):
#             port_opt.add_constraint(x[i] >= 0.01 * y[i])
#         for i in range(0, assets_num):
#             port_opt.add_constraint(x[i] <= y[i])
#         port_opt.solve()
#         mean_return = port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)).solution_value
#         variance = port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)).solution_value
#
#         print(mean_return,variance,compare_port(y,eval(port_optimal[int(lam*49)])))

# for i in range(5):
#     print("第{0}个数据集".format(i+1))
#     assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(ls_files[i])
#     dt = pd.read_csv(ls_files_experiments_re_final_51[i])
#     dt1 = pd.read_csv(ls_comparison_lambda_final[i])
#     port_ga = dt.iloc[0:dt.size,9]
#     port_optimal = dt1.iloc[0:dt1.size,5]
#     lam = 0
#     for y in port_ga:
#         y = eval(y)
#         lam = lam+1/50
#         port_opt = Model(name='PortfolioOptimization')
#         port_opt.context.cplex_parameters.threads = 1
#
#         x = {i: port_opt.continuous_var(name='x_{0}'.format(i)) for i in range(0, assets_num)}
#         port_opt.minimize(
#             lam * port_opt.sum(
#                 x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)) - (
#                         1 - lam) * port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)))
#
#         port_opt.add_constraint(port_opt.sum(x[i] for i in range(0, assets_num)) == 1)
#         for i in range(0, assets_num):
#             port_opt.add_constraint(x[i] >= 0.01 * y[i])
#         for i in range(0, assets_num):
#             port_opt.add_constraint(x[i] <= y[i])
#         port_opt.solve()
#         mean_return = port_opt.sum(x[i] * asset_return[i] for i in range(0, assets_num)).solution_value
#         variance = port_opt.sum(x[i] * x[j] * asset_cov[i][j] for i in range(0, assets_num) for j in range(0, assets_num)).solution_value
#
#         print(mean_return,variance,compare_port(y,eval(port_optimal[int(lam*49)])))


# for i in range(5):
#     print("第{0}个数据集：".format(i+1))
#     data1 = pd.read_csv(ls_files_ef[i],encoding='gb2312')
#     data2 = pd.read_csv(ls_comparison_lambda_final[i],encoding='gb2312')
#
#     data1 = data1.iloc[0:data1.size + 1, 0]
#     data1 = data1.str.split(' ', expand=True)
#     xx = data1[2].astype(float)
#     yy = data1[4].astype(float)
#     ga = data2.iloc[0:data2.size,27]
#     x = data2.iloc[0:data2.size,26]
#     MPE = []
#     for j in range(50):
#         if np.isnan(ga[j]):
#             continue
#         x_direct = x[j]
#         y_direct = ga[j]
#         for n in range(1998):
#             if xx[n]>=x_direct and xx[n+1]<=x_direct:
#                 y_coordinate = yy[n]+(x_direct-xx[n])*(yy[n]-yy[n+1])/(xx[n]-xx[n+1])
#                 y_MPE = 100*abs(math.sqrt(y_direct)-math.sqrt(y_coordinate))/math.sqrt(y_coordinate)
#                 y_MPE_1 = 100 * abs(y_direct - y_coordinate) / y_coordinate
#                 break
#             else:
#                 continue
#         for n in range(1998):
#             if yy[n]>=y_direct and yy[n+1]<=y_direct:
#                 x_coordinate = xx[n]+(y_direct-yy[n])*(xx[n]-xx[n+1])/(yy[n]-yy[n+1])
#                 x_MPE = 100*abs(x_direct-x_coordinate)/x_coordinate
#                 break
#             else:
#                 continue
#         MPE.append(min(x_MPE,y_MPE))
#         print(x_MPE,y_MPE,y_MPE_1,min(x_MPE,y_MPE))
#     print("mean:",sum(MPE)/len(MPE),",median", np.median(MPE),",min:",min(MPE),",max:",max(MPE))




#
# for i in range(5):
#     print("第{0}个数据集：".format(i+1))
#     data1 = pd.read_csv(ls_files_ef[i],encoding='gb2312')
#     data2 = pd.read_csv(ls_files_experiments_re_final_51[i],encoding='gb2312')
#
#     data1 = data1.iloc[0:data1.size + 1, 0]
#     data1 = data1.str.split(' ', expand=True)
#     xx = data1[2].astype(float)
#     yy = data1[4].astype(float)
#     ga = data2.iloc[0:data2.size,11]
#     x = data2.iloc[0:data2.size,10]
#     MEUCD = []
#     VRE = []
#     MRE = []
#     for j in range(50):
#         if np.isnan(ga[j]):
#             continue
#         x_direct = x[j]
#         y_direct = ga[j]
#         MEUCD_temp = math.inf
#         VRE_temp = np.nan
#         MRE_temp = np.nan
#         for n in range(1999):
#             temp = math.sqrt((xx[n]-x_direct)*(xx[n]-x_direct)+(yy[n]-y_direct)*(yy[n]-y_direct))
#             if temp<MEUCD_temp:
#                 MEUCD_temp = temp
#                 VRE_temp = 100*abs((yy[n]-y_direct)/y_direct)
#                 MRE_temp = 100*abs((xx[n]-x_direct)/x_direct)
#             else:
#                 continue
#         MEUCD.append(MEUCD_temp)
#         VRE.append(VRE_temp)
#         MRE.append(MRE_temp)
#         print(MEUCD_temp,VRE_temp,MRE_temp)
#     print("MEUCD:",sum(MEUCD)/len(MEUCD),"VRE:", sum(VRE)/len(VRE),"MRE:",sum(MRE)/len(MRE))

# for i in range(5):
#
#
# print("第{0}个数据集：".format(i+1))
# data1 = pd.read_csv(ls_files_ef[i],encoding='gb2312')
# data2 = pd.read_csv(ls_comparison_lambda_final[i],encoding='gb2312')
#
# data1 = data1.iloc[0:data1.size + 1, 0]
# data1 = data1.str.split(' ', expand=True)
# xx = data1[2].astype(float)
# yy = data1[4].astype(float)
# ga = data2.iloc[0:data2.size,27]
# x = data2.iloc[0:data2.size,26]
# MEUCD = []
# VRE = []
# MRE = []
# for j in range(50):
#     if np.isnan(ga[j]):
#         continue
#     x_direct = x[j]
#     y_direct = ga[j]
#     MEUCD_temp = math.inf
#     VRE_temp = np.nan
#     MRE_temp = np.nan
#     for n in range(1999):
#         temp = math.sqrt((xx[n]-x_direct)*(xx[n]-x_direct)+(yy[n]-y_direct)*(yy[n]-y_direct))
#         if temp<MEUCD_temp:
#             MEUCD_temp = temp
#             VRE_temp = 100*abs((yy[n]-y_direct)/y_direct)
#             MRE_temp = 100*abs((xx[n]-x_direct)/x_direct)
#         else:
#             continue
#     MEUCD.append(MEUCD_temp)
#     VRE.append(VRE_temp)
#     MRE.append(MRE_temp)
#     print(MEUCD_temp,VRE_temp,MRE_temp)
# print("MEUCD:",sum(MEUCD)/len(MEUCD),"VRE:", sum(VRE)/len(VRE),"MRE:",sum(MRE)/len(MRE))

def MPE(x_array,y_array,x,y):
    for n in range(len(x_array)-1):
        if x_array[n+1]<=x and x_array[n]>=x:
            y_coordinate = y_array[n]+(x-x_array[n])*(y_array[n+1]-y_array[n])/(x_array[n+1]-x_array[n])
            y_MPE = 100*abs(y-y_coordinate)/y_coordinate
            break
        else:
            y_MPE = np.nan
    for n in range(len(x_array) - 1):
        if y_array[n + 1] <= y and y_array[n] >= y:
            x_coordinate = x_array[n] + (y - y_array[n]) * (x_array[n + 1] - x_array[n]) / (y_array[n + 1] - y_array[n])
            x_MPE = 100 * abs(x - x_coordinate) / x_coordinate
            break
        else:
            x_MPE = np.nan
    return x_MPE,y_MPE,min(x_MPE,y_MPE)

# for i in range(5):
#     print("第{0}个数据集：".format(i+1))
#     data1 = pd.read_csv(ls_files_ef[i],encoding='gb2312')
#     data2 = pd.read_csv(ls_files_experiments_re_final[i],encoding='gb2312')
#     assets_num, asset_return, asset_cov, asset_cov_inv = get_inf(ls_files[i])
#     data1 = data1.iloc[0:data1.size + 1, 0]
#     data1 = data1.str.split(' ', expand=True)
#
#     xx = data1[2].astype(float)
#     yy = data1[4].astype(float)
#     yy = np.sqrt(yy)
#     ga = data2.iloc[0:data2.size,11]
#     ga = np.sqrt(ga)
#     x = data2.iloc[0:data2.size,10]
#     MPE_re = data2.iloc[0:data2.size,15]
#     adjusted = [np.nan]*5
#     for j in range(50):
#         if np.isnan(ga[j]):
#             continue
#         for m in range(len(xx)):
#             if abs(xx[m]-x[j])<=0.003*x[j]:
#                 try:
#                     primal = math.sqrt(primal_solution(xx[m],10,assets_num, asset_return, asset_cov)[0])
#                 except:
#                     continue
#                 x_MPE,y_MPE,min_MPE = MPE(xx,yy,xx[m],primal)
#                 if min_MPE<MPE_re[j]:
#                     MPE_re[j] = min_MPE
#                     adjusted = [xx[m],primal*primal,x_MPE,y_MPE,min_MPE]
#         print(adjusted[0],adjusted[1],adjusted[2],adjusted[3],adjusted[4])



