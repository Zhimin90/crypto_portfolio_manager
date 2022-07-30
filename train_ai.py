import pandas as pd
import numpy as np
import pygad
import torch
from pygad import torchga
from portfolio_model import LSTMModel

def run_ai(daily_history_df, asset_ls=['BTC-USD','ETH-USD'], adjustment_period=7):
    global model
    global optimized_portfolio, portfolio_values, portfolio_value_df
    global prev_pred, prev_portfolio, sol_pred, sol_portfolio
    global epoch, period 
    print('executing ai porfolio manager training routing for historical data')
    daily_df = daily_history_df[daily_history_df['ticker'] == asset_ls[-1]].copy() #last asset ETH-USD has shorter index

    def get_daily_return(ticker, time, df=daily_history_df):
        if ticker == 'USD':
            return 1.0
        else:
            df = df.loc[time]
            asset_df = df[df['ticker'].apply(lambda x: ticker in x)]
            return (asset_df['close']/asset_df['open']).values[0]
            
    def calc_asset_daily_returns(row):
        asset_ls = row.index
        time = row.name
        daily_returns_ls = np.array([get_daily_return(asset, time) for asset in asset_ls])
        return daily_returns_ls
        
    initial_allocation_ratio = np.array([[.6],[.2],[.1]])
    portfolio_df = pd.DataFrame(np.repeat(initial_allocation_ratio, daily_df.shape[0], axis=1).T,columns=['BTC','ETH','USD'],index=daily_df.index)
    asset_daily_returns_df = portfolio_df.apply(lambda x: pd.Series(calc_asset_daily_returns(x),index=portfolio_df.columns), axis=1)
    
    def get_accumulated_return(initial_allocation_total_value=1000.00, initial_allocation_ratio=np.array([[.6],[.3],[.1]]), asset_daily_returns_df=asset_daily_returns_df):
        ratio_total = round(initial_allocation_ratio.sum(),6)
        assert ratio_total == 1, f"ratio total is {ratio_total} and doesnt add up to 1"
        initial_allocation = initial_allocation_ratio * initial_allocation_total_value 
        portfolio_df = pd.DataFrame(np.repeat(initial_allocation_ratio, daily_df.shape[0], axis=1).T,columns=['BTC','ETH','USD'],index=daily_df.index)
        
        return_df = (asset_daily_returns_df).shift(1)
        accumulator_df = pd.DataFrame(columns=asset_daily_returns_df.columns,index=asset_daily_returns_df.index)
        accumulator_df.iloc[0] = pd.Series(initial_allocation.flatten(),index=asset_daily_returns_df.columns) # initialize portfolio value of first index

        shape = accumulator_df.shape[0]
        cummulated_returns = (np.cumprod(return_df.shift(-1).values, axis=0)-1)*np.tile(initial_allocation,[shape,1]).reshape(shape, accumulator_df.columns.shape[0]) +\
            np.tile(initial_allocation,[shape,1]).reshape(shape,accumulator_df.columns.shape[0])

        accumulator_df.iloc[1:] = pd.DataFrame(cummulated_returns, index=accumulator_df.index).shift(1)
        return accumulator_df

    gains_ls = []
    porfolio_details = pd.DataFrame()
    for i in range(0,asset_daily_returns_df.shape[0]-adjustment_period, adjustment_period):
        current_portfolio_total = initial_allocation_total_value=1000.00 if i==0 else periodic_returns.tail(1).sum(axis=1).values[0]
        periodic_returns = get_accumulated_return(current_portfolio_total, initial_allocation_ratio=np.array([[.6],[.3],[.1]]),
                                            asset_daily_returns_df= asset_daily_returns_df.iloc[i:i+adjustment_period])
        gains_ls+=[(periodic_returns.tail(1).sum(axis=1).values[0],
                    asset_daily_returns_df.iloc[i+adjustment_period].name)]
        porfolio_details=porfolio_details.append(periodic_returns.iloc[-1])
        
    portfolio_value_df = porfolio_details.copy()

    def generate_row_data(index=0, period=adjustment_period, portfolio=[600,300,100], allocation_ratio=np.array([[.6],[.3],[.1]])):
        output_portfolio = get_accumulated_return(initial_allocation_total_value=sum(portfolio), initial_allocation_ratio=allocation_ratio, asset_daily_returns_df=asset_daily_returns_df.iloc[index:index+period])
        return output_portfolio.iloc[-1].values.astype(float)

    print("test generate_row_data", generate_row_data(index=0, period=adjustment_period, portfolio=[800,300,100], allocation_ratio=np.array([[.6],[.3],[.1]])))

    def fitness_func(solution, sol_idx):
        global torch_ga, model,  sol_pred, sol_portfolio
        model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                            weights_vector=solution)
        #print(model_weights_dict['lstm.weight_ih_l0'])
        # Use the current solution as the model parameters.
        model.load_state_dict(model_weights_dict)
        #print(f'{sol_idx} prev_portfolio',prev_portfolio)
        data_inputs = generate_row_data(index=epoch*period, period=period, portfolio=prev_portfolio, allocation_ratio=prev_pred)
        sol_portfolio[sol_idx] = data_inputs

        pred = torch.nn.functional.normalize(torch.abs(model(torch.from_numpy(data_inputs).float())),p=1,dim=0).reshape(3,1).detach().numpy()
        #print(pred)
        sol_pred[sol_idx] = pred
        
        updated_pred = generate_row_data(index=epoch*period, period=period, portfolio=prev_portfolio, allocation_ratio=pred) 
        #print(updated_pred)
        
        reg_fact1 = 1e-1
        reg_fact2 = 1e-3
        #fitness function is regularized for portfolio ratio changes and even distribution
        solution_fitness = (updated_pred).sum() - reg_fact1*(updated_pred).sum()*(np.abs(prev_pred - pred)).sum() - reg_fact2/np.cumprod(pred,axis=0)[-1]*(updated_pred).sum()
        #solution_fitness = (updated_pred).sum()*(1-reg_fact1*(np.abs(prev_pred - pred)).sum())*(1-reg_fact2/np.cumprod(pred,axis=0)[-1]) 
        return solution_fitness[0]

    def callback_generation(ga_instance):
        global prev_pred, prev_portfolio, epoch, optimized_portfolio, portfolio_values
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
        print(f"Date {portfolio_value_df.index[epoch]}")
        sol_idx = ga_instance.best_solution()[2]
        prev_pred = sol_pred[sol_idx]
        print(sol_pred[sol_idx].flatten())
        prev_portfolio = sol_portfolio[sol_idx]

        current_portfolio = pd.DataFrame([sol_pred[sol_idx].flatten()], columns=['BTC','ETH','USD'], index=[portfolio_value_df.index[epoch]])
        optimized_portfolio = pd.concat([ optimized_portfolio, current_portfolio])
        
        current_portfolio_values = pd.DataFrame([sol_portfolio[sol_idx].flatten()], columns=['BTC','ETH','USD'], index=[portfolio_value_df.index[epoch]])
        portfolio_values = pd.concat([ portfolio_values, current_portfolio_values])
        
        epoch += 1

    np.random.seed(0) #fix weights update
    torch.manual_seed(0) #fix init weights

    model = LSTMModel(3,32,4,3,.2)
    torch_ga = torchga.TorchGA(model=model, num_solutions=120)

    num_generations = np.floor(daily_df.shape[0]/adjustment_period).astype(int) #Each generation is adjustment_period day period weight update based on how well to optimize this current period
    print('num_generations', num_generations)
    num_parents_mating = 12
    initial_population = torch_ga.population_weights

    print(initial_population[0])
    ga_instance = pygad.GA(num_generations=num_generations, 
                        num_parents_mating=num_parents_mating, 
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        crossover_type =  "uniform",
                        mutation_type = None,
                        on_generation=callback_generation)

    optimized_portfolio = pd.DataFrame()
    portfolio_values = pd.DataFrame()

    prev_pred = np.array([[.6],[.3],[.1]])
    prev_portfolio = np.array([600,300,100])

    sol_pred = {}
    sol_portfolio = {}

    epoch = 0
    period = adjustment_period

    ga_instance.run()
    print(prev_portfolio, prev_portfolio.sum())
    portfolio_values.to_parquet('portfolio.pqt')
    optimized_portfolio.to_parquet('portfolio_ratio.pqt')