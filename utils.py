import pickle
import os

def nice_table(dict, title=''):
    '''
    Given a dictionary, it returns an HTML tables with the key-value pairs arranged in rows or columns.
    '''
    # make a copy
    dict = dict.copy()
    html = f'<h2 style="text-align:left;">{title}</h2>'
    html += '<table style="width:50%; border-collapse: collapse; font-size: 16px; text-align:center; padding: 10px; border: 1px solid #fff;">'
    html += '<tr>'
    for key, value in dict.items():
        html += f'<td style="border: 1px solid #fff; text-align:center; padding: 10px; color: white; border-right: 1px solid #fff;">{key}</td>'
    html += '</tr>'
    
    # check if type of value is scalar and if it is, convert it to a list
    for key, value in dict.items():
        if not isinstance(value, list):
            if isinstance(value, float):
                if value < 1:
                    value = round(value, 5)
                else:
                    value = round(value, 3) 
            dict[key] = [value]

    for i in range(max([len(value) for value in dict.values()])):
        html += '<tr>'
        for key, value in dict.items():
            html += f'<td style="border: 1px solid #fff; text-align:center; padding: 10px; color: white; opacity: 0.8; border-left: 1px solid #fff;">{value[i]}</td>'
        html += '</tr>'
            
    return html


def load_hyperparameters(model_name):
    '''
    Given model name, it returns the hyperparameters found by hyperparameter search.
    '''
    # if file exists
    if os.path.isfile(f'../../Saved/{model_name}_opt_params.pkl'):
        with open(f'../../Saved/{model_name}_opt_params.pkl', 'rb') as f:
            opt_params = pickle.load(f)
        return opt_params
    else:
        return {}

def save_hyperparameters(model_name, opt_params):
    '''
    Given model name and hyperparameters, it saves the hyperparameters found by hyperparameter search.
    '''
    with open(f'../../Saved/{model_name}_opt_params.pkl', 'wb') as f:
        pickle.dump(opt_params, f)

def load_model(model_name):
    '''
    Given model name, it returns the model.
    '''
    if not os.path.isfile(f'../../Saved/{model_name}.pkl'):
        return None
    with open(f'../../Saved/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def save_model(model_name, model):
    '''
    Given model name and model, it saves the model.
    '''
    with open(f'../../Saved/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
        
def get_metrics(report):
    '''
    Get useful metrics from classification report.
    '''
    acc, wf1 = report.split('\n')[-2].split()[3:5]
    acc, wf1 = float(acc), float(wf1)
    return acc, wf1