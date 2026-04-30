import os
import json
import matplotlib.pyplot as plt

def save_results(experiment_params,results_dir):
    exp_id = experiment_params['id']
    iteration = experiment_params['iteration']
    
    experiment_desc = experiment_params['description']
    save_dir = results_dir / f"{exp_id}" / f"replica-{iteration}"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(save_dir / f'{exp_id}.json', 'r') as f:
            results = json.load(f)
        print(f'Loaded results for experiment {exp_id}\n{experiment_desc}')
    except FileNotFoundError:
        print(f'Creating new results file for experiment {exp_id}\n{experiment_desc}')
        # create new results file
        results = {}
        results['id'] = exp_id
        results['description'] = experiment_desc
        results['experiments'] = []

    results['experiments'] = [
        experiment
        for experiment in results.get('experiments', [])
        if experiment.get('params', {}).get('iteration') != iteration
    ]
    results['experiments'].append({'params': experiment_params,
                                    'accuracies_by_round': [],
                                    'loss_by_round': []
                                    })
    # load all nodes metrics
    node_metrics_dir = results_dir / f"{exp_id}" / f"replica-{iteration}" / "node_metrics"
    node_metrics_dir.mkdir(parents=True, exist_ok=True)
    avg_accuracies_by_round = [0]*experiment_params['rounds']
    avg_losses_by_round = [0]*experiment_params['rounds']
    num_benign_nodes = 0
    for node_hash_json in os.listdir(node_metrics_dir):
        with open(os.path.join(node_metrics_dir,node_hash_json),'r') as f:
            node_metrics = json.load(f)
        node_accuracies = node_metrics['accuracies']
        node_losses = node_metrics['losses']
        for r in range(experiment_params['rounds']):
            if r >= len(node_accuracies) or r >= len(node_losses):
                raise ValueError(f"Missing round {r} metrics for node {node_hash_json}")
            if node_accuracies[r] is None or node_losses[r] is None:
                raise ValueError(f"Incomplete round {r} metrics for node {node_hash_json}")
            avg_accuracies_by_round[r] += node_accuracies[r]
            avg_losses_by_round[r] += node_losses[r]
        num_benign_nodes += 1
    if num_benign_nodes == 0:
        raise ValueError(f"No node metrics found in {node_metrics_dir}")
    results['experiments'][-1]['accuracies_by_round'] = [a/num_benign_nodes for a in avg_accuracies_by_round]
    results['experiments'][-1]['loss_by_round'] = [loss/num_benign_nodes for loss in avg_losses_by_round]

    with open(save_dir / f'{exp_id}.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved results to", save_dir / f'{exp_id}.json')



def save_node_metrics(
        node_hash,
        accuracy,
        loss,
        exp_id,
        iteration,
        results_dir,
        round_num=None,
        is_malicious=None):
    '''
    Save node metrics to a json file.
    '''
    save_dir = results_dir / f"{exp_id}" / f"replica-{iteration}" / "node_metrics"
    save_dir.mkdir(parents=True, exist_ok=True)
    node_file = save_dir / f"{node_hash}.json"

    try:
        with open(node_file,'r') as f:
            results = json.load(f)
        print(f'Loaded results for node {node_hash}')
    except FileNotFoundError:
        print(f'Creating new results file for node {node_hash}')
        # create new results file
        results = {}
        results['node_hash'] = node_hash
        results['is_malicious'] = is_malicious
        results['accuracies'] = []
        results['losses'] = []

    if is_malicious is not None:
        results['is_malicious'] = is_malicious

    if round_num is None:
        results['accuracies'].append(accuracy)
        results['losses'].append(loss)
    else:
        while len(results['accuracies']) <= round_num:
            results['accuracies'].append(None)
        while len(results['losses']) <= round_num:
            results['losses'].append(None)
        results['accuracies'][round_num] = accuracy
        results['losses'][round_num] = loss

    with open(node_file,'w') as f:
        json.dump(results, f,indent=4)
    print("Saved results to", node_file)

def make_plot(exp_id,iteration,results_dir):
    experiment_json_path = results_dir / f"{exp_id}" / f"replica-{iteration}" / f'{exp_id}.json'
    with open(experiment_json_path,'r') as f:
        results = json.load(f)

    print(len(results['experiments']))
    line_type = { 'scale-free': '-', 'small-world': '--', 'two-f-1': '-.'}
    line_color = {0.0: 'green', 0.3: 'blue', 0.6: 'red', .15: 'pink', .45: 'orange', 0.1: 'pink', 0.05: 'purple', 0.25: 'black'}
    for i in range(len(results['experiments'])):
        accuracies_by_round = results['experiments'][i]['accuracies_by_round']
        # experiment_params = results['experiments'][i]['params']
        byzantine_proportion = results['experiments'][i]['params']['malicious_proportion']
        topology = results['experiments'][i]['params']['topology']
        plt.plot(range(1,len(accuracies_by_round)+1), 
                 accuracies_by_round, 
                 label=f'{topology} b={byzantine_proportion}', 
                 linestyle=line_type[topology], 
                 color=line_color[byzantine_proportion],
                 marker='o')

    plt.legend()

    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Trimmed Mean Accuracy by Round \nStrategic Byzantine Placement')
    plt.savefig(results_dir / f'{exp_id}' / f"replica-{iteration}" / "accuracy_by_round.png")

    plt.clf()
