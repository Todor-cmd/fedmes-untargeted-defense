from loguru import logger
import torch
import time
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets import FashionMNISTDataset
from federated_learning.datasets.data_distribution import distribute_batches_equally,distribute_batches_bias, distribute_batches_1_class, distribute_batches_2_class, distribute_batches_dirichlet, distribute_batches_iid_ba, distribute_batches_iid_dba, distribute_batches_dirichlet_ba, distribute_batches_dirichlet_dba
from federated_learning.datasets.data_distribution import distribute_batches_noniid_mal
from federated_learning.utils import average_nn_parameters, fed_average_nn_parameters
from federated_learning.utils.aggregation import krum_nn_parameters, multi_krum_nn_parameters, bulyan_nn_parameters, trmean_nn_parameters, median_nn_parameters, fgold_nn_parameters
from federated_learning.utils.attack import reverse_nn_parameters, ndss_nn_parameters, reverse_last_parameters, lie_nn_parameters, free_nn_parameters,free_last_nn_parameters, free_rand_nn_parameters, fang_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements, identify_random_elements_inc_49
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader, load_ba_data_loader, load_dba_data_loader
from federated_learning.utils import load_test_data_loader, load_backdoor_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client
from federated_learning.nets import NetGenMnist, NetGenCifar, FashionMNISTCNNMAL, Cifar10CNNMAL




def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(args,
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    previous_weight = []
    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch),
                               str(clients[client_idx].get_client_index()))
        previous_weight = clients[0].get_nn_parameters()
        clients[client_idx].train(epoch)

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    dict_parameters = {client_idx: clients[client_idx].get_nn_parameters() for client_idx in random_workers}

    # defenses
    new_nn_params = {}

    # single server FL
    if args.get_topology() == "single":

        if args.get_aggregation_method() == "fedavg":
            parameters = {client_idx: clients[client_idx].get_nn_parameters() for client_idx in random_workers}
            sizes = {client_idx: clients[client_idx].get_client_datasize() for client_idx in random_workers}
            new_nn_params = fed_average_nn_parameters(parameters,sizes)

        elif args.get_aggregation_method() == "single_fedsgd":
            new_nn_params = average_nn_parameters(list(dict_parameters.values()))

        for client in clients:
            args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
            client.update_nn_parameters(new_nn_params)

    elif args.get_topology() == "multi_cross":
        args.get_logger().info("working on multi server crossing topology")
        multi_server_list = []

        # 1 overlapping
        # server_control_dict = {0: [0, 1, 2, 3, 4], 1: [2, 4, 5, 6, 7], 2: [3, 4, 7, 8, 9]}
        # 2 overlapping
        server_control_dict = {0: [0, 1, 2, 3, 4, 5], 1: [1, 2, 5, 6, 7, 8], 2: [3, 4, 5, 7, 8, 9]}

        for server in server_control_dict.keys():
            control_list = []
            for control_client in server_control_dict[server]:
                control_list.append({control_client: dict_parameters[control_client]})
            multi_server_list.append({server: control_list})

        if args.get_aggregation_method() == "fedsgd":
            args.get_logger().info("working on no defense")

            new_nn_params_list = []
            for server in range(len(multi_server_list)):
                params_dict = {}
                for i in multi_server_list[server][server]:
                    params_dict.update(i)
                new_nn_params_list.append(average_nn_parameters(list(params_dict.values())))

        elif args.get_aggregation_method() == "trmean":
            args.get_logger().info("working on trmean defense")

            new_nn_params_list = []
            for server in range(len(multi_server_list)):
                params_dict = {}
                for i in multi_server_list[server][server]:
                    params_dict.update(i)
                new_nn_params_list.append(trmean_nn_parameters(list(params_dict.values()), args = args))

        elif args.get_aggregation_method() == "krum":
            args.get_logger().info("working on krum defense")

            new_nn_params_list = []
            for server in range(len(multi_server_list)):
                params_dict = {}
                for i in multi_server_list[server][server]:
                    params_dict.update(i)
                new_nn_params_list.append(krum_nn_parameters(params_dict, args = args))

        elif args.get_aggregation_method() == "median":
            args.get_logger().info("working on median defense")

            new_nn_params_list = []
            for server in range(len(multi_server_list)):
                params_dict = {}
                for i in multi_server_list[server][server]:
                    params_dict.update(i)
                new_nn_params_list.append(median_nn_parameters(list(params_dict.values()), args = args))

        # updating parameters on 1 overlapping
        for client in clients:
            args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
            if args.get_aggregation_method() == "multi":
                if client.get_client_index() in [0,1]:
                    client.update_nn_parameters(new_nn_params_list[0])
                elif client.get_client_index() in [5,6]:
                    client.update_nn_parameters(new_nn_params_list[1])
                elif client.get_client_index() in [8,9]:
                    client.update_nn_parameters(new_nn_params_list[2])
                elif client.get_client_index() == 4:
                    comb_all = average_nn_parameters(new_nn_params_list)
                    client.update_nn_parameters(comb_all)
                    args.get_logger().info("server agg: " + str(clients[4].test()))
                elif client.get_client_index() == 2:
                    comb_0_1 = average_nn_parameters(new_nn_params_list[:2])
                    client.update_nn_parameters(comb_0_1)
                elif client.get_client_index() == 7:
                    comb_1_2 = average_nn_parameters(new_nn_params_list[1:])
                    client.update_nn_parameters(comb_1_2)
                elif client.get_client_index() == 3:
                    comb_0_2 = average_nn_parameters([new_nn_params_list[0],new_nn_params_list[2]])
                    client.update_nn_parameters(comb_0_2)
        args.get_logger().info("server 1: "+ str(clients[0].test()))
        args.get_logger().info("server 2: "+ str(clients[5].test()))
        args.get_logger().info("server 3: "+ str(clients[8].test()))

        # updating parameters on 2 overlapping
        for client in clients:
            args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
            if args.get_aggregation_method() == "multi":
                if client.get_client_index() in [0,1]:
                    client.update_nn_parameters(new_nn_params_list[0])
                elif client.get_client_index() in [5,6]:
                    client.update_nn_parameters(new_nn_params_list[1])
                elif client.get_client_index() in [8,9]:
                    client.update_nn_parameters(new_nn_params_list[2])
                elif client.get_client_index() == 4:
                    comb_all = average_nn_parameters(new_nn_params_list)
                    client.update_nn_parameters(comb_all)
                    args.get_logger().info("server agg: " + str(clients[4].test()))
                elif client.get_client_index() == 2:
                    comb_0_1 = average_nn_parameters(new_nn_params_list[:2])
                    client.update_nn_parameters(comb_0_1)
                elif client.get_client_index() == 7:
                    comb_1_2 = average_nn_parameters(new_nn_params_list[1:])
                    client.update_nn_parameters(comb_1_2)
                elif client.get_client_index() == 3:
                    comb_0_2 = average_nn_parameters([new_nn_params_list[0],new_nn_params_list[2]])
                    client.update_nn_parameters(comb_0_2)
        args.get_logger().info("server 1: "+ str(clients[0].test()))
        args.get_logger().info("server 2: "+ str(clients[5].test()))
        args.get_logger().info("server 3: "+ str(clients[8].test()))

    elif args.get_topology() == "multi_line":
        args.get_logger().info("working on multi server line topology")

        if args.get_aggregation_method() == "fedsgd":
            args.get_logger().info("working on no defense")

            multi_server_list = []
            server_control_dict = {0:[0,1,2,3], 1:[3,4,5,6],2: [6,7,8,9]}
            for server in server_control_dict.keys():
                control_list = []
                for control_client in server_control_dict[server]:
                    control_list.append({control_client: dict_parameters[control_client]})
                multi_server_list.append({server : control_list})

            new_nn_params_list = []
            for server in range(len(multi_server_list)):
                params_dict = {}
                for i in multi_server_list[server][server]:
                    params_dict.update(i)
                new_nn_params_list.append(average_nn_parameters(list(params_dict.values())))

            for client in clients:
                args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
                if args.get_aggregation_method() == "multi":
                    if client.get_client_index() in [0,1,2]:
                        client.update_nn_parameters(new_nn_params_list[0])
                    elif client.get_client_index() in [4,5]:
                        client.update_nn_parameters(new_nn_params_list[1])
                    elif client.get_client_index() in [7,8,9]:
                        client.update_nn_parameters(new_nn_params_list[2])
                    elif client.get_client_index() == 3:
                        comb_all = average_nn_parameters(new_nn_params_list)
                        client.update_nn_parameters(comb_all)
                        print("server agg: " + str(clients[3].test()))
                        comb_0_1 = average_nn_parameters(new_nn_params_list[:2])
                        client.update_nn_parameters(comb_0_1)
                    elif client.get_client_index() == 6:
                        comb_1_2 = average_nn_parameters(new_nn_params_list[1:])
                        client.update_nn_parameters(comb_1_2)
            args.get_logger().info("server 1: "+ str(clients[0].test()))
            args.get_logger().info("server 2: "+ str(clients[4].test()))
            args.get_logger().info("server 3: "+ str(clients[7].test()))


    all = 0
    select = 0

    return clients[4].test(), clients[4].backdoor_test(), clients[0].test(),  clients[5].test(), clients[8].test(),  random_workers, all, select


def create_clients(args, train_data_loaders, test_data_loader, backdoor_test_data_loader, distributed_train_dataset):
    """
    Create a set of clients.
    """
    clients = []
    if args.get_attack_strategy() == "cua" and (args.get_dataset() == "mnist" or args.get_dataset() == "fashion_mnist"):
        for idx in range(int(args.get_num_workers()*(1-args.get_mal_prop()))):
            clients.append(Client(args = args, client_idx = idx, is_mal= 'False', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, distributed_train_dataset = distributed_train_dataset[idx], gen_net = NetGenMnist(z_dim=args.n_dim)))
        for idx in range(int(args.get_num_workers()*(1-args.get_mal_prop())),int(args.get_num_workers())):
            if args.get_cua_syn_data_version() == "generator":
                gen_net = NetGenMnist(z_dim=args.n_dim)
            else:
                gen_net = FashionMNISTCNNMAL()
            clients.append(Client(args = args, client_idx = idx, is_mal= 'CUA', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, distributed_train_dataset = distributed_train_dataset[idx], gen_net = gen_net))
    if args.get_attack_strategy() == "cua" and (args.get_dataset() == "cifar_10" or args.get_dataset() == "cifar_100"):
        for idx in range(int(args.get_num_workers()*(1-args.get_mal_prop()))):
            clients.append(Client(args = args, client_idx = idx, is_mal= 'False', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, distributed_train_dataset = distributed_train_dataset[idx], gen_net = NetGenCifar(z_dim=args.n_dim)))
        for idx in range(int(args.get_num_workers()*(1-args.get_mal_prop())),int(args.get_num_workers())):
            if args.get_cua_syn_data_version() == "generator":
                gen_net = NetGenCifar(z_dim=args.n_dim)
            else:
                gen_net = Cifar10CNNMAL()
            clients.append(Client(args = args, client_idx = idx, is_mal= 'CUA', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, distributed_train_dataset = distributed_train_dataset[idx], gen_net = gen_net))
    else:
        for idx in range(int(args.get_num_workers())):
            clients.append(Client(args = args, client_idx = idx, is_mal= 'False', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, backdoor_test_data_loader = backdoor_test_data_loader,distributed_train_dataset = distributed_train_dataset[idx], gen_net = NetGenMnist(z_dim=args.n_dim)))

    return clients

def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    epoch_test_set_results_0 = []
    epoch_test_set_results_1 = []
    epoch_test_set_results_2 = []
    worker_selection = []
    all_worker_nums = []
    select_attacker_nums = []


    for epoch in range(1, args.get_num_epochs() + 1):
        results, backdoor_results, results_0, results_1, results_2, workers_selected, all_worker_num, select_attacker_num = train_subset_of_clients(epoch, args, clients, poisoned_workers)
        epoch_test_set_results.append(results + (backdoor_results,))
        epoch_test_set_results_0.append(results_0)
        epoch_test_set_results_1.append(results_1)
        epoch_test_set_results_2.append(results_2)
        worker_selection.append(workers_selected)
        all_worker_nums.append(all_worker_num)
        select_attacker_nums.append(select_attacker_num)

    return convert_results_to_csv(epoch_test_set_results), convert_results_to_csv(epoch_test_set_results_0), convert_results_to_csv(epoch_test_set_results_1),convert_results_to_csv(epoch_test_set_results_2), worker_selection, all_worker_nums, select_attacker_nums


def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, backdoor_results_files, results_0_files, results_1_files, results_2_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)
    ba_data_loader = load_ba_data_loader(logger, args)
    dba_data_loader = load_dba_data_loader(logger, args)
    backdoor_test_data_loader = load_backdoor_test_data_loader(logger, args)

    # Distribute batches

    # if args.get_distribution_method() == "bias":
    #     distributed_train_dataset = distribute_batches_bias(train_data_loader, args.get_num_workers())
    # elif args.get_distribution_method() == "iid":
    #     distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    if args.get_distribution_method() == "noniid_1":
        distributed_train_dataset = distribute_batches_1_class(train_data_loader, args.get_num_workers(), args = args)
    elif args.get_distribution_method() == "noniid_2":
        distributed_train_dataset = distribute_batches_2_class(train_data_loader, args.get_num_workers(), args = args)
    elif args.get_distribution_method() == "noniid_dir_0":
        distributed_train_dataset = distribute_batches_dirichlet(train_data_loader, args.get_num_workers(), args.get_mal_prop(), args = args, type=0)
    elif args.get_distribution_method() == "noniid_dir_1":
        distributed_train_dataset = distribute_batches_dirichlet(train_data_loader, args.get_num_workers(), args.get_mal_prop(), args = args, type=1)
    elif args.get_distribution_method() == "noniid_dir_2":
        distributed_train_dataset = distribute_batches_dirichlet(train_data_loader, args.get_num_workers(), args.get_mal_prop(), args = args, type=2)
    elif args.get_distribution_method() == "iid_ba":
        distributed_train_dataset = distribute_batches_iid_ba(train_data_loader, ba_data_loader, args.get_mal_prop(), args.get_num_workers(), args = args)
    elif args.get_distribution_method() == "noniid_dir_2_ba":
        distributed_train_dataset = distribute_batches_dirichlet_ba(train_data_loader, ba_data_loader, args.get_mal_prop(), args.get_num_workers(), args = args, type=2)
    elif args.get_distribution_method() == "iid_dba":
        distributed_train_dataset = distribute_batches_iid_dba(train_data_loader, ba_data_loader, args.get_mal_prop(), args.get_num_workers(), args = args)
    elif args.get_distribution_method() == "noniid_dir_2_dba":
        distributed_train_dataset = distribute_batches_dirichlet_dba(train_data_loader, ba_data_loader, args.get_mal_prop(), args.get_num_workers(), args = args, type=2)

    else:
        distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())


    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers,
                                            replacement_method, args.get_poison_effort)

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset,
                                                                        args.get_batch_size())

    clients = create_clients(args, train_data_loaders, test_data_loader, backdoor_test_data_loader, distributed_train_dataset)

    results, results_0, results_1, results_2, worker_selection, all_worker_nums, select_attacker_nums = run_machine_learning(clients, args, poisoned_workers)
    max = 0
    for i in results:
        if i[0]>max:
            max = i[0]
    print(max)
    # print(sum(select_attacker_nums))
    # print(sum(all_worker_nums))
    args.get_logger().info("random all attacker num is #{}, selected attacker num is #{}, best acc is #{} ", str(sum(all_worker_nums)), str(sum(select_attacker_nums)), str(max))

    result_name = args.get_result_name() + "rep_"
    save_results(results, args.get_dataset() + "_" + args.get_aggregation_method() + "_" +args.get_attack_strategy() + "_" +str(args.get_mal_prop()) + "_" + args.get_distribution_method() + "_" + str(args.get_beta())  + "_" + results_files[0] )
    save_results(worker_selection, args.get_dataset() + "_" + args.get_aggregation_method() + "_" +args.get_attack_strategy() + "_" +str(args.get_mal_prop()) + "_" + args.get_distribution_method() + "_" + str(args.get_beta()) + "_" + worker_selections_files[0])
    save_results(results_0, args.get_dataset() + "_" + args.get_aggregation_method() + "_" +args.get_attack_strategy() + "_" +str(args.get_mal_prop()) + "_" + args.get_distribution_method() + "_" + str(args.get_beta())  + "_" + results_0_files[0] )
    save_results(results_1, args.get_dataset() + "_" + args.get_aggregation_method() + "_" +args.get_attack_strategy() + "_" +str(args.get_mal_prop()) + "_" + args.get_distribution_method() + "_" + str(args.get_beta())  + "_" + results_1_files[0] )
    save_results(results_2, args.get_dataset() + "_" + args.get_aggregation_method() + "_" +args.get_attack_strategy() + "_" +str(args.get_mal_prop()) + "_" + args.get_distribution_method() + "_" + str(args.get_beta())  + "_" + results_2_files[0] )
    # save_results(results, result_name + results_files[0] )
    # save_results(worker_selection, result_name + worker_selections_files[0])

    logger.remove(handler)
