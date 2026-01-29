import torch
import random
from torch.utils.data.dataset import Subset
from continual_datasets.base_datasets import *
from continual_datasets.dataset_utils import build_transform, UnknownWrapper, get_dataset

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = domain_list = None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    mode = args.IL_mode

    if mode == 'cil':
        if 'iDigits' in args.dataset:
            dataset_list = args.id_datasets
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )

                splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
                mask.append(class_mask)

                for i in range(len(splited_dataset)):
                    train.append(splited_dataset[i][0])
                    val.append(splited_dataset[i][1])
            
            #mask = [[[0,1], [2,3], [4,5], [6,7], [8,9]] , [[0,1], [2,3], [4,5], [6,7], [8,9]], ...] 4개의 데이터셋에 대해 각각의 태스크별 클래스 마스크
            #train = [train, train, ... ] 20개
            #val = [val, val, ... ] 20개
                    
            splited_dataset = list()
            for i in range(args.num_tasks): #5  
                t = [train[i+args.num_tasks*j] for j in range(len(dataset_list))]
                v = [val[i+args.num_tasks*j] for j in range(len(dataset_list))]
                splited_dataset.append((torch.utils.data.ConcatDataset(t), torch.utils.data.ConcatDataset(v)))

            #splited_dataset = [(train, val), (train, val), (train, val), (train, val), (train, val)]
            #class_mask = [[0,1], [2,3], [4,5], [6,7], [8,9]]
            args.num_classes = len(splited_dataset[0][1].datasets[0].dataset.classes)
            class_mask = np.unique(np.array(mask), axis=0).tolist()[0] 
            #domain_list = ["D0123", "D0123", "D0123", "D0123", "D0123"]
            domain_list = [f'D{"".join(map(str, range(len(dataset_list))))}'] * args.num_tasks
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )   

            splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
            domain_list = ["D0"] * args.num_tasks
            args.num_classes = len(dataset_val.classes)

    elif mode in ['dil', 'vil']:
        if 'iDigits' in args.dataset:
            dataset_list = args.id_datasets
            splited_dataset = list()

            for i in range(len(dataset_list)):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset_list[i],
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                splited_dataset.append((dataset_train, dataset_val))
            #splited_dataset = [(train, val), (train, val), (train, val), (train, val)] 각 d0,d1,d2,d3
            args.num_classes = len(dataset_val.classes)
            if mode == 'dil':
                class_mask = [[j for j in range(args.num_classes)] for i in range(len(splited_dataset)) ]
                domain_list = [f'D{i}' for i in range(len(splited_dataset)) ]
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )
            domain_list = [f'D{i}' for i in range(len(dataset_train))]

            if args.dataset in ['CORe50']:
                splited_dataset = [(dataset_train[i], dataset_val) for i in range(len(dataset_train))]
                args.num_classes = len(dataset_val.classes)
                # 문자열 대신 정수 인덱스(0부터 num_classes-1까지)로 생성
                class_mask = [[j for j in range(args.num_classes)] for i in range(args.num_tasks)]
            else:
                splited_dataset = [(dataset_train[i], dataset_val[i]) for i in range(len(dataset_train))]
                args.num_classes = len(dataset_val[0].classes)
                # 문자열 대신 정수 인덱스(0부터 num_classes-1까지)로 생성
                class_mask = [[j for j in range(args.num_classes)] for i in range(args.num_tasks)]
    
    elif mode in ['joint']:
        if 'iDigits' in args.dataset:
            dataset_list = args.id_datasets
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                train.append(dataset_train)
                val.append(dataset_val)
                args.num_classes = len(dataset_val.classes)

            dataset_train = torch.utils.data.ConcatDataset(train)
            dataset_val = torch.utils.data.ConcatDataset(val)
            splited_dataset = [(dataset_train, dataset_val)]        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset = [(dataset_train, dataset_val)]

            args.num_classes = len(dataset_val.classes)

    else:
        raise ValueError(f'Invalid mode: {mode}')
                

    if args.IL_mode in ['vil']:
        splited_dataset, class_mask, domain_list, args = build_vil_scenario(splited_dataset, args)

    for i in range(len(splited_dataset)):
        dataset_train, dataset_val = splited_dataset[i]

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})
    
    args.develop_tasks = False
    args.verbose = False
    if args.verbose or args.develop_tasks:
        print(f"{'TASK INFO':=^60}")
        print(f"{'IL mode':<20} => {args.IL_mode}")
        print(f"{'Dataset':<20} => {args.dataset}")
        print(f"{'Number of tasks':<20} => {args.num_tasks}")
        print(f"{'Number of classes':<20} => {args.num_classes}")
        print(f"{'Number of domains':<20} => {args.num_domains}")
        print("domain_list:", domain_list)
        print("class_mask:", class_mask)
        print("dataloader: ",len(dataloader))
        print(f"{'Sequence of Tasks':=^60}")    
        for t_id in range(args.num_tasks):
            dom_info = domain_list[t_id] if domain_list is not None else "N/A"
            cls_info = class_mask[t_id] if class_mask is not None else "N/A"
            print(f"Task {t_id+1} => domain(s): {dom_info}, classes: {cls_info}")
        print(f"{'':=^60}")

    return dataloader, class_mask, domain_list

def split_single_dataset(dataset_train, dataset_val, args):
    #CIL 세팅
    num_classes = len(dataset_val.classes) # [0,1,2,3,4,5,6,7,8,9] -> 10
    assert num_classes % args.num_tasks == 0 # 10 % 5 = 0
    classes_per_task = num_classes // args.num_tasks # 10 // 5 = 2

    labels = [i for i in range(num_classes)] # [0,1,2,3,4,5,6,7,8,9]
    
    split_datasets = list() # [[train, val], [train, val], [train, val], [train, val], [train, val]]
    mask = list() # [[0,1], [2,3], [4,5], [6,7], [8,9]]

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks): # 5
        train_split_indices = list()  # [0,1]
        test_split_indices = list() # [2,3,4,5,6,7,8,9]
        
        scope = labels[:classes_per_task] # [0,1]
        labels = labels[classes_per_task:] # [2,3,4,5,6,7,8,9]

        mask.append(scope)

        #학습 데이터셋과 검증 데이터셋의 각 샘플에 대해, 해당 샘플의 레이블이 현재 scope에 속하면 그 인덱스를 선택
        for k in range(len(dataset_train.targets)): 
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)

        #선택된 인덱스들을 이용해 torch.utils.data.Subset을 생성하고, 이 서브셋을 현재 태스크의 학습/검증 데이터로 사용
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask # [[train, val], [train, val], ...], [[0,1], [2,3], [4,5], [6,7], [8,9]]

def build_vil_scenario(splited_dataset, args):
    datasets = list()
    class_mask = list()
    domain_list = list()
    
    num_tasks = args.num_tasks

    #split_single_dataset에서 cil 세팅을 위해 잠시 num_tasks를 변경
    args.num_tasks = int(args.num_tasks / len(splited_dataset))
    #splited_dataset = [(train, val), (train, val), (train, val), (train, val)] 각 d0,d1,d2,d3
    for i in range(len(splited_dataset)):
        dataset, mask = split_single_dataset(splited_dataset[i][0], splited_dataset[i][1], args)
        datasets.append(dataset)
        class_mask.append(mask)
        for _ in range(len(dataset)):
            domain_list.append(f'D{i}')

    splited_dataset = sum(datasets, [])
    class_mask = sum(class_mask, [])

    assert num_tasks == len(splited_dataset)
    args.num_tasks = num_tasks

    zipped = list(zip(splited_dataset, class_mask, domain_list))
    random.shuffle(zipped)
    splited_dataset, class_mask, domain_list = zip(*zipped)

    return splited_dataset, class_mask, domain_list, args

