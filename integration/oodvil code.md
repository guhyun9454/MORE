
def main(args):
    args = set_data_config(args)
    device = torch.device(args.device)

    seed_everything(args.seed)
    
    data_loader, class_mask, domain_list = build_continual_dataloader(args)
    if args.ood_dataset:
        data_loader[-1]['ood'] = get_ood_dataset(args.ood_dataset, args)
        
    
    print(args)
    args.wandb = False
    if args.wandb_run and args.wandb_project:
        import wandb
        import getpass

        args.wandb = True
        wandb.init(entity="OODVIL",project=args.wandb_project, name=args.wandb_run, config=args)
        wandb.config.update({"username": getpass.getuser()})
        
      
      train_and_evaluate()
      
    def train_and_evaluate(self, model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, args):
        """
        전체 incremental learning 과정을 수행합니다.
        각 task에 대해 지정된 epoch만큼 fine-tuning 후,
        지금까지의 task에 대해 평가합니다.
        """
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            print(f"{f'Training on Task {task_id+1}/{args.num_tasks}':=^60}")
            train_start = time.time()
            for epoch in range(args.epochs):
                epoch_start = time.time()
                epoch_avg_loss, epoch_avg_acc = self.train_one_epoch(model, criterion, data_loader[task_id]['train'], optimizer, device, epoch, args)
                epoch_duration = time.time() - epoch_start  
                print(f"Epoch [{epoch+1}/{args.epochs}] Completed in {str(datetime.timedelta(seconds=int(epoch_duration)))}: Avg Loss = {epoch_avg_loss:.4f}, Avg Acc@1 = {epoch_avg_acc:.2f}")

                if lr_scheduler is not None:
                    lr_scheduler.step(epoch)
            train_duration = time.time() - train_start
            print(f"Task {task_id+1} training completed in {str(datetime.timedelta(seconds=int(train_duration)))}")

         

            print(f'{f"Testing on Task {task_id+1}/{args.num_tasks}":=^60}')
            eval_start = time.time()
            self.evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args)
            eval_duration = time.time() - eval_start
            print(f"Task {task_id+1} evaluation completed in {str(datetime.timedelta(seconds=int(eval_duration)))}")

            if args.ood_dataset:
                print(f"{f'OOD Evaluation':=^60}")
                ood_start = time.time()
                # 현재 task까지의 ID 데이터셋만 사용
                all_id_datasets = torch.utils.data.ConcatDataset([data_loader[t]['val'].dataset for t in range(task_id+1)])
                ood_loader = data_loader[-1]['ood']
                self.evaluate_ood(model, all_id_datasets, ood_loader, device, args, task_id)
                ood_duration = time.time() - ood_start
                print(f"OOD evaluation after Task {task_id+1} completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")
                

def evaluate_till_now(self, model, data_loader, device, task_id, class_mask, acc_matrix, args):
        """
        현재까지의 모든 task에 대해 평가하고,
        A_last, A_avg, Forgetting 지표를 계산하여 출력합니다.
        """
        for t in range(task_id+1):
            acc_matrix[t, task_id] = self.evaluate_task(model, data_loader[t]['val'], device, t, class_mask, args)

        A_i = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id+1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)

        result_str = "[Average accuracy till task{}] A_last: {:.2f} A_avg: {:.2f}".format(task_id+1, A_last, A_avg)
        
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            result_str += " Forgetting: {:.4f}".format(forgetting)
        else:
            forgetting = 0

        if args.wandb:
            import wandb
            wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting, "TASK": task_id})
        
        print(result_str)
        if args.verbose or args.wandb:
            sub_matrix = acc_matrix[:task_id+1, :task_id+1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            heatmap_path = save_accuracy_heatmap(result, task_id, args)
            if args.wandb:
                import wandb
                wandb.log({"Accuracy Heatmap": wandb.Image(heatmap_path)})
        
        return {"Acc@1": A_last}
        
        
def evaluate_task(self, model, data_loader, device, task_id, class_mask, args):
        """
        한 task에 대해 evaluation을 수행하며, 매 print_freq 배치마다 중간 결과를 출력합니다.
        """
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.eval()
        total_acc = 0.0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if args.develop and batch_idx > 20: break

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                batch_size = inputs.size(0)
                
                total_acc += acc1.item() * batch_size
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                if batch_idx % args.print_freq == 0:
                    running_avg_loss = total_loss / total_samples
                    running_avg_acc = total_acc / total_samples
                    print(f"Task {task_id+1}, Batch [{batch_idx}/{len(data_loader)}]: "
                        f"Running Avg Loss = {running_avg_loss:.2f}, Running Avg Acc@1 = {running_avg_acc:.2f}")

        avg_acc = total_acc / total_samples
        avg_loss = total_loss / total_samples
        print(f"Task {task_id+1}: Final Avg Loss = {avg_loss:.2f} | Final Avg Acc@1 = {avg_acc:.2f}")
        return avg_acc
        
        
def evaluate_ood(self, model, id_datasets, ood_dataset, device, args, task_id=None):
        model.eval()
        
        # === New unified OOD evaluation (adapter 기반) ===
        ood_method = args.ood_method.upper()
        
        id_size, ood_size = len(id_datasets), len(ood_dataset)
        min_size = min(id_size, ood_size)
        if args.develop:
            min_size = 1000
        if args.ood_develop:
            min_size = args.ood_develop
        if args.verbose:
            print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")
        
        id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
        ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset
        
        id_loader = torch.utils.data.DataLoader(id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ood_loader = torch.utils.data.DataLoader(ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        if ood_method == "ALL":
            methods = SUPPORTED_METHODS
        else:
            # 쉼표로 구분된 메소드들 처리
            methods = [method.strip().upper() for method in ood_method.split(',')]
            # 지원되지 않는 메소드 확인
            unsupported = [m for m in methods if m not in SUPPORTED_METHODS]
            if unsupported:
                raise ValueError(f"지원되지 않는 OOD 메소드: {unsupported}. 지원되는 메소드: {SUPPORTED_METHODS}")
        
        from sklearn import metrics
        results = {}
        
        for method in methods:
            id_scores, ood_scores = compute_ood_scores(method, model, id_loader, ood_loader, device)
            
            if args.verbose or args.wandb:
                hist_path = save_anomaly_histogram(id_scores.numpy(), ood_scores.numpy(), args, suffix=method.lower(), task_id=task_id)
                if args.wandb:
                    import wandb
                    wandb.log({f"Anomaly Histogram TASK {task_id}": wandb.Image(hist_path)})
            
            binary_labels = np.concatenate([np.ones(id_scores.shape[0]), np.zeros(ood_scores.shape[0])])
            all_scores = np.concatenate([id_scores.numpy(), ood_scores.numpy()])
            
            fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
            auroc = metrics.auc(fpr, tpr)
            idx_tpr95 = np.abs(tpr - 0.95).argmin()
            fpr_at_tpr95 = fpr[idx_tpr95]
            
            print(f"[{method}]: AUROC {auroc * 100:.2f}% | FPR@TPR95 {fpr_at_tpr95 * 100:.2f}%")
            if args.wandb:
                import wandb
                wandb.log({f"{method}_AUROC (↑)": auroc * 100, f"{method}_FPR@TPR95 (↓)": fpr_at_tpr95 * 100, "TASK": task_id})
            
            results[method] = {"auroc": auroc, "fpr_at_tpr95": fpr_at_tpr95, "scores": all_scores}
        
        return results
