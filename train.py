import os, argparse, sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model import fair_resnet18, Vanilla_ResNet18
from util import get_et_time, set_seed, DualWriter, AverageMeter
from dataset import load_dataset
from util import  calculate_metrics, optimize_auc, optimize_acc
from tqdm import tqdm


def train(model, train_loader, criterion, optimizer, device, epoch, args):

    model.train()
    if not args.no_progress:
        p_bar = tqdm(range(len(train_loader)))
    total_loss = AverageMeter()
    
    for batch_idx, (data, target, sensitive_attr) in enumerate(train_loader):
        
        data, target, sensitive_attr= data.to(device), target.to(device), sensitive_attr.to(device)
        outputs_full, virtual_centers, rep_feature_full, attr_prediction = model(data)
        batchsize = data.size(0)
        optimizer.zero_grad()
        
        loss_conditional_classifier = criterion(attr_prediction, sensitive_attr)
        
        loss_decouple_classifier = 0
        for i in range(args.sensitive_attributes):
            current_idx = (sensitive_attr == i).nonzero().squeeze()
            if(current_idx.numel() )> 0:
                decouple_group_logits = outputs_full[i][current_idx]            
                decouple_group_targets = target[current_idx].unsqueeze(0) if batchsize == 1 else target[current_idx]
                loss_decouple_classifier += criterion(decouple_group_logits, decouple_group_targets)
            
        loss_virtual_center = 0
        virtual_centers_combined = torch.zeros(batchsize, model.num_classes).to(data.device)
        for i in range(args.sensitive_attributes):
            current_idx = (sensitive_attr == i).nonzero().squeeze()
            if(current_idx.numel() )> 0:
                decouple_group_cosine_similarity = virtual_centers[i][:batchsize][current_idx] / 0.1   
                decouple_group_targets = target[current_idx].unsqueeze(0) if batchsize == 1 else target[current_idx]
                softmax_similariry = F.softmax(decouple_group_cosine_similarity, dim=1)
                one_hot_targets = torch.zeros_like(softmax_similariry)
                one_hot_targets.scatter_(1, decouple_group_targets.unsqueeze(1), 1)
                loss_virtual_center += -torch.sum(one_hot_targets * torch.log(softmax_similariry)) / batchsize
                virtual_centers_combined[current_idx] = virtual_centers[i][:batchsize][current_idx]
        
        
        ############### Pairwise Alignment #################################
        
        
        mask_sensi = sensitive_attr.unsqueeze(0) == sensitive_attr.unsqueeze(1)
        mask_target_same = target.unsqueeze(0) == target.unsqueeze(1)
        mask_same_sensi_same_target = mask_sensi & mask_target_same
        mask_diff_sensi_same_target = (sensitive_attr.unsqueeze(0) != sensitive_attr.unsqueeze(1)) & mask_target_same
        
        z_i = F.normalize(rep_feature_full, dim=1)
        z_plus = z_i
        
        num_samples = sensitive_attr.size(0)
        random_indices = torch.full((num_samples,), -1, dtype=torch.long)
        for i in range(num_samples):
            same_sensi_same_target_indices = torch.where(mask_same_sensi_same_target[i])[0]
            same_sensi_same_target_indices = same_sensi_same_target_indices[same_sensi_same_target_indices != i]
            if len(same_sensi_same_target_indices) > 0:
                random_index = np.random.choice(same_sensi_same_target_indices.cpu().numpy())
                random_indices[i] = random_index
            else:
                random_indices[i] = i
                
        exp_positive = torch.exp(torch.sum(z_i * z_plus[random_indices], dim=1)  / args.temperature)  
        one_hot_targets = torch.eye(virtual_centers_combined.size(1)).to(device)[target]
        exp_positive = exp_positive + torch.exp(torch.sum(one_hot_targets * virtual_centers_combined, dim=1)  / args.temperature)  
        
        random_indices_nega = torch.full((num_samples,), -1, dtype=torch.long)
        for i in range(num_samples):
            dff_sensi_same_target_indices = torch.where(mask_diff_sensi_same_target[i])[0]
            dff_sensi_same_target_indices = dff_sensi_same_target_indices[dff_sensi_same_target_indices != i]
            if len(dff_sensi_same_target_indices) > 0:
                random_index = np.random.choice(dff_sensi_same_target_indices.cpu().numpy())
                random_indices_nega[i] = random_index
            else:
                random_indices_nega[i] = 0
        
        
        mask_logits = torch.ones_like(outputs_full.permute(1,0,2), dtype=torch.bool) # group, batch, class
        mask_logits[torch.arange(batchsize), sensitive_attr, target] = False
        masked_output = virtual_centers.permute(1,0,2)[mask_logits].view(batchsize, -1)
        exp_negatives_sum_diff_sensi_diff_target = torch.exp(masked_output).sum(dim=1)/ args.temperature
        exp_negatives_sum_total = exp_negatives_sum_diff_sensi_diff_target 
        exp_negatives_sum_total = torch.where(exp_negatives_sum_total == 0, torch.tensor(1.0).to(device), exp_negatives_sum_total)
        
        loss_fair_distri= -torch.log(exp_positive / ( torch.exp(torch.sum(z_i * z_plus[random_indices_nega], dim=1)  / args.temperature)  + exp_negatives_sum_total)).mean()
        
        loss = loss_decouple_classifier + args.lambda1*loss_conditional_classifier + args.lambda2 * loss_virtual_center +  args.lambda3 * loss_fair_distri 

        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
        
        if not args.no_progress:
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}.  ".format(
                epoch=epoch+1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=len(train_loader),
                loss=total_loss.avg,)
                )
            p_bar.update()
    if not args.no_progress:
        p_bar.close()

    print(f'\n#####################################Train {epoch} #######################################')    
    print(f'Average loss for epoch {epoch}: {total_loss.avg}')
    return total_loss.avg
    
    
def test(model, erm_model, test_loader, criterion, device, epoch, args, use_erm, use_erm_acc, v_a_optimal, v_a_optimal_acc):
    model.eval()
    erm_model.eval()
    
    correct = 0
    correct_subgroups = [0 for _ in range(args.sensitive_attributes)]
    total = [0 for _ in range(args.sensitive_attributes)]
    correct_erm = 0
    correct_erm_subgroups = [0 for _ in range(args.sensitive_attributes)]
    
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    tol_output_erm  = []
    
    with torch.no_grad():
        for data, target, sensitive_attr in test_loader:
            data, target, sensitive_attr = data.to(device), target.to(device), sensitive_attr.to(device)
            erm_output = erm_model(data)
            outputs, _, _, _ = model(data)
            pred_erm = erm_output.argmax(dim=1, keepdim=True)
            correct_erm += pred_erm.eq(target.view_as(pred_erm)).sum().item()
            
            erm_prob = F.softmax(erm_output,dim=1)
            tol_output_erm += erm_prob.cpu().data.numpy().tolist()
            
            for j in range(args.sensitive_attributes):
                current_idx = (sensitive_attr == j).squeeze()
                if(current_idx.numel() )> 0:
                    correct_erm_subgroups[j] +=  pred_erm[current_idx].eq(target[current_idx].view_as(pred_erm[current_idx])).sum().item()
                    total[j] += current_idx.sum().item()
                    
            
            outputs_memory = torch.zeros(data.size(0), erm_output.size(1)).to(data.device)
            for i in range(args.sensitive_attributes):
                current_idx = (sensitive_attr == i).nonzero().squeeze()
                if(current_idx.numel() )> 0:
                    outputs_memory[current_idx] = outputs[i][current_idx]
                    
            pred = outputs_memory.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            for j in range(args.sensitive_attributes):
                current_idx = (sensitive_attr == j).squeeze()
                if(current_idx.numel() )> 0:
                    correct_subgroups[j] += pred[current_idx].eq(target[current_idx].view_as(pred[current_idx])).sum().item()

            prob = F.softmax(outputs_memory,dim=1)
            tol_output += prob.cpu().data.numpy().tolist()
            tol_target += target.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            
    total_accuracy = 100. * correct / len(test_loader.dataset)
    subgroup_acc = [100. * correct_subgroups[i] / total[i] if total[i] > 0 else 0  for i in range(args.sensitive_attributes) ]  
    
          
    erm_accuracy = 100. * correct_erm / len(test_loader.dataset)
    erm_subgroup_accuracy = [100. * correct_erm_subgroups[i] / total[i] if total[i] > 0 else 0   for i in range(args.sensitive_attributes) ]
    
    erm_fairness = max(erm_subgroup_accuracy) - min(erm_subgroup_accuracy)
    print(f'\n#####################################Test {epoch} #######################################')    
    print(f'erm Accuracy: {erm_accuracy:.2f}%, erm Subgroup Accuracy: {[ str(item)[:5] for item in erm_subgroup_accuracy]}%,  Group Fairness: {erm_fairness}')
    print(f'Max Accuracy: {total_accuracy:.2f}%, Max Subgroup Accuracy: {[ str(item)[:5] for item in subgroup_acc]}%, Max Group Fairness: {max(subgroup_acc) - min(subgroup_acc)}')

    
    addjust_subgroups = copy.deepcopy(correct_subgroups)
    for j in range(args.sensitive_attributes):
        if(use_erm_acc[j]):
            addjust_subgroups[j] = correct_erm_subgroups[j]
    addjust_subgroup_acc = [100. * addjust_subgroups[i] / total[i] if total[i] > 0 else 0  for i in range(args.sensitive_attributes) ]  
    addjust_acc = 100. * sum(addjust_subgroups)/ len(test_loader.dataset)
    print(f'Accuracy MIN: {addjust_acc:.2f}%, Subgroup Accuracy: {[ str(item)[:5] for item in addjust_subgroup_acc]}%,  Group Fairness: {max(addjust_subgroup_acc) - min(addjust_subgroup_acc)}')
    
    
    addjust_subgroups_group = copy.deepcopy(correct_subgroups)
    for j in range(args.sensitive_attributes):
        if(v_a_optimal_acc[j] == 1):
            addjust_subgroups_group[j] = correct_erm_subgroups[j]
    addjust_subgroup_acc_groupfairness = [100. * addjust_subgroups_group[i] / total[i] if total[i] > 0 else 0  for i in range(args.sensitive_attributes) ]  
    addjust_acc_groupfairness = 100. * sum(addjust_subgroups_group)/ len(test_loader.dataset)
    print(f'Accuracy GP: {addjust_acc_groupfairness:.2f}%, Subgroup Accuracy: {[ str(item)[:5] for item in addjust_subgroup_acc_groupfairness]}%,  Group Fairness: {max(addjust_subgroup_acc_groupfairness) - min(addjust_subgroup_acc_groupfairness)}')
    
    log_dict_erm, _, aucs_subgroup = calculate_metrics(tol_output_erm, tol_target, tol_sensitive, tol_index, args.sensitive_attributes)
    print("erm: ", log_dict_erm, max(aucs_subgroup)-min(aucs_subgroup))
    
    log_dict, _, aucs_subgroup = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, args.sensitive_attributes)
    print("Max: ", log_dict, max(aucs_subgroup)-min(aucs_subgroup))
    
    tempo = np.array([int(use_erm[i]) for i in range(args.sensitive_attributes)])
    mask = np.isin(tol_sensitive, np.where(tempo == 1)[0])
    tol_output_final_greedy = np.where(mask[:, None], tol_output_erm, tol_output)
    
    log_dict_final_greedy, _, aucs_subgroup_adjust = calculate_metrics(tol_output_final_greedy, tol_target, tol_sensitive, tol_index, args.sensitive_attributes)
    print("Adjust MIN: ", log_dict_final_greedy, max(aucs_subgroup_adjust)-min(aucs_subgroup_adjust))
    
    tempo = v_a_optimal 
    mask = np.isin(tol_sensitive, np.where(tempo == 1)[0])
    tol_output_final= np.where(mask[:, None], tol_output_erm, tol_output)
    
    log_dict_final, _, aucs_subgroup_adjust = calculate_metrics(tol_output_final, tol_target, tol_sensitive, tol_index, args.sensitive_attributes)
    print("Adjust GP: ", log_dict_final, max(aucs_subgroup_adjust)-min(aucs_subgroup_adjust))
    
    return erm_accuracy,  erm_fairness, min(erm_subgroup_accuracy),   erm_subgroup_accuracy, tol_output, tol_target, tol_sensitive, tol_index


def validate(model, erm_model, val_loader, criterion, device, epoch, args):
    print(f'#####################################validation {epoch} #######################################')
    model.eval()
    erm_model.eval()
    
    correct = 0
    correct_subgroups = [0  for _ in range(args.sensitive_attributes)] 
    total = [0 for _ in range(args.sensitive_attributes)]
    correct_erm = 0
    correct_erm_subgroups = [0 for _ in range(args.sensitive_attributes)]
    
    tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
    tol_output_erm = []
    val_loss, auc = 0., 0.
    no_iter = 0
    
    with torch.no_grad():
        for data, target, sensitive_attr in val_loader:
            data, target, sensitive_attr = data.to(device), target.to(device), sensitive_attr.to(device)
            outputs, _, _, _ = model(data)
            erm_output = erm_model(data)
            loss = criterion(erm_output, target)
            try:
                val_loss += loss.item()
            except:
                val_loss += loss.mean().item()
                
            pred_erm = erm_output.argmax(dim=1, keepdim=True)
            correct_erm += pred_erm.eq(target.view_as(pred_erm)).sum().item()
            tol_output_erm += F.softmax(erm_output,dim=1).cpu().data.numpy().tolist()
            
            for j in range(args.sensitive_attributes):
                current_idx = (sensitive_attr == j).squeeze()
                if(current_idx.numel() )> 0:
                    correct_erm_subgroups[j] +=  pred_erm[current_idx].eq(target[current_idx].view_as(pred_erm[current_idx])).sum().item()
                    total[j] += current_idx.sum().item()
                    
            outputs_memory = torch.zeros(data.size(0), erm_output.size(1)).to(data.device) 
            for i in range(args.sensitive_attributes):
                current_idx = (sensitive_attr == i).nonzero().squeeze()
                if(current_idx.numel() )> 0:
                    outputs_memory[current_idx] = outputs[i][current_idx]
                    
            pred = outputs_memory.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            for j in range(args.sensitive_attributes):
                current_idx = (sensitive_attr == j).squeeze()
                if(current_idx.numel() )> 0:
                    correct_subgroups[j] += pred[current_idx].eq(target[current_idx].view_as(pred[current_idx])).sum().item()    
            
            prob = F.softmax(outputs_memory,dim=1)
            tol_output += prob.cpu().data.numpy().tolist()
            tol_target += target.cpu().data.numpy().tolist()
            tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            
            no_iter += 1
            
    val_loss /= no_iter

    total_accuracy = 100. * correct / len(val_loader.dataset)
    subgroup_acc = [100. * correct_subgroups[i] / total[i] if total[i] > 0 else 0  for i in range(args.sensitive_attributes) ]     
    

    log_dict, t_predictions,aucs_subgroup  = calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, args.sensitive_attributes)
    log_dict_erm, t_predictions_erm,aucs_subgroup_erm  = calculate_metrics(tol_output_erm, tol_target, tol_sensitive, tol_index, args.sensitive_attributes)
    
    auc = 100 * log_dict["Overall AUC"]
    erm_auc = 100 * log_dict_erm["Overall AUC"]
    erm_accuracy = 100. * correct_erm / len(val_loader.dataset)
    erm_accuracy_subgroups = [ (100. * correct_erm_subgroups[j] / total[j]) if total[j] > 0 else 0  for j in range(args.sensitive_attributes)]
    
    v_a_optimal_acc, overall_acc_final = optimize_acc(total_accuracy, subgroup_acc, erm_accuracy, erm_accuracy_subgroups, total, args)
    addjust_subgroups_group = copy.deepcopy(correct_subgroups)
    for j in range(args.sensitive_attributes):
        if(v_a_optimal_acc[j] == 1):
            addjust_subgroups_group[j] = correct_erm_subgroups[j]
    addjust_subgroup_acc_groupfairness = [100. * addjust_subgroups_group[i] / total[i] if total[i] > 0 else 0  for i in range(args.sensitive_attributes) ]  
    
    print(f'ERM Accuracy: {erm_accuracy:.2f}%, ERM Subgroups Accuracy: {[ str(item)[:5] for item in erm_accuracy_subgroups]}%, Group Fairness: {max(erm_accuracy_subgroups) - min(erm_accuracy_subgroups)}')
    
    print(f'Max Accuracy: {total_accuracy:.2f}%, Max Subgroup Accuracy: {[ str(item)[:5] for item in subgroup_acc]}%, Max Group Fairness: {max(subgroup_acc) - min(subgroup_acc)}')
    print(f'Accuracy GP: {overall_acc_final:.2f}%, Virtual Subgroup Accuracy: {[ str(item)[:5] for item in addjust_subgroup_acc_groupfairness]}%, Max Group Fairness: {max(addjust_subgroup_acc_groupfairness) - min(addjust_subgroup_acc_groupfairness)}')
    
    
    print(f'ERM AUC: {erm_auc:.2f}%, Subgroups AUC: {[ str(item)[:5] for item in aucs_subgroup_erm]}%, AUC Group Fairness: {max(aucs_subgroup_erm) - min(aucs_subgroup_erm)}\n')
    print(f'AUC: {auc:.2f}%, Subgroups AUC: {[ str(item)[:5] for item in aucs_subgroup]}%, AUC Group Fairness: {max(aucs_subgroup) - min(aucs_subgroup)}')
    
    use_erm_acc = np.array([ correct_erm_subgroups[idx] > correct_subgroups[idx] for idx in range(args.sensitive_attributes) ])
    use_erm = np.array([ aucs_subgroup_erm[idx] > aucs_subgroup[idx] for idx in range(args.sensitive_attributes) ])
    v_a_optimal, overall_auc_final, log_dict_final,aucs_subgroup_adjust = optimize_auc(tol_output, tol_output_erm, tol_target, tol_sensitive, tol_index, args.sensitive_attributes)
    print(f'Adjusted GP AUC: {100 *log_dict_final["Overall AUC"]:.2f}%, Subgroups AUC: {[ str(item)[:5] for item in aucs_subgroup_adjust]}%, AUC Group Fairness: {max(aucs_subgroup_adjust) - min(aucs_subgroup_adjust)}\n')
    
    return auc, val_loss, log_dict, use_erm, aucs_subgroup, v_a_optimal, use_erm_acc, v_a_optimal_acc


def main(args):
    train_set, val_set, test_set,sensitive_attributes, num_classes = load_dataset(args)
    args.sensitive_attributes = sensitive_attributes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fair_resnet18(args, num_classes=num_classes,sensitive_attributes=args.sensitive_attributes)
    model = model.to(device)
    erm_model = Vanilla_ResNet18(args, num_classes=num_classes)
    erm_model = erm_model.to(device)
    

    train_loader = DataLoader(train_set, batch_size=args.bs, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.bs,num_workers=4,   shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.bs,num_workers=4,  shuffle=False)
    criterion = nn.CrossEntropyLoss()
    

    optimizer = optim.SGD([param for name, param in model.named_parameters()], lr=args.lr,momentum=0.9,  weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.StepLR_size, gamma=args.gamma)
    
    
    auc, val_loss, log_dict, use_erm,aucs_subgroup, v_a_optimal,use_erm_acc, v_a_optimal_acc = validate(model, erm_model, val_loader, criterion, device,-1, args)
    for epoch in (range(args.epochs)):
        
        train_loss = train(model,train_loader, criterion, optimizer, device, epoch, args)
        if(os.path.exists('model/erm_checkpoint_{epoch}.pth')):
            erm_state = torch.load('model/erm_checkpoint_{epoch}.pth')
            erm_model.load_state_dict(erm_state)
        else:
            print('No ERM model loaded')
        auc, val_loss, log_dict, use_erm,aucs_subgroup, v_a_optimal,use_erm_acc, v_a_optimal_acc = validate(model,erm_model, val_loader, criterion, device,epoch, args)
        test(model, erm_model, test_loader, criterion, device,epoch, args, use_erm,use_erm_acc, v_a_optimal, v_a_optimal_acc)
        scheduler.step()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50, help='train rounds')
    parser.add_argument('--bs', type=int, default=256, help='batch_size')
    parser.add_argument('--dataset', type=str, default="utk", help='')
    parser.add_argument('--data_path', type=str, default="../data")
    parser.add_argument('--save_path', type=str, default="debug")
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--StepLR_size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--no_progress', action='store_true', default=False)
    parser.add_argument('--experiment', type=str, default="debug")
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=0.1)
    parser.add_argument('--ta', type=int, default=0)
    parser.add_argument('--sa', type=str, default='race')
    args = parser.parse_args()
    
    # Basic Init
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    run_detail = f'exp_lr_{args.lr}_{args.experiment}_seed_{args.seed}'
    current_time = get_et_time()
    args.save_path = os.path.join(args.save_path, run_detail, current_time)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Log file
    log_file_path = os.path.join(args.save_path, "log.txt")
    sys.stdout = DualWriter(log_file_path)
    print(args)
    print(f'save path: {args.save_path}')
    
    set_seed(args)
    main(args)
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal  
