import numpy as np
import torch

from attack_detection import *
from datetime import datetime

B_SIZE = 1

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


def prediction_step(model_c, model_c_name, input, target, loss_fn, attack):
    pgd_steps, step_size, epsilonMax = attack
    model_t = deepcopy(model_c)
    model_t_name = model_c_name
    writer = SummaryWriter()
    modc_hooks = get_all_hooks_non_distributed(model_c, is_t2t='t2t' in model_c_name,
                                               is_performer=model_c_name.split('_')[3] == 'p')
    modt_hooks = get_all_hooks_non_distributed(model_t, is_t2t='t2t' in model_t_name,
                                               is_performer=model_t_name.split('_')[3] == 'p')
    metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)

    input_orig = input.clone()
    for _ in range(pgd_steps):
        input.requires_grad = True
        output = model_c(input)
        model_c.zero_grad()
        cost = loss_fn(output, target)
        grad = torch.autograd.grad(cost, input, retain_graph=False, create_graph=False)[0]
        input = input + step_size * grad.sign()
        input = input_orig + torch.clamp(input - input_orig, -epsilonMax, epsilonMax)
        input = torch.clamp(input, -1, 1).detach()
    _ = model_c(input)
    output = model_t(input)
    update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", 0, writer, True)
    for hook0 in modc_hooks:
        for hook1 in modt_hooks:
            hook0.clear()
            hook1.clear()
    sim_mat = get_simmat_from_metrics(metrics_ct)
    return torch.argmax(output, axis=1), sim_mat


def train_one_epoch(victim_model, victim_name, detect_model, loader, loss_victim, loss_detect, optimizer, epoch, num_epochs, saver):
    attacks = [[0, 0, 0], [1, 1, 0.062], [1, 1, 0.031]]
    # attacks = attacks + [[40, 0.025, 0.001], [40, 0.025, 0.003], [40, 0.025, 0.005], [40, 0.025, 0.01]]
    attacks_name = ['clean', 'FGSM-0.062', 'FGSM-0.031', 'PGD-0.001', 'PGD-0.003', 'PGD-0.005', 'PGD-0.01']
    losses_m = AverageMeter()

    victim_model.eval()
    detect_model.train()

    for atk_id, atk in enumerate(attacks):
        for it, (input, target) in enumerate(loader):
            input, target = input.cuda(), target.cuda()
            input = input.repeat(5 - B_SIZE, 1, 1, 1)
            victim_out, sim_mat = prediction_step(victim_model, victim_name, input, target, loss_victim, atk)
            sim_mat = torch.unsqueeze(torch.flatten(sim_mat).detach().cuda(), 0)
            print(sim_mat)
            detect_out = detect_model(sim_mat)
            detect_y = torch.from_numpy(np.asarray([[atk_id>0, (victim_out[0]==target[0]).cpu().numpy()]]))
            detect_y = detect_y.cuda().float()
            loss = loss_detect(detect_out, detect_y)
            losses_m.update(loss.item(), sim_mat.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if it % 400*len(attacks) == 0:
                print('Epoch [{}/{}], Step [{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, it, loss.item()))
                saver.save_recovery(epoch, batch_idx=it*atk_id)
    return OrderedDict([('loss', losses_m.avg)])


def eval_one_epoch(victim_model, victim_name, detect_model, loader, loss_victim, loss_detect, epoch, num_epochs):
    attacks = [[0, 0, 0], [1, 1, 0.062], [1, 1, 0.031]]
    # attacks = attacks + [[40, 0.025, 0.001], [40, 0.025, 0.003], [40, 0.025, 0.005], [40, 0.025, 0.01]]
    attacks_name = ['clean', 'FGSM-0.062', 'FGSM-0.031', 'PGD-0.001', 'PGD-0.003', 'PGD-0.005', 'PGD-0.01']
    losses_m = AverageMeter()

    victim_model.eval()
    detect_model.eval()

    for atk_id, atk in enumerate(attacks):
        for it, (input, target) in enumerate(loader):
            input, target = input.cuda(), target.cuda()
            input = input.repeat(5 - B_SIZE, 1, 1, 1)
            victim_out, sim_mat = prediction_step(victim_model, victim_name, input, target, loss_victim, atk)
            sim_mat = torch.unsqueeze(torch.flatten(sim_mat).detach().cuda(), 0)
            detect_out = detect_model(sim_mat)
            detect_y = torch.from_numpy(np.asarray([[atk_id > 0, (victim_out[0] == target[0]).cpu().numpy()]]))
            detect_y = detect_y.cuda().float()
            loss = loss_detect(detect_out, detect_y)
            losses_m.update(loss.item(), sim_mat.size(0))
            if it % 100*len(attacks) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, it, loader.shape[0], loss.item()))
    return OrderedDict([('loss', losses_m.avg)])


def main(adv_atk=0, detect_atk=1, version=1, ckpt='t', p=True, data='/media/hugo/Data/ImageNet'):
    torch.cuda.set_device(0)
    attacks = [[0, 0, 0], [1, 1, 0.062], [1, 1, 0.031], [40, 0.025, 0.001], [40, 0.025, 0.003], [40, 0.025, 0.005],
               [40, 0.025, 0.01]]
    attacks_name = ['clean', 'FGSM-0.062', 'FGSM-0.031', 'PGD-0.001', 'PGD-0.003', 'PGD-0.005', 'PGD-0.01']
    tested_models = ['vit_base_patch16_224', 'vit_base_patch32_224', 't2t_vit_14']
    vit_versions = ['doexp5', 'donegexp025l']
    t2t_versions = ['t', 'p', 't_doexp05l', 't_donegexp05l', 't_donegexp025l']
    train_path = 'output/train/'
    ext = '/model_best.pth.tar'
    if version < 0 or version >= len(tested_models):
        print("Error: Version asked does not exist.")
        return

    if 't2t' not in tested_models[version]:
        if ckpt in vit_versions:
            ckpt_path = train_path + tested_models[version] + '_' + ckpt
            victim_name = tested_models[version] + '_' + ckpt
            model_name = 'custom_' + tested_models[version]
        else:
            ckpt_path = train_path + tested_models[version]
            victim_name = tested_models[version] + ('_pretrained' if p else '_scratch')
            model_name = tested_models[version]
    elif ckpt in t2t_versions:
        ckpt_path = train_path + tested_models[version] + '_' + ckpt
        victim_name = tested_models[version] + '_' + ckpt
        if ckpt in ['t', 'p']:
            model_name = tested_models[version] + '_' + ckpt
        else:
            model_name = 'custom_' + tested_models[version] + '_' + ckpt.split('_')[0]
    else:
        return

    ckpt_file = ckpt_path + ext
    if not p and not os.path.exists(ckpt_file):
        return
    if 't2t' not in model_name:
        if p:
            victim_model = timm.create_model(model_name, pretrained=True)
        else:
            victim_model = timm.create_model(model_name, checkpoint_path=ckpt_file)
    else:
        if ckpt in ['t', 'p']:
            victim_model = models.T2T.load_t2t_vit(model_name, ckpt_file)
        else:
            victim_model = models.Custom_T2T.load_custom_t2t_vit(model_name, ckpt_file)
    victim_model = victim_model.cuda()

    if not (os.path.exists(ckpt_path) or p):
        return
    loader_train = get_val_loader(data, batch_size=B_SIZE)
    loader_val = get_val_loader(data, batch_size=B_SIZE)
    random_index = np.random.randint(0, 10)
    val_path = 'output/CKA_detect/' + victim_name
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    val_path = val_path + '/setup_' + attacks_name[adv_atk] + '_' + attacks_name[detect_atk]
    if not os.path.exists(val_path):
        os.mkdir(val_path)
        os.mkdir(val_path + '/correct_preds/')
        os.mkdir(val_path + '/incorrect_preds/')

    loss_fn = CrossEntropyLoss().cuda()

    num_epochs = 50
    learning_rate = 0.001
    hidden_dim = 128
    nb_block_victim = len(victim_model.blocks)*3
    if 't2t' in victim_name:
        nb_block_victim += 6
    detect_model = MLP(input_dim=nb_block_victim**2, hidden_dim=hidden_dim).cuda()
    optimizer = torch.optim.Adam(detect_model.parameters(), lr=learning_rate)

    output_dir = './output/train/' + '-'.join([datetime.now().strftime("%m-%d_%H:%M"), 'CKA_adversarial_detector'])
    saver = CheckpointSaver(model=detect_model, optimizer=optimizer, max_history=3, checkpoint_dir=output_dir, decreasing=True)

    for epoch in range(num_epochs):
        train_one_epoch(victim_model, victim_name, detect_model, loader_train, loss_fn, loss_fn, optimizer, epoch, num_epochs, saver)
        eval_metrics = eval_one_epoch(victim_model, victim_name, detect_model, loader_val, loss_fn, loss_fn, epoch, num_epochs)
        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = eval_metrics['loss']
            _, _ = saver.save_checkpoint(epoch, metric=save_metric)


if __name__ == '__main__':
    main()
