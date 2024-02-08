import torch
from torch import nn
from loss.tv_loss import TVLoss
from loss.diff_loss import DiffLoss
from loss.style_loss import StyleLoss
from loss.attack_loss import AttackLoss
from net.synthesized_image import SynthesizedImage
from utilities import calculate_attack_acc
from typing import Dict, Tuple, Any
import argparse
import numpy as np
from torch.autograd import Variable


def extract_features(model: nn.Module, img, layers):
    features = []
    embed = img
    for i in range(len(model.features)):
        embed = model.features[i](embed)
        if i in layers:
            features.append(embed)
    return features, embed


def style_trans_attack(
        model: nn.Module,
        data: Tuple[Any, Any],
        args: argparse.Namespace
):
    gen_img = SynthesizedImage(data[1]).to(args.device)
    optimizer = torch.optim.Adam(gen_img.parameters(), args.lr)
    ori_features, ori_embed = extract_features(model, data[0], args.layers)
    attack_criterion = AttackLoss(args.threshold, args.attack_weight).to(args.device)
    tv_criterion = TVLoss(weight=args.tv_weight).to(args.device)
    style_criterion = [StyleLoss(weight=args.style_weight, target_feature=feature).to(args.device)
                       for feature in ori_features]
    diff_criterion = DiffLoss(weight=args.diff_weight)
    attack_loss_his = []
    tv_loss_his = []
    style_loss_his = []
    diff_loss_his = []
    for i in range(args.epochs):
        optimizer.zero_grad()
        gen_features, gen_embed = extract_features(model, gen_img(), args.layers)
        attack_loss = attack_criterion(ori_embed, gen_embed)
        tv_loss = tv_criterion(gen_img())
        style_loss = []
        for sl, gen_feature in zip(style_criterion, gen_features):
            style_loss.append(sl(gen_feature))
        style_loss = sum(style_loss)
        diff_loss = diff_criterion(data[1], gen_img())

        attack_loss_his.append(attack_loss.detach().item())
        tv_loss_his.append(tv_loss.detach().item())
        style_loss_his.append(style_loss.detach().item())
        diff_loss_his.append(diff_loss.detach().item())

        (attack_loss + tv_loss + style_loss + diff_loss).backward()
        optimizer.step()
    with torch.no_grad():
        gen_features, gen_embed = extract_features(model, gen_img(), args.layers)
        acc = 1 if torch.pairwise_distance(ori_embed, gen_embed, p=2) < attack_criterion.threshold else 0
        diff = torch.sum(gen_img() - data[1]).item()
    return gen_img().detach(), acc, {'attack': attack_loss_his, 'tv': tv_loss_his, 'sty': style_loss_his,
                                     'diff': diff_loss_his}, diff


def no_style_trans_attack(
        model: nn.Module,
        data: Tuple[Any, Any],
        args: argparse.Namespace
):
    gen_img = SynthesizedImage(data[1]).to(args.device)
    optimizer = torch.optim.Adam(gen_img.parameters(), args.lr)
    ori_features, ori_embed = extract_features(model, data[0], args.layers)
    attack_criterion = AttackLoss(args.threshold, args.attack_weight).to(args.device)
    tv_criterion = TVLoss(weight=args.tv_weight).to(args.device)
    style_criterion = [StyleLoss(weight=args.style_weight, target_feature=feature).to(args.device)
                       for feature in ori_features]
    diff_criterion = DiffLoss(weight=args.diff_weight)
    attack_loss_his = []
    tv_loss_his = []
    diff_loss_his = []
    for i in range(args.epochs):
        optimizer.zero_grad()
        gen_features, gen_embed = extract_features(model, gen_img(), args.layers)
        attack_loss = attack_criterion(ori_embed, gen_embed)
        tv_loss = tv_criterion(gen_img())
        diff_loss = diff_criterion(data[1], gen_img())
        attack_loss_his.append(attack_loss.detach().item())
        tv_loss_his.append(tv_loss.detach().item())
        diff_loss_his.append(diff_loss.detach().item())
        (attack_loss + tv_loss + diff_loss).backward()
        optimizer.step()
        gen_img.weight.data.clamp_(0, 1)
    with torch.no_grad():
        gen_features, gen_embed = extract_features(model, gen_img(), args.layers)
        acc = 1 if torch.pairwise_distance(ori_embed, gen_embed, p=2) < attack_criterion.threshold else 0
        diff = torch.sum(gen_img() - data[1]).item()
    return gen_img().detach(), acc, {'attack': attack_loss_his, 'tv': tv_loss_his, 'diff': diff_loss_his}, diff


def fgsm_attack(
        model: nn.Module,
        data: Tuple[Any, Any],
        args: argparse.Namespace
):
    ori_img, gen_img = data[0], data[1].clone().detach()
    gen_img.requires_grad = True
    embed1, embed2 = model(ori_img), model(gen_img)
    loss = args.contrastive_loss_fn(embed1, embed2, args.attack_label)
    model.zero_grad()
    loss.backward()
    gen_adv = gen_img + args.epsilon * gen_img.grad.sign()
    gen_adv = torch.clamp(gen_adv, 0, 1)
    embed2_adv = model(gen_adv)
    acc = calculate_attack_acc(embed1, embed2_adv, args.threshold, args.attack_label)
    return gen_adv, acc, None, None


def mim_attack(
        model: nn.Module,
        data: Tuple[Any, Any],
        args: argparse.Namespace
):
    ori_img, gen_img = data[0], data[1].clone().detach()
    momentum = 0
    embed1 = model(ori_img)
    for _ in range(args.epochs):
        gen_img.requires_grad = True
        embed2 = model(gen_img)
        loss = args.contrastive_loss_fn(embed1, embed2, args.attack_label)
        model.zero_grad()
        loss.backward()
        grad = gen_img.grad
        momentum = args.mu * momentum + grad / torch.norm(grad, p=1)
        gen_img = gen_img + args.epsilon * torch.sign(momentum)
        gen_img = torch.clamp(gen_img, 0, 1).detach()

    with torch.no_grad():
        embed2_adv = model(gen_img)
        acc = calculate_attack_acc(embed1, embed2_adv, args.threshold, args.attack_label)
    return gen_img, acc, None, None


def igs_attack(
        model: nn.Module,
        data: Tuple[Any, Any],
        args: argparse.Namespace
):
    ori_img, gen_img = data[0], data[1].clone().detach()
    embed1 = model(ori_img)
    for _ in range(args.epochs):
        gen_img.requires_grad_()
        embed2 = model(gen_img)
        loss = args.contrastive_loss_fn(embed1, embed2, args.attack_label)
        model.zero_grad()
        loss.backward()
        gen_img = gen_img + args.epsilon * torch.sign(gen_img.grad)
        gen_img = torch.clamp(gen_img, 0, 1).detach()

    with torch.no_grad():
        embed2_adv = model(gen_img)
        acc = calculate_attack_acc(embed1, embed2_adv, args.threshold, args.attack_label)
    return gen_img, acc, None, None


def pgd_attack(
        model: nn.Module,
        data: Tuple[Any, Any],
        args: argparse.Namespace
):
    ref_img, query_img, gen_img = data[0], data[1], data[1].detach().clone()
    embed1 = model(ref_img)
    ori_features, _ = extract_features(model, ref_img, args.layers)
    gen_img.requires_grad = True
    for i in range(args.epochs):
        gen_img.requires_grad = True
        embed2 = model(gen_img)
        model.zero_grad()
        loss = args.contrastive_loss_fn(embed1, embed2, args.attack_label)
        loss.backward()
        adv_img = gen_img + args.alpha * gen_img.grad.sign()
        # adv_img.require_grad = True
        eta = torch.clamp(adv_img - query_img, min=-args.epsilon, max=args.epsilon)
        gen_img = torch.clamp(query_img + eta, min=0, max=1).detach_()

    embed2_adv = model(gen_img)
    acc = calculate_attack_acc(embed1, embed2_adv, args.threshold, args.attack_label)
    return gen_img, acc, None, None


def pgd_attack_fg(
        model: nn.Module,
        data: Tuple[Any, Any],
        args: argparse.Namespace
):
    ref_img, query_img, gen_img = data[0], data[1], data[1].detach().clone()
    mask = query_img > 125 / 255
    embed1 = model(ref_img)
    ori_features, _ = extract_features(model, ref_img, args.layers)
    gen_img.requires_grad = True
    for i in range(args.epochs):
        gen_img.requires_grad = True
        embed2 = model(gen_img)
        model.zero_grad()
        loss = args.contrastive_loss_fn(embed1, embed2, args.attack_label)
        loss.backward()
        masked_grad = torch.where(mask, gen_img.grad.sign(), 0)
        adv_img = gen_img + args.alpha * masked_grad
        # adv_img.require_grad = True
        eta = torch.clamp(adv_img - query_img, min=-args.epsilon, max=args.epsilon)
        gen_img = torch.clamp(query_img + eta, min=0, max=1).detach_()

    embed2_adv = model(gen_img)
    acc = calculate_attack_acc(embed1, embed2_adv, args.threshold, args.attack_label)
    return gen_img, acc, None, None


def pgd_attack_st(
        model: nn.Module,
        data: Tuple[Any, Any],
        args: argparse.Namespace
):
    ref_img, query_img, gen_img = data[0], data[1], data[1].detach().clone()
    gen_img.requires_grad = True
    ori_features, embed1 = extract_features(model, ref_img, args.layers)
    tv_criterion = TVLoss(weight=args.tv_weight).to(args.device)
    style_criterion = [StyleLoss(weight=args.style_weight, target_feature=feature).to(args.device)
                       for feature in ori_features]
    for i in range(args.epochs):
        gen_img.requires_grad = True
        model.zero_grad()
        gen_features, embed2 = extract_features(model, gen_img, args.layers)
        tv_loss = tv_criterion(gen_img)
        style_loss = []
        for sl, gen_feature in zip(style_criterion, gen_features):
            style_loss.append(sl(gen_feature))
        style_loss = sum(style_loss)
        loss = args.contrastive_loss_fn(embed1, embed2, args.attack_label)
        (loss + tv_loss + style_loss).backward()
        adv_img = gen_img + args.alpha * gen_img.grad.sign()
        # adv_img.require_grad = True
        eta = torch.clamp(adv_img - query_img, min=-args.epsilon, max=args.epsilon)
        gen_img = torch.clamp(query_img + eta, min=0, max=1).detach_()

    embed2_adv = model(gen_img)
    acc = calculate_attack_acc(embed1, embed2_adv, args.threshold, args.attack_label)
    return gen_img, acc, None, None


def cw_attack(model, data, args):
    perturbation = torch.zeros_like(data[1], requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=args.lr)
    target = 0 if args.attack_label == 1 else 1
    ori_embed = model(data[0])
    for i in range(args.epochs):
        adv_img = data[1] + perturbation
        gen_embed = model(adv_img)
        loss = args.contrastive_loss_fn(ori_embed, gen_embed, target)
        loss += args.c * torch.norm(perturbation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        gen_img = data[1] + perturbation.detach()
        acc = calculate_attack_acc(ori_embed, model(gen_img), args.threshold, args.attack_label)

    return gen_img, acc, None, None


def vmi_fgsm_attack(model: nn.Module,
                    data: Tuple[Any, Any],
                    args: argparse.Namespace):
    model = model.eval()
    ref_img = data[0]
    x = data[1]
    x = x * 2 - 1
    num_iter = args.epochs
    eps = args.epsilon / 255 * 2.0
    alpha = eps / num_iter  # attack step size
    momentum = args.mu
    number = args.num_VT
    beta = args.beta
    grads = torch.zeros_like(x, requires_grad=False)
    variance = torch.zeros_like(x, requires_grad=False)
    min_x = x - eps
    max_x = x + eps

    adv = x.clone()
    embed_ref = model(ref_img)
    with torch.enable_grad():
        for i in range(num_iter):
            adv.requires_grad = True
            embed_adv = model(adv)
            loss = args.contrastive_loss_fn(embed_ref, embed_adv, args.attack_label)
            loss.backward()
            new_grad = adv.grad
            noise = momentum * grads + (new_grad + variance) / torch.norm(new_grad + variance, p=1)

            # update variance
            sample = adv.clone().detach()
            global_grad = torch.zeros_like(x, requires_grad=False)
            for _ in range(number):
                sample = sample.detach()
                sample.requires_grad = True
                rd = (torch.rand_like(x) * 2 - 1) * beta * eps
                sample = sample + rd
                embed_sample = model(sample)
                loss_sample = args.contrastive_loss_fn(embed_ref, embed_sample, args.attack_label)
                global_grad += torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]
            variance = global_grad / (number * 1.0) - new_grad

            adv = adv + alpha * noise.sign()
            adv = torch.clamp(adv, -1.0, 1.0).detach()  # range [-1, 1]
            adv = torch.max(torch.min(adv, max_x), min_x).detach()
            grads = noise

    output = model(adv)
    acc = calculate_attack_acc(embed_ref, output, args.threshold, args.attack_label)
    return adv, acc, None, None
