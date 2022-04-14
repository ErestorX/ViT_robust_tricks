import models
from timm import create_model


def recursive_merge_dictionaries(dica, dicb, tree=None):
    if tree is None:
        tree = []
    for key in dicb:
        if key in dica:
            if isinstance(dicb[key], dict) and isinstance(dica[key], dict):
                next_tree = tree + [key]
                recursive_merge_dictionaries(dica[key], dicb[key], next_tree)
            elif dicb[key] == dica[key]:
                pass
            else:
                raise ValueError('Conflict at %s' % '.'.join(tree + [key]))
        else:
            dica[key] = dicb[key]
    return dica


def beautiful_model_name(model_name):
    final_tokens = []
    for token in model_name.split("_"):
        if token == 't2t':
            final_tokens.append('T2T-')
        elif token == 'vit':
            final_tokens.append('ViT')
        elif token == 'base':
            final_tokens.append('-B')
        elif token == 'small':
            final_tokens.append('-S')
        elif token == 'tiny':
            final_tokens.append('-T')
        elif token == '14':
            pass
        elif token == 'patch16':
            final_tokens.append('/16')
        elif token == 'patch32':
            final_tokens.append('/32')
        elif token == '224':
            pass
        elif token == 'p':
            final_tokens.append('-p')
        elif token == 't':
            final_tokens.append('-t')
        elif token == 'pretrained':
            final_tokens.append(' Pretrained')
        elif token == 'scratch':
            final_tokens.append(' INet training')
        elif token == 'doexp05l':
            final_tokens.append(' - dropout: 0.5/exp(0.5*l*d)')
        elif token == 'donegexp05l':
            final_tokens.append(' - dropout: 0.5-0.5/exp(0.5*l*d)')
        elif token == 'donegexp025l':
            final_tokens.append(' - dropout: 0.5-0.5/exp(0.25*l*d)')
        elif token == 'donegexp075l':
            final_tokens.append(' - dropout: 0.5-0.5/exp(0.75*l*d)')
        elif token == 'doexp5':
            final_tokens.append(' - dropout: 0.5/exp(5*d)')
        elif token == 'finetuned':
            final_tokens.append(' for INet fine tuning')
        elif token == 'custom':
            pass
        elif token == 'depth':
            final_tokens.append(' 12 ViT blocks' if 't2t' in model_name else ' 14 ViT blocks')
        else:
            final_tokens.append('-'+token)
    return ''.join(final_tokens)


def string_params2dict(string_params):
    int_params = ['patch_size', 'embed_dim', 'depth', 'num_heads', 'exp_mul']
    params_dict = {}
    params = string_params.split('-')
    for param in params:
        if '=' in param:
            key, value = param.split('=')
            if key in int_params:
                value = int(value)
            if value == 'True':
                value = True
            if value == 'False':
                value = False
            params_dict[key] = value
        else:
            params_dict[param] = True
    return params_dict


def load_model_and_make_name(args):
    params_dict = {}
    if args.model_param is not None:
        params_dict = recursive_merge_dictionaries(params_dict, string_params2dict(args.model_param))
    if args.ckpt_file is None:
        params_dict['pretrained'] = True
        experiment_name = beautiful_model_name(args.model + '_pretrained')
    else:
        experiment_name = beautiful_model_name(args.ckpt_file.replace('-', '_'))
        params_dict['checkpoint_path'] = 'output/train/' + args.ckpt_file + '/model_best.pth.tar'
    if 't2t' in args.model:
        if 'custom' not in args.model:
            model = models.T2T.load_t2t_vit(args.model, **params_dict)
        else:
            model = models.Custom_T2T.load_custom_t2t_vit(args.model, **params_dict)
    else:
        model = create_model(args.model, **params_dict)
    return model, experiment_name