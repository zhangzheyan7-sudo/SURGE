"""Dataset-independent train augmentation modes with native-size validation."""
from pathlib import Path


def add_data_arguments(parser):
    parser.add_argument('--data-root', default=None, help='Dataset root in the existing manifest/image-gt-flow format')
    parser.add_argument('--dataset-name', default=None, help='Dataset identifier; defaults to the root folder name')
    parser.add_argument('--augmentation', choices=('none', 'gdros'), default=None)
    parser.add_argument('--output-dir', default=None, help='Override experiment output directory')


def apply_data_arguments(config, args):
    data = config['data']
    root = getattr(args, 'data_root', None)
    mode = getattr(args, 'augmentation', None) or data.get('augmentation_mode')
    if root:
        root = Path(root).expanduser().resolve()
        name = getattr(args, 'dataset_name', None) or root.name
        data['datasets'] = [{'name': name, 'root': str(root)}]
    elif getattr(args, 'dataset_name', None):
        if len(data['datasets']) != 1:
            raise ValueError('--dataset-name requires a single dataset or --data-root')
        data['datasets'][0]['name'] = args.dataset_name
    if mode:
        data['augmentation_mode'] = mode
        data['center_crop_size'] = [400, 400]
        data.setdefault('evaluation_crop_size', None)
        augment = data.setdefault('augment', {})
        augment['enabled'] = mode == 'gdros'
        spatial = augment.setdefault('gdros_spatial', {})
        spatial.update(enabled=mode == 'gdros', crop_size=[400, 400])
        if root or getattr(args, 'augmentation', None):
            names = '_'.join(item['name'] for item in data['datasets'])
            output = Path(config['_project_root']) / 'runs' / f'{names}_{mode}_valnative'
            config['training']['output_dir'] = str(output)
    if getattr(args, 'output_dir', None):
        output = Path(args.output_dir).expanduser()
        if not output.is_absolute():
            output = Path(config['_project_root']) / output
        config['training']['output_dir'] = str(output.resolve())
    if mode or getattr(args, 'output_dir', None):
        output = Path(config['training']['output_dir'])
        for section, folder in [('evaluation', 'evaluation'), ('visualization', 'visualizations')]:
            config[section]['checkpoint'] = str(output / 'best.pt')
            config[section]['output_dir'] = str(output / folder)
    return config
