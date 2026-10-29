"""Optional W&B metrics with explicit run identity and local provenance.

Importing this module never initializes W&B. Callers opt in with ``--wandb``
and authenticate separately with the W&B CLI or environment. Project visibility
is managed in W&B, never changed by this code. Model/data/source uploads are not
performed by these helpers.
"""

import importlib
import json
from pathlib import Path


DEFAULT_PROJECT = 'Philips_Assignment_4_Video_Action_Recognition'
DEFAULT_ENTITY = 'rphilip4-johns-hopkins-university'


def _configuration(values):
    """Make CLI/JSON configuration serializable without copying credential fields."""
    private = {'api_key', 'wandb_api_key', 'password', 'access_token', 'secret'}
    if isinstance(values, dict):
        return {
            str(key): _configuration(value) for key, value in values.items()
            if str(key).lower() not in private
        }
    if isinstance(values, (list, tuple)):
        return [_configuration(value) for value in values]
    if values is None or isinstance(values, (str, int, float, bool)):
        return values
    return str(values)


def _save_metadata(run, directory, mode, job_type):
    """Save a run pointer while preserving pointers to other runs in that directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'tracking.json'
    previous = {}
    if target.exists():
        try:
            previous = json.loads(target.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            previous = {}
        if not isinstance(previous, dict) or previous.get('run_id') != run.id:
            target = directory / f'tracking_{job_type}_{run.id}.json'
            previous = {}
    record = {
        **previous, 'run_id': run.id, 'entity': run.entity, 'project': run.project,
        'url': run.url, 'mode': mode,
        'job_type': previous.get('job_type', job_type), 'last_job_type': job_type,
    }
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(target)
    print(f'W&B run metadata saved to: {target}')


def _run_options(wandb, args, run_dir, configuration, job_type):
    """Build SDK options without publishing code or replacing training configuration."""
    mode = getattr(args, 'wandb_mode', 'online')
    run_id = getattr(args, 'wandb_run_id', None)
    settings = {
        'project': getattr(args, 'wandb_project', DEFAULT_PROJECT),
        'entity': getattr(args, 'wandb_entity', DEFAULT_ENTITY) or None,
        'mode': mode, 'job_type': job_type,
        'save_code': False, 'settings': wandb.Settings(disable_git=True),
    }
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        settings['dir'] = str(run_dir)
    name = getattr(args, 'wandb_name', None)
    if name:
        settings['name'] = name
    elif run_dir is not None and not run_id:
        settings['name'] = Path(run_dir).name
    if run_id:
        settings.pop('job_type')
        settings.update(
            id=run_id, resume='must', allow_val_change=True,
            config={'evaluation_config': configuration},
        )
    else:
        settings['config'] = configuration
    if job_type == 'import':
        settings['tags'] = ['imported-history']
        settings['notes'] = (
            'Historical training metrics imported from CSV. W&B execution time '
            'and system information describe this import process, not the '
            'original training execution.'
        )
    return settings


def start_tracking(args, run_dir=None, config=None, job_type='train'):
    """Start an explicitly enabled W&B run, otherwise return ``None``.

    ``args`` supplies optional ``wandb``, ``wandb_project``, ``wandb_entity``,
    ``wandb_mode``, ``wandb_name``, and evaluation-only ``wandb_run_id`` settings.
    Missing project/entity settings follow the documented project default and
    W&B account defaults. Online/offline behavior is explicit; ``disabled``
    imports no SDK. A resumed evaluation adds nested evaluation configuration
    instead of overwriting the run's original training hyperparameters.
    """
    mode = getattr(args, 'wandb_mode', 'online')
    if not getattr(args, 'wandb', False) or mode == 'disabled':
        return None
    if mode not in ('online', 'offline'):
        raise ValueError('W&B mode must be online, offline, or disabled')
    run_id = getattr(args, 'wandb_run_id', None)
    if run_id and job_type not in ('eval', 'evaluation', 'test'):
        raise ValueError('An existing W&B run ID is only used for evaluation')
    if run_id and mode != 'online':
        raise ValueError('Resuming an existing W&B run requires online mode')
    try:
        wandb = importlib.import_module('wandb')
    except ImportError as error:
        raise RuntimeError(
            'W&B logging requires wandb. Install it with python -m pip install wandb '
            'or omit --wandb.'
        ) from error
    configuration = dict(vars(args))
    configuration.update(config or {})
    configuration = _configuration(configuration)
    settings = _run_options(wandb, args, run_dir, configuration, job_type)
    run = wandb.init(**settings)
    try:
        run.define_metric('epoch')
        for name in ('train/*', 'val/*', 'learning_rate', 'cnn_learning_rate'):
            run.define_metric(name, step_metric='epoch')
        run.define_metric('val/accuracy', summary='max', step_metric='epoch')
        run.define_metric('val/loss', summary='min', step_metric='epoch')
        if run_dir is not None:
            _save_metadata(run, run_dir, mode, job_type)
    except BaseException:
        run.finish(exit_code=1)
        raise
    print(f'W&B run: {run.url or run.id}')
    return run


def log_epoch(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    run, epoch, learning_rate, cnn_learning_rate, train_loss, train_accuracy,
    val_loss, val_accuracy,
):
    """Log one completed epoch and identify the best validation-accuracy epoch.

    Accuracies are fractions from zero to one. ``epoch`` is the plotted custom
    axis, not W&B's internal step, so later evaluation can append test results.
    Historical CSVs may supply ``None`` for an unrecorded CNN learning rate.
    """
    if run is None:
        return
    values = {
        'epoch': int(epoch), 'learning_rate': float(learning_rate),
        'train/loss': float(train_loss), 'train/accuracy': float(train_accuracy),
        'val/loss': float(val_loss), 'val/accuracy': float(val_accuracy),
    }
    if cnn_learning_rate is not None:
        values['cnn_learning_rate'] = float(cnn_learning_rate)
    run.log(values)
    best = run.summary.get('best_val_accuracy')
    if best is None or val_accuracy > best:
        run.summary['best_val_accuracy'] = float(val_accuracy)
        run.summary['best_epoch'] = int(epoch)
        run.summary['val_loss_at_best_accuracy'] = float(val_loss)


def finish_tracking(run, exit_code=0):
    """Close an enabled W&B run, including when training/evaluation raises."""
    if run is not None:
        run.finish(exit_code=exit_code)
