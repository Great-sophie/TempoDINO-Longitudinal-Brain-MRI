from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]

def test_public_scripts_parse():
    for script in (ROOT / 'scripts').glob('*.py'):
        ast.parse(script.read_text())

def test_no_personal_path_literals_in_scripts():
    for script in (ROOT / 'scripts').glob('*.py'):
        text = script.read_text()
        assert '/media/sophie/' not in text
        assert '/home/sophie/' not in text

def test_train_checkpoint_source_and_output_are_separate():
    t = (ROOT/'scripts/19_train_tempodino_v7_unified_axis.py').read_text()
    assert 'V6_CHECKPOINT_ROOT = (\n    PROJECT_DIR' in t
    assert 'CHECKPOINT_ROOT = (\n    RUN_ROOT' in t

def test_locked_results_structure():
    d = json.loads((ROOT/'results/locked_oof_metrics.json').read_text())
    assert d['locked_complete'] is True
    assert d['metrics']['n_mri']==114 and d['metrics']['n_subjects']==23

def test_no_private_files():
    forbidden={'.nii','.npy','.pt','.pth','.mgz','.mif','.dcm'}
    for p in ROOT.rglob('*'):
        if p.is_file() and not any(part in {'__pycache__','.pytest_cache'} for part in p.parts):
            assert p.suffix not in forbidden
