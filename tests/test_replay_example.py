"""Path portability and preservation of saved experiment parameters."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('replay_example',REPO/'tools/replay_example.py')
replay=importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.example=self.root/'example'
        (self.example/'inputs').mkdir(parents=True)
        (self.example/'inputs/all.npz').write_bytes(b'input bytes')
        self.original={'run_dir':'/old/case','spaceflow_config':'/old/config.yaml','variants':[
            {'name':'spaceflow','argv':['--shape_superquadric_path','/old/case/inputs/all.npz',
                                      '--texture_optim_steps','300'],'seed':1,'low_tau':3.0},
            {'name':'copy','runner':'copy_variant','source_variant':'spaceflow',
             'source_output_dir':'/old/case/output/spaceflow'}]}
        (self.example/'experiment_runner_config.json').write_text(json.dumps(self.original))
    def tearDown(self): self.temp.cleanup()
    def test_relocation_preserves_parameters_and_adds_dependencies(self):
        destination=self.root/'replay'
        config=json.loads(replay.prepare(self.example,destination,['copy']).read_text())
        self.assertEqual([v['name'] for v in config['variants']],['spaceflow','copy'])
        self.assertEqual(config['variants'][0]['argv'][-1],'300')
        self.assertEqual(config['variants'][0]['seed'],1)
        self.assertEqual(config['variants'][0]['low_tau'],3.0)
        self.assertEqual(config['variants'][0]['argv'][1],str(destination/'inputs/all.npz'))
        self.assertEqual((destination/'inputs/all.npz').read_bytes(),b'input bytes')
        self.assertEqual(json.loads((self.example/'experiment_runner_config.json').read_text()),self.original)
    def test_unknown_variant_leaves_no_output(self):
        destination=self.root/'unknown'
        with self.assertRaises(ValueError): replay.prepare(self.example,destination,['unknown'])
        self.assertFalse(destination.exists())
    def test_existing_output_is_preserved(self):
        destination=self.root/'existing';destination.mkdir()
        (destination/'keep').write_text('preserve')
        with self.assertRaises(FileExistsError):replay.prepare(self.example,destination,[])
        self.assertEqual((destination/'keep').read_text(),'preserve')

if __name__=='__main__': unittest.main()
